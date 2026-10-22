"""Train the tracklet quality head on the objective it is actually used for.

The first attempt trained it on GSR, track against track: two disjoint frame sets of one annotated
tracklet, negatives being the other tracks in the same thirty-second sequence. That task turned
out to be saturated — a plain mean of eight frames separates eight tracks at 99% — so weighting
had no headroom to help, while the softmax had every freedom to concentrate. The head duly
concentrated to about 4x and scored five points *below* the plain mean. A head that cannot beat
the average it replaces is not worth carrying.

So the task here is the deployment one: several broadcast views of a player against a clean
reference photograph, with the other players of the same match as negatives. That is where view
quality bites, because a back-turned or motion-blurred crop actively misleads a match to a
studio portrait in a way it does not mislead a match to another broadcast crop of the same
instant. Groups are formed per player per match, which is what a tracker accumulates.

The backbone stays frozen. GSR is nine source games; letting a 300-million-parameter tower adapt
to nine games would trade a general encoder for a specialised one, which is the opposite of what
the head is for. The projection is trainable by default so the shared space can accommodate
tracklet embeddings, and `L_frame` runs alongside `L_track` so per-frame quality cannot rot while
aggregation covers for it.
"""
import argparse,json,math,time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.common import read_rgb, seed_everything, seed_for, digest_file, atomic_json, atomic_write
from soccer_gif_reid.augment import preprocess
from soccer_gif_reid.model import build_model
from soccer_gif_reid.train import cross_domain_circle
from .augment_v2 import CropAugment
from .metadata import ROOT, DATA
from .quality_head import QualityHead, entropy_penalty


class LabelledGroups(Dataset):
    """A player's views within one match against their reference photograph; negatives share the match."""

    def __init__(self, query, reference, steps=400, batch=8, frames=8, augment=True, min_views=2):
        from .augment_v2 import QueryAugment, ReferenceAugment
        self.refs = defaultdict(list)
        for r in reference: self.refs[r['pid']].append(r)
        groups = defaultdict(list)
        for r in query:
            if r['pid'] in self.refs: groups[(r['pid'], r['game'])].append(r)
        self.by_match = defaultdict(list)
        for (pid, game), rows in groups.items():
            if len(rows) >= min_views: self.by_match[game].append((pid, rows))
        self.matches = [v for v in self.by_match.values() if len(v) >= 2]
        assert self.matches, 'no match has two players with several views'
        self.people = [x for v in self.matches for x in v]
        self.steps, self.batch, self.frames, self.epoch = steps, batch, frames, 0
        self.query_augment = QueryAugment('strong') if augment else None
        self.reference_augment = ReferenceAugment() if augment else None

    def __len__(self): return self.steps

    def __getitem__(self, index):
        from soccer_gif_reid.augment import load_masks
        rng = np.random.default_rng(seed_for(13, self.epoch, index))
        pool = self.matches[int(rng.integers(len(self.matches)))]
        picks = list(rng.permutation(len(pool))[:self.batch])
        while len(picks) < self.batch: picks.append(None)
        views, refs, mask, labels = [], [], [], []
        for label, i in enumerate(picks):
            pid, rows = pool[int(i)] if i is not None else self.people[int(rng.integers(len(self.people)))]
            chosen = [rows[int(j)] for j in rng.permutation(len(rows))[:self.frames]]
            frames, keep = [], []
            for r in chosen:
                image = read_rgb(r['path'])
                if self.query_augment is not None:
                    image = self.query_augment(image, load_masks(r, image.size), rng)
                frames.append(preprocess(image)); keep.append(True)
            while len(frames) < self.frames:
                frames.append(torch.zeros_like(frames[0])); keep.append(False)
            reference = self.refs[pid][int(rng.integers(len(self.refs[pid])))]
            image = read_rgb(reference['path'])
            if self.reference_augment is not None: image = self.reference_augment(image, rng)
            views.append(torch.stack(frames)); refs.append(preprocess(image))
            mask.append(torch.tensor(keep)); labels.append(label)
        return torch.stack(views), torch.stack(refs), torch.stack(mask), torch.tensor(labels, dtype=torch.long)


class Tracklets(Dataset):
    """GSR: one batch is one sequence. Kept for reference; the labelled task replaced it."""

    def __init__(self, tracks, steps=400, batch=8, frames=8, augment=True):
        self.by_sequence = defaultdict(list)
        for t in tracks:
            if len(t['views']) >= 2 * frames: self.by_sequence[t['sequence']].append(t)
        self.sequences = [v for v in self.by_sequence.values() if len(v) >= 2]
        assert self.sequences, 'no sequence has two tracklets long enough'
        self.steps, self.batch, self.frames, self.epoch = steps, batch, frames, 0
        self.augment = CropAugment() if augment else None

    def __len__(self): return self.steps

    def load(self, views, rng):
        out = []
        for v in views:
            image = read_rgb(v['path'])
            out.append(preprocess(self.augment(image, rng) if self.augment else image))
        return torch.stack(out)

    def __getitem__(self, index):
        rng = np.random.default_rng(seed_for(11, self.epoch, index))
        pool = self.sequences[int(rng.integers(len(self.sequences)))]
        picks = rng.permutation(len(pool))[:self.batch]
        a, b, labels = [], [], []
        for label, i in enumerate(picks):
            views = pool[int(i)]['views']
            # Disjoint halves, each spread across the whole tracklet rather than consecutive.
            order = rng.permutation(len(views))
            first, second = order[:self.frames], order[self.frames:2 * self.frames]
            a.append(self.load([views[int(j)] for j in sorted(first)], rng))
            b.append(self.load([views[int(j)] for j in sorted(second)], rng))
            labels.append(label)
        return torch.stack(a), torch.stack(b), torch.tensor(labels, dtype=torch.long)


def encode(model, batch, microbatch=32):
    """[B, T, 3, H, W] -> [B, T, D] through the existing encoder."""
    b, t = batch.shape[:2]
    flat = batch.flatten(0, 1).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        base = torch.cat([model.encode_base(x) for x in flat.split(microbatch)])
    with torch.autocast('cuda', enabled=False):
        return model.projection(base.float()).view(b, t, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'runs/quality_head_v2')
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--track-weight', type=float, default=1.)
    ap.add_argument('--entropy-weight', type=float, default=.05)
    ap.add_argument('--head-lr', type=float, default=.001)
    ap.add_argument('--projection-lr', type=float, default=.00005)
    ap.add_argument('--freeze-projection', action='store_true')
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    seed_everything(42); torch.set_num_threads(4)

    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    sets = {s: {d: [r for r in rows if r['split'] == s and r['domain'] == d]
                for d in ('query', 'reference')} for s in ('train', 'val')}
    train = LabelledGroups(sets['train']['query'], sets['train']['reference'], args.steps, args.batch, args.frames)
    validate = LabelledGroups(sets['val']['query'], sets['val']['reference'], 40, args.batch, args.frames, augment=False)

    model = build_model('cuda:0', ROOT/'weights')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model']); model.set_trainable_blocks(0)
    for p in model.projection.parameters(): p.requires_grad = not args.freeze_projection
    head = QualityHead().cuda()
    criterion = CircleLoss(m=.4, gamma=80)
    groups = [{'params': list(head.parameters()), 'lr': args.head_lr}]
    if not args.freeze_projection:
        groups.append({'params': list(model.projection.parameters()), 'lr': args.projection_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)

    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(task='labelled groups against reference photographs', train_matches=len(train.matches),
                  validation_matches=len(validate.matches), code_sha256=digest_file(Path(__file__)),
                  head_sha256=digest_file(ROOT/'experiments/soccernet_full_v2/quality_head.py'))
    atomic_json(args.out/'config.json', config)
    best = -1.

    for epoch in range(args.epochs):
        model.train(); head.train(); totals = defaultdict(float); began = time.monotonic()
        train.epoch = epoch
        loader = DataLoader(train, batch_size=None, num_workers=args.workers, pin_memory=True)
        for step, (views, refs, mask, labels) in enumerate(loader):
            factor = min(1., (step + 1) / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * epoch / args.epochs)))
            for group, lr in zip(optimizer.param_groups, [args.head_lr, args.projection_lr]):
                group['lr'] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            frames = encode(model, views)
            reference = encode(model, refs.unsqueeze(1)).squeeze(1)
            labels, mask = labels.cuda(), mask.cuda()
            flat, flat_labels = frames.flatten(0, 1), labels.repeat_interleave(views.shape[1])
            keep = mask.flatten()
            frame = cross_domain_circle(criterion, F.normalize(flat[keep], dim=-1),
                F.normalize(reference.repeat_interleave(views.shape[1], 0)[keep], dim=-1), flat_labels[keep])
            track, weights = head(frames, mask)
            aggregated = cross_domain_circle(criterion, track, F.normalize(reference, dim=-1), labels)
            spread = entropy_penalty(weights, mask)
            loss = frame + args.track_weight * aggregated + args.entropy_weight * spread
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 10.)
            optimizer.step()
            for key, value in (('loss', loss), ('frame', frame), ('track', aggregated), ('spread', spread)):
                totals[key] += float(value.detach())
            if (step + 1) % 25 == 0:
                atomic_json(args.out/'progress.json', {'epoch': epoch + 1, 'step': step + 1,
                    'steps': len(train), **{k: v / (step + 1) for k, v in totals.items()},
                    'seconds': time.monotonic() - began})
        metrics = evaluate(model, head, validate)
        improved = metrics['weighted_rank1'] > best
        if improved:
            best = metrics['weighted_rank1']
            atomic_write(args.out/'best.pt', lambda p: torch.save({'head': head.state_dict(),
                'projection': model.projection.state_dict(), 'config': config,
                'epoch': epoch + 1, 'metrics': metrics}, p))
        with (args.out/'history.jsonl').open('a') as f:
            f.write(json.dumps({'epoch': epoch + 1, **{k: v / len(train) for k, v in totals.items()},
                                **metrics, 'promoted': improved}) + '\n')
        print(f"epoch {epoch+1} loss {totals['loss']/len(train):.2f} "
              f"weighted {100*metrics['weighted_rank1']:.2f}% mean {100*metrics['mean_rank1']:.2f}% "
              f"single {100*metrics['single_rank1']:.2f}% ratio {metrics['weight_max_over_mean']:.2f}"
              f"{' PROMOTED' if improved else ''}", flush=True)
    atomic_json(args.out/'status.json', {'status': 'complete', 'best_weighted_rank1': best})


@torch.inference_mode()
def evaluate(model, head, data):
    """Held-out matches: does weighting the views beat averaging them, and beat one crop."""
    model.eval(); head.eval()
    weighted, mean, single, ratios = [], [], [], []
    for index in range(len(data)):
        views, refs, mask, labels = data[index]
        frames = encode(model, views)
        reference = F.normalize(encode(model, refs.unsqueeze(1)).squeeze(1), dim=-1)
        mask = mask.cuda()
        track, w = head(frames, mask)
        normalized = F.normalize(frames, dim=-1) * mask.unsqueeze(-1)
        plain = F.normalize(normalized.sum(1) / mask.sum(1, keepdim=True).clamp(min=1), dim=-1)
        first = F.normalize(frames[:, 0], dim=-1)
        for store, embedding in ((weighted, track), (mean, plain), (single, first)):
            store += (torch.argmax(embedding @ reference.T, dim=1).cpu() == labels).tolist()
        ratios.append(float((w.max(-1).values / w.mean(-1)).mean()))
    return {'weighted_rank1': float(np.mean(weighted)), 'mean_rank1': float(np.mean(mean)),
            'single_rank1': float(np.mean(single)), 'weight_max_over_mean': float(np.mean(ratios)),
            'batches': len(data)}


if __name__ == '__main__': main()
