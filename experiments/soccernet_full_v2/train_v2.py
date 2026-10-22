"""Three-stage trainer: self-supervised soccer adaptation, then the labelled cross-domain stages.

What changed from `train.py`, and why:

* **Self-supervised stage on unlabelled SoccerNet.** 282 labelled identities is the ceiling on the
  supervised stages, while the archives hold tens of thousands of players annotated across linked
  camera views of one action. Those give positive pairs from annotation rather than from a
  tracker, so no identity switch can poison them, and they cover the crops with no readable shirt
  number that the labelled pipeline cannot use at all.
* **Same-kit negatives.** Both the self-supervised and the labelled batches are drawn from a
  single match, so the players a batch has to separate wear one of two kits. Uniform sampling over
  the whole pool made almost every negative a different match in a different kit, which is
  solvable by colour — and the R3 error sheets showed the model doing exactly that. Labelled
  matches rarely hold sixteen eligible players, so a match supplies the batch's core and the rest
  is filled at random instead of shrinking the batch.
* **Reference augmentation.** The reference kit is recoloured too, so colour cannot be the shared
  cue inside a positive pair, and the shirt number is erased outright part of the time.
* **Validation features are saved**, so reference-aggregation rules can be chosen on validation
  rather than on test.
* **The shirt number, in the same embedding.** The error anatomy priced the levers: reading the
  number is worth about forty-six points of tracklet ceiling where every appearance-side change
  measured so far is worth two or three. A frozen CLIP text anchor per number is placed in the
  projection space by one linear map inside this model, and the query embedding is trained to
  land on its own number's anchor alongside the identity loss. There is no second model and no
  second score: at inference each identity's prototype is part photograph, part number. The
  number loss is skipped for a crop whose digits the augmentation has just painted over, and
  `--flat-kit` lowers how often that happens, because erasing numbers half the time was teaching
  the head to discard exactly the cue with the most headroom.
"""
import argparse,json,math,random,time,hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.common import read_rgb, seed_everything, seed_for, digest_file, atomic_json, atomic_write
from soccer_gif_reid.augment import load_masks, preprocess
from soccer_gif_reid.model import build_model
from soccer_gif_reid.train import cross_domain_circle
from .augment_v2 import CropAugment, QueryAugment, ReferenceAugment
from .metadata import ROOT, DATA
from .numbers import load as load_anchors, label as number_label
from .train import Images, evaluate, paired_forward


def identity_disjoint_ssl(people, rows):
    """Exclude whole auxiliary matches containing any held-out roster identity.

    Match-only separation permits the same player in another match. Anonymous
    auxiliary crops cannot reliably be filtered one player at a time, so unknown
    rosters and every match with a held-out roster member are excluded entirely.
    """
    from .finalize import normalized
    held = [r for r in rows if r['split'] in ('val', 'test')]
    ids = {r['pid'] for r in held}
    names = {normalized(r['name']) for r in held}
    allowed = {}
    for person in people:
        game = person['match']
        if game not in allowed:
            path = DATA/'metadata'/hashlib.sha256(game.encode()).hexdigest()[:12]/'Labels-caption.json'
            if not path.exists():
                allowed[game] = False
                continue
            caption = json.loads(path.read_text())
            roster = [p for side in ('home', 'away') for p in caption['lineup'][side]['players']]
            allowed[game] = not any(p['hash'] in ids or normalized(p.get('long_name') or p['name']) in names
                                    for p in roster)
    return [p for p in people if allowed[p['match']]]


class MatchPairs(Dataset):
    """Labelled cross-domain batches whose core comes from one match, so negatives share a kit."""

    def __init__(self, query, reference, steps=500, batch=16, match_fraction=.75, flat_kit=.5):
        self.q, self.g = defaultdict(list), defaultdict(list)
        for r in query: self.q[r['pid']].append(r)
        for r in reference: self.g[r['pid']].append(r)
        self.ids = sorted(self.q)
        assert set(self.ids) == set(self.g) and len(self.ids) > 1
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        matches = defaultdict(set)
        for r in query: matches[r['game']].add(r['pid'])
        self.matches = [sorted(v) for v in matches.values() if len(v) >= 3]
        self.steps, self.batch = steps, min(batch, len(self.ids))
        self.match_fraction, self.epoch = match_fraction, 0
        self.query_augment = QueryAugment('strong', flat_kit_probability=flat_kit)
        self.reference_augment = ReferenceAugment()

    def identities(self, rng):
        """A match's players first, then random identities to fill the batch."""
        chosen = []
        if self.matches and rng.random() < self.match_fraction:
            core = self.matches[int(rng.integers(len(self.matches)))]
            chosen = list(rng.permutation(core))[:self.batch]
        pool = [p for p in self.ids if p not in set(chosen)]
        extra = rng.choice(len(pool), self.batch - len(chosen), replace=False)
        return chosen + [pool[int(i)] for i in extra]

    def __len__(self): return self.steps

    def __getitem__(self, index):
        rng = np.random.default_rng(seed_for(42, self.epoch, index))
        qs, gs, labels, numbers = [], [], [], []
        people = self.identities(rng)
        for pid in people:
            r = self.q[pid][int(rng.integers(len(self.q[pid])))]
            ref = self.g[pid][int(rng.integers(len(self.g[pid])))]
            image = read_rgb(r['path']); donor = None
            if rng.random() < .5:
                other = people[int(rng.integers(len(people)))]
                d = self.q[other][int(rng.integers(len(self.q[other])))]
                di = read_rgb(d['path']); donor = (di, load_masks(d, di.size)['person'])
            qs.append(preprocess(self.query_augment(image, load_masks(r, image.size), rng, donor)))
            gs.append(preprocess(self.reference_augment(read_rgb(ref['path']), rng)))
            labels.append(self.index[pid])
            # A crop whose digits were just painted over has no number to read: -2 marks it, so the
            # loss can either skip it (v6) or teach the embedding to hold no preference over numbers.
            numbers.append(-2 if self.query_augment.erased else number_label(r.get('number')))
        return (torch.stack(qs), torch.stack(gs), torch.tensor(labels, dtype=torch.long),
                torch.tensor(numbers, dtype=torch.long))


class ActionPairs(Dataset):
    """Self-supervised batches: two annotated views of one player, negatives from the same action."""

    def __init__(self, people, steps=500, batch=16):
        self.actions = defaultdict(list)
        for person in people: self.actions[(person['match'], person['action'])].append(person)
        self.groups = [v for v in self.actions.values() if len(v) >= 2]
        self.people = people
        self.steps, self.batch, self.epoch = steps, batch, 0
        self.augment = CropAugment()

    def identities(self, rng):
        chosen = []
        if self.groups:
            group = self.groups[int(rng.integers(len(self.groups)))]
            chosen = [group[int(i)] for i in rng.permutation(len(group))[:self.batch]]
        while len(chosen) < self.batch:
            chosen.append(self.people[int(rng.integers(len(self.people)))])
        return chosen

    def __len__(self): return self.steps

    def __getitem__(self, index):
        rng = np.random.default_rng(seed_for(7, self.epoch, index))
        a, b, labels = [], [], []
        for label, person in enumerate(self.identities(rng)):
            frames = defaultdict(list)
            for view in person['views']: frames[view['frame_idx']].append(view)
            keys = list(frames)
            first, second = rng.permutation(len(keys))[:2] if len(keys) > 1 else (0, 0)
            one = frames[keys[int(first)]][0]; two = frames[keys[int(second)]][0]
            a.append(preprocess(self.augment(read_rgb(one['path']), rng)))
            b.append(preprocess(self.augment(read_rgb(two['path']), rng)))
            labels.append(label)
        # Unlabelled crops carry no roster number, so the number loss is masked out entirely here.
        return (torch.stack(a), torch.stack(b), torch.tensor(labels, dtype=torch.long),
                torch.full((len(labels),), -1, dtype=torch.long))


def run_epoch(model, loader, optimizer, criterion, schedule, progress, anchors=None, number_weight=0.,
              unknown='mask'):
    """One epoch. `unknown` decides what an erased-number crop teaches.

    'mask' skips it, which is what v6 did, and it left the model reading a confident number off
    crops that show none: on the Argentina-France clip, three common numbers took 297 of 511 reads.
    'uniform' trains such a crop toward equal probability over every number anchor. An embedding
    that sits equidistant from all anchors makes the number half of every prototype score the
    same, so the number stops deciding anything exactly when it cannot be read -- no gate, no
    second head, the same loss on the same embedding.
    """
    model.train(); total = 0.; number_total = 0.; read = 0; seen = 0; began = time.monotonic()
    unknown_total = 0.; unknown_peak = 0.; unknown_seen = 0; number_steps = 0; unknown_steps = 0
    for step, (x, y, labels, numbers) in enumerate(loader):
        for group, lr in zip(optimizer.param_groups, schedule(step)): group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        xf, yf = paired_forward(model, x, y)
        loss = cross_domain_circle(criterion, xf, yf, labels.cuda())
        if number_weight and anchors is not None:
            numbers = numbers.cuda(); valid = numbers >= 0
            erased = (numbers == -2) if unknown == 'uniform' else torch.zeros_like(valid)
            if bool(valid.any()) or bool(erased.any()):
                with torch.autocast('cuda', enabled=False):
                    placed = model.number_adapter(anchors)
                    scale = model.number_adapter.logit_scale.exp().clamp(max=200.)
                    terms, count = [], 0
                    if bool(valid.any()):
                        logits = scale * (xf[valid].float() @ placed.T)
                        number_loss = torch.nn.functional.cross_entropy(logits, numbers[valid], reduction='sum')
                        terms.append(number_loss); count += int(valid.sum())
                        number_total += float(number_loss.detach()) / int(valid.sum()); number_steps += 1
                        read += int((logits.argmax(1) == numbers[valid]).sum()); seen += int(valid.sum())
                    if bool(erased.any()):
                        blank = torch.log_softmax(scale * (xf[erased].float() @ placed.T), dim=1)
                        # Cross-entropy against the uniform distribution over all anchors.
                        unknown_loss = -blank.mean(1).sum()
                        terms.append(unknown_loss); count += int(erased.sum())
                        unknown_total += float(unknown_loss.detach()) / int(erased.sum()); unknown_steps += 1
                        unknown_peak += float(blank.exp().max(1).values.sum()); unknown_seen += int(erased.sum())
                loss = loss + number_weight * torch.stack(terms).sum() / count
        assert torch.isfinite(loss)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.)
        assert torch.isfinite(grad)
        optimizer.step(); total += float(loss.detach())
        if (step + 1) % 25 == 0:
            progress({'step': step + 1, 'loss': total / (step + 1), 'seconds': time.monotonic() - began,
                      'number_loss': number_total / number_steps if number_steps else None,
                      'number_top1': read / seen if seen else None,
                      'unknown_loss': unknown_total / unknown_steps if unknown_steps else None,
                      # Mean top probability on erased crops: 0.01 is perfectly undecided over 100.
                      'unknown_peak_probability': unknown_peak / unknown_seen if unknown_seen else None})
    return total / max(1, len(loader))


def fused_validation(model, anchors, path, split):
    """Per-match validation with fused prototypes, alpha swept on validation itself.

    This is the setting the model is deployed in -- one roster per match, one prototype per
    player that is part photograph and part number -- and it is the only validation number the
    number channel can move. Whole-split photo mAP, which v6 selected on, cannot see it at all.
    """
    from .number_channel import score_split
    saved = torch.load(path, map_location='cpu', weights_only=True)
    q = torch.nn.functional.normalize(saved['query'].float(), dim=-1)
    g = torch.nn.functional.normalize(saved['gallery'].float(), dim=-1)
    model.eval()
    with torch.no_grad(): adapted = model.number_adapter(anchors).float().cpu()
    meta = split['query']; numbers = {}
    for r in meta: numbers.setdefault(r['pid'], number_label(r.get('number')))
    ref_ids = [r['pid'] for r in split['reference']]
    sweep = {a: score_split(q, g, meta, ref_ids, adapted, numbers, a)
             for a in (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)}
    alpha = max(sweep, key=lambda a: sweep[a]['MRR'])
    return {'alpha': alpha, 'MRR': sweep[alpha]['MRR'], 'rank1': sweep[alpha]['rank1'],
            'photo_only_MRR': sweep[1.]['MRR'], 'photo_only_rank1': sweep[1.]['rank1']}


def payload(model, config, metrics, stage, epoch, manifest_hash):
    return {'format': 'mupnit-native-quality-v3', 'kind': 'clip',
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': dict(config, stage=stage, unfrozen_blocks=model.unfrozen_blocks),
        'epoch': epoch, 'metrics': metrics, 'manifest_sha256': manifest_hash}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'runs/soccernet_full_v3')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--ssl-epochs', type=int, default=0,
        help='optional auxiliary pretraining; whole-match roster checks enforce player separation')
    ap.add_argument('--head-epochs', type=int, default=30)
    ap.add_argument('--finetune-epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--ssl-persons', type=int, default=40000)
    ap.add_argument('--number-weight', type=float, default=.5,
        help='weight on the shirt-number term; 0 reproduces the run before the number channel')
    ap.add_argument('--number-unknown', choices=('mask', 'uniform'), default='mask',
        help='what an erased-number crop teaches: skipped (v6) or no preference over numbers')
    ap.add_argument('--select', choices=('photo', 'fused'), default='photo',
        help='checkpoint selection: whole-split photo mAP (v6), or per-match fused validation MRR')
    ap.add_argument('--flat-kit', type=float, default=.2,
        help='probability that a training crop has its shirt number painted over')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); out = args.out
    seed_everything(42); torch.set_num_threads(4)
    audit = json.loads((DATA/'final_audit.json').read_text())
    assert audit['review_passed'] is True
    manifest_hash = digest_file(DATA/'manifest.jsonl')
    assert audit['manifest_sha256'] == manifest_hash
    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    sets = {s: {d: [r for r in rows if r['split'] == s and r['domain'] == d]
                for d in ['query', 'reference']} for s in ['train', 'val', 'test']}
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        for key in ['pid','name','sha256']:
            assert not {r[key] for d in sets[a].values() for r in d} & {r[key] for d in sets[b].values() for r in d}
        assert not {r['game'] for r in sets[a]['query']} & {r['game'] for r in sets[b]['query']}

    people = ([json.loads(s) for s in (DATA/'ssl_pairs.jsonl').read_text().splitlines()]
              if args.ssl_epochs else [])
    people = identity_disjoint_ssl(people, rows)
    held_out = {r.get('game') for r in rows if r['split'] in ('val','test')} - {None}
    leaked = [p for p in people if p['match'] in held_out]
    assert not leaked, f'{len(leaked)} self-supervised persons come from a held-out match'
    if args.ssl_persons: people = people[:args.ssl_persons]
    if args.ssl_epochs and len(people) < args.batch:
        raise ValueError('Not enough identity-disjoint auxiliary groups; use --ssl-epochs 0')

    out.mkdir(exist_ok=True, parents=True)
    if any(out.iterdir()) and not args.resume: raise FileExistsError('Existing run: use --resume')
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status') == 'complete': return
    config = {'experiment': out.name, 'steps': args.steps, 'batch': args.batch,
        'ssl_epochs': args.ssl_epochs, 'head_epochs': args.head_epochs, 'finetune_epochs': args.finetune_epochs,
        'ssl_persons': len(people), 'seed': 42,
        'ssl_identity_policy': 'exclude entire matches with held-out roster identities or unavailable rosters',
        'ssl_source_sha256': digest_file(DATA/'ssl_pairs.jsonl') if args.ssl_epochs else None,
        'ssl_selected_people_sha256': hashlib.sha256(json.dumps([p['person'] for p in people]).encode()).hexdigest(),
        'selection': 'validation per-match fused MRR' if args.select == 'fused' else 'validation image-gallery mAP',
        'quality_degradation': False, 'reference_augmentation': True, 'match_batches': True,
        'number_weight': args.number_weight, 'flat_kit_probability': args.flat_kit,
        'number_unknown': args.number_unknown, 'checkpoint_selection': args.select,
        'number_anchors_sha256': digest_file(ROOT/'weights/number_text.pt') if args.number_weight else None,
        'training_code_sha256': digest_file(Path(__file__)),
        'augmentation_code_sha256': digest_file(ROOT/'experiments/soccernet_full_v2/augment_v2.py'),
        'model_code_sha256': digest_file(ROOT/'src/soccer_gif_reid/model.py')}
    state = torch.load(out/'last.pt', map_location='cpu', weights_only=False) if args.resume and (out/'last.pt').exists() else None
    if state:
        assert state['manifest_sha256'] == manifest_hash
        for key in config: assert state['config'][key] == config[key], key
    atomic_json(out/'config.json', dict(config, manifest_sha256=manifest_hash))

    model = build_model('cuda:0', ROOT/'weights', numbers=bool(args.number_weight))
    anchors = load_anchors('cuda:0') if args.number_weight else None
    criterion = CircleLoss(m=.4, gamma=80)
    labelled = MatchPairs(sets['train']['query'], sets['train']['reference'], args.steps, args.batch,
                          flat_kit=args.flat_kit)
    unlabelled = ActionPairs(people, args.steps, args.batch)
    best = state['best'] if state else -1.
    best_photo = state.get('best_photo', -1.) if state else -1.
    stages = [('ssl', unlabelled, 2, args.ssl_epochs, .0005, .000002),
              ('head', labelled, 0, args.head_epochs, .001, 0.),
              ('finetune', labelled, 2, args.finetune_epochs, .0001, .000001)]

    for stage_index, (stage, data, blocks, epochs, head_lr, backbone_lr) in enumerate(stages):
        if state and stage_index < state['stage_index']: continue
        if not epochs: continue
        model.set_trainable_blocks(blocks)
        head = list(model.projection.parameters())
        if model.number_adapter is not None: head += list(model.number_adapter.parameters())
        groups = [{'params': head, 'lr': head_lr}]
        if blocks: groups.append({'params': [p for p in model.visual.parameters() if p.requires_grad], 'lr': backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
        start, stale = 0, 0
        if state and stage_index == state['stage_index']:
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start, stale = state['epoch'] + 1, state['stale']
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif stage_index and (out/'best.pt').exists():
            model.load_state_dict(torch.load(out/'best.pt', map_location='cpu', weights_only=False)['model'])
        for epoch in range(start, epochs):
            if stale >= 8: break
            data.epoch = epoch + stage_index * 1000
            loader = DataLoader(data, batch_size=None, num_workers=args.workers, pin_memory=True)
            def schedule(step, epoch=epoch, epochs=epochs):
                fraction = (epoch + (step + 1) / len(data)) / epochs
                warmup = min(1., (epoch * len(data) + step + 1) / (2 * len(data)))
                factor = warmup * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
                return [head_lr * factor, backbone_lr * factor]
            began = time.monotonic()
            loss = run_epoch(model, loader, optimizer, criterion, schedule,
                lambda p, stage=stage, epoch=epoch: atomic_json(out/'progress.json',
                    dict(p, stage=stage, epoch=epoch + 1, steps=len(data))),
                anchors=anchors, number_weight=args.number_weight, unknown=args.number_unknown)
            fused = None
            if model.number_adapter is not None and stage != 'ssl':
                metrics = evaluate(model, sets['val']['query'], sets['val']['reference'], workers=args.workers,
                                   artifact=out/'val_epoch_features.pt')
                fused = fused_validation(model, anchors, out/'val_epoch_features.pt', sets['val'])
            else:
                metrics = evaluate(model, sets['val']['query'], sets['val']['reference'], workers=args.workers)
            # The self-supervised stage never selects a checkpoint: it has not seen a label yet,
            # and its job is to hand the labelled stages a better starting point, not to win.
            score = fused['MRR'] if args.select == 'fused' and fused else metrics['mAP']
            improved = score > best and stage != 'ssl'
            if improved: best, stale = score, 0
            # Under fused selection, also keep what v6's rule would have kept, so the effect of the
            # selection rule and the effect of the training change can be measured separately.
            if args.select == 'fused' and stage != 'ssl' and metrics['mAP'] > best_photo:
                best_photo = metrics['mAP']
                photo_ckpt = payload(model, config, metrics, stage, epoch, manifest_hash)
                atomic_write(out/'best_photo.pt', lambda p: torch.save(photo_ckpt, p))
                atomic_json(out/'best_photo_metrics.json', {'stage': stage, 'epoch': epoch + 1, 'fused': fused, **metrics})
            if not improved and stage != 'ssl': stale += 1
            ckpt = payload(model, config, metrics, stage, epoch, manifest_hash)
            if improved or (stage == 'ssl' and epoch == epochs - 1):
                atomic_write(out/('best.pt' if improved else 'ssl_final.pt'), lambda p: torch.save(ckpt, p))
                if improved: atomic_json(out/'best_metrics.json', {'stage': stage, 'epoch': epoch + 1,
                                                                   'selected_on': args.select, 'fused': fused, **metrics})
            ckpt.update(optimizer=optimizer.state_dict(), stage_index=stage_index, best=best, stale=stale,
                best_photo=best_photo,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            atomic_write(out/'last.pt', lambda p: torch.save(ckpt, p))
            with (out/'history.jsonl').open('a') as f:
                f.write(json.dumps({'stage': stage, 'epoch': epoch + 1, 'loss': loss, 'validation': metrics,
                    'fused_validation': fused, 'promoted': improved, 'seconds': time.monotonic() - began}) + '\n')
            print(stage, epoch + 1, metrics['rank1'], metrics['mAP'],
                  f"fused MRR {fused['MRR']:.4f} alpha {fused['alpha']}" if fused else '', flush=True)
        if stage == 'ssl' and (out/'ssl_final.pt').exists():
            atomic_write(out/'best.pt', lambda p: torch.save(torch.load(out/'ssl_final.pt', map_location='cpu', weights_only=False), p))
        state = None

    model.load_state_dict(torch.load(out/'best.pt', map_location='cpu', weights_only=False)['model'])
    evaluate(model, sets['val']['query'], sets['val']['reference'], workers=args.workers, artifact=out/'val_features.pt')
    test = evaluate(model, sets['test']['query'], sets['test']['reference'], workers=args.workers, artifact=out/'test_features.pt')
    atomic_json(out/'test_metrics.json', test)
    baselines = {}
    # The baselines are image-only by definition, so they are built without the number adapter.
    fresh = build_model('cuda:0', ROOT/'weights')
    baselines['frozen_clip'] = evaluate(fresh, sets['test']['query'], sets['test']['reference'], workers=args.workers, base=True, artifact=out/'frozen_clip_test_features.pt')
    nba = torch.load(ROOT/'runs/ap3d_no_quality_v2/clip/best.pt', map_location='cpu', weights_only=False)
    fresh.load_state_dict(nba['model'])
    baselines['nba_projection'] = evaluate(fresh, sets['test']['query'], sets['test']['reference'], workers=args.workers, artifact=out/'nba_test_features.pt')
    atomic_json(out/'baselines.json', baselines)
    atomic_json(out/'status.json', {'status': 'complete', 'test': test, 'manifest_sha256': manifest_hash})


if __name__ == '__main__': main()
