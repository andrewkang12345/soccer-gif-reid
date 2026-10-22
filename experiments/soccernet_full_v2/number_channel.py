"""The shirt number as part of each identity's prototype, in the model's own embedding space.

A roster gives every player a photograph *and* a number. Until now only the photograph was used,
and the error anatomy says that is where the loss is: within a match both candidates wear the same
kit at the same distance, and the number is the only field that is unique by construction.

This adds the number without adding a second model. One frozen CLIP text anchor per number is
placed into the projection space by a single linear map trained inside this network, and each
identity's prototype becomes one vector -- part photograph, part number:

    p = normalize(alpha * photo_prototype + (1 - alpha) * number_anchor)

Retrieval stays a single dot product against a single prototype. There is no second score to
combine, no gate, and no separate recogniser: the query embedding that matches the photograph is
the same embedding that has to match the number.

`alpha` is chosen on validation and applied unchanged to test.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import atomic_json,digest_file
from soccer_gif_reid.model import build_model
from .metadata import DATA,ROOT
from .numbers import load as load_anchors,label
from .train import Images


class NumberAdapter(nn.Module):
    """Places the frozen text anchors in the projection space. 590k parameters, one linear map."""

    def __init__(self, dim=768):
        super().__init__()
        self.map = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(1/.07).log())

    def forward(self, anchors):
        return F.normalize(self.map(anchors), dim=-1)


def encode(model, rows, workers=6, batch=16):
    model.eval(); out = []
    with torch.inference_mode():
        for x in DataLoader(Images(rows), batch_size=batch, num_workers=workers, pin_memory=True):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(model(x.cuda()).float().cpu())
    return torch.cat(out)


def cached(model, rows, path, workers=6):
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['paths'] == [r['path'] for r in rows]: return saved['features']
    features = encode(model, rows, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'features': features, 'paths': [r['path'] for r in rows]}, path)
    return features


def fit(features, labels, anchors, steps=3000, batch=256, lr=1e-3, seed=42):
    """Train the adapter so a query embedding lands on its own number's anchor."""
    torch.manual_seed(seed)
    device = 'cuda:0'
    adapter = NumberAdapter(anchors.shape[1]).to(device)
    anchors = anchors.to(device)
    x = F.normalize(features, dim=-1).to(device); y = labels.to(device)
    keep = y >= 0; x, y = x[keep], y[keep]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    history = []
    for step in range(steps):
        index = torch.randint(0, len(x), (min(batch, len(x)),), generator=generator).to(device)
        logits = adapter.logit_scale.exp() * (x[index] @ adapter(anchors).T)
        loss = F.cross_entropy(logits, y[index])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad(): adapter.logit_scale.clamp_(0, np.log(200))
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                accuracy = float(((x @ adapter(anchors).T).argmax(1) == y).float().mean())
            history.append({'step': step + 1, 'loss': float(loss), 'train_top1': accuracy})
            print(f"  step {step+1} loss {float(loss):.3f} train top-1 over 100 numbers {100*accuracy:.1f}%", flush=True)
    return adapter.eval(), history


def groups(meta):
    out = defaultdict(list)
    for i, r in enumerate(meta): out[(r['pid'], r['game'])].append(i)
    return out


def score_split(q, refs, meta, ref_ids, adapted, numbers, alpha, tracklet=False):
    """Per-match retrieval against one fused prototype per roster player."""
    photo = {}
    rows = defaultdict(list)
    for i, pid in enumerate(ref_ids): rows[pid].append(i)
    for pid, idx in rows.items(): photo[pid] = F.normalize(refs[idx].mean(0), dim=0)
    units = groups(meta) if tracklet else {(r['pid'], r['game'], i): [i] for i, r in enumerate(meta)}
    by_game = defaultdict(list)
    for key in units: by_game[key[1]].append(key)
    ranks = []
    for game, keys in by_game.items():
        roster = sorted({k[0] for k in keys if k[0] in photo})
        if len(roster) < 2: continue
        parts = []
        for pid in roster:
            anchor = adapted[numbers[pid]] if numbers.get(pid, -1) >= 0 else torch.zeros_like(photo[pid])
            parts.append(F.normalize(alpha * photo[pid] + (1 - alpha) * anchor, dim=0))
        matrix = torch.stack(parts); arr = np.asarray(roster)
        for key in keys:
            if key[0] not in photo: continue
            e = F.normalize(q[units[key]].mean(0), dim=0)
            order = torch.argsort(matrix @ e, descending=True, stable=True).numpy()
            ranks.append(1 + int(np.flatnonzero(arr[order] == key[0])[0]))
    ranks = np.asarray(ranks)
    return {'rank1': float(np.mean(ranks == 1)), 'top3': float(np.mean(ranks <= 3)),
            'MRR': float(np.mean(1 / ranks)), 'median_rank': int(np.median(ranks)), 'units': len(ranks)}


def number_accuracy(q, meta, adapted, restrict_roster=True):
    """How often the fused embedding's nearest number anchor is the right one."""
    by_game = defaultdict(set)
    for r in meta:
        n = label(r.get('number'))
        if n >= 0: by_game[r['game']].add(n)
    hits, chance = [], []
    scores = F.normalize(q, dim=-1) @ adapted.T
    for i, r in enumerate(meta):
        n = label(r.get('number'))
        if n < 0: continue
        candidates = sorted(by_game[r['game']]) if restrict_roster else list(range(adapted.shape[0]))
        pick = candidates[int(torch.argmax(scores[i, candidates]))]
        hits.append(pick == n); chance.append(1 / len(candidates))
    return {'top1': float(np.mean(hits)), 'chance': float(np.mean(chance)), 'queries': len(hits)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    sets = {s: {d: [r for r in rows if r['split'] == s and r['domain'] == d]
                for d in ('query', 'reference')} for s in ('train', 'val', 'test')}
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    trained_in = 'number_adapter.map.weight' in payload['model']
    model = build_model('cuda:0', ROOT/'weights', numbers=trained_in)
    model.load_state_dict(payload['model']); model.eval()
    # One cache per checkpoint: a run directory can hold more than one (best.pt, best_photo.pt),
    # and features cached for one must never be scored as the other's.
    stem = args.checkpoint.stem
    cache = args.checkpoint.parent/('number_cache' if stem == 'best' else f'number_cache_{stem}')
    anchors = load_anchors()

    if trained_in:
        # The adapter trained jointly with the image head: take it as it is and only choose alpha.
        adapter, history = model.number_adapter.eval(), []
        print('using the adapter trained inside the checkpoint', flush=True)
    else:
        train_q = cached(model, sets['train']['query'], cache/'train_query.pt', args.workers)
        labels = torch.tensor([label(r.get('number')) for r in sets['train']['query']])
        print(f"fitting the adapter on {int((labels>=0).sum())} labelled train crops "
              f"against a frozen model", flush=True)
        adapter, history = fit(train_q, labels, anchors, steps=args.steps)
    with torch.no_grad(): adapted = adapter(anchors.cuda()).cpu()

    out = {}
    for split in ('val', 'test'):
        q = cached(model, sets[split]['query'], cache/f'{split}_query.pt', args.workers)
        g = cached(model, sets[split]['reference'], cache/f'{split}_reference.pt', args.workers)
        q = F.normalize(q, dim=-1); g = F.normalize(g, dim=-1)
        meta = sets[split]['query']
        numbers = {}
        for r in meta: numbers.setdefault(r['pid'], label(r.get('number')))
        ref_ids = [r['pid'] for r in sets[split]['reference']]
        out[split] = {'number_reading': number_accuracy(q, meta, adapted),
                      'number_reading_over_100': number_accuracy(q, meta, adapted, restrict_roster=False),
                      'sweep': {}, 'tracklet_sweep': {}}
        for alpha in np.round(np.arange(0, 1.05, .1), 2):
            out[split]['sweep'][float(alpha)] = score_split(q, g, meta, ref_ids, adapted, numbers, float(alpha))
            out[split]['tracklet_sweep'][float(alpha)] = score_split(q, g, meta, ref_ids, adapted, numbers, float(alpha), tracklet=True)
    best = max(out['val']['sweep'], key=lambda a: out['val']['sweep'][a]['MRR'])
    best_track = max(out['val']['tracklet_sweep'], key=lambda a: out['val']['tracklet_sweep'][a]['MRR'])
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest_file(args.checkpoint),
        'adapter_parameters': sum(p.numel() for p in adapter.parameters()),
        'adapter_trained_jointly': trained_in,
        'training_history': history,
        'alpha_chosen_on_validation': {'per_crop': best, 'per_tracklet': best_track},
        'validation': {'per_crop': out['val']['sweep'][best], 'per_tracklet': out['val']['tracklet_sweep'][best_track]},
        'test': {'per_crop': out['test']['sweep'][best], 'per_tracklet': out['test']['tracklet_sweep'][best_track],
                 'photo_only_per_crop': out['test']['sweep'][1.0], 'photo_only_per_tracklet': out['test']['tracklet_sweep'][1.0],
                 'number_only_per_crop': out['test']['sweep'][0.0], 'number_only_per_tracklet': out['test']['tracklet_sweep'][0.0]},
        'number_reading': {s: out[s]['number_reading'] for s in ('val', 'test')},
        'number_reading_over_100': {s: out[s]['number_reading_over_100'] for s in ('val', 'test')},
        'full_sweeps': {s: {'per_crop': out[s]['sweep'], 'per_tracklet': out[s]['tracklet_sweep']} for s in ('val', 'test')},
        'caveat': ('The adapter trained jointly with the image head.' if trained_in else
                   'The adapter is fitted on train queries against a frozen re-identification '
                   'model, so the image head is still actively discarding number information.')}
    name = 'number_channel.json' if args.checkpoint.parent.name == 'soccernet_full_v6' and stem == 'best' else \
        f'number_channel_{args.checkpoint.parent.name}_{stem}.json'
    atomic_json(ROOT/'reports/soccernet_full_v2'/name, report)
    cache.mkdir(parents=True, exist_ok=True)
    torch.save({'adapter': adapter.state_dict(), 'adapted_anchors': adapted}, cache/'adapter.pt')
    print(json.dumps({k: report[k] for k in ('alpha_chosen_on_validation', 'validation', 'test',
                                             'number_reading', 'number_reading_over_100')}, indent=2))


if __name__ == '__main__': main()
