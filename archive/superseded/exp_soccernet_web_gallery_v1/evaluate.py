"""Evaluate our v2 checkpoints against independently supplied web references."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from soccer_gif_reid.common import atomic_json, digest_file, read_rgb, write_manifest
from soccer_gif_reid.metrics import retrieval_metrics
from experiments.ap3d_no_quality_v2.data import transform
from experiments.ap3d_no_quality_v2.evaluate import load_model
from experiments.ap3d_no_quality_v2.model import encode_clip


class Images(Dataset):
    def __init__(self, items, kind):
        self.items, self.kind = items, kind

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        paths = self.items[index]['paths']
        frames = [transform(read_rgb(p), self.kind) for p in paths]
        if self.kind == 'ap3d' and len(frames) == 1:
            frames *= 4  # A still image is a static clip, not four observations.
        return torch.stack(frames)


@torch.inference_mode()
def extract(model, kind, items, device, base=False):
    out = []
    loader = DataLoader(Images(items, kind), batch_size=8, num_workers=4, pin_memory=True)
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=str(device).startswith('cuda')):
            if base:
                b, t, c, h, w = batch.shape
                features = F.normalize(model.encode_base(batch.reshape(b*t,c,h,w)).reshape(b,t,-1).mean(1), dim=-1)
            else:
                features = encode_clip(model, kind, batch)
        out.append(features.float().cpu())
    result = torch.cat(out)
    assert torch.isfinite(result).all()
    assert torch.all(result.norm(dim=1) > 0)
    return F.normalize(result, dim=1)


def validate_inputs(queries, gallery, references_per_player):
    if not queries or not gallery:
        raise ValueError('Queries and independently labeled reference images are required')
    counts = Counter(r['pid'] for r in gallery)
    if references_per_player and set(counts.values()) != {references_per_player}:
        raise ValueError(f'Expected exactly {references_per_player} references per player: {counts}')
    if set(r['pid'] for r in queries) - set(counts):
        raise ValueError('Queries without a verified reference identity')
    qpaths = {str(Path(p).resolve()) for r in queries for p in r['paths']}
    gpaths = {str(Path(r['path']).resolve()) for r in gallery}
    if qpaths & gpaths:
        raise ValueError('Query/gallery image overlap')
    qhashes = {digest_file(p) for p in qpaths}
    ghashes = [digest_file(r['path']) for r in gallery]
    if len(set(ghashes)) != len(ghashes) or qhashes & set(ghashes):
        raise ValueError('Duplicate reference bytes or query/reference leakage')
    for row in queries:
        if len(row['paths']) not in (1,4) or len(set(row['paths'])) != len(row['paths']):
            raise ValueError('Queries must contain one or four distinct real observations')


def score(query_features, gallery_features, queries, gallery):
    qids, gids = [r['pid'] for r in queries], [r['pid'] for r in gallery]
    metrics = retrieval_metrics(query_features, gallery_features, qids, gids)
    ids = sorted(set(gids))
    prototypes = torch.stack([F.normalize(gallery_features[[i for i,p in enumerate(gids) if p == pid]].mean(0),dim=0) for pid in ids])
    identity_metrics = retrieval_metrics(query_features, prototypes, qids, ids)
    # With one prototype per identity AP is reciprocal rank; name it explicitly.
    identity_metrics['MRR'] = identity_metrics.pop('mAP')
    identity_metrics['macro_identity_MRR'] = identity_metrics.pop('macro_identity_mAP')
    for entry in identity_metrics['per_identity'].values():
        entry['MRR'] = entry.pop('mAP')
    sims = query_features @ prototypes.T
    predictions = []
    for row, values in zip(queries, sims):
        order = torch.argsort(values, descending=True, stable=True).tolist()
        predictions.append({'query_id': row['query_id'], 'gt_pid': row['pid'],
            'paths': row['paths'], 'ranked_identities': [
                {'pid':ids[i], 'similarity':float(values[i])} for i in order]})
    return {'image_gallery':metrics, 'roster_prototypes':identity_metrics}, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', required=True)
    p.add_argument('--gallery')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--references-per-player', type=int, default=8)
    p.add_argument('--base', action='store_true', help='Frozen CLIP backbone without trained projection, same letterbox inputs')
    a = p.parse_args(); torch.set_num_threads(4)
    queries = [json.loads(s) for s in Path(a.queries).read_text().splitlines()]
    gallery = [json.loads(s) for s in Path(a.gallery).read_text().splitlines()] if a.gallery else None
    if gallery:
        validate_inputs(queries, gallery, a.references_per_player)
    model, payload = load_model(a.checkpoint, a.device)
    kind = payload['kind']
    if a.base and kind != 'clip':
        raise ValueError('--base requires a CLIP checkpoint')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    features = extract(model, kind, queries, a.device, base=a.base)
    torch.save({'features':features, 'queries':queries,
                'checkpoint_sha256':digest_file(a.checkpoint), 'base':a.base,
                'queries_sha256':digest_file(a.queries)}, out/'query_features.pt')
    if gallery is None:
        atomic_json(out/'status.json', {'query_features_extracted':len(queries),
            'web_gallery_evaluated':False, 'reason':'Verified named web gallery has not been supplied'})
        print(f'Cached {len(queries)} query features; no gallery metric claimed.', flush=True)
        return
    gf = extract(model, kind, [dict(r,paths=[r['path']]) for r in gallery], a.device, base=a.base)
    metrics, predictions = score(features, gf, queries, gallery)
    result = {'kind':kind, 'trained_projection_used':kind=='clip' and not a.base,
        'our_trained_checkpoint':not a.base, 'authors_released_reid_checkpoint':False,
        'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':digest_file(a.checkpoint),
        'queries_sha256':digest_file(a.queries),'gallery_sha256':digest_file(a.gallery),
        'reference_counts':dict(Counter(r['pid'] for r in gallery)),
        'preprocessing':'Checkpoint-native aspect-preserving letterbox; no test augmentation or quality degradation',
        'score_type':'Cosine similarity; not a calibrated probability', **metrics}
    torch.save({'query':features,'gallery':gf},out/'features.pt')
    atomic_json(out/'metrics.json', result); write_manifest(out/'predictions.jsonl', predictions)
    print(json.dumps({k:{m:result[k][m] for m in ('rank1','mAP' if k=='image_gallery' else 'MRR')} for k in metrics}),flush=True)


if __name__ == '__main__':
    main()
