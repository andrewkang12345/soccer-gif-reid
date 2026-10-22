"""Audit the completed run and recount rankings without using its scoring functions.

This consumes the frozen split and saved embeddings. It does not select a checkpoint,
change alpha, train, or turn identity-by-match groups into genuine tracker output.
"""
import argparse
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from soccer_gif_reid.common import atomic_json, digest_file
from .metadata import DATA, ROOT
from .finalize import normalized


def unit(x):
    a = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    assert np.isfinite(a).all() and np.all(n > 0)
    return a / n


def ranking(q, refs, meta, ref_ids, anchors, alpha, pooled):
    ref_ids = np.asarray(ref_ids)
    photo = {pid: unit(refs[ref_ids == pid].mean(0)) for pid in sorted(set(ref_ids))}
    game_rows = defaultdict(list)
    for i, r in enumerate(meta):
        game_rows[r['game']].append(i)
    ranks = []
    for game, indices in sorted(game_rows.items()):
        roster = sorted({meta[i]['pid'] for i in indices})
        numbers = {pid: {int(meta[i]['number']) for i in indices if meta[i]['pid'] == pid} for pid in roster}
        assert all(len(ns) == 1 for ns in numbers.values())
        prototypes = np.stack([unit(alpha * photo[pid] + (1-alpha) * anchors[next(iter(numbers[pid]))]) for pid in roster])
        if pooled:
            observations = [(pid, unit(q[[i for i in indices if meta[i]['pid'] == pid]].mean(0))) for pid in roster]
        else:
            observations = [(meta[i]['pid'], q[i]) for i in indices]
        scores = np.stack([e for _, e in observations]) @ prototypes.T
        for (pid, _), row in zip(observations, scores):
            order = np.argsort(-row, kind='stable')
            ranks.append(1 + int(np.flatnonzero(np.asarray(roster)[order] == pid)[0]))
    a = np.asarray(ranks)
    return {'rank1': float(np.mean(a == 1)), 'top3': float(np.mean(a <= 3)),
            'MRR': float(np.mean(1/a)), 'median_rank': int(np.median(a)), 'units': len(a)}


def image_retrieval(q, g, qids, gids, games=None):
    gids = np.asarray(gids); qids = np.asarray(qids); values = []
    if games is None:
        groups = {'all': list(range(len(q)))}
    else:
        groups = defaultdict(list)
        for i, game in enumerate(games): groups[game].append(i)
    for indices in groups.values():
        allowed = np.arange(len(g)) if games is None else np.flatnonzero(np.isin(gids, qids[indices]))
        sim = q[indices] @ g[allowed].T
        for i, row in zip(indices, sim):
            hits = gids[allowed[np.argsort(-row, kind='stable')]] == qids[i]
            positions = np.flatnonzero(hits)
            assert len(positions)
            values.append((float(hits[0]), float(np.mean(np.arange(1,len(positions)+1)/(positions+1)))))
    return {'rank1': float(np.mean([v[0] for v in values])), 'mAP': float(np.mean([v[1] for v in values])), 'queries': len(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'runs/soccernet_full_v7')
    args = ap.parse_args(); run = args.run
    torch.set_num_threads(4)
    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    audit = json.loads((DATA/'final_audit.json').read_text())
    config = json.loads((run/'config.json').read_text())
    manifest_hash = digest_file(DATA/'manifest.jsonl')
    assert manifest_hash == audit['manifest_sha256'] == config['manifest_sha256']
    sets = {s: [r for r in rows if r['split'] == s] for s in ('train','val','test')}
    for a,b in itertools.combinations(sets,2):
        for key in ('pid','sha256'):
            assert not {r[key] for r in sets[a]} & {r[key] for r in sets[b]}, (a,b,key)
        assert not {normalized(r['name']) for r in sets[a]} & {normalized(r['name']) for r in sets[b]}
        assert not {r['game'] for r in sets[a] if r['domain']=='query'} & {r['game'] for r in sets[b] if r['domain']=='query'}
    source_q = {r['sha256'] for r in rows if r['domain']=='query'}
    source_g = {r['sha256'] for r in rows if r['domain']=='reference'}
    stored_g = {r['gallery_sha256'] for r in rows if r['domain']=='reference'}
    assert not source_q & (source_g | stored_g)
    assert len(stored_g) == sum(r['domain']=='reference' for r in rows)
    def check(r):
        expected = r.get('gallery_sha256',r['sha256'])
        assert digest_file(r['path']) == expected, r['path']
        if r['domain']=='reference': assert digest_file(r['candidate_path']) == r['sha256'],r['candidate_path']
        if r['domain']=='query': assert Path(r['person_mask']).is_file(),r['person_mask']
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(check, rows))
    print('Manifest, all stored image/source hashes, masks and split separation passed',flush=True)
    checkpoint_hash = digest_file(run/'best.pt')
    report_path = ROOT/'reports/soccernet_full_v2'/f'number_channel_{run.name}_best.json'
    old = json.loads(report_path.read_text())
    assert checkpoint_hash == old['checkpoint_sha256']
    adapter = torch.load(run/'number_cache/adapter.pt',map_location='cpu',weights_only=True)
    anchors = unit(adapter['adapted_anchors'].numpy())
    out = {}
    for split in ('val','test'):
        meta = [r for r in sets[split] if r['domain']=='query']
        refs = [r for r in sets[split] if r['domain']=='reference']
        qc = torch.load(run/f'number_cache/{split}_query.pt',map_location='cpu',weights_only=True)
        gc = torch.load(run/f'number_cache/{split}_reference.pt',map_location='cpu',weights_only=True)
        assert qc['paths']==[r['path'] for r in meta] and gc['paths']==[r['path'] for r in refs]
        q,g = unit(qc['features'].numpy()),unit(gc['features'].numpy())
        out[split] = {}
        for key,pooled in [('per_crop',False),('per_tracklet',True)]:
            alpha = old['alpha_chosen_on_validation'][key]
            result = ranking(q,g,meta,[r['pid'] for r in refs],anchors,alpha,pooled)
            expected = old['validation' if split=='val' else 'test'][key]
            for k in result: assert abs(result[k]-expected[k])<1e-6,(split,key,k,result[k],expected[k])
            out[split][key] = result
        out[split]['image_retrieval'] = image_retrieval(q,g,[r['pid'] for r in meta],[r['pid'] for r in refs])
        out[split]['per_match_image_retrieval'] = image_retrieval(q,g,[r['pid'] for r in meta],[r['pid'] for r in refs],[r['game'] for r in meta])
    saved=torch.load(run/'test_features.pt',map_location='cpu',weights_only=True)
    cached=torch.load(run/'number_cache/test_query.pt',map_location='cpu',weights_only=True)
    assert saved['query_paths']==cached['paths']
    max_delta=float((saved['query']-cached['features']).abs().max())
    assert max_delta<1e-5,max_delta
    baselines={}
    for name,filename in [('frozen_clip','frozen_clip_test_features.pt'),('nba_projection','nba_test_features.pt')]:
        x=torch.load(run/filename,map_location='cpu',weights_only=True)
        assert x['query_paths']==saved['query_paths'] and x['gallery_paths']==saved['gallery_paths']
        q,g=unit(x['query'].numpy()),unit(x['gallery'].numpy())
        baselines[name]={'image_retrieval':image_retrieval(q,g,x['query_ids'],x['gallery_ids']),
                         'per_match_image_retrieval':image_retrieval(q,g,x['query_ids'],x['gallery_ids'],[r['game'] for r in sets['test'] if r['domain']=='query'])}
    result={'passed':True,'run':str(run),'manifest_sha256':manifest_hash,'checkpoint_sha256':checkpoint_hash,
        'verification_scope':'Stored data, supervised split separation and independent score recount; auxiliary player separation is reported separately.',
        'stored_images_verified':len(rows),'source_photos_verified':len(source_g),'supervised_split_separation':['roster player ID','normalized name','match','source hash','stored crop hash'],
        'auxiliary_training_disabled':config.get('ssl_epochs',0)==0,
        'soccer_training_identity_disjoint':config.get('ssl_epochs',0)==0,
        'cached_vs_training_feature_max_abs_difference':max_delta,'metrics':out,'baselines':baselines,
        'image_retrieval_precision_note':'Independent float64 scoring can reorder nearly tied images relative to float32 training metrics.',
        'caveats':['Pooled units group crops by ground-truth player and match, not detected McByte++ tracklets.',
                   'Candidate rosters contain reference-covered identities observed in the labelled match queries, not every active squad member.',
                   'The test was reused in the recorded experiment series; this recount is verification, not a new untouched test.',
                   'Weak labels and unknown CLIP-pretraining overlap remain.']}
    if config.get('ssl_epochs',0):
        result['caveats'].append('Auxiliary pretraining is not certified player-disjoint by the supervised-manifest checks. See ssl_identity_audit.json for proven overlap in v7.')
    path=ROOT/'reports/soccernet_full_v2'/f'verification_{run.name}.json'
    atomic_json(path,result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
