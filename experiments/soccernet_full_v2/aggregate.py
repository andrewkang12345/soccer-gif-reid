"""Compare ways of turning a player's several reference photographs into a score.

Inference currently averages a player's reference embeddings into one vector. That is the right
baseline, but a single centroid blurs a head-and-shoulders portrait, a full-body action shot and
an off-axis view into one point, and the true appearance is closer to a few modes than to one.
Compared here: every reference ranked on its own, one mean prototype, the best single reference,
and k prototypes per player from clustering.

Each rule is scored per match, because that is the deployment setting, and the winner is chosen on
**validation**. Picking it by its test score would be choosing a method on the test set; test is
reported for every rule only so the choice can be audited.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT
from .report import query_games


def unit(x):return x/np.clip(np.linalg.norm(x,axis=-1,keepdims=True),1e-9,None)


def prototypes(vectors,k,seed=0):
    """k modes of one player's references; plain Lloyd iterations, deterministic seeding."""
    if k<=1 or len(vectors)<=k:return unit(vectors if k>1 else vectors.mean(0,keepdims=True))
    rng=np.random.default_rng(seed);centres=vectors[rng.permutation(len(vectors))[:k]].copy()
    for _ in range(25):
        assign=np.argmax(vectors@centres.T,axis=1)
        moved=np.stack([vectors[assign==i].mean(0) if np.any(assign==i) else centres[i] for i in range(k)])
        if np.allclose(moved,centres):break
        centres=moved
    return unit(centres)


def score(rule,query,references,seed=0,sources=None):
    """One similarity per identity for this query, under the given aggregation rule."""
    if rule=='max':return max(float(query@r) for r in references)
    if rule=='mean':return float(query@unit(references.mean(0)))
    if rule=='source-mean':
        # One photo gallery must not outvote every other source a player has: Vicente Guaita has
        # 57 references from a single forum page. Average within a source, then across sources,
        # which costs no photographs at all.
        groups={}
        for vector,key in zip(references,sources):groups.setdefault(key,[]).append(vector)
        centres=unit(np.stack([unit(np.stack(v).mean(0)) for v in groups.values()]))
        return float(query@unit(centres.mean(0)))
    k=int(rule.split('-')[1])
    return max(float(query@c) for c in prototypes(references,k,seed))


def reference_sources():
    """The page each reference came from, so one gallery can be collapsed to one vote."""
    return {r['path']:r.get('source_url','') for r in
            map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['domain']=='reference'}


def evaluate(path,rules):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    q=unit(saved['query'].numpy());g=unit(saved['gallery'].numpy())
    gids=np.asarray(saved['gallery_ids']);qids=np.asarray(saved['query_ids'])
    games=query_games();groups=defaultdict(list)
    for i,p in enumerate(saved['query_paths']):groups[games[p]].append(i)
    by_identity={pid:g[gids==pid] for pid in set(gids)}
    page=reference_sources();paths=np.asarray(saved['gallery_paths'])
    by_source={pid:[page.get(x,'') for x in paths[gids==pid]] for pid in set(gids)}
    out={}
    for rule in rules:
        hits,reciprocal=[],[]
        for rows in groups.values():
            roster=sorted(set(qids[rows]))
            for i in rows:
                scores=np.array([score(rule,q[i],by_identity[pid],hash(pid)%10**6,by_source[pid]) for pid in roster])
                order=np.argsort(-scores,kind='stable')
                rank=int(np.flatnonzero(np.asarray(roster)[order]==qids[i])[0])+1
                hits.append(rank==1);reciprocal.append(1/rank)
        out[rule]={'rank1':float(np.mean(hits)),'mean_reciprocal_rank':float(np.mean(reciprocal)),
                   'queries':len(hits),'identities':len(by_identity)}
    return out


def main(run=None):
    run=Path(run) if run else ROOT/'runs/soccernet_full_v3'
    rules=['mean','source-mean','max','proto-2','proto-3']
    validation=evaluate(run/'val_features.pt',rules)
    chosen=max(rules,key=lambda r:validation[r]['mean_reciprocal_rank'])
    test=evaluate(run/'test_features.pt',rules)
    result={'chosen_on_validation':chosen,'validation':validation,'test':test,
     'protocol':'Per-match retrieval; the gallery is that match roster. The rule was chosen by validation mean reciprocal rank and every rule is shown on test only for audit.',
     'caveat':'"max" is the best single reference. Image-level ranking, where every reference photograph is ranked on its own, is reported by report.py instead.'}
    atomic_json(ROOT/'reports/soccernet_full_v2/aggregation.json',result)
    width=max(len(r) for r in rules)
    print(f'{"rule":<{width}}  val Rank-1  val MRR   test Rank-1  test MRR')
    for r in rules:
        v,t=validation[r],test[r]
        star=' <- chosen on validation' if r==chosen else ''
        print(f'{r:<{width}}  {100*v["rank1"]:9.2f}%  {v["mean_reciprocal_rank"]:.3f}   {100*t["rank1"]:10.2f}%  {t["mean_reciprocal_rank"]:.3f}{star}')


if __name__=='__main__':main(*sys.argv[1:])
