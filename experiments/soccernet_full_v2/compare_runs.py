"""Is the difference between two runs larger than the noise in the evaluation?

R5 scored 2.4 points below R4 on per-match Rank-1 while scoring higher on validation, and the
paired interval on that difference ran from -12.6 to +6.6. A single run's number carries far less
information than it appears to, so any claim that one change beat another has to come with an
interval, and it has to be **paired**: both runs answer the same queries, so comparing their
totals independently throws away the pairing and widens the interval for no reason.

Resampling is clustered by player rather than by query. Queries of one player are not independent
— they come from the same match, often seconds apart — and resampling them individually would
report an interval several times too narrow.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from soccer_gif_reid.common import atomic_json
from .metadata import ROOT
from .report import query_games


def per_query(path):
    """Per-match retrieval outcome for every query: was the top-ranked roster identity right."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    q=saved['query'].numpy();g=saved['gallery'].numpy()
    q=q/np.clip(np.linalg.norm(q,axis=1,keepdims=True),1e-9,None)
    g=g/np.clip(np.linalg.norm(g,axis=1,keepdims=True),1e-9,None)
    gid=np.asarray(saved['gallery_ids']);qid=np.asarray(saved['query_ids'])
    games=query_games();groups=defaultdict(list)
    for i,p in enumerate(saved['query_paths']):groups[games[p]].append(i)
    out={}
    for game,rows in groups.items():
        keep=np.flatnonzero(np.isin(gid,list(set(qid[rows]))))
        for i in rows:
            order=keep[np.argsort(-(g[keep]@q[i]),kind='stable')]
            hits=gid[order]==qid[i];ranks=np.flatnonzero(hits)
            out[saved['query_paths'][i]]={'hit':float(hits[0]),'player':qid[i],'match':game,
                'ap':float(np.mean(np.arange(1,len(ranks)+1)/(ranks+1)))}
    return out


def bootstrap(rows,statistic,cluster,draws=5000,seed=4):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[cluster]].append(i)
    keys=sorted(groups);rng=np.random.default_rng(seed);out=[]
    for _ in range(draws):
        picked=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        out.append(statistic([rows[i] for i in picked]))
    return float(np.quantile(out,.025)),float(np.quantile(out,.975)),out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',help='run directories holding test_features.pt')
    ap.add_argument('--cluster',default='player',choices=['player','match'])
    ap.add_argument('--out',type=Path,default=ROOT/'reports/soccernet_full_v2/run_comparison.json')
    args=ap.parse_args()
    scored={Path(r).name:per_query(Path(r)/'test_features.pt') for r in args.runs}
    shared=sorted(set.intersection(*(set(v) for v in scored.values())))
    assert shared,'these runs share no queries, so they cannot be compared'
    result={'queries_shared':len(shared),'cluster':args.cluster,'runs':{}}
    for name,rows in scored.items():
        data=[rows[k] for k in shared]
        for key in ('hit','ap'):
            lo,hi,_=bootstrap(data,lambda d,k=key:float(np.mean([x[k] for x in d])),args.cluster)
            result['runs'].setdefault(name,{})[{'hit':'rank1','ap':'mAP'}[key]]={
                'value':float(np.mean([x[key] for x in data])),'ci95':[lo,hi]}
    names=list(scored)
    result['paired']={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            paired=[{'player':scored[a][k]['player'],'match':scored[a][k]['match'],
                     'hit':scored[a][k]['hit']-scored[b][k]['hit'],
                     'ap':scored[a][k]['ap']-scored[b][k]['ap']} for k in shared]
            entry={}
            for key,label in (('hit','rank1'),('ap','mAP')):
                lo,hi,draws=bootstrap(paired,lambda d,k=key:float(np.mean([x[k] for x in d])),args.cluster)
                entry[label]={'difference':float(np.mean([x[key] for x in paired])),'ci95':[lo,hi],
                              'fraction_of_resamples_favouring_first':float(np.mean(np.array(draws)>0)),
                              'distinguishable':bool(lo>0 or hi<0)}
            result['paired'][f'{a} minus {b}']=entry
    atomic_json(args.out,result)
    width=max(len(n) for n in names)
    print(f'{"run":<{width}}  per-match Rank-1 (95% CI)        mAP (95% CI)')
    for name,m in result['runs'].items():
        print(f"{name:<{width}}  {100*m['rank1']['value']:6.2f}%  [{100*m['rank1']['ci95'][0]:5.2f}, {100*m['rank1']['ci95'][1]:5.2f}]"
              f"    {100*m['mAP']['value']:6.2f}%  [{100*m['mAP']['ci95'][0]:5.2f}, {100*m['mAP']['ci95'][1]:5.2f}]")
    print(f'\npaired differences, clustered by {args.cluster}, on {len(shared)} shared queries:')
    for pair,m in result['paired'].items():
        r=m['rank1']
        verdict='distinguishable' if r['distinguishable'] else 'NOT distinguishable'
        print(f"  {pair}: Rank-1 {100*r['difference']:+.2f} pts "
              f"[{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] -> {verdict}")


if __name__=='__main__':main()
