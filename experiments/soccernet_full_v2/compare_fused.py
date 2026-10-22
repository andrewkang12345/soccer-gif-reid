"""Is one run's fused photo+number score distinguishable from another's on the same test units?

compare_runs.py pairs photo-only retrieval. This pairs the deployment metric instead: the per-match
fused prototype ranking scored by number_channel.py, per crop and per ground-truth player-by-match
pool, each run at its own validation-chosen alpha. It reads the embeddings and adapted anchors that
number_channel.py cached, so it needs no GPU, and it checks its own aggregates against that run's
report before comparing anything. Resampling is clustered by player, as in compare_runs.py.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT
from .numbers import label
from .number_channel import groups


def unit_ranks(q,refs,meta,ref_ids,adapted,numbers,alpha,tracklet=False):
    """number_channel.score_split, keeping each unit's rank instead of only the summary."""
    photo={};rows=defaultdict(list)
    for i,pid in enumerate(ref_ids):rows[pid].append(i)
    for pid,idx in rows.items():photo[pid]=F.normalize(refs[idx].mean(0),dim=0)
    units=groups(meta) if tracklet else {(r['pid'],r['game'],i):[i] for i,r in enumerate(meta)}
    by_game=defaultdict(list)
    for key in units:by_game[key[1]].append(key)
    out={}
    for game,keys in by_game.items():
        roster=sorted({k[0] for k in keys if k[0] in photo})
        if len(roster)<2:continue
        parts=[]
        for pid in roster:
            anchor=adapted[numbers[pid]] if numbers.get(pid,-1)>=0 else torch.zeros_like(photo[pid])
            parts.append(F.normalize(alpha*photo[pid]+(1-alpha)*anchor,dim=0))
        matrix=torch.stack(parts);arr=np.asarray(roster)
        for key in keys:
            if key[0] not in photo:continue
            e=F.normalize(q[units[key]].mean(0),dim=0)
            order=torch.argsort(matrix@e,descending=True,stable=True).numpy()
            out[key]={'player':key[0],'rank':1+int(np.flatnonzero(arr[order]==key[0])[0])}
    return out


def summary(ranks):
    r=np.asarray(ranks)
    return {'rank1':float(np.mean(r==1)),'top3':float(np.mean(r<=3)),'MRR':float(np.mean(1/r))}


def load_run(run,rows):
    meta=[r for r in rows if r['split']=='test' and r['domain']=='query']
    refs=[r for r in rows if r['split']=='test' and r['domain']=='reference']
    cache=run/'number_cache'
    q=torch.load(cache/'test_query.pt',map_location='cpu',weights_only=True)
    g=torch.load(cache/'test_reference.pt',map_location='cpu',weights_only=True)
    assert q['paths']==[r['path'] for r in meta] and g['paths']==[r['path'] for r in refs],\
        f'{run.name}: cached features do not match the manifest'
    adapted=torch.load(cache/'adapter.pt',map_location='cpu',weights_only=False)['adapted_anchors']
    report=json.loads((ROOT/'reports/soccernet_full_v2'/f'number_channel_{run.name}_best.json').read_text())
    numbers={}
    for r in meta:numbers.setdefault(r['pid'],label(r.get('number')))
    q=F.normalize(q['features'],dim=-1);g=F.normalize(g['features'],dim=-1)
    out={}
    for unit,tracklet,key in (('per_crop',False,'per_crop'),('per_tracklet',True,'per_tracklet')):
        alpha=float(report['alpha_chosen_on_validation'][key])
        ranks=unit_ranks(q,g,meta,[r['pid'] for r in refs],adapted,numbers,alpha,tracklet)
        mine=summary([v['rank'] for v in ranks.values()])
        for m,v in mine.items():
            assert abs(v-report['test'][key][m])<1e-9,f'{run.name} {unit} {m}: {v} vs reported {report["test"][key][m]}'
        out[unit]={'alpha':alpha,'ranks':ranks,'summary':mine}
    return out


def bootstrap(paired,draws=5000,seed=4):
    clusters=defaultdict(list)
    for i,p in enumerate(paired):clusters[p['player']].append(i)
    keys=sorted(clusters);rng=np.random.default_rng(seed)
    stats={m:[] for m in ('rank1','top3','MRR')}
    a=np.array([p['a'] for p in paired]);b=np.array([p['b'] for p in paired])
    for _ in range(draws):
        idx=np.concatenate([clusters[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        stats['rank1'].append(np.mean(a[idx]==1)-np.mean(b[idx]==1))
        stats['top3'].append(np.mean(a[idx]<=3)-np.mean(b[idx]<=3))
        stats['MRR'].append(np.mean(1/a[idx])-np.mean(1/b[idx]))
    point={'rank1':np.mean(a==1)-np.mean(b==1),'top3':np.mean(a<=3)-np.mean(b<=3),'MRR':np.mean(1/a)-np.mean(1/b)}
    out={}
    for m,draws_ in stats.items():
        lo,hi=float(np.quantile(draws_,.025)),float(np.quantile(draws_,.975))
        out[m]={'difference':float(point[m]),'ci95':[lo,hi],'distinguishable':bool(lo>0 or hi<0),
                'fraction_of_resamples_favouring_first':float(np.mean(np.array(draws_)>0))}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('first',type=Path,help='run directory, e.g. runs/soccernet_full_v8')
    ap.add_argument('second',type=Path,help='run directory to subtract')
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    a,b=load_run(args.first,rows),load_run(args.second,rows)
    result={'first':args.first.name,'second':args.second.name,'cluster':'player','draws':5000,'units':{}}
    for unit in ('per_crop','per_tracklet'):
        shared=sorted(set(a[unit]['ranks'])&set(b[unit]['ranks']),key=str)
        paired=[{'player':a[unit]['ranks'][k]['player'],'a':a[unit]['ranks'][k]['rank'],
                 'b':b[unit]['ranks'][k]['rank']} for k in shared]
        result['units'][unit]={'shared_units':len(shared),
            'alpha':{args.first.name:a[unit]['alpha'],args.second.name:b[unit]['alpha']},
            'summary':{args.first.name:a[unit]['summary'],args.second.name:b[unit]['summary']},
            'paired':bootstrap(paired)}
    out=args.out or ROOT/'reports/soccernet_full_v2'/f'fused_comparison_{args.first.name}_vs_{args.second.name}.json'
    atomic_json(out,result)
    print(f'{args.first.name} minus {args.second.name}, clustered by player:')
    for unit,u in result['units'].items():
        print(f"  {unit} ({u['shared_units']} units)")
        for m,v in u['paired'].items():
            verdict='distinguishable' if v['distinguishable'] else 'NOT distinguishable'
            print(f"    {m:5s} {100*v['difference']:+6.2f} pts [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}] -> {verdict}")
    print('wrote',out)


if __name__=='__main__':main()
