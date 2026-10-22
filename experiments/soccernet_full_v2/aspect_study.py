"""Does a whole-body reference gallery retrieve better than a head-and-shoulders one?

Every stored reference is already a dominant-person crop, so the crop's height/width *is* the
shape of the detected player: about 1.0-1.4 for a head-and-shoulders press portrait and above
2.0 for a standing whole player. Queries are whole players by construction -- the test crops sit
at median aspect 2.0 and the Argentina-France track boxes at 2.27 -- while the reference gallery
sits at 1.35. This measures what that mismatch costs, on the frozen test split and from saved
embeddings, by rebuilding each identity's prototype from the taller half of its own references
and from the shorter half. Splitting each player in half keeps the number of references equal
between the two arms, so the comparison is about shape and not about how many photographs a
player happens to have.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT


def aspects(paths):
 out=[]
 for p in paths:
  with Image.open(p) as im:out.append(im.height/im.width)
 return np.asarray(out)


def prototypes(features,ids,chosen):
 """Unit-norm mean of the selected reference rows, per identity."""
 keys=sorted({ids[i] for i in chosen})
 rows={k:[i for i in chosen if ids[i]==k] for k in keys}
 matrix=np.stack([features[rows[k]].mean(0) for k in keys])
 return keys,matrix/np.linalg.norm(matrix,axis=1,keepdims=True)


def evaluate(q,qids,games,features,ids,chosen):
 """Per-match retrieval against one prototype per roster player; returns per-query outcomes."""
 keys,matrix=prototypes(features,ids,chosen)
 index={k:i for i,k in enumerate(keys)}
 out=[]
 by_game=defaultdict(list)
 for i,g in enumerate(games):by_game[g].append(i)
 for game,rows in by_game.items():
  roster=sorted({qids[i] for i in rows if qids[i] in index})
  if len(roster)<2:continue
  take=np.array([index[p] for p in roster])
  scores=q[rows]@matrix[take].T
  for row,i in zip(scores,rows):
   if qids[i] not in roster:continue
   order=np.argsort(-row,kind='stable')
   rank=int(np.flatnonzero(np.asarray(roster)[order]==qids[i])[0])+1
   out.append({'pid':qids[i],'rank1':float(rank==1),'rr':1./rank})
 return out


def summarise(values):
 return {'rank1':float(np.mean([v['rank1'] for v in values])),
         'MRR':float(np.mean([v['rr'] for v in values])),'queries':len(values)}


def paired(a,b,seed=4471):
 """Player-clustered bootstrap on the per-query difference between two reference selections."""
 assert [x['pid'] for x in a]==[x['pid'] for x in b]
 pids=sorted({x['pid'] for x in a});rows=defaultdict(list)
 for x,y in zip(a,b):rows[x['pid']].append((x['rank1']-y['rank1'],x['rr']-y['rr']))
 rng=np.random.default_rng(seed);draws=rng.integers(0,len(pids),size=(4000,len(pids)))
 out={}
 for j,name in enumerate(('rank1','MRR')):
  per=[np.array([v[j] for v in rows[p]]) for p in pids]
  totals=np.array([p.sum() for p in per]);sizes=np.array([len(p) for p in per])
  samples=totals[draws].sum(1)/sizes[draws].sum(1)
  out[name]={'difference':float(totals.sum()/sizes.sum()),
             'ci95':[float(v) for v in np.quantile(samples,[.025,.975])]}
  out[name]['distinguishable']=bool(out[name]['ci95'][0]>0 or out[name]['ci95'][1]<0)
 return out


def main(run='runs/soccernet_full_v5'):
 saved=torch.load(ROOT/run/'test_features.pt',map_location='cpu',weights_only=True)
 q=saved['query'].numpy();g=saved['gallery'].numpy()
 q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
 ids=list(saved['gallery_ids']);qids=list(saved['query_ids'])
 shape=aspects(saved['gallery_paths'])
 games={r['path']:r['game'] for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())
        if r['domain']=='query'}
 qgames=[games[p] for p in saved['query_paths']]
 per=defaultdict(list)
 for i,pid in enumerate(ids):per[pid].append(i)
 tall,short,all_rows=[],[],list(range(len(ids)))
 for pid,rows in per.items():
  ordered=sorted(rows,key=lambda i:-shape[i]);half=max(1,len(ordered)//2)
  tall+=ordered[:half];short+=ordered[-half:]
 arms={'all':all_rows,'taller half':tall,'shorter half':short}
 results={name:evaluate(q,qids,qgames,g,ids,rows) for name,rows in arms.items()}
 report={'run':run,'reference_aspect':{'median':float(np.median(shape)),
   'quartiles':[float(v) for v in np.percentile(shape,[25,50,75])],
   'share_above_2':float(np.mean(shape>2))},
  'query_aspect_median':float(np.median(aspects(saved['query_paths'][:400]))),
  'arms':{name:dict(summarise(v),references=len(arms[name])) for name,v in results.items()},
  'taller_minus_shorter':paired(results['taller half'],results['shorter half']),
  'taller_minus_all':paired(results['taller half'],results['all']),
  'note':'Each identity is split at its own median, so the two half-arms hold equal reference counts.'}
 out=ROOT/'reports/soccernet_full_v2/aspect_study.json';atomic_json(out,report)
 print(json.dumps(report,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
