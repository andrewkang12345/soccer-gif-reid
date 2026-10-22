"""Where the per-match retrieval loss actually is, before choosing what to fix next.

Per-match retrieval mixes two very different questions: which of the two teams is this, and
which of that team's eleven. Pooled Rank-1 hides the split. This separates them on the frozen
test split from saved embeddings, and adds the two things a fix would have to beat -- what
knowing the team is worth, and what reading the shirt number would be worth.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT


def load(run):
 saved=torch.load(ROOT/run/'test_features.pt',map_location='cpu',weights_only=True)
 q=saved['query'].numpy();g=saved['gallery'].numpy()
 q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
 rows={r['path']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())}
 meta=[rows[p] for p in saved['query_paths']]
 return saved,q,g,meta


def prototypes(g,ids):
 keys=sorted(set(ids));index={k:i for i,k in enumerate(keys)}
 rows=defaultdict(list)
 for i,k in enumerate(ids):rows[k].append(i)
 m=np.stack([g[rows[k]].mean(0) for k in keys])
 return keys,index,m/np.linalg.norm(m,axis=1,keepdims=True)


def main(run='runs/soccernet_full_v5'):
 saved,q,g,meta=load(run)
 keys,index,protos=prototypes(g,list(saved['gallery_ids']))
 team={r['pid']:r['team'] for r in meta};number={r['pid']:str(r.get('number','')) for r in meta}
 by_game=defaultdict(list)
 for i,r in enumerate(meta):by_game[r['game']].append(i)
 pixels=np.array([int(r['width'])*int(r['height']) for r in meta])
 out=[]
 for game,rowset in by_game.items():
  roster=sorted({meta[i]['pid'] for i in rowset if meta[i]['pid'] in index})
  if len(roster)<2:continue
  take=np.array([index[p] for p in roster]);arr=np.asarray(roster)
  scores=q[rowset]@protos[take].T
  for row,i in zip(scores,rowset):
   truth=meta[i]['pid']
   if truth not in index:continue
   order=np.argsort(-row,kind='stable');ranked=arr[order]
   same=[j for j,p in enumerate(roster) if team[p]==team[truth]]
   other=[j for j,p in enumerate(roster) if team[p]!=team[truth]]
   samerank=np.asarray(roster)[np.array(same)[np.argsort(-row[same],kind='stable')]]
   numeric=[j for j,p in enumerate(roster) if number[p] and number[p]==number[truth]]
   out.append({'pid':truth,'game':game,'pixels':int(pixels[i]),'view':meta[i].get('view'),
    'roster':len(roster),'teammates':len(same),
    'rank1':float(ranked[0]==truth),
    'rank':1+int(np.flatnonzero(ranked==truth)[0]),
    'top_is_teammate':float(team[ranked[0]]==team[truth]),
    'team_correct':float(team[ranked[0]]==team[truth]),
    'rank1_same_team_only':float(samerank[0]==truth),
    'rank_same_team_only':1+int(np.flatnonzero(samerank==truth)[0]),
    'number_unique_in_roster':len(numeric)==1,
    'best_other_team':float(np.max(row[other])) if other else None,
    'best_same_team':float(np.max(row[same])),'truth_score':float(row[roster.index(truth)])})
 def share(rows,key):return float(np.mean([r[key] for r in rows])) if rows else None
 errors=[r for r in out if not r['rank1']]
 quart=np.percentile([r['pixels'] for r in out],[25,50,75])
 bands={}
 for name,test in [('smallest quarter',lambda r:r['pixels']<=quart[0]),
                   ('second',lambda r:quart[0]<r['pixels']<=quart[1]),
                   ('third',lambda r:quart[1]<r['pixels']<=quart[2]),
                   ('largest quarter',lambda r:r['pixels']>quart[2])]:
  rows=[r for r in out if test(r)]
  bands[name]={'queries':len(rows),'median_pixels':int(np.median([r['pixels'] for r in rows])),
   'rank1':share(rows,'rank1'),'rank1_within_team':share(rows,'rank1_same_team_only'),
   'team_correct':share(rows,'team_correct')}
 views=defaultdict(list)
 for r in out:views[r['view']].append(r)
 report={'run':run,'queries':len(out),
  'mean_roster':float(np.mean([r['roster'] for r in out])),
  'mean_teammates':float(np.mean([r['teammates'] for r in out])),
  'rank1':share(out,'rank1'),
  'picks_a_teammate_when_wrong':share(errors,'top_is_teammate'),
  'chance_of_a_teammate_at_random':float(np.mean([(r['teammates']-1)/(r['roster']-1) for r in errors])),
  'team_identified':share(out,'team_correct'),
  'team_chance':float(np.mean([r['teammates']/r['roster'] for r in out])),
  'rank1_if_team_were_known':share(out,'rank1_same_team_only'),
  'rank1_if_number_were_read':float(np.mean([r['number_unique_in_roster'] for r in out])),
  'by_query_size':bands,
  'by_view':{k:{'queries':len(v),'rank1':share(v,'rank1'),
    'rank1_within_team':share(v,'rank1_same_team_only')} for k,v in sorted(views.items())},
  'note':'"If the team were known" ranks only the truth team. "If the number were read" is the '
         'share of queries whose shirt number is unique on the roster, the ceiling a perfect '
         'number reader plus roster metadata would reach.'}
 atomic_json(ROOT/'reports/soccernet_full_v2/error_anatomy.json',report)
 print(json.dumps(report,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
