"""Render what the deployed model predicts once the shirt number is part of each prototype.

`visualize_test` shows photograph-only retrieval, which is no longer what the model does. Here
each roster player is the single fused prototype the model is actually scored against, so the
sheets show the prediction that would be made in deployment, and every tile carries the number
the roster assigns the predicted player next to the number the model read off the crop.

The summary counts what the channel changed rather than only what it scores: how many queries the
number fixed, how many it broke, and what happened to same-team confusions, which the error
anatomy identified as the failure mode that survives once kit colour stops being usable.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT
from .numbers import label as number_label
from .visualize_test import names,teams,sheets,pair,CELL_W,CELL_H,IMG_H

REPORT=ROOT/'reports/soccernet_full_v2/test_visualisation_fused'


def rankings(run,alpha,adapted,roster_number,shirt):
 """Rank each query against its own match's roster of fused prototypes."""
 x=torch.load(run/'test_features.pt',map_location='cpu',weights_only=True)
 q=F.normalize(x['query'].float(),dim=-1);g=F.normalize(x['gallery'].float(),dim=-1)
 gids=np.asarray(x['gallery_ids']);qids=np.asarray(x['query_ids'])
 paths=x['gallery_paths']
 photo,best_photo={},{}
 for pid in set(gids.tolist()):
  rows=np.flatnonzero(gids==pid)
  photo[pid]=F.normalize(g[rows].mean(0),dim=0);best_photo[pid]=rows
 games={r['path']:r['game'] for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())
        if r['domain']=='query'}
 by_game=defaultdict(list)
 for i,p in enumerate(x['query_paths']):by_game[games[p]].append(i)
 out=[]
 for game,rows in sorted(by_game.items()):
  roster=sorted(set(qids[rows]))
  fused,plain=[],[]
  for pid in roster:
   v=photo[pid];plain.append(v)
   n=roster_number.get(pid,-1)
   fused.append(F.normalize(alpha*v+(1-alpha)*adapted[n],dim=0) if n>=0 else v)
  fused=torch.stack(fused);plain=torch.stack(plain);arr=np.asarray(roster)
  for i in rows:
   scores=fused@q[i];order=torch.argsort(scores,descending=True,stable=True).numpy()
   winner=arr[order[0]]
   before=arr[int(torch.argmax(plain@q[i]))]
   # Show a real photograph of the predicted player: its closest reference to this crop.
   pool=best_photo[winner];pick=pool[int(torch.argmax(g[pool]@q[i]))]
   read=int(torch.argmax(adapted@q[i]))
   out.append({'query_path':x['query_paths'][i],'truth':str(qids[i]),'predicted':str(winner),
    'match_path':paths[pick],'similarity':float(scores[order[0]]),
    'correct':bool(winner==qids[i]),
    'rank_of_truth':int(np.flatnonzero(arr[order]==qids[i])[0])+1,
    'match':game,'roster_size':len(roster),
    'photo_only_predicted':str(before),'photo_only_correct':bool(before==qids[i]),
    'truth_number':shirt.get(str(qids[i])),'predicted_number':shirt.get(str(winner)),
    'number_read':read})
 return out


def fused_pair(row,label):
 """The standard tile plus the numbers, which are what this model added."""
 cell=pair(row,label);draw=ImageDraw.Draw(cell)
 changed=('number fixed it' if row['correct'] and not row['photo_only_correct'] else
          'number broke it' if row['photo_only_correct'] and not row['correct'] else '')
 draw.text((7,IMG_H+58),f"shirt truth {row['truth_number']}   predicted {row['predicted_number']}"
                        f"   model read {row['number_read']}   {changed}",fill=(40,70,140))
 return cell


def main(run='runs/soccernet_full_v6',alpha=None):
 global REPORT
 run=ROOT/run if not str(run).startswith('/') else Path(run)
 # v6 keeps its original folder and report; every later run gets its own, never overwriting.
 name='number_channel.json' if run.name=='soccernet_full_v6' else f'number_channel_{run.name}_best.json'
 if run.name!='soccernet_full_v6':REPORT=REPORT.with_name(REPORT.name+'_'+run.name.split('_')[-1])
 report=json.loads((ROOT/'reports/soccernet_full_v2'/name).read_text())
 alpha=float(alpha) if alpha is not None else report['alpha_chosen_on_validation']['per_crop']
 adapted=torch.load(run/'number_cache/adapter.pt',map_location='cpu',weights_only=True)['adapted_anchors']
 adapted=F.normalize(adapted.float(),dim=-1)
 shirt={}
 for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()):
  if r['domain']=='query':shirt.setdefault(r['pid'],str(r.get('number')))
 roster_number={pid:number_label(v) for pid,v in shirt.items()}
 name=names();club=teams();label=lambda pid:name.get(pid,pid)
 rows=rankings(run,alpha,adapted,roster_number,shirt)
 for r in rows:r['same_team']=club.get(r['truth'])==club.get(r['predicted'])
 REPORT.mkdir(parents=True,exist_ok=True)
 for old in REPORT.glob('*.jpg'):old.unlink()
 import experiments.soccernet_full_v2.visualize_test as vt
 original=vt.pair;vt.pair=fused_pair
 try:
  wrong=sorted([r for r in rows if not r['correct']],key=lambda r:-r['similarity'])
  right=sorted([r for r in rows if r['correct']],key=lambda r:-r['similarity'])
  fixed=[r for r in rows if r['correct'] and not r['photo_only_correct']]
  broke=[r for r in rows if r['photo_only_correct'] and not r['correct']]
  same=[r for r in wrong if r['same_team']]
  pages=(sheets(fixed,REPORT,'number_fixed',label)+sheets(broke,REPORT,'number_broke',label)
         +sheets(same,REPORT,'same_team',label)+sheets(wrong,REPORT,'incorrect',label)
         +sheets(right,REPORT,'correct',label))
 finally:
  vt.pair=original
 photo_wrong=[r for r in rows if not r['photo_only_correct']]
 photo_same=sum(1 for r in photo_wrong if club.get(r['truth'])==club.get(r['photo_only_predicted']))
 confusion=defaultdict(int)
 for r in wrong:confusion[f'{label(r["truth"])} -> {label(r["predicted"])}']+=1
 summary={'run':str(run.relative_to(ROOT)),'alpha':alpha,'queries':len(rows),
  'fused_rank1':len(right)/len(rows),
  'photo_only_rank1':sum(r['photo_only_correct'] for r in rows)/len(rows),
  'number_fixed':len(fixed),'number_broke':len(broke),
  'same_team_confusions_fused':len(same),'same_team_confusions_photo_only':photo_same,
  'number_read_matches_truth':sum(1 for r in rows if r['number_read']==roster_number.get(r['truth'],-2))/len(rows),
  'sheets':pages,'folder':str(REPORT.relative_to(ROOT)),
  'most_frequent_confusions':sorted(confusion.items(),key=lambda kv:-kv[1])[:12]}
 atomic_json(REPORT/'rankings.json',rows)
 atomic_json(REPORT/'summary.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='most_frequent_confusions'},indent=2))
 print('top confusions:',summary['most_frequent_confusions'][:6])


if __name__=='__main__':main(*sys.argv[1:])
