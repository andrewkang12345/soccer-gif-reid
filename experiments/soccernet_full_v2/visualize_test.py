"""Render every test query beside the reference it retrieved, labelled with truth and prediction.

A retrieval number says how often the model is right; it does not say what it confuses. These
sheets pair each broadcast crop with the top-ranked gallery photograph and mark the tile green
when the identity matches and red when it does not, so the failures can be read directly. One
folder per scored model, correct and incorrect on separate sheets, ordered by confidence.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from soccer_gif_reid.common import read_rgb,atomic_json
from .metadata import DATA,ROOT
from .report import match_groups

REPORT=ROOT/'reports/soccernet_full_v2/test_visualisation'
COLUMNS,ROWS=4,4
CELL_W,CELL_H,IMG_H=390,306,240
MODELS={'soccer_full':'test_features.pt','frozen_clip':'frozen_clip_test_features.pt','nba_projection':'nba_test_features.pt'}


def names():
 """Every roster player, so an archived run whose split has since changed still labels."""
 return {r['pid']:r['name'] for r in json.loads((DATA/'players.json').read_text())}


def teams():
 """Club per player, so a confusion can be split into same-team and opponent."""
 return {r['pid']:r['team'] for r in json.loads((DATA/'players.json').read_text())}


def rankings(path):
 """Rank each query against its own match's roster, which is how the model is used."""
 x=torch.load(path,map_location='cpu',weights_only=True)
 q=torch.nn.functional.normalize(x['query'].float(),dim=-1).numpy()
 g=torch.nn.functional.normalize(x['gallery'].float(),dim=-1).numpy()
 gids=np.asarray(x['gallery_ids']);qids=np.asarray(x['query_ids'])
 out=[]
 for game,rows in sorted(match_groups(x).items()):
  keep=np.flatnonzero(np.isin(gids,list(set(qids[rows]))))
  for i in rows:
   order=keep[np.argsort(-(g[keep]@q[i]),kind='stable')];best=order[0]
   out.append({'query_path':x['query_paths'][i],'truth':qids[i],'predicted':str(gids[best]),
    'match_path':x['gallery_paths'][best],'similarity':float(g[best]@q[i]),
    'correct':bool(gids[best]==qids[i]),'rank_of_truth':int(np.flatnonzero(gids[order]==qids[i])[0])+1,
    'match':game,'roster_size':len(set(qids[rows]))})
 return out


def pair(row,label):
 cell=Image.new('RGB',(CELL_W,CELL_H),'white');draw=ImageDraw.Draw(cell)
 for offset,path in ((5,row['query_path']),(CELL_W//2+2,row['match_path'])):
  try:thumb=ImageOps.contain(read_rgb(path),(CELL_W//2-9,IMG_H-6))
  except Exception:continue
  cell.paste(thumb,(offset+(CELL_W//2-9-thumb.width)//2,IMG_H-thumb.height))
 colour=(24,132,60) if row['correct'] else (190,38,38)
 draw.rectangle([0,0,CELL_W-1,CELL_H-1],outline=colour,width=3)
 draw.text((7,IMG_H+4),f"truth: {label(row['truth'])[:38]}",fill=(20,20,20))
 draw.text((7,IMG_H+22),f"top-1: {label(row['predicted'])[:38]}",fill=colour)
 draw.text((7,IMG_H+40),f"sim {row['similarity']:.3f}   true identity at rank {row['rank_of_truth']}"
                        f"   roster {row['roster_size']}   [query | retrieved]",fill=(90,90,90))
 return cell


def sheets(rows,folder,stem,label):
 folder.mkdir(parents=True,exist_ok=True)
 per=COLUMNS*ROWS
 for page in range((len(rows)+per-1)//per):
  canvas=Image.new('RGB',(CELL_W*COLUMNS,CELL_H*ROWS),'white')
  for i,row in enumerate(rows[page*per:(page+1)*per]):
   canvas.paste(pair(row,label),(i%COLUMNS*CELL_W,i//COLUMNS*CELL_H))
  canvas.save(folder/f'{stem}_{page:02d}.jpg',quality=92)
 return (len(rows)+per-1)//per


def main(run=None):
 run=Path(run) if run else ROOT/'runs/soccernet_full_v2'
 label=names();club=teams();summary={}
 for model,filename in MODELS.items():
  path=run/filename
  if not path.exists():continue
  rows=rankings(path)
  for r in rows:r['same_team']=club.get(r['truth'])==club.get(r['predicted'])
  naming=lambda pid:label.get(pid,pid)
  folder=REPORT/model
  for old in folder.glob('*.jpg'):old.unlink()
  right=sorted([r for r in rows if r['correct']],key=lambda r:-r['similarity'])
  wrong=sorted([r for r in rows if not r['correct']],key=lambda r:-r['similarity'])
  # Same-team confusions get their own sheets: after kit colour stopped being usable they are
  # the failure mode that remains, and they are the ones worth looking at.
  same=[r for r in wrong if r['same_team']]
  pages=(sheets(same,folder,'same_team',naming)+sheets(wrong,folder,'incorrect',naming)
         +sheets(right,folder,'correct',naming))
  confusion=defaultdict(int)
  for r in wrong:confusion[f'{label.get(r["truth"],r["truth"])} -> {label.get(r["predicted"],r["predicted"])}']+=1
  summary[model]={'queries':len(rows),'correct':len(right),'incorrect':len(wrong),
   'per_match_rank1':len(right)/len(rows),'sheets':pages,'folder':str(folder.relative_to(ROOT)),
   'most_frequent_confusions':sorted(confusion.items(),key=lambda kv:-kv[1])[:12],
   'incorrect_with_true_identity_in_top_five':sum(1 for r in wrong if r['rank_of_truth']<=5),
   'same_team_confusions':len(same),'opponent_confusions':len(wrong)-len(same)}
  atomic_json(folder/'rankings.json',rows)
  print(model,'->',summary[model]['folder'],f"{len(right)}/{len(rows)} correct, {pages} sheets",flush=True)
 atomic_json(REPORT/'summary.json',summary)
 print(json.dumps({k:{a:b for a,b in v.items() if a!='most_frequent_confusions'} for k,v in summary.items()},indent=2))


if __name__=='__main__':main(*sys.argv[1:])
