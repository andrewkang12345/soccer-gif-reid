"""Find references that do not look like the rest of their player's gallery.

Every automatic screen so far asks whether a photograph shows one dominant person on a page that
names the player. None asks whether it shows the *same* person as the player's other photographs.
A spectator in the crowd, a team-mate from the same photo gallery, or a namesake all pass the
earlier screens and only reveal themselves next to the rest of the gallery.

Each reference is scored by its mean cosine similarity to the other references of the same player.
That is a suggestion, not a verdict: a legitimate photograph can be an outlier because it is the
only headshot among action shots, so everything flagged is rendered for review beside the two
references it least resembles and the two it most resembles, and nothing is excluded automatically.

This scan is applied to every split on the same rule. The first instance was noticed in a test
error sheet, so to be explicit: what is judged here is whether a photograph depicts the named
player, never whether removing it helps a score.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from torch.utils.data import DataLoader
from soccer_gif_reid.common import read_rgb,atomic_json
from soccer_gif_reid.model import build_model
from .metadata import DATA,ROOT
from .train import Images

REPORT=ROOT/'reports/soccernet_full_v2/reference_outliers'


def embed(rows,checkpoint,device='cuda:0'):
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert payload.get('kind')=='clip',payload.get('kind')
 model=build_model(device,ROOT/'weights');model.load_state_dict(payload['model']);model.eval()
 out=[]
 with torch.inference_mode():
  for x in DataLoader(Images(rows),batch_size=32,num_workers=4):
   with torch.autocast('cuda',dtype=torch.bfloat16):out.append(model(x.to(device)).float().cpu())
 x=torch.cat(out).numpy()
 return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),1e-9,None)


def main(checkpoint=None,threshold=.60,per_sheet=12):
 threshold=float(threshold)
 checkpoint=checkpoint or ROOT/'runs/soccernet_full_v3/best.pt'
 rows=[json.loads(s) for s in (DATA/'gallery.jsonl').read_text().splitlines()]
 features=embed(rows,checkpoint)
 by_player=defaultdict(list)
 for i,r in enumerate(rows):by_player[r['pid']].append(i)
 flagged=[]
 for pid,idx in by_player.items():
  if len(idx)<3:continue
  block=features[idx];sim=block@block.T;np.fill_diagonal(sim,np.nan)
  mean=np.nanmean(sim,axis=1)
  for position,i in enumerate(idx):
   if mean[position]<threshold:
    order=np.argsort(-np.nan_to_num(sim[position],nan=-9))
    flagged.append({'candidate_id':rows[i]['candidate_id'],'pid':pid,'name':rows[i]['name'],
     'split':rows[i]['split'],'path':rows[i]['path'],'mean_similarity':float(mean[position]),
     'gallery_size':len(idx),'evidence':rows[i].get('name_evidence'),'source_url':rows[i]['source_url'],
     'nearest':[rows[idx[j]]['path'] for j in order[:2]],
     'furthest':[rows[idx[j]]['path'] for j in order[-2:]]})
 flagged.sort(key=lambda r:r['mean_similarity'])
 REPORT.mkdir(parents=True,exist_ok=True)
 for old in REPORT.glob('*.jpg'):old.unlink()
 cell_w,cell_h=560,232
 for page in range((len(flagged)+per_sheet-1)//per_sheet):
  canvas=Image.new('RGB',(cell_w*2,cell_h*(per_sheet//2)),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(flagged[page*per_sheet:(page+1)*per_sheet]):
   x=i%2*cell_w;y=i//2*cell_h
   draw.rectangle([x,y,x+cell_w-1,y+cell_h-1],outline=(190,38,38),width=2)
   for slot,path in enumerate([r['path']]+r['nearest']+r['furthest'][:1]):
    try:thumb=ImageOps.contain(read_rgb(path),(128,168))
    except Exception:continue
    canvas.paste(thumb,(x+8+slot*136+(128-thumb.width)//2,y+26))
   draw.text((x+8,y+6),f"{page*per_sheet+i:03d} {r['name'][:30]} [{r['split']}] mean sim {r['mean_similarity']:.3f} of {r['gallery_size']}",fill=(20,20,20))
   draw.text((x+8,y+cell_h-18),'flagged | two most similar | least similar   '+str(r['evidence'])[:34],fill=(110,110,110))
  canvas.save(REPORT/f'page_{page:02d}.jpg',quality=93)
 atomic_json(REPORT/'flagged.json',{'checkpoint':str(checkpoint),'threshold':threshold,
  'gallery_references':len(rows),'players_scored':sum(1 for v in by_player.values() if len(v)>=3),
  'flagged':len(flagged),'by_split':{s:sum(1 for r in flagged if r['split']==s) for s in ('train','val','test')},
  'by_evidence':{e:sum(1 for r in flagged if r['evidence']==e) for e in {r['evidence'] for r in flagged}},
  'rows':flagged})
 print(json.dumps({'flagged':len(flagged),'of':len(rows),'pages':len(list(REPORT.glob('*.jpg'))),
  'by_split':{s:sum(1 for r in flagged if r['split']==s) for s in ('train','val','test')}},indent=2))


if __name__=='__main__':main(*sys.argv[1:])
