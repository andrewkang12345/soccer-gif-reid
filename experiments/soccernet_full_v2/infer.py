"""Rank roster identities using clean reference centroids and genuine crop/tracklet observations."""
import argparse,json,re,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import image_paths,digest_file,atomic_json,atomic_write
from experiments.ap3d_no_quality_v2.evaluate import load_model
from .screen_references import CROP_MARGIN,MIN_CONFIDENCE,MIN_SUBJECT_PIXELS,crop_box
from .train import Images


def roster_crops(paths,cache_dir,device='cuda:0',weights='/mnt/data/sports_reID/yolo11m-seg.pt'):
 """Crop each roster photograph to its dominant person, as the training gallery was built.

 The gallery stores detector crops, so a full-frame roster photograph must be reduced the
 same way or the reference and the broadcast crop no longer describe the same region. A photo
 with no single dominant person is used whole and reported, rather than silently dropped.
 """
 from ultralytics import YOLO
 from soccer_gif_reid.common import read_rgb
 model=YOLO(weights);out=[];skipped=[]
 cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
 for path in paths:
  image=read_rgb(path)
  pred=model.predict(image,imgsz=640,conf=.25,classes=[0],device=device,verbose=False)[0]
  boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
  order=sorted(range(len(boxes)),key=lambda j:float(np.prod(boxes[j,2:]-boxes[j,:2])),reverse=True)
  keep=None
  if order:
   areas=[float(np.prod(boxes[j,2:]-boxes[j,:2])) for j in order]
   box=boxes[order[0]];short=float(min(box[2]-box[0],box[3]-box[1]))
   if (areas[1]/areas[0] if len(areas)>1 else 0.)<.35 and short>=MIN_SUBJECT_PIXELS and float(scores[order[0]])>=MIN_CONFIDENCE:
    keep=crop_box(box,image.width,image.height)
  if keep is None:skipped.append(str(path))
  target=cache_dir/(hashlib.sha256(str(path).encode()).hexdigest()[:20]+'.png')
  (image if keep is None else image.crop(tuple(int(round(v)) for v in keep))).save(target,'PNG')
  out.append(target)
 return out,skipped

@torch.inference_mode()
def rank(checkpoint,roster,query,device='cuda:0',top_k=10,max_frames=64,cache=None,roster_crop=True,crop_dir=None,roster_numbers=None,photo_weight=1.):
 if top_k<1 or max_frames<1:raise ValueError('top-k and max-frames must be positive')
 if not 0<=photo_weight<=1:raise ValueError('photo-weight must be between 0 and 1')
 if photo_weight<1 and roster_numbers is None:raise ValueError('Number fusion requires --roster-numbers')
 torch.set_num_threads(4);roster=Path(roster).resolve();refs=image_paths(roster)
 ids=[]
 for path in refs:
  rel=path.relative_to(roster)
  if len(rel.parts)<2:raise ValueError('Put reference photos under roster/<player_name>/')
  ids.append(rel.parts[0])
 fingerprint=hashlib.sha256(json.dumps({'checkpoint':digest_file(checkpoint),'files':[(str(p.relative_to(roster)),digest_file(p)) for p in refs],'preprocess':'letterbox_clip_224_v1','roster_crop':roster_crop,'aggregation':'normalized mean of individually normalized embeddings'}).encode()).hexdigest()
 undetected=[]
 if roster_crop:refs,undetected=roster_crops(refs,crop_dir or (Path(cache).parent if cache else roster.parent)/'roster_person_crops',device)
 model,payload=load_model(checkpoint,device)
 if payload['kind']!='clip':raise ValueError('This experiment expects a GIF CLIP projection checkpoint')
 numbers=None;adapted=None
 if roster_numbers is not None:
  from .numbers import label,load as load_anchors,CACHE as ANCHOR_CACHE
  raw=json.loads(Path(roster_numbers).read_text())
  if not isinstance(raw,dict):raise ValueError('roster-numbers must map exact roster folder names to numbers')
  missing=sorted(set(ids)-set(raw))
  if missing:raise ValueError(f'Missing roster numbers for: {missing}')
  numbers={identity:label(raw[identity]) for identity in set(ids)}
  if any(n<0 for n in numbers.values()):raise ValueError('Roster numbers must be integers from 0 through 99')
  if photo_weight<1:
   if model.number_adapter is None:raise ValueError('Checkpoint has no trained number adapter')
   if not ANCHOR_CACHE.exists():raise FileNotFoundError('Missing trained number anchors: '+str(ANCHOR_CACHE))
   expected=payload.get('config',{}).get('number_anchors_sha256')
   if expected and expected!=digest_file(ANCHOR_CACHE):raise ValueError('Number anchors differ from training')
   adapted=model.number_adapter(load_anchors(device)).float().cpu()
 def encode(paths):
  out=[]
  for x in DataLoader(Images([{'path':str(p)} for p in paths]),batch_size=16,num_workers=0):
   with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):out.append(model(x.to(device)).float().cpu())
  return F.normalize(torch.cat(out),dim=-1)
 cached=torch.load(cache,map_location='cpu',weights_only=True) if cache and Path(cache).exists() else None
 if cached and cached.get('fingerprint')==fingerprint:
  identities=cached['identities'];centroids=cached['centroids']
 else:
  features=encode(refs);identities=sorted(set(ids));centroids=torch.stack([F.normalize(features[[i for i,pid in enumerate(ids) if pid==name]].mean(0),dim=0) for name in identities])
  if cache:atomic_write(cache,lambda p:torch.save({'fingerprint':fingerprint,'identities':identities,'centroids':centroids},p))
 query=Path(query);paths=image_paths(query) if query.is_dir() else [query]
 paths=sorted(paths,key=lambda p:[(0,int(s)) if s.isdigit() else (1,s.lower()) for s in re.split(r'(\d+)',p.name)])
 paths=[paths[i] for i in np.linspace(0,len(paths)-1,min(len(paths),max_frames),dtype=int)]
 prototypes=centroids
 if adapted is not None:
  prototypes=F.normalize(photo_weight*centroids+(1-photo_weight)*torch.stack([adapted[numbers[name]] for name in identities]),dim=-1)
 feature=F.normalize(encode(paths).mean(0),dim=0);scores=prototypes@feature;order=torch.argsort(scores,descending=True,stable=True)[:top_k]
 return {'checkpoint':str(Path(checkpoint).resolve()),'frames_used':len(paths),'identities':len(identities),'reference_images':len(refs),'score_type':'Cosine similarity; not calibrated identity probability','roster_crop':roster_crop,'roster_photos_without_a_dominant_person':undetected,'aggregation':'Mean normalized reference features per identity; mean normalized genuine query observations','photo_weight':photo_weight,'uses_roster_numbers':adapted is not None,'ranked_identities':[{'identity':identities[i],'similarity':float(scores[i]),**({'number':numbers[identities[i]]} if numbers is not None else {})} for i in order.tolist()]}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['checkpoint','roster','query']:p.add_argument('--'+key,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--top-k',type=int,default=10);p.add_argument('--max-frames',type=int,default=64);p.add_argument('--cache');p.add_argument('--output')
 p.add_argument('--no-roster-crop',action='store_true',help='Use whole roster photographs instead of dominant-person crops')
 p.add_argument('--crop-dir',help='Where cropped roster photographs are written')
 p.add_argument('--roster-numbers',help='JSON mapping roster folder names to jersey numbers; duplicate numbers are allowed')
 p.add_argument('--photo-weight',type=float,default=1.,help='Photo fraction in a fused prototype; use a weight chosen on validation')
 a=p.parse_args();result=rank(a.checkpoint,a.roster,a.query,a.device,a.top_k,a.max_frames,a.cache,not a.no_roster_crop,a.crop_dir,a.roster_numbers,a.photo_weight)
 if a.output:atomic_json(a.output,result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
