"""Cut player tracklets out of SoccerNet Game State Reconstruction for the quality head.

GSR annotates every box with a `track_id` inside a thirty-second sequence, so a tracklet is
ground truth rather than tracker output and cannot suffer an identity switch. It also records the
jersey number per box, marked absent when unreadable, which is the per-crop legibility signal the
labelled set structurally cannot provide: there, every query carries a number because the number
is the roster join key, whether or not it is visible.

Frames are sampled evenly across a tracklet rather than consecutively. Adjacent frames at 25 fps
are nearly identical, so eight consecutive ones teach an attention head almost nothing, while
eight spread over thirty seconds span the pose, scale and occlusion changes it has to learn to
weigh. Each source frame is opened once and every box wanted from it is cut in that pass.
"""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import DATA

GSR=DATA.parent/'sn_gsr_2025'
OUT=DATA/'gsr_crops'
MARGIN=.08


def tracklets(labels,per_track=32,min_side=20):
 """Group a sequence's boxes by track, keeping players and goalkeepers with a usable box."""
 tracks=defaultdict(list)
 for a in labels['annotations']:
  if a['category_id'] not in (1,2) or a.get('track_id') is None:continue
  b=a['bbox_image']
  if min(b['w'],b['h'])<min_side:continue
  tracks[a['track_id']].append(a)
 out={}
 for track,items in tracks.items():
  items.sort(key=lambda a:a['image_id'])
  if len(items)<8:continue
  keep=np.linspace(0,len(items)-1,min(len(items),per_track)).astype(int)
  out[track]=[items[i] for i in sorted(set(keep))]
 return out


def main(per_track=32,limit_sequences=0):
 per_track=int(per_track);OUT.mkdir(parents=True,exist_ok=True)
 sequences=sorted(GSR.glob('SNGS-*/Labels-GameState.json'))
 if limit_sequences:sequences=sequences[:int(limit_sequences)]
 manifest=[];skipped=0
 for index,path in enumerate(sequences):
  labels=json.loads(path.read_text());folder=path.parent
  frames={i['image_id']:i['file_name'] for i in labels['images']}
  chosen=tracklets(labels,per_track)
  wanted=defaultdict(list)
  for track,items in chosen.items():
   for a in items:wanted[a['image_id']].append((track,a))
  cut=defaultdict(list)
  for image_id,items in sorted(wanted.items()):
   source=folder/'img1'/frames[image_id]
   if not source.exists():skipped+=len(items);continue
   with Image.open(source) as frame:
    frame=frame.convert('RGB')
    for track,a in items:
     b=a['bbox_image'];mx,my=MARGIN*b['w'],MARGIN*b['h']
     box=(max(0,b['x']-mx),max(0,b['y']-my),
          min(frame.width,b['x']+b['w']+mx),min(frame.height,b['y']+b['h']+my))
     target=OUT/folder.name/f'{track:03d}'/f"{image_id}.jpg"
     target.parent.mkdir(parents=True,exist_ok=True)
     if not target.exists():frame.crop(box).save(target,quality=92)
     cut[track].append({'path':str(target),'frame':int(image_id),
      'jersey':a.get('attributes',{}).get('jersey'),'height':b['h'],'width':b['w']})
  for track,views in cut.items():
   if len(views)>=8:
    a=chosen[track][0]
    manifest.append({'track':f'{folder.name}_{track:03d}','sequence':folder.name,
     'game':labels['info'].get('game_id'),'role':a.get('attributes',{}).get('role'),
     'team':a.get('attributes',{}).get('team'),
     'numbered_views':sum(1 for v in views if str(v['jersey'] or '').isdigit()),
     'views':sorted(views,key=lambda v:v['frame'])})
  if (index+1)%20==0:
   write_manifest(DATA/'gsr_tracklets.jsonl',manifest)
   print(f'{index+1}/{len(sequences)} sequences, tracklets {len(manifest)}, '
         f'crops {sum(len(m["views"]) for m in manifest)}',flush=True)
 write_manifest(DATA/'gsr_tracklets.jsonl',manifest)
 audit={'sequences':len(sequences),'tracklets':len(manifest),
  'crops':sum(len(m['views']) for m in manifest),'frames_missing':skipped,
  'source_games':len({m['game'] for m in manifest}),'frames_per_tracklet_cap':per_track,
  'tracklets_with_any_readable_number':sum(1 for m in manifest if m['numbered_views']),
  'note':'Frames are sampled evenly across the thirty-second sequence, not consecutively, so a set of eight spans real pose and scale change.'}
 atomic_json(DATA/'gsr_tracklets_audit.json',audit)
 print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main(*sys.argv[1:])
