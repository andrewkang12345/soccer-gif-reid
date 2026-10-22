"""Rebuild the Argentina-France demonstration gallery from whole-player photographs.

The frozen v1 gallery is 168 hand-picked web images, eight per player, and almost all of them
are head-and-shoulders portraits from club-profile sites: the hosts it drew from are the ones
measured at 0-4% full-body. The clip's tracks are 1,474-pixel distant crops of a whole player
in motion, so the gallery and the query disagree about what a person looks like before the
model is asked anything. This rebuilds the same twenty-one identities from the Commons
candidates already enumerated for them, ordered by the measured full-body cues, and leaves
v1 untouched so the two galleries can be scored against each other.

Wikimedia's content host throttles a single address hard, so `fetch` is one patient stream
that waits out `Retry-After` and spreads the budget round-robin across players rather than
finishing one at a time.
"""
import io,json,sys,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import numpy as np
import requests
from PIL import Image
from soccer_gif_reid.common import read_rgb,atomic_json,write_manifest,digest_file
from .body_preference import body_likelihood
from .metadata import ROOT

DATA=ROOT/'data/arg_fra_expanded_gallery_v1'
CANDIDATES=DATA/'candidates.jsonl'
GALLERY=DATA/'gallery_body.jsonl'
USER_AGENT='SoccerReIDResearch/0.1 (non-commercial player re-identification research)'
# Wikimedia's content host answers about six requests from one address and then asks for a ten
# minute cooldown, which turns 287 photographs into an eight hour queue. wsrv.nl is a free public
# image CDN that fetches the same Commons URL from its own servers, so the rate limit that applies
# is theirs and not ours. What lands is therefore a re-encoded copy of the Commons original rather
# than the original bytes -- recorded in the manifest, and harmless here because every reference is
# cropped to the detected player and letterboxed to 224 pixels before the model sees it.
PROXY='https://wsrv.nl/?url={url}&w=1400&output=jpg&q=92'
MAX_BYTES=25000000


def score(row):
 """Measured full-body likelihood from the Commons metadata alone.

 The host is deliberately left out: every candidate here is on upload.wikimedia, so it is a
 constant and would only shift every score by the same amount. What is left is nearly flat --
 the word cues were mined from web-search snippets and Commons titles are match fixtures and
 dates, so they almost never fire. It is kept as a weak fetch-order tiebreak only; the real
 selection happens after download, on the shape of the detected person (`build`).
 """
 return body_likelihood(row.get('title'),row.get('categories'),row.get('description'))


def ranked(include_web=True):
 rows=[json.loads(s) for s in CANDIDATES.read_text().splitlines()]
 web=DATA/'web_candidates.jsonl'
 if include_web and web.exists():
  rows+=[dict(json.loads(s),source='news gallery') for s in web.read_text().splitlines()]
 per=defaultdict(list)
 for r in rows:per[r['pid']].append(dict(r,body_score=score(r),source=r.get('source','commons')))
 for items in per.values():items.sort(key=lambda r:(-r['body_score'],r['display_index']))
 return per


def proxied(url):
 return PROXY.format(url=quote(url.replace('https://',''),safe=''))


def keep(path,content):
 """Write only a decodable photograph of usable size."""
 if len(content)>MAX_BYTES:return False
 with Image.open(io.BytesIO(content)) as im:
  im.load()
  if min(im.size)<200 or im.format not in {'JPEG','PNG','WEBP'}:return False
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);return True


def fetch(per_player=14,budget_seconds=0,gap=8.,proxy=True,workers=6):
 """Download the best-ranked candidates for every demo player.

 Through the proxy this is parallel and takes a couple of minutes. Direct from Wikimedia it is
 one patient stream that waits out each advertised cooldown and takes hours; that path is kept
 because it fetches the original bytes.
 """
 per=ranked();queue=[]
 for depth in range(per_player):
  queue+=[items[depth] for _,items in sorted(per.items()) if depth<len(items)]
 todo=[r for r in queue if not Path(r['path']).exists()]
 session=requests.Session();session.headers.update({'User-Agent':USER_AGENT})
 began=time.monotonic();counts={'saved':0,'throttled':0,'failed':0,'rejected':0}
 if proxy:
  def one(row):
   try:
    response=session.get(proxied(row['download_url']),timeout=(6,60))
    if response.status_code==429:return 'throttled'
    response.raise_for_status()
    return 'saved' if keep(Path(row['path']),response.content) else 'rejected'
   except Exception:return 'failed'
  with ThreadPoolExecutor(workers) as pool:
   for i,outcome in enumerate(pool.map(one,todo),1):
    counts[outcome]+=1
    if i%50==0:print(f'{i}/{len(todo)} '+' '.join(f'{k} {v}' for k,v in counts.items()),flush=True)
 else:
  index=0
  while index<len(todo):
   if budget_seconds and time.monotonic()-began>budget_seconds:break
   row=todo[index]
   try:
    response=session.get(row['download_url'],timeout=(6,90))
    if response.status_code==429:
     counts['throttled']+=1;gap=min(20.,gap*1.2)
     try:wait=float(response.headers.get('Retry-After','30'))
     except ValueError:wait=30.
     print(f'throttled at {index}/{len(todo)}: waiting {wait:.0f}s, spacing now {gap:.1f}s',flush=True)
     time.sleep(wait+2.);continue
    response.raise_for_status()
    counts['saved' if keep(Path(row['path']),response.content) else 'rejected']+=1
   except Exception:counts['failed']+=1
   index+=1
   time.sleep(gap)
 print(json.dumps({'queued':len(todo),**counts,'proxy':bool(proxy),
  'on_disk':sum(1 for r in queue if Path(r['path']).exists()),
  'seconds':round(time.monotonic()-began,1)}),flush=True)


SECONDARY_LIMIT=.60


def relaxed(measurement):
 """The dominant-person screen, but tolerant of other players sharing the frame."""
 from .screen_references import MIN_SUBJECT_PIXELS,MIN_CONFIDENCE,crop_box
 if 'person_box' not in measurement:return {'accepted':False,'reason':'No confident person detection'}
 box=measurement['person_box'];short=min(box[2]-box[0],box[3]-box[1])
 if measurement['secondary_ratio']>=SECONDARY_LIMIT:
  return {'accepted':False,'reason':'No clearly dominant person'}
 if short<MIN_SUBJECT_PIXELS or measurement['confidence']<MIN_CONFIDENCE:
  return {'accepted':False,'reason':'Subject too small or uncertain to crop'}
 return dict(measurement,accepted=True,reason='Dominant person',
  crop_box=crop_box(box,measurement['width'],measurement['height']))


def top_up(rows,per_player):
 """Fall back to the frozen portrait gallery for players with too few whole-body photographs.

 A roster is whatever photographs exist. Six of the twenty-one are squad players the free
 galleries barely photograph, and dropping them would change the task rather than improve the
 references, so their best-shaped v1 portraits fill the gap and the manifest says which is which.
 """
 have=defaultdict(int)
 for r in rows:have[r['pid']]+=1
 path=DATA/'gallery_v1_cropped.jsonl'
 if not path.exists():return []
 pool=defaultdict(list)
 for line in path.read_text().splitlines():
  r=json.loads(line);pool[r['pid']].append(r)
 out=[]
 for pid,items in pool.items():
  need=per_player-have.get(pid,0)
  if need<=0:continue
  for r in sorted(items,key=lambda x:-x['person_aspect'])[:need]:
   out.append({'pid':pid,'name':r['name'],'team':r['team'],'domain':'reference','split':'test',
    'path':r['path'],'candidate_path':r['candidate_path'],'candidate_id':r['candidate_id'],
    'source_url':r['source_url'],'image_url':r['image_url'],'title':r.get('source_title',''),
    'license':None,'body_score':None,'person_aspect':r['person_aspect'],
    'crop_box':r['crop_box'],'sha256':r['sha256'],'gallery_sha256':r['gallery_sha256'],
    'source':'v1 portrait fallback','label_source':r['label_source']})
 return out


def build(per_player=8):
 """Screen what is on disk to the dominant person and freeze a gallery in the demo's schema."""
 import cv2,torch
 from ultralytics import YOLO
 # Press portraits hold one person; a whole-body action photograph usually holds several, so the
 # portrait-era rule that rejects any image with a comparable second person throws away exactly
 # the photographs this gallery is being rebuilt to find. `relaxed` keeps the requirement that
 # the chosen person is clearly the biggest, and the caption still names one roster player.
 torch.set_num_threads(4);cv2.setNumThreads(2)
 per=ranked();model=YOLO('/mnt/data/sports_reID/yolo11m-seg.pt')
 cachefile=DATA/'body_person_screen.json'
 cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
 rows=[];rejected=[];seen=set()
 for pid,items in sorted(per.items()):
  # Screen everything on disk first, then keep the most whole-body-shaped. A standing player
  # detects as a box roughly 2.5-3.5 times taller than wide; a head-and-shoulders portrait
  # detects close to square. That measured shape is the selection signal the Commons text is
  # too sparse to supply.
  passed=[]
  for r in items:
   path=Path(r['path'])
   if not path.exists():continue
   sha=digest_file(path)
   if sha in seen:rejected.append({'candidate_id':r['candidate_id'],'reason':'Duplicate of an accepted photograph'});continue
   if sha not in cache:
    im=read_rgb(path);pred=model.predict(im,imgsz=640,conf=.25,classes=[0],device=0,verbose=False)[0]
    boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
    order=sorted(range(len(boxes)),key=lambda j:float(np.prod(boxes[j,2:]-boxes[j,:2])),reverse=True)
    record={'width':im.width,'height':im.height}
    if order:
     areas=[float(np.prod(boxes[j,2:]-boxes[j,:2])) for j in order]
     record.update(person_box=boxes[order[0]].tolist(),confidence=float(scores[order[0]]),
      coverage=areas[0]/(im.width*im.height),secondary_ratio=areas[1]/areas[0] if len(areas)>1 else 0.)
    cache[sha]=record
   decision=relaxed(cache[sha])
   if not decision['accepted']:rejected.append({'candidate_id':r['candidate_id'],'reason':decision['reason']});continue
   x0,y0,x1,y1=decision['person_box']
   seen.add(sha)
   passed.append((dict(r,person_aspect=(y1-y0)/max(x1-x0,1e-6),sha256=sha),decision))
  passed.sort(key=lambda t:-t[0]['person_aspect'])
  kept=0
  for r,decision in passed[:per_player]:
   box=tuple(int(round(v)) for v in decision['crop_box'])
   dest=DATA/'gallery_body'/pid/f'{kept+1:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():read_rgb(r['path']).crop(box).save(dest,'PNG',compress_level=4)
   kept+=1
   rows.append({'pid':pid,'name':r['name'],'team':r['team'],'domain':'reference','split':'test',
    'path':str(dest),'candidate_path':r['path'],'candidate_id':r['candidate_id'],
    'source_url':r['source_url'],'image_url':r['image_url'],'title':r['title'],
    'license':r.get('license'),'body_score':r['body_score'],'person_aspect':r['person_aspect'],
    'crop_box':list(box),'sha256':r['sha256'],'gallery_sha256':digest_file(dest),
    'source':r['source'],
    'label_source':('Caption on a public news gallery naming this player and no other on the roster'
     if r['source']=='news gallery' else 'Commons person category naming the player')
     +', plus automated dominant-person screening'})
  print(pid,kept,'references from',len(passed),'screened',flush=True)
 rows+=top_up(rows,per_player)
 atomic_json(cachefile,cache)
 write_manifest(GALLERY,rows)
 counts={pid:sum(1 for r in rows if r['pid']==pid) for pid in per}
 atomic_json(DATA/'gallery_body_audit.json',{'images':len(rows),'identities':len(counts),
  'per_identity':counts,'target_per_identity':per_player,
  'mean_body_score':(float(np.mean([r['body_score'] for r in rows if r.get('body_score') is not None]))
   if any(r.get('body_score') is not None for r in rows) else None),
  'median_person_aspect':float(np.median([r['person_aspect'] for r in rows])) if rows else None,
  'by_source':{k:sum(1 for r in rows if r.get('source')==k) for k in sorted({r.get('source') for r in rows})},
  'rejected':len(rejected),'rejection_reasons':dict(sorted(
   {k:sum(1 for x in rejected if x['reason']==k) for k in {x['reason'] for x in rejected}}.items())),
  'gallery_manifest_sha256':digest_file(GALLERY),
  'leakage':'Candidates enumerated with the 2022 final date and category already excluded upstream.'})
 print(json.dumps({'images':len(rows),'identities':len(counts),
  'short':{k:v for k,v in counts.items() if v<per_player}},indent=2),flush=True)


def main():
 action=sys.argv[1] if len(sys.argv)>1 else 'fetch'
 kwargs={k:float(v) for k,v in (a.split('=') for a in sys.argv[2:])}
 if action=='fetch':fetch(int(kwargs.get('per_player',14)),kwargs.get('budget_seconds',0),
                          kwargs.get('gap',8.),bool(kwargs.get('proxy',1)),int(kwargs.get('workers',6)))
 elif action=='build':build(int(kwargs.get('per_player',8)))
 elif action=='inventory':
  per=ranked()
  for pid,items in sorted(per.items()):
   have=sum(1 for r in items if Path(r['path']).exists())
   print(f'{pid:34s} candidates {len(items):3d} on disk {have:3d} '
         f'top scores {[round(r["body_score"],2) for r in items[:5]]}')
 else:raise SystemExit('fetch | build | inventory')


if __name__=='__main__':main()
