"""Uncapped source-context, person-composition and duplicate screening.
Automatic screening is weak supervision, not facial identity certification.
"""
import json,re,cv2
from urllib.parse import urlparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageDraw
from ultralytics import YOLO
from .metadata import DATA,ROOT
from .collect_pages import named,norm
from soccer_gif_reid.common import read_rgb,atomic_json,write_manifest

REPORT=ROOT/'reports/soccernet_full_v2'
MIN_SUBJECT_PIXELS=56
MIN_CONFIDENCE=.30
CROP_MARGIN=.10


FOOTBALL_WORDS=('football','futbol','soccer','calcio','futebol','player','joueur','spieler','sport',
 'asroma','realmadrid','fcbarcelona','premierleague','bundesliga','uefa','fifa')


def football_context(team,source):
 """Accept a source that is about the player's own club as football context.

 The keyword list alone rejected official club sites whose address is an abbreviation the
 roster never uses -- mancity.com for Manchester City, bayer04.de for Bayer Leverkusen,
 herthabsc.com for Hertha Berlin, psv.nl for PSV. A club name token, a host label that is the
 club name, or a host label spelled as a subsequence of it all count as club context.
 Entertainment sites that merely share a player's name still fail every test; a short club
 name is only accepted as the site's own label, because such a fragment turns up by accident
 inside ordinary words.
 """
 lowered=source.lower()
 if any(word in lowered for word in FOOTBALL_WORDS):return True
 flat=norm(source);club=norm(team)
 if len(club)>=5 and club in flat:return True
 if any(t in flat for t in re.split(r'[^a-z0-9]+',norm_spaced(team)) if len(t)>=5):return True
 host=re.search(r'https?://([^/]+)',source)
 for label in ([norm(part) for part in host.group(1).split('.')] if host else []):
  if len(club)>=3 and (label==club or label.startswith(club)):return True
  if len(label)>=5 and subsequence(label,club):return True
 return False


def subsequence(short,long):
 """True when every letter of `short` appears in `long` in the same order."""
 it=iter(long)
 return all(c in it for c in short)


def norm_spaced(s):
 import unicodedata
 return re.sub('[^a-z0-9 ]',' ',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())


def judge(measurement):
 """Apply the current thresholds to a cached detection.

 Measurements are cached, decisions are not, so changing a threshold does not require the
 whole gallery to be detected again. The gallery stores the dominant person rather than the
 whole photograph, so a subject that is small inside a large press photo stays usable as long
 as it is alone; the query crops in this dataset are themselves around 60 pixels wide.
 """
 if 'person_box' not in measurement:return {'accepted':False,'reason':'No confident person detection'}
 box=measurement['person_box'];short=min(box[2]-box[0],box[3]-box[1])
 if measurement['secondary_ratio']>=.35:return {'accepted':False,'reason':'Several comparable people'}
 if short<MIN_SUBJECT_PIXELS or measurement['confidence']<MIN_CONFIDENCE:
  return {'accepted':False,'reason':'Subject too small or uncertain to crop'}
 return dict(measurement,accepted=True,reason='Dominant person',
  crop_box=crop_box(box,measurement['width'],measurement['height']))


def crop_box(box,width,height):
 """Padded, clipped person box used as the stored reference region."""
 x0,y0,x1,y1=[float(v) for v in box];mx=CROP_MARGIN*(x1-x0);my=CROP_MARGIN*(y1-y0)
 return [max(0.,x0-mx),max(0.,y0-my),min(float(width),x1+mx),min(float(height),y1+my)]

def pilot_exclusions():
 old=ROOT/'data/soccernet_named_v1';excluded=set()
 audit=json.loads((old/'final_audit.json').read_text())
 for r in audit['reference_exclusions']:excluded.add(r['candidate_id'])
 manual=json.loads((old/'manual_reference_exclusions.json').read_text())
 for r in map(json.loads,(old/'web_downloaded.jsonl').read_text().splitlines()):
  if r['display_index'] in manual.get(r['name'],[]):excluded.add(r['candidate_id'])
 return excluded


def main():
 torch.set_num_threads(4);cv2.setNumThreads(2)
 players={r['pid']:r for r in json.loads((DATA/'selected_players.json').read_text())}
 rows=[json.loads(s) for s in (DATA/'web_downloaded.jsonl').read_text().splitlines()]
 bad=pilot_exclusions();manualfile=DATA/'manual_reference_exclusions.json'
 if manualfile.exists():bad|=set(json.loads(manualfile.read_text()).get('candidate_ids',[]))
 eligible=[];reject=[]
 for r in rows:
  reason=None;source=r['source_title']+' '+r['source_url']+' '+r['image_url']
  if r['candidate_id'] in bad:reason='Previously reviewed exclusion'
  elif any(t in r['source_url'].lower() for t in ['futbin','futwiz','fifarosters','pesmaster','efootball','futgg','sofifa','pesdb']):reason='Game asset source'
  elif any(t in r['source_title'].lower() for t in ['drawing','cartoon','caricature','illustration','figurine']):reason='Nonphotographic source'
  elif r.get('reference_type')=='wikidata_entity':reason=None if r.get('wikidata_qid') else 'Missing Wikidata entity link'
  # The image's own caption is the evidence `collect_pages`/`mine_pages` selected on, and it is
  # more specific than the page title: one article names a dozen players in its captions.
  elif not (named(r['name'],source) or named(r['name'],r.get('image_alt',''))):reason='Neither the source title, the image URL nor the image caption names the player'
  # "Bruno" is a substring of "Bruno Genesio", so a one-word roster name inside a caption is not
  # evidence; it needs the page title or address to agree.
  elif len(r['name'].split())==1 and not named(r['name'],source):reason='One-word roster name evidenced only by an image caption'
  elif r.get('image_role')=='page_featured_image' and not named(r['name'],r['image_url']) and not named(r['name'],urlparse(r['source_url']).path):reason='Featured image is neither named itself nor on a page whose address names the player'
  elif (len(r['name'].split())==1 or r.get('reference_type')=='named_web_page') and not football_context(r['team'],source):reason='Player page lacks football-specific source context'
  if reason:reject.append({'candidate_id':r['candidate_id'],'reason':reason});continue
  if r.get('reference_type')=='wikidata_entity':evidence='wikidata entity link'
  elif r.get('image_role')=='page_featured_image':evidence='image URL names the player' if named(r['name'],r['image_url']) else 'page address names the player'
  elif named(r['name'],r.get('image_alt','')):evidence='image caption names the player'
  else:evidence='source title or image URL names the player'
  eligible.append(dict(r,name_evidence=evidence))
 cachefile=DATA/'reference_person_screen.json';cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
 model=YOLO('/mnt/data/sports_reID/yolo11m-seg.pt');composed=[]
 for i,r in enumerate(eligible):
  key=r['sha256']
  if key not in cache or 'measured' not in cache[key]:
   im=read_rgb(r['path']);pred=model.predict(im,imgsz=640,conf=.25,classes=[0],device=0,verbose=False)[0]
   boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
   order=sorted(range(len(boxes)),key=lambda j:float(np.prod(boxes[j,2:]-boxes[j,:2])),reverse=True)
   record={'measured':True,'width':im.width,'height':im.height}
   if order:
    areas=[float(np.prod(boxes[j,2:]-boxes[j,:2])) for j in order]
    record.update(person_box=boxes[order[0]].tolist(),confidence=float(scores[order[0]]),
     coverage=areas[0]/(im.width*im.height),secondary_ratio=areas[1]/areas[0] if len(areas)>1 else 0.)
   cache[key]=record
  decision=judge(cache[key])
  if decision['accepted']:composed.append(dict(r,person_screen=decision))
  else:reject.append({'candidate_id':r['candidate_id'],'reason':decision['reason']})
  if (i+1)%100==0:atomic_json(cachefile,cache);print('person screen',i+1,len(eligible),'passing',len(composed),flush=True)
 atomic_json(cachefile,cache)
 # An identical photo associated with different roster identities is ambiguous: remove all assignments.
 hashes=defaultdict(set)
 for r in composed:hashes[r['sha256']].add(r['pid'])
 ambiguous={h for h,pids in hashes.items() if len(pids)>1}
 kept=defaultdict(list);features={};orb=cv2.ORB_create(nfeatures=700)
 def feat(r):
  key=r['sha256']
  if key not in features:
   a=np.asarray(read_rgb(r['path']));scale=min(1.,600/max(a.shape[:2]));a=cv2.resize(a,(round(a.shape[1]*scale),round(a.shape[0]*scale)))
   kp,des=orb.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None);features[key]=(kp,des,a.shape)
  return features[key]
 def duplicate(a,b):
  if a['sha256']==b['sha256']:return True
  ka,da,sa=feat(a);kb,db,sb=feat(b)
  if da is None or db is None or len(db)<2:return False
  matches=[m for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
  if len(matches)<25:return False
  aa=np.float32([ka[m.queryIdx].pt for m in matches]);bb=np.float32([kb[m.trainIdx].pt for m in matches])
  _,mask=cv2.findHomography(aa,bb,cv2.RANSAC,3.)
  if mask is None:return False
  mask=mask.ravel().astype(bool)
  return bool(mask.sum()>=25 and mask.mean()>=.65 and np.all(np.ptp(aa[mask],axis=0)>np.array([sa[1],sa[0]])*.3))
 for index,r in enumerate(sorted(composed,key=lambda r:(r['pid'],-r['width']*r['height'],r['candidate_id']))):
  if index%200==0:print('duplicate screen',index,len(composed),flush=True)
  if r['sha256'] in ambiguous:reject.append({'candidate_id':r['candidate_id'],'reason':'Same photo assigned to multiple players'});continue
  old=next((a for a in kept[r['pid']] if duplicate(r,a)),None)
  if old:reject.append({'candidate_id':r['candidate_id'],'reason':'Verified duplicate','duplicate_of':old['candidate_id']});continue
  kept[r['pid']].append(r)
 result=[r for pid in sorted(kept) for r in kept[pid]]
 write_manifest(DATA/'screened_gallery.jsonl',result);atomic_json(DATA/'reference_screen_rejections.json',reject)
 audit={'candidate_photos':len(rows),'source_name_passing':len(eligible),'person_composition_passing':len(composed),'unique_screened_photos':len(result),'players':len(kept),'photos_per_player':dict(Counter(len(v) for v in kept.values())),'reasons':dict(Counter(r['reason'] for r in reject)),'reference_cap':None,'identity_certification':False,'thresholds':{'person_detection_confidence':.25,'dominant_person_confidence':MIN_CONFIDENCE,'minimum_subject_short_side_pixels':MIN_SUBJECT_PIXELS,'secondary_to_main_area_max':.35,'crop_margin':CROP_MARGIN},'name_evidence':dict(Counter(r.get('name_evidence') for r in result)),'gallery_region':'Stored references are the padded dominant-person box of the source photograph, not the whole image.','manual_review_complete':False}
 atomic_json(DATA/'reference_screen_audit.json',audit)
 sheets=REPORT/'reference_review';sheets.mkdir(exist_ok=True)
 # All retained candidates receive a source-indexed review sheet; no clipping at eight images.
 for pid,rs in kept.items():
  cols=6;canvas=Image.new('RGB',(1200,290*((len(rs)+cols-1)//cols)),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(rs):
   x=i%cols*200;y=i//cols*290
   draw.text((x+3,y+3),f'{i:02d} {r["name"][:25]}',fill='black')
   draw.text((x+3,y+18),r['candidate_id'][-16:],fill='black')
   thumb=ImageOps.contain(read_rgb(r['path']).crop(tuple(int(round(v)) for v in r['person_screen']['crop_box'])),(190,245));canvas.paste(thumb,(x+(200-thumb.width)//2,y+38))
  canvas.save(sheets/(pid+'.jpg'),quality=92)
 print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
