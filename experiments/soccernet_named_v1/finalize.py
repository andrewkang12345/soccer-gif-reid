"""Freeze reviewed references and enforce real-player and content split separation."""
import json,shutil,hashlib,cv2
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from soccer_gif_reid.common import digest_file
from .metadata import DATA,ROOT

def main():
 manual=json.loads((DATA/'manual_reference_exclusions.json').read_text());extra={'David Jones':[22,24,27,28],'Lukasz Skorupski':[22],'Giorgio Chiellini':[22],'Scott Dann':[24,25,26],'Jordon Mutch':[5],'Joel Ward':[24,28]}
 for k,v in extra.items():manual.setdefault(k,[]).extend(v)
 players={r['pid']:r for r in json.loads((DATA/'selected_players.json').read_text())};rows=[json.loads(s) for s in (DATA/'provisional_gallery.jsonl').read_text().splitlines()];kept=defaultdict(list);excluded=[];orb=cv2.ORB_create(nfeatures=700);feats={}
 def features(r):
  if r['candidate_id'] not in feats:
   im=cv2.imread(r['path']);scale=min(1.,600/max(im.shape[:2]));im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));kp,des=orb.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None);feats[r['candidate_id']]=(kp,des,im.shape)
  return feats[r['candidate_id']]
 def duplicate(a,b):
  if a['sha256']==b['sha256']:return True
  ka,da,sa=features(a);kb,db,sb=features(b)
  if da is None or db is None:return False
  good=[m for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
  if len(good)<25:return False
  aa=np.float32([ka[m.queryIdx].pt for m in good]);bb=np.float32([kb[m.trainIdx].pt for m in good]);h,mask=cv2.findHomography(aa,bb,cv2.RANSAC,3.)
  if mask is None:return False
  m=mask.ravel().astype(bool)
  if m.sum()<25 or m.mean()<.65:return False
  span=np.ptp(aa[m],axis=0);return bool(span[0]>.3*sa[1] and span[1]>.3*sa[0])
 for r in rows:
  if r['display_index'] in manual.get(r['name'],[]):excluded.append({'candidate_id':r['candidate_id'],'reason':'manual composition/duplicate/namesake exclusion'});continue
  duplicate_of=next((a for a in kept[r['pid']] if duplicate(r,a)),None)
  if duplicate_of:excluded.append({'candidate_id':r['candidate_id'],'reason':'geometrically verified near-duplicate','duplicate_of':duplicate_of['candidate_id']});continue
  kept[r['pid']].append(r)
 # At least three separately sourced photographs; never pad with duplicate crops.
 eligible={pid for pid,rs in kept.items() if len(rs)>=3};query=[r for r in map(json.loads,(DATA/'masked_queries.jsonl').read_text().splitlines()) if r['pid'] in eligible]
 eligible &= {r['pid'] for r in query};query=[r for r in query if r['pid'] in eligible];gallery=[]
 for pid in sorted(eligible):
  p=players[pid]
  for i,r in enumerate(kept[pid]):
   with Image.open(r['path']) as im:ext={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(im.format,'.img')
   path=DATA/'gallery'/p['split']/(p['name'].replace(' ','_')+'_'+pid)/(f'{i+1:02d}'+ext);path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r['path'],path)
   gallery.append(dict(r,path=str(path),candidate_path=r['path'],split=p['split'],domain='reference',label_source='Explicit name in source title/URLs plus manual composition/namesake review; source-context labels not independently identity-certified'))
 allrows=query+gallery
 for key in ('pid','name','sha256'):
  sets={s:{r[key] for r in allrows if r['split']==s} for s in ('train','val','test')}
  for a,b in [('train','val'),('train','test'),('val','test')]:assert not sets[a]&sets[b],(key,a,b,sets[a]&sets[b])
 assert not {r['sha256'] for r in query}&{r['sha256'] for r in gallery}
 assert len({r['sha256'] for r in gallery})==len(gallery)
 for filename,rs in [('gallery.jsonl',gallery),('final_queries.jsonl',query),('manifest.jsonl',allrows)]:
  (DATA/filename).write_text(''.join(json.dumps(r)+'\n' for r in rs))
 audit={'frozen_utc':datetime.now(timezone.utc).isoformat(),'gallery_manifest_sha256':digest_file(DATA/'gallery.jsonl'),'query_manifest_sha256':digest_file(DATA/'final_queries.jsonl'),'splits':{s:{'players':len({r['pid'] for r in query if r['split']==s}),'queries':sum(r['split']==s for r in query),'references':sum(r['split']==s for r in gallery),'games':len({r['game'] for r in query if r['split']==s})} for s in ('train','val','test')},'reference_count_distribution':dict(Counter(len(kept[pid]) for pid in eligible)),'excluded_players':[p for pid,p in players.items() if pid not in eligible],'reference_exclusions':excluded,'disjointness':'Real roster player hash, exact name, source game and original image hashes checked. No query/reference exact-byte overlap.','caveats':['Small curated pilot, not all SoccerNet.','Automatic roster-evidence labels and source-caption reference labels may contain residual errors.','References vary from 3 to 8 photos/player after screening.','SoccerNet original action IDs are not treated as global identities.','Training includes Messi; Argentina-France is not an independent evaluation for this run.']}
 atomic=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');atomic(DATA/'final_audit.json',audit)
 # Preserve full source pixels; review sheets alone are resized.
 out=ROOT/'reports/soccernet_named_v1/final_gallery';out.mkdir(exist_ok=True)
 for pid in eligible:
  rs=[r for r in gallery if r['pid']==pid];canvas=Image.new('RGB',(1200,400),'white');d=ImageDraw.Draw(canvas);d.text((5,5),players[pid]['name'],fill='black')
  for i,r in enumerate(rs):
   with Image.open(r['path']) as im:t=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(145,350));canvas.paste(t,(150*i,35))
  canvas.save(out/f'{pid}.jpg')
 print(json.dumps({k:audit[k] for k in ('splits','reference_count_distribution','excluded_players')},indent=2))
if __name__=='__main__':main()
