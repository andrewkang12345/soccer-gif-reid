"""Deterministic, stratified visual-review sample and independent roster join audit."""
import json,random,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from .metadata import DATA,ROOT

def main():
 rows=[json.loads(s) for s in (DATA/'mapped_boxes.jsonl').read_text().splitlines()]
 available={(r['game'],r['uai']):r for r in map(json.loads,(DATA/'queries.jsonl').read_text().splitlines())}
 rng=random.Random(91726);sample=[]
 for split,n in [('train',120),('val',40),('test',40)]:
  group=[r for r in rows if r['split']==split and (r['game'],r['uai']) in available]
  bygame=defaultdict(list)
  for r in group:bygame[r['game']].append(r)
  games=sorted(bygame);rng.shuffle(games)
  for i in range(min(n,len(group))):
   game=games[i%len(games)];choices=bygame[game]
   if not choices:continue
   r=choices.pop(rng.randrange(len(choices)));sample.append(dict(r,path=available[(r['game'],r['uai'])]['path']))
 failures=[]
 for r in rows:
  key=hashlib.sha256(r['game'].encode()).hexdigest()[:12]
  # Reading every game's metadata once keeps this audit independent and inexpensive.
  # Identity, jersey and club must all agree with exactly one explicit side's roster.
  if key not in cache:cache[key]=json.loads((DATA/'metadata'/key/'Labels-caption.json').read_text())
  cap=cache[key];matches=[]
  for side in ('home','away'):
   if cap[side]['name']!=r['team']:continue
   for p in cap['lineup'][side]['players']:
    if p['hash']==r['pid'] and str(p.get('shirt_number'))==str(r['number']) and (p.get('long_name') or p['name'])==r['name']:matches.append(p)
  if len(matches)!=1:failures.append({'pid':r['pid'],'game':r['game'],'uai':r['uai']})
 out=ROOT/'reports/soccernet_full_v2/label_review';out.mkdir(exist_ok=True)
 for page in range((len(sample)+39)//40):
  canvas=Image.new('RGB',(1600,1350),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(sample[page*40:(page+1)*40]):
   x=i%8*200;y=i//8*270
   draw.text((x+3,y+2),f"{page*40+i:03d} #{r['number']} {r['split']}",fill='black')
   draw.text((x+3,y+17),r['name'][:28],fill='black');draw.text((x+3,y+32),r['team'][:28],fill='black')
   with Image.open(r['path']) as im:thumb=ImageOps.contain(im.convert('RGB'),(190,215));canvas.paste(thumb,(x+(200-thumb.width)//2,y+48))
  canvas.save(out/f'page_{page:02d}.jpg',quality=94)
 (out/'sample.json').write_text(json.dumps(sample,indent=2))
 (out/'automated_audit.json').write_text(json.dumps({'rows_checked':len(rows),'roster_join_failures':failures,'sample':len(sample),'sample_splits':dict(Counter(r['split'] for r in sample)),'visual_review_complete':False},indent=2))
 print('audit',len(rows),'failures',len(failures),'sample',len(sample))
cache={}
if __name__=='__main__':main()
