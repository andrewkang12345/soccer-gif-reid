import json,re,unicodedata
from collections import defaultdict,Counter
from PIL import Image,ImageOps,ImageDraw
from .metadata import DATA,ROOT
REPORT=ROOT/'reports/soccernet_named_v1'
def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode())
def main():
 players=json.loads((DATA/'selected_players.json').read_text());byid=defaultdict(list)
 for r in map(json.loads,(DATA/'web_downloaded.jsonl').read_text().splitlines()):byid[r['pid']].append(r)
 selected=[];reject=[]
 for p in players:
  rows=byid[p['pid']];chosen=[]
  tokens=[norm(t) for t in p['name'].split() if len(norm(t))>=4];target=norm(p['name']);eligible=[]
  if p['name']=='Juan Manuel Mata': tokens=['juan','mata'];target='juanmata'
  for r in rows:
   source=norm(r['source_title']+' '+r['source_url']+' '+r['image_url']);desc=norm(r['search_description'])
   if any(t in r['source_url'].lower() for t in ['futbin','futwiz','fifarosters','pesmaster','efootball','futgg','sofifa','pesdb']):reason='game asset source'
   elif not ((tokens and all(t in source for t in tokens)) or target in source):reason='insufficient explicit name in source title/URLs'
   elif any(t in r['source_title'].lower() for t in ['drawing','cartoon','caricature','illustration','figurine']):reason='nonphotographic source'
   else:reason=None
   if reason:reject.append({'pid':p['pid'],'candidate_id':r['candidate_id'],'reason':reason});continue
   eligible.append(r)
  # Fill both view categories first, then supplement; no score-based selection.
  for category,limit in [('portrait_search',4),('fullbody_search',4),(None,8)]:
   start=len(chosen);target_count=8 if category is None else start+limit
   for r in eligible:
    if len(chosen)>=min(8,target_count):break
    if category and r['reference_type']!=category:continue
    if any(r['candidate_id']==a['candidate_id'] or r['sha256']==a['sha256'] or (int(r['dhash'],16)^int(a['dhash'],16)).bit_count()<=8 for a in chosen):continue
    chosen.append(r)
  selected+=chosen
 (DATA/'provisional_gallery.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected));(DATA/'reference_screen_rejections.json').write_text(json.dumps(reject,indent=2))
 for page in range((len(players)+7)//8):
  canvas=Image.new('RGB',(1450,8*195),'white');draw=ImageDraw.Draw(canvas)
  for j,p in enumerate(players[page*8:(page+1)*8]):
   y=j*195;draw.text((3,y+8),p['name'],fill='black');draw.text((3,y+26),p['pid']+' '+p['split'],fill='black')
   rows=[r for r in selected if r['pid']==p['pid']]
   for i,r in enumerate(rows):
    x=180+i*158
    with Image.open(r['path']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(150,165));canvas.paste(thumb,(x,y+20))
    draw.text((x,y+3),str(r['display_index']),fill='black')
  canvas.save(REPORT/f'review_{page:02d}.jpg',quality=93)
 print('selected',len(selected),'counts',Counter(Counter(r['pid'] for r in selected).values()));print('shortfalls',[(p['name'],sum(r['pid']==p['pid'] for r in selected)) for p in players if sum(r['pid']==p['pid'] for r in selected)<8])
if __name__=='__main__':main()
