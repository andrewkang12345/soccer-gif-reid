import json,hashlib,io
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from PIL import Image
from .metadata import DATA

def main():
 players={r['pid']:r for r in json.loads((DATA/'selected_players.json').read_text())}
 boxes={(r['game'],r['uai']):r for r in map(json.loads,(DATA/'mapped_boxes.jsonl').read_text().splitlines()) if r['pid'] in players}
 found={};reject=[]
 for archive in sorted((DATA/'archives').glob('*.zip')):
  with ZipFile(archive) as z:
   for info in z.infolist():
    if not info.filename.endswith('.png'):continue
    parts=info.filename.split('/');game='/'.join(parts[-5:-2]);uai=parts[-1].split('-')[-2];key=(game,uai)
    if key not in boxes:continue
    b=boxes[key];data=z.read(info);sha=hashlib.sha256(data).hexdigest()
    if key in found:
     assert found[key]['sha256']==sha;continue
    with Image.open(io.BytesIO(data)) as im:w,h=im.size;im.verify()
    if min(w,h)<16 or h<48:reject.append({'uai':uai,'reason':'small decoded crop'});continue
    p=DATA/'queries'/b['pid']/(uai+'.png');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    found[key]=dict(b,path=str(p),width=w,height=h,sha256=sha,domain='query',source_archive=str(archive),source_member=info.filename)
 rows=list(found.values());hashids={}
 for r in rows:
  if r['sha256'] in hashids and hashids[r['sha256']]!=r['pid']:raise ValueError('Cross-identity duplicate crop')
  hashids[r['sha256']]=r['pid']
 rows=list({r['sha256']:r for r in rows}.values())
 (DATA/'queries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 report={'mapped_selected':len(boxes),'found_unique_crops':len(rows),'missing':len(boxes)-len(found),'splits':dict(Counter(r['split'] for r in rows)),'rejected':reject}
 (DATA/'extraction_audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
