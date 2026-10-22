"""Deterministic review sample of the screened gallery, stratified by split and name evidence.

Sheets show the stored region -- the dominant-person crop -- because that, not the whole source
photograph, is what training and retrieval see. The sample is drawn after every automatic screen
so the residual error rate is judged on the gallery that is actually used, and each evidence type
is sampled separately so a widened rule can be checked on its own.
"""
import json,random
from collections import defaultdict
from PIL import Image,ImageOps,ImageDraw
from .metadata import DATA,ROOT
from soccer_gif_reid.common import read_rgb,atomic_json,digest_file

PER_EVIDENCE=90
COLUMNS,ROWS=6,6


def main():
 rows=[json.loads(s) for s in (DATA/'screened_gallery.jsonl').read_text().splitlines()]
 players={r['pid']:r for r in json.loads((DATA/'players.json').read_text())}
 rng=random.Random(97261);groups=defaultdict(lambda:defaultdict(list));sample=[]
 for r in rows:groups[r.get('name_evidence','unrecorded')][r['pid']].append(r)
 for evidence in sorted(groups):
  ids=sorted(groups[evidence]);rng.shuffle(ids);taken=0
  for depth in range(3):
   for pid in ids:
    items=groups[evidence][pid]
    if depth<len(items) and taken<PER_EVIDENCE:
     sample.append(dict(items[depth],split=players[pid]['split'],evidence=evidence));taken+=1
 out=ROOT/'reports/soccernet_full_v2/reference_sample';out.mkdir(parents=True,exist_ok=True)
 for old in out.glob('page_*.jpg'):old.unlink()
 page_size=COLUMNS*ROWS
 for page in range((len(sample)+page_size-1)//page_size):
  canvas=Image.new('RGB',(240*COLUMNS,260*ROWS),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(sample[page*page_size:(page+1)*page_size]):
   x=i%COLUMNS*240;y=i//COLUMNS*260
   draw.text((x+3,y+3),f'{page*page_size+i:03d} {r["name"][:27]}',fill='black')
   draw.text((x+3,y+17),f'{r["split"]} {r["evidence"][:26]}',fill='black')
   crop=read_rgb(r['path']).crop(tuple(int(round(v)) for v in r['person_screen']['crop_box']))
   thumb=ImageOps.contain(crop,(230,208));canvas.paste(thumb,(x+(240-thumb.width)//2,y+34))
  canvas.save(out/f'page_{page:02d}.jpg',quality=93)
 atomic_json(out/'sample.json',{'screened_manifest_sha256':digest_file(DATA/'screened_gallery.jsonl'),
  'per_evidence_target':PER_EVIDENCE,'rows':sample})
 print(json.dumps({'sampled':len(sample),'players':len({r['pid'] for r in sample}),
  'pages':len(list(out.glob('page_*.jpg'))),
  'by_evidence':{k:sum(1 for r in sample if r['evidence']==k) for k in sorted(groups)}},indent=2))


if __name__=='__main__':main()
