"""Freeze manually curated references before evaluation; retain source provenance."""
import hashlib,json,shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from .collect import DATA,REPORT

def main():
 s=json.loads((DATA/'provisional_selection.json').read_text())
 candidates=[json.loads(x) for x in (DATA/'web_downloaded.jsonl').read_text().splitlines()]
 rows=[]
 for pid,indices in sorted(s.items()):
  assert len(indices)==8 and len(set(indices))==8
  sheet=Image.new('RGB',(1200,680),'white');draw=ImageDraw.Draw(sheet)
  for j,i in enumerate(indices):
   r=next(dict(r) for r in candidates if r['pid']==pid and r['display_index']==i)
   src=Path(r['path'])
   with Image.open(src) as im:
    ext={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(im.format,'.'+im.format.lower())
    thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(290,290))
   dst=DATA/'gallery'/pid/f'{j+1:02d}{ext}';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
   r.update(path=str(dst),candidate_path=str(src),domain='reference',split='test',label_source='Public image-search source title/caption context; manual composition and duplicate review; not independently identity-certified')
   rows.append(r);x=(j%4)*300;y=(j//4)*340;sheet.paste(thumb,(x,y+25));draw.text((x+4,y+4),f'{j+1:02d} candidate {i}',fill='black')
  out=REPORT/'gallery_sheets';out.mkdir(parents=True,exist_ok=True);sheet.save(out/f'{pid}.jpg')
 assert len(rows)==168 and len({r['sha256'] for r in rows})==168
 manifest=DATA/'gallery.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 audit={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'gallery_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'images':168,'identities':21,'images_per_identity':8,'actual_training_references_per_identity':8,'raw_mupnit_mean_reference_count':144.32,'raw_mupnit_median_reference_count':173,'selection_before_new_inference':True,'candidate_downloads':len(candidates),'selection':s,'reference_search_types':dict(Counter(r['reference_type'] for r in rows)),'pixel_changes':'None; original downloaded image bytes copied. Model-native letterbox at inference.','review':'All player candidate sheets reviewed for composition and crop duplicates. Exact hashes unique. dHash/ORB candidates reviewed; De Paul 0/1 same-source portrait removed, others flagged by shared kit/templates or distinct photographs. Ambiguous source contexts removed for Tagliafico, Enzo, Molina, Tchouameni, Theo; Upamecano possible rendered portrait replaced.','limitations':['Source-context labels are not independently identity-certified.','Web imagery spans years, clubs and kits; some small background people remain.','This is an exploratory previously inspected single-match evaluation, not an untouched production test.']}
 (DATA/'gallery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({k:audit[k] for k in ('images','identities','gallery_manifest_sha256')}))
if __name__=='__main__':main()
