import json,torch
from collections import Counter
from PIL import Image
from ultralytics import YOLO
from soccer_gif_reid.masks import generate_person_mask
from soccer_gif_reid.common import read_rgb
from .metadata import DATA

def main():
 torch.set_num_threads(4);model=YOLO('/mnt/data/sports_reID/yolo11m-seg.pt');rows=[json.loads(s) for s in (DATA/'queries.jsonl').read_text().splitlines()];good=[];errors=[]
 for i,r in enumerate(rows):
  p=DATA/'masks'/(r['sha256']+'.png');p.parent.mkdir(exist_ok=True)
  try:
   if not p.exists():
    mask,method=generate_person_mask(model,read_rgb(r['path']),0);Image.fromarray(mask).save(p)
   good.append(dict(r,person_mask=str(p)))
  except ValueError as e:errors.append({'pid':r['pid'],'path':r['path'],'reason':str(e)})
  if (i+1)%100==0:print(i+1,len(good),len(errors),flush=True)
 (DATA/'masked_queries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in good));(DATA/'mask_errors.json').write_text(json.dumps(errors,indent=2));print('done',Counter(r['split'] for r in good),flush=True)
if __name__=='__main__':main()
