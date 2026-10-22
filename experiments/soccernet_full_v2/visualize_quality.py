"""Show the tracklets the model aggregates, and what the quality head thought of each view.

The head's effect on retrieval was not distinguishable from noise, which leaves an obvious
question: is it doing anything structured at all, or is it noise with a softmax on it. A weight
is only interesting next to the crop that earned it, so every group is drawn as a row, views
ordered by weight, each with its weight as a number and a bar.

Groups are shown most-opinionated first — largest ratio of top weight to mean weight — because a
head that is merely uniform has nothing to show, and the failure to look for would be confident
weight on a crop a person can see is useless. Alongside the sheets, the weights are correlated
with what a person would guess quality tracks: how many pixels the crop has, and how elongated it
is, a proxy for a standing player against a sprawled or partly occluded one.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import read_rgb,atomic_json
from soccer_gif_reid.model import build_model
from .metadata import DATA,ROOT
from .quality_head import QualityHead
from .train import Images

REPORT=ROOT/'reports/soccernet_full_v2/quality_weights'
CELL=118
ROWS_PER_SHEET=14


def render(groups,order,label,folder,stem):
    folder.mkdir(parents=True,exist_ok=True)
    for old in folder.glob(f'{stem}*.jpg'):old.unlink()
    widest=max(len(groups[k]['views']) for k in order)
    width=CELL*min(widest,12)+250
    for page in range((len(order)+ROWS_PER_SHEET-1)//ROWS_PER_SHEET):
        rows=order[page*ROWS_PER_SHEET:(page+1)*ROWS_PER_SHEET]
        canvas=Image.new('RGB',(width,(CELL+46)*len(rows)),'white');draw=ImageDraw.Draw(canvas)
        for i,key in enumerate(rows):
            g=groups[key];y=i*(CELL+46)
            correct=g['correct']
            draw.text((6,y+4),f"{label(key)}  {len(g['views'])} views  "
                             f"top/mean {g['ratio']:.2f}  {'correct' if correct else 'WRONG: '+label((g['predicted'],))}",
                      fill=(24,132,60) if correct else (190,38,38))
            for j,(view,weight) in enumerate(zip(g['views'],g['weights'])):
                if j>=12:break
                x=250+j*CELL if False else 6+j*CELL
                try:thumb=ImageOps.contain(read_rgb(view['path']),(CELL-14,CELL-16))
                except Exception:continue
                canvas.paste(thumb,(x+(CELL-14-thumb.width)//2,y+20))
                bar=int((CELL-16)*weight/max(g['weights']))
                draw.rectangle([x,y+CELL+22,x+bar,y+CELL+28],fill=(40,90,200))
                draw.text((x,y+CELL+30),f'{weight:.3f}  {view["width"]}x{view["height"]}',fill=(90,90,90))
        canvas.save(folder/f'{stem}_{page:02d}.jpg',quality=92)
    return (len(order)+ROWS_PER_SHEET-1)//ROWS_PER_SHEET


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--head',type=Path,default=ROOT/'runs/quality_head_v2/best.pt')
    ap.add_argument('--split',default='test')
    ap.add_argument('--sheets',type=int,default=4)
    args=ap.parse_args()
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    queries=[r for r in rows if r['split']==args.split and r['domain']=='query']
    references=[r for r in rows if r['split']==args.split and r['domain']=='reference']
    names={r['pid']:r['name'] for r in rows}

    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=build_model('cuda:0',ROOT/'weights');model.load_state_dict(payload['model']);model.eval()
    head=QualityHead().cuda()
    saved=torch.load(args.head,map_location='cpu',weights_only=False)
    head.load_state_dict(saved['head'])
    if 'projection' in saved:model.projection.load_state_dict(saved['projection'])
    feats=[]
    with torch.inference_mode():
        for x in DataLoader(Images(queries+references),batch_size=32,num_workers=4):
            with torch.autocast('cuda',dtype=torch.bfloat16):base=model.encode_base(x.cuda())
            feats.append(model.projection(base.float()).cpu())
    feats=torch.cat(feats)
    gallery=F.normalize(feats[len(queries):],dim=-1).numpy()
    gallery_ids=np.asarray([r['pid'] for r in references])

    grouped=defaultdict(list)
    for i,r in enumerate(queries):grouped[(r['pid'],r['game'])].append(i)
    groups={};pairs=[]
    for key,idx in grouped.items():
        block=feats[idx]
        with torch.inference_mode():z,w=head(block.unsqueeze(0).cuda())
        w=w[0].cpu().numpy();z=z[0].cpu().numpy()
        roster=sorted({queries[j]['pid'] for j in grouped if j[1]==key[1]} if False else
                      {k[0] for k in grouped if k[1]==key[1]})
        keep=np.flatnonzero(np.isin(gallery_ids,roster))
        best=gallery_ids[keep][int(np.argmax(gallery[keep]@z))]
        groups[key]={'views':[{'path':queries[j]['path'],'width':int(queries[j]['width']),
                               'height':int(queries[j]['height'])} for j in idx],
                     'weights':w.tolist(),'ratio':float(w.max()/w.mean()),
                     'correct':bool(best==key[0]),'predicted':best}
        for j,weight in zip(idx,w):
            pairs.append((float(weight)*len(idx),int(queries[j]['width'])*int(queries[j]['height']),
                          int(queries[j]['height'])/max(1,int(queries[j]['width']))))
    label=lambda k:names.get(k[0],k[0])
    opinionated=sorted(groups,key=lambda k:-groups[k]['ratio'])[:args.sheets*ROWS_PER_SHEET]
    uniform=sorted(groups,key=lambda k:groups[k]['ratio'])[:ROWS_PER_SHEET]
    pages=render(groups,opinionated,label,REPORT,'opinionated')+render(groups,uniform,label,REPORT,'uniform')

    relative=np.array([p[0] for p in pairs]);area=np.array([p[1] for p in pairs]);aspect=np.array([p[2] for p in pairs])
    summary={'checkpoint':str(args.checkpoint),'head':str(args.head),'head_epoch':saved.get('epoch'),
     'split':args.split,'groups':len(groups),'views':len(pairs),'sheets':pages,
     'weight_ratio':{'mean':float(np.mean([g['ratio'] for g in groups.values()])),
                     'max':float(np.max([g['ratio'] for g in groups.values()]))},
     'correlation_of_relative_weight_with':{
        'crop_pixels':float(np.corrcoef(relative,np.log(area))[0,1]),
        'crop_aspect_ratio':float(np.corrcoef(relative,aspect)[0,1])},
     'note':'Relative weight is the weight times the group size, so 1.0 means the view got exactly its equal share.'}
    atomic_json(REPORT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
