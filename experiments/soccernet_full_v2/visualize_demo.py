"""Each Argentina-France track beside the roster photographs it was matched to.

Scores are centred — each prototype's mean over this match's own tracks subtracted — because the
raw cosines are dominated by hubness: three prototypes sit closest to nearly everything and take
eighteen of twenty-one tracks. Centring is available at inference, since a tracker hands you all
the match's tracks before any name is committed.

Each row is one track: a few of its crops, then the three roster photographs it scored highest,
and the true player's photograph with its rank when it is not already among them. That makes the
failure legible — whether the model is choosing a plausible lookalike or something unrelated.
"""
import argparse,json,re,unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import read_rgb,atomic_json
from soccer_gif_reid.model import build_model
from .metadata import ROOT
from .train import Images

GALLERY=ROOT/'data/arg_fra_expanded_gallery_v1/gallery.jsonl'
PROBE=ROOT/'data/arg_fra_roster_probe_v1/manifest.jsonl'
REPORT=ROOT/'reports/soccernet_full_v2/demo_arg_fra'
CROPS,CANDIDATES,CELL,ROW=4,3,104,150


def norm(s):return re.sub('[^a-z]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--raw-scores',action='store_true')
    args=ap.parse_args()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=build_model('cuda:0',ROOT/'weights');model.load_state_dict(payload['model']);model.eval()

    def encode(rows):
        out=[]
        with torch.inference_mode():
            for x in DataLoader(Images(rows),batch_size=32,num_workers=4):
                with torch.autocast('cuda',dtype=torch.bfloat16):base=model.encode_base(x.cuda())
                out.append(model.projection(base.float()).cpu())
        return F.normalize(torch.cat(out),dim=-1).numpy()

    gallery=[json.loads(s) for s in GALLERY.read_text().splitlines()]
    features=encode(gallery)
    roster,photo,names=[],{},{}
    for key in sorted({norm(r['name']) for r in gallery}):
        rows=[i for i,r in enumerate(gallery) if norm(r['name'])==key]
        vector=features[rows].mean(0)
        roster.append(key);photo[key]=gallery[rows[0]]['path'];names[key]=gallery[rows[0]]['name']
        features[rows[0]]=vector/np.linalg.norm(vector)
    matrix=np.stack([features[[i for i,r in enumerate(gallery) if norm(r['name'])==k][0]] for k in roster])

    probe=[json.loads(s) for s in PROBE.read_text().splitlines() if json.loads(s).get('domain')=='query']
    crops=encode(probe)
    tracks=defaultdict(list)
    for i,r in enumerate(probe):tracks[norm(r['name'])].append(i)
    keys=sorted(tracks)
    stacked=np.stack([crops[tracks[k]].mean(0)/np.linalg.norm(crops[tracks[k]].mean(0)) for k in keys])
    scores=stacked@matrix.T
    if not args.raw_scores:scores=scores-scores.mean(0)

    rows=[]
    for j,key in enumerate(keys):
        order=np.argsort(-scores[j])
        rank=int(np.flatnonzero(np.asarray(roster)[order]==key)[0])+1
        rows.append({'truth':key,'rank':rank,'margin':float(scores[j][order[0]]-scores[j][order[1]]),
                     'top':[(roster[i],float(scores[j][i])) for i in order[:CANDIDATES]],
                     'views':len(tracks[key]),'crops':[probe[i]['path'] for i in tracks[key]]})
    rows.sort(key=lambda r:r['rank'])

    width=CELL*(CROPS+CANDIDATES+1)+40
    per=11
    REPORT.mkdir(parents=True,exist_ok=True)
    for old in REPORT.glob('predictions_*.jpg'):old.unlink()
    for page in range((len(rows)+per-1)//per):
        block=rows[page*per:(page+1)*per]
        canvas=Image.new('RGB',(width,ROW*len(block)),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(block):
            y=i*ROW;correct=r['rank']==1
            draw.text((6,y+4),f"{names[r['truth']]}  {r['views']} views  truth at rank {r['rank']}"
                              f"  margin {r['margin']:+.3f}",fill=(24,132,60) if correct else (190,38,38))
            picks=np.linspace(0,len(r['crops'])-1,min(CROPS,len(r['crops']))).astype(int)
            for c,index in enumerate(picks):
                try:thumb=ImageOps.contain(read_rgb(r['crops'][index]),(CELL-12,ROW-38))
                except Exception:continue
                canvas.paste(thumb,(6+c*CELL+(CELL-12-thumb.width)//2,y+18))
            draw.line([(CELL*CROPS+14,y+16),(CELL*CROPS+14,y+ROW-6)],fill=(200,200,200),width=2)
            shown=list(r['top'])
            if r['rank']>CANDIDATES:shown.append((r['truth'],dict(r['top']).get(r['truth'],float(0))))
            for c,(key,score) in enumerate(shown):
                x=CELL*CROPS+24+c*CELL
                try:thumb=ImageOps.contain(read_rgb(photo[key]),(CELL-12,ROW-46))
                except Exception:continue
                canvas.paste(thumb,(x+(CELL-12-thumb.width)//2,y+18))
                hit=key==r['truth']
                draw.text((x,y+ROW-26),f"{names[key].split()[-1][:13]}",fill=(24,132,60) if hit else (60,60,60))
                draw.text((x,y+ROW-14),f"{score:+.3f}" if c<CANDIDATES else f"truth, rank {r['rank']}",
                          fill=(24,132,60) if hit else (120,120,120))
        canvas.save(REPORT/f'predictions_{page:02d}.jpg',quality=93)

    summary={'scores':'raw cosine' if args.raw_scores else 'centred by prototype mean over the match',
             'tracks':len(rows),'rank1':sum(1 for r in rows if r['rank']==1),
             'truth_in_top_3':sum(1 for r in rows if r['rank']<=3),
             'median_rank':float(np.median([r['rank'] for r in rows])),
             'sheets':len(list(REPORT.glob('predictions_*.jpg'))),
             'per_track':[{'truth':names[r['truth']],'predicted':names[r['top'][0][0]],
                           'rank_of_truth':r['rank'],'margin':r['margin']} for r in rows]}
    atomic_json(REPORT/'predictions.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='per_track'},indent=2))


if __name__=='__main__':main()
