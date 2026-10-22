"""Does the shirt-number channel transfer to the Argentina-France clip?

On SoccerNet test the fused prototype is the largest gain in the project. On the clip it named
17 of 21 tracks at 23.5% precision, with two roster players absorbing most of them. Two readings
predict that differently. If the clip's crops are simply too small to show a number, the model's
number read should collapse toward chance there, and on SoccerNet it should fall with crop size
too. If the reading holds up and the damage comes from how the demo streams and commits, the
full-clip ranks should look fine. This measures both, from the same checkpoint.
"""
import json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from soccer_gif_reid.common import atomic_json
from soccer_gif_reid.model import build_model
from .demo_argfra import encode,norm,PROBE,GALLERY
from .metadata import DATA,ROOT
from .numbers import load as load_anchors,label as number_label


def load(checkpoint):
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=build_model('cuda:0',ROOT/'weights',numbers='number_adapter.map.weight' in payload['model'])
 model.load_state_dict(payload['model']);return model.eval()


def clip_ranks(q,tracks,keys,gallery_features,gallery,numbers,adapted,alpha,centred):
 """Full-clip rank of the truth for every track, from the mean of all its views."""
 proto=[]
 for key in keys:
  rows=[i for i,r in enumerate(gallery) if norm(r['name'])==key]
  v=F.normalize(gallery_features[rows].mean(0),dim=0)
  if adapted is not None and alpha<1:v=F.normalize(alpha*v+(1-alpha)*adapted[numbers[key]],dim=0)
  proto.append(v)
 M=torch.stack(proto)
 E=torch.stack([F.normalize(q[tracks[k]].mean(0),dim=0) for k in keys])
 S=(E@M.T).numpy()
 if centred:S=S-S.mean(0)
 ranks=[1+int(np.flatnonzero(np.argsort(-row)==j)[0]) for j,row in enumerate(S)]
 winners=Counter(keys[int(np.argmax(row))] for row in S)
 return np.asarray(ranks),winners


def main(checkpoint='runs/soccernet_full_v6/best.pt'):
 checkpoint=ROOT/checkpoint
 model=load(checkpoint)
 anchors=load_anchors('cuda:0')
 with torch.inference_mode():adapted=F.normalize(model.number_adapter(anchors).float(),dim=-1).cpu()
 gallery=[json.loads(s) for s in GALLERY.read_text().splitlines()]
 probe=[json.loads(s) for s in PROBE.read_text().splitlines()]
 probe=[r for r in probe if r.get('domain')=='query' and 'annotation_id' in r]
 q=encode(probe,model);g=encode(gallery,model)
 numbers={}
 for r in gallery:numbers.setdefault(norm(r['name']),number_label(re.search(r'_(\d+)_',r['pid']).group(1)))
 tracks=defaultdict(list)
 for i,r in enumerate(probe):tracks[norm(r['name'])].append(i)
 keys=sorted(k for k in tracks if k in numbers)
 roster_numbers=sorted(set(numbers[k] for k in keys))
 # 1. Can the model read the number off these crops at all?
 scores=q@adapted.T
 pixels=np.array([int(r['width'])*int(r['height']) for r in probe])
 truth=np.array([numbers.get(norm(r['name']),-1) for r in probe])
 read_roster=np.array([roster_numbers[int(torch.argmax(scores[i,roster_numbers]))] for i in range(len(probe))])
 read_any=scores.argmax(1).numpy()
 per_track_read={}
 for k in keys:
  e=F.normalize(q[tracks[k]].mean(0),dim=0)
  per_track_read[k]=roster_numbers[int(torch.argmax(e@adapted[roster_numbers].T))]
 clip={'crops':len(probe),'median_pixels':int(np.median(pixels)),
  'roster_distinct_numbers':len(roster_numbers),
  'read_within_roster_numbers':float(np.mean(read_roster==truth)),
  'chance_within_roster_numbers':1/len(roster_numbers),
  'read_over_100':float(np.mean(read_any==truth)),
  'per_track_read_correct':int(sum(per_track_read[k]==numbers[k] for k in keys)),
  'tracks':len(keys),
  'most_read_numbers':Counter(read_roster.tolist()).most_common(6)}
 # 2. Full-clip ranks, photographs against fused, raw against centred.
 sweeps={}
 for centred in (False,True):
  for alpha in (1.,.8,.6,.4,.2):
   r,w=clip_ranks(q,tracks,keys,g,gallery,numbers,adapted,alpha,centred)
   sweeps[f"alpha {alpha} {'centred' if centred else 'raw'}"]={'rank1':int((r==1).sum()),
    'top3':int((r<=3).sum()),'median_rank':float(np.median(r)),
    'top_attractors':[(n,c) for n,c in w.most_common(3)]}
 # 3. On SoccerNet test, does the number read fall with crop size?
 rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
 test=[r for r in rows if r['split']=='test' and r['domain']=='query']
 cache=ROOT/'runs'/checkpoint.parent.name/'number_cache/test_query.pt'
 tq=F.normalize(torch.load(cache,map_location='cpu',weights_only=True)['features'],dim=-1)
 by_game=defaultdict(set)
 for r in test:
  n=number_label(r.get('number'))
  if n>=0:by_game[r['game']].add(n)
 tp=np.array([int(r['width'])*int(r['height']) for r in test])
 hit=np.array([sorted(by_game[r['game']])[int(torch.argmax(tq[i]@adapted[sorted(by_game[r['game']])].T))]==number_label(r.get('number'))
               for i,r in enumerate(test)])
 edges=[0,2000,4000,8000,16000,64000,10**9]
 by_size=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(tp>=lo)&(tp<hi)
  if m.sum():by_size.append({'pixels':f'{lo}-{hi if hi<10**9 else "max"}','crops':int(m.sum()),
   'read_within_roster':float(hit[m].mean())})
 report={'checkpoint':str(checkpoint.relative_to(ROOT)),'clip_number_reading':clip,
  'clip_full_ranks':sweeps,'soccernet_test_reading_by_crop_pixels':by_size,
  'soccernet_test_median_pixels':int(np.median(tp))}
 atomic_json(ROOT/'reports/soccernet_full_v2/number_transfer.json',report)
 print(json.dumps(report,indent=1,ensure_ascii=False))


if __name__=='__main__':main(*sys.argv[1:])
