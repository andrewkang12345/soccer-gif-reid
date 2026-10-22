"""Stream a real broadcast clip, accumulate each track, and name a player only when confident.

Argentina against France, 2022 final: 31 annotated frames outside SoccerNet entirely. Each box
carries the player's name and shirt number, so the annotation supplies both the track identity a
tracker would give us and the ground truth to check against. The demonstration therefore assumes
a perfect tracker and asks only the question this model answers: from appearance alone, which of
this match's twenty-two is this.

Two deliberate choices. Views are combined with a **plain mean**, because the learned quality head
never beat one by a distinguishable margin. And a name is **committed only when the margin over
the runner-up clears a threshold calibrated on our own validation split** — never on this clip —
so the clip cannot flatter the threshold it is scored against. Below the threshold the box stays
blank, which is the honest output and the one deployment wants.

This is a demonstration, not a measurement. Argentina-France identities were forced into the
training split on purpose, so the model has seen some of these players.
"""
import argparse,json,re,subprocess,unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import read_rgb,atomic_json
from soccer_gif_reid.model import build_model
from .metadata import DATA,ROOT
from .numbers import load as load_anchors,label as number_label
from .train import Images

CLIP=Path('/mnt/data/sports_reID/groundTruth_AllTracking_ARG_FRA_183303')
GALLERY=ROOT/'data/arg_fra_expanded_gallery_v1/gallery.jsonl'
PROBE=ROOT/'data/arg_fra_roster_probe_v1/manifest.jsonl'
REPORT=ROOT/'reports/soccernet_full_v2/demo_arg_fra'


def norm(s):return re.sub('[^a-z]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())


DEVICE='cuda:0'


def encode(rows,model,device=None):
    device=device or DEVICE
    out=[]
    with torch.inference_mode():
        for x in DataLoader(Images(rows),batch_size=32,num_workers=4):
            with torch.autocast(device.split(':')[0],dtype=torch.bfloat16,
                                enabled=device.startswith('cuda')):
                base=model.encode_base(x.to(device))
            out.append(model.projection(base.float()).cpu())
    return F.normalize(torch.cat(out),dim=-1)


def calibrate(model,target_precision=.90,centred=True,alpha=None,adapted=None):
    """Find the smallest commit margin reaching a wanted precision on validation — or report that none does.

    An earlier version always returned a number. When no threshold on validation reaches the
    requested precision, a number is worse than useless: it looks like a calibrated gate while
    encoding only where the search happened to stop. It now returns None in that case and the
    caller must decide what to do about a precision the model cannot supply.
    """
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    queries=[r for r in rows if r['split']=='val' and r['domain']=='query']
    references=[r for r in rows if r['split']=='val' and r['domain']=='reference']
    feats=encode(queries+references,model)
    q,g=feats[:len(queries)],feats[len(queries):]
    ids=np.asarray([r['pid'] for r in references])
    # Calibrate on the same kind of prototype the clip will be scored against, or the margin
    # distribution the threshold comes from is not the one it is applied to.
    roster_number={}
    for r in queries:roster_number.setdefault(r['pid'],number_label(r.get('number')))
    groups=defaultdict(list)
    for i,r in enumerate(queries):groups[(r['pid'],r['game'])].append(i)
    by_match=defaultdict(list)
    for key in groups:by_match[key[1]].append(key)
    margins=[]
    for game,keys in by_match.items():
        roster=sorted({k[0] for k in keys})
        proto=[]
        for p in roster:
            v=F.normalize(g[np.flatnonzero(ids==p)].mean(0),dim=0).numpy()
            if adapted is not None and roster_number.get(p,-1)>=0:
                v=alpha*v+(1-alpha)*adapted[roster_number[p]]
                v=v/max(np.linalg.norm(v),1e-9)
            proto.append(v)
        proto=np.stack(proto)
        tracks=np.stack([F.normalize(q[groups[k]].mean(0),dim=0).numpy() for k in keys])
        scores=tracks@proto.T
        if centred:scores=scores-scores.mean(0)
        for row,key in zip(scores,keys):
            order=np.argsort(-row)
            margins.append((float(row[order[0]]-row[order[1]]) if len(order)>1 else 1.,
                            roster[order[0]]==key[0]))
    curve=[]
    for cut in np.arange(0,.40,.005):
        kept=[m for m in margins if m[0]>=cut]
        if len(kept)<10:break
        curve.append({'margin':float(cut),'precision':float(np.mean([m[1] for m in kept])),
                      'coverage':len(kept)/len(margins)})
    reached=[c for c in curve if c['precision']>=target_precision]
    if reached:
        return reached[0]['margin'],curve,None
    best=max(curve,key=lambda c:c['precision']) if curve else None
    reason=(f"no validation margin reaches {100*target_precision:.0f}% precision; the best is "
            f"{100*best['precision']:.1f}% at margin {best['margin']:.3f} covering "
            f"{100*best['coverage']:.0f}% of tracks" if best else 'validation produced no usable curve')
    return None,curve,reason


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--precision',type=float,default=.90)
    ap.add_argument('--fps',type=int,default=4)
    ap.add_argument('--raw-scores',action='store_true',help='skip the hubness correction')
    ap.add_argument('--gallery',type=Path,default=GALLERY,help='reference manifest to name players from')
    ap.add_argument('--tag',default='',help='suffix for the output folder, to keep runs side by side')
    ap.add_argument('--device',default='cuda:0',help='cuda:0, or cpu when the GPU is full')
    ap.add_argument('--alpha',type=float,default=None,
        help='weight on the photograph in each roster prototype; the rest is the shirt number. '
             'Omit to use photographs alone.')
    args=ap.parse_args()
    global DEVICE;DEVICE=args.device
    report=REPORT.with_name(REPORT.name+args.tag) if args.tag else REPORT
    report.mkdir(parents=True,exist_ok=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=build_model(args.device,ROOT/'weights',
                      numbers='number_adapter.map.weight' in payload['model'])
    model.load_state_dict(payload['model']);model.eval()

    adapted=None
    if args.alpha is not None:
        assert model.number_adapter is not None,'this checkpoint has no number channel'
        with torch.inference_mode():
            adapted=model.number_adapter(load_anchors(args.device)).cpu().numpy()
    threshold,curve,unreachable=calibrate(model,args.precision,centred=not args.raw_scores,
                                          alpha=args.alpha,adapted=adapted)
    if unreachable:
        best=max(curve,key=lambda c:c['precision'])
        threshold=best['margin']
        print(f'WARNING: {unreachable}',flush=True)
        print(f'falling back to the best achievable operating point, margin {threshold:.3f}',flush=True)
    else:
        print(f'commit threshold {threshold:.3f} from validation at {100*args.precision:.0f}% target precision',flush=True)

    gallery=[json.loads(s) for s in args.gallery.read_text().splitlines()]
    features=encode(gallery,model)
    # A roster carries a number as well as a photograph. When the model holds the number adapter,
    # each prototype becomes part photograph and part number in the one embedding space, exactly
    # as on the benchmark; without it the prototype is the photographs alone.
    prototypes,names,numbers={},{},{}
    for key in {norm(r['name']) for r in gallery}:
        rows=[i for i,r in enumerate(gallery) if norm(r['name'])==key]
        photo=F.normalize(features[rows].mean(0),dim=0).numpy()
        names[key]=gallery[rows[0]]['name']
        numbers[key]=number_label(re.search(r'_(\d+)_',gallery[rows[0]]['pid']).group(1))
        if adapted is not None and numbers[key]>=0:
            photo=args.alpha*photo+(1-args.alpha)*adapted[numbers[key]]
            photo=photo/max(np.linalg.norm(photo),1e-9)
        prototypes[key]=photo
    roster=sorted(prototypes)
    matrix=np.stack([prototypes[k] for k in roster])

    # The probe manifest also carries a few reference portraits; only the clip crops are tracks.
    probe=[json.loads(s) for s in PROBE.read_text().splitlines()]
    probe=[r for r in probe if r.get('domain')=='query' and 'annotation_id' in r]
    crops=encode(probe,model).numpy()
    coco=json.loads((CLIP/'_annotations.coco.json').read_text())
    frames={i['id']:i['file_name'] for i in coco['images']}
    boxes={str(a['id']):a for a in coco['annotations']}
    order=[i['id'] for i in sorted(coco['images'],key=lambda x:x['file_name'])]
    position={f:k for k,f in enumerate(order)}

    tracks=defaultdict(list)
    for i,r in enumerate(probe):
        a=boxes.get(str(r['annotation_id']))
        if a is None:continue
        tracks[norm(r['name'])].append({'frame':a['image_id'],'bbox':a['bbox'],'index':i,'truth':norm(r['name'])})
    for v in tracks.values():v.sort(key=lambda x:position.get(x['frame'],0))

    log=[];committed=defaultdict(lambda:None)
    for step,frame_id in enumerate(order):
        picture=Image.open(CLIP/frames[frame_id]).convert('RGB');draw=ImageDraw.Draw(picture)
        live={}
        for key,views in tracks.items():
            seen=[v for v in views if position[v['frame']]<=step]
            if not seen or not any(v['frame']==frame_id for v in views):continue
            embedding=crops[[v['index'] for v in seen]].mean(0)
            embedding/=max(np.linalg.norm(embedding),1e-9)
            live[key]=matrix@embedding
        if not live:continue
        # Hubness correction: a few prototypes sit close to everything, and on this clip the spread
        # of prototype means is 0.185 against track margins near 0.04, so the winner is decided by
        # which prototype is generally popular rather than by the match. Subtracting each
        # prototype's mean over the match's own tracks removes that, and needs nothing the tracker
        # does not already have.
        stacked=np.stack(list(live.values()))
        offset=stacked.mean(0) if not args.raw_scores else 0.
        for key,scores in live.items():
            scores=scores-offset
            ranked=np.argsort(-scores)
            margin=float(scores[ranked[0]]-scores[ranked[1]])
            guess=roster[ranked[0]];commit=margin>=threshold
            here=[v for v in tracks[key] if v['frame']==frame_id]
            if commit:committed[key]=guess
            shown=committed[key]
            for v in here:
                x,y,w,h=v['bbox']
                right=shown==key if shown else None
                colour=(150,150,150) if shown is None else ((24,200,90) if right else (235,60,60))
                draw.rectangle([x,y,x+w,y+h],outline=colour,width=2)
                caption=names[shown].split()[-1] if shown else '?'
                draw.text((x,max(0,y-11)),f'{caption} {margin:.02f}',fill=colour)
            log.append({'frame':step,'track':key,'views':len(seen),'margin':margin,
                        'guess':guess,'committed':bool(shown),'correct':bool(shown==key) if shown else None})
        draw.text((8,8),f'frame {step+1}/{len(order)}   commit margin >= {threshold:.3f} '
                        f'(validation, {100*args.precision:.0f}% precision target)',fill=(255,255,255))
        draw.text((8,22),'green named correctly   red named wrongly   grey not confident enough',fill=(255,255,255))
        picture.save(report/f'frame_{step:03d}.png')

    final={k:committed[k] for k in tracks}
    named=[k for k,v in final.items() if v]
    audit={'checkpoint':str(args.checkpoint),'gallery':str(args.gallery),
     'gallery_images':len(gallery),'photo_weight_alpha':args.alpha,
     'roster_numbers':{names[k]:numbers[k] for k in sorted(numbers)},
     'threshold':threshold,'precision_target':args.precision,
     'calibration_curve':curve,'tracks':len(tracks),'roster':len(roster),'frames':len(order),
     'tracks_named':len(named),'tracks_named_correctly':sum(1 for k in named if final[k]==k),
     'coverage':len(named)/len(tracks),
     'precision':(sum(1 for k in named if final[k]==k)/len(named)) if named else None,
     'per_track':[{'truth':names[k],'named':names[final[k]] if final[k] else None,
                   'views':len(tracks[k])} for k in sorted(tracks)],
     'caveat':'Argentina-France identities are in the training split by design; this demonstrates the deployed behaviour and is not a held-out measurement.'}
    atomic_json(report/'summary.json',audit)
    exe=__import__('imageio_ffmpeg').get_ffmpeg_exe()
    subprocess.run([exe,'-y','-loglevel','error','-framerate',str(args.fps),'-i',str(report/'frame_%03d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(report/'arg_fra_demo.mp4')],check=True)
    for f in report.glob('frame_*.png'):f.unlink()
    print(json.dumps({k:v for k,v in audit.items() if k not in ('calibration_curve','per_track')},indent=2))


if __name__=='__main__':main()
