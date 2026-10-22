"""Does weighting a player's views beat averaging them, on the labelled evaluation?

The quality head is trained on GSR, which has temporal tracklets but only nine games and no
names. The question that matters is whether it transfers: when several crops of one player are
available in *our* test matches, does a quality-weighted combination retrieve better than a plain
mean, and better than a single crop.

Two groupings, because they answer different questions. Per action is several cameras on one
instant — what SoccerNet-v3 actually provides. Per match accumulates everything seen of a player
across the match, which is what a tracker hands you in deployment and is the more generous case.

Every variant is scored per match against that match's roster, and differences are reported with
a paired interval clustered by player, because the same players recur across variants.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from soccer_gif_reid.common import atomic_json
from soccer_gif_reid.model import build_model
from .metadata import DATA, ROOT
from .quality_head import QualityHead
from .train import Images


def embed(rows,checkpoint,head_checkpoint,device='cuda:0'):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(device,ROOT/'weights');model.load_state_dict(payload['model']);model.eval()
    head=QualityHead().to(device)
    saved=torch.load(head_checkpoint,map_location='cpu',weights_only=False)
    head.load_state_dict(saved['head']);head.eval()
    if 'projection' in saved:model.projection.load_state_dict(saved['projection'])
    out=[]
    with torch.inference_mode():
        for x in DataLoader(Images(rows),batch_size=32,num_workers=4,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                base=model.encode_base(x.to(device))
            out.append(model.projection(base.float()).cpu())
    return torch.cat(out),head,saved


def variants(features,groups,head,sizes):
    """One embedding per group per rule, including a hand-written size weighting.

    The learned weights correlate 0.52 with log crop area and only -0.11 with aspect ratio, so the
    head looks like a resolution detector. If weighting by crop size alone matches it, the hundred
    thousand parameters are buying nothing a one-line heuristic does not already give.
    """
    single,mean,weighted,by_size,spreads={},{},{},{},[]
    for key,rows in groups.items():
        block=features[rows]
        normalized=F.normalize(block,dim=-1)
        single[key]=normalized[0]
        mean[key]=F.normalize(normalized.mean(0),dim=-1)
        area=torch.tensor([float(sizes[i]) for i in rows]).log()
        share=torch.softmax((area-area.mean())/max(area.std().item(),1e-6),dim=0)
        by_size[key]=F.normalize((share.unsqueeze(-1)*normalized).sum(0),dim=-1)
        with torch.inference_mode():
            z,w=head(block.unsqueeze(0).cuda())
        weighted[key]=z[0].cpu()
        if len(rows)>1:spreads.append(float((w.max()/w.mean()).cpu()))
    return {'single crop':single,'plain mean':mean,'size weighted':by_size,'quality weighted':weighted},spreads


def score(embeddings,gallery,gallery_ids,keys,truth,match):
    """Per-match retrieval: rank each group against the roster of its own match."""
    by_match=defaultdict(list)
    for k in keys:by_match[match[k]].append(k)
    hits,aps,rows=[],[],[]
    for game,ks in by_match.items():
        roster=sorted({truth[k] for k in ks})
        keep=np.flatnonzero(np.isin(gallery_ids,roster))
        sub=gallery[keep];ids=gallery_ids[keep]
        for k in ks:
            sims=(sub@embeddings[k].numpy())
            order=np.argsort(-sims,kind='stable')
            found=ids[order]==truth[k];ranks=np.flatnonzero(found)
            hits.append(float(found[0]));aps.append(float(np.mean(np.arange(1,len(ranks)+1)/(ranks+1))))
            rows.append({'player':truth[k],'hit':hits[-1],'ap':aps[-1]})
    return float(np.mean(hits)),float(np.mean(aps)),rows


def paired(a,b,draws=5000,seed=4):
    groups=defaultdict(list)
    for i,r in enumerate(a):groups[r['player']].append(i)
    keys=sorted(groups);rng=np.random.default_rng(seed);out=[]
    for _ in range(draws):
        picked=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        out.append(np.mean([a[i]['hit'] for i in picked])-np.mean([b[i]['hit'] for i in picked]))
    return float(np.mean(out)),float(np.quantile(out,.025)),float(np.quantile(out,.975))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'runs/soccernet_full_v5/best.pt')
    ap.add_argument('--head',type=Path,default=ROOT/'runs/quality_head_v1/best.pt')
    ap.add_argument('--split',default='test')
    ap.add_argument('--out',type=Path,default=ROOT/'reports/soccernet_full_v2/tracklet_evaluation.json')
    args=ap.parse_args()
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    queries=[r for r in rows if r['split']==args.split and r['domain']=='query']
    references=[r for r in rows if r['split']==args.split and r['domain']=='reference']
    features,head,saved=embed(queries+references,args.checkpoint,args.head)
    gallery=F.normalize(features[len(queries):],dim=-1).numpy()
    gallery_ids=np.asarray([r['pid'] for r in references])
    query_features=features[:len(queries)]
    sizes=[int(r['width'])*int(r['height']) for r in queries]
    result={'checkpoint':str(args.checkpoint),'head':str(args.head),'split':args.split,
            'head_epoch':saved.get('epoch'),'head_metrics':saved.get('metrics'),'groupings':{}}
    for name,key in (('per action',lambda r:(r['pid'],r['game'],r['action'])),('per match',lambda r:(r['pid'],r['game']))):
        groups=defaultdict(list)
        for i,r in enumerate(queries):groups[key(r)].append(i)
        truth={k:k[0] for k in groups};match={k:k[1] for k in groups}
        made,spreads=variants(query_features,groups,head,sizes)
        entry={'groups':len(groups),'multi_view_groups':sum(1 for v in groups.values() if len(v)>1),
               'mean_group_size':float(np.mean([len(v) for v in groups.values()])),
               'weight_max_over_mean':float(np.mean(spreads)) if spreads else None,'rules':{}}
        scored={}
        for rule,emb in made.items():
            rank1,mAP,detail=score(emb,gallery,gallery_ids,list(groups),truth,match)
            scored[rule]=detail;entry['rules'][rule]={'rank1':rank1,'mAP':mAP}
        for rule in ('plain mean','size weighted','quality weighted'):
            d,lo,hi=paired(scored[rule],scored['single crop'])
            entry['rules'][rule]['vs_single_crop']={'difference':d,'ci95':[lo,hi],'distinguishable':bool(lo>0 or hi<0)}
        for rule,against in (('quality weighted','plain mean'),('size weighted','plain mean'),
                             ('quality weighted','size weighted')):
            d,lo,hi=paired(scored[rule],scored[against])
            entry[f'{rule} vs {against}']={'difference':d,'ci95':[lo,hi],'distinguishable':bool(lo>0 or hi<0)}
        result['groupings'][name]=entry
        print(f"{name}: {entry['groups']} groups, {entry['multi_view_groups']} with more than one view, "
              f"mean size {entry['mean_group_size']:.2f}")
        for rule,m in entry['rules'].items():
            extra=f"  vs single crop {100*m['vs_single_crop']['difference']:+.2f} [{100*m['vs_single_crop']['ci95'][0]:+.2f}, {100*m['vs_single_crop']['ci95'][1]:+.2f}]" if 'vs_single_crop' in m else ''
            print(f"   {rule:17s} Rank-1 {100*m['rank1']:6.2f}%  mAP {100*m['mAP']:6.2f}%{extra}")
        for pair in ('quality weighted vs plain mean','size weighted vs plain mean',
                     'quality weighted vs size weighted'):
            q=entry[pair]
            print(f"   {pair}: {100*q['difference']:+.2f} pts [{100*q['ci95'][0]:+.2f}, {100*q['ci95'][1]:+.2f}] -> "
                  f"{'distinguishable' if q['distinguishable'] else 'NOT distinguishable'}")
    atomic_json(args.out,result)


if __name__=='__main__':main()
