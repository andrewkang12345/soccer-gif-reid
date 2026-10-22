"""Resumable native-quality soccer ReID: frozen head, then two visual blocks.
Selection uses validation mAP only. Test is evaluated once both stages finish.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.common import read_rgb, seed_everything, seed_for, digest_file, atomic_json, atomic_write
from soccer_gif_reid.augment import load_masks, preprocess
from soccer_gif_reid.model import build_model
from soccer_gif_reid.train import cross_domain_circle
from soccer_gif_reid.metrics import retrieval_metrics
from experiments.ap3d_no_quality_v2.augment import BroadcastAugment
from .metadata import ROOT, DATA


class Pairs(Dataset):
    def __init__(self, q, g, steps=500, batch=16):
        self.q, self.g = defaultdict(list), defaultdict(list)
        for r in q: self.q[r['pid']].append(r)
        for r in g: self.g[r['pid']].append(r)
        self.ids = sorted(self.q)
        assert set(self.ids) == set(self.g) and len(self.ids) > 1
        self.steps, self.batch, self.epoch = steps, min(batch, len(self.ids)), 0

    def __len__(self): return self.steps

    def __getitem__(self, index):
        rng = np.random.default_rng(seed_for(42, self.epoch, index))
        ids = rng.choice(len(self.ids), self.batch, replace=False)
        qs, gs = [], []
        for idx in ids:
            pid = self.ids[idx]
            r = self.q[pid][int(rng.integers(len(self.q[pid])))]
            ref = self.g[pid][int(rng.integers(len(self.g[pid])))]
            im = read_rgb(r['path']); donor = None
            if rng.random() < .5:
                other = self.ids[(idx + int(rng.integers(1, len(self.ids)))) % len(self.ids)]
                d = self.q[other][int(rng.integers(len(self.q[other])))]
                di = read_rgb(d['path']); donor = (di, load_masks(d, di.size)['person'])
            aug = BroadcastAugment('strong')(im, load_masks(r, im.size), rng, donor)
            qs.append(preprocess(aug)); gs.append(preprocess(read_rgb(ref['path'])))
        return torch.stack(qs), torch.stack(gs), torch.tensor(ids, dtype=torch.long)


class Images(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return preprocess(read_rgb(self.rows[i]['path']))


def paired_forward(model, q, g, microbatch=8):
    # Encode in chunks, but run the head jointly so batch normalization sees both domains.
    # Fine-tuning retains the final-block graphs across chunks, not a full CLIP graph.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        base = torch.cat([model.encode_base(x) for x in torch.cat([q, g]).cuda().split(microbatch)])
    with torch.autocast('cuda', enabled=False):
        return F.normalize(model.projection(base.float()), dim=-1).chunk(2)


@torch.inference_mode()
def evaluate(model, q, g, workers=4, base=False, artifact=None):
    model.eval(); features = []
    for rows in [q, g]:
        out = []
        for x in DataLoader(Images(rows), batch_size=16, num_workers=workers, pin_memory=True):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append((model.encode_base(x.cuda()) if base else model(x.cuda())).float().cpu())
        features.append(torch.cat(out))
    result = retrieval_metrics(*features, [r['pid'] for r in q], [r['pid'] for r in g])
    if artifact:
        atomic_write(artifact, lambda p: torch.save({'query': features[0], 'gallery': features[1],
            'query_ids': [r['pid'] for r in q], 'gallery_ids': [r['pid'] for r in g],
            'query_paths': [r['path'] for r in q], 'gallery_paths': [r['path'] for r in g]}, p))
    return result


def payload(model, config, metrics, stage, epoch, manifest_hash):
    return {'format': 'mupnit-native-quality-v2', 'kind': 'clip',
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': dict(config, stage=stage, unfrozen_blocks=model.unfrozen_blocks),
        'epoch': epoch, 'metrics': metrics, 'manifest_sha256': manifest_hash}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'runs/soccernet_full_v2')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--head-epochs', type=int, default=30)
    ap.add_argument('--finetune-epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); out = args.out
    seed_everything(42); torch.set_num_threads(4)
    audit = json.loads((DATA/'final_audit.json').read_text())
    assert audit['review_passed'] is True
    manifest_hash = digest_file(DATA/'manifest.jsonl')
    assert audit['manifest_sha256'] == manifest_hash
    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    sets = {s: {d: [r for r in rows if r['split']==s and r['domain']==d]
                   for d in ['query', 'reference']} for s in ['train', 'val', 'test']}
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        for key in ['pid','name','sha256']:
            assert not {r[key] for d in sets[a].values() for r in d} & {r[key] for d in sets[b].values() for r in d}
        assert not {r['game'] for r in sets[a]['query']} & {r['game'] for r in sets[b]['query']}
    for s in sets:
        assert sets[s]['query'] and sets[s]['reference']
        assert {r['pid'] for r in sets[s]['query']} == {r['pid'] for r in sets[s]['reference']}
    out.mkdir(exist_ok=True, parents=True)
    if any(out.iterdir()) and not args.resume: raise FileExistsError('Existing run: use --resume')
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete': return
    config = {'experiment': 'soccernet_full_v2', 'steps': args.steps, 'batch': args.batch,
        'head_epochs': args.head_epochs, 'finetune_epochs': args.finetune_epochs,
        'seed': 42, 'selection': 'validation image-gallery mAP', 'quality_degradation': False,
        'training_code_sha256': digest_file(Path(__file__)),
        'augmentation_code_sha256': digest_file(ROOT/'experiments/ap3d_no_quality_v2/augment.py'),
        'model_code_sha256': digest_file(ROOT/'src/soccer_gif_reid/model.py')}
    state = torch.load(out/'last.pt', map_location='cpu', weights_only=False) if args.resume and (out/'last.pt').exists() else None
    if state:
        assert state['manifest_sha256'] == manifest_hash
        for key in config: assert state['config'][key] == config[key], key
    atomic_json(out/'config.json', dict(config, manifest_sha256=manifest_hash))
    model = build_model('cuda:0', ROOT/'weights')
    pairs = Pairs(sets['train']['query'], sets['train']['reference'], args.steps, args.batch)
    criterion = CircleLoss(m=.4, gamma=80)
    best = state['best'] if state else -1.
    stages = [('head',0,args.head_epochs,.001,0.), ('finetune',2,args.finetune_epochs,.0001,.000001)]
    for stage_index, (stage, blocks, epochs, head_lr, backbone_lr) in enumerate(stages):
        if state and stage_index < state['stage_index']: continue
        model.set_trainable_blocks(blocks)
        groups = [{'params': list(model.projection.parameters()), 'lr': head_lr}]
        if blocks: groups.append({'params': [p for p in model.visual.parameters() if p.requires_grad], 'lr': backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
        start, stale = 0, 0
        if state and stage_index == state['stage_index']:
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start, stale = state['epoch']+1, state['stale']
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            if stage_index: model.load_state_dict(torch.load(out/'best.pt', map_location='cpu', weights_only=False)['model'])
            baseline = evaluate(model, **{'q':sets['val']['query'], 'g':sets['val']['reference']}, workers=args.workers)
            atomic_json(out/(stage+'_initial_validation.json'), baseline)
            if baseline['mAP'] > best:
                best = baseline['mAP']
                atomic_write(out/'best.pt', lambda p: torch.save(payload(model,config,baseline,stage,-1,manifest_hash),p))
                atomic_json(out/'best_metrics.json', {'stage':stage,'epoch':0,**baseline})
        for epoch in range(start, epochs):
            if stale >= 8: break
            pairs.epoch = epoch + stage_index*1000
            model.train(); loss_sum = 0.; began = time.monotonic()
            for step, (q,g,labels) in enumerate(DataLoader(pairs,batch_size=None,num_workers=args.workers,pin_memory=True)):
                fraction = (epoch + (step+1)/len(pairs)) / epochs
                warmup = min(1., (epoch*len(pairs)+step+1)/(2*len(pairs)))
                factor = warmup*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
                for group, lr in zip(optimizer.param_groups,[head_lr,backbone_lr]): group['lr']=lr*factor
                optimizer.zero_grad(set_to_none=True)
                qf,gf = paired_forward(model,q,g)
                loss = cross_domain_circle(criterion,qf,gf,labels.cuda())
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.)
                assert torch.isfinite(grad)
                optimizer.step(); loss_sum += float(loss.detach())
                if (step+1)%25==0:
                    atomic_json(out/'progress.json', {'stage':stage,'epoch':epoch+1,'step':step+1,
                        'steps':len(pairs),'loss':loss_sum/(step+1),'seconds':time.monotonic()-began})
            metrics = evaluate(model,sets['val']['query'],sets['val']['reference'],workers=args.workers)
            improved = metrics['mAP'] > best
            if improved: best=metrics['mAP']; stale=0
            else: stale+=1
            ckpt = payload(model,config,metrics,stage,epoch,manifest_hash)
            if improved:
                atomic_write(out/'best.pt',lambda p:torch.save(ckpt,p))
                atomic_json(out/'best_metrics.json',{'stage':stage,'epoch':epoch+1,**metrics})
            ckpt.update(optimizer=optimizer.state_dict(),stage_index=stage_index,best=best,stale=stale,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            atomic_write(out/'last.pt',lambda p:torch.save(ckpt,p))
            record={'stage':stage,'epoch':epoch+1,'loss':loss_sum/len(pairs),'validation':metrics,
                'promoted':improved,'seconds':time.monotonic()-began}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(stage,epoch+1,metrics['rank1'],metrics['mAP'],flush=True)
        state = None
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'])
    test=evaluate(model,sets['test']['query'],sets['test']['reference'],workers=args.workers,artifact=out/'test_features.pt')
    atomic_json(out/'test_metrics.json',test)
    # Baselines on the identical held-out test; these never affect checkpoint selection.
    baselines = {}
    fresh=build_model('cuda:0',ROOT/'weights')
    baselines['frozen_clip']=evaluate(fresh,sets['test']['query'],sets['test']['reference'],workers=args.workers,base=True,artifact=out/'frozen_clip_test_features.pt')
    source=ROOT/'runs/ap3d_no_quality_v2/clip/best.pt'
    nba=torch.load(source,map_location='cpu',weights_only=False)
    fresh.load_state_dict(nba['model'])
    baselines['nba_projection']=evaluate(fresh,sets['test']['query'],sets['test']['reference'],workers=args.workers,artifact=out/'nba_test_features.pt')
    atomic_json(out/'baselines.json',baselines)
    atomic_json(out/'status.json',{'status':'complete','test':test,'manifest_sha256':manifest_hash})

if __name__=='__main__': main()
