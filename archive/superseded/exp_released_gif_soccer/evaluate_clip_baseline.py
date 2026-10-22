"""Run GIF's published *vanilla CLIP* image-only baseline on real soccer.

This is not a trained GIF CLIP-Proj/AP3D checkpoint and must never be labeled so.
"""
from collections import defaultdict
from pathlib import Path

import clip
import torch
from torch.nn import functional as F

from soccer_gif_reid.common import atomic_json, digest_file, load_manifest, read_rgb, write_manifest
from soccer_gif_reid.metrics import retrieval_metrics

ROOT=Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    manifest=ROOT/'data/arg_fra_roster_probe_v1/manifest.jsonl'
    rows=load_manifest(manifest)
    q=[r for r in rows if r['domain']=='query'];g=[r for r in rows if r['domain']=='reference']
    # Match Phase 1 of upstream pretrained_clipreid_eval_textque.py:
    # official CLIP preprocessing, native pretrained image encoder, cosine scores.
    model,preprocess=clip.load('ViT-L/14',device='cuda:0',jit=False,download_root=str(ROOT/'weights'))
    model.eval()
    def extract(items):
        outputs=[]
        for start in range(0,len(items),16):
            images=torch.stack([preprocess(read_rgb(r['path'])) for r in items[start:start+16]]).cuda()
            outputs.append(F.normalize(model.encode_image(images).float(),dim=-1).cpu())
        return torch.cat(outputs)
    qf,gf=extract(q),extract(g)
    scores=qf@gf.T
    metrics=retrieval_metrics(qf,gf,[r['pid'] for r in q],[r['pid'] for r in g])
    predictions=[]
    for i,row in enumerate(q):
        order=torch.argsort(scores[i],descending=True,stable=True)
        predictions.append(dict(annotation_id=row['annotation_id'],query_path=row['path'],gt_pid=row['pid'],
             ranked=[dict(pid=g[j]['pid'],name=g[j]['name'],similarity=float(scores[i,j])) for j in order.tolist()]))
    # Four successive sampled frames grouped by GT identity; not McByte tracking.
    by_pid=defaultdict(list)
    for i,row in enumerate(q):by_pid[row['pid']].append((row['source_index'],i))
    pooled=[];ids=[]
    for pid,items in sorted(by_pid.items()):
        items.sort();runs=[];run=[]
        for frame,i in items:
            if run and frame!=run[-1][0]+1:runs.append(run);run=[]
            run.append((frame,i))
        if run:runs.append(run)
        for run in runs:
            for start in range(0,len(run)-3,4):
                indices=[i for frame,i in run[start:start+4]]
                pooled.append(F.normalize(qf[indices].mean(0),dim=0));ids.append(pid)
    pooled_metrics=retrieval_metrics(torch.stack(pooled),gf,ids,[r['pid'] for r in g]) if pooled else None
    result=dict(model='GIF vanilla CLIP-ReID baseline: official pretrained OpenAI ViT-L/14',
        gif_trained_reid_checkpoint=False,projection_used=False,text_or_ocr_used=False,
        preprocessing='Official CLIP Resize + CenterCrop, as in GIF baseline script; no added quality degradation.',
        manifest_sha256=digest_file(manifest),weights_sha256=digest_file(ROOT/'weights/ViT-L-14.pt'),
        upstream_script_sha256=digest_file(ROOT/'vendor/GIF/reid/mupnit_clip_reid/pretrained_clipreid_eval_textque.py'),
        image_queries=metrics,oracle_four_frame_pooling=pooled_metrics,
        temporal_protocol='Non-overlapping groups of four adjacent annotation sample indices, grouped by GT identity; leftovers excluded from pooled result only. Not dense-frame AP3D or tracker performance.',
        caveats=['One existing headshot per identity; semantic reference labels not independently certified.',
                 'GT boxes supplied; duplicate-label conflicts removed before inference; other label errors possible.',
                 'Small reused diagnostic clip, not an independent multi-match soccer benchmark.',
                 'Official center-crop preprocessing can remove head/feet from tall bounding boxes.',
                 'This does not measure the authors trained GIF projection or AP3D, whose weights remain unavailable.'])
    out=ROOT/'reports/released_gif_soccer_check'
    atomic_json(out/'vanilla_clip_soccer_metrics.json',result)
    write_manifest(out/'vanilla_clip_predictions.jsonl',predictions)
    print({k:{m:result[k][m] for m in ('rank1','mAP','queries','identities')} for k in ('image_queries','oracle_four_frame_pooling')},flush=True)


if __name__=='__main__':main()
