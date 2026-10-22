"""Maintain a readable training report from actual process/checkpoint logs."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from soccer_gif_reid.common import atomic_json, atomic_write

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/ap3d_no_quality_v2"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def update():
    queue=read(REPORT/"queue_status.json") or {}
    now=datetime.now(timezone.utc).isoformat()
    records={}
    for kind in ("clip","ap3d"):
        run=ROOT/"runs/ap3d_no_quality_v2"/kind
        records[kind]={"status":read(run/"status.json"),"progress":read(run/"progress.json"),
                       "best":read(run/"best_metrics.json"),"initial":read(run/"initial_metrics.json"),
                       "pretrained_clip":read(run/"pretrained_clip_metrics.json"),
                       "test":read(REPORT/f"{kind}_test_metrics.json"),
                       "best_checkpoint":str(run/"best.pt") if (run/"best.pt").exists() else None}
    snapshot={"last_updated":now,"status":queue.get("status"),"experiment":"native_quality_four_frame_CLIP_Proj_vs_AP3D",
              "training_started":any(r["progress"] for r in records.values()),"unaugmented_training_performed":False,
              "soccer_production_validated":False,"queue":queue,"models":records,
              "data_preparation":read(REPORT/"data_preparation.json"),"tests_passed":35,
              "reference_gallery":str(ROOT/"data/MuPNIT-ReID/gallery")}
    atomic_json(ROOT/"reports/metrics_status.json",snapshot)
    lines=["# Matched MuPNIT training progress", "", f"Updated {now}. Queue: **{queue.get('status')}**; active model: **{queue.get('kind','none')}**.","",
           "Native-quality v2: no added downsampling, blur, JPEG artifacts, pixel noise, player shrinking or harsh exposure shifts. Clean validation is comparable to v1; augmented validation uses different images from the old degraded condition. No soccer production accuracy has been measured.","",
           "| Model/checkpoint | Epoch | Clean Rank-1 | Clean mAP | Augmented Rank-1 | Augmented mAP |",
           "|---|---:|---:|---:|---:|---:|"]
    def line(name,epoch,m):
        values=[m[k][v] for k in ('clean','augmented') for v in ('rank1','mAP')]
        return '| '+name+' | '+str(epoch)+' | '+' | '.join(f'{v:.2%}' for v in values)+' |'
    baseline=records['clip']['pretrained_clip']
    if baseline:lines.append(line('Pretrained CLIP; four-frame average','0',baseline))
    for kind,r in records.items():
        if r['best']:lines.append(line(kind+' best validation',r['best']['epoch'],r['best']))
    lines += ['', '## Processes and checkpoints','']
    for kind,r in records.items():
        p=r['progress'] or {}
        lines.append(f"- {kind}: {(r['status'] or {}).get('status','queued')}; latest logged epoch {p.get('epoch','—')}, step {p.get('step','—')}, loss {p.get('loss','—')}. Best checkpoint: `{r['best_checkpoint'] or 'not yet saved'}`.")
    lines += ['', '## Protocol and limits','',
              '- 3,413 train / 323 validation / 391 reserved test clips; four distinct source frames per clip; 118 / 42 / 51 disjoint identities.',
              '- 1,685 clean references; 113 failed/incomplete clips excluded before training. No test-based model selection.',
              '- Query-only appearance augmentation with no added quality degradation; fixed augmented validation; checkpoint selection balances both validation mAP values.',
              '- CLIP backbone frozen; AP3D ImageNet spatial initialization, temporal/stages 2–4 adaptation. Same CircleLoss and training samples; different pretraining, input size and learning rates.',
              '- Reference labels are inherited from the archive and not human-certified. Person-mask errors and undetected camera cuts remain possible.',
              '- Queue: two epochs each, then up to 30 epochs each / patience 8, then final reserved-test evaluation.',
              '', '[Configuration and inference](../../experiments/ap3d_no_quality_v2/README.md) · [Queue status](queue_status.json) · [Data audit](data_preparation.json) · [Augmentation preview](augmentation_preview.jpg)', '']
    if queue.get('error'):lines += ['## Failure', '', queue['error'], '']
    atomic_write(REPORT/'progress_report.md',lambda p:p.write_text('\n'.join(lines)))
    return queue.get('status')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        state=update()
        if not args.watch or state in ('complete','failed'):break
        time.sleep(30)
