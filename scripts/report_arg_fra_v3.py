"""Refresh the detached soccer-validation run summary until queue completion."""
import json,time
from pathlib import Path
from soccer_gif_reid.common import atomic_write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/arg_fra_validation_v3'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
while True:
 queue=read(OUT/'queue_status.json')
 lines=['# Soccer-validation v3 live progress','',f"Queue: {queue.get('status','starting')}",'','Argentina–France is development validation. Best checkpoints are selected by soccer individual-photo mAP. NBA metrics do not select checkpoints.','', '| Model | Status | Epochs complete | Best epoch | Soccer R1 / mAP | NBA clean R1 / mAP |','|---|---|---:|---:|---:|---:|']
 for kind in ('clip','ap3d'):
  run=ROOT/'runs/arg_fra_validation_v3'/kind;s=read(run/'status.json');b=read(run/'best_metrics.json');soc=b.get('soccer',{});nba=b.get('clean',{})
  fmt=lambda m:f"{100*m['rank1']:.2f}% / {100*m['mAP']:.2f}%" if m else 'pending'
  lines.append(f"| {kind} | {s.get('status','queued')} | {s.get('completed_epochs',0)} | {b.get('epoch','—')} | {fmt(soc)} | {fmt(nba)} |")
 if queue.get('error'):lines+=['',queue['error']]
 lines+=['','[Protocol](report.md). Full metrics and histories: `runs/arg_fra_validation_v3/{clip,ap3d}/`.']
 atomic_write(OUT/'progress_report.md',lambda p:p.write_text('\n'.join(lines)+'\n'))
 if queue.get('status') in ('complete','failed'):break
 time.sleep(30)
