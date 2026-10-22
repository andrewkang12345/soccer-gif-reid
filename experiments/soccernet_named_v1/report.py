import json
from pathlib import Path
from .metadata import DATA,ROOT
from soccer_gif_reid.common import atomic_json
REPORT=ROOT/'reports/soccernet_named_v1';RUN=ROOT/'runs/soccernet_named_v1'
def main():
 results=json.loads((REPORT/'results.json').read_text());audit=json.loads((DATA/'final_audit.json').read_text());checks={}
 for label,scores in results.items():
  predictions=[json.loads(s) for s in (REPORT/f'{label}_test_rankings.jsonl').read_text().splitlines()];aps=[];correct=0
  for p in predictions:
   ranks=[i+1 for i,r in enumerate(p['ranked_gallery']) if r['pid']==p['gt_pid']];assert ranks
   aps.append(sum(i/rank for i,rank in enumerate(ranks,1))/len(ranks));correct+=ranks[0]==1
  counted={'queries':len(predictions),'correct':correct,'rank1':correct/len(predictions),'mAP':sum(aps)/len(aps)}
  for metric in ('rank1','mAP'):assert abs(counted[metric]-scores['test'][metric])<1e-6
  checks[label]=counted
 atomic_json(REPORT/'independent_metric_check.json',checks)
 names={'frozen_clip':'Frozen CLIP','nba_projection':'NBA-trained projection','soccer_fresh':'Soccer-trained fresh head','soccer_finetune':'NBA head fine-tuned on soccer'}
 lines=['# SoccerNet crop-to-web-photo training results','', 'Training and held-out test scoring are complete. This is a small custom named-player retrieval task, not the standard action-restricted SoccerNet challenge.','', '| Model | Validation Rank-1 / mAP | Test Rank-1 / mAP | Correct test crops |','|---|---:|---:|---:|']
 for label,scores in results.items():
  fmt=lambda s:f"{100*s['rank1']:.2f}% / {100*s['mAP']:.2f}%"
  lines.append(f"| {names[label]} | {fmt(scores['val'])} | {fmt(scores['test'])} | {checks[label]['correct']}/{checks[label]['queries']} |")
 lines+=['', 'Validation uses 16 players / 53 broadcast crops; test uses 24 different players / 134 broadcast crops. Training uses 38 other players / 766 broadcast and replay crops across nine matches. Gallery references: 260 train, 112 validation, 163 test, four to eight screened web photos per player. Real roster IDs, exact names and source matches are disjoint across custom splits.','']
 for label in ('soccer_fresh','soccer_finetune'):
  s=json.loads((RUN/label/'status.json').read_text());lines.append(f"{names[label]}: best epoch {s['best_epoch']}, stopped after {s['epochs_completed']} epochs. Epoch zero was eligible. The selected checkpoints depend only on validation mAP, not test results.")
 lines+=['','Both models keep CLIP ViT-L/14 frozen and train the 790,272-parameter GIF projection head. Existing kit-invariance augmentation is retained, without additional image-quality degradation; sixteen deterministic augmented variants per training crop were cached. References remain clean.','', 'All four test scores were independently recounted from saved full gallery rankings. Metric mAP counts every reference of the correct player as a positive. Neither team nor match restricts test candidate identities.','', 'Limits: small and imbalanced query counts; validation covers only two matches; source-caption and automatically roster-mapped labels may contain residual errors; references vary in count and era. These results cannot establish production accuracy or isolate augmentation/data quantity as the cause. Test images were reviewed for data curation before scoring, but were not used for checkpoint selection. Argentina–France is not scored because several of its players occur in training.','', '[Preparation/training protocol](protocol.md) · [Full metrics](results.json) · [Independent recount](independent_metric_check.json) · [Dataset audit](../../data/soccernet_named_v1/final_audit.json) · [Gallery](../../data/soccernet_named_v1/gallery/)','', 'Inference with either self-contained checkpoint uses the existing `experiments.ap3d_no_quality_v2.infer` module. Supply `--roster data/soccernet_named_v1/gallery/test` (or your own identity-subfolder directory) and `--query /path/to/crop.png`. It averages references per identity and returns cosine scores, not calibrated probabilities; that aggregation differs from the individual-photo benchmark above.']
 (REPORT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
