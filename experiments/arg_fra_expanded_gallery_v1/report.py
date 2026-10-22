"""Independently recount saved rankings and document the fixed-gallery evaluation."""
import json
from pathlib import Path
from datetime import datetime,timezone
from .collect import DATA,REPORT,ROOT

def main():
 results=json.loads((REPORT/'results.json').read_text());verification={}
 for name,r in results.items():
  preds=[json.loads(x) for x in (REPORT/name/'ranked_images.jsonl').read_text().splitlines()]
  aps=[];correct=0
  for p in preds:
   matches=[i+1 for i,x in enumerate(p['ranked_gallery']) if x['pid']==p['gt_pid']]
   assert len(matches)==8 and len(p['ranked_gallery'])==168
   correct+=matches[0]==1;aps.append(sum(j/rank for j,rank in enumerate(matches,1))/8)
  measured={'rank1':correct/len(preds),'mAP':sum(aps)/len(aps),'correct':correct,'queries':len(preds)}
  for k in ('rank1','mAP'):assert abs(measured[k]-r['image_gallery'][k])<1e-6
  ip=[json.loads(x) for x in (REPORT/name/'ranked_identities.jsonl').read_text().splitlines()]
  pc=sum(p['ranked_identities'][0]['pid']==p['gt_pid'] for p in ip)
  assert abs(pc/len(ip)-r['roster_prototypes']['rank1'])<1e-6
  measured['prototype_correct']=pc;verification[name]=measured
 (REPORT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 lines=['# Argentina–France expanded-gallery evaluation','',
 'Used **8 reference photographs per player**, matching the manifest used by our actual training run: 944 references / 118 training identities. This is not a 200-image experiment. The full raw MuPNIT gallery averages 144.32 images per identity (median 173), but those were not all used by our training run.','',
 'The frozen gallery contains **168 photos / 21 players**. Evaluation reused exactly the same **109 four-observation queries / 20 players**, checkpoints and checkpoint-native letterbox preprocessing as the previous Argentina–France diagnostic. No retraining or added image-quality degradation. Query features were reused only after verifying checkpoint and query-manifest hashes.','',
 '| Model | Soccer Rank-1 | Soccer mAP | NBA clean validation Rank-1 | NBA clean validation mAP | Soccer minus NBA (R1 / mAP, pp) |','|---|---:|---:|---:|---:|---:|']
 for name in ('clip','ap3d','base'):
  r=results[name];m=r['image_gallery'];v=r.get('nba_clean_validation');d=r.get('soccer_minus_nba_validation_percentage_points')
  label={'clip':'Our CLIP-Proj','ap3d':'Our AP3D','base':'Frozen CLIP (matched preprocessing)'}[name]
  lines.append(f"| {label} | {100*m['rank1']:.2f}% | {100*m['mAP']:.2f}% | "+(f"{100*v['rank1']:.2f}% | {100*v['mAP']:.2f}% | {d['rank1']:+.2f} / {d['mAP']:+.2f} |" if v else '— | — | — |'))
 lines+=['','These are individual-photo retrieval metrics, with all eight photos of the correct player counted as positives for AP, matching the NBA metric definition. NBA validation has 323 queries, 42 identities and 333 references (41 identities with eight, one with five); the soccer test has fewer distractor identities.','', 'For deployment-style ranking using one mean reference embedding per identity, Rank-1 is '+', '.join(f"{name}: {100*results[name]['roster_prototypes']['rank1']:.2f}% ({verification[name]['prototype_correct']}/109)" for name in ('clip','ap3d','base'))+'. This is a separate aggregation rule, not the NBA validation metric.','',
 'The previous one-portrait gallery gave CLIP-Proj 12.84% Rank-1 and AP3D 11.01%; frozen CLIP gave 5.50%. The new experiment changes both reference count and photo variety, so improvements cannot be attributed to count alone. Previous one-positive mAP was reciprocal rank; it should not be interpreted as the same eight-positive retrieval task.','',
 'Gallery labels come from public image-search source titles/caption context, with retained original links and search records. All candidate sheets were reviewed for composition and duplicate photographs; ambiguous contexts, obvious renderings and duplicate source-photo crops were excluded before new inference. Exact reference hashes are unique and no reference bytes overlap query bytes. Original downloaded reference pixels are preserved. Source labels are not independently identity-certified.','',
 'This remains an exploratory single-match diagnostic using existing ground-truth annotations and sparse four-observation groups, not McByte++ tracks or dense AP3D video clips. The match was inspected previously; labels may contain residual errors. This is not a completed SoccerNet web-gallery benchmark or evidence of production-level accuracy.','',
 '[Gallery manifest](../../data/arg_fra_expanded_gallery_v1/gallery.jsonl) · [Gallery folder](../../data/arg_fra_expanded_gallery_v1/gallery/) · [Contact sheets](gallery_sheets/) · [Gallery audit](../../data/arg_fra_expanded_gallery_v1/gallery_audit.json) · [Full results](results.json) · [Independent metric recount](independent_verification.json)','',
 'Reproduce scoring: `.venv/bin/python -m experiments.arg_fra_expanded_gallery_v1.run`; recount and report: `.venv/bin/python -m experiments.arg_fra_expanded_gallery_v1.report`. Gallery manifest and checkpoint hashes are saved in each model’s metrics.']
 (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
 status={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'references_per_player':8,'reference_images':168,'queries':109,'independent_metric_recount':'passed','results':verification}
 (REPORT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 log=ROOT/'reports/SOCCER_EVALUATION_LOG_2026-09-12.md'
 with log.open('a') as f:
  f.write('\n## Expanded Argentina–France gallery\n\nMatched the actual training manifest with 8 photos/player (168 total), not 200. Fixed 109 queries and original v2 checkpoints; no retraining or added degradation. Gallery curated and frozen before scoring.\n\n')
  for name in ('clip','ap3d','base'):
   m=verification[name];f.write(f"- {name}: Rank-1 {100*m['rank1']:.2f}%, mAP {100*m['mAP']:.2f}%; {m['correct']}/109 correct.\n")
  f.write('\n[Expanded-gallery report](arg_fra_expanded_gallery_v1/report.md). Independent ranked-output recount passed. SoccerNet named mapping remains unresolved; no new SoccerNet score claimed.\n')
 print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
