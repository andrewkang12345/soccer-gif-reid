"""Write a run-specific report after training and the independent recount finish."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from soccer_gif_reid.common import atomic_json
from .metadata import ROOT,DATA

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
 out=ROOT/'reports/soccernet_full_v2';v=json.loads((out/f'verification_{run.name}.json').read_text());b=json.loads((run/'best_metrics.json').read_text());c=json.loads((run/'config.json').read_text());audit=json.loads((DATA/'final_audit.json').read_text())
 assert v['passed'] and json.loads((run/'status.json').read_text())['status']=='complete'
 assert c['ssl_epochs']==0 and v['auxiliary_training_disabled'], 'This report is for the clean paired-data-only run'
 f=json.loads((out/f'number_channel_{run.name}_best.json').read_text());photo=v['metrics']['test']['per_match_image_retrieval']
 lines=[f'# {run.name}: completed soccer ReID run','',f"Verified {datetime.now(timezone.utc).isoformat()}. Selected {b['stage']} epoch {b['epoch']} using validation only.",'',f"Auxiliary pretraining epochs: {c['ssl_epochs']}. Manifest: `{v['manifest_sha256']}`.",'','| Photo retrieval within match | Rank-1 | mAP |','|---|---:|---:|']
 for name,m in [('Frozen CLIP',v['baselines']['frozen_clip']['per_match_image_retrieval']),('NBA projection',v['baselines']['nba_projection']['per_match_image_retrieval']),('Soccer projection',photo)]:lines.append(f"| {name} | {100*m['rank1']:.2f}% | {100*m['mAP']:.2f}% |")
 lines+=['','| Photo + known roster number | Rank-1 | Top-3 | MRR |','|---|---:|---:|---:|']
 for label,key in [('Individual query crop','per_crop'),('Player-by-match pooled observations','per_tracklet')]:
  m=v['metrics']['test'][key];lines.append(f"| {label} | {100*m['rank1']:.2f}% | {100*m['top3']:.2f}% | {100*m['MRR']:.2f}% |")
 lines+=['',f"Photo weights chosen on validation: `{json.dumps(f['alpha_chosen_on_validation'])}`.",'','Dataset:','','| Split | Players | Queries | References | Matches |','|---|---:|---:|---:|---:|']
 for s,d in audit['splits'].items():lines.append(f"| {s} | {d['players']} | {d['queries']} | {d['references']} | {d['games']} |")
 lines+=['','The photo table ranks individual reference photographs. Fusion ranks one prototype per identity, so MRR and image-gallery mAP should be interpreted separately.','', 'Supervised identities, normalized names, matches and hashes are disjoint. This run starts from pretrained CLIP and uses no earlier soccer checkpoint. No added blur/downsampling/JPEG/noise degradation was enabled.','', 'The older v7 auxiliary stage contained evaluation identities, so its scores are historical diagnostics. This test set has been reused during development; these results are not a newly untouched final test. Ground-truth player-by-match pooling assumes perfect grouping and is not McByte++ tracking performance. Candidate rosters include reference-covered labelled players rather than every active squad member.','', 'There is no validated production commitment threshold. The earlier Argentina-France demo failed to transfer; a completed SoccerNet score does not establish production readiness.','',f"Checkpoint: `{run/'best.pt'}`. Detailed audit: `verification_{run.name}.json`. Training history: `{run/'history.jsonl'}`."]
 path=out/f'results_{run.name}.md';path.write_text('\n'.join(lines)+'\n');atomic_json(run/'completion_report.json',{'report':str(path),'verified':True,'manifest_sha256':v['manifest_sha256'],'checkpoint_sha256':v['checkpoint_sha256']});print(path)
if __name__=='__main__':main()
