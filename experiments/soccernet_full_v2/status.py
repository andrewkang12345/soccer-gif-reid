"""Report preparation artifacts and the latest run compatible with the frozen manifest."""
import argparse
import json
import re
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from .metadata import ROOT,DATA
from soccer_gif_reid.common import atomic_json,digest_file


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--run',type=Path,help='Inspect an explicit run instead of the latest compatible version')
 args=ap.parse_args()
 def rows(name):
  p=DATA/name
  return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
 def read(path):return json.loads(path.read_text()) if path.exists() else {}
 def counts(rs):return {'images':len(rs),'players':len({r['pid'] for r in rs}),'splits':dict(Counter(r.get('split','unassigned') for r in rs))}
 ps=read(DATA/'players.json');audit=read(DATA/'final_audit.json')
 manifest_hash=digest_file(DATA/'manifest.jsonl') if (DATA/'manifest.jsonl').exists() else None
 compatible=[]
 for path in (ROOT/'runs').glob('soccernet_full_v*'):
  match=re.fullmatch(r'soccernet_full_v(\d+)',path.name)
  if match and read(path/'config.json').get('manifest_sha256')==manifest_hash:compatible.append((int(match[1]),path))
 run=args.run if args.run else max(compatible,default=(0,None))[1]
 record=read(run/'status.json') if run else {};best=read(run/'best_metrics.json') if run else {}
 state={'utc':datetime.now(timezone.utc).isoformat(),'candidate_players':len(ps),'named_source_searches':len(list((DATA/'page_search').glob('*.json'))),'queries':counts(rows('masked_queries.jsonl')),'downloaded_reference_candidates':counts(rows('web_downloaded.jsonl')),'screened_reference_snapshot':counts(rows('screened_gallery.jsonl')),'final_manifest_frozen':bool(audit),'manifest_sha256':manifest_hash,'frozen_splits':audit.get('splits'),
  'latest_run':str(run) if run else None,'recorded_training_status':record.get('status','not recorded'),'full_training_started':bool(run and (run/'progress.json').exists()),'full_training_completed':record.get('status')=='complete','selected_stage':best.get('stage'),'selected_epoch':best.get('epoch')}
 if run:
  state['latest_progress']=read(run/'progress.json')
  state['auxiliary_training_epochs']=read(run/'config.json').get('ssl_epochs')
  state['image_gallery_test']={k:record.get('test',{}).get(k) for k in ['rank1','mAP','queries','identities']}
  fused=read(ROOT/'reports/soccernet_full_v2'/f'number_channel_{run.name}_best.json')
  if fused:
   state['fused_test']=fused.get('test')
   state['fused_evaluation_note']='Per-match rosters of reference-covered labelled players; pooled groups use ground-truth player and match, not tracker output.'
 atomic_json(ROOT/'reports/soccernet_full_v2/status.json',state);print(json.dumps(state,indent=2))
if __name__=='__main__':main()
