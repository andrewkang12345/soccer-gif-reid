"""Independently recount saved test retrieval and report selected checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from soccer_gif_reid.common import atomic_json
from .metadata import ROOT,DATA

def recount(path):
 x=torch.load(path,map_location='cpu',weights_only=True)
 q=x['query'].numpy();g=x['gallery'].numpy();q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
 ids=np.asarray(x['gallery_ids']);values=[]
 for vector,pid in zip(q,x['query_ids']):
  matches=ids[np.argsort(-(g@vector),kind='stable')]==pid
  ranks=np.flatnonzero(matches);assert len(ranks)
  values.append((int(matches[0]),float(np.mean(np.arange(1,len(ranks)+1)/(ranks+1)))))
 return {'rank1':float(np.mean([v[0] for v in values])),'mAP':float(np.mean([v[1] for v in values])),'queries':len(values)}
def query_games():
 """Map a stored query image path to the match it came from."""
 return {r['path']:r['game'] for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())
         if r['domain']=='query'}


def match_groups(saved):
 """Per-match evaluation: the gallery is the roster of that one match.

 This is how the model is used. It is harder than whole-split retrieval in one way — every
 candidate wears one of two kits, so colour carries almost no information — and easier in
 another, because the roster bounds the answer to about twenty people instead of the whole
 test split.
 """
 games=query_games();groups={}
 for i,path in enumerate(saved['query_paths']):
  groups.setdefault(games[path],[]).append(i)
 return groups


def per_match(path):
 saved=torch.load(path,map_location='cpu',weights_only=True)
 q=saved['query'].numpy();g=saved['gallery'].numpy()
 q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
 gids=np.asarray(saved['gallery_ids']);qids=np.asarray(saved['query_ids'])
 out={};pooled=[]
 for game,rows in sorted(match_groups(saved).items()):
  roster=set(qids[rows]);keep=np.flatnonzero(np.isin(gids,list(roster)))
  values=[]
  for i in rows:
   order=keep[np.argsort(-(g[keep]@q[i]),kind='stable')]
   hits=gids[order]==qids[i];ranks=np.flatnonzero(hits)
   values.append((float(hits[0]),float(np.mean(np.arange(1,len(ranks)+1)/(ranks+1)))))
  pooled+=values
  out[game]={'players':len(roster),'gallery_images':len(keep),'queries':len(rows),
   'rank1':float(np.mean([v[0] for v in values])),'mAP':float(np.mean([v[1] for v in values]))}
 return {'per_match':out,'pooled_rank1':float(np.mean([v[0] for v in pooled])),
  'pooled_mAP':float(np.mean([v[1] for v in pooled])),
  'macro_match_rank1':float(np.mean([m['rank1'] for m in out.values()])),
  'macro_match_mAP':float(np.mean([m['mAP'] for m in out.values()])),'queries':len(pooled)}


def main(run=None):
 out=Path(run) if run else ROOT/'runs/soccernet_full_v2';report=ROOT/'reports/soccernet_full_v2'
 assert json.loads((out/'status.json').read_text())['status']=='complete'
 trained=json.loads((out/'test_metrics.json').read_text());baselines=json.loads((out/'baselines.json').read_text());allmetrics={'soccer_full':trained,**baselines}
 files={'soccer_full':'test_features.pt','frozen_clip':'frozen_clip_test_features.pt','nba_projection':'nba_test_features.pt'}
 checks={}
 for name,f in files.items():
  checks[name]=recount(out/f)
  for key in ['rank1','mAP']:assert abs(checks[name][key]-allmetrics[name][key])<1e-6,(name,key)
 atomic_json(report/'independent_metric_check.json',checks)
 matches={name:per_match(out/f) for name,f in files.items()}
 atomic_json(report/'per_match_metrics.json',matches)
 ids=sorted(trained['per_identity']);rng=np.random.default_rng(91726);intervals={}
 for name,m in baselines.items():
  assert set(ids)==set(m['per_identity'])
  sizes=np.array([trained['per_identity'][pid]['queries'] for pid in ids]);deltas=np.array([trained['per_identity'][pid]['mAP']-m['per_identity'][pid]['mAP'] for pid in ids])
  draws=rng.integers(0,len(ids),size=(5000,len(ids)));samples=(deltas[draws]*sizes[draws]).sum(1)/sizes[draws].sum(1)
  intervals[name]={'mAP_difference':trained['mAP']-m['mAP'],'player_cluster_bootstrap_95_ci':np.quantile(samples,[.025,.975]).tolist(),'caveat':'Player-cluster resampling does not account for match correlation or weak-label errors.'}
 atomic_json(report/'bootstrap_deltas.json',intervals)
 audit=json.loads((DATA/'final_audit.json').read_text());best=json.loads((out/'best_metrics.json').read_text())
 lines=['# Full named SoccerNet v2 result','',f"Selected stage: {best['stage']}, epoch {best['epoch']}; validation Rank-1 {100*best['rank1']:.2f}%, mAP {100*best['mAP']:.2f}%.",'',
  'Per-match retrieval is the deployment setting: the gallery is the roster of the one match the',
  'crop came from, so every candidate wears one of two kits and the answer is bounded to about',
  'twenty people. Whole-split retrieval ranks against every test identity at once.','',
  '| Model | Per-match Rank-1 | Per-match mAP | Whole-split Rank-1 | Whole-split mAP |','|---|---:|---:|---:|---:|']
 for name,m in allmetrics.items():
  mm=matches[name]
  lines.append(f"| {name} | {100*mm['pooled_rank1']:.2f}% | {100*mm['pooled_mAP']:.2f}% | {100*m['rank1']:.2f}% | {100*m['mAP']:.2f}% |")
 lines+=['','Per match, for the selected model:','','| match | players | queries | Rank-1 | mAP |','|---|---:|---:|---:|---:|']
 for game,m in matches['soccer_full']['per_match'].items():
  lines.append(f"| {game.split('/')[-1][:52]} | {m['players']} | {m['queries']} | {100*m['rank1']:.2f}% | {100*m['mAP']:.2f}% |")
 lines+=['',f'Player-cluster bootstrap mAP differences versus baselines: `{json.dumps(intervals,sort_keys=True)}`','', 'The test players and source games are disjoint from training and validation. Checkpoint selection used validation only. Test metrics were independently recounted from saved embeddings.','',f"Dataset: `{json.dumps(audit['splits'],sort_keys=True)}`",'', 'All screened distinct reference photos were retained, with at least three per eligible player and no maximum cap. Labels remain automatically joined roster/source-context supervision with sampled visual review.','', 'No additional image-quality degradation was used. Query augmentation randomizes kit/background and modest geometry/occlusion; references remain clean.','', 'Inference averages reference features per identity; reported Rank-1/mAP above use image-gallery retrieval, so they measure different aggregation rules.','', 'Argentina-France was development data and is not an independent test for this run. The new test has different players and gallery composition from the small pilot, so cross-run raw percentages are not directly comparable.']
 (report/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':
 import sys
 main(*sys.argv[1:])
