import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from soccer_gif_reid.common import load_manifest,atomic_json
from experiments.ap3d_no_quality_v2.evaluate import load_model,Validation
ROOT=Path('/mnt/data/soccer_gif_reid');OUT=ROOT/'reports/diagnosis_audit_v1'
def main():
 torch.set_num_threads(4);result={}
 for kind in ('clip','ap3d'):
  histories={}
  for run in ('ap3d_no_quality_v2','arg_fra_validation_v3'):
   h=[json.loads(s) for s in (ROOT/f'runs/{run}/{kind}/history.jsonl').read_text().splitlines()]
   histories[run]=[{k:r[k] for k in ('epoch','train_loss','clean','augmented','soccer') if k in r} for r in h]
   for r in histories[run]:
    for k in ('clean','augmented','soccer'):
     if k in r:r[k]={m:r[k][m] for m in ('rank1','mAP')}
  pred=[json.loads(s) for s in (ROOT/f'reports/arg_fra_expanded_gallery_v1/{kind}/ranked_images.jsonl').read_text().splitlines()]
  qs={r['query_id']:r for r in map(json.loads,(ROOT/'data/soccernet_web_gallery_v1/arg_fra_auxiliary/four_queries.jsonl').read_text().splitlines())}
  sizes={p['query_id']:float(np.mean([Image.open(x).height for x in qs[p['query_id']]['paths']])) for p in pred};median=float(np.median(list(sizes.values())))
  wrong=[p for p in pred if p['gt_pid']!=p['ranked_gallery'][0]['pid']]
  same=sum(p['gt_pid'].split('_')[0]==p['ranked_gallery'][0]['pid'].split('_')[0] for p in wrong)
  team_correct=[]
  for p in pred:
   g=[x for x in p['ranked_gallery'] if x['pid'].split('_')[0]==p['gt_pid'].split('_')[0]]
   team_correct.append(g[0]['pid']==p['gt_pid'])
  strata={}
  for label,selected in [('smaller',[p for p in pred if sizes[p['query_id']]<=median]),('larger',[p for p in pred if sizes[p['query_id']]>median])]:
   strata[label]={'queries':len(selected),'rank1':sum(p['gt_pid']==p['ranked_gallery'][0]['pid'] for p in selected)/len(selected)}
  checkpoint=ROOT/f'runs/ap3d_no_quality_v2/{kind}/best.pt';model,payload=load_model(checkpoint,'cuda:0');cfg=payload['config'];rows=load_manifest(ROOT/cfg['manifest']);clips=[json.loads(s) for s in (ROOT/cfg['clips']).read_text().splitlines()]
  # Deterministic per-identity subset, independent of recognition scores.
  grouped=defaultdict(list)
  for c in clips:
   if c['split']=='train':grouped[c['pid']].append(c)
  chosen=[c for pid,cs in sorted(grouped.items()) for c in sorted(cs,key=lambda c:c['clip_id'])[:8]]
  metrics=Validation(model,kind,rows,chosen,'cuda:0',split='train').evaluate(model)
  result[kind]={'histories':histories,'train_subset_metrics':metrics,'train_query_selection':'first eight sorted clip IDs per training identity; all training references','selected_train_clip_ids':[c['clip_id'] for c in chosen],'soccer_wrong':len(wrong),'soccer_wrong_same_team':same,'soccer_oracle_team_rank1':float(np.mean(team_correct)),'soccer_height_median':median,'soccer_height_strata':strata}
  atomic_json(OUT/f'{kind}.json',result[kind]);print(kind,'train', {k:{m:metrics[k][m] for m in ('rank1','mAP')} for k in ('clean','augmented')},'same team wrong',same,len(wrong),'oracle',np.mean(team_correct),'size',strata,flush=True)
  del model,payload;torch.cuda.empty_cache()
 atomic_json(OUT/'results.json',result)
if __name__=='__main__':main()
