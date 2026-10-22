"""Verify latest and legacy checkpoint loading and actual crop/pooled inference."""
import json,tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from soccer_gif_reid.common import atomic_json,read_rgb,digest_file
from soccer_gif_reid.augment import preprocess
from experiments.ap3d_no_quality_v2.evaluate import load_model
from .infer import rank
from .numbers import load as load_anchors
from .metadata import DATA,ROOT

@torch.inference_mode()
def main():
 torch.set_num_threads(4);checkpoint=ROOT/'runs/soccernet_full_v7/best.pt'
 rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
 queries=[r for r in rows if r['split']=='test' and r['domain']=='query']
 gallery=[r for r in rows if r['split']=='test' and r['domain']=='reference']
 first=queries[0];pids=sorted({r['pid'] for r in queries if r['game']==first['game']})[:2]
 selected=[r for r in queries if r['pid']==pids[0] and r['game']==first['game']][:2]
 assert len(selected)==2
 refs=[r for r in gallery if r['pid'] in pids]
 names={r['pid']:Path(r['path']).parent.name for r in refs}
 numbers={names[pid]:next(r['number'] for r in queries if r['pid']==pid and r['game']==first['game']) for pid in pids}
 with tempfile.TemporaryDirectory(prefix='soccer_inference_',dir='/mnt/data') as td:
  td=Path(td);roster=td/'roster';track=td/'observations';track.mkdir()
  for r in refs:
   path=roster/names[r['pid']]/Path(r['path']).name;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(r['path'])
  for i,r in enumerate(selected):(track/f'{i:03d}.png').symlink_to(r['path'])
  numberfile=td/'numbers.json';numberfile.write_text(json.dumps(numbers))
  uncached=rank(checkpoint,roster,track,top_k=2,cache=td/'gallery.pt',roster_crop=False,roster_numbers=numberfile,photo_weight=.4)
  cached=rank(checkpoint,roster,track,top_k=2,cache=td/'gallery.pt',roster_crop=False,roster_numbers=numberfile,photo_weight=.4)
  assert cached==uncached
  model,payload=load_model(checkpoint,'cuda:0');assert model.number_adapter is not None
  def encode(rs):
   with torch.autocast('cuda',dtype=torch.bfloat16):return model(torch.stack([preprocess(read_rgb(r['path'])) for r in rs]).cuda()).float().cpu()
  q=F.normalize(encode(selected).mean(0),dim=0);g=encode(refs)
  anchors=model.number_adapter(load_anchors('cuda:0')).float().cpu()
  cached_anchors=torch.load(ROOT/'runs/soccernet_full_v7/number_cache/adapter.pt',map_location='cpu',weights_only=True)['adapted_anchors']
  anchor_error=float((anchors-cached_anchors).abs().max());assert anchor_error<1e-6,anchor_error
  expected={}
  for pid in pids:
   photo=F.normalize(g[[i for i,r in enumerate(refs) if r['pid']==pid]].mean(0),dim=0)
   prototype=F.normalize(.4*photo+.6*anchors[numbers[names[pid]]],dim=0)
   expected[names[pid]]=float(prototype@q)
  score_error=max(abs(r['similarity']-expected[r['identity']]) for r in uncached['ranked_identities'])
  assert score_error<1e-3,score_error
  # The latest weights must reproduce the existing embeddings, not merely load their keys.
  golden=torch.load(ROOT/'runs/soccernet_full_v7/test_features.pt',map_location='cpu',weights_only=True)
  indices=[golden['query_paths'].index(r['path']) for r in selected]
  fresh=encode(selected);agreement=float(F.cosine_similarity(fresh,golden['query'][indices]).min());assert agreement>.9999,agreement
  del model,payload;torch.cuda.empty_cache()
  legacy,_=load_model(ROOT/'runs/soccernet_full_v5/best.pt','cpu');assert legacy.number_adapter is None
  result={'passed':True,'checkpoint_sha256':digest_file(checkpoint),'legacy_checkpoint_loading':True,
          'number_checkpoint_loading':True,'number_anchor_max_difference':anchor_error,'fresh_cached_embedding_min_cosine':agreement,
          'cached_uncached_ranking_equal':True,'independent_fused_score_max_difference':score_error,
          'test_fixture':'Two-player roster and two genuine observations from the frozen test; software correctness check, not an accuracy estimate',
          'ranking':uncached}
 atomic_json(ROOT/'reports/soccernet_full_v2/current_inference_check.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
