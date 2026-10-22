"""Integration check on a tiny subset of the already-used pilot; not a new score."""
import json,sys,tempfile,shutil,gc
from pathlib import Path
import torch
from . import train
from soccer_gif_reid.common import digest_file,atomic_json

def main():
 root=Path(tempfile.mkdtemp(prefix='soccer_full_resume_',dir='/mnt/data'))
 try:
  source=[json.loads(s) for s in (train.ROOT/'data/soccernet_named_v1/manifest.jsonl').read_text().splitlines()]
  rows=[]
  for split,n in [('train',4),('val',2),('test',2)]:
   pids=sorted({r['pid'] for r in source if r['split']==split})[:n]
   for pid in pids:
    for domain in ['query','reference']:
     rows.extend([r for r in source if r['pid']==pid and r['domain']==domain][:2])
  data=root/'data';data.mkdir();manifest=data/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
  atomic_json(data/'final_audit.json',{'review_passed':True,'manifest_sha256':digest_file(manifest),'purpose':'Previously reviewed pilot-only integration smoke'})
  train.DATA=data
  argv=['train','--steps','1','--head-epochs','2','--finetune-epochs','2','--workers','0','--batch','4']
  original=train.atomic_write
  def interrupt(path,writer):
   original(path,writer)
   if Path(path).name=='last.pt':
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['stage_index']==1 and ck['epoch']==0:raise RuntimeError('simulated interruption after durable checkpoint')
  train.atomic_write=interrupt;sys.argv=argv+['--out',str(root/'resumed')]
  try:train.main()
  except RuntimeError as e:
   if 'simulated interruption' not in str(e):raise
  else:raise AssertionError('Interruption did not occur')
  train.atomic_write=original;gc.collect();torch.cuda.empty_cache()
  sys.argv=argv+['--out',str(root/'resumed'),'--resume'];train.main();gc.collect();torch.cuda.empty_cache()
  sys.argv=argv+['--out',str(root/'continuous')];train.main();gc.collect();torch.cuda.empty_cache()
  a=torch.load(root/'resumed/last.pt',map_location='cpu',weights_only=False)
  b=torch.load(root/'continuous/last.pt',map_location='cpu',weights_only=False)
  deltas={k:float((a['model'][k].float()-b['model'][k].float()).abs().max()) for k in a['model']}
  worst=max(deltas.values());assert worst<1e-6,worst
  assert a['metrics']==b['metrics']
  atomic_json(train.ROOT/'reports/soccernet_full_v2/resume_check.json',{'passed':True,'max_parameter_difference':worst,'data':'Tiny already-used pilot subset; no full-v2 test images evaluated','stages':'2 head epochs + 2 final-block epochs, 1 batch each','interruption':'After first fine-tuning epoch durable checkpoint'})
  print('resume check passed',worst,flush=True)
 finally:shutil.rmtree(root)
if __name__=='__main__':main()
