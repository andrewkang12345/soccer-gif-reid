"""Cache frozen CLIP features; sixteen reproducible kit-invariant variants/query."""
import json
from pathlib import Path
import torch,numpy as np
from torch.utils.data import Dataset,DataLoader
from experiments.ap3d_no_quality_v2.evaluate import load_model
from experiments.ap3d_no_quality_v2.augment import BroadcastAugment
from soccer_gif_reid.augment import load_masks,preprocess
from soccer_gif_reid.common import read_rgb,digest_file,seed_for
from .metadata import DATA,ROOT
VARIANTS=16
class Queries(Dataset):
 def __init__(self,rows,augmented):
  self.rows=rows;self.augmented=augmented;self.n=VARIANTS if augmented else 1
 def __len__(self):return len(self.rows)*self.n
 def __getitem__(self,i):
  row=self.rows[i//self.n];im=read_rgb(row['path'])
  if self.augmented:
   rng=np.random.default_rng(seed_for(42,row['sha256'],i%self.n));donor=None
   if rng.random()<.5:
    candidates=[r for r in self.rows if r['pid']!=row['pid']];d=candidates[int(rng.integers(len(candidates)))];di=read_rgb(d['path']);donor=(di,load_masks(d,di.size)['person'])
   im=BroadcastAugment('strong')(im,load_masks(row,im.size),rng,donor)
  return preprocess(im)
@torch.inference_mode()
def encode(model,ds):
 out=[]
 for i,x in enumerate(DataLoader(ds,batch_size=64,num_workers=4,pin_memory=True)):
  with torch.autocast('cuda',dtype=torch.bfloat16):out.append(model.encode_base(x.cuda()).cpu())
  if i%20==0:print('encoded',min((i+1)*64,len(ds)),len(ds),flush=True)
 return torch.cat(out)
def main():
 torch.set_num_threads(4);rows=[json.loads(s) for s in (DATA/'masked_queries.jsonl').read_text().splitlines()];model,_=load_model(ROOT/'runs/ap3d_no_quality_v2/clip/best.pt','cuda:0')
 cache={'rows':rows,'clean':encode(model,Queries(rows,False)),'manifest_sha256':digest_file(DATA/'masked_queries.jsonl'),'variants':VARIANTS,'backbone':'Frozen CLIP ViT-L/14 shared with NBA checkpoint'}
 train=[r for r in rows if r['split']=='train'];cache['train_rows']=train;cache['train_augmented']=encode(model,Queries(train,True)).reshape(len(train),VARIANTS,-1)
 torch.save(cache,DATA/'query_features.pt');print('query feature cache complete',flush=True)
if __name__=='__main__':main()
