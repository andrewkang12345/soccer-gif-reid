"""Train small projection heads on soccer features; test only after selection."""
import json,copy,time,hashlib,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from pytorch_metric_learning.losses import CircleLoss
from soccer_gif_reid.model import ProjectionModel
from soccer_gif_reid.train import cross_domain_circle
from soccer_gif_reid.common import atomic_json,digest_file,seed_everything
from soccer_gif_reid.metrics import retrieval_metrics
from experiments.ap3d_no_quality_v2.evaluate import load_model
from experiments.soccernet_web_gallery_v1.evaluate import Images
from .cache import encode
from .metadata import DATA,ROOT
OUT=ROOT/'runs/soccernet_named_v1';REPORT=ROOT/'reports/soccernet_named_v1'
@torch.inference_mode()
def evaluate(head,q,g,qids,gids,rank_path=None):
 if head is not None:
  head.eval();q=F.normalize(head(q),dim=-1);g=F.normalize(head(g),dim=-1)
 metrics=retrieval_metrics(q.cpu(),g.cpu(),qids,gids)
 if rank_path:
  sims=(q@g.T).cpu();pred=[]
  for pid,s in zip(qids,sims):
   order=torch.argsort(s,descending=True,stable=True).tolist();pred.append({'gt_pid':pid,'ranked_gallery':[{'pid':gids[i],'index':i,'similarity':float(s[i])} for i in order]})
  rank_path.write_text(''.join(json.dumps(p)+'\n' for p in pred))
 return metrics

def main():
 seed_everything(42);torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'complete.json').exists():raise FileExistsError('Experiment already completed; use a new directory')
 audit=json.loads((DATA/'final_audit.json').read_text());gallery=[json.loads(s) for s in (DATA/'gallery.jsonl').read_text().splitlines()];qs=[json.loads(s) for s in (DATA/'final_queries.jsonl').read_text().splitlines()]
 assert audit['gallery_manifest_sha256']==digest_file(DATA/'gallery.jsonl')
 assert audit['query_manifest_sha256']==digest_file(DATA/'final_queries.jsonl')
 cache=torch.load(DATA/'query_features.pt',weights_only=True,map_location='cpu');assert cache['manifest_sha256']==digest_file(DATA/'masked_queries.jsonl')
 model,payload=load_model(ROOT/'runs/ap3d_no_quality_v2/clip/best.pt','cuda:0');nba=copy.deepcopy(model.projection).eval()
 # The frozen backbone is identical for every compared model.
 # Images yields T,C,H,W; flatten singleton time for shared encoder.
 class References(Images):
  def __getitem__(self,i):return super().__getitem__(i)[0]
 gf=encode(model,References([dict(r,paths=[r['path']]) for r in gallery],'clip')).cuda()
 qindex={r['sha256']:i for i,r in enumerate(cache['rows'])};qf=cache['clean'][[qindex[r['sha256']] for r in qs]].cuda()
 sets={}
 for split in ('train','val','test'):
  qi=[i for i,r in enumerate(qs) if r['split']==split];gi=[i for i,r in enumerate(gallery) if r['split']==split]
  sets[split]=(qf[qi],gf[gi],[qs[i]['pid'] for i in qi],[gallery[i]['pid'] for i in gi])
 trainids=set(sets['train'][2]);qtrain=defaultdict(list);gtrain=defaultdict(list)
 for i,r in enumerate(cache['train_rows']):
  if r['pid'] in trainids:qtrain[r['pid']].append(i)
 for i,r in enumerate(gallery):
  if r['split']=='train':gtrain[r['pid']].append(i)
 ids=sorted(trainids);assert set(qtrain)==set(gtrain)==trainids
 qa=cache['train_augmented'].cuda();variants=qa.shape[1];criterion=CircleLoss(m=.4,gamma=80)
 atomic_json(OUT/'protocol.json',{'trainable':'GIF projection head, 790272 parameters; CLIP ViT-L/14 frozen','gallery_sha256':digest_file(DATA/'gallery.jsonl'),'query_sha256':digest_file(DATA/'final_queries.jsonl'),'query_cache_sha256':digest_file(DATA/'query_features.pt'),'nba_checkpoint_sha256':digest_file(ROOT/'runs/ap3d_no_quality_v2/clip/best.pt'),'augmentation':'appearance_only_v2, no added quality degradation, sixteen fixed seeded variants per training query; no clean-query bypass','selection':'validation individual-photo mAP; epoch zero is eligible','epochs':30,'steps_per_epoch':200,'identities_per_batch':min(16,len(ids)),'variants':variants,'seed':42,'code_sha256':digest_file(Path(__file__))})
 selected={};validation={}
 for label,lr in [('soccer_fresh',.001),('soccer_finetune',.0001)]:
  seed_everything(42);rng=np.random.default_rng(42);head=ProjectionModel().cuda() if label=='soccer_fresh' else copy.deepcopy(nba)
  out=OUT/label;out.mkdir(exist_ok=True);optim=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.0001)
  initial=evaluate(head,*sets['val']);best=initial['mAP'];best_epoch=0;best_state=copy.deepcopy(head.state_dict());stale=0
  atomic_json(out/'initial_validation.json',initial)
  for epoch in range(1,31):
   head.train();loss_sum=0.;began=time.monotonic()
   for step in range(200):
    sampled=rng.choice(len(ids),size=min(16,len(ids)),replace=False);qsamp=[int(rng.choice(qtrain[ids[x]])) for x in sampled];gsamp=[int(rng.choice(gtrain[ids[x]])) for x in sampled];vsamp=rng.integers(variants,size=len(sampled))
    features=head(torch.cat([qa[qsamp,vsamp],gf[gsamp]]));features=F.normalize(features,dim=-1);a,b=features.chunk(2);labels=torch.tensor(sampled,device='cuda',dtype=torch.long)
    loss=cross_domain_circle(criterion,a,b,labels);assert torch.isfinite(loss)
    optim.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),10.);assert torch.isfinite(grad);optim.step();loss_sum+=float(loss.detach())
   metrics=evaluate(head,*sets['val']);improved=metrics['mAP']>best
   if improved:best=metrics['mAP'];best_epoch=epoch;best_state=copy.deepcopy(head.state_dict());stale=0
   else:stale+=1
   record={'epoch':epoch,'loss':loss_sum/200,'validation':metrics,'promoted':improved,'seconds':time.monotonic()-began}
   with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   atomic_json(out/'status.json',{'status':'training','epoch':epoch,'best_epoch':best_epoch,'best_mAP':best});print(label,epoch,loss_sum/200,metrics['rank1'],metrics['mAP'],flush=True)
   torch.save({'head':best_state,'epoch':best_epoch},out/'best_head.pt')
   if stale>=8:break
  head.load_state_dict(best_state);head.eval();selected[label]=head;validation[label]=evaluate(head,*sets['val'])
  model.projection.load_state_dict(best_state)
  torch.save({'format':'mupnit-native-quality-v2','kind':'clip','model':{k:v.cpu() for k,v in model.state_dict().items()},'config':{'experiment':'soccernet_named_v1','gallery':str(DATA/'gallery.jsonl'),'queries':str(DATA/'final_queries.jsonl')},'epoch':best_epoch-1,'metrics':{'soccer_validation':validation[label]},'gallery_sha256':digest_file(DATA/'gallery.jsonl'),'query_sha256':digest_file(DATA/'final_queries.jsonl')},out/'best.pt')
  atomic_json(out/'status.json',{'status':'complete','epochs_completed':epoch,'best_epoch':best_epoch,'validation':validation[label]})
 # Test scores become visible only after BOTH model selections have finished.
 results={}
 for label,head in [('frozen_clip',None),('nba_projection',nba),*selected.items()]:
  results[label]={split:evaluate(head,*sets[split],rank_path=REPORT/f'{label}_{split}_rankings.jsonl' if split=='test' else None) for split in ('val','test')}
  atomic_json(REPORT/f'{label}_metrics.json',results[label]);print('FINAL',label,{s:{k:m[k] for k in ('rank1','mAP')} for s,m in results[label].items()},flush=True)
 atomic_json(REPORT/'results.json',results);atomic_json(OUT/'complete.json',{'status':'complete','models':list(results)})
if __name__=='__main__':main()
