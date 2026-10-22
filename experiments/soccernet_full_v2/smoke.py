"""Exercise the actual two-stage gradient path on pilot training rows only."""
import json,time
import torch
from pytorch_metric_learning.losses import CircleLoss
from .train import Pairs, paired_forward, ROOT
from soccer_gif_reid.model import build_model
from soccer_gif_reid.train import cross_domain_circle
from soccer_gif_reid.common import atomic_json

def main():
 torch.set_num_threads(4);torch.manual_seed(42)
 rows=[json.loads(s) for s in (ROOT/'data/soccernet_named_v1/manifest.jsonl').read_text().splitlines()]
 ds=Pairs([r for r in rows if r['split']=='train' and r['domain']=='query'],[r for r in rows if r['split']=='train' and r['domain']=='reference'],steps=1,batch=16)
 q,g,labels=ds[0];model=build_model('cuda:0',ROOT/'weights');report={}
 for blocks in [0,2]:
  model.set_trainable_blocks(blocks);model.train();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6)
  optimizer.zero_grad(set_to_none=True);a,b=paired_forward(model,q,g)
  loss=cross_domain_circle(CircleLoss(m=.4,gamma=80),a,b,labels.cuda());loss.backward()
  frozen=[p.grad for p in model.visual.transformer.resblocks[0].parameters()];assert all(x is None for x in frozen)
  last=list(model.visual.transformer.resblocks[-1].parameters());assert all((p.grad is not None)==bool(blocks) for p in last)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  optimizer.step();torch.cuda.synchronize()
  report[str(blocks)]={'loss':float(loss.detach()),'seconds':time.monotonic()-began,'peak_gb':torch.cuda.max_memory_allocated()/1e9,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
 atomic_json(ROOT/'reports/soccernet_full_v2/training_smoke.json',report);print(report)
if __name__=='__main__':main()
