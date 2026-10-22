"""NBA diagnostics plus fixed soccer validation; only soccer mAP selects weights."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.ap3d_no_quality_v2.evaluate import Validation as NBAValidation, extract, load_model
from experiments.soccernet_web_gallery_v1.evaluate import Images, score, validate_inputs
from soccer_gif_reid.common import digest_file

class Validation(NBAValidation):
    def __init__(self, model, kind, rows, clips, device, soccer_queries, soccer_gallery, **kwargs):
        super().__init__(model,kind,rows,clips,device,**kwargs)
        self.sq=[json.loads(s) for s in Path(soccer_queries).read_text().splitlines()]
        self.sg=[json.loads(s) for s in Path(soccer_gallery).read_text().splitlines()]
        validate_inputs(self.sq,self.sg,8)
        train_hashes={r.get('sha256') or digest_file(r['path']) for r in rows if r['split']=='train'}
        soccer_paths={p for q in self.sq for p in q['paths']} | {g['path'] for g in self.sg}
        if train_hashes & {digest_file(p) for p in soccer_paths}:raise ValueError('Training/soccer validation image overlap')
        self.soccer_content_sha256=__import__('hashlib').sha256(''.join(sorted(digest_file(p) for p in soccer_paths)).encode()).hexdigest()
        self.sds={'query':Images(self.sq,kind),'gallery':Images([dict(r,paths=[r['path']]) for r in self.sg],kind)}
        self.scache=None
        if kind=='clip':
            self.scache={k:extract(model,kind,ds,device,self.batch_size,self.workers,base=True) for k,ds in self.sds.items()}

    @torch.inference_mode()
    def evaluate(self,model,base=False):
        metrics=super().evaluate(model,base=base);model.eval();features={}
        for k,ds in self.sds.items():
            if self.scache is not None:
                value=self.scache[k];b,t,d=value.shape
                if not base:
                    value=torch.cat([model.projection(v.to(self.device)).float().cpu() for v in value.reshape(b*t,d).split(512)]).reshape(b,t,-1)
                    value=F.normalize(value,dim=-1)
                # Frozen baseline uses raw frame-mean; trained head normalizes each frame.
                features[k]=F.normalize(value.mean(1),dim=-1)
            else:
                features[k]=F.normalize(extract(model,self.kind,ds,self.device,self.batch_size,self.workers),dim=-1)
        sm,_=score(features['query'],features['gallery'],self.sq,self.sg)
        metrics.update(nba_selection_score_diagnostic=metrics['selection_score'],soccer=sm['image_gallery'],soccer_roster_prototypes=sm['roster_prototypes'],selection_score=sm['image_gallery']['mAP'],selection_metric='soccer individual-photo mAP',soccer_content_sha256=self.soccer_content_sha256,soccer_role='development validation; not independent test')
        return metrics
