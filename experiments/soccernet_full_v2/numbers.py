"""Shirt numbers as text anchors in the model's own embedding space.

The roster already tells us every player's number, so the number does not have to be recognised
in a reference photograph -- it only has to be read off the query crop. CLIP's text tower turns
each number into a vector without any new recogniser: one frozen 768-dimensional anchor per
number, built once here and cached, then placed in the projection space by a single linear map
that trains inside the re-identification model rather than beside it.

Nothing about this is a second model. There is one image encoder, one projection, one embedding
per crop; the anchors are constants and the adapter is 590k parameters of the same network.
"""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from soccer_gif_reid.common import atomic_json,digest_file
from .metadata import ROOT

CACHE=ROOT/'weights/number_text.pt'
MAX_NUMBER=100
TEMPLATES=('a photo of a soccer player wearing number {n}',
 'a football player with the number {n} on his shirt',
 'a cropped photo of a footballer whose jersey number is {n}',
 'number {n}',
 'a soccer player in a jersey numbered {n}')


def build(device='cuda:0'):
 """Encode the prompt ensemble once; the text tower is never needed again."""
 import clip
 model,_=clip.load('ViT-L/14',device=device,jit=False,download_root=str(ROOT/'weights'))
 model.eval()
 rows=[]
 with torch.inference_mode():
  for n in range(MAX_NUMBER):
   tokens=clip.tokenize([t.format(n=n) for t in TEMPLATES]).to(device)
   text=F.normalize(model.encode_text(tokens).float(),dim=-1)
   rows.append(F.normalize(text.mean(0),dim=0))
 anchors=torch.stack(rows).cpu()
 CACHE.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'anchors':anchors,'templates':list(TEMPLATES),'max_number':MAX_NUMBER,
  'backbone':'CLIP ViT-L/14 text tower, frozen'},CACHE)
 print(json.dumps({'anchors':list(anchors.shape),'cache':str(CACHE),
  'sha256':digest_file(CACHE),
  'mean_cosine_between_anchors':float((anchors@anchors.T).masked_fill(
   torch.eye(len(anchors),dtype=torch.bool),0).sum()/(len(anchors)*(len(anchors)-1)))},indent=2))
 return anchors


def load(device='cpu'):
 if not CACHE.exists():build()
 return torch.load(CACHE,map_location=device,weights_only=True)['anchors']


def label(value):
 """Roster number as an anchor index, or -1 when it is missing or out of range."""
 try:n=int(str(value).strip())
 except (TypeError,ValueError):return -1
 return n if 0<=n<MAX_NUMBER else -1


if __name__=='__main__':build(*sys.argv[1:])
