"""Rank a crop/tracklet against roster/<identity>/*.jpg with either trained model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from soccer_gif_reid.common import atomic_json, image_paths, read_rgb
from .data import transform
from .evaluate import load_model
from .model import encode_clip


@torch.inference_mode()
def rank(checkpoint, roster, query, device="cuda:0", top_k=10, max_frames=64):
    if top_k < 1 or max_frames < 1:
        raise ValueError("top-k and max-frames must be positive")
    torch.set_num_threads(4)
    model, payload = load_model(checkpoint, device); kind = payload["kind"]
    roster = Path(roster).resolve()
    refs = image_paths(roster)
    grouped = {}
    for path in refs:
        rel = path.relative_to(roster)
        identity = rel.parts[0] if len(rel.parts) > 1 else path.stem
        tensor = transform(read_rgb(path), kind)[None,None].to(device)
        if kind == "ap3d":
            tensor = tensor.expand(-1,4,-1,-1,-1)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            feature = encode_clip(model, kind, tensor).float().cpu()[0]
        grouped.setdefault(identity, []).append(feature)
    ids = sorted(grouped)
    gallery = torch.stack([F.normalize(torch.stack(grouped[i]).mean(0), dim=0) for i in ids])
    query = Path(query)
    paths = image_paths(query) if query.is_dir() else [query]
    # Numeric ordering avoids frame_10 sorting before frame_2.
    import re
    paths = sorted(paths, key=lambda p: [(0,int(s)) if s.isdigit() else (1,s.lower()) for s in re.split(r"(\d+)",p.name)])
    paths = [paths[i] for i in np.linspace(0,len(paths)-1,min(len(paths),max_frames),dtype=int)]
    outputs, weights = [], []
    for start in range(0,len(paths),4):
        block = paths[start:start+4]
        frames = [transform(read_rgb(p),kind) for p in block]
        if kind == "ap3d":
            frames += [frames[-1]]*(4-len(frames))
        tensor = torch.stack(frames)[None].to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            outputs.append(encode_clip(model,kind,tensor).float().cpu()[0])
        weights.append(len(block))
    q = F.normalize((torch.stack(outputs)*torch.tensor(weights)[:,None]).sum(0),dim=0)
    scores = gallery @ q
    order = torch.argsort(scores,descending=True,stable=True)[:top_k]
    return {"model":kind,"checkpoint":str(Path(checkpoint).resolve()),"frames_used":len(paths),
            "reference_images":len(refs),"identities":len(ids),
            "score_type":"cosine similarity, not a calibrated probability; reference features averaged per identity",
            "ranked_identities":[{"identity":ids[i],"similarity":float(scores[i])} for i in order.tolist()]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--roster",required=True)
    p.add_argument("--query",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--top-k",type=int,default=10);p.add_argument("--max-frames",type=int,default=64)
    p.add_argument("--output");a=p.parse_args()
    result=rank(a.checkpoint,a.roster,a.query,a.device,a.top_k,a.max_frames)
    if a.output:atomic_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
