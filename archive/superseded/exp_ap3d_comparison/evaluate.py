import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from soccer_gif_reid.common import atomic_json, digest_file, load_manifest
from soccer_gif_reid.metrics import retrieval_metrics, selection_score
from .data import ClipEvaluation
from .model import AP3DEncoder, encode_clip


@torch.inference_mode()
def extract(model, kind, dataset, device, batch_size=8, workers=4, base=False):
    model.eval(); features = []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            if base:
                b, t, c, h, w = x.shape
                f = model.encode_base(x.reshape(b*t, c, h, w)).reshape(b, t, -1)
            else:
                f = encode_clip(model, kind, x)
        features.append(f.float().cpu())
    return torch.cat(features)


class Validation:
    def __init__(self, model, kind, rows, clips, device, split="val", batch_size=8, workers=4, seed=1701):
        self.q = [c for c in clips if c["split"] == split]
        self.g = [r for r in rows if r["split"] == split and r["domain"] == "reference"]
        self.kind, self.device, self.split, self.seed = kind, device, split, seed
        self.batch_size, self.workers = batch_size, workers
        self.datasets = {"gallery": ClipEvaluation(rows, self.g, kind, reference=True),
                         "clean": ClipEvaluation(rows, self.q, kind, seed=seed),
                         "degraded": ClipEvaluation(rows, self.q, kind, degraded=True, seed=seed)}
        self.cache = None
        if kind == "clip":
            self.cache = {k: extract(model, kind, ds, device, batch_size, workers, base=True)
                          for k, ds in self.datasets.items()}

    @torch.inference_mode()
    def evaluate(self, model, base=False):
        model.eval(); features = {}; started = time.monotonic()
        for name, ds in self.datasets.items():
            if self.cache is not None:
                value = self.cache[name]
                b, t, d = value.shape
                if not base:
                    value = torch.cat([model.projection(v.to(self.device)).float().cpu()
                                       for v in value.reshape(b*t, d).split(512)]).reshape(b, t, -1)
                features[name] = F.normalize(F.normalize(value, dim=-1).mean(1), dim=-1)
            else:
                features[name] = extract(model, self.kind, ds, self.device, self.batch_size, self.workers)
        metrics = {k: retrieval_metrics(features[k], features["gallery"],
                      [c["pid"] for c in self.q], [r["pid"] for r in self.g]) for k in ("clean", "degraded")}
        metrics.update(selection_score=selection_score(metrics), split=self.split, degradation_seed=self.seed,
                       evaluation_seconds=time.monotonic()-started,
                       protocol="Four-frame query to individual clean web photographs; all positive references count for AP; fractions in [0,1].")
        return metrics


def load_model(checkpoint, device):
    p = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = p["kind"]
    if p.get("format") != "mupnit-matched-clips-v1":
        raise ValueError("Unsupported checkpoint")
    if kind == "ap3d":
        model = AP3DEncoder()
    else:
        from clip.model import VisionTransformer
        from soccer_gif_reid.model import ReIDModel
        model = ReIDModel(VisionTransformer(224, 14, 1024, 24, 16, 768))
    model.load_state_dict(p["model"], strict=True)
    return model.to(device).eval(), p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    torch.set_num_threads(4)
    model, payload = load_model(args.checkpoint, args.device)
    cfg = payload["config"]
    rows = load_manifest(cfg["manifest"])
    clips = [json.loads(s) for s in Path(cfg["clips"]).read_text().splitlines()]
    from soccer_gif_reid.prepare import audit
    stats = audit(rows, require_masks=True)
    if (digest_file(cfg["manifest"]) != payload["manifest_sha256"] or
        digest_file(cfg["clips"]) != payload["clips_sha256"] or stats["dataset_sha256"] != payload["dataset_sha256"]):
        raise ValueError("Evaluation data changed since training")
    result = Validation(model, payload["kind"], rows, clips, args.device, split=args.split).evaluate(model)
    result.update(checkpoint_sha256=digest_file(args.checkpoint), kind=payload["kind"])
    atomic_json(args.output, result)
    print(json.dumps({k: {m: result[k][m] for m in ("rank1", "mAP")} for k in ("clean", "degraded")}), flush=True)


if __name__ == "__main__":
    main()
