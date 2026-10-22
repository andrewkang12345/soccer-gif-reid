from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import atomic_json, digest_file, load_manifest
from .data import EvalDataset
from .metrics import retrieval_metrics, selection_score
from .model import from_checkpoint
from .prepare import audit


class FrozenValidation:
    """Compute fixed CLIP features once, then re-evaluate the learned head each epoch.

    Caching is allowed only while the visual encoder remains completely frozen.
    Online training augmentations are never cached or reused across epochs.
    """

    def __init__(self, model, rows, device, batch_size=32, workers=4, seed=1701):
        if model.unfrozen_blocks:
            raise ValueError("Cannot cache validation for a trainable backbone")
        self.q = [r for r in rows if r["split"] == "val" and r["domain"] == "query"]
        self.g = [r for r in rows if r["split"] == "val" and r["domain"] == "reference"]
        self.seed, self.device = seed, device
        args = dict(model=model, device=device, batch_size=batch_size, workers=workers, seed=seed, base=True)
        self.features = {"gallery": extract(rows=self.g, **args), "clean": extract(rows=self.q, **args),
                         "degraded": extract(rows=self.q, degraded=True, **args)}

    @torch.inference_mode()
    def evaluate(self, model):
        if model.unfrozen_blocks:
            raise ValueError("Visual encoder unfrozen after cache creation")
        model.eval()
        projected = {}
        for name, values in self.features.items():
            projected[name] = torch.cat([model.projection(batch.to(self.device)).float().cpu()
                                         for batch in values.split(512)])
        out = {name: retrieval_metrics(projected[name], projected["gallery"], [r["pid"] for r in self.q],
                                       [r["pid"] for r in self.g]) for name in ("clean", "degraded")}
        out.update({"selection_score": selection_score(out), "split": "val", "degradation_seed": self.seed,
                    "protocol": "image-gallery retrieval; fractions in [0,1]; fixed heavy queries, same clean references"})
        return out


@torch.inference_mode()
def extract(model, rows, device, batch_size=32, workers=4, degraded=False, seed=1701, base=False):
    loader = DataLoader(EvalDataset(rows, degraded, seed), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=str(device).startswith("cuda"))
    features = []
    model.eval()
    for batch in loader:
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            images = batch.to(device, non_blocking=True)
            out = model.encode_base(images) if base else model(images)
        features.append(out.float().cpu())
    if not features:
        raise ValueError("Empty evaluation domain")
    return torch.cat(features)


def evaluate(model, rows, device, split="val", batch_size=32, workers=4, seed=1701, base=False):
    query = [r for r in rows if r["split"] == split and r["domain"] == "query"]
    gallery = [r for r in rows if r["split"] == split and r["domain"] == "reference"]
    common = dict(model=model, device=device, batch_size=batch_size, workers=workers, seed=seed, base=base)
    g = extract(rows=gallery, **common)
    out = {}
    for name, degraded in [("clean", False), ("degraded", True)]:
        q = extract(rows=query, degraded=degraded, **common)
        out[name] = retrieval_metrics(q, g, [r["pid"] for r in query], [r["pid"] for r in gallery])
    out.update({"selection_score": selection_score(out), "split": split, "degradation_seed": seed,
                "protocol": "image-gallery retrieval; fractions in [0,1]; fixed heavy queries, same clean references"})
    return out


def main():
    p = argparse.ArgumentParser(description="Measure unaugmented and fixed heavily-degraded NBA retrieval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1701)
    args = p.parse_args()
    rows = load_manifest(args.manifest)
    audit(rows, require_masks=True)
    model, _ = from_checkpoint(args.checkpoint, args.device)
    result = evaluate(model, rows, args.device, args.split, args.batch_size, args.workers, args.seed)
    result.update({"checkpoint_sha256": digest_file(args.checkpoint), "manifest_sha256": digest_file(args.manifest)})
    atomic_json(args.output, result)
    print({k: {m: result[k][m] for m in ("rank1", "mAP")} for k in ("clean", "degraded")})


if __name__ == "__main__":
    main()
