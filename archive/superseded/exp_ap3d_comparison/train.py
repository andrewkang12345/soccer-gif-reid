"""A matched augmented-data experiment, not a reproduction of GIF Table 4."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.common import atomic_json, atomic_write, digest_file, implementation_digest, load_manifest, seed_everything
from soccer_gif_reid.data import IdentityBatchSampler
from soccer_gif_reid.prepare import audit
from soccer_gif_reid.train import cross_domain_circle
from soccer_gif_reid.model import build_model
from .data import ClipPairs
from .model import AP3DEncoder, encode_pair
from .evaluate import Validation, load_model


def code_hash():
    root = Path(__file__).parent
    parts = [(str(p.relative_to(root)), digest_file(p)) for p in sorted(root.rglob("*.py"))]
    return hashlib.sha256(json.dumps([implementation_digest(), parts]).encode()).hexdigest()


def validate_clips(rows, clips):
    by_path = {r["path"]: r for r in rows}
    seen = set()
    for clip in clips:
        paths, frames = clip["paths"], clip["frames"]
        if len(paths) != 4 or len(set(paths)) != 4 or len(frames) != 4 or sorted(set(frames)) != frames:
            raise ValueError("Each query needs four distinct ordered frames")
        for path, frame in zip(paths, frames):
            row = by_path[path]
            video, actual_frame, local, *_ = Path(path).stem.split("_")
            if (row["domain"] != "query" or row["pid"] != clip["pid"] or row["split"] != clip["split"] or
                video != clip["video"] or local != clip["local_id"] or int(actual_frame) != frame):
                raise ValueError("Mixed identity, split, source video or local track")
            if path in seen:
                raise ValueError("Overlapping query clips")
            seen.add(path)
    if seen != {r["path"] for r in rows if r["domain"] == "query"}:
        raise ValueError("Manifest/query-clip coverage mismatch")


def train(cfg, kind, resume=False, until_epoch=None):
    seed_everything(cfg["seed"]); torch.set_num_threads(4)
    device = cfg["device"]
    rows = load_manifest(cfg["manifest"])
    clips = [json.loads(s) for s in Path(cfg["clips"]).read_text().splitlines()]
    validate_clips(rows, clips)
    stats = audit(rows, require_masks=True)
    out = Path(cfg["output_root"]) / kind
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("Use --resume or a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"manifest_sha256": digest_file(cfg["manifest"]), "clips_sha256": digest_file(cfg["clips"]),
                  "dataset_sha256": stats["dataset_sha256"], "implementation_sha256": code_hash(),
                  "initial_weights_sha256": digest_file(Path(cfg["clip_cache"])/"ViT-L-14.pt" if kind == "clip" else cfg["spatial_weights"])}
    loaded = None
    if resume:
        model, loaded = load_model(out / "last.pt", device)
        if loaded["config"] != cfg or loaded["kind"] != kind or any(loaded[k] != v for k, v in provenance.items()):
            raise ValueError("Resume config/data/code mismatch")
    else:
        model = build_model(device, cfg["clip_cache"]) if kind == "clip" else AP3DEncoder(cfg["spatial_weights"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg[f"{kind}_lr"], weight_decay=cfg["weight_decay"])
    start, best, stale = 0, -1., 0
    if loaded:
        optimizer.load_state_dict(loaded["optimizer"])
        start, best, stale = loaded["epoch"] + 1, loaded["best_score"], loaded["stale_epochs"]
        torch.set_rng_state(loaded["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(loaded["cuda_rng"])
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "data_audit.json", {**stats, **provenance})
    atomic_json(out / "status.json", {"status": "preparing_validation", "kind": kind, "completed_epochs": start})
    print(json.dumps({"event": "start", "kind": kind, "trainable_parameters": sum(p.numel() for p in params),
                      "initialization": "CLIP ViT-L/14, frozen" if kind == "clip" else "ImageNet1K V1 ResNet50; temporal branches newly initialized"}), flush=True)
    validation = Validation(model, kind, rows, clips, device, batch_size=cfg["eval_batch_size"], workers=cfg["workers"], seed=cfg["eval_seed"])
    if not loaded:
        initial = validation.evaluate(model)
        atomic_json(out / "initial_metrics.json", initial)
        if kind == "clip":
            atomic_json(out / "pretrained_clip_metrics.json", validation.evaluate(model, base=True))
        print(json.dumps({"event": "initial_validation", **brief(initial)}), flush=True)
    data = ClipPairs(rows, clips, kind, cfg["seed"])
    sampler = IdentityBatchSampler(data, cfg["identities_per_batch"], 1, cfg["steps_per_epoch"])
    criterion = CircleLoss(m=.4, gamma=80)
    target = min(cfg["epochs"], until_epoch or cfg["epochs"])
    completed = start
    for epoch in range(start, target):
        if stale >= cfg["patience"]:
            break
        data.epoch = epoch
        loader = DataLoader(data, batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=True,
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        model.train(); loss_sum = 0.; began = time.monotonic()
        atomic_json(out / "status.json", {"status": "training", "kind": kind, "epoch": epoch+1, "completed_epochs": epoch})
        for step, (q, g, labels) in enumerate(loader):
            current = epoch*len(loader)+step
            warm = cfg["warmup_steps"]
            factor = (current+1)/max(1,warm) if current < warm else .5*(1+math.cos(math.pi*(current-warm)/max(1,cfg["epochs"]*len(loader)-warm)))
            for group in optimizer.param_groups:
                group["lr"] = cfg[f"{kind}_lr"]*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                qf, gf = encode_pair(model, kind, q.to(device), g.to(device))
            loss = cross_domain_circle(criterion, qf.float(), gf.float(), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite gradient; optimizer not updated")
            optimizer.step(); loss_sum += float(loss.detach())
            if step == 0 or (step+1) % cfg["log_every"] == 0:
                progress = {"event": "step", "kind": kind, "epoch": epoch+1, "step": step+1,
                            "loss": loss_sum/(step+1), "seconds": time.monotonic()-began,
                            "lr": optimizer.param_groups[0]["lr"]}
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        metrics = validation.evaluate(model)
        improved = metrics["selection_score"] > best
        if improved:
            best, stale = metrics["selection_score"], 0
        else:
            stale += 1
        payload = {"format": "mupnit-matched-clips-v1", "kind": kind, "config": cfg, **provenance,
                   "model": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best,
                   "stale_epochs": stale, "metrics": metrics, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        atomic_write(out / "last.pt", lambda p: torch.save(payload, p))
        if improved:
            atomic_write(out / "best.pt", lambda p: torch.save(payload, p))
            atomic_json(out / "best_metrics.json", {"epoch": epoch+1, **metrics})
        record = {"epoch": epoch+1, "train_loss": loss_sum/len(loader), "optimizer_updates": len(loader),
                  "elapsed_seconds": time.monotonic()-began, "promoted": improved, **metrics}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps({"event": "validation", "kind": kind, **{k: record[k] for k in ("epoch", "train_loss", "elapsed_seconds", "promoted")}, **brief(metrics)}), flush=True)
        completed = epoch + 1
        del payload
    done = completed >= cfg["epochs"] or stale >= cfg["patience"]
    atomic_json(out / "status.json", {"status": "complete" if done else "paused_for_other_model", "completed_epochs": completed,
                 "best_score": best, "production_validated": False})


def brief(metrics):
    return {k: {m: metrics[k][m] for m in ("rank1", "mAP")} for k in ("clean", "degraded")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--kind", choices=["clip", "ap3d"], required=True)
    p.add_argument("--resume", action="store_true"); p.add_argument("--until-epoch", type=int)
    args = p.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    train(cfg, args.kind, args.resume, args.until_epoch)


if __name__ == "__main__":
    main()
