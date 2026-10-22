from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml
from pytorch_metric_learning.losses import CircleLoss

from .common import GIF_COMMIT, atomic_json, atomic_write, digest_file, implementation_digest, load_manifest, seed_everything
from .data import IdentityBatchSampler, PairDataset
from .evaluate import FrozenValidation, evaluate
from .model import build_model, from_checkpoint
from .prepare import audit
from .metrics import retrieval_metrics, selection_score

DEFAULTS = {
    "seed": 42, "epochs": 50, "steps_per_epoch": 500, "identities_per_batch": 8,
    "queries_per_identity": 2, "workers": 4, "device": "cuda:0", "lr": 0.001,
    "backbone_lr": 0.000001, "weight_decay": 0.0001, "grad_clip": 10.0,
    "warmup_steps": 100, "eval_batch_size": 32, "eval_seed": 1701,
    "severity": "strong", "unfreeze_blocks": 0, "init_checkpoint": None,
    "clip_cache": "weights", "patience": 10, "log_every": 25,
    "max_clean_map_drop": 0.02, "max_clean_rank1_drop": 0.02,
    "amp": True, "output": "runs/augmented_head", "manifest": "data/mupnit_masked.jsonl",
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a YAML mapping")
    if set(raw) - DEFAULTS.keys():
        raise ValueError(f"Unknown config keys: {set(raw) - DEFAULTS.keys()}")
    cfg = {**DEFAULTS, **raw}
    for key in ("epochs", "steps_per_epoch", "identities_per_batch", "queries_per_identity", "eval_batch_size", "patience", "log_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["workers"] < 0 or cfg["warmup_steps"] < 0 or not 0 <= cfg["unfreeze_blocks"] <= 24:
        raise ValueError("Invalid workers/warmup/unfreeze count")
    if cfg["unfreeze_blocks"] and not cfg["init_checkpoint"]:
        raise ValueError("Fine-tuning requires an augmented-head checkpoint; train frozen CLIP first")
    for key in ("lr", "backbone_lr", "grad_clip"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("weight_decay", "max_clean_map_drop", "max_clean_rank1_drop"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    return cfg


def cross_domain_circle(criterion, query, gallery, labels):
    # Use the repo's CircleLoss, but only broadcast↔reference positive/negative pairs.
    # Cloned labels avoid any shared-object self-match semantics in metric learning.
    ref_labels = labels.clone()
    return 0.5 * (criterion(query, labels, ref_emb=gallery, ref_labels=ref_labels) +
                  criterion(gallery, ref_labels, ref_emb=query, ref_labels=labels))


def checkpoint_payload(model, optimizer, scaler, cfg, epoch, metrics, best, stale, manifest_hash, anchor, dataset_hash):
    return {
        "format_version": 1, "architecture": "GIF-CLIP-ViT-L14-Proj",
        "preprocess": "letterbox_clip_224_v1", "gif_commit": GIF_COMMIT,
        "implementation_sha256": implementation_digest(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "config": cfg, "epoch": epoch, "metrics": metrics,
        "best_score": best, "stale_epochs": stale, "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "finetune_anchor": anchor, "training": "strongly_augmented_query_to_clean_reference",
    }


def can_promote(metrics, best, anchor, cfg):
    if metrics["selection_score"] <= best:
        return False
    if anchor is None:
        return True
    return (metrics["clean"]["mAP"] >= anchor["clean"]["mAP"] - cfg["max_clean_map_drop"] and
            metrics["clean"]["rank1"] >= anchor["clean"]["rank1"] - cfg["max_clean_rank1_drop"] and
            metrics["degraded"]["mAP"] > anchor["degraded"]["mAP"])


def train(cfg, resume=None):
    seed_everything(cfg["seed"])
    torch.set_num_threads(4)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select CPU explicitly for a smoke test")
    manifest = Path(cfg["manifest"]).resolve()
    rows = load_manifest(manifest)
    # Test rows never enter the optimizer, validation, or checkpoint selection.
    stats = audit(rows, require_masks=True)
    if not stats["identities"].get("train") or not stats["identities"].get("val"):
        raise ValueError("Training requires both train and validation splits")
    manifest_hash = digest_file(manifest)
    out = Path(cfg["output"]).resolve()
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError(f"Nonempty run directory: {out}; use --resume or a new output")
    if resume and Path(resume).resolve().parent != out:
        raise ValueError("Resume in the same run directory so best.pt/history remain consistent")
    out.mkdir(parents=True, exist_ok=True)
    data = PairDataset(rows, cfg["seed"], cfg["severity"])
    sampler = IdentityBatchSampler(data, cfg["identities_per_batch"], cfg["queries_per_identity"], cfg["steps_per_epoch"])
    source = resume or cfg["init_checkpoint"]
    loaded = None
    if source:
        model, loaded = from_checkpoint(source, device)
        if loaded["manifest_sha256"] != manifest_hash:
            raise ValueError("Checkpoint/manifest fingerprint mismatch")
        if loaded["dataset_sha256"] != stats["dataset_sha256"]:
            raise ValueError("Dataset image or mask contents changed since the checkpoint")
        if resume:
            if loaded.get("implementation_sha256") != implementation_digest():
                raise ValueError("Pipeline code changed since checkpoint; start a separate initialized run")
            variable = {"device", "workers", "log_every"}
            if any(cfg[k] != loaded["config"].get(k) for k in cfg if k not in variable):
                raise ValueError("Resume config differs from checkpoint; only device/workers/log_every may change")
    else:
        model = build_model(device, cfg["clip_cache"])
    model.set_trainable_blocks(cfg["unfreeze_blocks"])
    groups = [{"params": model.projection.parameters(), "lr": cfg["lr"], "base_lr": cfg["lr"]}]
    if cfg["unfreeze_blocks"]:
        groups.append({"params": [p for p in model.visual.parameters() if p.requires_grad],
                       "lr": cfg["backbone_lr"], "base_lr": cfg["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = CircleLoss(m=0.4, gamma=80)
    best, stale, start, anchor = -1.0, 0, 0, None
    if resume:
        optimizer.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        start, best, stale = loaded["epoch"] + 1, loaded["best_score"], loaded["stale_epochs"]
        anchor = loaded["finetune_anchor"]
        torch.set_rng_state(loaded["torch_rng"])
        if device.type == "cuda" and loaded["cuda_rng"]:
            torch.cuda.set_rng_state_all(loaded["cuda_rng"])
    elif cfg["init_checkpoint"]:
        anchor = evaluate(model, rows, device, batch_size=cfg["eval_batch_size"], workers=cfg["workers"], seed=cfg["eval_seed"])
        best = anchor["selection_score"]
        atomic_json(out / "finetune_anchor.json", anchor)
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "data_audit.json", {**stats, "manifest_sha256": manifest_hash})
    atomic_json(out / "status.json", {"status": "preparing_validation", "completed_epochs": start,
                "manifest_sha256": manifest_hash, "production_validated": False})
    validation = None
    if not cfg["unfreeze_blocks"]:
        print(json.dumps({"event": "cache_fixed_validation_features"}), flush=True)
        validation = FrozenValidation(model, rows, device, cfg["eval_batch_size"], cfg["workers"], cfg["eval_seed"])
        if not resume:
            base_metrics = {name: retrieval_metrics(validation.features[name], validation.features["gallery"],
                [r["pid"] for r in validation.q], [r["pid"] for r in validation.g]) for name in ("clean", "degraded")}
            base_metrics.update(selection_score=selection_score(base_metrics),
                scope="Pretrained frozen CLIP evaluation only; no unaugmented training")
            atomic_json(out / "pretrained_clip_metrics.json", base_metrics)
            atomic_json(out / "initial_projection_metrics.json", validation.evaluate(model))
            print(json.dumps({"event": "pretrained_clip_validation",
                **{name: {k: base_metrics[name][k] for k in ("rank1", "mAP")} for name in ("clean", "degraded")}}), flush=True)
    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    print(json.dumps({"event": "start", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "frozen_clip": not cfg["unfreeze_blocks"], "output": str(out)}), flush=True)
    for epoch in range(start, cfg["epochs"]):
        atomic_json(out / "status.json", {"status": "training", "epoch": epoch + 1,
                    "completed_epochs": epoch, "best_score": best, "production_validated": False})
        data.epoch = epoch
        # New workers each epoch see updated epoch; sample RNG is independent of worker scheduling.
        loader = DataLoader(data, batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        model.train()
        loss_sum, seen, updates = 0.0, 0, 0
        begin = time.monotonic()
        for step, (q, g, labels) in enumerate(loader):
            global_step = epoch * len(loader) + step
            warm = cfg["warmup_steps"]
            scale = (global_step + 1) / max(1, warm) if global_step < warm else 0.5 * (1 + math.cos(math.pi * (global_step - warm) / max(1, total_steps - warm)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            q, g, labels = q.to(device, non_blocking=True), g.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                features = model(torch.cat([q, g], dim=0))
            qf, gf = features.float().chunk(2)
            loss = cross_domain_circle(criterion, qf, gf, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            if not torch.isfinite(grad):
                if not amp:
                    raise FloatingPointError("Nonfinite gradient")
                print(json.dumps({"event": "amp_overflow", "epoch": epoch + 1, "step": step + 1}), flush=True)
            scaler.step(optimizer)
            scaler.update()
            updates += int(bool(torch.isfinite(grad)))
            loss_sum += float(loss.detach())
            seen += 1
            if (step + 1) % cfg["log_every"] == 0 or step == 0:
                progress = {"event": "step", "epoch": epoch + 1, "step": step + 1,
                            "loss": loss_sum / seen, "seconds": time.monotonic() - begin}
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if updates == 0:
            raise FloatingPointError("All optimizer steps overflowed; no trained checkpoint can be promoted")
        metrics = validation.evaluate(model) if validation else evaluate(model, rows, device, batch_size=cfg["eval_batch_size"], workers=cfg["workers"], seed=cfg["eval_seed"])
        improved = can_promote(metrics, best, anchor, cfg)
        if improved:
            best, stale = metrics["selection_score"], 0
        else:
            stale += 1
        payload = checkpoint_payload(model, optimizer, scaler, cfg, epoch, metrics, best, stale, manifest_hash, anchor, stats["dataset_sha256"])
        if improved:
            atomic_write(out / "best.pt", lambda p: torch.save(payload, p))
            atomic_json(out / "best_metrics.json", {"epoch": epoch + 1, **metrics})
        atomic_write(out / "last.pt", lambda p: torch.save(payload, p))
        record = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "optimizer_updates": updates, "promoted": improved,
                  "elapsed_seconds": time.monotonic() - begin, **metrics}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({"event": "validation", **{k: v for k, v in record.items() if k not in {"clean", "degraded"}},
                          "clean": {k: metrics["clean"][k] for k in ("rank1", "mAP")},
                          "degraded": {k: metrics["degraded"][k] for k in ("rank1", "mAP")}}), flush=True)
        del payload
        if stale >= cfg["patience"]:
            break
    atomic_json(out / "status.json", {"status": "complete", "best_score": best,
                "best_checkpoint": str(out / "best.pt") if (out / "best.pt").exists() else cfg["init_checkpoint"],
                "finetune_promoted": (out / "best.pt").exists() if anchor else None,
                "production_validated": False})


def main():
    p = argparse.ArgumentParser(description="Train GIF projection directly on augmented MuPNIT-ReID")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
