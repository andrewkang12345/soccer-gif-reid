"""Exercise real ViT-L/14, head optimization and offline checkpoint reload.

Uses two local crops solely as implementation fixtures. This is NOT MuPNIT training,
does not estimate identity accuracy, and retains no deployable checkpoint.
"""
from pathlib import Path
import argparse
import gc
import tempfile
import time

import numpy as np
import torch
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.augment import BroadcastAugment, load_masks, preprocess
from soccer_gif_reid.common import atomic_json, load_manifest, read_rgb, seed_everything
from soccer_gif_reid.infer import encode_paths, make_prototypes, rank_tracklet
from soccer_gif_reid.model import build_model, from_checkpoint
from soccer_gif_reid.train import DEFAULTS, checkpoint_payload, cross_domain_circle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/local_preview_v2/masked.jsonl"))
    p.add_argument("--output", type=Path, default=Path("reports/real_clip_smoke.json"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    seed_everything(42)
    torch.set_num_threads(4)
    start = time.monotonic()
    rows = [r for r in load_manifest(args.manifest) if r.get("person_mask")][:2]
    if len(rows) < 2:
        raise ValueError("Need two successfully segmented crop fixtures")
    queries, references = [], []
    for i, row in enumerate(rows):
        im = read_rgb(row["path"])
        queries.append(preprocess(BroadcastAugment()(im, load_masks(row, im.size), np.random.default_rng(i))))
        references.append(preprocess(im))
    model = build_model(args.device, "weights")
    head_before = model.projection.net[0].weight.detach().clone()
    # Spot-check backbone tensor values and all visual gradient flags.
    base_before = model.visual.conv1.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    model.train()
    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.float16,
                        enabled=args.device.startswith("cuda")):
        z = model(torch.stack(queries + references).to(args.device))
    loss = cross_domain_circle(CircleLoss(), *z.float().chunk(2), torch.tensor([0, 1], device=args.device))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 10)
    scaler.step(optimizer)
    scaler.update()
    frozen = all(p.grad is None for p in model.visual.parameters()) and torch.equal(base_before, model.visual.conv1.weight)
    changed = not torch.equal(head_before, model.projection.net[0].weight)
    assert frozen and changed and torch.isfinite(loss)
    model.eval()
    with torch.inference_mode():
        batch = torch.stack(references).to(args.device)
        expected = model(batch).cpu()
    payload = checkpoint_payload(model, optimizer, scaler, DEFAULTS, 0, {}, -1, 0, "SMOKE_ONLY", None, "SMOKE_ONLY")
    payload["training"] = "SMOKE_ONLY_NOT_A_TRAINED_REID_MODEL"
    with tempfile.TemporaryDirectory(prefix="smoke_", dir="runs") as temp:
        ckpt = Path(temp) / "temporary.pt"
        torch.save(payload, ckpt)
        del payload, model, optimizer, z
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        restored, _ = from_checkpoint(ckpt, args.device)
        with torch.inference_mode():
            actual = restored(batch).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        features = encode_paths(restored, [Path(r["path"]) for r in rows], args.device, 2)
        ids, prototypes = make_prototypes(features, ["fixture_1", "fixture_2"])
        ranked = rank_tracklet(features[:1], prototypes, ids)
        assert len(ranked["ranked_identities"]) == 2
        restored.set_trainable_blocks(1)
        restored.train()
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            z = restored(torch.stack(queries + references).to(args.device))
        finetune_loss = cross_domain_circle(CircleLoss(), *z.float().chunk(2), torch.tensor([0, 1], device=args.device))
        finetune_loss.backward()
        final_block_gradient = any(p.grad is not None and p.grad.norm() > 0 for p in restored.visual.transformer.resblocks[-1].parameters())
        early_blocks_frozen = all(p.grad is None for b in list(restored.visual.transformer.resblocks)[:-1] for p in b.parameters())
        assert final_block_gradient and early_blocks_frozen
    atomic_json(args.output, {
        "status": "passed", "architecture": "real OpenAI CLIP ViT-L/14 + official GIF projection head",
        "device": args.device, "head_updated": changed, "frozen_backbone_unchanged": frozen,
        "head_AMP": "float16 CLIP / float32 projection / GradScaler",
        "head_loss_finite": bool(torch.isfinite(loss)), "offline_reload_matches": True,
        "single_final_block_gradient": bool(final_block_gradient), "earlier_blocks_frozen": early_blocks_frozen,
        "inference_ranking_exercised": True, "deployable_checkpoint_retained": False,
        "MuPNIT_Rank1": None, "MuPNIT_mAP": None,
        "scope": "Implementation smoke test on two local crop fixtures; no accuracy claim",
        "seconds": time.monotonic() - start,
    })
    print(args.output.read_text())


if __name__ == "__main__":
    main()
