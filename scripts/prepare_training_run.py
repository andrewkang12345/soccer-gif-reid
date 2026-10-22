"""Deterministic first-run MuPNIT subset, reference screening, and real masks.

Every exclusion is recorded before model training. Frozen tests are reserved.
No trained ReID prediction or player-name guess is used to choose data.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from soccer_gif_reid.common import atomic_json, digest_file, read_rgb, seed_for, write_manifest
from soccer_gif_reid.prepare import audit
from soccer_gif_reid.masks import select_person, generate_person_mask, MASK_VERSION
from soccer_gif_reid.augment import BroadcastAugment, load_masks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/augmented_head_v1"
REPORTS = ROOT / "reports/augmented_head_v1"
DATA = ROOT / "data/MuPNIT-ReID"
SEGMENTER = Path("/mnt/data/sports_reID/yolo11m-seg.pt")
QUERY_CAP = {"train": 128, "val": 32, "test": 32}
REF_CAP = 8


def log(event, **fields):
    print(json.dumps({"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields}), flush=True)
    atomic_json(REPORTS / "preparation_progress.json", {"event": event, **fields})


def image_info(path):
    try:
        im = read_rgb(path)
        w, h = im.size
        thumbnail = im.copy(); thumbnail.thumbnail((256, 256))
        gray = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2GRAY)
        return {"path": str(path), "pid": str(int(path.parent.name.split("_")[-1])),
                "sha256": digest_file(path), "width": w, "height": h,
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var())}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "error": str(exc)}


def sample_queries(folder, cap):
    # Round-robin over sequences; choose distinct video/frame pairs and spread
    # each sequence's selected samples over its complete temporal extent.
    sequences = defaultdict(dict)
    for p in sorted(folder.glob("*.jpg")):
        fields = p.stem.split("_")
        if len(fields) < 3 or not fields[1].isdigit():
            raise ValueError(f"Unrecognized query filename: {p}")
        video, frame = fields[0], int(fields[1])
        sequences[video].setdefault(frame, p)
    ordered = sorted(sequences, key=lambda s: seed_for(42, folder.name, s))
    per_video = max(1, int(np.ceil(cap / len(ordered))))
    pools = {}
    for video in ordered:
        frames = sorted(sequences[video])
        order = np.linspace(0, len(frames)-1, min(len(frames), per_video), dtype=int)
        pools[video] = [sequences[video][frames[i]] for i in order]
    selected = []
    while len(selected) < cap and any(pools.values()):
        for video in ordered:
            if pools[video] and len(selected) < cap:
                selected.append(pools[video].pop(0))
    return selected


def reference_screen(model, candidates, split_for):
    selected, diagnostics = [], []
    by_id = defaultdict(list)
    for r in candidates:
        by_id[r["pid"]].append(r)
    for number, pid in enumerate(sorted(by_id, key=int), 1):
        # Source size/shape/blur only; never retrieval scores or test outcomes.
        rows = sorted(by_id[pid], key=lambda r: (-min(r["width"]*r["height"], 1_000_000),
                      -min(r["sharpness"], 300), r["path"]))[:48]
        kept = []
        for start in range(0, len(rows), 8):
            if len(kept) >= REF_CAP:
                break
            chunk = rows[start:start+8]
            imgs = [np.asarray(read_rgb(r["path"]))[..., ::-1] for r in chunk]
            results = model.predict(imgs, classes=[0], device="cuda:0", imgsz=640, conf=.25, verbose=False)
            for r, result in zip(chunk, results):
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                reason, main = "no_person", None
                if len(boxes):
                    areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
                    i = int(areas.argmax()); box = boxes[i]; fraction = float(areas[i]/(r["width"]*r["height"]))
                    significant = sum(a >= .20*areas[i] for a in areas)
                    central = box[0] <= .5*r["width"] <= box[2] and box[1] <= .5*r["height"] <= box[3]
                    good = significant == 1 and central and fraction >= .22 and confs[i] >= .45
                    reason = "eligible" if good else "multiple_or_small_or_offcenter_person"
                    main = {"bbox": box.tolist(), "fraction": fraction, "significant_people": int(significant), "score": float(confs[i])}
                    if good and len(kept) < REF_CAP:
                        kept.append(dict(r, split=split_for[pid], domain="reference", reference_screen=main,
                            label_source="MuPNIT_external_gallery_directory", quality_source="automatic_single_dominant_person_screen"))
                diagnostics.append({"path": r["path"], "pid": pid, "result": reason, "person": main})
        selected.extend(kept)
        if number % 10 == 0:
            log("reference_screen", identities=number, total_identities=len(by_id), retained=len(selected))
    write_manifest(OUT / "reference_screen.jsonl", diagnostics)
    missing = set(split_for) - {r["pid"] for r in selected}
    atomic_json(REPORTS / "reference_screen_summary.json", {"references": len(selected), "identities": len(set(r["pid"] for r in selected)),
                "missing_identities": sorted(missing), "criterion": "One dominant detected person; center; >=22% source area; detection score >=0.45; minimum dimensions and exact duplicate screening.",
                "scope": "Automatic visual quality screening, not human-verified identity or watermark removal."})
    if missing:
        write_manifest(OUT / "references_partial.jsonl", selected)
        raise RuntimeError(f"Reference review needed for IDs {sorted(missing)}")
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError("Prepared manifest already exists; use a new run directory")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    inventory = json.loads((ROOT / "reports/local_dataset_inventory.json").read_text())
    ids = {k: set(v) for k, v in inventory["identities_by_section"].items()}
    train, test = ids["train"], ids["test"]
    assert not train & test
    val = ids["val"] - train - test
    split_for = {pid: split for split, group in [("train", train), ("val", val), ("test", test)] for pid in group}
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": 42,
                "ids": {s: sorted(p for p, t in split_for.items() if t == s) for s in ["train", "val", "test"]},
                "query_cap_per_identity": QUERY_CAP, "reference_cap_per_identity": REF_CAP,
                "sampling": "Round-robin source video sequences with evenly spaced distinct video/frame crops; no recognition scores.",
                "excluded_val_ids": sorted(ids["val"] - val),
                "mask_failure_policy": "Record and exclude unsuccessful segmentation from this fixed initial subset before any training; report retention by split/identity. No fabricated masks or silent denominator changes.",
                "reference_policy": "Decode, size/shape screening, remove exact duplicates and cross-ID ambiguous copies, then single-dominant-person detection. Original reference pixels are retained.",
                "segmenter_sha256": digest_file(SEGMENTER), "preparation_code_sha256": digest_file(__file__),
                "training_scope": "Initial identity-balanced subset from full MuPNIT-ReID; not all 365,924 broadcast images; no unaugmented training."}
    atomic_json(OUT / "protocol.json", protocol)
    log("gallery_decode_and_hash", images=inventory["files_by_section"]["gallery"])
    cache = OUT / "gallery_inventory.jsonl"
    if cache.exists():
        gallery = [json.loads(s) for s in cache.read_text().splitlines()]
    else:
        paths = sorted((DATA / "gallery").glob("player_*/*.jpg"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            gallery = list(pool.map(image_info, paths))
        write_manifest(cache, gallery)
    owners = defaultdict(set)
    for r in gallery:
        if "sha256" in r: owners[r["sha256"]].add(r["pid"])
    seen, candidates, rejects = set(), [], []
    for r in gallery:
        reason = None
        if "error" in r: reason = "decode_failed"
        elif len(owners[r["sha256"]]) > 1: reason = "identical_bytes_different_player_labels"
        elif r["sha256"] in seen: reason = "duplicate_within_identity"
        elif r["width"] < 96 or r["height"] < 128 or r["width"]/r["height"] > 1.35 or r["sharpness"] < 8:
            reason = "size_shape_or_blur"
        if reason:
            rejects.append({"path": r["path"], "reason": reason}); continue
        seen.add(r["sha256"]); candidates.append(r)
    write_manifest(OUT / "gallery_exclusions.jsonl", rejects)
    log("gallery_inventory_complete", candidates=len(candidates), exclusions=dict(Counter(r["reason"] for r in rejects)))
    from ultralytics import YOLO
    model = YOLO(str(SEGMENTER))
    references_path = OUT / "references.jsonl"
    if references_path.exists():
        references = [json.loads(s) for s in references_path.read_text().splitlines()]
    else:
        references = reference_screen(model, candidates, split_for)
        write_manifest(references_path, references)
    query_paths = [(split, pid, p) for split in ("train", "val", "test") for pid in sorted(p for p,t in split_for.items() if t == split)
                   for p in sample_queries(DATA / split / f"player_{int(pid):04d}", QUERY_CAP[split])]
    info = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(image_info, [p for _,_,p in query_paths]): info[r["path"]] = r
    reference_hashes = {r["sha256"] for r in references}
    qs, exclusions, seen_q = [], [], {}
    for split, pid, p in query_paths:
        r = info[str(p)]
        if "error" in r or r["width"] < 14 or r["height"] < 24:
            exclusions.append({"path": str(p), "split": split, "pid": pid, "reason": "decode_or_too_small"}); continue
        key = r["sha256"]
        if key in reference_hashes:
            raise ValueError("Identical query and reference image")
        if key in seen_q:
            if seen_q[key] != (split, pid): raise ValueError("Cross-ID/split duplicate query")
            exclusions.append({"path": str(p), "split": split, "pid": pid, "reason": "duplicate_query_bytes"}); continue
        seen_q[key] = (split, pid)
        qs.append(dict(r, split=split, domain="query", label_source="MuPNIT_broadcast_directory", sample_id=str(p.relative_to(DATA))))
    mask_dir = OUT / "masks"; mask_dir.mkdir(exist_ok=True)
    masked, failures, methods = [], [], Counter()
    weight_hash = protocol["segmenter_sha256"]
    for start in range(0, len(qs), 16):
        batch = qs[start:start+16]
        pending = []
        for r in batch:
            path = mask_dir / f'{MASK_VERSION}_{weight_hash[:12]}_{r["sha256"]}.png'
            r["person_mask"] = str(path)
            if path.exists():
                load_masks(r, (r["width"], r["height"])); masked.append(r); methods["cached"] += 1
            else: pending.append(r)
        if pending:
            imgs = [read_rgb(r["path"]) for r in pending]
            results = model.predict([np.asarray(im)[..., ::-1] for im in imgs], classes=[0], device="cuda:0", imgsz=640,
                                    conf=.15, retina_masks=True, verbose=False)
            for r, im, result in zip(pending, imgs, results):
                try:
                    try:
                        mask = select_person(result, (im.height, im.width)); method = "original_batch"
                    except ValueError:
                        mask, method = generate_person_mask(model, im, "cuda:0")
                    Image.fromarray(mask).save(r["person_mask"])
                    load_masks(r, im.size); masked.append(r); methods[method] += 1
                except (ValueError, OSError) as exc:
                    failures.append({"path": r["path"], "pid": r["pid"], "split": r["split"], "reason": str(exc)})
        if start % 256 == 0:
            log("query_masks", processed=min(start+16,len(qs)), total=len(qs), retained=len(masked), failed=len(failures), methods=methods)
    write_manifest(OUT / "query_exclusions.jsonl", exclusions + failures)
    missing = set(split_for) - {r["pid"] for r in masked}
    if missing: raise RuntimeError(f"No usable query masks for IDs {sorted(missing)}")
    rows = sorted(masked + references, key=lambda r: (r["split"], r["domain"], int(r["pid"]), r["path"]))
    # Verify every included image and mask and prevent cross-domain/split leakage.
    stats = audit(rows, require_masks=True)
    write_manifest(manifest, rows)
    per_split = {}
    for split in ("train", "val", "test"):
        per_split[split] = {"sampled_query_candidates": sum(s == split for s,_,_ in query_paths),
                           "retained_queries": sum(r["split"] == split for r in masked),
                           "query_exclusions": sum(r["split"] == split for r in exclusions),
                           "mask_failures": sum(r["split"] == split for r in failures),
                           "references": sum(r["split"] == split for r in references)}
    stats.update(per_split=per_split, mask_methods=dict(methods), manifest_sha256=digest_file(manifest),
                 stage="initial_subset", full_dataset_training=False, test_used_for_training_or_selection=False,
                 quality_curation="Automatic; semantic reference identity not human-certified", prepared_at=datetime.now(timezone.utc).isoformat())
    atomic_json(REPORTS / "data_preparation.json", stats)
    # Seeded cross-identity preview: original, real mask, strong augmentation, clean reference.
    chosen = sorted([r for r in masked if r["split"] == "train"], key=lambda r: seed_for(1701,r["path"]))
    unique = {}; ref_by = defaultdict(list)
    for r in references: ref_by[r["pid"]].append(r)
    for r in chosen:
        unique.setdefault(r["pid"],r)
        if len(unique) == 12: break
    sheet = Image.new("RGB", (960, 260*len(unique)), "#14212b"); draw = ImageDraw.Draw(sheet)
    for i,r in enumerate(unique.values()):
        im = read_rgb(r["path"]); masks = load_masks(r, im.size)
        augmented = BroadcastAugment("strong")(im,masks,np.random.default_rng(seed_for(1701,r["path"])))
        mask_img = Image.fromarray((masks["person"]*255).astype(np.uint8)).convert("RGB")
        images = [im,mask_img,augmented,read_rgb(ref_by[r["pid"]][0]["path"])]
        for j,(label,image) in enumerate(zip(["Original query","Person mask","Augmented query","Clean reference"],images)):
            image.thumbnail((220,220));sheet.paste(image,(j*240+(240-image.width)//2,i*260+28))
            draw.text((j*240+8,i*260+6),f'ID {r["pid"]} | {label}',fill="white")
    sheet.save(REPORTS / "training_preview.jpg")
    log("preparation_complete", manifest=str(manifest), counts=stats["counts"], identities=stats["identities"], per_split=per_split)


if __name__ == "__main__": main()
