"""Freeze a shared four-frame protocol before either model is trained."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from soccer_gif_reid.common import atomic_json, digest_file, load_manifest, read_rgb, seed_for, write_manifest
from soccer_gif_reid.masks import MASK_VERSION, generate_person_mask, select_person
from soccer_gif_reid.augment import load_masks
from soccer_gif_reid.prepare import audit
from scripts.prepare_training_run import image_info

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/ap3d_comparison_v1"
REPORT = ROOT / "reports/ap3d_comparison_v1"


def windows(paths):
    """Same video/local ID, monotonically adjacent frames; never cross a gap.

    Source is 60 fps, sampled every four frames. Four observations are drawn
    from each disjoint block of 16 available crops (typically one second).
    A missing interval >8 source frames splits a sequence. No frame repeats.
    """
    groups = defaultdict(dict)
    for path in sorted(paths):
        video, frame, local, *_ = path.stem.split("_")
        groups[video, local].setdefault(int(frame), path)
    result = []
    for (video, local), mapping in sorted(groups.items()):
        runs, run = [], []
        for frame in sorted(mapping):
            if run and frame - run[-1] > 8:
                runs.append(run); run = []
            run.append(frame)
        runs.append(run)
        for run in runs:
            for start in range(0, len(run), 16):
                block = run[start:start + 16]
                if len(block) < 4:
                    continue
                frames = [block[i] for i in np.linspace(0, len(block)-1, 4, dtype=int)]
                result.append({"video": video, "local_id": local, "frames": frames,
                               "paths": [str(mapping[f]) for f in frames]})
    return result


def choose(candidates, cap, pid):
    # Spread across videos first, then uniformly over temporal windows.
    by_video = defaultdict(list)
    for item in candidates:
        by_video[item["video"]].append(item)
    pools = {}
    for video, items in by_video.items():
        indices = np.linspace(0, len(items)-1, min(cap, len(items)), dtype=int)
        pools[video] = [items[i] for i in indices]
    order = sorted(pools, key=lambda v: seed_for(42, pid, v))
    result = []
    while len(result) < cap and any(pools.values()):
        for video in order:
            if pools[video] and len(result) < cap:
                result.append(pools[video].pop(0))
    return result


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.jsonl").exists():
        raise FileExistsError("Protocol already frozen; choose a new version to change it")
    original = load_manifest(ROOT / "data/augmented_head_v1/manifest.jsonl")
    refs = [r for r in original if r["domain"] == "reference"]
    split_for = {r["pid"]: r["split"] for r in refs}
    candidates = []
    for pid, split in sorted(split_for.items(), key=lambda x: int(x[0])):
        folder = ROOT / "data/MuPNIT-ReID" / split / f"player_{int(pid):04d}"
        chosen = choose(windows(folder.glob("*.jpg")), 32 if split == "train" else 8, pid)
        for item in chosen:
            item.update(pid=pid, split=split,
                        clip_id=f'{split}/{pid}/{item["video"]}/{item["local_id"]}/{item["frames"][0]}')
        candidates.extend(chosen)
    paths = sorted({p for item in candidates for p in item["paths"]})
    with ThreadPoolExecutor(max_workers=4) as pool:
        info = {r["path"]: r for r in pool.map(image_info, map(Path, paths))}
    print(json.dumps({"event": "tracklets_selected", "candidates": len(candidates), "unique_frames": len(paths)}), flush=True)
    from ultralytics import YOLO
    weight = Path("/mnt/data/sports_reID/yolo11m-seg.pt")
    weight_hash = digest_file(weight)
    model = YOLO(str(weight))
    cache = ROOT / "data/augmented_head_v1/masks"
    failures, methods = {}, Counter()
    for start in range(0, len(paths), 16):
        pending = []
        for path in paths[start:start+16]:
            row = info[path]
            if "error" in row or row["width"] < 14 or row["height"] < 24:
                failures[path] = "decode_or_too_small"; continue
            row["person_mask"] = str(cache / f'{MASK_VERSION}_{weight_hash[:12]}_{row["sha256"]}.png')
            if Path(row["person_mask"]).exists():
                load_masks(row, (row["width"], row["height"])); methods["cached"] += 1
            else:
                pending.append(row)
        if pending:
            images = [read_rgb(r["path"]) for r in pending]
            results = model.predict([np.asarray(im)[..., ::-1] for im in images], classes=[0],
                device="cuda:0", imgsz=640, conf=.15, retina_masks=True, verbose=False)
            for row, im, result in zip(pending, images, results):
                try:
                    try:
                        mask = select_person(result, (im.height, im.width)); method = "batch"
                    except ValueError:
                        mask, method = generate_person_mask(model, im, "cuda:0")
                    Image.fromarray(mask).save(row["person_mask"])
                    load_masks(row, im.size); methods[method] += 1
                except (ValueError, OSError) as exc:
                    failures[row["path"]] = str(exc)
        if start % 256 == 0:
            progress = {"event": "tracklet_masks", "processed": min(start+16, len(paths)),
                        "total": len(paths), "failed_frames": len(failures), "methods": dict(methods)}
            atomic_json(REPORT / "preparation_progress.json", progress)
            print(json.dumps(progress), flush=True)
    clips, rejected, rows = [], [], []
    for clip in candidates:
        bad = {p: failures[p] for p in clip["paths"] if p in failures}
        hashes = [info[p].get("sha256") for p in clip["paths"]]
        if bad or len(set(hashes)) != 4:
            rejected.append({**clip, "failures": bad, "duplicate_frames": len(set(hashes)) != 4})
            continue
        clips.append(clip)
        for path in clip["paths"]:
            rows.append(dict(info[path], split=clip["split"], domain="query", clip_id=clip["clip_id"], sample_id=path))
    write_manifest(OUT / "rejected_clips.jsonl", rejected)
    missing = set(split_for) - {c["pid"] for c in clips}
    if missing:
        raise ValueError(f"Identities with no eligible complete clip: {sorted(missing)}")
    rows = sorted(rows + refs, key=lambda r: (r["split"], r["domain"], int(r["pid"]), r["path"]))
    stats = audit(rows, require_masks=True)
    write_manifest(OUT / "manifest.jsonl", rows)
    write_manifest(OUT / "clips.jsonl", clips)
    stats.update(candidates_by_split=dict(Counter(c["split"] for c in candidates)),
                 retained_by_split=dict(Counter(c["split"] for c in clips)),
                 rejected_by_split=dict(Counter(c["split"] for c in rejected)),
                 source_fps=60, frames_per_clip=4, candidate_block_frames=16,
                 max_gap_between_available_frames=8,
                 max_selected_clip_span=max(c["frames"][-1]-c["frames"][0] for c in clips),
                 manifest_sha256=digest_file(OUT / "manifest.jsonl"),
                 clips_sha256=digest_file(OUT / "clips.jsonl"),
                 semantic_reference_identity_human_verified=False,
                 shot_boundary_detection=False,
                 limitations="Short uninterrupted source-ID windows; camera cuts are not explicitly detected. Failed-mask clips excluded before training; this measures the retained subset, not full-release coverage.",
                 test_used_for_training_or_selection=False)
    atomic_json(REPORT / "data_preparation.json", stats)
    print(json.dumps({"event": "preparation_complete", **{k: stats[k] for k in ("counts", "retained_by_split", "rejected_by_split")}}), flush=True)


if __name__ == "__main__":
    main()
