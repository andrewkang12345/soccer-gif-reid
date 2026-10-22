from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import atomic_json, digest_file, image_paths, load_manifest, read_rgb, seed_for, write_manifest


def scan(root, domain, split):
    rows = []
    for p in image_paths(root):
        # Full release: the directory supplies global ID; the filename starts
        # with a video ID. Flat light-release filenames use a global-ID prefix.
        folders = [match.group(1) for parent in p.parents
                   if (match := re.fullmatch(r"player_(\d+)", parent.name))]
        if len(set(folders)) > 1:
            raise ValueError(f"Conflicting identity directories: {p}")
        token = folders[0] if folders else p.stem.split("_")[0]
        if not token.isdecimal() or int(token) <= 0:
            raise ValueError(f"Expected <global-player-id>_... image filename: {p}")
        rows.append({"path": str(p), "pid": str(int(token)), "domain": domain, "split": split})
    return rows


def audit(rows, check_images=True, require_masks=False):
    identities = defaultdict(set)
    groups = defaultdict(lambda: defaultdict(set))
    seen_paths, seen_hashes = {}, {}
    content_records = []
    for row in rows:
        pid, split, domain = row["pid"], row["split"], row["domain"]
        identities[split].add(pid)
        groups[split][pid].add(domain)
        p = str(Path(row["path"]).resolve())
        if p in seen_paths:
            raise ValueError(f"Duplicate image path: {p}")
        seen_paths[p] = (split, domain)
        if not Path(p).is_file():
            raise ValueError(f"Missing image: {p}")
        if check_images:
            read_rgb(p)
            h = digest_file(p)
            if row.get("sha256") and row["sha256"] != h:
                raise ValueError(f"Image changed since manifest was prepared: {p}")
            # Copies inside a domain are reported; across domains/splits are leakage.
            if h in seen_hashes and seen_hashes[h] != (split, domain, pid):
                raise ValueError(f"Identical bytes across domains/splits/identities: {p}")
            seen_hashes[h] = (split, domain, pid)
            content_records.append((p, h))
        if require_masks and domain == "query":
            from .augment import load_masks
            load_masks(row, read_rgb(p).size)
            for name in ("person_mask", "shirt_mask", "shorts_mask"):
                if row.get(name):
                    content_records.append((row[name], digest_file(row[name])))
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = identities[s1] & identities[s2]
        if overlap:
            raise ValueError(f"Identity leakage between {s1}/{s2}: {sorted(overlap)[:10]}")
    for split, pids in groups.items():
        if len(pids) < 2:
            raise ValueError(f"{split} needs at least two identities")
        for pid, domains in pids.items():
            if domains != {"query", "reference"}:
                raise ValueError(f"{split} identity {pid} lacks a query or clean reference; no self-image fallback")
    return {"counts": dict(Counter(r["split"] + "/" + r["domain"] for r in rows)),
            "identities": {s: len(p) for s, p in identities.items()},
            "content_hashes_checked": check_images,
            "dataset_sha256": hashlib.sha256(json.dumps(sorted(content_records)).encode()).hexdigest(),
            "duplicate_content_within_domain": len(rows) - len(seen_hashes) if check_images else None}


def holdout_train_identities(rows, fraction=0.2, seed=42):
    ids = sorted({r["pid"] for r in rows if r["split"] == "train"}, key=lambda p: seed_for(seed, p))
    if not 0 < fraction < 1 or len(ids) < 4:
        raise ValueError("Need at least four train identities and 0 < val-fraction < 1")
    count = min(len(ids) - 2, max(2, round(len(ids) * fraction)))
    held = set(ids[:count])
    return [{**r, "split": "val" if r["split"] == "train" and r["pid"] in held else r["split"]}
            for r in rows]


def main():
    p = argparse.ArgumentParser(description="Build/audit explicit query-reference manifests; never infer web references from broadcast quality.")
    p.add_argument("--train-query", type=Path)
    p.add_argument("--train-reference", type=Path)
    p.add_argument("--val-query", type=Path)
    p.add_argument("--val-reference", type=Path)
    p.add_argument("--test-query", type=Path)
    p.add_argument("--test-reference", type=Path)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=Path, help="Audit an existing custom JSONL manifest instead")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    if args.manifest:
        rows = load_manifest(args.manifest)
    else:
        if not args.train_query or not args.train_reference:
            p.error("Provide --train-query and --train-reference, or --manifest")
        rows = scan(args.train_query, "query", "train") + scan(args.train_reference, "reference", "train")
        for split in ("val", "test"):
            q, g = getattr(args, split + "_query"), getattr(args, split + "_reference")
            if bool(q) != bool(g):
                p.error(f"Both {split} query and reference directories are required")
            if q:
                rows += scan(q, "query", split) + scan(g, "reference", split)
        if not args.val_query:
            rows = holdout_train_identities(rows, args.val_fraction, args.seed)
    summary = audit(rows)
    write_manifest(args.output, rows)
    summary.update({"manifest_sha256": digest_file(args.output), "seed": args.seed,
                    "validation": "explicit" if args.manifest or args.val_query else "identity_holdout_from_training"})
    atomic_json(args.output.with_suffix(".audit.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
