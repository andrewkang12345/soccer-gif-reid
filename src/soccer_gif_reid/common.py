from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GIF_COMMIT = "304370c107d9091bea47bb9c5432ac76020c24c3"


def image_paths(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images in {root}")
    return paths


def read_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if min(im.size) < 2:
            raise ValueError(f"Image too small: {path}")
        return im.copy()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_digest():
    """Fingerprint executable pipeline code separately from upstream GIF provenance."""
    root = Path(__file__).resolve().parent
    files = [(p.name, digest_file(p)) for p in sorted(root.glob("*.py"))]
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path, value):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temp))
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_manifest(path):
    path = Path(path).resolve()
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(k in row for k in ("path", "pid", "domain", "split")):
            raise ValueError(f"{path}:{i}: missing path/pid/domain/split")
        if row["domain"] not in {"query", "reference"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"{path}:{i}: invalid domain or split")
        if row["pid"] is None or not str(row["pid"]).strip():
            raise ValueError(f"{path}:{i}: empty player identity")
        row["pid"] = str(row["pid"]).strip()
        if row["pid"].isdecimal():
            row["pid"] = str(int(row["pid"]))
        for k in ("path", "person_mask", "shirt_mask", "shorts_mask"):
            if row.get(k):
                p = Path(row[k])
                row[k] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def write_manifest(path, rows):
    atomic_write(path, lambda p: p.write_text("".join(json.dumps(r) + "\n" for r in rows)))
