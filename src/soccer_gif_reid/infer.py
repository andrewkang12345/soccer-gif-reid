from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .augment import preprocess
from .common import EXTENSIONS, atomic_json, digest_file, image_paths, read_rgb
from .model import from_checkpoint


def roster_images(root):
    root = Path(root).resolve()
    files = image_paths(root)
    flat = [p.parent == root for p in files]
    if any(flat) and not all(flat):
        raise ValueError("Use either roster/<identity>/*.jpg or flat roster/<identity>.jpg; do not mix layouts")
    labels = [p.stem if all(flat) else p.relative_to(root).parts[0] for p in files]
    return files, labels


def make_prototypes(features, labels):
    if not len(features) or len(features) != len(labels) or not torch.isfinite(features).all():
        raise ValueError("Invalid roster features")
    ids = sorted(set(labels))
    features = F.normalize(features.float(), dim=-1)
    prototypes = torch.stack([features[[i for i, pid in enumerate(labels) if pid == target]].mean(0) for target in ids])
    if (prototypes.norm(dim=-1) < 1e-8).any():
        raise ValueError("Degenerate roster prototype")
    return ids, F.normalize(prototypes, dim=-1)


def rank_tracklet(features, prototypes, ids, top_k=5, min_similarity=None, min_margin=None):
    if top_k < 1 or len(features) == 0 or len(ids) == 0:
        raise ValueError("Need query frames, roster identities and top_k >= 1")
    if min_similarity is not None and (not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("min_similarity must be finite and between -1 and 1")
    if min_margin is not None and (not math.isfinite(min_margin) or not 0 <= min_margin <= 2):
        raise ValueError("min_margin must be finite and between 0 and 2")
    if features.ndim != 2 or prototypes.ndim != 2 or features.shape[1] != prototypes.shape[1] or len(prototypes) != len(ids):
        raise ValueError("Mismatched query/roster features")
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
        raise ValueError("Nonfinite feature")
    if (features.norm(dim=-1) < 1e-8).any() or (prototypes.norm(dim=-1) < 1e-8).any():
        raise ValueError("Zero feature")
    q = F.normalize(features.float(), dim=-1)
    # Mean per-frame cosine keeps scores bounded without turning weak agreement into
    # artificially high confidence by re-normalizing a cancelling tracklet average.
    frame_scores = q @ F.normalize(prototypes.float(), dim=-1).T
    scores = frame_scores.mean(0)
    order = torch.argsort(scores, descending=True, stable=True)
    best = float(scores[order[0]])
    margin = float(best - scores[order[1]]) if len(ids) > 1 else None
    if min_margin is not None and margin is None:
        raise ValueError("Margin rejection needs at least two roster identities")
    accepted = (min_similarity is None or best >= min_similarity) and (min_margin is None or margin >= min_margin)
    return {"ranked_identities": [{"identity": ids[i], "similarity": float(scores[i])} for i in order[:top_k].tolist()],
            "prediction": ids[int(order[0])] if accepted else "unknown", "accepted": accepted,
            "margin": margin, "frames_used": len(q),
            "frame_top1_agreement": float((frame_scores.argmax(1) == order[0]).float().mean()),
            "rejection_enabled": min_similarity is not None or min_margin is not None,
            "score_type": "mean_frame_cosine_to_identity_reference_centroid", "scores_are_probabilities": False}


@torch.inference_mode()
def encode_paths(model, paths, device, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(read_rgb(p)) for p in paths[start:start + batch_size]]).to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            result.append(model(batch).float().cpu())
    return torch.cat(result)


class RosterMatcher:
    """Load once and reuse in a service; input is already-detected player crops."""

    def __init__(self, checkpoint, roster, device="cuda:0", batch_size=16):
        self.device, self.batch_size = device, batch_size
        self.model, metadata = from_checkpoint(checkpoint, device)
        self.checkpoint_sha256 = digest_file(checkpoint)
        files, labels = roster_images(roster)
        self.ids, self.prototypes = make_prototypes(encode_paths(self.model, files, device, batch_size), labels)
        self.roster_references = len(files)
        self.training = metadata.get("training")

    def match(self, crop_or_tracklet, top_k=5, max_frames=64, min_similarity=None, min_margin=None):
        path = Path(crop_or_tracklet)
        if path.is_dir():
            paths = image_paths(path)
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:
            paths = [path.resolve()]
        else:
            raise ValueError("Query must be a crop image or directory of crops from one player's tracklet")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        total = len(paths)
        if total > max_frames:
            paths = [paths[i] for i in np.linspace(0, total - 1, max_frames, dtype=int)]
        features = encode_paths(self.model, paths, self.device, self.batch_size)
        result = rank_tracklet(features, self.prototypes, self.ids, top_k, min_similarity, min_margin)
        return {**result, "frames_available": total, "selected_frames": [str(p) for p in paths],
                "roster_size": len(self.ids), "reference_images": self.roster_references,
                "checkpoint_sha256": self.checkpoint_sha256, "training": self.training}


def main():
    p = argparse.ArgumentParser(description="Rank roster identities for a crop or single-player tracklet; cosine scores are not probabilities")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--roster", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=64)
    p.add_argument("--min-similarity", type=float, help="Optional threshold calibrated on held-out soccer data")
    p.add_argument("--min-margin", type=float, help="Optional top1-top2 threshold calibrated on held-out soccer data")
    args = p.parse_args()
    import json
    matcher = RosterMatcher(args.checkpoint, args.roster, args.device, args.batch_size)
    result = matcher.match(args.query, args.top_k, args.max_frames, args.min_similarity, args.min_margin)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
