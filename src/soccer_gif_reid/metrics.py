from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def retrieval_metrics(query, gallery, query_ids, gallery_ids, chunk_size=256):
    """Image-gallery AP/CMC, all reference images relevant, no camera filtering.

    Missing positives fail instead of silently inflating mAP by skipping queries.
    The manifest audit forbids self matches and duplicate cross-domain image bytes.
    Stable tie-breaking follows sorted manifest/gallery order.
    """
    if len(query) != len(query_ids) or len(gallery) != len(gallery_ids) or not len(query) or not len(gallery):
        raise ValueError("Empty/misaligned evaluation inputs")
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("Invalid feature dimensions")
    if chunk_size < 1 or not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("Invalid chunk size/nonfinite features")
    if (query.norm(dim=-1) < 1e-8).any() or (gallery.norm(dim=-1) < 1e-8).any():
        raise ValueError("Zero feature vector")
    missing = set(query_ids) - set(gallery_ids)
    if missing:
        raise ValueError(f"Queries without reference positives: {sorted(missing)}")
    q = F.normalize(query.float().cpu(), dim=-1)
    g = F.normalize(gallery.float().cpu(), dim=-1)
    gpids = np.array(gallery_ids)
    values, per_identity = [], defaultdict(list)
    for start in range(0, len(q), chunk_size):
        sims = (q[start:start + chunk_size] @ g.T).numpy()
        for offset, sim in enumerate(sims):
            pid = query_ids[start + offset]
            matches = gpids[np.argsort(-sim, kind="stable")] == pid
            ranks = np.flatnonzero(matches)
            ap = float(np.mean(np.arange(1, len(ranks) + 1) / (ranks + 1)))
            v = (float(matches[0]), ap, float(ranks[0] < 5))
            values.append(v)
            per_identity[pid].append(v)
    mean = np.mean(values, axis=0)
    per = {str(pid): {"rank1": float(np.mean(v, axis=0)[0]), "mAP": float(np.mean(v, axis=0)[1]),
                      "queries": len(v)} for pid, v in per_identity.items()}
    return {"rank1": float(mean[0]), "mAP": float(mean[1]), "rank5": float(mean[2]),
            "queries": len(query), "gallery_images": len(gallery), "identities": len(per),
            "macro_identity_rank1": float(np.mean([v["rank1"] for v in per.values()])),
            "macro_identity_mAP": float(np.mean([v["mAP"] for v in per.values()])),
            "per_identity": per}


def selection_score(metrics):
    """Balance both validation domains, primarily using mAP."""
    clean, heavy = metrics["clean"]["mAP"], metrics["degraded"]["mAP"]
    return 2 * clean * heavy / (clean + heavy) if clean + heavy else 0.0
