from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from torch.utils.data import Dataset, Sampler

from .augment import BroadcastAugment, load_masks, preprocess
from .common import read_rgb, seed_for


class PairDataset(Dataset):
    def __init__(self, rows, seed=42, severity="strong"):
        self.queries = [r for r in rows if r["split"] == "train" and r["domain"] == "query"]
        self.refs = defaultdict(list)
        for r in rows:
            if r["split"] == "train" and r["domain"] == "reference":
                self.refs[r["pid"]].append(r)
        self.labels = {pid: i for i, pid in enumerate(sorted(self.refs))}
        self.by_pid = defaultdict(list)
        for i, q in enumerate(self.queries):
            if q["pid"] not in self.refs:
                raise ValueError(f"No clean train reference for {q['pid']}")
            self.by_pid[q["pid"]].append(i)
        self.seed, self.epoch = seed, 0
        self.augment = BroadcastAugment(severity)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        # occurrence gives repeated small-identity samples different augmentation.
        idx, occurrence = index if isinstance(index, tuple) else (index, 0)
        q = self.queries[idx]
        rng = np.random.default_rng(seed_for(self.seed, self.epoch, idx, occurrence))
        ref = self.refs[q["pid"]][int(rng.integers(len(self.refs[q["pid"]])))]
        image = read_rgb(q["path"])
        masks = load_masks(q, image.size)
        donor = None
        if rng.random() < 0.5:
            other_ids = sorted(set(self.by_pid) - {q["pid"]})
            if other_ids:
                pid = other_ids[int(rng.integers(len(other_ids)))]
                candidates = self.by_pid[pid]
                d = self.queries[candidates[int(rng.integers(len(candidates)))]]
                di = read_rgb(d["path"])
                donor = di, load_masks(d, di.size)["person"]
        query = self.augment(image, masks, rng, donor)
        return preprocess(query), preprocess(read_rgb(ref["path"])), self.labels[q["pid"]]


class IdentityBatchSampler(Sampler):
    """P distinct identities × K queries, with paired references, every step."""

    def __init__(self, dataset, identities=8, queries_per_identity=2, steps=500):
        if not 2 <= identities <= len(dataset.by_pid) or queries_per_identity < 1 or steps < 1:
            raise ValueError("Need 2 <= identities_per_batch <= train identities, K >= 1, steps >= 1")
        self.dataset, self.p, self.k, self.steps = dataset, identities, queries_per_identity, steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(seed_for(self.dataset.seed, self.dataset.epoch, "sampler"))
        ids = sorted(self.dataset.by_pid)
        for step in range(self.steps):
            batch = []
            for pid in rng.choice(ids, self.p, replace=False):
                candidates = self.dataset.by_pid[pid]
                sampled = rng.choice(candidates, self.k, replace=len(candidates) < self.k)
                batch.extend((int(idx), step * self.p * self.k + len(batch) + j) for j, idx in enumerate(sampled))
            yield batch


class EvalDataset(Dataset):
    def __init__(self, rows, degraded=False, seed=1701):
        self.rows, self.degraded, self.seed = rows, degraded, seed
        self.augment = BroadcastAugment("heavy")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        row = self.rows[idx]
        im = read_rgb(row["path"])
        if self.degraded:
            # Same query receives exactly the same corruption at every epoch/checkpoint.
            key = row.get("sample_id", row.get("sha256", row["path"]))
            im = self.augment(im, load_masks(row, im.size), np.random.default_rng(seed_for(self.seed, key)))
        return preprocess(im)
