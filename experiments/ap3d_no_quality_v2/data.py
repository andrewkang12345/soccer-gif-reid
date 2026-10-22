from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from soccer_gif_reid.augment import load_masks, preprocess
from .augment import BroadcastAugment
from soccer_gif_reid.common import read_rgb, seed_for


def transform(image, kind):
    if kind == "clip":
        return preprocess(image)
    canvas = ImageOps.pad(image, (128, 256), method=Image.Resampling.BICUBIC, color=(124, 116, 104))
    value = torch.from_numpy(np.array(canvas, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
    return (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def render_frames(clip, rows_by_path, kind, augmented=False, seed=1701, severity="strong", donor=None):
    result = []
    for path in clip["paths"]:
        row = rows_by_path[path]; im = read_rgb(path)
        if augmented:
            # Reset to a common clip seed: persistent kit/scene augmentation,
            # rather than independent clothing identities in adjacent frames.
            im = BroadcastAugment(severity)(im, load_masks(row, im.size), np.random.default_rng(seed), donor,
                                            temporal_seed=seed)
        result.append(transform(im, kind))
    return torch.stack(result)


class ClipPairs(Dataset):
    def __init__(self, rows, clips, kind, seed=42):
        self.rows = {r["path"]: r for r in rows}
        self.queries = [c for c in clips if c["split"] == "train"]
        self.refs, self.by_pid = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["split"] == "train" and r["domain"] == "reference":
                self.refs[r["pid"]].append(r)
        for i, c in enumerate(self.queries):
            self.by_pid[c["pid"]].append(i)
        self.labels = {pid: i for i, pid in enumerate(sorted(self.by_pid))}
        self.seed, self.epoch, self.kind = seed, 0, kind

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        idx, occurrence = index
        clip = self.queries[idx]
        seed = seed_for(self.seed, self.epoch, idx, occurrence)
        rng = np.random.default_rng(seed)
        ref = self.refs[clip["pid"]][int(rng.integers(len(self.refs[clip["pid"]])))]
        donor = None
        if rng.random() < .5:
            others = sorted(set(self.by_pid) - {clip["pid"]})
            pid = others[int(rng.integers(len(others)))]
            d = self.queries[self.by_pid[pid][int(rng.integers(len(self.by_pid[pid])))]]
            row = self.rows[d["paths"][0]]; im = read_rgb(row["path"])
            donor = im, load_masks(row, im.size)["person"]
        q = render_frames(clip, self.rows, self.kind, True, seed, "strong", donor)
        return q, transform(read_rgb(ref["path"]), self.kind), self.labels[clip["pid"]]


class ClipEvaluation(Dataset):
    def __init__(self, rows, items, kind, reference=False, augmented=False, seed=1701):
        self.rows = {r["path"]: r for r in rows}
        self.items, self.kind, self.reference = items, kind, reference
        self.augmented, self.seed = augmented, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        item = self.items[idx]
        if self.reference:
            im = transform(read_rgb(item["path"]), self.kind)
            return im[None].repeat(4 if self.kind == "ap3d" else 1, 1, 1, 1)
        return render_frames(item, self.rows, self.kind, self.augmented, seed_for(self.seed, item["clip_id"]))
