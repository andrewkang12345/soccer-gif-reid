from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .common import seed_for

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


def preprocess(image):
    """Preserve the whole player/headshot; never center-crop away head or feet."""
    image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC,
                         color=(123, 117, 104), centering=(0.5, 0.5))
    x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    return (x - CLIP_MEAN) / CLIP_STD


def load_masks(row, size):
    masks = {}
    for name in ("person", "shirt", "shorts"):
        if row.get(name + "_mask"):
            with Image.open(row[name + "_mask"]) as im:
                if im.size != size:
                    raise ValueError(f"Mask/image size mismatch: {row['path']}")
                masks[name] = np.asarray(im.convert("L")) > 127
    if "person" not in masks:
        raise ValueError(f"Missing person mask for query: {row['path']}; run soccer-reid-masks")
    if not 0.03 <= masks["person"].mean() <= 0.99:
        raise ValueError(f"Implausible person mask: {row['path']}")
    return masks


def kit_masks(rgb, masks):
    """Use parsing masks when supplied; otherwise conservative torso/shorts bands.

    Geometry is an approximation, not a semantic clothing parser. Protect the head,
    exposed skin, and outer arms; preview before large-scale training.
    """
    h, w = rgb.shape[:2]
    y, x = np.mgrid[:h, :w]
    yy, xx = y / h, x / w
    yc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = (yc[..., 1] > 132) & (yc[..., 1] < 180) & (yc[..., 2] > 76) & (yc[..., 2] < 135)
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    person = masks["person"]
    bands = {"shirt": (yy > 0.25) & (yy < 0.56) & (xx > 0.27) & (xx < 0.73),
             "shorts": (yy >= 0.56) & (yy < 0.73) & (xx > 0.24) & (xx < 0.76)}
    return {name: masks.get(name, band) & person & ~skin & (yy > 0.23)
            for name, band in bands.items()}


def stadium(h, w, rng):
    """Procedural turf or out-of-focus stands, independent of player identity."""
    if rng.random() < 0.65:
        color = np.array([rng.uniform(25, 70), rng.uniform(70, 145), rng.uniform(20, 65)])
        out = np.broadcast_to(color, (h, w, 3)).copy()
        out += rng.normal(0, 9, (h, w, 1))
        if rng.random() < 0.4:
            yy = int(rng.uniform(0.2, 0.95) * h)
            out[max(0, yy - 1):yy + 2] = rng.uniform(160, 220)
    else:
        small = rng.uniform(20, 170, (12, 16, 3)).astype(np.float32)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 255).astype(np.uint8)


class BroadcastAugment:
    """Seeded, query-only soccer degradation; evaluation uses fixed seeds.

    Degradation is unavoidable (no clean-query branch). Severity controls resolution,
    compression, blur and nuisance probabilities. Geometry does not synthesize unseen
    3D views: perspective/crop jitter only approximates camera/crop variation.
    """

    def __init__(self, severity="strong"):
        if severity not in {"strong", "heavy"}:
            raise ValueError(severity)
        self.heavy = severity == "heavy"

    def __call__(self, image, masks, rng, donor=None, temporal_seed=None):
        # Bound work for large source photos; masks share exactly the same resize.
        scale = min(1.0, 384 / max(image.size))
        size = tuple(max(2, round(v * scale)) for v in image.size)
        a = np.array(image.resize(size, Image.Resampling.BICUBIC))
        h, w = a.shape[:2]
        masks = {k: cv2.resize(v.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                 for k, v in masks.items()}
        person = masks["person"]
        kits = kit_masks(a, masks)
        yy, xx = np.mgrid[:h, :w]
        for name, mask in kits.items():
            if temporal_seed is not None:
                rng = np.random.default_rng(seed_for(temporal_seed, "kit", name))
            if rng.random() < 0.85 and mask.any():
                color = rng.uniform(15, 245, 3)
                shade = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 180
                replacement = np.clip(color[None, None] * np.clip(shade[..., None], 0.45, 1.3), 0, 255)
                if rng.random() < 0.7:
                    spacing = max(2, int(rng.uniform(0.05, 0.18) * w))
                    mode = int(rng.integers(3))
                    stripes = ((xx if mode == 0 else yy if mode == 1 else xx + yy) // spacing) % 2 == 0
                    replacement[stripes] = rng.uniform(10, 245, 3)
                a[mask] = replacement[mask].astype(np.uint8)
            # Several small logo/number patches restricted to clothing, avoiding face.
            if mask.any() and rng.random() < 0.75:
                ys, xs = np.where(mask)
                for _ in range(int(rng.integers(1, 4))):
                    idx = int(rng.integers(len(xs)))
                    radius = max(1, int(w * rng.uniform(0.04, 0.12)))
                    patch = (abs(xx - xs[idx]) < radius) & (abs(yy - ys[idx]) < radius) & mask
                    a[patch] = np.median(a[mask], axis=0).astype(np.uint8)

        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "background"))
        bg = stadium(h, w, rng)
        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "background_blend"))
        if rng.random() < 0.85:
            alpha = cv2.GaussianBlur(person.astype(np.float32), (3, 3), 0.5)[..., None]
            a = np.clip(alpha * a + (1 - alpha) * bg, 0, 255).astype(np.uint8)
        else:
            a[~person] = np.clip(a[~person].astype(float) * rng.uniform(0.5, 1.5), 0, 255)

        # Smaller apparent player and modest perspective/translation; retain central head.
        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "geometry"))
        s = rng.uniform(0.60 if self.heavy else 0.70, 0.98)
        affine = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), s)
        affine[:, 2] += [rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.025, 0.025) * h]
        perspective = np.vstack([affine, [rng.uniform(-0.05, 0.05) / w,
                                         rng.uniform(-0.05, 0.05) / h, 1]])
        warped = cv2.warpPerspective(a, perspective, (w, h), borderValue=(40, 95, 35))
        valid = cv2.warpPerspective(np.ones((h, w), np.uint8), perspective, (w, h))
        a = np.where(valid[..., None] > 0, warped, bg)
        if rng.random() < 0.5:
            a = np.ascontiguousarray(a[:, ::-1])

        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "occlusion"))
        if rng.random() < (0.65 if self.heavy else 0.45):
            # Cropped donor player at the side; never cover the central head entirely.
            ow = max(2, int(w * rng.uniform(0.15, 0.34)))
            oh = max(2, int(h * rng.uniform(0.50, 0.85)))
            x = 0 if rng.random() < 0.5 else w - ow
            y = h - oh
            if donor is not None:
                di, dm = donor
                patch = np.array(di.resize((ow, oh), Image.Resampling.BILINEAR))
                alpha = cv2.resize(dm.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                # Fallback is explicitly a player-shaped silhouette, not an object box.
                patch = np.full((oh, ow, 3), rng.integers(15, 230, 3), np.uint8)
                alpha = np.zeros((oh, ow), np.uint8)
                cv2.ellipse(alpha, (ow // 2, max(1, oh // 9)), (max(1, ow // 5), max(1, oh // 9)), 0, 0, 360, 1, -1)
                cv2.rectangle(alpha, (ow // 5, oh // 4), (4 * ow // 5, oh - 1), 1, -1)
                alpha = alpha.astype(bool)
            view = a[y:y + oh, x:x + ow]
            view[alpha] = patch[alpha]

        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "photometric"))
        a = a.astype(np.float32) / 255
        gamma = rng.uniform(0.55, 1.6)
        a = np.power(np.clip(a, 0, 1), gamma) * rng.uniform(0.65, 1.4)
        a *= rng.uniform(0.80, 1.20, (1, 1, 3))
        a *= np.linspace(rng.uniform(0.65, 1.2), rng.uniform(0.75, 1.3), w)[None, :, None]
        gray = a.mean(-1, keepdims=True)
        a = gray + rng.uniform(0.3, 1.6) * (a - gray)
        a = np.clip(a * 255, 0, 255).astype(np.uint8)

        # At most source resolution: never invent detail by first enlarging a tiny crop.
        if temporal_seed is not None:
            rng = np.random.default_rng(seed_for(temporal_seed, "degradation"))
        low_h = min(h, int(rng.integers(20 if self.heavy else 28, 73 if self.heavy else 113)))
        low_w = max(2, min(w, round(w * low_h / h)))
        a = cv2.resize(a, (low_w, low_h), interpolation=cv2.INTER_AREA)
        use_gaussian = rng.random() < 0.7
        if use_gaussian:
            a = cv2.GaussianBlur(a, (3, 3), rng.uniform(0.35, 1.0 if self.heavy else 0.75))
        if rng.random() < 0.55 and (self.heavy or not use_gaussian):
            k = int(rng.choice([3, 5, 7] if self.heavy else [3, 5]))
            # Don't smear over the full width of an already tiny broadcast player.
            k = min(k, max(3, (low_w // 3) | 1))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2] = 1
            rot = cv2.getRotationMatrix2D((k // 2, k // 2), float(rng.uniform(0, 180)), 1)
            kernel = cv2.warpAffine(kernel, rot, (k, k))
            kernel /= max(float(kernel.sum()), 1e-8)
            a = cv2.filter2D(a, -1, kernel)
        a = np.clip(a.astype(float) + rng.normal(0, rng.uniform(1, 9), a.shape), 0, 255).astype(np.uint8)
        out = Image.fromarray(a)
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=int(rng.integers(12 if self.heavy else 20, 65)))
        buf.seek(0)
        with Image.open(buf) as jpeg:
            return jpeg.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
