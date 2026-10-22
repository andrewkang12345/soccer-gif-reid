from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionModel(nn.Module):
    """Exact GIF ProjectionModel, including parameter names for upstream weights."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 768), nn.BatchNorm1d(768),
        )

    def forward(self, x):
        return self.net(x)


class NumberAdapter(nn.Module):
    """Places frozen CLIP text anchors for shirt numbers into the projection space.

    This is not a second model and produces no second score. It gives each roster number a point
    in the one embedding space the image branch already uses, so an identity's prototype can be
    part photograph and part number while retrieval stays a single dot product.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.map = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(1 / .07).log())

    def forward(self, anchors):
        return F.normalize(self.map(anchors), dim=-1)


class ReIDModel(nn.Module):
    def __init__(self, visual, numbers=False):
        super().__init__()
        self.visual = visual.float()
        self.projection = ProjectionModel()
        self.number_adapter = NumberAdapter() if numbers else None
        self.unfrozen_blocks = 0
        self.set_trainable_blocks(0)

    def set_trainable_blocks(self, count):
        blocks = self.visual.transformer.resblocks
        if count < 0 or count > len(blocks):
            raise ValueError(f"Invalid final-block count: {count}")
        self.visual.requires_grad_(False)
        if count:
            for block in list(blocks)[-count:]:
                block.requires_grad_(True)
            self.visual.ln_post.requires_grad_(True)
            self.visual.proj.requires_grad_(True)
        self.unfrozen_blocks = count
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        if mode and self.unfrozen_blocks:
            for block in list(self.visual.transformer.resblocks)[-self.unfrozen_blocks:]:
                block.train()
        return self

    def encode_base(self, images):
        context = nullcontext() if self.unfrozen_blocks and self.training else torch.no_grad()
        with context:
            return F.normalize(self.visual(images).float(), dim=-1)

    def forward(self, images):
        base = self.encode_base(images)
        # Keep the small head/BN in FP32 in train, cached validation and inference.
        # CLIP still benefits from the caller's mixed precision context.
        with torch.autocast(device_type=images.device.type, enabled=False):
            return F.normalize(self.projection(base.float()), dim=-1)


def build_model(device="cpu", cache_dir=None, numbers=False):
    import clip
    base, _ = clip.load("ViT-L/14", device="cpu", jit=False, download_root=cache_dir)
    return ReIDModel(base.visual, numbers=numbers).to(device)


def from_checkpoint(path, device="cpu"):
    # Self-contained visual weights: inference requires no network or CLIP download.
    from clip.model import VisionTransformer
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("architecture") != "GIF-CLIP-ViT-L14-Proj":
        raise ValueError("Unsupported checkpoint architecture/format")
    if payload.get("preprocess") != "letterbox_clip_224_v1":
        raise ValueError("Unsupported checkpoint preprocessing")
    visual = VisionTransformer(224, 14, 1024, 24, 16, 768)
    model = ReIDModel(visual, numbers="number_adapter.map.weight" in payload["model"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
