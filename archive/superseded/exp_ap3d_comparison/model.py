"""GIF's AP3DResNet50 with a consistent train/eval metric embedding."""
import torch
from torch import nn
from torch.nn import functional as F

from .upstream.ResNet import AP3DResNet50


class AP3DEncoder(nn.Module):
    def __init__(self, spatial_weights=None):
        super().__init__()
        self.network = AP3DResNet50(1, spatial_weights=spatial_weights)
        # Cross-domain CircleLoss has no training-ID classifier.
        self.network.classifier = nn.Identity()
        # Freeze stem and first residual stage; learn temporal blocks and stages 2–4.
        for name in ("conv1", "bn1", "layer1"):
            getattr(self.network, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for name in ("conv1", "bn1", "layer1"):
            getattr(self.network, name).eval()
        return self

    def forward(self, clips):
        # B,T,C,H,W -> B,C,T,H,W. Keep upstream spatial-max / temporal-mean.
        n = self.network
        x = clips.permute(0, 2, 1, 3, 4)
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        x = x.amax(dim=(-1, -2)).mean(dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.normalize(n.bn(x.float()), dim=-1)


def encode_clip(model, kind, clips):
    if kind == "ap3d":
        return model(clips)
    b, t, c, h, w = clips.shape
    # Same four genuine observations as AP3D; normalize each frame before pooling.
    return F.normalize(model(clips.reshape(b*t, c, h, w)).reshape(b, t, -1).mean(1), dim=-1)


def encode_pair(model, kind, query, gallery):
    if kind == "ap3d":
        # A single clean photo is a static clip; never mix different web images.
        repeated = gallery[:, None].expand(-1, query.shape[1], -1, -1, -1)
        return model(torch.cat([query, repeated])).chunk(2)
    b, t, c, h, w = query.shape
    features = model(torch.cat([query.reshape(b*t, c, h, w), gallery]))
    return F.normalize(features[:b*t].reshape(b, t, -1).mean(1), dim=-1), features[b*t:]
