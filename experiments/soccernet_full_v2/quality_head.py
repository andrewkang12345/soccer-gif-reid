"""A tracklet is one embedding: frames weighted by a quality score the identity loss discovers.

    z_t     = f(x_t)                      the existing encoder, unchanged, per crop
    q_t     = W2 GELU(W1 z_t)             768 -> 128 -> 1, about 100k parameters
    w       = softmax(q / tau)
    z_track = normalize( sum_t w_t normalize(z_t) )

Nothing labels a frame good or bad. If leaning on a blurred crop hurts retrieval the gradient
lowers its score, and if a clear three-quarter view helps, its score rises. That is the whole
reason this is one model rather than a quality classifier bolted to the side: there is no second
objective and no logits to fuse by hand.

Two failure modes are guarded explicitly. Softmax attention will happily collapse onto a single
frame, so the temperature starts near 2 and an entropy term pushes back on peaked weights. And
aggregation can paper over rotten per-frame embeddings, so the loss keeps a per-frame term
alongside the tracklet term: `L = L_frame + lambda * L_track`.
"""
import torch
from torch import nn
from torch.nn import functional as F


class QualityHead(nn.Module):
    def __init__(self, dim=768, hidden=128, temperature=2.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        # Start every frame equal: a head that begins opinionated biases the first epochs.
        nn.init.zeros_(self.net[2].weight); nn.init.zeros_(self.net[2].bias)

    def weights(self, frames, mask=None):
        """frames [B, T, D] -> [B, T] softmax weights; `mask` marks padded slots."""
        scores = self.net(frames).squeeze(-1) / self.log_temperature.exp().clamp(.25, 10.)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return torch.softmax(scores, dim=-1)

    def forward(self, frames, mask=None):
        """Weighted mean of the normalized frame embeddings, renormalized."""
        normalized = F.normalize(frames, dim=-1)
        w = self.weights(frames, mask)
        return F.normalize((w.unsqueeze(-1) * normalized).sum(1), dim=-1), w


def entropy_penalty(w, mask=None):
    """Push back on weights that collapse onto one frame; zero when they are uniform."""
    count = (mask.sum(-1) if mask is not None else torch.full(w.shape[:1], w.shape[1], device=w.device)).clamp(min=1)
    entropy = -(w.clamp_min(1e-9).log() * w).sum(-1)
    return (count.float().log() - entropy).clamp_min(0).mean()
