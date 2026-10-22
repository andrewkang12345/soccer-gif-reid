from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.ap3d_comparison.prepare import windows
from experiments.ap3d_comparison.train import validate_clips
from experiments.ap3d_comparison.model import AP3DEncoder


def test_windows_do_not_stitch_gaps_tracks_or_duplicate_suffixes():
    paths = [Path(f"game_{f:06d}_1_g.jpg") for f in [0,4,8,12,100,104,108,112]]
    paths += [Path("game_000004_1_z.jpg")]
    paths += [Path(f"game_{f:06d}_2_g.jpg") for f in [0,4,8,12]]
    clips = windows(paths)
    assert len(clips) == 3
    assert all(len(c["frames"]) == len(set(c["frames"])) == 4 for c in clips)
    assert all(c["frames"][-1]-c["frames"][0] == 12 for c in clips)
    assert len({p for c in clips for p in c["paths"]}) == 12


def test_clip_validation_rejects_label_leakage():
    paths = [f"game_{f}_1_g.jpg" for f in range(0,16,4)]
    clip = dict(paths=paths, frames=list(range(0,16,4)), video="game", local_id="1", pid="3", split="train")
    rows = [dict(path=p, domain="query", pid="3", split="train") for p in paths]
    validate_clips(rows, [clip])
    rows[-1]["split"] = "val"
    with pytest.raises(ValueError, match="Mixed"):
        validate_clips(rows, [clip])


def test_ap3d_temporal_branch_learns_and_roundtrips(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(18)
    model = AP3DEncoder().train()
    x = torch.randn(2,4,3,64,32)
    branch = model.network.layer2[0].conv2.temporal_conv3d.weight
    before = branch.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    output = model(x)
    loss = (output * torch.randn_like(output)).sum()
    loss.backward()
    assert branch.grad is not None and torch.isfinite(branch.grad).all()
    optimizer.step()
    assert not torch.equal(branch, before)
    assert all(p.grad is None for p in model.network.layer1.parameters())
    model.eval()
    with torch.inference_mode():
        expected = model(x)
    torch.save(model.state_dict(), tmp_path/"state.pt")
    other = AP3DEncoder().eval()
    other.load_state_dict(torch.load(tmp_path/"state.pt", weights_only=True))
    with torch.inference_mode():
        torch.testing.assert_close(other(x), expected)
    assert expected.shape == (2,2048) and torch.isfinite(expected).all()
