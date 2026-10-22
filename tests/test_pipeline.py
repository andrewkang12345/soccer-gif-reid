import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from PIL import Image
from pytorch_metric_learning.losses import CircleLoss

from soccer_gif_reid.augment import BroadcastAugment, kit_masks, preprocess
from soccer_gif_reid.common import read_rgb, write_manifest
from soccer_gif_reid.data import EvalDataset, IdentityBatchSampler, PairDataset
from soccer_gif_reid.infer import make_prototypes, rank_tracklet, roster_images
from soccer_gif_reid.metrics import retrieval_metrics
from soccer_gif_reid.model import ProjectionModel, ReIDModel
from soccer_gif_reid.prepare import audit, holdout_train_identities
from soccer_gif_reid.train import can_promote, cross_domain_circle, load_config


@pytest.fixture
def rows(tmp_path):
    rows = []
    for pid in range(1, 7):
        split = "train" if pid < 5 else "val"
        for domain in ("query", "reference"):
            for sample in range(2):
                rng = np.random.default_rng(pid * 100 + sample + (10 if domain == "reference" else 0))
                image = rng.integers(0, 255, (96, 48, 3), dtype=np.uint8)
                p = tmp_path / f"{pid}_{domain}_{sample}.png"
                Image.fromarray(image).save(p)
                r = {"pid": str(pid), "path": str(p), "split": split, "domain": domain}
                if domain == "query":
                    m = np.zeros((96, 48), np.uint8)
                    m[5:92, 8:40] = 255
                    mp = tmp_path / f"mask_{pid}_{sample}.png"
                    Image.fromarray(m).save(mp)
                    r["person_mask"] = str(mp)
                rows.append(r)
    return rows


def test_metric_hand_computed_ap_and_denominators():
    # Relevant gallery images at ranks 1 and 3 -> AP = (1 + 2/3)/2.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    g = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]])
    result = retrieval_metrics(q, g, ["a", "b"], ["a", "b", "a"], chunk_size=1)
    assert result["rank1"] == 0.5
    assert result["mAP"] == pytest.approx(((1 + 2 / 3) / 2 + 0.5) / 2)
    assert result["queries"] == 2


def test_metric_missing_positive_and_nonfinite_fail():
    with pytest.raises(ValueError, match="without reference"):
        retrieval_metrics(torch.ones(1, 2), torch.ones(1, 2), ["a"], ["b"])
    with pytest.raises(ValueError, match="nonfinite"):
        retrieval_metrics(torch.tensor([[float("nan"), 1]]), torch.ones(1, 2), ["a"], ["a"])


def test_identity_split_moves_query_and_reference_together(rows):
    moved = holdout_train_identities(rows, 0.5, 17)
    audit(moved)
    moved2 = holdout_train_identities(list(reversed(rows)), 0.5, 17)
    assert {(r["path"], r["split"]) for r in moved} == {(r["path"], r["split"]) for r in moved2}


def test_audit_rejects_identity_leak(rows):
    rows[0]["split"] = "val"
    with pytest.raises(ValueError, match="Identity leakage"):
        audit(rows)


def test_audit_rejects_copied_query_as_reference(rows):
    q = next(r for r in rows if r["domain"] == "query")
    g = next(r for r in rows if r["domain"] == "reference")
    Path(g["path"]).write_bytes(Path(q["path"]).read_bytes())
    with pytest.raises(ValueError, match="Identical bytes"):
        audit(rows)


def test_missing_reference_is_not_replaced_with_query(rows):
    rows = [r for r in rows if not (r["pid"] == "1" and r["domain"] == "reference")]
    with pytest.raises(ValueError, match="lacks a query or clean reference"):
        audit(rows)


def test_pairs_are_asymmetric_and_repeatable(rows):
    data = PairDataset(rows)
    q, g, label = data[(0, 3)]
    q2, g2, label2 = data[(0, 3)]
    assert torch.equal(q, q2) and torch.equal(g, g2) and label == label2
    refs = [preprocess(read_rgb(r["path"])) for r in rows if r["pid"] == "1" and r["domain"] == "reference"]
    assert any(torch.equal(g, ref) for ref in refs)
    assert not torch.equal(q, preprocess(read_rgb(rows[0]["path"])))
    data.epoch = 1
    assert not torch.equal(q, data[(0, 3)][0])


def test_validation_corruption_is_frozen(rows):
    qs = [r for r in rows if r["domain"] == "query"]
    first = EvalDataset(qs, True, 10)
    second = EvalDataset(qs, True, 10)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[0], EvalDataset(qs, True, 11)[0])
    assert torch.equal(EvalDataset(qs)[0], preprocess(read_rgb(qs[0]["path"])))


def test_sampler_has_positives_negatives_and_deterministic_restart(rows):
    data = PairDataset(rows)
    sampler = IdentityBatchSampler(data, identities=3, queries_per_identity=2, steps=3)
    batches = list(sampler)
    assert batches == list(sampler)
    for batch in batches:
        labels = [data.queries[i]["pid"] for i, _ in batch]
        assert len(set(labels)) == 3 and all(labels.count(label) == 2 for label in labels)
    data.epoch = 1
    assert list(sampler) != batches


@pytest.mark.parametrize("height,width", [(2, 2), (21, 9), (96, 48), (512, 128)])
def test_augmentation_handles_low_resolution(height, width):
    im = Image.new("RGB", (width, height), (100, 80, 160))
    masks = {"person": np.ones((height, width), bool)}
    out = BroadcastAugment("heavy")(im, masks, np.random.default_rng(1))
    assert out.mode == "RGB" and torch.isfinite(preprocess(out)).all()


def test_kit_regions_exclude_head_and_background():
    im = np.full((100, 50, 3), (20, 50, 210), dtype=np.uint8)
    person = np.zeros((100, 50), bool)
    person[:, 10:40] = True
    masks = kit_masks(im, {"person": person})
    for mask in masks.values():
        assert not mask[:23].any() and not mask[:, :10].any()
        assert mask.any()


def test_cross_domain_circle_has_both_branch_gradients():
    torch.manual_seed(1)
    query = torch.randn(4, 32, requires_grad=True)
    reference = torch.randn(4, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = cross_domain_circle(CircleLoss(), query, reference, labels)
    loss.backward()
    assert torch.isfinite(loss) and query.grad.norm() > 0 and reference.grad.norm() > 0


class TinyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
        self.ln_post = torch.nn.LayerNorm(3)
        self.proj = torch.nn.Parameter(torch.randn(3, 768))

    def forward(self, x):
        return self.ln_post(self.transformer.resblocks(x.mean((-1, -2)))) @ self.proj


def test_frozen_clip_and_optional_final_blocks():
    model = ReIDModel(TinyVisual()).train()
    before = {k: v.clone() for k, v in model.visual.state_dict().items()}
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=1e-3)
    features = model(torch.randn(8, 3, 12, 12))
    loss = cross_domain_circle(CircleLoss(), *features.chunk(2), torch.tensor([0, 0, 1, 1]))
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in model.visual.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.visual.state_dict().items())
    model.set_trainable_blocks(1)
    assert not any(p.requires_grad for p in model.visual.transformer.resblocks[0].parameters())
    assert all(p.requires_grad for p in model.visual.transformer.resblocks[1].parameters())
    model.zero_grad()
    model(torch.randn(8, 3, 12, 12)).square().mean().backward()
    assert model.visual.proj.grad is not None


def test_cached_validation_agrees_with_uncached(rows):
    from soccer_gif_reid.evaluate import FrozenValidation, evaluate
    model = ReIDModel(TinyVisual()).eval()
    cache = FrozenValidation(model, rows, "cpu", batch_size=4, workers=0)
    assert cache.evaluate(model) == evaluate(model, rows, "cpu", batch_size=4, workers=0)
    model.set_trainable_blocks(1)
    with pytest.raises(ValueError, match="unfrozen"):
        cache.evaluate(model)


def test_projection_is_upstream_parameter_compatible():
    model = ProjectionModel()
    assert model.net[0].weight.shape == (512, 768)
    assert model.net[3].weight.shape == (768, 512)
    assert set(model.state_dict()) == {"net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", "net.1.running_mean",
                                      "net.1.running_var", "net.1.num_batches_tracked", "net.3.weight", "net.3.bias", "net.4.weight",
                                      "net.4.bias", "net.4.running_mean", "net.4.running_var", "net.4.num_batches_tracked"}


def test_multi_reference_inference_and_tracklet_rejection():
    ids, prototypes = make_prototypes(torch.tensor([[1., 0], [1., 0], [0., 1]]), ["a", "a", "b"])
    result = rank_tracklet(torch.tensor([[1., 0], [0.8, 0.6]]), prototypes, ids)
    assert result["ranked_identities"][0]["identity"] == "a"
    assert result["ranked_identities"][0]["similarity"] == pytest.approx(0.9)
    assert result["frames_used"] == 2 and result["frame_top1_agreement"] == 1
    assert rank_tracklet(torch.ones(1, 2), prototypes, ids, min_margin=0.2)["prediction"] == "unknown"


def test_inference_rejects_invalid_thresholds():
    for kwargs in ({"min_similarity": float("nan")}, {"min_similarity": 1.1},
                   {"min_margin": -0.1}, {"min_margin": float("inf")}):
        with pytest.raises(ValueError, match="must be finite"):
            rank_tracklet(torch.ones(1, 2), torch.ones(2, 2), ["a", "b"], **kwargs)


def test_tight_crop_segmentation_retry_uses_predicted_mask():
    from soccer_gif_reid.masks import generate_person_mask
    class Predictor:
        calls = 0
        def predict(self, im, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [SimpleNamespace(masks=None)]
            # Original 20x40 image padded left/right=8, top/bottom=8.
            polygon = np.array([[12, 10], [24, 10], [24, 46], [12, 46]])
            return [SimpleNamespace(masks=SimpleNamespace(xy=[polygon]), boxes=SimpleNamespace(
                xyxy=torch.tensor([[12., 10, 24, 46]]), conf=torch.tensor([0.9])))]
    predictor = Predictor()
    mask, method = generate_person_mask(predictor, Image.new("RGB", (20, 40)), "cpu")
    assert method == "padded" and predictor.calls == 2
    assert mask.shape == (40, 20) and mask[20, 10] == 255 and mask[20, 0] == 0


def test_changed_data_content_is_detected(rows):
    from soccer_gif_reid.common import digest_file
    rows[0]["sha256"] = digest_file(rows[0]["path"])
    Image.new("RGB", (48, 96)).save(rows[0]["path"])
    with pytest.raises(ValueError, match="Image changed"):
        audit(rows)


def test_invalid_training_learning_rate_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("lr: .nan\n")
    with pytest.raises(ValueError, match="positive and finite"):
        load_config(path)


def test_finetune_needs_improvement_and_clean_retention():
    anchor = {"clean": {"mAP": 0.7, "rank1": 0.8}, "degraded": {"mAP": 0.5}, "selection_score": 0.58}
    metrics = copy.deepcopy(anchor)
    metrics["selection_score"] = 0.65
    metrics["degraded"]["mAP"] = 0.6
    cfg = {"max_clean_map_drop": 0.02, "max_clean_rank1_drop": 0.02}
    assert can_promote(metrics, 0.58, anchor, cfg)
    metrics["clean"]["mAP"] = 0.65
    assert not can_promote(metrics, 0.58, anchor, cfg)


def test_roster_rejects_ambiguous_layout(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    (tmp_path / "b").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "b" / "photo.jpg")
    with pytest.raises(ValueError, match="do not mix"):
        roster_images(tmp_path)


def test_train_epoch_checkpoint_resume_and_validation(tmp_path, rows, monkeypatch):
    import soccer_gif_reid.train as trainer
    import soccer_gif_reid.evaluate as evaluator
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    cfg = {**trainer.DEFAULTS, "manifest": str(manifest), "output": str(tmp_path / "run"), "device": "cpu",
           "epochs": 2, "steps_per_epoch": 1, "identities_per_batch": 2, "queries_per_identity": 1,
           "workers": 0, "eval_batch_size": 4, "warmup_steps": 0, "amp": False}
    monkeypatch.setattr(trainer, "build_model", lambda *a: ReIDModel(TinyVisual()))
    evaluate_original = evaluator.evaluate
    checkpoints = []
    def fake_load(path, device):
        state = torch.load(path, weights_only=True)
        model = ReIDModel(TinyVisual())
        model.load_state_dict(state["model"])
        return model.to(device), state
    monkeypatch.setattr(trainer, "from_checkpoint", fake_load)
    write_original = trainer.atomic_write
    def capture(path, writer):
        write_original(path, writer)
        if Path(path).name == "last.pt":
            checkpoints.append(Path(path).read_bytes())
    monkeypatch.setattr(trainer, "atomic_write", capture)
    trainer.train(cfg)
    first_final = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert first_final["epoch"] == 1
    assert (tmp_path / "run" / "best.pt").is_file()
    assert set(first_final["metrics"]) >= {"clean", "degraded"}
    # Resume from epoch 1 in the same directory and compare the entire final model.
    path = tmp_path / "run" / "last.pt"
    path.write_bytes(checkpoints[0])
    trainer.train(cfg, path)
    second_final = torch.load(path, weights_only=True)
    assert all(torch.equal(first_final["model"][k], v) for k, v in second_final["model"].items())
