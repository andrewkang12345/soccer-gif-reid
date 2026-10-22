from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.ap3d_no_quality_v2 import augment as native
from experiments.ap3d_no_quality_v2.data import ClipEvaluation, render_frames
from experiments.ap3d_no_quality_v2.evaluate import selection_score


def test_native_augmentation_never_blurs_rgb_or_encodes_jpeg(monkeypatch):
    rgb = np.random.default_rng(10).integers(0, 255, (600, 300, 3), dtype=np.uint8)
    mask = np.zeros((600, 300), bool); mask[10:580, 20:280] = True
    gaussian = cv2.GaussianBlur

    def only_mask_blending(value, *args, **kwargs):
        assert value.ndim == 2, "RGB image must not be blurred"
        return gaussian(value, *args, **kwargs)

    def reject(*args, **kwargs):
        raise AssertionError("Lossy compression or motion blur must not run")

    monkeypatch.setattr(cv2, "GaussianBlur", only_mask_blending)
    monkeypatch.setattr(cv2, "filter2D", reject)
    monkeypatch.setattr(Image.Image, "save", reject)
    for seed in range(8):
        result = native.BroadcastAugment()(Image.fromarray(rgb), {"person": mask},
                                            np.random.default_rng(seed), temporal_seed=seed)
        assert result.size == (300, 600)  # No old 384px cap or 28–112px reduction.
        assert result.mode == "RGB"


def test_native_augmentation_preserves_player_texture(monkeypatch):
    # Isolate the player's texture from background, clothing, and geometric changes.
    # A former downsample/blur/JPEG/noise or exposure operation would change it.
    y, x = np.mgrid[:128, :64]
    rgb = np.repeat((((x // 2 + y // 2) % 2) * 160 + 40)[..., None], 3, axis=2).astype(np.uint8)
    mask = np.zeros((128, 64), bool); mask[2:-2, 2:-2] = True
    monkeypatch.setattr(native, "kit_masks", lambda rgb, masks: {})
    monkeypatch.setattr(cv2, "warpPerspective", lambda value, *args, **kwargs: value.copy())
    class Fixed:
        def random(self): return .99  # No occluder or flip; choose unchanged background.
        def uniform(self, low, high=None, size=None):
            value = 1.0 if low <= 1.0 <= high else (low + high) / 2
            return np.full(size, value) if size else value
    monkeypatch.setattr(native, "stadium", lambda h, w, rng: np.zeros((h, w, 3), np.uint8))
    result = np.asarray(native.BroadcastAugment()(Image.fromarray(rgb), {"person": mask}, Fixed()))
    np.testing.assert_array_equal(result[mask], rgb[mask])


def test_evaluation_uses_native_augmentation_and_keeps_references_clean(tmp_path, monkeypatch):
    im = Image.fromarray(np.full((80, 40, 3), 120, np.uint8))
    path=tmp_path/'image.png'; im.save(path)
    mask=np.zeros((80,40),np.uint8);mask[2:-2,2:-2]=255
    mp=tmp_path/'mask.png';Image.fromarray(mask).save(mp)
    row=dict(path=str(path), person_mask=str(mp))
    clip=dict(paths=[str(path)]*4, clip_id='fixture')
    calls=[]
    def augment(self, image, masks, rng, donor=None, temporal_seed=None):
        calls.append(temporal_seed);return image
    monkeypatch.setattr(native.BroadcastAugment, '__call__', augment)
    original=ClipEvaluation([row],[clip],'clip')[0]
    augmented=ClipEvaluation([row],[clip],'clip',augmented=True)[0]
    assert len(calls)==4 and len(set(calls))==1
    torch.testing.assert_close(original,augmented)
    ClipEvaluation([row],[row],'clip',reference=True,augmented=True)[0]
    assert len(calls)==4  # Reference bypasses augmentation regardless of query flag.


def test_selection_uses_new_validation_conditions():
    assert selection_score({'clean':{'mAP':.2},'augmented':{'mAP':.1}})==pytest.approx(2*.2*.1/.3)
    assert selection_score({'clean':{'mAP':0.},'augmented':{'mAP':0.}})==0.
