"""Augmentation that removes the two shortcuts the R3 error sheets exposed.

R1 and R3 both randomised the kit of the query and left the reference untouched, so a training
pair stayed solvable by colour, and the error sheets showed exactly that: every Roma-kit query
returning a Roma-kit reference. Here the reference is recoloured too, so colour cannot survive as
a shared cue.

The second shortcut is the shirt number. Identity labels in this dataset come from a readable
number, so the data is biased towards frames where one is visible and the model can simply read
it. The previous kit recolour preserved the digits' luminance pattern; filling the clothing flat
erases them. Both are applied as probabilities rather than always, so the model still sees the
real cue some of the time and the two regimes can be compared.
"""
import cv2
import numpy as np
from PIL import Image
from experiments.ap3d_no_quality_v2.augment import BroadcastAugment, kit_masks


def recolour(a, mask, rng, flat):
    """Repaint one clothing region. `flat` drops the luminance shading, taking the digits with it."""
    colour = rng.uniform(15, 245, 3)
    if flat:
        a[mask] = colour.astype(np.uint8)
        return
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    shade = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 180
    replacement = np.clip(colour[None, None] * np.clip(shade[..., None], 0.45, 1.3), 0, 255)
    if rng.random() < 0.7:
        spacing = max(2, int(rng.uniform(0.05, 0.18) * w))
        mode = int(rng.integers(3))
        stripes = ((xx if mode == 0 else yy if mode == 1 else xx + yy) // spacing) % 2 == 0
        replacement[stripes] = rng.uniform(10, 245, 3)
    a[mask] = replacement[mask].astype(np.uint8)


class QueryAugment(BroadcastAugment):
    """Broadcast augmentation plus a chance of erasing the shirt number outright.

    `erased` records what happened to the last crop. The number channel is trained on the shirt
    number that is actually in the picture, so a crop whose digits this augmentation has just
    painted over must not also be asked to predict them -- that teaches invention, not reading.
    The probability is a real trade-off: erasing stops the identity loss from reading the number
    as a shortcut, and not erasing is what leaves the number legible at all.
    """

    def __init__(self, severity="strong", flat_kit_probability=0.5):
        super().__init__(severity)
        self.flat_kit_probability = flat_kit_probability
        self.erased = False

    def __call__(self, image, masks, rng, donor=None, temporal_seed=None):
        self.erased = rng.random() < self.flat_kit_probability
        if self.erased:
            a = np.array(image.convert("RGB"), copy=True)
            h, w = a.shape[:2]
            sized = {k: cv2.resize(v.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                     for k, v in masks.items()}
            for mask in kit_masks(a, sized).values():
                if mask.any():
                    recolour(a, mask, rng, flat=True)
            image = Image.fromarray(a)
        return super().__call__(image, masks, rng, donor, temporal_seed)


class ReferenceAugment:
    """Kit-only augmentation for the clean reference photograph.

    The reference keeps its own background, pose, framing and photographic quality — it is a
    studio or press picture and should stay one. Only the clothing is repainted, so that kit
    colour cannot be the thing that links a pair. A reference is a padded person crop, so the
    clothing bands are taken from crop geometry rather than a segmentation mask.
    """

    def __init__(self, probability=0.7, flat_probability=0.5):
        self.probability, self.flat_probability = probability, flat_probability

    def __call__(self, image, rng):
        if rng.random() >= self.probability:
            return image
        a = np.array(image.convert("RGB"), copy=True)
        person = np.ones(a.shape[:2], bool)
        flat = rng.random() < self.flat_probability
        for mask in kit_masks(a, {"person": person}).values():
            if mask.any():
                recolour(a, mask, rng, flat=flat)
        return Image.fromarray(a)


class CropAugment:
    """Augmentation for an unlabelled broadcast crop, which has no segmentation mask.

    Self-supervised views come straight out of the SoccerNet archives with no person mask, so the
    clothing regions are taken from crop geometry. Kit colour and the shirt number are the two
    cues a contrastive objective would otherwise latch onto immediately — two views of one player
    in one action share both exactly — so both are attacked here, and the geometric jitter stops
    the pair being matched on framing alone.
    """

    def __init__(self, kit_probability=0.8, flat_probability=0.5):
        self.kit_probability, self.flat_probability = kit_probability, flat_probability

    def __call__(self, image, rng):
        a = np.array(image.convert("RGB"), copy=True)
        if rng.random() < self.kit_probability:
            person = np.ones(a.shape[:2], bool)
            flat = rng.random() < self.flat_probability
            for mask in kit_masks(a, {"person": person}).values():
                if mask.any():
                    recolour(a, mask, rng, flat=flat)
        h, w = a.shape[:2]
        affine = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.92, 1.08))
        affine[:, 2] += [rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h]
        a = cv2.warpAffine(a, affine, (w, h), borderMode=cv2.BORDER_REPLICATE)
        a = np.clip(a.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            a = np.ascontiguousarray(a[:, ::-1])
        return Image.fromarray(a)
