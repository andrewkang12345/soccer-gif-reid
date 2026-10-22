# MuPNIT comparison with no added visual-quality degradation

User correction on 2026-09-12: remove the artificial reduction in image quality.
This isolated experiment restarts both GIF CLIP-Proj and AP3D from the same original
pretrained weights used by v1. Previous code, checkpoints and reports are preserved.

## Augmentation change

Removed from both training and augmented validation:

- Downsample–upsample degradation and the extra 384-pixel preprocessing cap.
- Gaussian/motion blur on the player image, JPEG recompression, and pixel noise.
- Artificial player shrinking and aggressive exposure/gamma/saturation shifts.

Retained on queries: jersey/shorts recoloring, kit patterns, logo/number masking,
background replacement/perturbation, modest affine/perspective shifts, flips and
player-like occlusion. Clip seeds maintain consistent appearance choices across
frames. These operations and the normal model input resize still use interpolation;
this does not restore detail missing from the original broadcast crops. Reference
images bypass augmentation. Original source files are not rewritten.

See the [original / old / new / clean-reference preview](../../reports/ap3d_no_quality_v2/augmentation_preview.jpg).

## Training and validation

The same v1 manifest, source images, masks, reference pool, splits and four-frame
clips are reused: 3,413 train / 323 validation / 391 test clips; 118 / 42 / 51
identity-disjoint players; 1,685 references. No additional filtering or data selection.
Reference labels remain archive-provided and are not human-certified.

CLIP uses the frozen ViT-L/14 backbone and GIF projection head; AP3D uses ImageNet
ResNet50 initialization with temporal/stages 2–4 adaptation. Both use the same
cross-domain CircleLoss, batches, learning rates and schedule as v1. The new
configuration explicitly requires `quality_degradation: false` and
`augmentation_profile: appearance_only_v2`.

Validation now reports **clean** and **augmented** conditions. The latter has
appearance augmentation at native source quality. It is a different condition
from v1's heavily **degraded** validation, so those mAP values are not comparable
as if the test images were unchanged. Clean validation remains directly comparable.
Select checkpoints by the harmonic mean of clean/augmented validation mAP.

The queue runs two epochs each first, then continues up to 30 epochs each with
patience 8. It evaluates validation-selected winners on test afterward. This same
test subset was previously evaluated by v1; it is not newly unseen test data. Test
images/metrics never enter optimization or checkpoint selection.

```bash
cd /mnt/data/soccer_gif_reid
.venv/bin/python -m experiments.ap3d_no_quality_v2.run_queue
```

Check existing queue processes before running again. Checkpoints include optimizer
state and code/data/configuration hashes. Do not edit executable pipeline code
while this queue is running. The new checkpoint format is `mupnit-native-quality-v2`.

- [Live results](../../reports/ap3d_no_quality_v2/progress_report.md)
- [Queue status](../../reports/ap3d_no_quality_v2/queue_status.json)
- [Configuration](../../configs/ap3d_no_quality_v2.yaml)
- Logs: `reports/ap3d_no_quality_v2/{clip,ap3d}_training.log`
- Checkpoints: `runs/ap3d_no_quality_v2/{clip,ap3d}/{best,last}.pt`

## Roster inference

Put reference photos in `roster/<identity>/*.jpg`. Use the matching v2 loader:

```bash
.venv/bin/python -m experiments.ap3d_no_quality_v2.infer \
  --checkpoint runs/ap3d_no_quality_v2/clip/best.pt \
  --roster /path/to/roster --query /path/to/player_crop.jpg

.venv/bin/python -m experiments.ap3d_no_quality_v2.infer \
  --checkpoint runs/ap3d_no_quality_v2/ap3d/best.pt \
  --roster /path/to/roster --query /path/to/tracklet --top-k 5
```

Scores are cosine similarities, not calibrated identity probabilities. No soccer
production accuracy or commit threshold is established by this experiment.
