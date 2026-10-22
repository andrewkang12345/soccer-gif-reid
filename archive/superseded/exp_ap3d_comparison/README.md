# Augmented MuPNIT: GIF CLIP-Proj vs AP3D

This is a separate, matched four-frame experiment. It trains directly on augmented
broadcast queries paired with clean web photographs. It does **not** reproduce
GIF's unaugmented training or claim soccer production accuracy.

Run commands from `/mnt/data/soccer_gif_reid` using `.venv/bin/python`.

## Protocol

- Full MuPNIT-ReID archive; an initial subset, not every available crop.
- 118 training identities, 42 identities exclusive to validation, 51 reserved test
  identities. The archive's 91 overlapping validation identities are excluded.
- At most 32 training clips / 8 validation or test clips per identity. Each clip has
  four distinct, ordered crops from the same source video and local track ID.
  Source footage is 60 fps. Windows contain at most 16 consecutive available crops;
  available-frame gaps over 8 frames split windows. No duplicate suffix variants,
  repeated query frames, overlapping windows, or merged web-photo sequences.
- Up to eight automatically screened reference images per identity. Reference
  pixels stay clean; preprocessing preserves aspect ratio. Screening rejects
  exact duplicate/ambiguous image bytes, tiny/blurred images, and multiple prominent
  people. **It does not certify semantic identity labels or remove watermarks.**
- Actual segmentation masks drive query-only kit recoloring, patterns, logo/number
  masking, background replacement, smaller scale, crop/perspective shifts,
  player-shaped occlusion, lighting/color shifts, blur, downsampling, JPEG and noise.
  Clip seeds are separated by augmentation stage so changing crop size does not
  randomly flip geometry or kit color between adjacent frames. This is approximate
  temporal consistency, not optical-flow-tracked clothing masks.
- Any clip with a failed mask is excluded **before** either model runs. Counts and
  exact rejected clips are saved. Metrics describe the retained subset.
- Camera cuts are not explicitly detected. Short uninterrupted source-ID windows
  are an approximation of deployment tracklets, not McByte++ output.

## Models and objective

**CLIP-Proj:** official pretrained ViT-L/14, frozen, with GIF's exact
768→512→768 BatchNorm/ReLU projection head. Normalize per-frame embeddings, then
average the same four observations provided to AP3D. CLIP uses 224×224 letterboxing.
One clean reference is encoded once.

**AP3D:** GIF's vendored AP3DResNet50 / APP3DC implementation. ImageNet1K V1
ResNet50 spatial initialization (`resnet50-0676ba61.pth`); no trained GIF/AP3D
ReID checkpoint was located in the checked public repositories. Stem and first
residual stage are frozen; stages 2–4, temporal modules and embedding BatchNorm
are trained. AP3D uses 256×128 letterboxing, spatial max pooling and temporal mean
pooling. A single clean reference photo is repeated as a static four-frame clip.

Both use bidirectional cross-domain CircleLoss (margin .4, gamma 80), identical
identity sampling and augmentation seeds. AP3D's original classifier/CE+triplet
training is replaced by this shared metric objective, and its embedding BatchNorm
is used consistently in train/eval. Thus **this is an adaptation comparison,
not the published AP3D recipe**. Pretraining, input geometry, learning rates and
number of trainable parameters differ; it is not a controlled architecture-only
ablation. Original files, hashes, license and compatibility changes are in `upstream/`.

## Run and monitor

Configuration: `configs/ap3d_comparison_v1.yaml` (30 epochs, 500 steps/epoch,
4 identities per batch, fresh augmentation each sample, patience 8).

```bash
.venv/bin/python -m experiments.ap3d_comparison.prepare
.venv/bin/python -m experiments.ap3d_comparison.run_queue
```

The serial queue first runs two epochs of CLIP-Proj and two of AP3D, then continues
both to the epoch/patience limit. `last.pt` permits resumption; configuration,
source code, manifests and image/mask contents are fingerprinted and checked.
Do not change executable pipeline files while this queue is running.

- `reports/ap3d_comparison_v1/queue_status.json`: active phase and process IDs.
- `reports/ap3d_comparison_v1/{clip,ap3d}_training.log`: step/validation/error logs.
- `runs/ap3d_comparison_v1/{clip,ap3d}/progress.json`: current epoch, step and loss.
- `runs/ap3d_comparison_v1/{clip,ap3d}/history.jsonl`: per-epoch validation metrics.
- `runs/ap3d_comparison_v1/{clip,ap3d}/{best,last}.pt`: self-contained checkpoints.
- `reports/ap3d_comparison_v1/data_preparation.json`: retained/excluded counts.

Rank-1/mAP are measured against **individual** clean gallery images, with all
same-identity references relevant. Report both original and fixed heavily degraded
validation queries. Select the best checkpoint by the harmonic mean of the two
mAP values. The pretrained CLIP comparison is **evaluation only**. Test data never
enters optimization or checkpoint selection; the queue evaluates each validation
winner on the reserved test subset once after training finishes.

## Inference

Put clean images in `roster/<identity>/*.jpg`. Identity may be a player name or
your application ID. For this new checkpoint format use:

```bash
.venv/bin/python -m experiments.ap3d_comparison.infer \
  --checkpoint runs/ap3d_comparison_v1/ap3d/best.pt \
  --roster /path/to/roster --query /path/to/player_crop.jpg

.venv/bin/python -m experiments.ap3d_comparison.infer \
  --checkpoint runs/ap3d_comparison_v1/clip/best.pt \
  --roster /path/to/roster --query /path/to/ordered_tracklet_directory \
  --top-k 5 --output reports/tracklet_ranking.json
```

Returns ranked identities and cosine similarity scores, **not calibrated
probabilities**. Inference averages clean reference embeddings per identity;
this roster-prototype ranking differs from the image-gallery mAP protocol.
AP3D repeats a singleton crop, which supplies no new temporal evidence. Tracklet
input should contain one player within one uninterrupted track. No soccer roster
has been configured and no production commit threshold has been calibrated.

## Published comparison

GIF Table 4 uses **MuPNIT-ReID-light**, a different protocol:

| Model | mAP | Rank-1 |
|---|---:|---:|
| Plain CLIP | 13.5% | 18.6% |
| CLIP-Proj | 25.3% | 32.2% |
| CLIP-Proj + text | 33.6% | 46.8% |
| AP3D | 48.1% | 30.3% |

AP3D leads image-only CLIP-Proj on mAP, but not Rank-1. These are published results,
not our measurements. Source: [GIF WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.pdf).
AP3D architecture and official code: [paper](https://arxiv.org/abs/2007.08434),
[repository](https://github.com/guxinqian/AP3D).
