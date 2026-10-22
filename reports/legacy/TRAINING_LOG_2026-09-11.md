# Training session — 2026-09-11

The user authorized augmented MuPNIT training and then added AP3D versus GIF
CLIP-Proj. OCR remains paused. Work is isolated in `/mnt/data/soccer_gif_reid`;
the original sports tracking project was not modified.

## Executed

1. Audited the full release. Corrected identity parsing to use `player_<ID>`
   folders, not the first query filename token. Kept 118 training identities,
   42 validation-only identities, and all 51 test identities.
2. Decoded/hashed all 30,452 external reference photos. Selected up to eight
   per identity using size/shape/sharpness, duplicate checks and single-dominant-
   person detection. This is automatic quality screening, not human identity
   certification. Result: 944 training / 333 validation / 408 test references.
3. Prepared the initial image subset and actual segmentation masks: 14,612
   training / 1,337 validation / 1,623 test queries. Kept this artifact at
   `data/augmented_head_v1/`; **no model was trained on this preliminary subset**.
4. Added the matched video experiment after the user's AP3D steering. Selected
   genuine four-frame source windows and regenerated/reused segmentation masks.
   Retained 3,413 train / 323 validation / 391 test clips (13,652 / 1,292 / 1,564
   distinct query images). Rejected 97 / 5 / 11 clips before training. All intended
   identities remain. No exact duplicate image content in the retained manifest.
5. Used GIF's AP3DResNet50 implementation with current-device/offline-loading
   fixes, a shared cross-domain CircleLoss, and consistent train/eval BatchNorm.
   Downloaded official ImageNet ResNet50 spatial weights, verified SHA256
   `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
6. Verified 31 tests, including temporal parameter updates, frozen early layers,
   model state roundtrip, gap/track boundaries, and leakage rejection. AP3D GPU
   smoke test at the actual input/batch size passed two optimizer steps; peak
   allocated memory ~1.35 GiB. These smoke checks are not accuracy measurements.
7. Inspected both the image and four-frame augmentation previews. Query
   degradation is intentionally severe. Some original queries contain occluding
   players; a detected central silhouette is not a guarantee of a perfect target
   mask. Web-image label noise, posters/watermarks and undetected camera cuts
   remain limitations of this initial experiment.
8. Launched durable serial training at **22:10:54 UTC**, supervisor PID **709123**.
   Schedule: CLIP-Proj two epochs → AP3D two epochs → continue CLIP-Proj → continue
   AP3D. Each has up to 30 epochs × 500 steps, patience 8; best checkpoint chosen
   by harmonic mean of clean/degraded validation mAP. Test is evaluated only after
   model selection/training finishes. A separate report watcher refreshes the
   machine-readable status and progress report every 30 seconds.

## First measured baseline

On the fixed **323 validation clips / 42 identities / 333 clean web references**,
pretrained CLIP with four-frame pooling, before projection training:

| Query condition | Rank-1 | mAP |
|---|---:|---:|
| Original NBA crops | 21.98% | 15.78% |
| Fixed heavy soccer degradation | 2.48% | 4.33% |

No unaugmented ReID training was performed. The severe baseline collapse is
measured on the synthetic stress split; it does not estimate soccer match accuracy.
For subsequent trained results use the automatically refreshed report below.

## Two-epoch checkpoint comparison

Both models completed 1,000 optimizer updates on the same training clips and
references. These are **early checkpoints**, not converged-model conclusions.

| Model at epoch 2 | Clean Rank-1 | Clean mAP | Degraded Rank-1 | Degraded mAP |
|---|---:|---:|---:|---:|
| GIF CLIP-Proj, four-frame mean | 9.91% | 12.08% | 2.79% | 5.37% |
| AP3D, four-frame input | 4.02% | 5.03% | 1.55% | 4.16% |

CLIP-Proj currently leads AP3D, but it sacrifices substantial clean accuracy
relative to the pretrained CLIP baseline. AP3D remains weak under heavy
degradation. Training-loss reduction has not established a useful production
model. More training is scheduled; any eventual claim must use held-out results.

- CLIP-Proj epoch losses: 62.6993 → 23.0771; 790,272 trainable parameters.
- AP3D epoch losses: 77.8552 → 76.2516; 24,017,984 trainable parameters.
- Both `best.pt` and `last.pt` were saved. Both checkpoint formats were reloaded
  for actual one-crop and four-frame roster inference; each returned four finite,
  ranked identity scores. This is an execution check on four validation identities,
  not another accuracy benchmark. Details: `ap3d_comparison_v1/inference_check/`.
- The queue advanced to resume CLIP-Proj, with the full AP3D continuation queued
  afterward. Reports update every 30 seconds. No reserved test evaluation has yet
  occurred at this two-epoch snapshot.

## Resume and inspect

- [Live progress and measured results](ap3d_comparison_v1/progress_report.md)
- [Queue phase, process IDs and errors](ap3d_comparison_v1/queue_status.json)
- [Experiment, inference commands and methodological differences](../experiments/ap3d_comparison/README.md)
- [Frozen data audit](ap3d_comparison_v1/data_preparation.json)
- [Actual clip augmentations](ap3d_comparison_v1/clip_preview.jpg)
- [Archived source at launch](ap3d_comparison_v1/training_source.tar.gz)
- [Environment versions](ap3d_comparison_v1/environment.json)

Check the live process before launching another queue. On a stopped job,
`.venv/bin/python -m experiments.ap3d_comparison.run_queue` resumes from
`last.pt` when present, checking configuration/code/data fingerprints. A failed
job with no saved checkpoint needs explicit diagnosis and a fresh run directory;
do not silently erase its logs. Do not edit executable pipeline code during this
run. Original image-only inference and the new comparison checkpoint formats
are different; use the new experiment's inference command for these checkpoints.

## Published AP3D claim checked

GIF Table 4 is **MuPNIT-ReID-light**. AP3D has mAP 48.1%, Rank-1 30.3%; image-only
CLIP-Proj has mAP 25.3%, Rank-1 32.2%. Thus AP3D's reported advantage is mAP, not
all retrieval metrics. The current run differs in split, filtering, augmentation,
temporal grouping and objective and is not a replication of those numbers.
[Primary source](https://openaccess.thecvf.com/content/WACV2026/papers/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.pdf).
