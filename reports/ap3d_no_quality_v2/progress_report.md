# Matched MuPNIT training progress

Updated 2026-09-12T06:11:18.213162+00:00. Queue: **complete**; active model: **ap3d**.

Native-quality v2: no added downsampling, blur, JPEG artifacts, pixel noise, player shrinking or harsh exposure shifts. Clean validation is comparable to v1; augmented validation uses different images from the old degraded condition. No soccer production accuracy has been measured.

| Model/checkpoint | Epoch | Clean Rank-1 | Clean mAP | Augmented Rank-1 | Augmented mAP |
|---|---:|---:|---:|---:|---:|
| Pretrained CLIP; four-frame average | 0 | 21.98% | 15.78% | 8.98% | 8.74% |
| clip best validation | 24 | 24.15% | 23.60% | 19.20% | 19.91% |
| ap3d best validation | 22 | 11.46% | 12.49% | 11.15% | 12.09% |

## Processes and checkpoints

- clip: complete; latest logged epoch 30, step 500, loss 2.694289920769035. Best checkpoint: `/mnt/data/soccer_gif_reid/runs/ap3d_no_quality_v2/clip/best.pt`.
- ap3d: complete; latest logged epoch 30, step 500, loss 4.080119632286099. Best checkpoint: `/mnt/data/soccer_gif_reid/runs/ap3d_no_quality_v2/ap3d/best.pt`.

## Protocol and limits

- 3,413 train / 323 validation / 391 reserved test clips; four distinct source frames per clip; 118 / 42 / 51 disjoint identities.
- 1,685 clean references; 113 failed/incomplete clips excluded before training. No test-based model selection.
- Query-only appearance augmentation with no added quality degradation; fixed augmented validation; checkpoint selection balances both validation mAP values.
- CLIP backbone frozen; AP3D ImageNet spatial initialization, temporal/stages 2–4 adaptation. Same CircleLoss and training samples; different pretraining, input size and learning rates.
- Reference labels are inherited from the archive and not human-certified. Person-mask errors and undetected camera cuts remain possible.
- Queue: two epochs each, then up to 30 epochs each / patience 8, then final reserved-test evaluation.

[Configuration and inference](../../experiments/ap3d_no_quality_v2/README.md) · [Queue status](queue_status.json) · [Data audit](data_preparation.json) · [Augmentation preview](augmentation_preview.jpg)
