# Matched MuPNIT training progress

Updated 2026-09-11T23:07:53.843822+00:00. Queue: **complete**; active model: **ap3d**.

These are measured results from the retained full-release subset, not GIF's published MuPNIT-ReID-light numbers. No soccer production accuracy has been measured.

| Model/checkpoint | Epoch | Clean Rank-1 | Clean mAP | Degraded Rank-1 | Degraded mAP |
|---|---:|---:|---:|---:|---:|
| Pretrained CLIP; four-frame average | 0 | 21.98% | 15.78% | 2.48% | 4.33% |
| clip best validation | 18 | 18.58% | 17.83% | 6.50% | 6.84% |
| ap3d best validation | 3 | 4.02% | 5.42% | 2.79% | 4.19% |

## Processes and checkpoints

- clip: complete; latest logged epoch 26, step 500, loss 10.957220894986763. Best checkpoint: `/mnt/data/soccer_gif_reid/runs/ap3d_comparison_v1/clip/best.pt`.
- ap3d: complete; latest logged epoch 11, step 500, loss 27.13993489265442. Best checkpoint: `/mnt/data/soccer_gif_reid/runs/ap3d_comparison_v1/ap3d/best.pt`.

## Protocol and limits

- 3,413 train / 323 validation / 391 reserved test clips; four distinct source frames per clip; 118 / 42 / 51 disjoint identities.
- 1,685 clean references; 113 failed/incomplete clips excluded before training. No test-based model selection.
- Strong query-only soccer augmentation; fixed heavy validation; checkpoint selection balances both validation mAP values.
- CLIP backbone frozen; AP3D ImageNet spatial initialization, temporal/stages 2–4 adaptation. Same CircleLoss and training samples; different pretraining, input size and learning rates.
- Reference labels are inherited from the archive and not human-certified. Person-mask errors and undetected camera cuts remain possible.
- Queue: two epochs each, then up to 30 epochs each / patience 8, then final reserved-test evaluation.

[Configuration and inference](../../experiments/ap3d_comparison/README.md) · [Queue status](queue_status.json) · [Data audit](data_preparation.json) · [Clip preview](clip_preview.jpg)
