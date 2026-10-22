# Completed training: no added visual-quality degradation

Both models completed 30 epochs. Queue finished 2026-09-12T06:11:08.334447+00:00.

Validation-selected checkpoints: CLIP-Proj epoch 24; AP3D epoch 22.

| Model | Clean test Rank-1 | Clean test mAP | Augmented test Rank-1 | Augmented test mAP |
|---|---:|---:|---:|---:|
| clip | 24.30% | 21.17% | 16.11% | 17.02% |
| ap3d | 10.23% | 9.64% | 7.67% | 9.29% |

On unchanged clean validation, CLIP-Proj Rank-1 increased from 18.58% to 24.15%; mAP increased from 17.83% to 23.60%.

CLIP-Proj is the stronger model in this experiment. Absolute retrieval accuracy remains far below the requested production goal. These NBA subset results do not establish soccer performance. The old degraded condition and the new augmented condition use different images and should not be treated as the same benchmark. Test was previously evaluated in v1; checkpoints were selected using validation only.

[Configuration and inference](../../experiments/ap3d_no_quality_v2/README.md) · [Full training history](progress_report.md)
