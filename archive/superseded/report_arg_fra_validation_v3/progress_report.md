# Soccer-validation v3 live progress

Queue: complete

Argentina–France is development validation. Best checkpoints are selected by soccer individual-photo mAP. NBA metrics do not select checkpoints.

| Model | Status | Epochs complete | Best epoch | Soccer R1 / mAP | NBA clean R1 / mAP |
|---|---|---:|---:|---:|---:|
| clip | complete | 9 | 1 | 10.09% / 13.20% | 14.86% / 15.81% |
| ap3d | complete | 12 | 4 | 9.17% / 9.00% | 8.36% / 8.42% |

[Protocol](report.md). Full metrics and histories: `runs/arg_fra_validation_v3/{clip,ap3d}/`.
