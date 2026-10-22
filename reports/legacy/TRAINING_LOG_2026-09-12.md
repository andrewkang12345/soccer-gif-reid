# Remove artificial visual-quality reduction — 2026-09-12

The user requested removing the visual-quality degradation from augmentation for
now. The previous queue had already completed, so no running job was interrupted.
V1 code, checkpoints, data manifests and reports remain intact.

New isolated implementation: `experiments/ap3d_no_quality_v2/`.
Configuration: `configs/ap3d_no_quality_v2.yaml`.

Removed the downsample–upsample stage, blur, JPEG recompression, pixel noise,
extra 384px working-resolution cap, artificial player shrinking, and harsh
exposure/gamma/saturation shifts. Retained kit/shorts recoloring, kit patterns,
logo/number masking, background changes, modest geometry/flips and player-like
occlusions. References stay clean. Source image files are unchanged; normal model
resizing and geometry still use interpolation.

The same fixed v1 clips, masks, reference images and disjoint identities are reused.
No re-selection based on recognition scores. Both models start again from the
original pretrained CLIP / ImageNet ResNet50 weights rather than v1 ReID weights.
The learning rates, batch composition, objective and epoch/early-stop schedule
are unchanged. Validation now records `clean` and `augmented` (no added quality
degradation). The latter must not be confused with the older degraded validation.

All **35 tests passed**, including four new checks for native dimensions, no RGB
blur/lossy encoding, preservation of player texture when other augmentations are
isolated, clean reference handling, and the new validation selection conditions.
The [preview](ap3d_no_quality_v2/augmentation_preview.jpg) was generated from actual
training crops and visually inspected before launch.

Detached queue launched **2026-09-12 04:50:17 UTC**, supervisor PID **739811**.
Schedule: two epochs CLIP-Proj → two epochs AP3D → continue each to its 30-epoch /
patience-8 limit. Both save `best.pt` and `last.pt`. A separate report watcher
(PID 740090) refreshes live results and `reports/metrics_status.json` every 30 seconds.

The held-out test subset was already evaluated in v1; it is not newly unseen data.
Test metrics are not used for training, augmentation choice, or checkpoint selection.
V2 evaluates the selected checkpoints after training. No soccer production claim
is established by this run.

- [Live progress and measurements](ap3d_no_quality_v2/progress_report.md)
- [Queue/process status](ap3d_no_quality_v2/queue_status.json)
- [Preview: original / previous augmentation / new augmentation / clean reference](ap3d_no_quality_v2/augmentation_preview.jpg)
- [Configuration and inference](../experiments/ap3d_no_quality_v2/README.md)
- [Machine-readable change log](ap3d_no_quality_v2/change_log.json)
- [Old completed run snapshot](ap3d_no_quality_v2/previous_run_status.json)

Use the v2 inference module for `mupnit-native-quality-v2` checkpoints. Do not
edit executable pipeline code while the queue runs; resume checks its fingerprint.

## Soccer validation v3

Launched 2026-09-12T22:58:22.729025+00:00, supervisor PID 869220. Same NBA training subset and kit-invariance augmentation; fresh pretrained initialization. Argentina–France mAP now selects checkpoints; NBA clean/augmented validation retained as diagnostics. Match is no longer independent test. Raw gallery mean144.32 photos/player confirmed, actual training subset remains8/player. Cached soccer evaluator reproduces prior verified CLIP-Proj metrics. See [v3 report](arg_fra_validation_v3/report.md).
