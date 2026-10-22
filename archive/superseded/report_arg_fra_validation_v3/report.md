# Argentina–France validation experiment v3

Authorized change: retain current NBA training data and appearance-only v2 augmentation, including kit recoloring/patterns/masking, while selecting checkpoints by Argentina–France individual-photo mAP. No added blur, downsampling or JPEG degradation. No soccer gradients or soccer training pairs.

Both CLIP-Proj and AP3D restart from their original pretrained backbones, with the same seed, learning rates, 500 steps/epoch, maximum 30 epochs and patience 8. This isolates checkpoint selection from training changes. CLIP backbone remains frozen; AP3D retains the v2 trainable layers.

Training: 118 NBA identities, 944 clean references (8/player), 3,413 four-frame query clips. Training manifest and clips are unchanged. Raw downloaded gallery: 30,452 images / 211 identities = 144.32/player; the raw inventory is not the manifest actually used in our run. No claim that the original authors trained on a particular reference count is established by the raw inventory alone.

Soccer validation: fixed 109 four-observation Argentina–France groups / 20 query identities; fixed 168 web photos / 21 gallery identities. The metric uses all eight positive gallery photos. This match is now development validation and cannot serve as an independent test of the selected model. Existing source-caption and annotation uncertainties remain. No team restriction is added. Roster-prototype metrics are diagnostic only.

NBA clean and augmented validation metrics remain recorded but do not select checkpoints. Best and last checkpoints, optimizer state, per-epoch metrics, data/code hashes and live status are saved separately for each model. Final NBA test scoring is diagnostic; the NBA test has been inspected previously.

Files:
- Config: `configs/arg_fra_validation_v3.yaml`
- Code: `experiments/arg_fra_validation_v3/`
- Runs: `runs/arg_fra_validation_v3/{clip,ap3d}/`
- Queue status and training logs: `reports/arg_fra_validation_v3/`
- Gallery: `data/arg_fra_expanded_gallery_v1/gallery.jsonl`

Queue runs two epochs CLIP, two AP3D, then finishes each to its patience/epoch limit. Soccer and NBA metrics will be in each run's `history.jsonl` and `best_metrics.json`; supervisor errors are recorded in `queue_status.json`.

## Completed results

Queue completed 2026-09-12 23:33 UTC. CLIP stopped after 9 epochs (best epoch 1): soccer Rank-1 10.09%, mAP 13.20%. AP3D stopped after 12 epochs (best epoch 4): soccer Rank-1 9.17%, mAP 9.00%. Both hit patience 8.

Previous NBA-selected checkpoints on the same gallery: CLIP 12.84% / 13.21%, AP3D 6.42% / 8.14% (Rank-1 / mAP). Soccer selection provided no meaningful CLIP gain; AP3D improved modestly but remains weak. Early soccer peaks suggest continued NBA optimization does not reliably improve transfer; this is not proof of a specific augmentation defect. These are development-validation results, not independent test accuracy.
