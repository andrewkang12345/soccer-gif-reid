# SoccerNet player ReID

Current state and full history: [EXPERIMENT_STATUS.md](../../reports/EXPERIMENT_STATUS.md).
The clean run is `runs/soccernet_full_v8/`. Older auxiliary-pretrained runs contain evaluation
players in other matches; their numbers are historical diagnostics. The October 20 audit is
`reports/soccernet_full_v2/ssl_identity_audit.json`.

The frozen paired subset contains 265 training players, 115 validation players and 182 test
players. References are uncapped screened public photographs, stored as padded person crops.
Training requires three references; evaluation allows one. Source labels remain weak supervision.

V8 starts from pretrained CLIP ViT-L/14 with fresh GIF projection and number-adapter weights,
trains the head first, then the last two CLIP blocks. It uses no auxiliary training images and
adds no blur, downsampling, JPEG or noise degradation. Kit/background/geometry augmentation
remains enabled. The number adapter and optional reference-kit augmentation are inherited from
the later recorded experiments. Selection uses validation only; the existing test is a reused
development benchmark and is not a newly untouched final test.

From `/mnt/data/soccer_gif_reid`:

```bash
.venv/bin/python -m experiments.soccernet_full_v2.status
# After an interruption of the clean run:
bash experiments/soccernet_full_v2/run_identity_disjoint.sh --resume
```

The clean run is already active in tmux `soccer_full_v8_clean`. Its wrapper trains, scores the
selected checkpoint and baselines, computes validation-selected roster-number fusion, and
independently recounts the saved rankings. A companion `soccer_full_v8_report` session writes
`reports/soccernet_full_v2/results_soccernet_full_v8.md` when verification finishes.

`train_v2.py` disables auxiliary pretraining by default. If `--ssl-epochs` is explicitly positive,
it excludes every auxiliary match with a held-out roster ID/name or missing roster metadata.
The source and selected auxiliary-group hashes are recorded. It never initializes from an old
soccer checkpoint on a new run. Checkpoints are atomic and training resumes using `--resume`.

After a checkpoint is selected, photo-only inference is:

```bash
.venv/bin/python -m experiments.soccernet_full_v2.infer \
  --checkpoint runs/soccernet_full_v8/best.pt \
  --roster /path/to/roster --query /path/to/crop_or_tracklet \
  --cache /path/to/roster_features.pt --top-k 5 --output /path/to/ranking.json
```

Use `roster/<player_name>/*.{jpg,png,webp}`. The query may be a crop or genuine track observations,
sampled evenly up to 64 frames. Reference and query embeddings are averaged and normalized.
Roster photographs are detector-cropped by default; use `--no-roster-crop` for prepared crops.

For number fusion, also pass `--roster-numbers /path/to/numbers.json --photo-weight VALUE` with the
run's validation-selected alpha. The JSON maps exact roster folder names to numbers 0–99;
duplicate numbers are allowed. Fusion needs the trained adapter and matching
`weights/number_text.pt`; the anchor hash is checked against the checkpoint. Scores are cosine
similarities, not calibrated probabilities. No production commitment threshold is established.

Photo retrieval metrics rank individual reference images. Fusion metrics rank one identity
prototype. The reported player-by-match pooling uses ground-truth grouping and should not be
called McByte++ end-to-end tracking accuracy.

Historical preparation scripts and their audits remain here for provenance. The current manifest
is frozen; collection/screening/repartition/finalize are dataset-building steps, not inference or
training-resume commands.
