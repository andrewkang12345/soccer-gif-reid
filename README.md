# Soccer GIF ReID

> **Repository note:** start with [INDEX.md](INDEX.md) (current layout) and [reports/EXPERIMENT_STATUS.md](reports/EXPERIMENT_STATUS.md). Datasets, checkpoints and media are not in this repository — see [DATA.md](DATA.md). Sections below are the original September working notes and reference paths on the training host.

Train a roster-conditioned image retrieval model on **MuPNIT-ReID queries with kit/background augmentation, retaining their original visual quality**, paired with clean reference images. The initial stage freezes OpenAI CLIP ViT-L/14 and trains the exact GIF `ProjectionModel` (768 → Linear 512 → BatchNorm → ReLU → Linear 768 → BatchNorm). Both the CLIP output and projected embedding are L2 normalized. The objective is GIF's Circle Loss, applied bidirectionally between query and reference branches.

**Current experiment (2026-09-12):** **CLIP-Proj vs AP3D with no added quality degradation**, following the user's correction. The old runs finished and are preserved. Both new models restart from their original pretrained weights; kit/background/occlusion/geometry augmentation remains, while added downsampling, blur, JPEG artifacts, pixel noise, shrinking and harsh exposure shifts are removed. See the [new experiment instructions](experiments/ap3d_no_quality_v2/README.md), [live queue status](reports/ap3d_no_quality_v2/queue_status.json), [training results](reports/ap3d_no_quality_v2/progress_report.md), and [before/after preview](reports/ap3d_no_quality_v2/augmentation_preview.jpg). The same 3,413 / 323 / 391 four-frame clips and 118 / 42 / 51 disjoint identities are reused, with 1,685 clean references. The full archive and its **30,452 external references / 211 players** remain at `data/MuPNIT-ReID/`. **No soccer production accuracy is established.** The earlier [v1 results](reports/ap3d_comparison_v1/progress_report.md) remain available; their degraded-validation condition differs from the new augmented-validation condition.

This is a separate project from `/mnt/data/sports_reID`. Existing tracking configurations, histogram features, and evaluation claims are not dependencies.

## Setup

The local environment is ready at `.venv`. Run all commands from this directory. A fresh host needs Python 3.10+, compatible PyTorch/torchvision, NumPy, Pillow, OpenCV, PyYAML, scikit-learn, ftfy, regex, and tqdm. The bootstrap script reuses the host PyTorch installation and installs the small project dependencies without upgrading CUDA/PyTorch. It checks missing imports explicitly.

```bash
cd /mnt/data/soccer_gif_reid
bash scripts/bootstrap.sh
.venv/bin/python scripts/download_backbone.py
.venv/bin/python -m pytest -q
```

Upstream code is checked out in `vendor/GIF` at commit `304370c107d9091bea47bb9c5432ac76020c24c3`. OpenAI CLIP is pinned to `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. The notebook is not imported because it has execution side effects and requires user-specific paths. The compatible head is implemented in `src/soccer_gif_reid/model.py`. The inference checkpoint contains all visual backbone and projection parameters, so deployment does not download weights.

## Prepare MuPNIT-ReID

**The archive is now available locally and extracted.** Use the actual `train/`, `val/`, `test/`, and `gallery/` layout described in the [resume report](reports/RESUME_2026-09-11.md). The generic flat-filename importer and placeholder commands below predate archive inspection. In this full release, read global identity from `player_<ID>` directories; the first crop filename field is a game/video ID. The supplied validation split shares identities with train/test, so prepare an explicit identity-disjoint manifest before training. The Baidu login/download notes below are historical and no new login is needed.

[Official access instructions](https://github.com/Wojak27/GIF#2-prepare-the-mupnit-dataset) currently point to [this Baidu share](https://pan.baidu.com/s/1F8zmO3_L4-G9rIxB8eXbbg?pwd=bw6F), password `bw6F`. The archive is `MuPNIT/Release_versions/MuPNIT-ReID.tar.gz`; `MuPNIT-ReID-light.tar.gz` is also present. Use the full archive for the requested training. Extract to a data directory with adequate space. This project does not download full MOT videos or use a substitute training dataset.

The exact share/password was also tested through Chromium's visible controls. The
password was accepted, the full ReID archive was selected alone, and Download supplied
the Baidu Linux client installer instead of the data. The official `bdpan` CLI is now
installed locally at `.tools/bdpan/bdpan`; its first attempted read explicitly required
account login. To authorize the CLI without sending account credentials through chat:

```bash
bash /mnt/data/soccer_gif_reid/scripts/baidu_login.sh
```

The official CLI asks you to accept its first-use terms, then offers device login on
Baidu's site. Configuration stays in this project's ignored `.tools/bdpan/config.json`.
No account authentication has been completed. After login, only the full ReID archive
should be selected, not the surrounding folder containing the much larger MOT videos.

Provide separate directories of **broadcast queries** and **clean web/roster references** for training. The official code references `all_train/`, `query/`, and `gallery_test/`, and its notebook separately references `train/`; the downloaded layout has not been inspected. Do not assume a mixed `all_train/` folder consists only of queries, or treat the test gallery as a training gallery. If sources are mixed, classify them with the archive's metadata and use the explicit JSONL schema below.

```bash
.venv/bin/soccer-reid-prepare \
  --train-query /data/MuPNIT-ReID/TRAIN_BROADCAST_DIRECTORY \
  --train-reference /data/MuPNIT-ReID/TRAIN_CLEAN_REFERENCE_DIRECTORY \
  --test-query /data/MuPNIT-ReID/query \
  --test-reference /data/MuPNIT-ReID/gallery_test \
  --val-fraction 0.2 --seed 42 \
  --output data/mupnit.jsonl
```

The uppercase directory names are placeholders to replace after inspecting the archive. By default, 20% of **training identities**, including all their queries/references, are held out for validation. Official test identities remain separate. If an identity-disjoint official validation set exists, pass both `--val-query` and `--val-reference`. No random frame split, test-set checkpoint selection, gallery fallback, or query-as-reference shortcut is used. At least two identities and both domains are required in every included split. Train with at least eight identities for the default batch configuration.

The first filename field is the global player ID, e.g. `0007_...jpg` → `7`. For custom paths or mixed directories, supply one row per image:

```json
{"path":"/data/crops/0007_game_frame_track.jpg","pid":"7","split":"train","domain":"query"}
{"path":"/data/web/0007_reference.jpg","pid":"7","split":"train","domain":"reference"}
```

Use `--manifest custom.jsonl --output data/mupnit.jsonl` to audit it. Relative image/mask paths resolve against the manifest directory. Optional fields: `sample_id`, `sha256`, `person_mask`, `shirt_mask`, `shorts_mask`. Masks are binary PNGs in EXIF-oriented image coordinates, 255 inside the region and 0 elsewhere. Audit checks identity separation, positive-gallery coverage, image decoding, repeated paths, and exact image copies across splits/domains/identities. Content hashes also detect changed images/masks when resuming. Exact hashes do not detect all near-duplicates; inspect dataset provenance and perceptual duplicates before a release.

## Masks and query augmentation

Generate person masks once with a person-segmentation model. The existing local YOLO segmentation checkpoint is suitable for this preprocessing step:

```bash
YOLO_CONFIG_DIR="$PWD/.yolo" .venv/bin/soccer-reid-masks \
  --manifest data/mupnit.jsonl --output data/mupnit_masked.jsonl \
  --mask-dir data/masks \
  --weights /mnt/data/sports_reID/yolo11m-seg.pt \
  --device cuda:0 --preview reports/mupnit_augmentation_preview.jpg
```

Review the preview and `.masks.json` failures. The script tries the original crop, then context padding for tight crops, maps predicted polygons back to the original pixels, saves successful masks, and exits unsuccessfully if any query lacks a plausible central player. Repair those masks, then rerun using the partially completed manifest and a **new output filename**, or edit mask paths explicitly. Training fails on unresolved masks. No synthetic background mask is silently substituted. Optional human-parsing shirt/shorts masks improve garment localization. Without them, kit edits use conservative vertical regions intersected with person masks and a skin-color exclusion; this is an approximation and can miss or misclassify clothing, especially with unusual pose or skin/kit colors.

Training queries receive resolution reduction to approximately 28–112 pixels tall (never above native resolution), upsampling, blur or motion blur, JPEG quality 20–64, noise, gamma/exposure/channel/saturation changes, independent jersey/shorts recoloring, randomized stripes/patterns, clothing-restricted logo/number patches, turf/stands backgrounds, reduced apparent scale, rotation/perspective/translation/crop jitter, flips, and partial player occlusion. Donors come only from other **training identities**; a player-shaped silhouette is also supported. Heavy validation uses approximately 20–72 pixels, JPEG quality 12–64, and stronger corruption. The exact ranges and probabilities live in `augment.py`.

The central head is excluded from clothing edits; small crops have a capped motion kernel to avoid routinely erasing the whole player. Some heavily degraded crops remain intrinsically ambiguous. This pipeline does not generate new 3D viewpoints or restore absent face detail. Reference images get only deterministic aspect-preserving letterboxing and CLIP normalization. Letterboxing to 224×224 is a deliberate departure from upstream center-cropping to preserve heads/feet; the same versioned transform is used for training and inference.

## Train and evaluate

```bash
.venv/bin/soccer-reid-train --config configs/head.yaml

# Resume the same run after interruption, from the last completed epoch:
.venv/bin/soccer-reid-train --config configs/head.yaml \
  --resume runs/augmented_head/last.pt

# Optional: start a separate final-two-block experiment from the best augmented head:
.venv/bin/soccer-reid-train --config configs/finetune.yaml
```

The initial run trains directly on augmented queries. Eight identities × two queries per identity provide positives and negatives in each batch, with a clean same-identity reference for each query. Frozen CLIP runs without gradients. Fixed clean/degraded validation CLIP features are computed once and reused while the backbone is frozen; training queries receive fresh deterministic-per-epoch augmentations. The optional final-block run uses fresh visual features for every evaluation, a small backbone learning rate, and validation-based promotion.

`runs/augmented_head/` contains:

- `best.pt`: self-contained checkpoint selected after training an epoch.
- `last.pt`: model/optimizer/scaler/RNG/config state for epoch-boundary resume.
- `best_metrics.json`: Rank-1, Rank-5, mAP, per-identity metrics, counts, and seed for clean/degraded validation.
- `history.jsonl`, `config.json`, `data_audit.json`, `status.json`: reproducibility and progress records.

Selection maximizes the harmonic mean of **clean and degraded validation mAP**. Fine-tuning must also improve degraded mAP and retain clean mAP/Rank-1 within two absolute percentage points of the frozen-head anchor. If no candidate qualifies, keep the frozen-head checkpoint; the fine-tuning directory will not contain a promoted `best.pt`. Resume rejects changed training configuration, pipeline source code, or data. It restores the last completed epoch, not a partially executed minibatch. A run where every optimizer step overflows cannot promote a checkpoint.

All metrics are fractions in [0,1]. AP is calculated against every clean gallery image of the correct identity; Rank-1 is per query. Identity-macro metrics expose imbalance. No camera filtering is applied between broadcast queries and independent references. Missing positive references are errors. These image-gallery metrics differ from inference's per-identity reference centroids and from GIF's tracking metric also named mAP.

```bash
.venv/bin/soccer-reid-evaluate \
  --checkpoint runs/augmented_head/best.pt \
  --manifest data/mupnit_masked.jsonl --split val \
  --output reports/nba_validation.json

# Run test once after configuration and checkpoint selection are finalized:
.venv/bin/soccer-reid-evaluate \
  --checkpoint runs/augmented_head/best.pt \
  --manifest data/mupnit_masked.jsonl --split test \
  --output reports/nba_test.json
```

## Roster inference

Use identity directories containing one or more clean headshots, roster images, or full-body photographs:

```text
roster/
  player_001/
    headshot.jpg
    full_body.jpg
  player_002/
    roster_photo.jpg
tracklet/
  frame_000001.jpg
  frame_000002.jpg
```

Flat `roster/<identity>.jpg` also works for one reference per identity. Do not mix flat and identity-directory layouts. A tracklet directory must contain already-cropped images of **one player** and uses evenly spaced frames if it exceeds `--max-frames` (64 by default). Zero-pad frame filenames for chronological sorting.

```bash
.venv/bin/soccer-reid-infer \
  --checkpoint runs/augmented_head/best.pt \
  --roster /path/to/roster --query /path/to/player_crop.jpg \
  --top-k 5 --output ranked.json

.venv/bin/soccer-reid-infer \
  --checkpoint runs/augmented_head/best.pt \
  --roster /path/to/roster --query /path/to/tracklet \
  --top-k 5 --output tracklet_ranked.json
```

JSON includes ranked identities, cosine similarity scores, top-two margin, per-frame top-1 agreement, frame selection, roster size and checkpoint fingerprint. Multiple references form a normalized identity centroid. Tracklets average per-frame cosine scores, giving each selected frame equal weight. Scores are not probabilities. Mixed-identity tracklets are not corrected automatically. Calibrated `--min-similarity` and `--min-margin` thresholds can reject uncertain matches as `unknown`; none is presented as universally valid. For repeated service calls, instantiate `soccer_gif_reid.infer.RosterMatcher` once to keep the model/reference embeddings in memory.

## Validation status and release boundary

No MuPNIT training has run without the dataset. See [measured status](reports/metrics_status.json). Unit/integration tests and a real-CLIP implementation smoke test are separate from recognition evaluation. Local soccer crops are used only to inspect augmentations and exercise code; they are not substituted for MuPNIT or scored as a soccer benchmark.

Before calling the model production-ready, measure it on held-out **soccer** broadcasts with genuinely independent clean roster references, unseen identities/games/kits, difficult headshot/full-body combinations, unknown players, and mixed/impure tracklets. NBA and synthetic degradation results cannot establish that performance. Calibrate rejection thresholds and record latency on the intended hardware. Check the upstream code, weights, and data redistribution/commercial terms; the inspected GIF root has no explicit repository-wide license.
