# Data, checkpoints and media (not in this repository)

This repository holds code, configs, tests, and text reports only. Datasets, model checkpoints,
backbone weights, and all image/video media are excluded by `.gitignore` and will be shared
separately: project-built data and checkpoints on Google Drive (below), public datasets from their original sources.

Paths are relative to the repository root; the code expects these files at these paths.

## Public datasets (get from the original source)

| path | source |
|---|---|
| `data/MuPNIT-ReID/` (+ `-light`) | MuPNIT, via the [GIF repository instructions](https://github.com/Wojak27/GIF#2-prepare-the-mupnit-dataset) |
| `data/MuPNIT_60fps/` | MuPNIT full 60 fps release (same source) |
| `data/MuPnit_YoloX_weights/` | MuPNIT YOLOX detector weights (same source) |
| `data/sn_gsr_2025/` | [SoccerNet Game State Reconstruction 2025](https://github.com/SoccerNet/sn-gamestate) |
| `data/soccernet_named_v1/` (ReID archives) | [SoccerNet ReID](https://github.com/SoccerNet/sn-reid) |
| `weights/` | OpenAI CLIP ViT-L/14, fetched by `scripts/download_backbone.py` |
| `vendor/GIF/` | [Wojak27/GIF](https://github.com/Wojak27/GIF) at commit `304370c107d9091bea47bb9c5432ac76020c24c3` |

## Built by this project (shared as zip archives)

Two archives, both with paths relative to the repository root — unzip them at the root:

| archive | contents |
|---|---|
| `soccer_gif_reid_data_v2.zip` | `data/soccernet_full_v2/`: frozen `manifest.jsonl` (14,896 rows; 265 train / 115 val / 182 test players), every gallery reference, query crop and person mask it lists, split/audit files (`final_audit.json`, `players.json`, `repartition_audit.json`, …) and `backup_prerepartition/` for `repartition.py` |
| `soccer_gif_reid_checkpoints_v8.zip` | `runs/soccernet_full_v8/best.pt` + config/metrics/history (current clean model), `runs/ap3d_no_quality_v2/clip/best.pt` (NBA-trained GIF projection: the baseline that `train_v2.py` scores at the end of every run, so a run needs it to finish; v8 starts from pretrained CLIP with a fresh projection, not from this checkpoint), `weights/number_text.pt` (jersey-number text anchors) |

| archive | download | size | MD5 |
|---|---|---|---|
| `soccer_gif_reid_data_v2.zip` | [Google Drive](https://drive.google.com/file/d/1Gb3QOQFDKEGhxyuqoMnuxo54ZL3HKKs0/view) | 4.6 GB | `df4037b2f978c477d444d9e90b13839c` |
| `soccer_gif_reid_checkpoints_v8.zip` | [Google Drive](https://drive.google.com/file/d/1jBxY9HsWXgZvlLbJaIBXYuupSDmZ-dFm/view) | 1.6 GB | `b91e079ec287c8a2f11a5b366bad8496` |

```bash
git clone https://github.com/andrewkang12345/soccer-gif-reid && cd soccer-gif-reid
pip install gdown
gdown 1Gb3QOQFDKEGhxyuqoMnuxo54ZL3HKKs0   # soccer_gif_reid_data_v2.zip
gdown 1jBxY9HsWXgZvlLbJaIBXYuupSDmZ-dFm   # soccer_gif_reid_checkpoints_v8.zip
md5sum soccer_gif_reid_*.zip
unzip soccer_gif_reid_data_v2.zip
unzip soccer_gif_reid_checkpoints_v8.zip
python scripts/relocate_data.py        # manifest stores absolute paths from the original host
python scripts/download_backbone.py    # CLIP ViT-L/14 -> weights/
```

Not included, because the v8 configuration does not read them: the self-supervised crops and
`ssl_pairs.jsonl` (only used with `--ssl-epochs > 0`), the per-match SoccerNet metadata,
screened-out reference candidates and raw web downloads, and runs v2–v7 (v7 carries the
auxiliary-identity leakage caveat in `reports/EXPERIMENT_STATUS.md`).

Licensing: query crops come from SoccerNet (use is subject to the SoccerNet data terms) and
reference photos were collected from the web (largely Wikimedia Commons, per-image licenses
vary). Share the archives for research use only.

## Also excluded

- Per-sample ranking dumps (`ranked_images.jsonl`, `*_test_rankings.jsonl`) — regenerable by the scoring scripts.
- SoccerNet `Labels-v3.json` and the SoccerNet GSR paper — third-party; get from SoccerNet.
- Scraped search-engine probe pages (`*_probe.html`).
