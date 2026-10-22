# Data, checkpoints and media (not in this repository)

This repository holds code, configs, tests, and text reports only. Datasets, model checkpoints,
backbone weights, and all image/video media are excluded by `.gitignore` and will be shared
separately (Google Drive or the original dataset links). Download links will be added here.

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

## Built by this project (to be shared via Google Drive)

| path | contents | size |
|---|---|---|
| `data/soccernet_full_v2/` | frozen manifest, reference gallery, query crops, audits, self-supervised crops | ~51 GB |
| `data/arg_fra_expanded_gallery_v1/` | Argentina-France development clip and gallery | ~0.4 GB |
| `runs/soccernet_full_v8/` | clean paired-data-only run (current) | ~3.7 GB |
| `runs/soccernet_full_v7/` | latest completed model (auxiliary-identity leakage caveat, see `reports/EXPERIMENT_STATUS.md`) | ~4.9 GB |
| `runs/ap3d_no_quality_v2/clip/best.pt` | NBA-trained GIF projection, scored as a baseline | ~1.2 GB |
| `reports/**/*.jpg, *.mp4` | review sheets, error sheets, demo videos | ~0.5 GB |

Older runs (`runs/soccernet_full_v2`–`v6`, `archive/`) are not needed to reproduce current results.

## Also excluded

- Per-sample ranking dumps (`ranked_images.jsonl`, `*_test_rankings.jsonl`) — regenerable by the scoring scripts.
- SoccerNet `Labels-v3.json` and the SoccerNet GSR paper — third-party; get from SoccerNet.
- Scraped search-engine probe pages (`*_probe.html`).
