#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/soccer_gif_reid
.venv/bin/python -u -m experiments.soccernet_full_v2.train_v2 \
  --out runs/soccernet_full_v8 --ssl-epochs 0 \
  --head-epochs 30 --finetune-epochs 15 --steps 500 --batch 16 --workers 2 \
  --number-weight 2 --flat-kit .2 --number-unknown uniform --select fused "$@"
.venv/bin/python -u -m experiments.soccernet_full_v2.number_channel \
  --checkpoint runs/soccernet_full_v8/best.pt --workers 2
OPENBLAS_NUM_THREADS=4 .venv/bin/python -u -m experiments.soccernet_full_v2.verify_completed \
  --run runs/soccernet_full_v8
.venv/bin/python -m experiments.soccernet_full_v2.status
