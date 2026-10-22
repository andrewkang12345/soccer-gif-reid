#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/soccer_gif_reid
mkdir -p runs/soccernet_full_v2
if [ -f runs/soccernet_full_v2/last.pt ]; then
  .venv/bin/python -u -m experiments.soccernet_full_v2.train --resume --workers 6
else
  .venv/bin/python -u -m experiments.soccernet_full_v2.train --workers 6
fi
.venv/bin/python -m experiments.soccernet_full_v2.report
.venv/bin/python -m experiments.soccernet_full_v2.visualize_test
