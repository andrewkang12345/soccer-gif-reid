#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/soccer_gif_reid
while [ ! -f reports/soccernet_full_v2/verification_soccernet_full_v8.json ]; do
  if ! tmux has-session -t soccer_full_v8_clean 2>/dev/null; then
    echo 'Clean training/scoring session ended before verification; inspect training_v8_clean.log.'
    exit 1
  fi
  sleep 30
done
.venv/bin/python -m experiments.soccernet_full_v2.write_current_report --run runs/soccernet_full_v8
.venv/bin/python -m experiments.soccernet_full_v2.status
