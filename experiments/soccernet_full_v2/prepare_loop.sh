#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/soccer_gif_reid
while true; do
  .venv/bin/python -m experiments.soccernet_full_v2.collect_pages
  .venv/bin/python -m experiments.soccernet_full_v2.collect_web
  if [ -f data/soccernet_full_v2/page_search_complete.json ]; then
    .venv/bin/python -m experiments.soccernet_full_v2.collect_pages
    .venv/bin/python -m experiments.soccernet_full_v2.collect_web
    break
  fi
  sleep 30
done
.venv/bin/python -m experiments.soccernet_full_v2.screen_references
