#!/usr/bin/env bash
# Complete Baidu account authorization in your terminal and Baidu's own login page.
# This does not request or print account passwords, cookies, or stored access tokens.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cli="$project_dir/.tools/bdpan/bdpan"
export BDPAN_CONFIG_DIR="$project_dir/.tools/bdpan"
if [ ! -x "$cli" ]; then
  echo "Official bdpan CLI is missing from $cli" >&2
  exit 1
fi
exec "$cli" login --device-code --no-check-update \
  --config-path "$project_dir/.tools/bdpan/config.json"
