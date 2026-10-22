#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Install a matching torch/torchvision build for the host before running this script.
python -m venv --system-site-packages .venv
# AWS PyTorch images place packages in a parent venv, not the base interpreter.
python - <<'PY'
import site
from pathlib import Path
target = next(Path('.venv/lib').glob('python*/site-packages'))
target.joinpath('parent_environment.pth').write_text('\n'.join(site.getsitepackages()) + '\n')
PY
.venv/bin/python -m pip install --no-deps -e . pytorch-metric-learning==2.9.0
.venv/bin/python -m pip install --no-deps --force-reinstall 'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
.venv/bin/python -c 'import torch, torchvision, cv2, yaml, clip, ftfy, regex, tqdm, sklearn; from pytorch_metric_learning.losses import CircleLoss; print(torch.__version__, torch.cuda.is_available())'
if [ ! -d vendor/GIF/.git ]; then
  git clone https://github.com/Wojak27/GIF.git vendor/GIF
fi
git -C vendor/GIF checkout --detach 304370c107d9091bea47bb9c5432ac76020c24c3
