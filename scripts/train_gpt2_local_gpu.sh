#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints models

python3 -m pip install -r requirements.txt

python3 scripts/download_vit.py \
  --model-name google/vit-base-patch16-224 \
  --local-dir models/vit-base-patch16-224

python3 scripts/resize_images.py \
  --input-dir memes900k/images \
  --output-dir memes900k/images_224 \
  --size 224 \
  --mode resize

python3 scripts/train_captioner.py \
  --dataset-dir memes900k \
  --image-dir memes900k/images_224 \
  --vision-model models/vit-base-patch16-224 \
  --language-model gpt2 \
  --output-dir checkpoints/vit-gpt2-local-gpu \
  --batch-size 4 \
  --epochs 1 \
  --lr 5e-5 \
  --warmup-steps 500 \
  --max-length 64 \
  --visual-prefix-length 8 \
  --num-workers 2 \
  --device cuda \
  --freeze-vision \
  --no-freeze-language-model
