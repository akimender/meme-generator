#!/usr/bin/env bash
#SBATCH --job-name=meme-vit-gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints

export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

python3 -m pip install -r requirements.txt

python3 scripts/download_vit.py \
  --model-name google/vit-base-patch16-224 \
  --cache-dir "$HF_HOME" \
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
  --output-dir checkpoints/vit-gpt2 \
  --batch-size 8 \
  --epochs 1 \
  --lr 5e-5 \
  --warmup-steps 500 \
  --max-length 64 \
  --visual-prefix-length 8 \
  --num-workers 8 \
  --freeze-vision \
  --no-freeze-language-model
