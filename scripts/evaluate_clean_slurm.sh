#!/usr/bin/env bash
#SBATCH --job-name=meme-eval-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs reports

export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false

GENERATION_LIMIT="${GENERATION_LIMIT:-500}"
COMPUTE_CLIP="${COMPUTE_CLIP:-1}"
CLIP_FLAG=()
if [[ "$COMPUTE_CLIP" == "1" ]]; then
  CLIP_FLAG=(--compute-clip)
fi

python3 scripts/evaluate_captioner.py \
  --checkpoint checkpoints/vit-gpt2-clean/best.pt \
  --dataset-dir memes900k \
  --image-dir memes900k/images_224 \
  --split test \
  --batch-size 32 \
  --num-workers 4 \
  --generation-limit "$GENERATION_LIMIT" \
  --num-captions 1 \
  --max-new-tokens 24 \
  --temperature 0.7 \
  --top-p 0.9 \
  --device cuda \
  "${CLIP_FLAG[@]}" \
  --output-json reports/vit-gpt2-clean-test-metrics.json
