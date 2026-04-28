# Meme Caption Generator

Training scaffold for a meme caption generator that conditions an autoregressive
language model on a meme template image.

## Data

The `memes900k` dataset is expected to contain:

- `templates.txt`: template name, template slug, image URL
- `captions_train.txt`, `captions_val.txt`, `captions_test.txt`: template name, score, caption
- `images/`: one local image per template URL basename

Validate the local dataset with:

```bash
python3 scripts/check_dataset.py --dataset-dir memes900k
```

## Model

The default vision encoder is `google/vit-base-patch16-224`. Hugging Face's
`ViTImageProcessor` resizes, rescales, and normalizes dataset images to the
224x224 input resolution expected by the pretrained ViT.

The architecture is:

```text
image -> ViT -> [CLS] embedding -> projection layer -> visual prefix tokens
caption tokens -> causal LM embeddings
visual prefix + caption embeddings -> GPT-2/Llama-style autoregressive LM
```

The visual prefix positions are ignored in the LM loss, so training optimizes
caption token likelihood conditioned on the image prefix.

## Train

Install dependencies:

```bash
pip install -r requirements.txt
```

Small smoke run:

```bash
python3 scripts/train_captioner.py \
  --dataset-dir memes900k \
  --language-model gpt2 \
  --limit-train 32 \
  --limit-val 16 \
  --batch-size 2 \
  --epochs 1
```

Full GPT-2 baseline:

```bash
python3 scripts/train_captioner.py \
  --dataset-dir memes900k \
  --vision-model google/vit-base-patch16-224 \
  --language-model gpt2 \
  --output-dir checkpoints/vit-gpt2 \
  --batch-size 8 \
  --epochs 1
```

For Llama, pass a causal LM checkpoint you have access to:

```bash
python3 scripts/train_captioner.py \
  --dataset-dir memes900k \
  --language-model meta-llama/Llama-3.2-1B \
  --output-dir checkpoints/vit-llama \
  --batch-size 1 \
  --epochs 1
```

