from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, ViTImageProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meme_captioning.collator import MemeCaptionCollator
from meme_captioning.data import CAPTION_END, CAPTION_SEP, CAPTION_START, MemeCaptionDataset, is_unsafe_caption
from meme_captioning.model import VisionPrefixCausalLM


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vit-gpt2-clean/best.pt")
    parser.add_argument("--dataset-dir", default="memes900k")
    parser.add_argument("--image-dir", default="memes900k/images_224")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--generation-limit", type=int, default=500)
    parser.add_argument("--num-captions", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--compute-clip", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--filter-unsafe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-two-parts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--formatted-caption", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    train_args = checkpoint["args"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = ViTImageProcessor.from_pretrained(checkpoint_path.parent / "image_processor")

    model = VisionPrefixCausalLM(
        vision_model_name=train_args["vision_model"],
        language_model_name=train_args["language_model"],
        visual_prefix_length=train_args["visual_prefix_length"],
        freeze_vision=True,
        freeze_language_model=False,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    filter_config = {
        "min_score": args.min_score if args.min_score is not None else train_args.get("min_score"),
        "filter_unsafe": value_or_checkpoint(args.filter_unsafe, train_args.get("filter_unsafe", False)),
        "require_two_parts": value_or_checkpoint(args.require_two_parts, train_args.get("require_two_parts", False)),
        "formatted_caption": value_or_checkpoint(args.formatted_caption, train_args.get("formatted_caption", False)),
    }

    dataset = MemeCaptionDataset(
        args.dataset_dir,
        split=args.split,
        image_dir=args.image_dir,
        **filter_config,
    )
    print(f"split={args.split} examples={len(dataset):,} filters={filter_config}")

    max_length = args.max_length or train_args.get("max_length", 64)
    collator = MemeCaptionCollator(image_processor, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=args.device == "cuda",
    )

    loss_metrics = evaluate_perplexity(model, loader, args.device)
    print(f"test_loss={loss_metrics['loss']:.4f} perplexity={loss_metrics['perplexity']:.2f}")

    generation_metrics = {}
    if args.generation_limit > 0:
        generation_metrics = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset=dataset,
            args=args,
        )
        for key, value in generation_metrics.items():
            if isinstance(value, float):
                print(f"{key}={value:.4f}")
            else:
                print(f"{key}={value}")

    results = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_examples": len(dataset),
        "filters": filter_config,
        **loss_metrics,
        **generation_metrics,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def value_or_checkpoint(value, checkpoint_value):
    return checkpoint_value if value is None else value


@torch.no_grad()
def evaluate_perplexity(model, loader, device: str) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    for batch in tqdm(loader, desc="perplexity"):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        token_count = int((batch["labels"] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    loss = total_nll / max(total_tokens, 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "eval_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, image_processor, dataset, args) -> dict[str, float | int]:
    limit = min(args.generation_limit, len(dataset))
    generated_texts: list[str] = []
    reference_texts: list[str] = []
    generated_image_paths: list[Path] = []

    for idx in tqdm(range(limit), desc="generation"):
        example = dataset.examples[idx]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB")
        pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(args.device)
        for _ in range(args.num_captions):
            output_ids = model.generate(
                pixel_values=pixel_values,
                tokenizer=tokenizer,
                prompt=CAPTION_START,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=tokenizer.convert_tokens_to_ids(CAPTION_END),
            )
            generated_texts.append(clean_caption(tokenizer.decode(output_ids[0], skip_special_tokens=False)))
            reference_texts.append(clean_caption(example.caption))
            generated_image_paths.append(example.image_path)

    metrics = text_generation_metrics(generated_texts, reference_texts)
    metrics["generation_examples"] = len(generated_texts)
    if args.compute_clip:
        metrics["clip_score"] = clip_score(
            generated_texts=generated_texts,
            image_paths=generated_image_paths,
            clip_model_name=args.clip_model,
            device=args.device,
        )
    return metrics


def clean_caption(text: str) -> str:
    if CAPTION_START in text:
        text = text.split(CAPTION_START, maxsplit=1)[-1]
    if CAPTION_END in text:
        text = text.split(CAPTION_END, maxsplit=1)[0]
    text = text.replace(CAPTION_SEP, " ")
    for token in (CAPTION_START, CAPTION_END):
        text = text.replace(token, "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def text_generation_metrics(generated: list[str], references: list[str]) -> dict[str, float]:
    gen_tokens = [tokenize(text) for text in generated]
    ref_tokens = [tokenize(text) for text in references]
    unigram_precisions = [modified_precision(g, r, 1) for g, r in zip(gen_tokens, ref_tokens)]
    bigram_precisions = [modified_precision(g, r, 2) for g, r in zip(gen_tokens, ref_tokens)]
    lengths = [len(tokens) for tokens in gen_tokens]
    unsafe = [is_unsafe_caption(text) for text in generated]

    return {
        "bleu1_precision": mean(unigram_precisions) if unigram_precisions else 0.0,
        "bleu2_precision": mean(bigram_precisions) if bigram_precisions else 0.0,
        "distinct1": distinct_n(gen_tokens, 1),
        "distinct2": distinct_n(gen_tokens, 2),
        "repeated_3gram_rate": repeated_ngram_rate(gen_tokens, 3),
        "avg_generated_tokens": mean(lengths) if lengths else 0.0,
        "unsafe_generation_rate": sum(unsafe) / max(len(unsafe), 1),
    }


def modified_precision(generated: list[str], reference: list[str], n: int) -> float:
    gen_ngrams = Counter(ngrams(generated, n))
    ref_ngrams = Counter(ngrams(reference, n))
    total = sum(gen_ngrams.values())
    if total == 0:
        return 0.0
    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in gen_ngrams.items())
    return overlap / total


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokenized_texts: list[list[str]], n: int) -> float:
    all_ngrams = []
    for tokens in tokenized_texts:
        all_ngrams.extend(ngrams(tokens, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repeated_ngram_rate(tokenized_texts: list[list[str]], n: int) -> float:
    rates = []
    for tokens in tokenized_texts:
        grams = ngrams(tokens, n)
        if not grams:
            rates.append(0.0)
            continue
        rates.append(1.0 - (len(set(grams)) / len(grams)))
    return mean(rates) if rates else 0.0


@torch.no_grad()
def clip_score(generated_texts: list[str], image_paths: list[Path], clip_model_name: str, device: str) -> float:
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    scores = []
    for text, image_path in tqdm(list(zip(generated_texts, image_paths)), desc="clipscore"):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        scores.append(float((image_features * text_features).sum(dim=-1).item() * 100.0))
    return mean(scores) if scores else 0.0


if __name__ == "__main__":
    main()
