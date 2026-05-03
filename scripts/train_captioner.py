from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, ViTImageProcessor, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meme_captioning.collator import MemeCaptionCollator
from meme_captioning.data import SPECIAL_TOKENS, MemeCaptionDataset
from meme_captioning.model import VisionPrefixCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="memes900k")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--vision-model", default="google/vit-base-patch16-224")
    parser.add_argument("--language-model", default="gpt2")
    parser.add_argument("--output-dir", default="checkpoints/meme-captioner")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--visual-prefix-length", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--filter-unsafe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-two-parts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--formatted-caption", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-language-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def maybe_limit(dataset: MemeCaptionDataset, limit: int):
    if limit and limit < len(dataset):
        return torch.utils.data.Subset(dataset, range(limit))
    return dataset


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = ViTImageProcessor.from_pretrained(args.vision_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.formatted_caption:
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    train_dataset = maybe_limit(
        MemeCaptionDataset(
            args.dataset_dir,
            split="train",
            image_dir=args.image_dir,
            min_score=args.min_score,
            filter_unsafe=args.filter_unsafe,
            require_two_parts=args.require_two_parts,
            formatted_caption=args.formatted_caption,
        ),
        args.limit_train,
    )
    val_dataset = maybe_limit(
        MemeCaptionDataset(
            args.dataset_dir,
            split="val",
            image_dir=args.image_dir,
            min_score=args.min_score,
            filter_unsafe=args.filter_unsafe,
            require_two_parts=args.require_two_parts,
            formatted_caption=args.formatted_caption,
        ),
        args.limit_val,
    )
    print(f"train_examples={len(train_dataset)} val_examples={len(val_dataset)}")
    collator = MemeCaptionCollator(image_processor, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=args.device == "cuda",
    )

    model = VisionPrefixCausalLM(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        visual_prefix_length=args.visual_prefix_length,
        freeze_vision=args.freeze_vision,
        freeze_language_model=args.freeze_language_model,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / step)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation"):
                batch = {key: value.to(args.device) for key, value in batch.items()}
                val_loss += model(**batch).loss.item()
                val_steps += 1
        val_loss = val_loss / max(val_steps, 1)
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "tokenizer_name": args.language_model,
            "image_processor_name": args.vision_model,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    tokenizer.save_pretrained(output_dir / "tokenizer")
    image_processor.save_pretrained(output_dir / "image_processor")


if __name__ == "__main__":
    main()
