from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meme_captioning.model import VisionPrefixCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vit-gpt2-local-gpu/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint["args"]

    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint).parent / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = ViTImageProcessor.from_pretrained(Path(args.checkpoint).parent / "image_processor")
    model = VisionPrefixCausalLM(
        vision_model_name=train_args["vision_model"],
        language_model_name=train_args["language_model"],
        visual_prefix_length=train_args["visual_prefix_length"],
        freeze_vision=True,
        freeze_language_model=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    with Image.open(args.image) as image:
        image = image.convert("RGB")
    pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(args.device)

    for idx in range(args.num_captions):
        output_ids = model.generate(
            pixel_values=pixel_values,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"{idx + 1}. {text.strip()}")


if __name__ == "__main__":
    main()
