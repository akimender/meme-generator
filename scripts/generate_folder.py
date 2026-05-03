from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vit-gpt2-clean/best.pt")
    parser.add_argument("--image-dir", default="memes900k/oodimages_224")
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    generator = Path(__file__).resolve().parent / "generate_caption.py"
    for image_path in image_paths:
        print(f"\n=== {image_path.name} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--checkpoint",
                args.checkpoint,
                "--image",
                str(image_path),
                "--num-captions",
                str(args.num_captions),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--device",
                args.device,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
