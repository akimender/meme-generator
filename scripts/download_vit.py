from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import ViTImageProcessor, ViTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/vit-base-patch16-224")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory, e.g. /scratch/$USER/hf-cache.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional directory to materialize the model files, e.g. models/vit-base-patch16-224.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    local_dir = Path(args.local_dir).expanduser() if args.local_dir else None

    snapshot_path = snapshot_download(
        repo_id=args.model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        local_dir_use_symlinks=False if local_dir else "auto",
    )

    load_path = str(local_dir or snapshot_path)
    ViTImageProcessor.from_pretrained(load_path)
    model = ViTModel.from_pretrained(load_path)

    print(f"downloaded: {args.model_name}")
    print(f"files: {load_path}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"image_size: {model.config.image_size}")
    print(f"patch_size: {model.config.patch_size}")


if __name__ == "__main__":
    main()
