from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="memes900k/images")
    parser.add_argument("--output-dir", default="memes900k/images_224")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--mode",
        choices=("resize", "pad"),
        default="resize",
        help="resize warps to a square; pad preserves aspect ratio with padding.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in input_dir.iterdir() if path.is_file())
    written = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for image_path in tqdm(image_paths, desc="resizing"):
        output_path = output_dir / image_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if args.mode == "pad":
                    image = ImageOps.contain(image, (args.size, args.size), Image.Resampling.BICUBIC)
                    canvas = Image.new("RGB", (args.size, args.size), (0, 0, 0))
                    offset = ((args.size - image.width) // 2, (args.size - image.height) // 2)
                    canvas.paste(image, offset)
                    image = canvas
                else:
                    image = image.resize((args.size, args.size), Image.Resampling.BICUBIC)
                image.save(output_path, quality=95, optimize=True)
                written += 1
        except Exception as exc:
            failed.append((image_path, str(exc)))

    print(f"input_images: {len(image_paths)}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failed)}")
    if failed:
        print(f"failed_examples: {failed[:10]}")


if __name__ == "__main__":
    main()
