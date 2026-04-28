from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meme_captioning.data import iter_caption_rows, load_template_image_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="memes900k")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    template_to_image = load_template_image_map(dataset_dir)
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    bad_images: list[tuple[Path, str]] = []

    for image_path in sorted(set(template_to_image.values())):
        try:
            with Image.open(image_path) as image:
                sizes[image.size] += 1
                modes[image.mode] += 1
        except Exception as exc:
            bad_images.append((image_path, str(exc)))

    print(f"templates: {len(template_to_image)}")
    print(f"images: {sum(sizes.values())}")
    print(f"unique_sizes: {len(sizes)}")
    print(f"top_sizes: {sizes.most_common(20)}")
    print(f"modes: {modes.most_common()}")
    print(f"bad_images: {len(bad_images)}")
    if bad_images:
        print(f"bad_image_examples: {bad_images[:10]}")

    for split in ("train", "val", "test"):
        captions_path = dataset_dir / f"captions_{split}.txt"
        rows = 0
        missing_templates: Counter[str] = Counter()
        for template, _score, _caption in iter_caption_rows(captions_path):
            rows += 1
            if template not in template_to_image:
                missing_templates[template] += 1
        print(f"{split}_rows: {rows}")
        print(f"{split}_missing_templates: {len(missing_templates)}")


if __name__ == "__main__":
    main()
