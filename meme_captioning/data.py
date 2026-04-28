from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MemeExample:
    template: str
    score: int
    caption: str
    image_path: Path


def load_template_image_map(dataset_dir: str | Path) -> dict[str, Path]:
    """Map exact template names in templates.txt to local image paths."""
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / "images"
    mapping: dict[str, Path] = {}

    with (dataset_dir / "templates.txt").open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad templates.txt row {line_no}: expected 3 tab-separated fields")
            template, _slug, url = parts
            mapping[template] = image_dir / url.rsplit("/", 1)[-1]

    return mapping


def iter_caption_rows(captions_path: str | Path) -> Iterable[tuple[str, int, str]]:
    captions_path = Path(captions_path)
    with captions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Bad caption row {captions_path}:{line_no}: expected 3 tab-separated fields")
            template, score, caption = parts
            yield template, int(score), caption


class MemeCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        caption_separator: str = "\n",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.caption_separator = caption_separator
        captions_path = self.dataset_dir / f"captions_{split}.txt"
        if split == "all":
            captions_path = self.dataset_dir / "captions.txt"

        template_to_image = load_template_image_map(self.dataset_dir)
        examples: list[MemeExample] = []
        missing_templates: set[str] = set()
        missing_images: set[Path] = set()

        for template, score, caption in iter_caption_rows(captions_path):
            image_path = template_to_image.get(template)
            if image_path is None:
                missing_templates.add(template)
                continue
            if not image_path.exists():
                missing_images.add(image_path)
                continue
            examples.append(
                MemeExample(
                    template=template,
                    score=score,
                    caption=self.normalize_caption(caption),
                    image_path=image_path,
                )
            )

        if missing_templates:
            sample = ", ".join(sorted(missing_templates)[:5])
            raise ValueError(f"{len(missing_templates)} caption templates are missing from templates.txt: {sample}")
        if missing_images:
            sample = ", ".join(str(p) for p in sorted(missing_images)[:5])
            raise FileNotFoundError(f"{len(missing_images)} template images are missing: {sample}")

        self.examples = examples

    def normalize_caption(self, caption: str) -> str:
        return caption.replace(" <sep> ", self.caption_separator).replace("<emp>", "").strip()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        example = self.examples[idx]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB")
        return {
            "image": image,
            "caption": example.caption,
            "template": example.template,
            "score": example.score,
        }

