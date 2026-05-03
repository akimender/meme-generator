from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

CAPTION_START = "<caption>"
CAPTION_SEP = "<sep>"
CAPTION_END = "<end>"
SPECIAL_TOKENS = [CAPTION_START, CAPTION_SEP, CAPTION_END]

BLOCKED_PATTERNS = [
    r"\bf+u+c+k+\w*\b",
    r"\bs+h+i+t+\w*\b",
    r"\bb+i+t+c+h+\w*\b",
    r"\ba+s+s+h*o*l*e*\b",
    r"\bd+i+c+k+\w*\b",
    r"\bc+o+c+k+\w*\b",
    r"\bp+u+s+s+y+\w*\b",
    r"\bc+u+m+\w*\b",
    r"\bwhore\w*\b",
    r"\bslut\w*\b",
    r"\bporn\w*\b",
    r"\brape\w*\b",
    r"\bhand\s*job\w*\b",
    r"\bblow\s*job\w*\b",
    r"\bboob\w*\b",
    r"\btit+s*\b",
    r"\bpenis\w*\b",
    r"\bvagina\w*\b",
    r"\bfag+\w*\b",
    r"\bgay\b",
    r"\bretard\w*\b",
    r"\bn+i+g+g+\w*\b",
    r"\bnazi\w*\b",
    r"\bhitler\w*\b",
    r"\bholocaust\b",
    r"\bjew\w*\b",
    r"\bkkk\b",
    r"\bkill\s+yourself\b",
]

BLOCKED_RE = re.compile("|".join(BLOCKED_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class MemeExample:
    template: str
    score: int
    caption: str
    image_path: Path


def load_template_image_map(dataset_dir: str | Path, image_dir: str | Path | None = None) -> dict[str, Path]:
    """Map exact template names in templates.txt to local image paths."""
    dataset_dir = Path(dataset_dir)
    image_dir = Path(image_dir) if image_dir else dataset_dir / "images"
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
        image_dir: str | Path | None = None,
        min_score: int | None = None,
        filter_unsafe: bool = False,
        require_two_parts: bool = False,
        formatted_caption: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.caption_separator = caption_separator
        self.formatted_caption = formatted_caption
        captions_path = self.dataset_dir / f"captions_{split}.txt"
        if split == "all":
            captions_path = self.dataset_dir / "captions.txt"

        template_to_image = load_template_image_map(self.dataset_dir, image_dir=image_dir)
        examples: list[MemeExample] = []
        missing_templates: set[str] = set()
        missing_images: set[Path] = set()

        for template, score, caption in iter_caption_rows(captions_path):
            if min_score is not None and score < min_score:
                continue
            if filter_unsafe and is_unsafe_caption(caption):
                continue
            if require_two_parts and not has_two_nonempty_parts(caption):
                continue

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
        parts = normalize_caption_parts(caption)
        if self.formatted_caption:
            top, bottom = (parts + [""])[:2]
            return f"{CAPTION_START} {top} {CAPTION_SEP} {bottom} {CAPTION_END}"
        return self.caption_separator.join(parts).strip()

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


def normalize_caption_parts(caption: str) -> list[str]:
    parts = caption.split(" <sep> ")
    normalized = []
    for part in parts:
        part = part.replace("<emp>", "")
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            normalized.append(part)
    return normalized


def has_two_nonempty_parts(caption: str) -> bool:
    return len(normalize_caption_parts(caption)) >= 2


def is_unsafe_caption(caption: str) -> bool:
    return BLOCKED_RE.search(caption) is not None
