from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemeCaptionCollator:
    image_processor: object
    tokenizer: object
    max_length: int = 64

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        captions = [str(item["caption"]) for item in batch]

        pixel_batch = self.image_processor(images=images, return_tensors="pt")
        token_batch = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = token_batch["input_ids"].clone()
        labels[token_batch["attention_mask"] == 0] = -100

        return {
            "pixel_values": pixel_batch["pixel_values"],
            "input_ids": token_batch["input_ids"],
            "attention_mask": token_batch["attention_mask"],
            "labels": labels,
        }

