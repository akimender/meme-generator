from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, ViTModel


class VisionPrefixCausalLM(nn.Module):
    """ViT encoder + projection prefix + autoregressive language model."""

    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        language_model_name: str = "gpt2",
        visual_prefix_length: int = 8,
        freeze_vision: bool = True,
        freeze_language_model: bool = False,
    ) -> None:
        super().__init__()
        self.visual_prefix_length = visual_prefix_length
        self.vision = ViTModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)

        if freeze_vision:
            for param in self.vision.parameters():
                param.requires_grad = False
        if freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False

        vision_hidden = self.vision.config.hidden_size
        lm_hidden = self.language_model.config.hidden_size
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(vision_hidden),
            nn.Linear(vision_hidden, lm_hidden * visual_prefix_length),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        vision_outputs = self.vision(pixel_values=pixel_values)
        cls_features = vision_outputs.last_hidden_state[:, 0]
        visual_embeds = self.visual_projection(cls_features)
        visual_embeds = visual_embeds.view(
            pixel_values.size(0),
            self.visual_prefix_length,
            self.language_model.config.hidden_size,
        )

        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, token_embeds], dim=1)

        visual_attention = torch.ones(
            input_ids.size(0),
            self.visual_prefix_length,
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
            device=input_ids.device,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            visual_labels = torch.full(
                (labels.size(0), self.visual_prefix_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        prompt: str = "",
        max_new_tokens: int = 32,
        **generate_kwargs,
    ) -> torch.Tensor:
        self.eval()
        encoded = tokenizer(prompt, return_tensors="pt").to(pixel_values.device)
        if encoded["input_ids"].numel() == 0:
            encoded = tokenizer(tokenizer.bos_token or "", return_tensors="pt").to(pixel_values.device)

        vision_outputs = self.vision(pixel_values=pixel_values)
        cls_features = vision_outputs.last_hidden_state[:, 0]
        visual_embeds = self.visual_projection(cls_features).view(
            pixel_values.size(0),
            self.visual_prefix_length,
            self.language_model.config.hidden_size,
        )
        token_embeds = self.language_model.get_input_embeddings()(encoded["input_ids"])
        inputs_embeds = torch.cat([visual_embeds, token_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=pixel_values.device)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )

