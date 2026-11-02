"""Thin wrapper around the released SteerViT (github.com/JonaRuthardt/SteerViT).

Facts from the official code this wrapper relies on:
- ``model.forward(images, texts=None)`` skips every gated cross-attention block, so the
  output is exactly the frozen backbone (timm DINOv2). That is our F(x).
- ``model.forward(images, texts)`` returns prompt-conditioned tokens (prefix + patches).
- The native heatmap is ``softmax`` over patches of ``lin_seg_head(patch tokens)``; we keep
  the logits (rank-equivalent within an image, and comparable across prompts).
- ``model.set_gate_factor(w)`` scales the gate parameters (tanh(w * g)); w = 0 removes all
  text influence, w = 1 is the released model.
"""
from __future__ import annotations

import numpy as np
import torch


class SteerModel:
    def __init__(self, checkpoint: str = "steervit_dinov2_base.pth", device: str | None = None,
                 dtype: str = "float32"):
        from steervit import SteerViT  # official package

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SteerViT.from_pretrained(checkpoint, device=self.device)
        self.amp_dtype = {"float32": None, "bfloat16": torch.bfloat16,
                          "float16": torch.float16}[dtype]
        trunk = self.model.vision_model.trunk
        self.n_prefix = trunk.num_prefix_tokens
        self.resolution = self.model.image_size[0]
        self.patch = self.model.patch_size
        self.grid = self.resolution // self.patch
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=trunk)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.backbone_name = self.model.config["vision_encoder"]["model_name"]
        self._gate = 1.0

    # ------------------------------------------------------------------ preprocessing
    def to_tensor(self, images: np.ndarray) -> torch.Tensor:
        """(B, H, W, 3) float in [0, 1], already at ``self.resolution`` -> normalised tensor."""
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
        return ((x - self.mean) / self.std).to(self.device)

    # ------------------------------------------------------------------ forward
    def set_gate(self, w: float) -> None:
        if w != self._gate:
            self.model.set_gate_factor(w)
            self._gate = w

    @torch.no_grad()
    def tokens(self, x: torch.Tensor, prompt: str | None) -> torch.Tensor:
        """All output tokens (B, prefix + N, D). ``prompt=None`` -> unprompted backbone."""
        texts = None if prompt is None else [prompt] * x.shape[0]
        if self.amp_dtype is not None and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.amp_dtype):
                out = self.model(x, texts)
        else:
            out = self.model(x, texts)
        return out.float()

    def split(self, tokens: torch.Tensor):
        """-> cls (B, D) or None, patches (B, N, D)."""
        cls = tokens[:, 0] if self.n_prefix > 0 else None
        return cls, tokens[:, self.n_prefix:]

    @torch.no_grad()
    def native_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Native SteerViT localisation logits (B, N) from the released linear head."""
        return self.model.get_heatmap_logits(tokens).float()

    @torch.no_grad()
    def attention_map(self, x: torch.Tensor, prompt: str | None) -> torch.Tensor:
        """Official CLS-attention heatmap (B, grid, grid), min-max normalised, sinks removed."""
        texts = None if prompt is None else [prompt] * x.shape[0]
        hm = self.model.get_attention_heatmaps(x, texts=texts, interpolate=False)
        return hm.float()
