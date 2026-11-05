"""A tiny stand-in with the same interface as medsteer.model.SteerModel (no downloads).

Patch features are a fixed random projection of patch pixels. A prompt adds a
prompt-specific direction scaled by patch brightness and by how 'medical' the prompt is,
so on fake data (bright blobs = lesions) pathology prompts should localise and 'the cat'
should not. Used only to test the pipeline end to end.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


class FakeSteerModel:
    def __init__(self, resolution: int = 112, patch: int = 14, dim: int = 32):
        self.device = torch.device("cpu")
        self.resolution, self.patch, self.grid = resolution, patch, resolution // patch
        self.n_prefix = 1
        self.backbone_name = "fake"
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(3 * patch * patch, dim, generator=g) / patch
        self.head = torch.randn(dim, generator=g)
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.ones(1, 3, 1, 1)
        self._gate = 1.0

    def to_tensor(self, images):
        return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()

    def set_gate(self, w):
        self._gate = w

    def _patches(self, x):
        B, p = x.shape[0], self.patch
        u = x.unfold(2, p, p).unfold(3, p, p)  # B,3,g,g,p,p
        u = u.permute(0, 2, 3, 1, 4, 5).reshape(B, self.grid ** 2, -1)
        return u

    def _prompt_vec(self, text):
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        v = torch.randn(self.proj.shape[1], generator=torch.Generator().manual_seed(seed))
        medical = any(w in text for w in ("tumor", "lesion", "glioma", "meningioma", "pituitary",
                                          "benign", "malignant"))
        return v * (1.0 if medical else 0.15)

    @torch.no_grad()
    def tokens(self, x, prompt):
        pt = self._patches(x)
        feats = pt @ self.proj
        if prompt is not None and self._gate != 0:
            bright = pt.mean(dim=-1, keepdim=True)
            feats = feats + self._gate * bright * self._prompt_vec(prompt)
        cls = feats.mean(dim=1, keepdim=True)
        return torch.cat([cls, feats], dim=1)

    def split(self, tokens):
        return tokens[:, 0], tokens[:, 1:]

    @torch.no_grad()
    def native_logits(self, tokens):
        return tokens[:, 1:] @ self.head

    @torch.no_grad()
    def attention_map(self, x, prompt):
        return self._patches(x).mean(dim=-1)


class _FakeSteerNet(torch.nn.Module):
    """Trainable part of the fake model, with SteerViT-like parameter names
    (``connector.*``, ``*cross_attn*``, ``lin_seg_head.*``)."""

    def __init__(self, proj, head, patchify):
        super().__init__()
        self.patchify = patchify
        dim = proj.shape[1]
        self.register_buffer("proj", proj)
        self.connector = torch.nn.Linear(dim, dim)
        with torch.no_grad():
            self.connector.weight.copy_(torch.eye(dim))
            self.connector.bias.zero_()
        self.gated_cross_attn = torch.nn.Module()
        self.gated_cross_attn.attn_gate = torch.nn.Parameter(torch.tensor([1.0]))
        self.lin_seg_head = torch.nn.Linear(dim, 1)
        with torch.no_grad():
            self.lin_seg_head.weight.copy_(head[None])
            self.lin_seg_head.bias.zero_()

    def vision_model(self, x, text_vec):
        pt = self.patchify(x)
        feats = pt @ self.proj
        if text_vec is not None:
            bright = pt.mean(dim=-1, keepdim=True)
            feats = feats + torch.tanh(self.gated_cross_attn.attn_gate) * bright * self.connector(text_vec)
        return torch.cat([feats.mean(dim=1, keepdim=True), feats], dim=1)


class FakeTrainableSteerModel(FakeSteerModel):
    """FakeSteerModel with trainable steering modules and the TrainableSteer interface
    (select_trainable, param_groups, trainable_state, logits_grad, check_forward)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model = _FakeSteerNet(self.proj, self.head, self._patches)

    def select_trainable(self, train_head=False):
        names = []
        for n, p in self.model.named_parameters():
            p.requires_grad = "cross_attn" in n or n.startswith("connector.") or (
                train_head and n.startswith("lin_seg_head."))
            if p.requires_grad:
                names.append(n)
        return names

    def param_groups(self, *a, **k):
        return [{"params": [p for p in self.model.parameters() if p.requires_grad], "weight_decay": 0.0}]

    def trainable_state(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def _forward(self, x, prompt):
        tv = None if (prompt is None or self._gate == 0) else self._prompt_vec(prompt) * self._gate
        return self.model.vision_model(x, tv)

    def logits_grad(self, x, prompt):
        return self.model.lin_seg_head(self._forward(x, prompt)[:, 1:])[..., 0]

    @torch.no_grad()
    def tokens(self, x, prompt):
        return self._forward(x, prompt)

    @torch.no_grad()
    def native_logits(self, tokens):
        return self.model.lin_seg_head(tokens[:, 1:])[..., 0]

    @torch.no_grad()
    def check_forward(self, x, prompt):
        a, b = self.logits_grad(x, prompt), self.native_logits(self.tokens(x, prompt))
        return float((a - b).abs().max() / (b.abs().max() + 1e-8))
