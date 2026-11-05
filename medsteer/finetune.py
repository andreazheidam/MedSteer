"""Specificity-aware fine-tuning of SteerViT's steering modules (follow-up Sec. 2).

Trainable: the gated cross-attention blocks ("cross_attn" in the parameter name) and the
text->vision connector, exactly the modules SteerViT itself trains; optionally the linear
localisation head. The DINOv2 backbone and the RoBERTa text encoder stay frozen. Because the
text encoder is frozen and the prompts are a fixed set, its (normalised) token features are
computed once per prompt and cached; the connector is applied to them at every step.

Losses (patch grid, inside the evaluation region Omega; w = fraction of a patch in Omega):
  L_loc  = BCE(s+, frac)                         soft patch targets = lesion fraction
  q(s,M) = mean(s | M) - mean(s | Omega \\ M)      (patch-weighted)
  L_rank^(c) = [m - (q(s+) - q(s_c))]_+
  hard : L_spec = L_rank^(c*),  c* = argmax_c q(s_c)        (semantic hard-control mining)
  mean : L_spec = mean_c L_rank^(c)                          (mean-control ablation)
  L = L_loc + lambda_spec * L_spec  [+ lambda_pres * ||h_new(F(x)) - h_pre(F(x))||^2 if the head is trained]

q is computed on z-normalised maps by default (`q_space="z"`), the same normalisation as the
training-free analysis, so the margin m is in units of the map's standard deviation and the
loss cannot be satisfied by rescaling the logits.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from medsteer.model import SteerModel

EPS = 1e-6


# ---------------------------------------------------------------------------- losses
def znorm(s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """z-normalise maps s (B, N) with weights w (B, N) (fraction of each patch inside Omega)."""
    ws = w.sum(1, keepdim=True).clamp_min(EPS)
    mu = (w * s).sum(1, keepdim=True) / ws
    sd = ((w * (s - mu) ** 2).sum(1, keepdim=True) / ws).clamp_min(EPS).sqrt()
    return (s - mu) / sd


def to_q_space(s: torch.Tensor, w: torch.Tensor, space: str) -> torch.Tensor:
    if space == "z":
        return znorm(s, w)
    if space == "prob":
        return torch.sigmoid(s)
    if space == "logit":
        return s
    raise ValueError(space)


def q_score(s: torch.Tensor, frac: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """q(s, M) per sample: weighted mean inside the lesion minus weighted mean outside."""
    w_in, w_out = frac * w, (1 - frac) * w
    return ((w_in * s).sum(1) / w_in.sum(1).clamp_min(EPS)
            - (w_out * s).sum(1) / w_out.sum(1).clamp_min(EPS))


def loc_loss(logits: torch.Tensor, frac: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, frac, reduction="none")
    return ((w * bce).sum(1) / w.sum(1).clamp_min(EPS)).mean()


def spec_loss(q_plus: torch.Tensor, q_ctrl: torch.Tensor, margin: float, mode: str):
    """q_plus (B,), q_ctrl (B, C). Returns the loss and the index of the hardest control."""
    hard = q_ctrl.detach().argmax(1)
    if mode == "hard":
        qc = q_ctrl.gather(1, hard[:, None])[:, 0]
        return F.relu(margin - (q_plus - qc)).mean(), hard
    if mode == "mean":
        return F.relu(margin - (q_plus[:, None] - q_ctrl)).mean(), hard
    raise ValueError(mode)


# ---------------------------------------------------------------------------- model
class TrainableSteer(SteerModel):
    """SteerModel plus a differentiable, text-cached forward and trainable-parameter control."""

    def select_trainable(self, train_head: bool = False) -> list[str]:
        names = []
        for n, p in self.model.named_parameters():
            p.requires_grad = ("cross_attn" in n or n.startswith("connector.")
                               or (train_head and n.startswith("lin_seg_head.")))
            if p.requires_grad:
                names.append(n)
        self.model.eval()          # no dropout / drop-path anywhere; gradients still flow
        self._text_cache = {}
        return names

    def param_groups(self, weight_decay_connector=0.01, weight_decay_steer=0.1):
        con = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("connector.")]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("connector.")]
        return [{"params": con, "weight_decay": weight_decay_connector},
                {"params": rest, "weight_decay": weight_decay_steer}]

    def trainable_state(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def _text(self, prompt: str):
        if prompt not in self._text_cache:
            m = self.model
            tok = m.tokenizer([prompt], padding=True, truncation=True, max_length=512, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            feats = F.normalize(m.text_model(**tok).last_hidden_state, dim=-1)
            self._text_cache[prompt] = (feats, tok["attention_mask"].bool())
        return self._text_cache[prompt]

    def logits_grad(self, x: torch.Tensor, prompt: str | None) -> torch.Tensor:
        """Native localisation logits (B, N) with gradients to the trainable modules.
        Identical to ``model(x, [prompt] * B, return_segmentation_logits=True)``."""
        m = self.model
        B = x.shape[0]
        ctx = (torch.autocast("cuda", dtype=self.amp_dtype)
               if self.amp_dtype is not None and self.device.type == "cuda" else _null())
        with ctx:
            if prompt is None:
                feats = m.vision_model(x, None)
            else:
                tf, tmask = self._text(prompt)
                tf = m.connector(tf.expand(B, -1, -1))
                mask = torch.cat([torch.ones(B, m.num_img_tokens, dtype=torch.bool, device=x.device),
                                  tmask.expand(B, -1)], dim=-1)
                feats = m.vision_model(x, tf, attn_mask=mask)
            return m.get_heatmap_logits(feats).float()

    @torch.no_grad()
    def check_forward(self, x: torch.Tensor, prompt: str) -> float:
        """Max relative difference between the cached forward and the official forward."""
        a = self.logits_grad(x, prompt)
        b = self.native_logits(self.tokens(x, prompt))
        return float((a - b).abs().max() / (b.abs().max() + 1e-8))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def load_trainable(model_module: torch.nn.Module, state: dict) -> int:
    """Load a fine-tuned trainable-parameter state into a SteerViT module; every key must exist."""
    own = dict(model_module.named_parameters())
    missing = [k for k in state if k not in own]
    if missing:
        raise KeyError(f"{len(missing)} fine-tuned parameters not in the model, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    return len(state)


# ---------------------------------------------------------------------------- schedule
def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


# ---------------------------------------------------------------------------- data
def flip_batch(imgs: np.ndarray, grids: list[np.ndarray], rng: np.random.Generator):
    """Random horizontal / vertical flips, the same for the image and its patch grids."""
    imgs = imgs.copy()
    grids = [g.copy() for g in grids]
    for i in range(len(imgs)):
        if rng.random() < 0.5:
            imgs[i] = imgs[i][:, ::-1]
            for g in grids:
                g[i] = g[i][:, ::-1]
        if rng.random() < 0.5:
            imgs[i] = imgs[i][::-1]
            for g in grids:
                g[i] = g[i][::-1]
    return imgs, grids
