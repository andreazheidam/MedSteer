"""Clinical Steering Residual (CSR) and steering-effect maps.

All functions take patch features of shape (..., N, D) and return per-patch maps (..., N).
Maps are kept at patch resolution; upsampling happens only for pixel-level metrics/plots.

- ``csr_abs``   R_i = ||F_i(x,p) - F_i(x)||_2                 (the plan's definition)
- ``csr_rel``   R_i / ||F_i(x)||_2                             (controls for feature norm)
- ``csr_cos``   1 - cos(F_i(x,p), F_i(x))                      (direction-only change)
- ``pair_map``  ||F_i(x,p) - F_i(x,q)||_2                      (specificity: p vs control q)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def csr_abs(Fp: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return (Fp - F0).norm(dim=-1)


def csr_rel(Fp: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return (Fp - F0).norm(dim=-1) / (F0.norm(dim=-1) + EPS)


def csr_cos(Fp: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(Fp, F0, dim=-1)


def pair_map(Fp: torch.Tensor, Fq: torch.Tensor) -> torch.Tensor:
    return (Fp - Fq).norm(dim=-1)


def pooled_residual(dF: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Weighted mean residual vector (B, D). weights (B, N) e.g. per-patch lesion fraction;
    rows with zero total weight fall back to the plain mean over all patches."""
    if weights is None:
        return dF.mean(dim=1)
    w = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)
    pooled = (dF * w.unsqueeze(-1)).sum(dim=1)
    empty = weights.sum(dim=1) <= 0
    if empty.any():
        pooled[empty] = dF[empty].mean(dim=1)
    return pooled
