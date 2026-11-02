"""Prompt-free baseline maps. If the CSR map does not beat these, the prompt adds nothing
spatial beyond image salience / position."""
from __future__ import annotations

import numpy as np


def feature_norm(F0: np.ndarray) -> np.ndarray:
    """||F_i(x)|| per patch (N,)."""
    return np.linalg.norm(F0, axis=-1)


def salience(F0: np.ndarray) -> np.ndarray:
    """Distance of each patch from the image's mean patch feature (N,)."""
    return np.linalg.norm(F0 - F0.mean(axis=0, keepdims=True), axis=-1)


def center_gaussian(grid: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Centred isotropic Gaussian on the patch grid (N,)."""
    c = (np.arange(grid) + 0.5) / grid - 0.5
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_frac ** 2)).reshape(-1)
