"""Localisation metrics for one map against one ground-truth mask.

Two evaluation resolutions:
- pixel: the patch map is bilinearly upsampled to the mask resolution (plan: pixel AUROC/AUPRC)
- patch: a patch counts as lesion if its lesion fraction >= ``patch_thr`` (default 0.5),
  falling back to "touches the lesion" (> 0) when no patch reaches the threshold, so
  lesions smaller than one patch are still scored.

All metrics are threshold-free except Dice, which uses a fixed per-map Otsu threshold
(main result) and the best threshold (``dice_oracle``; report only as an upper bound).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-8


def upsample(patch_map: np.ndarray, size: int) -> np.ndarray:
    g = int(round(np.sqrt(patch_map.size)))
    t = torch.from_numpy(patch_map.reshape(1, 1, g, g).astype(np.float32))
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def patch_labels(frac: np.ndarray, thr: float = 0.5) -> np.ndarray:
    lab = frac.reshape(-1) >= thr
    if not lab.any():
        lab = frac.reshape(-1) > 0
    return lab


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    x = x.reshape(-1)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return hi
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + EPS))


def dice_oracle(score: np.ndarray, gt: np.ndarray, n: int = 64) -> float:
    qs = np.quantile(score, np.linspace(0.5, 0.995, n))
    return max(dice(score >= q, gt) for q in qs)


def css(score: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Clinical Steerability Score: (ratio, difference) of mean map value inside vs outside."""
    inside, outside = score[gt].mean(), score[~gt].mean()
    return float(inside / (outside + EPS)), float(inside - outside)


def localisation(patch_map: np.ndarray, mask: np.ndarray, frac: np.ndarray,
                 patch_thr: float = 0.5, with_dice: bool = True,
                 valid: np.ndarray | None = None, valid_frac: np.ndarray | None = None) -> dict:
    """All metrics for one (map, mask) pair. ``mask`` must contain lesion pixels.
    ``valid`` / ``valid_frac``: evaluation region (pixels / per-patch fraction), defined
    independently of the ground truth (``data.content_region``: the unpadded image). Pixels and
    patches outside it are excluded from every metric; target pixels are NOT added back.
    ``gt_outside_valid`` reports target pixels outside the region (0 for content_region)."""
    pm_all = patch_map.reshape(-1).astype(np.float64)
    pix = upsample(pm_all, mask.shape[0]).reshape(-1)
    gt = mask.reshape(-1)
    n_out = 0
    if valid is not None:
        keep = valid.reshape(-1)
        n_out = int((gt & ~keep).sum())
        pix, gt = pix[keep], gt[keep]
    pm = pm_all
    prev = float(gt.mean())
    out = {"prevalence": prev, "gt_outside_valid": n_out}
    if not gt.any() or gt.all():   # no target or no background inside the region
        out.update(auroc=np.nan, auprc=np.nan, auprc_lift=np.nan)
        out.update(pointing(pm_all, frac, valid_frac))
        return out
    if np.ptp(pix) < EPS:  # constant map: chance level by definition
        out.update(auroc=0.5, auprc=prev, auprc_lift=1.0)
    else:
        out["auroc"] = float(roc_auc_score(gt, pix))
        out["auprc"] = float(average_precision_score(gt, pix))
        out["auprc_lift"] = out["auprc"] / prev
    lab = patch_labels(frac, patch_thr)
    if valid_frac is not None:
        keep_p = valid_frac.reshape(-1) >= 0.5
        pm, lab = pm[keep_p], lab[keep_p]
    out["patch_prevalence"] = float(lab.mean()) if len(lab) else np.nan
    if not lab.any() or lab.all():
        out.update(auroc_patch=np.nan, auprc_patch=np.nan)
    elif np.ptp(pm) < EPS:
        out.update(auroc_patch=np.nan if lab.all() else 0.5,
                   auprc_patch=np.nan if lab.all() else float(lab.mean()))
    else:
        out["auroc_patch"] = float(roc_auc_score(lab, pm))
        out["auprc_patch"] = float(average_precision_score(lab, pm))
    out["css_ratio"], out["css_diff"] = css(pix, gt)
    out.update(pointing(pm_all, frac, valid_frac))
    if with_dice:
        out["dice_otsu"] = dice(pix >= otsu_threshold(pix), gt)
        out["dice_oracle"] = dice_oracle(pix, gt)
    return out


def pointing(patch_map: np.ndarray, frac: np.ndarray, valid_frac: np.ndarray | None = None) -> dict:
    """Pointing game: does the map's single highest patch touch the target?

    hit = 1 if the arg-max patch (among image-content patches) contains any target pixel.
    pointing_chance = the hit rate of a randomly chosen patch (share of eligible patches that
    touch the target), so hit rates can be compared to chance per image. Unlike AUROC this is
    not inflated by a broad map that merely separates the organ from a large background."""
    pm = patch_map.reshape(-1).astype(np.float64)
    tgt = frac.reshape(-1) > 0
    elig = np.ones_like(tgt) if valid_frac is None else (valid_frac.reshape(-1) >= 0.5)
    if not elig.any():
        return {"pointing": np.nan, "pointing_chance": np.nan}
    chance = float(tgt[elig].mean())
    if np.ptp(pm[elig]) < EPS:   # constant map: no preferred patch
        return {"pointing": chance, "pointing_chance": chance}
    idx = np.flatnonzero(elig)[np.argmax(pm[elig])]
    return {"pointing": float(tgt[idx]), "pointing_chance": chance}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    a, b = a.reshape(-1), b.reshape(-1)
    if np.ptp(a) < EPS or np.ptp(b) < EPS:
        return np.nan
    return float(spearmanr(a, b).statistic)
