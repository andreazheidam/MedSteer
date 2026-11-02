"""Statistics on per-image metrics. Every test is paired on the same images where possible;
effect sizes and bootstrap CIs are always reported next to p-values."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)


def bootstrap_ci(x: np.ndarray, stat=np.median, n: int = 5000, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for paired differences d."""
    d = d[(~np.isnan(d)) & (d != 0)]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def one_sample(x: np.ndarray, mu: float) -> dict:
    """Wilcoxon signed-rank of x against a reference value mu (e.g. AUROC 0.5, lift 1)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    d = x - mu
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 0 else np.nan
    lo, hi = bootstrap_ci(x)
    return {"n": len(x), "median": float(np.median(x)) if len(x) else np.nan,
            "ci_lo": lo, "ci_hi": hi, "ref": mu, "p": float(p), "r_rb": rank_biserial(d)}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank on paired samples a vs b (a - b > 0 means a is larger)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    lo, hi = bootstrap_ci(d)
    return {"n": int(ok.sum()), "median_diff": float(np.median(d)) if len(d) else np.nan,
            "ci_lo": lo, "ci_hi": hi, "p": float(p), "r_rb": rank_biserial(d)}


def friedman(wide: pd.DataFrame) -> dict:
    """Friedman test across columns (conditions) of a cases x conditions table."""
    w = wide.dropna()
    if w.shape[1] < 3 or len(w) < 3:
        return {"n": len(w), "chi2": np.nan, "p": np.nan, "kendall_w": np.nan}
    res = stats.friedmanchisquare(*[w[c].values for c in w.columns])
    k = w.shape[1]
    return {"n": len(w), "chi2": float(res.statistic), "p": float(res.pvalue),
            "kendall_w": float(res.statistic / (len(w) * (k - 1)))}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are kept and ignored)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def auroc_with_ci(scores: np.ndarray, labels: np.ndarray, n: int = 2000) -> dict:
    """Image-level AUROC (e.g. presence: tumour vs healthy) with a bootstrap CI."""
    from sklearn.metrics import roc_auc_score
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if y.all() or (~y).all():
        return {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "auroc": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(n):
        i = RNG.integers(0, len(s), len(s))
        if y[i].all() or (~y[i]).all():
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    return {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "auroc": float(auc),
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975))}


def spearman_ci(x: np.ndarray, y: np.ndarray, n: int = 2000) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return {"n": len(x), "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    res = stats.spearmanr(x, y)
    boots = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        boots.append(stats.spearmanr(x[i], y[i]).statistic)
    return {"n": len(x), "rho": float(res.statistic), "ci_lo": float(np.nanquantile(boots, 0.025)),
            "ci_hi": float(np.nanquantile(boots, 0.975)), "p": float(res.pvalue)}


def kruskal(groups: dict[str, np.ndarray]) -> dict:
    vals = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in groups.values()]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2:
        return {"H": np.nan, "p": np.nan}
    res = stats.kruskal(*vals)
    return {"H": float(res.statistic), "p": float(res.pvalue)}
