"""Paper figures (matplotlib -> PDF). Colours follow one fixed role assignment everywhere:
the same map type always has the same colour, controls/baselines are gray."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROLE = {"csr_abs": "#2a78d6", "native": "#eb6834", "attn": "#1baf7a", "spec": "#6250d6",
        "baseline": "#9a9893", "control": "#bdbbb5"}
CAT_ORDER = ["pathology", "class_correct", "class_wrong", "anatomy", "related_wrong",
             "nuisance", "irrelevant", "contrast_pos", "contrast_neg", "negation"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
SEQ = "magma"  # perceptually uniform, monotone lightness, for steering-effect maps

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def box_by_category(ax, data: dict[str, dict[str, np.ndarray]], ylabel: str, ref: float | None):
    """data[group][map_type] -> values. Groups on x, map types as side-by-side boxes."""
    groups = list(data)
    maps = sorted({m for g in data.values() for m in g}, key=lambda m: list(ROLE).index(_role(m)))
    width = 0.8 / max(1, len(maps))
    for k, m in enumerate(maps):
        pos = [i + (k - (len(maps) - 1) / 2) * width for i in range(len(groups))]
        vals = [data[g].get(m, np.array([np.nan])) for g in groups]
        vals = [v[~np.isnan(v)] if len(v[~np.isnan(v)]) else np.array([np.nan]) for v in vals]
        bp = ax.boxplot(vals, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, medianprops={"color": INK, "linewidth": 1})
        for b in bp["boxes"]:
            b.set(facecolor=ROLE[_role(m)], edgecolor="white", linewidth=1, alpha=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=INK2, linewidth=0.8)
        ax.plot([], [], color=ROLE[_role(m)], lw=6, label=_pretty(m))
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("_", " ") for g in groups], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
              handlelength=1.2, columnspacing=1.0)


def size_scatter(ax, df, x: str, y: str, hue_col: str, ref: float | None = 0.5):
    for name, sub in df.groupby(hue_col):
        c = ROLE[_role(name)]
        ax.scatter(sub[x], sub[y], s=8, color=c, alpha=0.35, linewidths=0, label=_pretty(name))
        # running median in 8 quantile bins of x
        q = np.quantile(sub[x], np.linspace(0, 1, 9))
        mids, meds = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            s = sub[(sub[x] >= lo) & (sub[x] <= hi)][y]
            if len(s):
                mids.append(np.sqrt(lo * hi) if lo > 0 else hi / 2)
                meds.append(s.median())
        ax.plot(mids, meds, color=c, lw=2)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.legend(frameon=False)


def cos_matrix(ax, mat, labels, title=""):
    im = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([l.replace("_", " ") for l in labels])
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color=INK)
    ax.set_title(title)
    return im


def _role(name: str) -> str:
    if name in ROLE:
        return name
    if name == "baseline_center":
        return "control"
    if name.startswith("baseline"):
        return "baseline"
    if name in ("attn0",):
        return "attn"
    if name == "native_delta":
        return "spec"
    if name == "native0":
        return "baseline"
    if name.startswith("csr"):
        return "csr_abs"
    return "control"


def _pretty(name: str) -> str:
    return {"csr_abs": "CSR", "csr_rel": "CSR (rel.)", "csr_cos": "CSR (cos)", "native": "native head",
            "attn": "CLS attention", "spec": "CSR specificity", "baseline_norm": "feature norm",
            "baseline_salience": "salience", "baseline_center": "centre prior",
            "native_delta": "head on residual", "native0": "head, no prompt"}.get(name, name)
