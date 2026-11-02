"""Combined paper figures from the BUSI and BRISC runs (after evaluate.py).

  fig_auroc.pdf   per-image pixel AUROC of every map/prompt, both datasets (Fig. 2)
  fig_size.pdf    AUROC vs lesion area: head without vs with prompt (Fig. 3)
  fig_panels.pdf  qualitative examples, 2 cases per dataset (Fig. 1, right)

Usage:
  python scripts/paper_figs.py --runs runs/steerdinov2/gate1/busi runs/steerdinov2/gate1/brisc \
      --manifest data/manifest.csv --out paper/figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import load_image, read_manifest  # noqa: E402
from medsteer.metrics import upsample  # noqa: E402
from medsteer.plotting import INK, INK2, GRID, SEQ, plt  # noqa: E402

DS_COLOR = {"busi": "#2a78d6", "brisc": "#eb6834", "kvasir": "#1baf7a", "sunnybrook": "#9b5de5"}
DS_NAME = {"busi": "BUSI (breast US)", "brisc": "BRISC (brain MRI)",
           "kvasir": "Kvasir-SEG (endoscopy)", "sunnybrook": "Sunnybrook (cardiac MRI)"}
DS_SHORT = {"busi": "BUSI (US)", "brisc": "BRISC (MRI)", "kvasir": "Kvasir (endosc.)",
            "sunnybrook": "Sunnybrook (MRI)"}

# (label, map, category or prompt id)  -- rows of the AUROC figure, top to bottom
ROWS = [
    ("SteerViT map, no prompt", "native0", "none"),
    ("SteerViT map + target prompt", "native", "pathology"),
    ("+ anatomy prompt", "native", "anatomy"),
    ("+ related-wrong prompt", "native", "related_wrong"),
    ('+ "the cat"', "native", "irrelevant"),
    ("prompt term w$^\\top\\Delta F$ (target)", "native_delta", "pathology"),
    ("CSR $\\|\\Delta F\\|$ (target)", "csr_abs", "pathology"),
    ("salience (no model prompt)", "baseline_salience", "none"),
    ("centre prior", "baseline_center", "none"),
]


def per_case(loc: pd.DataFrame, m: str, cat: str, metric: str = "auroc") -> pd.Series:
    d = loc[(loc["map"] == m) & (loc["category"] == cat) & loc["primary"]]
    return d.groupby("case_id")[metric].mean()


def fig_auroc(locs: dict, out: Path):
    dss = list(locs)
    step = min(0.34, 0.84 / len(dss))   # 2 datasets: unchanged; 4: thinner boxes, no overlap
    fig, ax = plt.subplots(figsize=(3.45, 3.1 + 0.45 * max(0, len(dss) - 2)))
    n = len(ROWS)
    off = {ds: -(k - (len(dss) - 1) / 2) * step for k, ds in enumerate(dss)}
    for ds, loc in locs.items():
        vals = [per_case(loc, m, c).dropna().values for _, m, c in ROWS]
        pos = [n - 1 - i + off[ds] for i in range(n)]
        keep = [i for i, v in enumerate(vals) if len(v)]
        bp = ax.boxplot([vals[i] for i in keep], positions=[pos[i] for i in keep], orientation="horizontal",
                        widths=0.82 * step, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.1})
        for b in bp["boxes"]:
            b.set(facecolor=DS_COLOR[ds], edgecolor="white", linewidth=0.8, alpha=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=INK2, linewidth=0.7)
        ax.plot([], [], color=DS_COLOR[ds], lw=6, label=f"{DS_NAME[ds]}, n={loc.case_id.nunique()}")
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("per-image pixel AUROC vs. target mask")
    ax.grid(axis="y", visible=False)
    for y in (n - 1 - 4.5, n - 1 - 6.5):   # separate head / residual / prompt-free groups
        ax.axhline(y, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=6.5)
    fig.savefig(out / "fig_auroc.pdf")
    fig.savefig(out / "fig_auroc.png", dpi=200)
    plt.close(fig)


def fig_size(locs: dict, out: Path):
    k = len(locs)
    ncol = k if k <= 2 else 2
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.45, 1.75 * nrow), sharey=True, squeeze=False)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    axes = axes.ravel()
    styles = [("native0", "none", "no prompt", "#9a9893", "--"),
              ("native", "pathology", "target prompt", None, "-")]
    for ax, (ds, loc) in zip(axes, locs.items()):
        size = loc.groupby("case_id")["lesion_frac"].first()
        for m, c, lab, col, ls in styles:
            s = per_case(loc, m, c).reindex(size.index).dropna()
            x = 100 * size.loc[s.index].values
            col = col or DS_COLOR[ds]
            ax.scatter(x, s.values, s=4, color=col, alpha=0.25, linewidths=0)
            q = np.quantile(x, np.linspace(0, 1, 7))
            mids, meds = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                sel = (x >= lo) & (x <= hi)
                if sel.sum() >= 5:
                    mids.append(np.sqrt(lo * hi)); meds.append(np.median(s.values[sel]))
            ax.plot(mids, meds, color=col, lw=2, ls=ls, label=lab)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(DS_NAME[ds] if k <= 2 else DS_SHORT[ds], fontsize=7)
        ax.set_xlabel("target area (% of image)", fontsize=7)
        ax.set_ylim(0, 1)
    for i in range(0, k, ncol):
        axes[i].set_ylabel("pixel AUROC")
    fig.tight_layout(pad=0.3)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(out / "fig_size.pdf")
    fig.savefig(out / "fig_size.png", dpi=200)
    plt.close(fig)


PANEL_COLS = [("image", None, None), ("SteerViT map, no prompt", "native0", None),
              ("+ target prompt", "native", "pathology"),
              ("prompt term w$^\\top\\Delta F$", "native_delta", "pathology"),
              ("CSR, target prompt", "csr_abs", "pathology"),
              ('CSR, "the cat"', "csr_abs", "irrelevant_cat")]


def pick(loc: pd.DataFrame, k: int = 2) -> list[str]:
    """Representative cases: the median-AUROC case (head + lesion prompt) in the smallest and
    largest lesion-size bin. Chosen by rule, not by eye."""
    d = per_case(loc, "native", "pathology")
    bins = loc.groupby("case_id")["size_bin"].first()
    out = []
    for b in (["small", "large"] if k == 2 else ["small", "medium", "large"])[:k]:
        s = d[bins.reindex(d.index) == b].sort_values()
        if len(s):
            out.append(s.index[len(s) // 2])
    return out


def fig_panels(runs: dict, locs: dict, cases: dict, out: Path, per_ds: int = 2):
    rows = [(ds, cid) for ds in runs for cid in pick(locs[ds], per_ds)]
    nc = len(PANEL_COLS)
    fig, axes = plt.subplots(len(rows), nc, figsize=(7.0, 1.2 * len(rows) + 0.25))
    axes = np.atleast_2d(axes)
    for i, (ds, cid) in enumerate(rows):
        run = runs[ds]
        R = json.loads((run / "meta.json").read_text())["resolution"]
        z = np.load(run / "cases" / f"{cid}.npz")
        pids = list(z["prompt_ids"])
        mask, valid = z["mask"], z["valid"]
        vgrid = (z["valid_frac"] >= 0.5).ravel()
        hide = lambda a: np.ma.masked_where(~valid, upsample(a, R))
        csr = [z["csr_abs"][pids.index(p)] for _, m, p in PANEL_COLS if m == "csr_abs"]
        lo = min(np.quantile(c[vgrid], 0.01) for c in csr)
        hi = max(np.quantile(c[vgrid], 0.99) for c in csr)
        for k, (title, m, p) in enumerate(PANEL_COLS):
            ax = axes[i, k]
            ax.set_facecolor("#d9d9d9")
            if m is None:
                ax.imshow(load_image(cases[cid].image_path, R))
            else:
                if m == "native0":
                    arr = z["native0"]
                elif m == "native_delta":   # stored implicitly: w.dF = native(x,p) - native(x)
                    arr = z["native"][pids.index(p)] - z["native0"]
                else:
                    arr = z[m][pids.index(p)]
                if m == "csr_abs":   # shared scale: magnitudes comparable across prompts
                    ax.imshow(hide(arr), cmap=SEQ, vmin=lo, vmax=hi)
                else:
                    ax.imshow(hide(arr), cmap=SEQ, vmin=np.quantile(arr[vgrid], 0.01),
                              vmax=np.quantile(arr[vgrid], 0.99))
            ax.contour(mask, levels=[0.5], colors="#1baf7a" if m is None else "white", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=6.5)
        axes[i, 0].set_ylabel(f"{ds.upper()}\n{cases[cid].label.replace('_', ' ')}, "
                              f"{100 * mask.mean():.1f}%", fontsize=6)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(out / "fig_panels.pdf")
    fig.savefig(out / "fig_panels.png", dpi=200)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, locs = {}, {}
    for r in args.runs:
        r = Path(r)
        ds = json.loads((r / "meta.json").read_text())["dataset"]
        runs[ds], locs[ds] = r, pd.read_csv(r / "loc.csv")
    cases = {c.case_id: c for c in read_manifest(args.manifest)}
    fig_auroc(locs, out)
    fig_size(locs, out)
    rows = fig_panels(runs, locs, cases, out)
    print("wrote fig_auroc, fig_size, fig_panels (cases:", [c for _, c in rows], ") ->", out)


if __name__ == "__main__":
    main()
