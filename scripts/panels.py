"""Qualitative figure: image | GT mask | native heatmap | CSR maps for pathology, anatomy,
related-but-wrong and irrelevant prompts. CSR columns share one colour scale per case, so
magnitudes are comparable across prompts (the plan's requirement).

By default picks, per size bin, the case with median pathology-CSR AUROC (representative,
not cherry-picked). Pass --cases to choose explicitly.

Usage:
  python scripts/panels.py --run runs/steerdinov2/gate1/busi --out paper/figs
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
from medsteer.plotting import SEQ, plt  # noqa: E402

COLUMNS = [("native0", None, "native head\n(no prompt)"),
           ("native", "pathology", "native head\n(pathology)"),
           ("csr_abs", "pathology", "CSR\npathology"),
           ("csr_abs", "anatomy", "CSR\nanatomy"),
           ("csr_abs", "related_wrong", "CSR\nrelated-wrong"),
           ("csr_abs", "irrelevant_cat", "CSR\nirrelevant")]


def pick_cases(run: Path) -> list[str]:
    loc = pd.read_csv(run / "loc.csv")
    d = loc[(loc["map"] == "csr_abs") & (loc["prompt"] == "pathology")]
    picks = []
    for b in ("small", "medium", "large"):
        s = d[d["size_bin"] == b].sort_values("auroc")
        if len(s):
            picks.append(s.iloc[len(s) // 2]["case_id"])
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    R = meta["resolution"]
    cases = {c.case_id: c for c in read_manifest(args.manifest, datasets=[meta["dataset"]])}
    ids = args.cases or pick_cases(run)

    fig, axes = plt.subplots(len(ids), 2 + len(COLUMNS), figsize=(1.25 * (2 + len(COLUMNS)), 1.3 * len(ids)))
    axes = np.atleast_2d(axes)
    for i, cid in enumerate(ids):
        z = np.load(run / "cases" / f"{cid}.npz")
        pids = list(z["prompt_ids"])
        img, mask = load_image(cases[cid].image_path, R), z["mask"]
        valid = z["valid"] if "valid" in z else np.ones_like(mask)
        vgrid = (z["valid_frac"] >= 0.5).ravel() if "valid_frac" in z else np.ones(z["csr_abs"].shape[1], bool)
        hide = lambda a: np.ma.masked_where(~valid, upsample(a, R))  # grey = not evaluated
        csr_cols = [z["csr_abs"][pids.index(p)] for m, p, _ in COLUMNS if m == "csr_abs" and p in pids]
        vmax = max(np.quantile(c[vgrid], 0.99) for c in csr_cols)
        vmin = min(np.quantile(c[vgrid], 0.01) for c in csr_cols)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(mask, cmap="gray")
        for k, (m, p, title) in enumerate(COLUMNS):
            ax = axes[i, 2 + k]
            ax.set_facecolor("#d9d9d9")
            if m == "native0" and "native0" in z:
                arr = z["native0"]
            elif p in pids:
                arr = z[m][pids.index(p)]
            else:
                continue
            if m.startswith("native"):   # logits; own robust scale per map
                ax.imshow(hide(arr), cmap=SEQ, vmin=np.quantile(arr[vgrid], 0.01), vmax=np.quantile(arr[vgrid], 0.99))
            else:                        # CSR columns share one scale per case
                ax.imshow(hide(arr), cmap=SEQ, vmin=vmin, vmax=vmax)
            ax.contour(mask, levels=[0.5], colors="white", linewidths=0.5)
            if i == 0:
                ax.set_title(title, fontsize=6)
        if i == 0:
            axes[i, 0].set_title("image", fontsize=6)
            axes[i, 1].set_title("GT mask", fontsize=6)
        axes[i, 0].set_ylabel(f"{cases[cid].label}\n{100 * mask.mean():.1f}% area", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    path = out / f"{meta['dataset']}_panels.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print("wrote", path, "cases:", ids)


if __name__ == "__main__":
    main()
