"""Fig. 1 of the ISBI paper: one representative case per dataset, editable vector output.

Columns (default), grouped under three headers:
  Image + GT
  SteerViT map:            target | cat | wrong-medical          (final localisation map)
  SteerViT representation: DSE (target) | SRM (target)            (w^T dF  and  ||dF||)
  SAM3 map:                target | cat
Rows: BRISC, BUSI, Kvasir-SEG, ISIC (one case each). The exact target and wrong-medical
prompt of each row is printed under the row label.

Every heat-map panel has its own colour scale (1st-99th percentile inside the evaluation
region): panels show WHERE a map responds, not how strongly, so brightness must not be compared
between panels. Ground truth as contour; grey = padding (outside the evaluation region).
The small number in each panel is that image's AUPRC lift L (1 = chance), read from loc.csv,
so the figure carries the same quantity as Table 2 (switch with --score auroc / none).

Outputs in --out:  fig1.pdf / fig1.svg (text stays editable: Type-42 fonts, SVG text as text),
fig1.png (preview), and with --panels every tile as a separate PNG in panels/ for re-layout.

  python scripts/paper_fig1.py --out paper_final2/fig1                    # default cases
  python scripts/paper_fig1.py --auto --out paper_final2/fig1_auto        # median-like cases
  python scripts/paper_fig1.py --cases brisc=brisc_brisc2025_test_00473_me_co_t1 --panels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import load_image, read_manifest  # noqa: E402

try:
    from medsteer.metrics import upsample          # the pipeline's own bilinear upsampling
except Exception:                                   # torch-free fallback, same arithmetic
    def upsample(patch_map, size):                  # bilinear, align_corners=False
        g = int(round(np.sqrt(patch_map.size)))
        m = patch_map.reshape(g, g).astype(np.float64)
        c = np.clip((np.arange(size) + 0.5) * g / size - 0.5, 0, g - 1)
        i0 = np.floor(c).astype(int); i1 = np.minimum(i0 + 1, g - 1); w = c - i0
        rows = m[i0] * (1 - w)[:, None] + m[i1] * w[:, None]
        return (rows[:, i0] * (1 - w)[None, :] + rows[:, i1] * w[None, :]).astype(np.float32)

DATASETS = [("brisc", "BRISC"), ("busi", "BUSI"), ("kvasir", "KVASIR"), ("isic", "ISIC")]
DEFAULT_CASES = {"brisc": "brisc_brisc2025_test_00086_gl_co_t1", "busi": "busi_benign_111",
                 "kvasir": "kvasir_cju1cvkfwqrec0993wbp1jlzm", "isic": "isic_isic_0018172"}
# column key: (group header, sub header, run, npz map, prompt id, loc.csv map, colour map)
COLS = {
    "img":       ("", "Image + GT", None, None, None, None, None),
    "sv_target": ("SteerViT map", "target", "steervit", "native", "pathology", "native", "magma"),
    "sv_cat":    ("SteerViT map", "cat", "steervit", "native", "irrelevant_cat", "native", "magma"),
    "sv_wrong":  ("SteerViT map", "wrong-medical", "steervit", "native", "wrong_medical", "native", "magma"),
    "sv_cf":     ("SteerViT map", "counterfactual", "steervit", "native", "counterfactual", "native", "magma"),
    "sv_dse":    ("SteerViT representation", r"DSE $w^{\top}\Delta F$", "steervit", "dse", "pathology",
                  "native_delta", "viridis"),
    "sv_srm":    ("SteerViT representation", r"SRM $\|\Delta F\|$", "steervit", "csr_abs", "pathology",
                  "csr_abs", "magma"),
    "sam_target": ("SAM3 map", "target", "sam3", "native", "pathology", "native", "magma"),
    "sam_cat":   ("SAM3 map", "cat", "sam3", "native", "irrelevant_cat", "native", "magma"),
    "sam_wrong": ("SAM3 map", "wrong-medical", "sam3", "native", "wrong_medical", "native", "magma"),
    "sam_empty": ("SAM3 map", "empty text", "sam3", "native0", None, "native0", "magma"),
    "med_target": ("MedSAM3 map", "target", "medsam3", "native", "pathology", "native", "magma"),
    "med_cat":   ("MedSAM3 map", "cat", "medsam3", "native", "irrelevant_cat", "native", "magma"),
    "med_wrong": ("MedSAM3 map", "wrong-medical", "medsam3", "native", "wrong_medical", "native", "magma"),
}
DEFAULT_COLS = ["img", "sv_target", "sv_cat", "sv_wrong", "sv_dse", "sv_srm", "sam_target", "sam_cat",
                "sam_wrong", "med_target", "med_cat", "med_wrong"]
GT_LIGHT, GT_DARK, GT_IMG = "white", "black", "#2BD99F"


def get_map(z, key, pid):
    pids = list(z["prompt_ids"])
    if key == "native0":
        return z["native0"]
    if key == "dse":
        return z["native"][pids.index(pid)] - z["native0"]
    return z[key][pids.index(pid)]


def pick_auto(loc, cases):
    """Representative case: lesion size and target-prompt AUROC both close to the median."""
    d = loc[(loc["map"] == "native") & (loc["prompt"] == "pathology")].dropna(subset=["auroc"])
    d = d[d["case_id"].isin(cases)]
    r = (d["lesion_frac"].rank(pct=True) - 0.5).abs() + (d["auroc"].rank(pct=True) - 0.5).abs()
    return d.loc[r.idxmin(), "case_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--config", default="configs/prompts_final2.yaml")
    ap.add_argument("--steervit", default="runs_final2/gate1")
    ap.add_argument("--sam3", default="runs_final2_compare/sam3/gate1")
    ap.add_argument("--medsam3", default="runs_final2_compare/medsam3/gate1")
    ap.add_argument("--cols", nargs="*", default=DEFAULT_COLS, choices=list(COLS))
    ap.add_argument("--cases", nargs="*", default=[], help="dataset=case_id overrides")
    ap.add_argument("--auto", action="store_true", help="pick median-like cases instead of defaults")
    ap.add_argument("--score", default="lift", choices=["lift", "auroc", "none"])
    ap.add_argument("--width", type=float, default=7.16, help="figure width in inches (ISBI: 7.16)")
    ap.add_argument("--panels", action="store_true", help="also save every tile as a PNG")
    ap.add_argument("--out", default="paper_final2/fig1")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())["datasets"]
    texts = {ds: {p["id"]: p["text"] for p in cfg[ds]["prompts"]} for ds, _ in DATASETS}
    runs = {"steervit": Path(a.steervit), "sam3": Path(a.sam3), "medsam3": Path(a.medsam3)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.panels:
        (out / "panels").mkdir(exist_ok=True)
    over = dict(s.split("=", 1) for s in a.cases)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                         "TeX Gyre Termes", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    nr, nc = len(DATASETS), len(a.cols)
    label_w = 0.42                                       # inches for the row-label column
    margin = 0.12                                        # inches of white space on the right and bottom
    tile = (a.width - label_w - margin) / nc
    compact = tile < 0.7                                 # narrow panels: shorter titles, smaller text
    SHORT = {"wrong-medical": "wrong-med.", r"DSE $w^{\top}\Delta F$": "DSE", r"SRM $\|\Delta F\|$": "SRM",
             "counterfactual": "counterf."}
    head = 0.42                                          # inches above the tiles (two header lines)
    fig = plt.figure(figsize=(a.width, nr * (tile + 0.02) + head + margin))
    top = 1 - head / fig.get_figheight()
    gs = fig.add_gridspec(nr, nc, left=label_w / a.width, right=1 - margin / a.width,
                          bottom=margin / fig.get_figheight(), top=top,
                          wspace=0.04, hspace=0.06)

    chosen = {}
    for r, (ds, dname) in enumerate(DATASETS):
        cases = {c.case_id: c for c in read_manifest(a.manifest, datasets=[ds])}
        locs = {k: (pd.read_csv(p / ds / "loc.csv") if (p / ds / "loc.csv").exists() else None)
                for k, p in runs.items()}
        cid = over.get(ds) or (pick_auto(locs["steervit"], cases) if a.auto else DEFAULT_CASES[ds])
        chosen[ds] = cid
        zs = {k: np.load(p / ds / "cases" / f"{cid}.npz") for k, p in runs.items()
              if (p / ds / "cases" / f"{cid}.npz").exists()}
        z0 = zs["steervit"]
        mask, valid = z0["mask"], z0["valid"]
        R = mask.shape[0]
        img = load_image(cases[cid].image_path, R)
        for c, key in enumerate(a.cols):
            group, sub, run, mkey, pid, lmap, cmap = COLS[key]
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if key == "img":
                ax.set_facecolor("#d0d0d0")
                ax.imshow(np.where(valid[..., None], img, np.nan), interpolation="lanczos")
                ax.contour(mask, [0.5], colors=GT_IMG, linewidths=0.8)
                tile_rgb = None
            elif run not in zs:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                continue
            else:
                m = upsample(get_map(zs[run], mkey, pid).astype(np.float32), R)
                lo, hi = np.percentile(m[valid], [1, 99])
                shown = np.where(valid, m, np.nan)
                ax.set_facecolor("#d0d0d0")
                ax.imshow(shown, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                ax.contour(mask, [0.5], colors=GT_DARK if cmap == "viridis" else GT_LIGHT, linewidths=0.6)
                loc = locs.get(run)
                if a.score != "none" and loc is not None:
                    q = loc[(loc["case_id"] == cid) & (loc["map"] == lmap)
                            & (loc["prompt"] == (pid if pid else "none"))]
                    col = "auprc_lift" if a.score == "lift" else "auroc"
                    if len(q) and pd.notna(q[col].iloc[0]):
                        v = q[col].iloc[0]
                        txt = (f"L={v:.1f}" if v < 10 else f"L={v:.0f}") if a.score == "lift" else f"{v:.2f}"
                        ax.text(0.04, 0.04, txt, transform=ax.transAxes, fontsize=4.8 if compact else 5.5, color="white",
                                ha="left", va="bottom",
                                bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))
            if a.panels:
                f2, a2 = plt.subplots(figsize=(2, 2), dpi=168)
                for im in ax.get_images():
                    a2.imshow(im.get_array(), cmap=im.get_cmap(), norm=im.norm, interpolation="nearest")
                a2.set_facecolor("#d0d0d0")
                a2.contour(mask, [0.5], colors=GT_IMG if key == "img" else
                           (GT_DARK if cmap == "viridis" else GT_LIGHT), linewidths=1)
                a2.axis("off"); f2.subplots_adjust(0, 0, 1, 1)
                f2.patch.set_facecolor("#d0d0d0")   # padding grey, as in the full figure
                f2.savefig(out / "panels" / f"{ds}_{key}.png"); plt.close(f2)
            if r == 0:
                ax.set_title(SHORT.get(sub, sub) if compact else sub, fontsize=6.5 if compact else 7, pad=2)
            if c == 0:
                pos = gs[r, 0].get_position(fig)
                fig.text(pos.x0 - 0.008, (pos.y0 + pos.y1) / 2, dname, fontsize=7, rotation=90,
                         ha="right", va="center")
    # group headers with a thin rule over their columns
    groups = []
    for c, key in enumerate(a.cols):
        g = COLS[key][0]
        if groups and groups[-1][0] == g:
            groups[-1][2] = c
        elif g:
            groups.append([g, c, c])
    ytop = top + 0.25 / fig.get_figheight()
    for g, c0, c1 in groups:
        x0 = gs[0, c0].get_position(fig).x0 + 0.004
        x1 = gs[0, c1].get_position(fig).x1 - 0.004
        fig.add_artist(plt.Line2D([x0, x1], [ytop - 0.012, ytop - 0.012], transform=fig.transFigure,
                                  lw=0.6, color="black"))
        fig.text((x0 + x1) / 2, ytop, g, ha="center", va="bottom", fontsize=7, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig1.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    (out / "cases.txt").write_text("\n".join(f"{ds}={cid}" for ds, cid in chosen.items()) + "\n")
    print("cases:", chosen)
    print(f"wrote {out}/fig1.pdf, fig1.svg, fig1.png" + (", panels/*.png" if a.panels else ""))


if __name__ == "__main__":
    main()
