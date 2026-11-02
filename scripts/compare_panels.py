"""Case figures for the final experiment, saved as PDF + SVG (+ PNG preview).

Columns: no prompt / empty text | anatomy | pathology | cat | wrong-medical | counterfactual.
Ground truth as contour.
Rows (choose with --rows):
  sv_map    SteerViT prompted map       per-panel scale  (location only)
  sv_dse    SteerViT DSE  w^T dF        shared scale (diverging around 0 only if both signs occur)
  sv_csr    SteerViT CSR  ||dF||        shared scale (magnitude)
  dino      DINOv2 CLS attention        per-panel
  clip_map  CLIP prompted map           per-panel
  sam_map   SAM3 prompted map           per-panel
  sam_dse   SAM3 output shift           shared scale (diverging around 0 only if both signs occur)
  sam_csr   SAM3 CSR ||dF||             shared
Shared scales are per case and row (1st-99th percentile inside the image region): they permit
within-case comparison across prompts, but not absolute magnitude comparison between cases.
Per-panel scales show WHERE a map responds only; do not compare their brightness.

  python scripts/compare_panels.py --dataset brisc --manifest data/manifest_final.csv \
      --out paper_final2/figs [--cases <case_id> ...] [--rows sv_map sv_dse sv_csr sam_map]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import load_image, read_manifest  # noqa: E402
from medsteer.metrics import upsample  # noqa: E402

COLS = [("none", "no prompt /\nempty text*"), ("anatomy", "anatomy"), ("pathology", "pathology"),
        ("irrelevant_cat", '"cat"'), ("wrong_medical", "wrong-medical"), ("counterfactual", "counterfactual")]
# key: (label, run, prompted map, 'none' map, scale)   scale: panel | shared | signed
ROWDEF = {
    "sv_map":   ("SteerViT\nmap", "steervit", "native", "native0", "panel"),
    "sv_dse":   ("SteerViT\nDSE $w^\\top\\Delta F$", "steervit", "native_delta", None, "signed"),
    "sv_csr":   ("SteerViT\nCSR $\\|\\Delta F\\|$", "steervit", "csr_abs", None, "shared"),
    "dino":     ("DINOv2\nCLS attention", "steervit", None, "attn0", "panel"),
    "clip_map": ("CLIP\nmap", "clip", "native", "native0", "panel"),
    "sam_map":  ("SAM3\nmap", "sam3", "native", "native0", "panel"),
    "sam_dse":  ("SAM3\noutput shift", "sam3", "native_delta", None, "signed"),
    "sam_csr":  ("SAM3\nCSR $\\|\\Delta F\\|$", "sam3", "csr_abs", None, "shared"),
}
DEFAULT_ROWS = list(ROWDEF)
SCALE_NOTE = {"panel": "(per-panel scale)", "shared": "(shared scale)", "signed": "(shared scale)"}
CAPTION = ("CSR and DSE rows use one colour scale per case, shared across prompts (1st-99th "
           "percentile in the image region; diverging and centred at 0 only where values take both "
           "signs). The shared scale permits "
           "within-case comparison across prompts, but not absolute magnitude comparison between "
           "cases. Map rows use per-panel scales and show only where each map responds; their "
           "brightness must not be compared between prompts. *First column: SteerViT and DINOv2 rows "
           "receive no text input; CLIP and SAM3 rows receive the empty string \"\" (empty-text "
           "control), since both models require a text input.")


def get_maps(z, prompted, none_key):
    pids = list(z["prompt_ids"])
    out = {}
    if none_key and none_key in z:
        out["none"] = z[none_key]
    if prompted:
        if prompted == "native_delta":
            if "native" in z and "native0" in z:
                for pid, _ in COLS[1:]:
                    if pid in pids:
                        out[pid] = z["native"][pids.index(pid)] - z["native0"]
        elif prompted in z:
            for pid, _ in COLS[1:]:
                if pid in pids:
                    out[pid] = z[prompted][pids.index(pid)]
    return out


def plot_case(cid, ds, runs, cases, rows, out_dir, fmts):
    zs = {k: np.load(Path(r) / ds / "cases" / f"{cid}.npz") for k, r in runs.items()
          if (Path(r) / ds / "cases" / f"{cid}.npz").exists()}
    if "steervit" not in zs:
        print(f"skip {cid}: no SteerViT maps"); return
    mask, valid = zs["steervit"]["mask"], zs["steervit"]["valid"]
    R = mask.shape[0]
    img = load_image(cases[cid].image_path, R)
    rr = []
    for key in rows:
        lab, run, pm, nm, scale = ROWDEF[key]
        if run in zs:
            m = get_maps(zs[run], pm, nm)
            if m:
                rr.append((lab, m, scale))

    fig, ax = plt.subplots(len(rr), len(COLS) + 1, figsize=(1.9 * (len(COLS) + 1), 1.9 * len(rr)),
                           squeeze=False)
    for i, (lab, maps, scale) in enumerate(rr):
        ax[i, 0].imshow(img)
        ax[i, 0].contour(mask, [0.5], colors="#2BD99F", linewidths=1)
        ax[i, 0].set_ylabel(f"{lab}\n{SCALE_NOTE[scale]}", fontsize=7)
        up = {k: upsample(v.astype(np.float32), R) for k, v in maps.items()}
        vals = np.concatenate([u[valid] for u in up.values()])
        norm = None
        lo_r, hi_r = np.percentile(vals, [1, 99])      # shared over all prompts of this row
        cmap = "magma"
        if scale == "signed":
            if lo_r < 0 < hi_r:   # both signs present: diverging map centred at 0
                cmap, norm = "RdBu_r", matplotlib.colors.TwoSlopeNorm(vmin=lo_r, vcenter=0.0, vmax=hi_r)
            else:                 # one sign only (e.g. SteerViT w^T dF < 0 everywhere): sequential
                cmap = "viridis"
        im = None
        for j, (pid, title) in enumerate(COLS, 1):
            a = ax[i, j]
            if i == 0:
                a.set_title(title, fontsize=9)
            if pid not in up:
                a.axis("off"); continue
            if scale == "panel":
                lo, hi = np.percentile(up[pid][valid], [1, 99])
                im = a.imshow(np.where(valid, up[pid], np.nan), cmap=cmap, vmin=lo, vmax=hi)
            elif norm is not None:
                im = a.imshow(np.where(valid, up[pid], np.nan), cmap=cmap, norm=norm)
            else:
                im = a.imshow(np.where(valid, up[pid], np.nan), cmap=cmap, vmin=lo_r, vmax=hi_r)
            a.set_facecolor("#d9d9d9")
            a.contour(mask, [0.5], colors="black" if cmap in ("RdBu_r", "viridis") else "white", linewidths=0.7)
        if im is not None and scale != "panel":
            fig.colorbar(im, ax=ax[i, 1:].tolist(), fraction=0.015, pad=0.01)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_title(f"{ds} ({100 * mask.mean():.1f}% target)", fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in fmts:
        fig.savefig(out_dir / f"case_{cid}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / f"case_{cid}.{{{','.join(fmts)}}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--rows", nargs="*", default=DEFAULT_ROWS, choices=list(ROWDEF))
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--steervit", default="runs_final2/gate1")
    ap.add_argument("--clip", default="runs_final2_compare/clip/gate1")
    ap.add_argument("--sam3", default="runs_final2_compare/sam3/gate1")
    ap.add_argument("--formats", nargs="*", default=["pdf", "svg", "png"])
    ap.add_argument("--out", default="paper_final2/figs")
    a = ap.parse_args()
    runs = {"steervit": a.steervit, "clip": a.clip, "sam3": a.sam3}
    cases = {c.case_id: c for c in read_manifest(a.manifest, datasets=[a.dataset])}
    ids = a.cases
    if not ids:   # lesion cases present in all runs, picked at lesion-size quantiles
        have = None
        for r in runs.values():
            d = Path(r) / a.dataset / "cases"
            if d.exists():
                s = {p.stem for p in d.glob("*.npz")}
                have = s if have is None else have & s
        cand = sorted(i for i in (have or []) if i in cases and cases[i].mask_paths and not cases[i].is_healthy)
        sizes = sorted((float(np.load(Path(a.steervit) / a.dataset / "cases" / f"{i}.npz")["mask"].mean()), i)
                       for i in cand)
        sizes = [x for x in sizes if x[0] > 0]
        ids = [sizes[int(q * (len(sizes) - 1))][1] for q in (0.15, 0.5, 0.85)] if sizes else []
    out = Path(a.out)
    for cid in ids:
        plot_case(cid, a.dataset, runs, cases, a.rows, out, a.formats)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CAPTION_NOTE.txt").write_text(CAPTION + "\n")


if __name__ == "__main__":
    main()
