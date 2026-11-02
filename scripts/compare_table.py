"""Final cross-model tables (supervisor feedback 2026-10-22 and 2026-10-28).

Localisation (headline): median per-image pixel AUROC and AUPRC lift, bootstrap 95% CI.
Secondary: pointing hit rate (mean), CSS difference (map units; not comparable across models).
Prompt specificity, per image, one gap per control (never averaged):
    G_cat     = AUROC(pathology) - AUROC(cat)              non-medical control
    G_context = AUROC(pathology) - AUROC(anatomy)          target-vs-context contrast (the
                context contains the target: not a direct measure of semantic failure)
    G_wrong   = AUROC(pathology) - AUROC(wrong_medical)    cross-medical discrimination
    G_cf      = AUROC(pathology) - AUROC(counterfactual)   presence vs healthy counterpart
Second specificity table on AUPRC (the SteerViT map is near the AUROC ceiling, so AUROC gaps
are compressed): R_x = log2(AUPRC(pathology) / AUPRC(x)) per image (= log2 of the AUPRC-lift
ratio, prevalence cancels). R_x = 1 means the pathology prompt doubles AUPRC.
Main evidence: median + bootstrap 95% CI. Images are not independent (no patient IDs; near
duplicates in BUSI), so Wilcoxon p-values (Holm per dataset) are only written to a separate
supplementary table.
Baseline column 'none': SteerViT / DINOv2 = no text input (texts=None);
CLIP / SAM3 = EMPTY-TEXT control (the string "" is tokenised and encoded: start + end token).
Maps (loc.csv names):
    csr_abs       steering magnitude      R_i = ||F_i(x,p) - F_i(x)||
    native_delta  directed steering effect S_i = w^T dF_i (SteerViT: linear head, bias cancels;
                  SAM3 / CLIP: change of the model's own output map, no linear head)
    native        prompted localisation map;  native0 = the same without prompt

  python scripts/compare_table.py --steervit runs_final2/gate1 \
      --clip runs_final2_compare/clip/gate1 --sam3 runs_final2_compare/sam3/gate1 --out paper_final2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["brisc", "busi", "kvasir", "isic"]
PROMPTS = {"none": "none", "anatomy": "anatomy", "pathology": "pathology",
           "irrelevant_cat": "cat", "wrong_medical": "wrong-medical", "counterfactual": "counterfactual"}
ORDER = ["none", "empty text", "anatomy", "pathology", "cat", "wrong-medical", "counterfactual"]
EMPTY_TEXT_RUNS = {"clip", "biomedclip", "sam3", "medsam3"}    # their 'none' is the empty-text control, not "no prompt"
GAINS = [("G_cat", "irrelevant_cat"), ("G_context", "anatomy"),
         ("G_wrong", "wrong_medical"), ("G_cf", "counterfactual")]
HEADLINE = ["auroc", "auprc_lift"]
SECONDARY = ["pointing", "css_diff"]

# (model, run, loc.csv map, label, prompt-free map shown in the 'none' column)
ROWS = [
    ("SteerViT", "steervit", "native", "prompted map", "native0"),
    ("SteerViT", "steervit", "native_delta", "DSE  w^T dF", None),
    ("SteerViT", "steervit", "csr_abs", "CSR  ||dF||", None),
    ("DINOv2", "steervit", None, "CLS attention", "attn0"),
    ("DINOv2", "steervit", None, "salience", "baseline_salience"),
    ("DINOv2", "steervit", None, "feature norm", "baseline_norm"),
    ("CLIP", "clip", "native", "prompted map", "native0"),
    ("CLIP", "clip", "native_delta", "output shift", None),
    ("BiomedCLIP", "biomedclip", "native", "prompted map", "native0"),
    ("BiomedCLIP", "biomedclip", "native_delta", "output shift", None),
    ("SAM3", "sam3", "native", "prompted map", "native0"),
    ("SAM3", "sam3", "native_delta", "output shift", None),
    ("SAM3", "sam3", "csr_abs", "CSR  ||dF||", None),
    ("MedSAM3", "medsam3", "native", "prompted map", "native0"),
    ("MedSAM3", "medsam3", "native_delta", "output shift", None),
    ("MedSAM3", "medsam3", "csr_abs", "CSR  ||dF||", None),
]
GAIN_MAPS = [r for r in ROWS if r[2] is not None]

PAIRS = [  # paired per-image AUROC comparisons between models
    ("SteerViT map vs SAM3 map (pathology)", ("steervit", "native", "pathology"), ("sam3", "native", "pathology")),
    ("SteerViT map (pathology) vs SAM3 map (empty text)", ("steervit", "native", "pathology"), ("sam3", "native0", "none")),
    ("SteerViT map (pathology) vs DINOv2 salience", ("steervit", "native", "pathology"), ("steervit", "baseline_salience", "none")),
    ("SteerViT map (pathology) vs CLIP map (pathology)", ("steervit", "native", "pathology"), ("clip", "native", "pathology")),
    ("SteerViT DSE vs SteerViT CSR (pathology)", ("steervit", "native_delta", "pathology"), ("steervit", "csr_abs", "pathology")),
    ("MedSAM3 map vs SAM3 map (pathology)", ("medsam3", "native", "pathology"), ("sam3", "native", "pathology")),
    ("BiomedCLIP map vs CLIP map (pathology)", ("biomedclip", "native", "pathology"), ("clip", "native", "pathology")),
    ("SteerViT map vs MedSAM3 map (pathology)", ("steervit", "native", "pathology"), ("medsam3", "native", "pathology")),
]


def boot(x, n=2000, seed=0, stat=np.median):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = stat(rng.choice(x, (n, len(x))), axis=1)
    return float(stat(x)), *np.percentile(m, [2.5, 97.5]).tolist()


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def units(d):
    """Analysis unit: patient if recorded (several images per patient), else image."""
    if "patient" in d and d["patient"].fillna("").astype(str).str.len().gt(0).any():
        return d["patient"].fillna(d["case_id"]).astype(str)
    return d["case_id"].astype(str)


def wilcoxon_p(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    return float(wilcoxon(x).pvalue)


def load(runs):
    locs = {}
    for ds in DATASETS:
        for key, root in runs.items():
            f = Path(root) / ds / "loc.csv"
            if f.exists():
                d = pd.read_csv(f)
                locs.setdefault(ds, {})[key] = d[d["prompt"].isin(PROMPTS)]
            else:
                print(f"skip {key}/{ds}: {f} not found")
    return locs


def summary(locs):
    rows = []
    for ds, by in locs.items():
        for model, run, m, label, m0 in ROWS:
            if run not in by:
                continue
            d = by[run]
            parts = []
            if m0 is not None:
                parts.append(d[(d["map"] == m0) & (d["prompt"] == "none")])
            if m is not None:
                parts.append(d[(d["map"] == m) & (d["prompt"] != "none")])
            for sub in parts:
                for pid, g in sub.groupby("prompt"):
                    shown = "empty text" if (pid == "none" and run in EMPTY_TEXT_RUNS) else PROMPTS[pid]
                    for met in HEADLINE + SECONDARY:
                        if met not in g:
                            continue
                        v, lo, hi = boot(g[met], stat=np.mean if met == "pointing" else np.median)
                        rows.append(dict(dataset=ds, model=model, map=label, prompt=shown, metric=met,
                                         n=int(g[met].notna().sum()), value=v, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def gains(locs, metric="auroc"):
    """metric 'auroc': AUROC(path) - AUROC(x); 'auprc': log2(AUPRC(path) / AUPRC(x))."""
    rows = []
    for ds, by in locs.items():
        for model, run, m, label, _ in GAIN_MAPS:
            if run not in by:
                continue
            d = by[run][by[run]["map"] == m]
            if d.empty:
                continue
            d = d.assign(unit=units(d))
            if metric not in d:
                continue
            w = d.pivot_table(index="unit", columns="prompt", values=metric, aggfunc="mean")
            if "pathology" not in w:
                continue
            if metric == "auprc":
                lw = np.log2(w.clip(lower=1e-6))
                defs = {gname.replace("G_", "R_"): lw["pathology"] - lw[pid] for gname, pid in GAINS if pid in w}
            else:
                defs = {gname: w["pathology"] - w[pid] for gname, pid in GAINS if pid in w}
            for gname, x in defs.items():
                v, lo, hi = boot(x.values)
                rows.append(dict(dataset=ds, model=model, map=label, gain=gname, n=int(x.notna().sum()),
                                 median=v, ci_lo=lo, ci_hi=hi, p=wilcoxon_p(x.values)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for ds, g in t.groupby("dataset"):
            ok = g["p"].notna()
            t.loc[g.index[ok], "p_holm"] = holm(g.loc[ok, "p"].values)
    return t


def paired(locs):
    from scipy.stats import wilcoxon
    rows = []
    for ds, by in locs.items():
        for label, a, b in PAIRS:
            if a[0] not in by or b[0] not in by:
                continue
            def get(run, m, pid):
                d = by[run]
                d = d[(d["map"] == m) & (d["prompt"] == pid)]
                return d.assign(unit=units(d)).groupby("unit")["auroc"].mean()
            x, y = get(*a), get(*b)
            j = x.index.intersection(y.index)
            x, y = x.loc[j], y.loc[j]
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            if len(x) < 5:
                continue
            p = float(wilcoxon(x, y).pvalue) if (x - y).abs().sum() > 0 else 1.0
            rows.append(dict(dataset=ds, comparison=label, n=len(x), median_a=float(x.median()),
                             median_b=float(y.median()), median_diff=float((x - y).median()), p=p))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for ds, g in t.groupby("dataset"):
            t.loc[g.index, "p_holm"] = holm(g["p"].values)
    return t


def fmt_p(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steervit", default="runs_final2/gate1")
    ap.add_argument("--clip", default="runs_final2_compare/clip/gate1")
    ap.add_argument("--sam3", default="runs_final2_compare/sam3/gate1")
    ap.add_argument("--medsam3", default="runs_final2_compare/medsam3/gate1")
    ap.add_argument("--biomedclip", default="runs_final2_compare/biomedclip/gate1")
    ap.add_argument("--out", default="paper_final2")
    a = ap.parse_args()
    out = Path(a.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    locs = load({"steervit": a.steervit, "clip": a.clip, "sam3": a.sam3, "medsam3": a.medsam3,
                 "biomedclip": a.biomedclip})
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)

    s = summary(locs)
    s.to_csv(out / "compare_long.csv", index=False)
    for met in HEADLINE + SECONDARY:
        w = s[s.metric == met].pivot_table(index=["dataset", "model", "map"], columns="prompt",
                                           values="value", aggfunc="first")
        w = w[[c for c in ORDER if c in w.columns]]
        w.round(3).to_csv(out / f"compare_{met}.csv")
        kind = "mean hit rate" if met == "pointing" else "median per image"
        note = "  (map units, not comparable across models)" if met == "css_diff" else ""
        (out / f"compare_{met}.tex").write_text(
            f"% {met}: {kind}{note}; '--' = not defined\n" + w.round(3).to_latex(na_rep="--", float_format="%.3f"))
        print(f"\n=== {met} ({kind}){note} ===\n{w.round(3).to_string()}")

    for metric, stem, title, unit in [
            ("auroc", "specificity_gains", "G_x = AUROC(pathology) - AUROC(x)", "AUROC gap"),
            ("auprc", "specificity_gains_auprc", "R_x = log2(AUPRC(pathology) / AUPRC(x))  (1 = 2x AUPRC)",
             "log2 AUPRC ratio")]:
        g = gains(locs, metric)
        if not len(g):
            continue
        g.to_csv(out / f"{stem}.csv", index=False)
        g["cell"] = g.apply(lambda r: f"{r['median']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1)
        pre = "R_" if metric == "auprc" else "G_"
        order = [pre + n[2:] for n, _ in GAINS]
        wide = g.pivot_table(index=["dataset", "model", "map"], columns="gain", values="cell", aggfunc="first")
        wide = wide[[c for c in order if c in wide.columns]]
        wide.to_csv(out / f"{stem}_wide.csv")
        (out / f"{stem}.tex").write_text(
            f"% median per-image {unit}, bootstrap 95% CI. Increasing difficulty: cat (non-medical),\n"
            "% context (target-vs-context contrast), wrong (cross-medical), cf (counterfactual)\n" +
            wide.to_latex(na_rep="--"))
        tex = g.assign(p=g["p"].map(fmt_p), p_holm=g["p_holm"].map(fmt_p))
        (out / f"{stem}_pvalues.tex").write_text(
            "% supplementary: Wilcoxon vs 0, Holm per dataset. Nominal only: images are not\n"
            "% independent (no patient IDs, near duplicates in BUSI), so these are optimistic.\n" +
            tex[["dataset", "model", "map", "gain", "n", "p", "p_holm"]].to_latex(index=False))
        print(f"\n=== prompt specificity: {title}, median [95% CI] ===")
        print(wide.to_string())

    pr = paired(locs)
    if len(pr):
        pr.to_csv(out / "compare_paired.csv", index=False)
        print("\n=== paired model comparisons (AUROC; p nominal, Holm per dataset) ===")
        print(pr.round(4).to_string(index=False))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
