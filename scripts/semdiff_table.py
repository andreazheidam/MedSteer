"""Tables for training-free semantic differential steering, from semdiff_eval.py outputs.

  python scripts/semdiff_table.py --dataset kvasir --root runs_semdiff --out paper_semdiff/kvasir

Reads <root>/<model>/gate1/<dataset>/semdiff_loc.csv.gz for every model found and writes
  localization.csv    median AUROC / AUPRC lift / q (css_diff) per model x family x variant x map
  specificity.csv     G_c = AUROC(s+) - AUROC(s_c), R_c = log2 AUPRC(s+)/AUPRC(s_c)  (single and ens)
  paired.csv          paired per-image comparisons: differential vs target, ensemble vs single
  paraphrase.csv      per-prompt localisation and paraphrase-map Spearman per category
  sanity_paper.csv    paraphrase-1 maps vs the paper's per-image results (results/runs/...)
  summary.md          the headline numbers in one file (paste this back for review)
All CIs: bootstrap over images (2000 resamples, seed 0), as in compare_table.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["steervit", "clip", "biomedclip", "sam3", "medsam3"]
CONTROLS = ["anatomy", "counterfactual", "wrong_medical", "irrelevant"]
P1 = {"pathology": "pathology", "anatomy": "anatomy", "counterfactual": "counterfactual",
      "wrong_medical": "wrong_medical", "irrelevant": "irrelevant_cat"}


def boot(x, n=2000, seed=0, stat=np.median):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = stat(rng.choice(x, (n, len(x))), axis=1)
    return float(stat(x)), *np.percentile(m, [2.5, 97.5]).tolist()


def wilcoxon_p(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    return float(wilcoxon(x).pvalue)


def fmt(v, lo=None, hi=None, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = f"{v:.{d}f}"
    if lo is not None and not np.isnan(lo):
        s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return s


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def localization(d):
    rows = []
    for (m, fam, var, mp), g in d[d.family != "prompt"].groupby(["model", "family", "variant", "map"], sort=False):
        r = dict(model=m, family=fam, variant=var, map=mp, n=len(g))
        for met in ("auroc", "auprc_lift", "css_diff"):
            r[met], r[met + "_lo"], r[met + "_hi"] = boot(g[met])
        r["pointing"] = float(g["pointing"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def specificity(d):
    rows = []
    n = d[d.family == "norm"]
    for (m, var), g in n.groupby(["model", "variant"], sort=False):
        w_auc = g.pivot_table(index="case_id", columns="map", values="auroc")
        w_pr = np.log2(g.pivot_table(index="case_id", columns="map", values="auprc").clip(lower=1e-6))
        for c in CONTROLS:
            if f"ctrl_{c}" not in w_auc:
                continue
            for name, x in ((f"G_{c}", w_auc["target"] - w_auc[f"ctrl_{c}"]),
                            (f"R_{c}", w_pr["target"] - w_pr[f"ctrl_{c}"])):
                v, lo, hi = boot(x.values)
                rows.append(dict(model=m, variant=var, gain=name, n=int(x.notna().sum()),
                                 median=v, ci_lo=lo, ci_hi=hi, p=wilcoxon_p(x.values)))
    return pd.DataFrame(rows)


def paired(d):
    rows = []
    key = ["model", "family", "variant", "map"]
    w = d[d.family.isin(["norm", "raw", "feat"])].pivot_table(index="case_id", columns=key, values="auroc")

    def add(label, a, b, model):
        if a not in w or b not in w:
            return
        x = (w[a] - w[b]).values
        v, lo, hi = boot(x)
        rows.append(dict(model=model, comparison=label, n=int(np.isfinite(x).sum()),
                         median_diff_auroc=v, ci_lo=lo, ci_hi=hi, p=wilcoxon_p(x)))

    for m in d.model.unique():
        for var in ("single", "ens"):
            for mp in ["D_mean", "D_env"] + [f"D_{c}" for c in CONTROLS]:
                add(f"{var}: {mp} - target", (m, "norm", var, mp), (m, "norm", var, "target"), m)
            add(f"{var}: D_env - D_mean", (m, "norm", var, "D_env"), (m, "norm", var, "D_mean"), m)
        for mp in ["target", "D_mean", "D_env"]:
            add(f"ens - single: {mp}", (m, "norm", "ens", mp), (m, "norm", "single", mp), m)
        if m == "steervit":
            for var in ("single", "ens"):
                add(f"{var}: feature ||D^F_mean|| - head D^H_mean", (m, "feat", var, "D_mean"),
                    (m, "raw", var, "D_mean"), m)
                add(f"{var}: normalised D_mean - head D^H_mean (raw)", (m, "norm", var, "D_mean"),
                    (m, "raw", var, "D_mean"), m)
    return pd.DataFrame(rows)


def paraphrase(d, para):
    rows = []
    p = d[d.family == "prompt"]
    for (m, pid), g in p.groupby(["model", "map"], sort=False):
        r = dict(model=m, prompt=pid, category=g["category"].iloc[0], n=len(g))
        r["auroc"], r["auroc_lo"], r["auroc_hi"] = boot(g["auroc"])
        r["auprc_lift"], r["lift_lo"], r["lift_hi"] = boot(g["auprc_lift"])
        rows.append(r)
    rho = para.groupby(["model", "category"])["spearman_mean"].median().rename("median_paraphrase_spearman")
    return pd.DataFrame(rows), rho.reset_index()


def sanity(d, paper_root, ds):
    rows = []
    for m in d.model.unique():
        f = Path(paper_root) / m / ds / "loc.csv.gz"
        if not f.exists():
            rows.append(dict(model=m, status=f"no paper file {f}"))
            continue
        ref = pd.read_csv(f)
        ref = ref[(ref["map"] == "native") & ref["prompt"].isin(P1.values())][["case_id", "prompt", "auroc", "auprc_lift"]]
        new = d[(d.model == m) & (d.family == "prompt") & d["map"].isin(P1.values())][["case_id", "map", "auroc", "auprc_lift"]]
        j = new.merge(ref, left_on=["case_id", "map"], right_on=["case_id", "prompt"], suffixes=("", "_paper"))
        if j.empty:
            rows.append(dict(model=m, status="no overlapping cases"))
            continue
        da = (j["auroc"] - j["auroc_paper"]).abs()
        dl = (j["auprc_lift"] - j["auprc_lift_paper"]).abs()
        rows.append(dict(model=m, status="ok", n_pairs=len(j), n_cases=j.case_id.nunique(),
                         max_abs_diff_auroc=float(da.max()), median_abs_diff_auroc=float(da.median()),
                         max_abs_diff_lift=float(dl.max()),
                         target_auroc_new=float(j[j["map"] == "pathology"]["auroc"].median()),
                         target_auroc_paper=float(j[j["map"] == "pathology"]["auroc_paper"].median())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="runs_semdiff", help="<root>/<model>/gate1/<dataset>/semdiff_loc.csv.gz")
    ap.add_argument("--paper-runs", default="results/runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="run names under <root> (e.g. steervit kvasir_target kvasir_spec_hard)")
    a = ap.parse_args()
    out = Path(a.out or f"paper_semdiff/{a.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    parts, paras = [], []
    for m in a.models:
        f = Path(a.root) / m / "gate1" / a.dataset / "semdiff_loc.csv.gz"
        if f.exists():
            parts.append(pd.read_csv(f))
            pf = f.parent / "semdiff_para.csv"
            if pf.exists() and pf.stat().st_size > 1:
                paras.append(pd.read_csv(pf))
        else:
            print(f"skip {m}: {f} not found")
    if not parts:
        raise SystemExit("no semdiff_loc.csv.gz found")
    d = pd.concat(parts, ignore_index=True)
    para = pd.concat(paras, ignore_index=True) if paras else pd.DataFrame(columns=["model", "category", "spearman_mean"])

    loc, spec, pr = localization(d), specificity(d), paired(d)
    per_prompt, rho = paraphrase(d, para)
    san = sanity(d, a.paper_runs, a.dataset)
    for name, t in [("localization", loc), ("specificity", spec), ("paired", pr),
                    ("paraphrase", per_prompt), ("paraphrase_spearman", rho), ("sanity_paper", san)]:
        t.to_csv(out / f"{name}.csv", index=False)

    md = [f"# Semantic differential steering, training-free: {a.dataset}\n"]
    md.append("## Sanity: paraphrase-1 maps vs paper per-image results\n")
    md.append(md_table(san.round(6)))
    md.append("\n## Localization (median per image [95% bootstrap CI])\n")
    t = loc[loc.family.isin(["norm", "feat", "raw"])].copy()
    t["AUROC"] = [fmt(r.auroc, r.auroc_lo, r.auroc_hi) for r in t.itertuples()]
    t["AUPRC lift"] = [fmt(r.auprc_lift, r.auprc_lift_lo, r.auprc_lift_hi, 2) for r in t.itertuples()]
    t["q"] = [fmt(r.css_diff, d=3) for r in t.itertuples()]
    t["pointing"] = t["pointing"].map(lambda v: f"{v:.2f}")
    md.append(md_table(t[["model", "family", "variant", "map", "n", "AUROC", "AUPRC lift", "q", "pointing"]]))
    md.append("\n## Specificity of the target map vs each control (single prompt vs 3-paraphrase ensemble)\n")
    s = spec.copy()
    s["value"] = [fmt(r.median, r.ci_lo, r.ci_hi) for r in s.itertuples()]
    md.append(md_table(s.pivot_table(index=["model", "gain"], columns="variant", values="value",
                                     aggfunc="first").reset_index()))
    md.append("\n## Paired comparisons (per-image AUROC differences)\n")
    p = pr.copy()
    p["median diff [CI]"] = [fmt(r.median_diff_auroc, r.ci_lo, r.ci_hi) for r in p.itertuples()]
    p["p"] = p["p"].map(lambda v: "–" if np.isnan(v) else f"{v:.1e}")
    md.append(md_table(p[["model", "comparison", "n", "median diff [CI]", "p"]]))
    md.append("\n## Paraphrase sensitivity\n")
    q = per_prompt.copy()
    q["AUROC"] = [fmt(r.auroc, r.auroc_lo, r.auroc_hi) for r in q.itertuples()]
    q["AUPRC lift"] = [fmt(r.auprc_lift, r.lift_lo, r.lift_hi, 2) for r in q.itertuples()]
    md.append(md_table(q[["model", "category", "prompt", "n", "AUROC", "AUPRC lift"]]))
    md.append("\nMedian per-image Spearman between paraphrase maps of the same category:\n")
    md.append(md_table(rho.round(3)))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote tables -> {out}")


if __name__ == "__main__":
    main()
