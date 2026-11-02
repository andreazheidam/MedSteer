"""All statistics and figures for the paper, from the CSVs written by evaluate.py.

Usage:
  python scripts/analyze.py --runs runs/steerdinov2/gate1/brisc runs/steerdinov2/gate1/busi \
      --out paper
Writes paper/tables/*.csv + *.tex and paper/figs/*.pdf. Holm correction is applied within
each research question; every test is paired on the same images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer import plotting as P  # noqa: E402
from medsteer import stats as S  # noqa: E402

MAIN = "csr_abs"
METRICS = ["auroc", "auprc_lift", "css_diff"]


# ----------------------------------------------------------------------------- helpers

def per_case_category(loc: pd.DataFrame, map_name: str, metric: str) -> pd.DataFrame:
    """cases x category table, averaging primary prompts within a category (e.g. cat + car)."""
    d = loc[(loc["map"] == map_name) & loc["primary"]]
    return d.pivot_table(index="case_id", columns="category", values=metric, aggfunc="mean")


def add_holm(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    if df.empty or "p" not in df:   # e.g. no class prompts (Kvasir, Sunnybrook)
        return df
    for _, idx in df.groupby(group_cols if len(group_cols) > 1 else group_cols[0]).groups.items():
        df.loc[idx, "p_holm"] = S.holm(df.loc[idx, "p"].values)
    return df


def to_tex(df: pd.DataFrame, path: Path, cols: list[str], caption: str = "") -> None:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "--"
            if abs(v) < 1e-3 and v != 0:
                return f"{v:.0e}"
            return f"{v:.3f}"
        return str(v).replace("_", r"\_")
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, r in df[cols].iterrows():
        lines.append(" & ".join(fmt(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.insert(0, f"% {caption}")
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- RQ1

def rq1(loc, ds):
    """Is each map localised? one-sample tests vs chance and vs the mask-permutation null."""
    rows = []
    d = loc[loc["primary"]]
    for (m, cat), g in d.groupby(["map", "category"]):
        pc = g.groupby("case_id")[["auroc", "auprc_lift", "css_diff", "null_auroc"]].mean()
        for metric, ref in (("auroc", 0.5), ("auprc_lift", 1.0), ("css_diff", 0.0)):
            rows.append({"dataset": ds, "map": m, "category": cat, "metric": metric, "test": "vs_chance"}
                        | S.one_sample(pc[metric].values, ref))
        if pc["null_auroc"].notna().any():
            r = S.paired(pc["auroc"].values, pc["null_auroc"].values)
            rows.append({"dataset": ds, "map": m, "category": cat, "metric": "auroc",
                         "test": "vs_mask_permutation", "median": np.nanmedian(pc["auroc"])} | r)
    return add_holm(pd.DataFrame(rows), ["dataset", "test", "metric"])


# ----------------------------------------------------------------------------- RQ2

COMPARISONS = [("pathology", "anatomy"), ("pathology", "irrelevant"), ("anatomy", "irrelevant"),
               ("pathology", "related_wrong"), ("pathology", "nuisance"),
               ("class_correct", "class_wrong"), ("contrast_pos", "contrast_neg"),
               ("contrast_pos", "negation")]


def rq2(loc, ds):
    fried, pairs = [], []
    for m in (MAIN, "spec", "native"):
        for metric in METRICS:
            wide = per_case_category(loc, m, metric)
            cols = [c for c in ("pathology", "anatomy", "irrelevant") if c in wide]
            if len(cols) == 3:
                fried.append({"dataset": ds, "map": m, "metric": metric,
                              "conditions": "+".join(cols)} | S.friedman(wide[cols]))
            for a, b in COMPARISONS:
                if a in wide and b in wide:
                    pairs.append({"dataset": ds, "map": m, "metric": metric, "a": a, "b": b}
                                 | S.paired(wide[a].values, wide[b].values))
    return pd.DataFrame(fried), add_holm(pd.DataFrame(pairs), ["dataset", "map", "metric"])


def vs_baselines(loc, ds):
    """Pathology-prompt maps vs prompt-free baselines (does the prompt add spatial info?)."""
    rows = []
    for metric in ("auroc", "auprc_lift"):
        base = {m: loc[loc["map"] == m].set_index("case_id")[metric]
                for m in ("baseline_norm", "baseline_salience", "baseline_center", "native0")
                if m in set(loc["map"])}
        for m in (MAIN, "spec", "native", "native_delta"):
            a = loc[(loc["map"] == m) & (loc["category"] == "pathology") & loc["primary"]]
            a = a.groupby("case_id")[metric].mean()
            for bname, b in base.items():
                j = pd.concat([a, b], axis=1, join="inner").dropna()
                rows.append({"dataset": ds, "metric": metric, "map": m, "baseline": bname}
                            | S.paired(j.iloc[:, 0].values, j.iloc[:, 1].values))
    return add_holm(pd.DataFrame(rows), ["dataset", "metric"])


def native_vs_csr(loc, ds):
    """Exp. 5: native heatmap vs CSR vs attention on the pathology prompt."""
    d = loc[(loc["category"] == "pathology") & loc["primary"]]
    rows = []
    maps = [m for m in (MAIN, "native", "native_delta", "attn", "spec") if m in set(d["map"])]
    for metric in ("auroc", "auprc_lift", "dice_otsu"):
        w = d.pivot_table(index="case_id", columns="map", values=metric, aggfunc="mean")
        for i, a in enumerate(maps):
            for b in maps[i + 1:]:
                if a in w and b in w:
                    rows.append({"dataset": ds, "metric": metric, "a": a, "b": b}
                                | S.paired(w[a].values, w[b].values))
    out = add_holm(pd.DataFrame(rows), ["dataset", "metric"])
    rho = d[d["map"] == MAIN]["rho_native"]
    corr = {"dataset": ds, "median_rho_csr_native": float(rho.median()),
            "median_rho_csr_featnorm": float(d[d["map"] == MAIN]["rho_featnorm"].median())}
    return out, corr


# ----------------------------------------------------------------------------- directions

def directions(dirs, ds):
    d = dirs[dirs["vector"] == "dF_lesion"]
    rows = []
    # median cosine between categories
    cats = [c for c in P.CAT_ORDER if c in set(d["cat_a"]) | set(d["cat_b"])]
    mat = np.full((len(cats), len(cats)), np.nan)
    for i, a in enumerate(cats):
        mat[i, i] = 1.0
        for j, b in enumerate(cats):
            if i < j:
                s = d[((d.cat_a == a) & (d.cat_b == b)) | ((d.cat_a == b) & (d.cat_b == a))]["cos"]
                if len(s):
                    mat[i, j] = mat[j, i] = s.median()
    # is pathology closer to class_correct than to controls?
    def cos_with(cat_x):
        s = d[((d.cat_a == "pathology") & (d.cat_b == cat_x)) | ((d.cat_b == "pathology") & (d.cat_a == cat_x))]
        return s.groupby("case_id")["cos"].mean()
    ref = cos_with("class_correct")
    for other in ("related_wrong", "irrelevant", "nuisance", "anatomy", "class_wrong"):
        o = cos_with(other)
        j = pd.concat([ref, o], axis=1, join="inner").dropna()
        if len(j):
            rows.append({"dataset": ds, "a": "cos(pathology, class_correct)",
                         "b": f"cos(pathology, {other})"} | S.paired(j.iloc[:, 0].values, j.iloc[:, 1].values))
    return add_holm(pd.DataFrame(rows), ["dataset"]), mat, cats


# ----------------------------------------------------------------------------- RQ3

SCORES = ["mag_mean", "mag_top5", "cls_norm", "native_max"]


def rq3(glob, loc, ds):
    rows = []
    g = glob[glob["primary"]]
    # presence: lesion image vs healthy image, per prompt and score
    for cat in ("pathology", "contrast_pos", "contrast_neg", "negation", "irrelevant", "anatomy"):
        sub = g[g["category"] == cat].groupby("case_id").agg({**{s: "mean" for s in SCORES}, "healthy": "first"})
        for s in SCORES:
            rows.append({"dataset": ds, "analysis": "presence", "prompt": cat, "score": s}
                        | S.auroc_with_ci(sub[s].values, ~sub["healthy"].values))
    # contrast: d = score(pos) - score(neg) separates lesion vs healthy?
    w = g.pivot_table(index="case_id", columns="category", values=SCORES, aggfunc="mean")
    healthy = g.groupby("case_id")["healthy"].first()
    for neg in ("contrast_neg", "negation"):
        for s in SCORES:
            if ("contrast_pos" in w[s]) and (neg in w[s]):
                dval = (w[s]["contrast_pos"] - w[s][neg]).reindex(healthy.index)
                rows.append({"dataset": ds, "analysis": f"contrast_pos-{neg}", "prompt": "diff", "score": s}
                            | S.auroc_with_ci(dval.values, ~healthy.values))
    # size contrast on lesion images: d = score(large) - score(small) vs GT size
    les = g[~g["healthy"]]
    ws = les.pivot_table(index="case_id", columns="category", values=SCORES, aggfunc="mean")
    size = les.groupby("case_id")["lesion_frac"].first()
    sizebin = les.groupby("case_id")["size_bin"].first()
    size_rows = []
    for s in SCORES:
        if "size_large" in ws[s] and "size_small" in ws[s]:
            dval = (ws[s]["size_large"] - ws[s]["size_small"]).reindex(size.index)
            size_rows.append({"dataset": ds, "score": s, "analysis": "large-small vs size"}
                             | S.spearman_ci(dval.values, np.log(size.values + 1e-8)))
            keep = sizebin.isin(["small", "large"])
            rows.append({"dataset": ds, "analysis": "size_contrast(large vs small bin)", "prompt": "diff", "score": s}
                        | S.auroc_with_ci(dval[keep].values, (sizebin[keep] == "large").values))
    # class prompts: localisation of correct vs wrong class prompt (paired, lesion images)
    cls_rows = []
    for metric in ("auroc", "auprc_lift", "css_diff"):
        wide = per_case_category(loc, MAIN, metric)
        if "class_correct" in wide and "class_wrong" in wide:
            cls_rows.append({"dataset": ds, "map": MAIN, "metric": metric}
                            | S.paired(wide["class_correct"].values, wide["class_wrong"].values))
    out = pd.DataFrame(rows)
    return out, pd.DataFrame(size_rows), pd.DataFrame(cls_rows)


def class_discrimination(glob, ds, labels_pairs):
    """Image-level: does score(prompt_A) - score(prompt_B) separate diagnosis A from B?"""
    rows = []
    g = glob[glob["primary"]]
    for a, b in labels_pairs:
        sub = g[g["label"].isin([a, b]) & g["prompt"].isin([a, b])]
        if sub.empty:   # labels without matching class prompts (e.g. Sunnybrook diagnoses)
            continue
        w = sub.pivot_table(index="case_id", columns="prompt", values=SCORES, aggfunc="mean")
        lab = sub.groupby("case_id")["label"].first()
        for s in SCORES:
            if a in w[s] and b in w[s]:
                dval = (w[s][a] - w[s][b]).reindex(lab.index)
                rows.append({"dataset": ds, "pair": f"{a} vs {b}", "score": s}
                            | S.auroc_with_ci(dval.values, (lab == a).values))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- size (RQ4)

def size_analysis(loc, ds):
    rows = []
    d = loc[loc["primary"]]
    for m in (MAIN, "native", "native_delta", "spec"):
        for cat in ("pathology", "anatomy", "irrelevant"):
            sub = d[(d["map"] == m) & (d["category"] == cat)].groupby("case_id").agg(
                {"auroc": "mean", "auprc_lift": "mean", "lesion_frac": "first", "size_bin": "first"})
            for metric in ("auroc", "auprc_lift"):
                rows.append({"dataset": ds, "map": m, "category": cat, "metric": metric, "test": "spearman_logsize"}
                            | S.spearman_ci(sub[metric].values, np.log(sub["lesion_frac"].values)))
                groups = {b: sub[sub["size_bin"] == b][metric].values for b in ("small", "medium", "large")}
                rows.append({"dataset": ds, "map": m, "category": cat, "metric": metric, "test": "kruskal_bins",
                             **{f"median_{b}": float(np.nanmedian(v)) if len(v) else np.nan for b, v in groups.items()}}
                            | S.kruskal(groups))
    # does the prompt help more for small lesions? benefit = pathology - irrelevant
    wide = per_case_category(loc, MAIN, "auroc")
    size = d.groupby("case_id")["lesion_frac"].first()
    if "pathology" in wide and "irrelevant" in wide:
        benefit = (wide["pathology"] - wide["irrelevant"]).reindex(size.index)
        rows.append({"dataset": ds, "map": MAIN, "category": "pathology-irrelevant", "metric": "auroc",
                     "test": "benefit_vs_logsize"} | S.spearman_ci(benefit.values, np.log(size.values)))
    return pd.DataFrame(rows)


def paraphrase(loc, ds):
    d = loc[(loc["map"] == MAIN) & (loc["category"] == "pathology")]
    w = d.pivot_table(index="case_id", columns="prompt", values="auroc")
    res = {"dataset": ds, "prompts": "+".join(w.columns)} | S.friedman(w) if w.shape[1] >= 3 else {"dataset": ds}
    res |= {f"median_{c}": float(w[c].median()) for c in w.columns}
    return pd.DataFrame([res])


# ----------------------------------------------------------------------------- pointing game

POINT_ROWS = [("native0", "none"), ("native", "pathology"), ("native", "anatomy"),
              ("native", "related_wrong"), ("native", "irrelevant"), ("native", "nuisance"),
              ("native_delta", "pathology"), ("csr_abs", "pathology"),
              ("baseline_salience", "none"), ("baseline_center", "none")]


def _units(d: pd.DataFrame) -> pd.Series:
    """Analysis unit per case: the patient when one is recorded (Sunnybrook), else the image."""
    if "patient" in d and d["patient"].fillna("").astype(str).str.len().gt(0).any():
        return d["patient"].fillna(d["case_id"]).astype(str)
    return d["case_id"]


def pointing_table(loc, ds):
    """Hit rate of the arg-max patch, with a bootstrap CI over units (patients if available)
    and the per-image chance rate. Units are averaged first, so a patient with 20 slices counts
    once."""
    if "pointing" not in loc:
        return pd.DataFrame()
    d = loc[loc["primary"]].copy()
    d["unit"] = _units(d)
    rows = []
    for m, cat in POINT_ROWS:
        sub = d[(d["map"] == m) & (d["category"] == cat)]
        if sub.empty:
            continue
        per_case = sub.groupby(["unit", "case_id"])[["pointing", "pointing_chance"]].mean()
        per_unit = per_case.groupby("unit").mean()
        lo, hi = S.bootstrap_ci(per_unit["pointing"].values, stat=np.mean)
        rows.append({"dataset": ds, "map": m, "category": cat, "n_images": len(per_case),
                     "n_units": len(per_unit), "hit_rate": float(per_unit["pointing"].mean()),
                     "ci_lo": lo, "ci_hi": hi, "chance": float(per_unit["pointing_chance"].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- prompt effects

def prompt_effects(loc, ds):
    """Split the gain from prompting into
         generic  = any prompt ("the cat") vs no prompt     (native[irrelevant] - native0)
         specific = target prompt vs a meaningless prompt   (native[pathology] - native[irrelevant])
       per case, then per unit (patient if recorded), median difference with bootstrap CI."""
    rows = []
    d = loc[loc["primary"]].copy()
    d["unit"] = _units(d)
    for metric in ("auroc", "pointing"):
        if metric not in d:
            continue
        nat = d[d["map"] == "native"].pivot_table(index="case_id", columns="category", values=metric, aggfunc="mean")
        n0 = d[d["map"] == "native0"].groupby("case_id")[metric].mean()
        unit = d.groupby("case_id")["unit"].first()
        if "pathology" not in nat or "irrelevant" not in nat:
            continue
        parts = {"generic (irrelevant - none)": nat["irrelevant"] - n0.reindex(nat.index),
                 "specific (target - irrelevant)": nat["pathology"] - nat["irrelevant"],
                 "total (target - none)": nat["pathology"] - n0.reindex(nat.index)}
        for name, diff in parts.items():
            u = diff.groupby(unit.reindex(diff.index)).mean().dropna().values
            lo, hi = S.bootstrap_ci(u, stat=np.median if metric == "auroc" else np.mean)
            rows.append({"dataset": ds, "metric": metric, "effect": name, "n_units": len(u),
                         "estimate": float(np.median(u) if metric == "auroc" else np.mean(u)),
                         "ci_lo": lo, "ci_hi": hi,
                         "unprompted": float(n0.groupby(unit.reindex(n0.index)).mean().median()
                                             if metric == "auroc" else n0.groupby(unit.reindex(n0.index)).mean().mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def fig_localisation(loc, ds, out):
    fig, axes = P.plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, metric, ref, lab in ((axes[0], "auroc", 0.5, "pixel AUROC"),
                                 (axes[1], "auprc_lift", 1.0, "AUPRC / prevalence")):
        d = loc[loc["primary"]]
        data = {}
        for cat in [c for c in P.CAT_ORDER[:7] if c in set(d["category"])]:
            data[cat] = {m: d[(d["category"] == cat) & (d["map"] == m)].groupby("case_id")[metric].mean().values
                         for m in (MAIN, "native", "native_delta") if m in set(d["map"])}
        data["no prompt"] = {m: d[d["map"] == m][metric].values
                             for m in ("native0", "baseline_salience", "baseline_center") if m in set(d["map"])}
        P.box_by_category(ax, data, lab, ref)
        if metric == "auprc_lift":
            ax.set_yscale("log")
    fig.suptitle(f"{ds}: where does the prompt change the representation?", x=0.01, ha="left", y=1.08)
    fig.savefig(out / f"{ds}_localisation.pdf")
    P.plt.close(fig)


def fig_size(loc, ds, out):
    d = loc[loc["primary"] & (loc["category"] == "pathology") & loc["map"].isin([MAIN, "native"])]
    d = d.groupby(["case_id", "map"]).agg({"auroc": "mean", "lesion_frac": "first"}).reset_index()
    fig, ax = P.plt.subplots(figsize=(3.4, 2.4))
    P.size_scatter(ax, d, "lesion_frac", "auroc", "map")
    ax.set_xlabel("lesion area / image area")
    ax.set_ylabel("pixel AUROC (pathology prompt)")
    fig.savefig(out / f"{ds}_size.pdf")
    P.plt.close(fig)


def fig_directions(mat, cats, ds, out):
    fig, ax = P.plt.subplots(figsize=(3.4, 3.0))
    P.cos_matrix(ax, mat, cats, f"{ds}: median cos of lesion residuals")
    fig.savefig(out / f"{ds}_directions.pdf")
    P.plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    acc = {k: [] for k in ("rq1", "rq2_friedman", "rq2_pairs", "baselines", "native_vs_csr",
                           "map_corr", "directions", "rq3", "rq3_size", "rq3_class", "class_disc",
                           "size", "paraphrase", "pointing", "prompt_effects")}
    for r in args.runs:
        r = Path(r)
        loc, glob, dirs = (pd.read_csv(r / f) for f in ("loc.csv", "global.csv", "direction.csv"))
        ds = loc["dataset"].iloc[0] if len(loc) else glob["dataset"].iloc[0]
        print(f"[{ds}] {loc.case_id.nunique()} lesion images, {glob.case_id.nunique()} images total")
        acc["rq1"].append(rq1(loc, ds))
        f, p = rq2(loc, ds); acc["rq2_friedman"].append(f); acc["rq2_pairs"].append(p)
        acc["baselines"].append(vs_baselines(loc, ds))
        n, c = native_vs_csr(loc, ds); acc["native_vs_csr"].append(n); acc["map_corr"].append(pd.DataFrame([c]))
        dr, mat, cats = directions(dirs, ds); acc["directions"].append(dr)
        a, b, c3 = rq3(glob, loc, ds); acc["rq3"].append(a); acc["rq3_size"].append(b); acc["rq3_class"].append(c3)
        labs = sorted(set(glob.loc[~glob["healthy"], "label"]))
        pairs = [(x, y) for i, x in enumerate(labs) for y in labs[i + 1:]]
        acc["class_disc"].append(class_discrimination(glob, ds, pairs))
        acc["size"].append(size_analysis(loc, ds))
        acc["paraphrase"].append(paraphrase(loc, ds))
        acc["pointing"].append(pointing_table(loc, ds))
        acc["prompt_effects"].append(prompt_effects(loc, ds))
        fig_localisation(loc, ds, out / "figs")
        fig_size(loc, ds, out / "figs")
        fig_directions(mat, cats, ds, out / "figs")

    tabs = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in acc.items()}
    for k, t in tabs.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)

    # compact LaTeX tables for the paper (edit columns as needed)
    t = tabs["rq1"]
    t = t[(t.test == "vs_chance") & (t.metric == "auroc") & (t.category.isin(["pathology", "anatomy", "irrelevant", "none"]))]
    to_tex(t.sort_values(["dataset", "map", "category"]), out / "tables" / "rq1.tex",
           ["dataset", "map", "category", "n", "median", "ci_lo", "ci_hi", "p_holm"],
           "RQ1: pixel AUROC vs 0.5")
    t = tabs["rq2_pairs"]
    to_tex(t[(t["map"] == MAIN) & (t.metric == "auroc")], out / "tables" / "rq2.tex",
           ["dataset", "a", "b", "n", "median_diff", "ci_lo", "ci_hi", "p_holm", "r_rb"],
           "RQ2: paired prompt comparisons, CSR pixel AUROC")
    t = tabs["rq3"]
    to_tex(t[t.score == "mag_top5"], out / "tables" / "rq3.tex",
           ["dataset", "analysis", "prompt", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"],
           "RQ3: image-level AUROC of steering magnitude (top-5% CSR)")
    print("tables ->", out / "tables", "| figures ->", out / "figs")


if __name__ == "__main__":
    main()
