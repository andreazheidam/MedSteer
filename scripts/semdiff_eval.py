"""Per-image metrics for training-free semantic differential steering (follow-up Sec. 1).

Works on any run whose case files store `native` (P, N) for every prompt of
configs/prompts_semdiff.yaml: scripts/extract_semdiff.py (SteerViT) and the unchanged
scripts/extract_compare.py / scripts/extract_medsam3.py (CLIP, BiomedCLIP, SAM3, MedSAM3).

Maps (all on the 24x24 patch grid, scored at pixel level inside the evaluation region Omega):
  family "norm"   s~(x,p) = (s - mu_Omega) / (sigma_Omega + eps), mu/sigma over the pixels of Omega
                  (computed on the bilinearly upsampled map; the upsampling is affine-equivariant,
                  so normalising the patch map with these statistics is exact)
      variant "single": paraphrase 1 of every category;  "ens": mean of the 3 normalised paraphrases
      maps  target             s+
            ctrl_<c>           s_c
            D_<c>              s+ - s_c                     pairwise semantic differential
            D_mean             s+ - mean_c s_c              multi-control (follow-up eq. 1.2)
            D_env              s+ - max_c s_c  (per patch)  control-envelope differential
  family "raw"    the same differences WITHOUT normalisation. For SteerViT these are exactly the
                  head-projected contrasts D^H_c = w^T D^F_c (linear head; see extract_semdiff.py).
  family "feat"   SteerViT only: ||D^F_c|| and ||D^F_mean|| feature-level maps (stored by extraction)
  family "prompt" every individual prompt (raw map; rank metrics are invariant to the normalisation)

Metrics per map: pixel AUROC, AUPRC, AUPRC lift, pointing game and css_diff = q(s, M) =
mean(s in M) - mean(s in Omega \\ M). For a D_c map of family "norm", q(D_c) = q(s+) - q(s_c), the
quantity whose margin the specificity loss of Sec. 2 enforces.

Also writes semdiff_para.csv: per image and category, the mean pairwise Spearman correlation of the
paraphrase maps (how much the map depends on the wording).

Usage:
  python scripts/semdiff_eval.py --run runs_semdiff/steervit/gate1/kvasir --manifest data/manifest_final.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import read_manifest  # noqa: E402
from medsteer.metrics import localisation, spearman, upsample  # noqa: E402

TARGET = "pathology"
CONTROLS = ["anatomy", "counterfactual", "wrong_medical", "irrelevant"]
EPS = 1e-8
METRICS = ["auroc", "auprc", "auprc_lift", "css_diff", "pointing", "pointing_chance", "prevalence"]


def znorm(pm: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Normalise a patch map with mean/std of its upsampled pixel map inside the valid region."""
    pix = upsample(pm.astype(np.float64), valid.shape[0] if valid is not None else 336).reshape(-1)
    if valid is not None:
        pix = pix[valid.reshape(-1)]
    return (pm - pix.mean()) / (pix.std() + EPS)


def case_rows(cid, run, model, groups, prim, pids, dataset):
    z = np.load(run / "cases" / f"{cid}.npz")
    mask, frac = z["mask"], z["frac"]
    valid = z["valid"] if "valid" in z else None
    vfrac = z["valid_frac"] if "valid_frac" in z else None
    if not mask.any():
        return [], []
    nat = z["native"].astype(np.float64)
    order = list(z["prompt_ids"])
    assert order == pids, f"{cid}: prompt order differs from meta.json"
    zn = np.stack([znorm(nat[j], valid) for j in range(len(pids))])
    base = {"case_id": cid, "dataset": dataset, "model": model, "lesion_frac": float(mask.mean())}
    rows = []

    def score(family, variant, name, pm, category=""):
        r = localisation(pm, mask, frac, with_dice=False, valid=valid, valid_frac=vfrac)
        rows.append(base | {"family": family, "variant": variant, "map": name, "category": category}
                    | {k: r.get(k, np.nan) for k in METRICS})

    for variant in ("single", "ens"):
        if variant == "single":
            S = {c: zn[prim[c]] for c in groups}
            Rw = {c: nat[prim[c]] for c in groups}
        else:
            S = {c: zn[groups[c]].mean(0) for c in groups}
            Rw = {c: nat[groups[c]].mean(0) for c in groups}
        ctrl = [c for c in CONTROLS if c in S]
        score("norm", variant, "target", S[TARGET], TARGET)
        for c in ctrl:
            score("norm", variant, f"ctrl_{c}", S[c], c)
            score("norm", variant, f"D_{c}", S[TARGET] - S[c], c)
            score("raw", variant, f"D_{c}", Rw[TARGET] - Rw[c], c)
        score("norm", variant, "D_mean", S[TARGET] - np.mean([S[c] for c in ctrl], 0))
        score("norm", variant, "D_env", S[TARGET] - np.max([S[c] for c in ctrl], 0))
        score("raw", variant, "D_mean", Rw[TARGET] - np.mean([Rw[c] for c in ctrl], 0))
        if "fd_ens" in z:
            fd = z["fd_single" if variant == "single" else "fd_ens"]
            for k, key in enumerate(z["fd_keys"]):
                score("feat", variant, f"D_{key}", fd[k], key if key != "mean" else "")
    cat_of = {j: c for c, idx in groups.items() for j in idx}
    for j, pid in enumerate(pids):
        score("prompt", "single", pid, nat[j], cat_of.get(j, ""))

    para = []
    vp = (vfrac.reshape(-1) >= 0.5) if vfrac is not None else slice(None)
    for c, idx in groups.items():
        if len(idx) < 2:
            continue
        rhos = [spearman(nat[a][vp], nat[b][vp]) for a, b in combinations(idx, 2)]
        para.append(base | {"category": c, "spearman_mean": float(np.nanmean(rhos)),
                            "spearman_min": float(np.nanmin(rhos))})
    return rows, para


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. runs_semdiff/steervit/gate1/kvasir")
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--model", default=None, help="default: taken from the run path <out>/<model>/gate1/<ds>")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    model = args.model or meta.get("model") or run.parts[-3]
    ds = meta["dataset"]
    prompts = meta["prompts"]
    pids = [p["id"] for p in prompts]
    groups = {}
    for j, p in enumerate(prompts):
        groups.setdefault(p["category"], []).append(j)
    groups = {c: groups[c] for c in [TARGET] + CONTROLS if c in groups}
    prim = {c: next((j for j in idx if prompts[j].get("primary", True)), idx[0]) for c, idx in groups.items()}
    print(f"[{model}] {ds}: categories " + ", ".join(f"{c}={len(i)}" for c, i in groups.items()), flush=True)

    cases = {c.case_id: c for c in read_manifest(args.manifest, datasets=[ds])}
    ids = sorted(p.stem for p in (run / "cases").glob("*.npz")
                 if p.stem in cases and cases[p.stem].mask_paths and not cases[p.stem].is_healthy)
    print(f"  {len(ids)} lesion cases", flush=True)
    fn = partial(case_rows, run=run, model=model, groups=groups, prim=prim, pids=pids, dataset=ds)
    rows, para = [], []
    with Pool(args.workers) as pool:
        for k, (r, p) in enumerate(pool.imap_unordered(fn, ids, chunksize=4)):
            rows += r
            para += p
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_csv(run / "semdiff_loc.csv.gz", index=False)
    pd.DataFrame(para).to_csv(run / "semdiff_para.csv", index=False)
    print(f"wrote {len(rows)} rows -> {run / 'semdiff_loc.csv.gz'}")


if __name__ == "__main__":
    main()
