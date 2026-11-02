"""Turn extracted per-case maps into tidy CSVs (one row per case x prompt x map).

Outputs in <run>/<dataset>/:
  loc.csv        localisation metrics on lesion images (RQ1, RQ2, RQ4-size, Exp. 5)
  global.csv     image-level steering magnitudes on ALL images incl. healthy (RQ3 presence)
  direction.csv  cosine similarity of pooled residual vectors between prompt pairs (RQ2/RQ3)

Usage:
  python scripts/evaluate.py --run runs/steerdinov2/gate1/busi --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.baselines import center_gaussian  # noqa: E402
from medsteer.config import Prompt, resolve_category  # noqa: E402
from medsteer.data import load_image, mask_to_grid, read_manifest, refine_valid  # noqa: E402  (legacy runs)
from medsteer.metrics import localisation, spearman  # noqa: E402

DICE_MAPS = {"csr_abs", "spec", "native", "native_delta", "attn"}   # Dice only where it is reported
NULL_MAPS = {"csr_abs", "spec", "native", "native_delta", "native0", "attn", "baseline_norm", "baseline_salience",
             "baseline_center", "attn0"}


def case_rows(cid, run, cases, prompts, null_ids, patch_thr, reference, legacy_valid=False):
    z = np.load(run / "cases" / f"{cid}.npz")
    c = cases[cid]
    mask, frac = z["mask"], z["frac"]
    valid = z["valid"] if "valid" in z else None
    vfrac = z["valid_frac"] if "valid_frac" in z else None
    g = frac.shape[0]
    if legacy_valid and valid is not None and c.dataset == "kvasir":   # only v1/v2 runs (old valid region)
        valid = refine_valid(load_image(c.image_path, mask.shape[0]), valid, c.dataset, mask)
        vfrac = mask_to_grid(valid, g)
    base = {"case_id": cid, "dataset": c.dataset, "split": c.split, "label": c.label,
            "plane": c.plane, "patient": c.patient, "healthy": c.is_healthy, "lesion_frac": float(mask.mean()),
            "lesion_patches": float(frac.sum())}
    pids = list(z["prompt_ids"])

    # ---------------- global (all images)
    glob = []
    for j, pid in enumerate(pids):
        p = prompts[pid]
        csr = z["csr_abs"][j]
        nat = z["native"][j]
        k = max(1, int(0.05 * len(csr)))
        glob.append(base | {"prompt": pid, "category": resolve_category(p, c.label),
                            "primary": p.primary,
                            "mag_mean": float(csr.mean()),
                            "mag_top5": float(np.sort(csr)[-k:].mean()),
                            "cls_norm": float(np.linalg.norm(z["dF_cls"][j].astype(np.float32))),
                            "native_max": float(nat.max()),
                            "native_lse": float(np.log(np.exp(nat - nat.max()).sum()) + nat.max())})

    # ---------------- direction (all images; lesion-pooled where a lesion exists)
    dirs = []
    prim = [i for i, pid in enumerate(pids) if prompts[pid].primary]
    for vec_name in ("dF_lesion", "dF_mean", "dF_cls"):
        if vec_name == "dF_lesion" and c.is_healthy:
            continue
        V = z[vec_name].astype(np.float32)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        S = Vn @ Vn.T
        for a_i, a in enumerate(prim):
            for b in prim[a_i + 1:]:
                dirs.append(base | {"vector": vec_name, "prompt_a": pids[a], "prompt_b": pids[b],
                                    "cat_a": resolve_category(prompts[pids[a]], c.label),
                                    "cat_b": resolve_category(prompts[pids[b]], c.label),
                                    "cos": float(S[a, b])})

    # ---------------- localisation (lesion images only)
    loc = []
    if mask.any() and not c.is_healthy:
        others = [np.load(run / "cases" / f"{o}.npz")["mask"] for o in null_ids.get(cid, [])]
        others_frac = [np.load(run / "cases" / f"{o}.npz")["frac"] for o in null_ids.get(cid, [])]
        maps = []
        for j, pid in enumerate(pids):
            p = prompts[pid]
            for m in ("csr_abs", "csr_rel", "csr_cos", "native") + (("attn",) if "attn" in z else ()):
                maps.append((m, pid, z[m][j]))
            if pid != reference:
                maps.append(("spec", pid, z["spec"][j]))
            if "native0" in z:   # projection of the residual onto the localisation head: w.dF
                maps.append(("native_delta", pid, z["native"][j] - z["native0"]))
        maps += [("baseline_norm", "none", z["f0_norm"]), ("baseline_salience", "none", z["salience"]),
                 ("baseline_center", "none", center_gaussian(g))]
        if "attn0" in z:
            maps.append(("attn0", "none", z["attn0"]))
        if "native0" in z:   # localisation head WITHOUT a prompt (does DINOv2 already localise?)
            maps.append(("native0", "none", z["native0"]))
        for m, pid, pmap in maps:
            p = prompts.get(pid)
            row = base | {"prompt": pid, "map": m,
                          "category": resolve_category(p, c.label) if p else "none",
                          "primary": p.primary if p else True}
            row |= localisation(pmap, mask, frac, patch_thr, with_dice=(m in DICE_MAPS and row["primary"]),
                                valid=valid, valid_frac=vfrac)
            if m in NULL_MAPS and row["primary"] and others:
                nulls = []
                for om, of in zip(others, others_frac):
                    if valid is not None:
                        om = om & valid  # the other mask, restricted to this image's content
                    if om.any():
                        nulls.append(localisation(pmap, om, of, patch_thr, with_dice=False,
                                                  valid=valid, valid_frac=vfrac))
                if nulls:
                    row["null_auroc"] = float(np.mean([n["auroc"] for n in nulls]))
                    row["null_auprc_lift"] = float(np.mean([n["auprc_lift"] for n in nulls]))
            if m == "csr_abs":
                vp = (vfrac.reshape(-1) >= 0.5) if vfrac is not None else slice(None)
                row["rho_featnorm"] = spearman(pmap[vp], z["f0_norm"][vp])
                row["rho_native"] = spearman(pmap[vp], z["native"][pids.index(pid)][vp])
            loc.append(row)
    return loc, glob, dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. runs/steerdinov2/gate1/busi")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--patch-thr", type=float, default=0.5)
    ap.add_argument("--null-perms", type=int, default=1)
    ap.add_argument("--size-bins", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    prompts = {p["id"]: Prompt(**p) for p in meta["prompts"]}
    cases = {c.case_id: c for c in read_manifest(args.manifest, datasets=[meta["dataset"]])}
    ids = sorted(p.stem for p in (run / "cases").glob("*.npz") if p.stem in cases)
    lesion_ids = [i for i in ids if cases[i].mask_paths and not cases[i].is_healthy]
    rng = np.random.default_rng(0)
    null_ids = {}
    for i in lesion_ids:  # mask-permutation null: masks of other lesion images
        pool = [o for o in lesion_ids if o != i]
        null_ids[i] = list(rng.choice(pool, size=min(args.null_perms, len(pool)), replace=False))
    print(f"{meta['dataset']}: {len(ids)} cases ({len(lesion_ids)} with lesion masks)", flush=True)

    fn = partial(case_rows, run=run, cases=cases, prompts=prompts, null_ids=null_ids,
                 patch_thr=args.patch_thr, reference=meta["reference"],
                 legacy_valid="valid_region" not in meta)
    print(f"evaluation region: {meta.get('valid_region', 'LEGACY intensity-based region')}", flush=True)
    loc, glob, dirs = [], [], []
    with Pool(args.workers) as pool:
        for k, (l, g, d) in enumerate(pool.imap_unordered(fn, ids, chunksize=4)):
            loc += l; glob += g; dirs += d
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(ids)}", flush=True)

    loc, glob, dirs = pd.DataFrame(loc), pd.DataFrame(glob), pd.DataFrame(dirs)
    # dataset-specific size bins from the GT masks only (independent of any result)
    if len(loc):
        sizes = loc.drop_duplicates("case_id").set_index("case_id")["lesion_frac"]
        labels = ["small", "medium", "large"] if args.size_bins == 3 else [f"q{i+1}" for i in range(args.size_bins)]
        bins = pd.qcut(sizes, args.size_bins, labels=labels)
        for df in (loc, glob, dirs):
            df["size_bin"] = df["case_id"].map(bins).astype(object)
        (run / "size_bins.json").write_text(json.dumps(
            {"edges": list(np.quantile(sizes, np.linspace(0, 1, args.size_bins + 1)))}, indent=2))
    loc.to_csv(run / "loc.csv", index=False)
    glob.to_csv(run / "global.csv", index=False)
    dirs.to_csv(run / "direction.csv", index=False)
    print(f"wrote {len(loc)} loc rows, {len(glob)} global rows, {len(dirs)} direction rows -> {run}")


if __name__ == "__main__":
    main()
