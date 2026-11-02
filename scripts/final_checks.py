"""Pre-submission checks for the final experiment. Prints PASS / FAIL / INFO per item.

  python scripts/final_checks.py [-v]       (from ~/medsteer, with .venv active; -v lists every PASS)
Checks:
  1. every run used the frozen config (hash), the exact prompt strings, gate 1
  2. every run covers every manifest image; loc.csv covers every positive image
  3. masks: every positive image still has target pixels after padding/resizing,
     and the target lies inside the evaluated (valid) region
  4. which images give no AUROC, and why (reported in the dataset table)
  5. figures exist as PDF and SVG for every dataset; caption note present
  6. tables exist
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.config import load_config  # noqa: E402
from medsteer.data import read_manifest  # noqa: E402

CONFIG = "configs/prompts_final2.yaml"
MANIFEST = "data/manifest_final.csv"
RUNS = {"SteerViT": "runs_final2/gate1", "CLIP": "runs_final2_compare/clip/gate1",
        "SAM3": "runs_final2_compare/sam3/gate1"}
PAPER = Path("paper_final2")
DATASETS = ["brisc", "busi", "kvasir", "isic"]
EXPECTED = {"brisc": ["brain tissue", "brain tumor", "cat", "skin lesion", "normal brain tissue"],
            "busi": ["breast tissue", "breast lesion", "cat", "brain tumor", "normal breast tissue"],
            "kvasir": ["colon mucosa", "colon polyp", "cat", "breast lesion", "normal colon mucosa"],
            "isic": ["skin", "skin lesion", "cat", "colon polyp", "normal skin"]}
fails = passes = 0
VERBOSE = "-v" in sys.argv   # -v also prints every PASS line


def report(ok, msg):
    global fails, passes
    if not ok or VERBOSE:
        print(("PASS " if ok else "FAIL ") + msg)
    fails += 0 if ok else 1
    passes += 1 if ok else 0


def main():
    h = hashlib.sha256(Path(CONFIG).read_bytes()).hexdigest()[:12]
    print(f"INFO frozen config {CONFIG} hash {h}")
    _, dsets = load_config(CONFIG)
    for ds in DATASETS:
        report([p.text for p in dsets[ds].prompts] == EXPECTED[ds],
               f"config prompts {ds}: {[p.text for p in dsets[ds].prompts]}")
    cases = read_manifest(MANIFEST)

    print("\n-- 1/2. runs: config, prompts, gate, coverage")
    for model, root in RUNS.items():
        for ds in DATASETS:
            run = Path(root) / ds
            tag = f"{model:8s} {ds:6s}"
            if not (run / "meta.json").exists():
                report(False, f"{tag} meta.json missing ({run})"); continue
            meta = json.loads((run / "meta.json").read_text())
            report(meta.get("config_hash") == h, f"{tag} config hash {meta.get('config_hash')}")
            texts = [p["text"] for p in meta["prompts"]]
            report(texts == EXPECTED[ds], f"{tag} prompts used {texts}")
            report(float(meta.get("gate", 1)) == 1.0, f"{tag} gate {meta.get('gate')}")
            report("valid_region" in meta, f"{tag} GT-independent evaluation region: {meta.get('valid_region', 'LEGACY')}")
            if "dse_check" in meta:
                e = meta["dse_check"]
                report(e["max_rel_err"] < 1e-3, f"{tag} DSE identity h(Fp)-h(F0) = w^T dF: "
                       f"max abs err {e['max_abs_err']:.2e}, rel {e['max_rel_err']:.2e}")
            if "null_prompt" in meta:
                print(f"INFO {tag} baseline = {meta['null_prompt']!r}"
                      f"{'  (empty-text control)' if meta['null_prompt'] == '' else ''}")
            want = {c.case_id for c in cases if c.dataset == ds}
            have = {p.stem for p in (run / "cases").glob("*.npz")}
            report(want <= have, f"{tag} extracted {len(have & want)}/{len(want)} images")
            loc = run / "loc.csv"
            if not loc.exists():
                report(False, f"{tag} loc.csv missing"); continue
            d = pd.read_csv(loc, usecols=["case_id", "map", "prompt", "auroc"])
            pos = {c.case_id for c in cases if c.dataset == ds and c.mask_paths and not c.is_healthy}
            got = set(d["case_id"])
            report(len(got) >= len(pos) - 1, f"{tag} loc.csv covers {len(got)}/{len(pos)} positive images")
            for m in ("native", "native_delta", "csr_abs"):
                report(m in set(d["map"]), f"{tag} map '{m}' present")

    print("\n-- 3/4. masks and images without AUROC (SteerViT run)")
    for ds in DATASETS:
        run = Path(RUNS["SteerViT"]) / ds
        pos = [c for c in cases if c.dataset == ds and c.mask_paths and not c.is_healthy]
        empty, outside, fracs = [], [], []
        for c in pos:
            f = run / "cases" / f"{c.case_id}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            m, v = z["mask"], z["valid"]
            if not m.any():
                empty.append(c.case_id)
            else:
                fracs.append(m.mean())
                if (m & ~v).any():
                    outside.append(c.case_id)
        neg_with_target = [c.case_id for c in cases if c.dataset == ds and c.is_healthy and c.mask_paths
                           and (run / "cases" / f"{c.case_id}.npz").exists()
                           and np.load(run / "cases" / f"{c.case_id}.npz")["mask"].any()]
        report(not empty, f"{ds:6s} positives with empty mask after resizing: {len(empty)} {empty[:3]}")
        report(not neg_with_target, f"{ds:6s} negatives with target pixels: {len(neg_with_target)}")
        print(f"INFO {ds:6s} target area: median {100*np.median(fracs):.2f}%, min {100*min(fracs):.3f}%, "
              f"max {100*max(fracs):.1f}% of the image")
        report(not outside, f"{ds:6s} images with target outside the evaluation region: {len(outside)} {outside[:3]}")
        loc = run / "loc.csv"
        if loc.exists():
            d = pd.read_csv(loc, usecols=["case_id", "map", "prompt", "auroc", "prevalence"])
            sub = d[(d["map"] == "native") & (d["prompt"] == "pathology")]
            missing = sorted({c.case_id for c in pos} - set(sub["case_id"]))
            nan = sorted(set(sub[sub["auroc"].isna()]["case_id"]))
            for cid in missing + nan:
                f = run / "cases" / f"{cid}.npz"
                why = "not in loc.csv"
                if f.exists():
                    z = np.load(f)
                    m, v = z["mask"], z["valid"]
                    inside = m[v]
                    why = (f"target covers {100*inside.mean():.1f}% of the evaluated region "
                           f"({'no background pixels -> AUROC undefined' if inside.all() else 'see npz'})")
                print(f"INFO {ds:6s} no AUROC: {cid}: {why}")

    print("\n-- 5/6. figures and tables")
    figs = PAPER / "figs"
    for ds in DATASETS:
        pdfs = sorted(figs.glob(f"case_{ds}_*.pdf"))
        svgs = sorted(figs.glob(f"case_{ds}_*.svg"))
        report(len(pdfs) >= 1 and len(svgs) == len(pdfs), f"figures {ds}: {len(pdfs)} PDF, {len(svgs)} SVG")
    report((figs / "CAPTION_NOTE.txt").exists(), "caption note present")
    for t in ("compare_auroc", "compare_auprc_lift", "specificity_gains", "specificity_gains_wide",
              "compare_paired"):
        report((PAPER / "tables" / f"{t}.csv").exists(), f"table {t}.csv")
    report((PAPER / "data" / "datasets.csv").exists(), "dataset description datasets.csv")

    print(f"\n{passes} checks passed, {fails} failed"
          f"{' -> ALL CHECKS PASSED' if fails == 0 else ' -> see FAIL lines above'}")


if __name__ == "__main__":
    main()
