"""SteerViT extraction for training-free semantic differential steering (follow-up Sec. 1).

Per case one .npz with
  prompt_ids, categories          (P,)    all prompts of configs/prompts_semdiff.yaml (3 per category)
  native                          (P, N)  SteerViT head logits h(F(x,p)) for every prompt
  native0                         (N,)    head on the unprompted F(x)
  fd_keys                         (K,)    control categories + "mean"
  fd_single                       (K, N)  ||F(x,p+) - F(x,p_c)||_2          (paraphrase 1 only)
  fd_ens                          (K, N)  ||Fbar_+(x) - Fbar_c(x)||_2       (Fbar = mean over paraphrases)
                                          "mean" row: ||Fbar_+ - (1/|C|) sum_c Fbar_c||_2
  mask, frac, valid, valid_frac   same geometry as scripts/extract.py

Head-projected contrasts need no extra storage: since h is linear,
  D^H_c = w^T (Fbar_+ - Fbar_c) = mean_j h(F(x,p+_j)) - mean_j h(F(x,p_c_j)),
i.e. a difference of stored `native` rows. This identity is checked numerically for every batch
and stored in meta.json (`head_identity_check`).

The output map files are compatible with scripts/semdiff_eval.py, which is also used for the
map-level runs of CLIP / BiomedCLIP / SAM3 / MedSAM3 (their existing extraction scripts already
store `native` for every prompt of the config).

Usage:
  python scripts/extract_semdiff.py --config configs/prompts_semdiff.yaml \
      --manifest data/manifest_final.csv --dataset kvasir --out runs_semdiff
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.config import config_hash, load_config  # noqa: E402
from medsteer.data import VALID_REGION, content_region, load_image, load_mask, mask_to_grid, read_manifest  # noqa: E402

TARGET = "pathology"
CONTROLS = ["anatomy", "counterfactual", "wrong_medical", "irrelevant"]


def get_model(args, cfg):
    if args.fake_model:  # tests only
        from tests.fake_model import FakeSteerModel, FakeTrainableSteerModel
        model = FakeTrainableSteerModel() if args.weights else FakeSteerModel()
    else:
        from medsteer.model import SteerModel
        m = cfg["model"]
        model = SteerModel(m["checkpoint"], device=args.device, dtype=m.get("dtype", "float32"))
    if args.weights:                       # fine-tuned steering modules (scripts/finetune.py)
        from medsteer.finetune import load_trainable
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        n = load_trainable(model.model, ck["state"])
        print(f"loaded {n} fine-tuned tensors from {args.weights} (epoch {ck.get('epoch')})", flush=True)
    return model


def head_weight(model) -> torch.Tensor:
    head = getattr(getattr(model, "model", None), "lin_seg_head", None)
    if head is not None:
        return head.weight.detach().float().reshape(-1)
    return model.head.float().reshape(-1)          # tests/fake_model.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompts_semdiff.yaml")
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="*", default=None, help="default: all splits in the manifest")
    ap.add_argument("--out", default="runs_semdiff")
    ap.add_argument("--name", default="steervit", help="run name: <out>/<name>/gate1/<dataset>")
    ap.add_argument("--weights", default=None, help="fine-tuned checkpoint from scripts/finetune.py")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-healthy", dest="lesion_only", action="store_false",
                    help="also extract images without a lesion mask (default: non-empty masks only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake-model", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    raw, dsets = load_config(args.config)
    dcfg = dsets[args.dataset]
    prompts = dcfg.prompts
    pids = [p.id for p in prompts]
    cats = [p.category for p in prompts]
    groups = {c: [j for j, p in enumerate(prompts) if p.category == c] for c in [TARGET] + CONTROLS}
    prim = {c: next(j for j in idx if prompts[j].primary) for c, idx in groups.items()}
    for c, idx in groups.items():
        assert idx, f"no prompts of category {c} for {args.dataset}"

    cases = read_manifest(args.manifest, datasets=[args.dataset], splits=args.splits)
    if args.lesion_only:
        cases = [c for c in cases if c.mask_paths and not c.is_healthy]
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out) / args.name / "gate1" / args.dataset
    (out / "cases").mkdir(parents=True, exist_ok=True)

    model = get_model(args, raw)
    model.set_gate(raw["model"].get("gate", 1.0))
    bs = raw["model"].get("batch_size", 32)
    R, g = model.resolution, model.grid
    w = head_weight(model).to(model.device)
    fd_keys = CONTROLS + ["mean"]

    meta = {"dataset": args.dataset, "model": args.name, "weights": args.weights, "n_cases": len(cases), "resolution": R, "grid": g,
            "backbone": getattr(model, "backbone_name", "?"), "checkpoint": raw["model"]["checkpoint"],
            "config": str(args.config), "config_hash": config_hash(args.config),
            "prompts": [p.__dict__ for p in prompts], "reference": dcfg.reference,
            "null_condition": "no text (texts=None)", "valid_region": VALID_REGION,
            "fd_keys": fd_keys, "lesion_only": args.lesion_only}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = [c for c in cases if not (out / "cases" / f"{c.case_id}.npz").exists()]
    print(f"{args.dataset}: {len(cases)} lesion cases, {len(todo)} to extract, {len(prompts)} prompts", flush=True)
    err_abs = err_rel = 0.0
    t0 = time.time()
    for b in range(0, len(todo), bs):
        batch = todo[b: b + bs]
        imgs = np.stack([load_image(c.image_path, R) for c in batch])
        masks = np.stack([load_mask(c.mask_paths, R, c.image_path) for c in batch])
        fracs = np.stack([mask_to_grid(m, g) for m in masks])
        valids = np.stack([content_region(c.image_path, R) for c in batch])
        vfracs = np.stack([mask_to_grid(v, g) for v in valids])
        x = model.to_tensor(imgs)
        B = len(batch)

        nat0 = model.native_logits(model.tokens(x, None)).cpu().numpy()
        native = np.zeros((B, len(prompts), g * g), np.float32)
        Fsum, Fprim = {}, {}
        for j, p in enumerate(prompts):
            tok = model.tokens(x, p.text)
            _, Fp = model.split(tok)
            native[:, j] = model.native_logits(tok).cpu().numpy()
            c = p.category
            if c not in groups:
                continue
            Fsum[c] = Fp.clone() if c not in Fsum else Fsum[c] + Fp
            if j == prim[c]:
                Fprim[c] = Fp
        Fbar = {c: Fsum[c] / len(groups[c]) for c in Fsum}

        fd_single = np.zeros((B, len(fd_keys), g * g), np.float32)
        fd_ens = np.zeros((B, len(fd_keys), g * g), np.float32)
        for k, c in enumerate(CONTROLS):
            dS = Fprim[TARGET] - Fprim[c]
            dE = Fbar[TARGET] - Fbar[c]
            fd_single[:, k] = dS.norm(dim=-1).cpu().numpy()
            fd_ens[:, k] = dE.norm(dim=-1).cpu().numpy()
            # identity: w^T (Fbar_+ - Fbar_c) == mean native(+) - mean native(c)
            lhs = (dE @ w).cpu().numpy()
            rhs = native[:, groups[TARGET]].mean(1) - native[:, groups[c]].mean(1)
            e = float(np.abs(lhs - rhs).max())
            err_abs = max(err_abs, e)
            err_rel = max(err_rel, e / (float(np.abs(rhs).max()) + 1e-12))
        Fctrl = sum(Fprim[c] for c in CONTROLS) / len(CONTROLS)
        fd_single[:, -1] = (Fprim[TARGET] - Fctrl).norm(dim=-1).cpu().numpy()
        Fctrl = sum(Fbar[c] for c in CONTROLS) / len(CONTROLS)
        fd_ens[:, -1] = (Fbar[TARGET] - Fctrl).norm(dim=-1).cpu().numpy()

        for i, c in enumerate(batch):
            np.savez_compressed(out / "cases" / f"{c.case_id}.npz", prompt_ids=np.array(pids),
                                categories=np.array(cats), native=native[i], native0=nat0[i],
                                fd_keys=np.array(fd_keys), fd_single=fd_single[i], fd_ens=fd_ens[i],
                                mask=masks[i], frac=fracs[i], valid=valids[i], valid_frac=vfracs[i])
        done = b + B
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)} ({rate:.2f} cases/s, eta {(len(todo) - done) / rate / 60:.1f} min)  "
              f"head identity max err {err_abs:.2e} (rel {err_rel:.2e})", flush=True)
    if todo:
        meta["head_identity_check"] = {"max_abs_err": err_abs, "max_rel_err": err_rel,
                                       "what": "max |w^T(Fbar_+ - Fbar_c) - (mean h(F_+) - mean h(F_c))|"}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done:", out)


if __name__ == "__main__":
    main()
