"""Run SteerViT once per (case, prompt) and store everything the evaluation needs.

Per case, one compressed .npz with patch-level maps (no full feature tensors by default;
those would be ~1 MB per case and prompt):
  prompt_ids                        (P,)
  csr_abs, csr_rel, csr_cos         (P, N)   steering-effect maps vs the unprompted F(x)
  spec                              (P, N)   ||F(x,p) - F(x,reference)||  (reference row = 0)
  native                            (P, N)   native SteerViT head logits = w.F(x,p) + b
  native0                           (N,)     the same head on the unprompted F(x)
                                             (native - native0 = w.dF: the head's view of the residual)
  attn, attn0                       (P, N), (N,)  CLS attention maps (only with --attention)
  dF_lesion, dF_outside, dF_mean, dF_cls   (P, D)  pooled residual vectors (for directions)
  f0_norm, salience                 (N,)     prompt-free baseline maps
  mask (R, R) bool, frac (g, g)     ground truth at model resolution / per-patch fraction
  valid (R, R) bool, valid_frac (g, g)  evaluation region = unpadded image (data.content_region;
                                             geometry only, independent of GT and intensities)

DSE check (feedback 2026-10-28, pt 6): for every batch and prompt, w^T dF is computed directly
from the released head weight and compared with native - native0 (= h(F_p) - h(F_0), logits).
The maximum absolute and relative difference is printed and stored in meta.json.

Usage:
  python scripts/extract.py --config configs/prompts.yaml --manifest data/manifest.csv \
      --dataset busi --out runs/steerdinov2
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
from medsteer import baselines, residuals  # noqa: E402
from medsteer.config import config_hash, load_config  # noqa: E402
from medsteer.data import VALID_REGION, content_region, load_image, load_mask, mask_to_grid, read_manifest  # noqa: E402


def get_model(args, cfg):
    if args.fake_model:  # tests only
        from tests.fake_model import FakeSteerModel
        return FakeSteerModel()
    from medsteer.model import SteerModel
    m = cfg["model"]
    return SteerModel(m["checkpoint"], device=args.device, dtype=m.get("dtype", "float32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompts.yaml")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="*", default=None, help="default: all splits")
    ap.add_argument("--out", default="runs/steerdinov2")
    ap.add_argument("--gate", type=float, default=None, help="override config gate factor")
    ap.add_argument("--limit", type=int, default=None, help="first N cases (pilot runs)")
    ap.add_argument("--attention", action="store_true", help="also store CLS attention maps")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake-model", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    raw, dsets = load_config(args.config)
    dcfg = dsets[args.dataset]
    gate = raw["model"].get("gate", 1.0) if args.gate is None else args.gate
    bs = raw["model"].get("batch_size", 32)
    cases = read_manifest(args.manifest, datasets=[args.dataset], splits=args.splits)
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out) / f"gate{gate:g}" / args.dataset
    (out / "cases").mkdir(parents=True, exist_ok=True)

    model = get_model(args, raw)
    model.set_gate(gate)
    R, g = model.resolution, model.grid
    prompts = dcfg.prompts
    pids = [p.id for p in prompts]
    ref_idx = pids.index(dcfg.reference)

    meta = {"dataset": args.dataset, "n_cases": len(cases), "gate": gate,
            "resolution": R, "grid": g, "backbone": getattr(model, "backbone_name", "?"),
            "checkpoint": raw["model"]["checkpoint"], "config_hash": config_hash(args.config),
            "prompts": [p.__dict__ for p in prompts], "reference": dcfg.reference,
            "attention": args.attention, "valid_region": VALID_REGION}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    head = getattr(getattr(model, "model", None), "lin_seg_head", None)
    w_head = head.weight.detach().float().reshape(-1) if head is not None else None
    dse_abs = dse_rel = 0.0

    todo = [c for c in cases if not (out / "cases" / f"{c.case_id}.npz").exists()]
    print(f"{args.dataset}: {len(cases)} cases, {len(todo)} to extract, grid {g}x{g}, "
          f"{len(prompts)} prompts, gate {gate}", flush=True)
    t0 = time.time()
    for b in range(0, len(todo), bs):
        batch = todo[b: b + bs]
        imgs = np.stack([load_image(c.image_path, R) for c in batch])
        masks = np.stack([load_mask(c.mask_paths, R, c.image_path) for c in batch])
        fracs = np.stack([mask_to_grid(m, g) for m in masks])
        valids = np.stack([content_region(c.image_path, R) for c in batch])
        vfracs = np.stack([mask_to_grid(v, g) for v in valids])
        x = model.to_tensor(imgs)
        w_les = torch.from_numpy(fracs.reshape(len(batch), -1)).to(x.device)

        tok0 = model.tokens(x, None)
        cls0, F0 = model.split(tok0)
        nat0_t = model.native_logits(tok0)
        nat0 = nat0_t.cpu().numpy()   # head on the UNPROMPTED features
        attn0 = model.attention_map(x, None).reshape(len(batch), -1) if args.attention else None
        # reference prompt first so specificity maps can be formed in one pass
        _, Fref = model.split(model.tokens(x, prompts[ref_idx].text))

        P, B, N, D = len(prompts), len(batch), F0.shape[1], F0.shape[2]
        res = {k: np.zeros((B, P, N), np.float32)
               for k in ("csr_abs", "csr_rel", "csr_cos", "spec", "native")}
        if args.attention:
            res["attn"] = np.zeros((B, P, N), np.float32)
        vec = {k: np.zeros((B, P, D), np.float16)
               for k in ("dF_lesion", "dF_outside", "dF_mean", "dF_cls")}
        for j, p in enumerate(prompts):
            tok = model.tokens(x, p.text)
            clsp, Fp = model.split(tok)
            dF = Fp - F0
            res["csr_abs"][:, j] = residuals.csr_abs(Fp, F0).cpu().numpy()
            res["csr_rel"][:, j] = residuals.csr_rel(Fp, F0).cpu().numpy()
            res["csr_cos"][:, j] = residuals.csr_cos(Fp, F0).cpu().numpy()
            res["spec"][:, j] = residuals.pair_map(Fp, Fref).cpu().numpy()
            nat_t = model.native_logits(tok)
            res["native"][:, j] = nat_t.cpu().numpy()
            if w_head is not None:   # DSE identity: h(F_p) - h(F_0) == w^T (F_p - F_0)
                diff = nat_t - nat0_t
                wdf = dF @ w_head.to(dF.device)
                err = (diff - wdf).abs().max().item()
                dse_abs = max(dse_abs, err)
                dse_rel = max(dse_rel, err / (diff.abs().max().item() + 1e-12))
            if args.attention:
                res["attn"][:, j] = model.attention_map(x, p.text).reshape(B, -1).cpu().numpy()
            vec["dF_lesion"][:, j] = residuals.pooled_residual(dF, w_les).cpu().numpy()
            vec["dF_outside"][:, j] = residuals.pooled_residual(dF, 1 - w_les).cpu().numpy()
            vec["dF_mean"][:, j] = dF.mean(dim=1).cpu().numpy()
            if clsp is not None:
                vec["dF_cls"][:, j] = (clsp - cls0).cpu().numpy()

        F0n = F0.cpu().numpy()
        for i, c in enumerate(batch):
            arrs = {k: v[i] for k, v in res.items()} | {k: v[i] for k, v in vec.items()}
            arrs["native0"] = nat0[i]
            if attn0 is not None:
                arrs["attn0"] = attn0[i].cpu().numpy()
            np.savez_compressed(
                out / "cases" / f"{c.case_id}.npz", prompt_ids=np.array(pids),
                f0_norm=baselines.feature_norm(F0n[i]).astype(np.float32),
                salience=baselines.salience(F0n[i]).astype(np.float32),
                mask=masks[i], frac=fracs[i], valid=valids[i], valid_frac=vfracs[i], **arrs)
        done = b + len(batch)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)}  ({rate:.1f} cases/s, eta {(len(todo) - done) / rate / 60:.1f} min)"
              f"  DSE check max|h(Fp)-h(F0) - w^T dF| = {dse_abs:.2e} (rel {dse_rel:.2e})", flush=True)
    if w_head is not None and todo:
        meta["dse_check"] = {"max_abs_err": dse_abs, "max_rel_err": dse_rel,
                             "what": "max over images, patches, prompts of |(native - native0) - w^T dF|"}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done:", out)


if __name__ == "__main__":
    main()
