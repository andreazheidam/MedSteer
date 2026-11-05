"""Fine-tune SteerViT's steering modules on one dataset (follow-up Sec. 2).

  # target-only baseline
  python scripts/finetune.py --dataset kvasir --loss target --run-name kvasir_target
  # specificity-aware, hard semantic-control mining over all four controls
  python scripts/finetune.py --dataset kvasir --loss spec --spec-mode hard --run-name kvasir_spec_hard

Uses the manifest split: `train` to optimise, `val` to select the epoch (and, across runs,
lambda_spec / margin), `test` is never touched here. Lesion images only.

Writes <out>/<run-name>/
  ckpt_best.pt    trainable parameters of the selected epoch (+ args)   -> scripts/extract_semdiff.py --weights
  ckpt_last.pt
  log.jsonl       one line per epoch: train losses, validation losses and monitoring metrics
  args.json
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
from medsteer.data import content_region, load_image, load_mask, mask_to_grid, read_manifest  # noqa: E402
from medsteer.finetune import (cosine_warmup, flip_batch, loc_loss, q_score, spec_loss,  # noqa: E402
                               to_q_space)

TARGET = "pathology"
CONTROLS = ["anatomy", "counterfactual", "wrong_medical", "irrelevant"]


def get_model(args, cfg):
    if args.fake_model:
        from tests.fake_model import FakeTrainableSteerModel
        return FakeTrainableSteerModel()
    from medsteer.finetune import TrainableSteer
    m = cfg["model"]
    return TrainableSteer(m["checkpoint"], device=args.device, dtype=m.get("dtype", "float32"))


def load_cases(cases, R, g):
    imgs = np.stack([load_image(c.image_path, R) for c in cases]).astype(np.float32)
    frac = np.stack([mask_to_grid(load_mask(c.mask_paths, R, c.image_path), g).reshape(g, g) for c in cases])
    vfrac = np.stack([mask_to_grid(content_region(c.image_path, R), g).reshape(g, g) for c in cases])
    return imgs, frac.astype(np.float32), vfrac.astype(np.float32)


def patch_auroc(s: np.ndarray, frac: np.ndarray, vfrac: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    keep = vfrac >= 0.5
    y = frac[keep] >= 0.5
    if y.all() or not y.any():
        return np.nan
    return float(roc_auc_score(y, s[keep]))


@torch.no_grad()
def validate(model, data, groups, args):
    """Validation losses (paraphrase 1, as in training with --train-paraphrase first) and
    monitoring metrics: patch-level target AUROC, G_c (paraphrase 1) and the ensemble q-margin
    q(s+) - q(s_c) per control, both in the q-space used for training."""
    imgs, frac, vfrac = data
    bs = args.batch_size
    first = {c: groups[c][0] for c in groups}
    n_loc = n_spec = 0.0
    margins = {c: [] for c in args.controls}
    auc = {c: [] for c in [TARGET] + args.controls}
    for b in range(0, len(imgs), bs):
        x = model.to_tensor(imgs[b: b + bs])
        fr = torch.from_numpy(frac[b: b + bs].reshape(len(x), -1)).to(model.device)
        w = torch.from_numpy(vfrac[b: b + bs].reshape(len(x), -1)).to(model.device)
        maps = {c: [model.logits_grad(x, t) for t in groups[c]] for c in [TARGET] + args.controls}
        s_plus = maps[TARGET][0]
        n_loc += float(loc_loss(s_plus, fr, w)) * len(x)
        qp = q_score(to_q_space(s_plus, w, args.q_space), fr, w)
        qc = torch.stack([q_score(to_q_space(maps[c][0], w, args.q_space), fr, w) for c in args.controls], 1)
        n_spec += float(spec_loss(qp, qc, args.margin, args.spec_mode)[0]) * len(x)
        q_ens = {c: torch.stack([q_score(to_q_space(s, w, args.q_space), fr, w) for s in maps[c]]).mean(0)
                 for c in maps}
        for c in args.controls:
            margins[c] += (q_ens[TARGET] - q_ens[c]).cpu().tolist()
        for c in auc:
            s1 = maps[c][0].cpu().numpy()
            for i in range(len(x)):
                auc[c].append(patch_auroc(s1[i], frac[b + i].reshape(-1), vfrac[b + i].reshape(-1)))
    n = len(imgs)
    out = {"val_loss_loc": n_loc / n, "val_loss_spec": n_spec / n}
    out["val_loss"] = out["val_loss_loc"] + (args.lambda_spec * out["val_loss_spec"] if args.loss == "spec" else 0)
    out["val_auroc_target"] = float(np.nanmedian(auc[TARGET]))
    for c in args.controls:
        out[f"val_G_{c}"] = float(np.nanmedian(np.array(auc[TARGET]) - np.array(auc[c])))
        out[f"val_dq_{c}"] = float(np.median(margins[c]))
    out["val_dq_min"] = min(out[f"val_dq_{c}"] for c in args.controls)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompts_semdiff.yaml")
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs_ft")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--loss", choices=["target", "spec"], default="spec")
    ap.add_argument("--spec-mode", choices=["hard", "mean"], default="hard")
    ap.add_argument("--controls", nargs="+", default=CONTROLS, choices=CONTROLS,
                    help="controls in the specificity loss (one entry = single-control ablation)")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--lambda-spec", type=float, default=1.0)
    ap.add_argument("--q-space", choices=["z", "prob", "logit"], default="z")
    ap.add_argument("--train-paraphrase", choices=["random", "first"], default="random")
    ap.add_argument("--train-head", action="store_true", help="also train the linear localisation head")
    ap.add_argument("--lambda-pres", type=float, default=0.0, help="preservation term (only with --train-head)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--select", choices=["val_loss", "val_loss_loc"], default="val_loss",
                    help="epoch selection criterion on the validation split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap train/val images (tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--fake-model", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg, dsets = load_config(args.config)
    prompts = dsets[args.dataset].prompts
    groups = {}
    for p in prompts:                                     # paraphrase 1 (primary) first
        groups.setdefault(p.category, []).append(p)
    groups = {c: [p.text for p in sorted(v, key=lambda p: not p.primary)] for c, v in groups.items()}

    out = Path(args.out) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    model = get_model(args, cfg)
    names = model.select_trainable(train_head=args.train_head)
    n_train = sum(p.numel() for n, p in model.model.named_parameters() if p.requires_grad)
    print(f"trainable: {len(names)} tensors, {n_train:,} parameters "
          f"({'steering + head' if args.train_head else 'steering only'})", flush=True)

    lesion = lambda cs: [c for c in cs if c.mask_paths and not c.is_healthy]  # noqa: E731
    tr = lesion(read_manifest(args.manifest, datasets=[args.dataset], splits=["train"]))
    va = lesion(read_manifest(args.manifest, datasets=[args.dataset], splits=["val"]))
    if args.limit:
        tr, va = tr[: args.limit], va[: args.limit]
    if not tr or not va:
        raise SystemExit(f"{args.dataset}: {len(tr)} train / {len(va)} val lesion images - check the manifest splits")
    R, g = model.resolution, model.grid
    t0 = time.time()
    train, val = load_cases(tr, R, g), load_cases(va, R, g)
    print(f"{args.dataset}: {len(tr)} train, {len(va)} val lesion images (loaded in {time.time() - t0:.0f}s)", flush=True)

    if not args.no_check:
        err = model.check_forward(model.to_tensor(val[0][:2]), groups[TARGET][0])
        print(f"check: cached forward vs official forward, max rel diff {err:.1e}", flush=True)
        if err > 1e-3:
            raise SystemExit("cached text forward differs from SteerViT's forward - stop")

    head_pre = None
    if args.train_head and args.lambda_pres > 0:
        head_pre = {k: v.detach().clone() for k, v in model.model.lin_seg_head.state_dict().items()}

    opt = torch.optim.AdamW(model.param_groups(), lr=args.lr)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch_size))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, total, int(args.warmup_frac * total)))
    meta = vars(args) | {"config_hash": config_hash(args.config), "trainable": names,
                         "n_trainable": n_train, "n_train": len(tr), "n_val": len(va),
                         "train_cases": [c.case_id for c in tr], "val_cases": [c.case_id for c in va]}
    (out / "args.json").write_text(json.dumps(meta, indent=2))

    def save(path, epoch, metrics):
        torch.save({"state": model.trainable_state(), "epoch": epoch, "metrics": metrics, "args": meta}, path)

    v0 = validate(model, val, groups, args)
    print("epoch 0 (pretrained): " + "  ".join(f"{k}={v:.4f}" for k, v in v0.items()), flush=True)
    log = open(out / "log.jsonl", "w")
    log.write(json.dumps({"epoch": 0} | v0) + "\n")
    best, best_epoch = v0[args.select], 0
    save(out / "ckpt_best.pt", 0, v0)

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(tr))
        tl = ts = tp = 0.0
        hard_counts = np.zeros(len(args.controls), int)
        t0 = time.time()
        for b in range(0, len(order), args.batch_size):
            idx = order[b: b + args.batch_size]
            imgs, (fr_np, vf_np) = flip_batch(train[0][idx], [train[1][idx], train[2][idx]], rng)
            x = model.to_tensor(imgs)
            fr = torch.from_numpy(fr_np.reshape(len(idx), -1).copy()).to(model.device)
            w = torch.from_numpy(vf_np.reshape(len(idx), -1).copy()).to(model.device)
            pick = (lambda c: groups[c][0]) if args.train_paraphrase == "first" else \
                   (lambda c: groups[c][rng.integers(len(groups[c]))])
            s_plus = model.logits_grad(x, pick(TARGET))
            loss = l_loc = loc_loss(s_plus, fr, w)
            l_spec = torch.zeros((), device=model.device)
            if args.loss == "spec":
                texts = {c: pick(c) for c in args.controls}
                qp = q_score(to_q_space(s_plus, w, args.q_space), fr, w)
                if args.spec_mode == "hard":
                    with torch.no_grad():   # find c* per sample, then recompute only those with gradients
                        q_ng = torch.stack([q_score(to_q_space(model.logits_grad(x, texts[c]), w, args.q_space), fr, w)
                                            for c in args.controls], 1)
                    hard = q_ng.argmax(1)
                    qc = q_ng.clone()
                    for k in hard.unique().tolist():
                        sel = hard == k
                        qk = q_score(to_q_space(model.logits_grad(x[sel], texts[args.controls[k]]), w[sel],
                                                args.q_space), fr[sel], w[sel])
                        qc[sel, k] = qk
                    l_spec, hard = spec_loss(qp, qc, args.margin, "hard")
                else:
                    qc = torch.stack([q_score(to_q_space(model.logits_grad(x, texts[c]), w, args.q_space), fr, w)
                                      for c in args.controls], 1)
                    l_spec, hard = spec_loss(qp, qc, args.margin, "mean")
                hard_counts += np.bincount(hard.cpu().numpy(), minlength=len(args.controls))
                loss = loss + args.lambda_spec * l_spec
            l_pres = torch.zeros((), device=model.device)
            if head_pre is not None:
                with torch.no_grad():
                    f0 = model.model.vision_model(x, None)[:, model.n_prefix:]
                pre = torch.nn.functional.linear(f0, head_pre["weight"], head_pre.get("bias"))[..., 0]
                l_pres = ((model.model.lin_seg_head(f0)[..., 0] - pre) ** 2).mean()
                loss = loss + args.lambda_pres * l_pres
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            tl += l_loc.item() * len(idx)
            ts += l_spec.item() * len(idx)
            tp += l_pres.item() * len(idx)
        v = validate(model, val, groups, args)
        rec = {"epoch": epoch, "train_loss_loc": tl / len(tr), "train_loss_spec": ts / len(tr),
               "train_loss_pres": tp / len(tr), "lr": sched.get_last_lr()[0], "sec": time.time() - t0,
               "hard_control_counts": dict(zip(args.controls, hard_counts.tolist()))} | v
        log.write(json.dumps(rec) + "\n")
        log.flush()
        improved = v[args.select] < best
        if improved:
            best, best_epoch = v[args.select], epoch
            save(out / "ckpt_best.pt", epoch, rec)
        print(f"epoch {epoch}/{args.epochs} ({rec['sec']:.0f}s)  train loc {rec['train_loss_loc']:.4f} "
              f"spec {rec['train_loss_spec']:.4f} | val loss {v['val_loss']:.4f} auroc {v['val_auroc_target']:.4f} "
              f"dq_min {v['val_dq_min']:.3f}" + ("  *" if improved else ""), flush=True)
    save(out / "ckpt_last.pt", args.epochs, rec)
    log.close()
    print(f"best epoch {best_epoch} ({args.select} = {best:.4f}) -> {out / 'ckpt_best.pt'}")


if __name__ == "__main__":
    main()
