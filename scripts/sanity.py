"""Phase-0 sanity checks. Run before any evaluation; stop if a check fails.

  1. unprompted SteerViT (texts=None) == vanilla timm backbone          (F(x) is clean)
  2. gate 0 with a prompt == unprompted                                 (gate knob works)
  3. same prompt twice gives identical features                        (determinism)
  4. natural-image positive control: CSR map lands on the named object (pipeline works)
  5. on a few medical cases (diagnostics, not pass/fail): residual inside the image vs in the
     padding, correlation with feature norm, how similar pathology and control maps are, a
     preview AUROC per map, and an overlay figure (mask grid + image area + maps)

Usage:
  python scripts/sanity.py --config configs/prompts.yaml --manifest data/manifest.csv \
      --dataset busi --natural-image dog.jpg --natural-prompt "the dog" --out sanity/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer import residuals  # noqa: E402
from medsteer.config import load_config  # noqa: E402
from medsteer.data import load_image, load_mask, mask_to_grid, read_manifest, valid_region  # noqa: E402
from medsteer.metrics import localisation, spearman, upsample  # noqa: E402
from medsteer.plotting import SEQ, plt  # noqa: E402


def check(name, ok, detail, report):
    report[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompts.yaml")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--dataset", default="busi")
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--natural-image", default=None, help="e.g. a photo with a dog")
    ap.add_argument("--natural-prompt", default="the dog")
    ap.add_argument("--out", default="sanity")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs diff counted as identical")
    ap.add_argument("--fake-model", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw, dsets = load_config(args.config)
    dcfg = dsets[args.dataset]
    report = {}

    if args.fake_model:
        from tests.fake_model import FakeSteerModel
        model = FakeSteerModel()
    else:
        from medsteer.model import SteerModel
        model = SteerModel(raw["model"]["checkpoint"], device=args.device, dtype="float32")
    R, g = model.resolution, model.grid
    print(f"device: {model.device}  (use a GPU runtime for extraction)", flush=True)
    cases = [c for c in read_manifest(args.manifest, datasets=[args.dataset])
             if c.mask_paths and not c.is_healthy][: args.n_cases]
    assert cases, "no lesion cases with masks found in the manifest"
    imgs = np.stack([load_image(c.image_path, R) for c in cases])
    masks = np.stack([load_mask(c.mask_paths, R, c.image_path) for c in cases])
    x = model.to_tensor(imgs)
    p_path = dcfg.prompt("pathology").text

    # 1. unprompted == vanilla backbone
    if not args.fake_model:
        import timm
        ref = timm.create_model(model.backbone_name, pretrained=True, img_size=R).to(model.device).eval()
        with torch.no_grad():
            van = ref.forward_features(x).float()
        mine = model.tokens(x, None)
        diff = (van - mine).abs().max().item()
        check("unprompted_equals_vanilla", diff < args.tol,
              {"max_abs_diff": diff, "feature_scale": van.abs().mean().item()}, report)
        del ref

    # 2. gate 0 removes the prompt
    model.set_gate(0.0)
    g0 = model.tokens(x, p_path)
    model.set_gate(1.0)
    diff = (g0 - model.tokens(x, None)).abs().max().item()
    check("gate0_equals_unprompted", diff < args.tol, {"max_abs_diff": diff}, report)

    # 3. determinism
    a, b = model.tokens(x, p_path), model.tokens(x, p_path)
    diff = (a - b).abs().max().item()
    check("deterministic", diff < args.tol, {"max_abs_diff": diff}, report)

    # 4. natural-image positive control
    if args.natural_image:
        nimg = load_image(args.natural_image, R)[None]
        nx = model.to_tensor(nimg)
        _, F0 = model.split(model.tokens(nx, None))
        tok = model.tokens(nx, args.natural_prompt)
        _, Fp = model.split(tok)
        csr = residuals.csr_abs(Fp, F0)[0].cpu().numpy()
        nat = model.native_logits(tok)[0].cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(7, 2.4))
        ax[0].imshow(nimg[0]); ax[0].set_title("image")
        ax[1].imshow(upsample(csr, R), cmap=SEQ); ax[1].set_title(f"CSR: {args.natural_prompt}")
        ax[2].imshow(upsample(nat, R), cmap=SEQ); ax[2].set_title("native head")
        for a_ in ax:
            a_.axis("off")
        fig.savefig(out / "natural_control.png")
        plt.close(fig)
        report["natural_control"] = {"pass": None, "note": "inspect natural_control.png: "
                                     "the CSR map should highlight the named object",
                                     "rho_csr_native": spearman(csr, nat)}
        print("[LOOK] natural_control.png", flush=True)

    # 5. medical cases: magnitudes, padding, artefact correlation, prompt specificity, overlay
    valids = np.stack([valid_region(im) for im in imgs])
    vgrid = np.stack([mask_to_grid(v, g) for v in valids]) >= 0.5          # content patches
    fracs = np.stack([mask_to_grid(m, g) for m in masks])
    tok0 = model.tokens(x, None)
    cls0, F0 = model.split(tok0)
    nat0 = model.native_logits(tok0).cpu().numpy()
    f0n = F0.norm(dim=-1).cpu().numpy()
    ref_text = dcfg.prompt(dcfg.reference).text
    _, Fref = model.split(model.tokens(x, ref_text))
    mags, maps, natives = {}, {}, {}
    for p in [q for q in dcfg.prompts if q.primary]:
        tok = model.tokens(x, p.text)
        _, Fp = model.split(tok)
        maps[p.id] = residuals.csr_abs(Fp, F0).cpu().numpy()
        natives[p.id] = model.native_logits(tok).cpu().numpy()
        mags[p.id] = float(np.mean([maps[p.id][i][vgrid[i].ravel()].mean() for i in range(len(cases))]))
    _, Fpath = model.split(model.tokens(x, p_path))
    spec = residuals.pair_map(Fpath, Fref).cpu().numpy()
    report["residual_magnitude_mean_in_image"] = mags

    pad_ratio = [maps["pathology"][i][~vgrid[i].ravel()].mean() / maps["pathology"][i][vgrid[i].ravel()].mean()
                 for i in range(len(cases)) if (~vgrid[i]).any()]
    rho_norm = [spearman(maps["pathology"][i][vgrid[i].ravel()], f0n[i][vgrid[i].ravel()]) for i in range(len(cases))]
    rho_ref = [spearman(maps["pathology"][i][vgrid[i].ravel()], maps[dcfg.reference][i][vgrid[i].ravel()])
               for i in range(len(cases))]
    info = {"padding_over_image_residual": float(np.median(pad_ratio)) if pad_ratio else None,
            "rho_csr_featnorm_in_image": float(np.nanmedian(rho_norm)),
            "rho_pathology_vs_reference_csr": float(np.nanmedian(rho_ref)),
            "note": "rho_pathology_vs_reference near 1 = the prompt barely changes WHERE steering acts; "
                    "then the specificity map (pathology vs reference) is the informative one"}
    report["diagnostics"] = info
    print("[INFO]", info, flush=True)
    preview = {}
    for name, arr in (("csr_pathology", maps["pathology"]), ("csr_" + dcfg.reference, maps[dcfg.reference]),
                      ("spec_pathology_vs_ref", spec), ("native_pathology", natives["pathology"]),
                      ("native_no_prompt", nat0), ("native_delta_pathology", natives["pathology"] - nat0),
                      ("feature_norm", f0n)):
        preview[name] = float(np.median([localisation(arr[i], masks[i], fracs[i], with_dice=False,
                                                      valid=valids[i], valid_frac=vgrid[i].astype(float))["auroc"]
                                         for i in range(len(cases))]))
    report["preview_median_pixel_auroc"] = preview
    print("[INFO] preview median pixel AUROC on these cases:", preview, flush=True)

    cols = [("native\n(pathology)", "native"), ("CSR\npathology", "pathology"), ("CSR\nanatomy", "anatomy"),
            (f"CSR\n{dcfg.reference}", dcfg.reference), ("specificity\npathology vs ref", "spec")]
    fig, axes = plt.subplots(len(cases), 2 + len(cols), figsize=(1.5 * (2 + len(cols)), 1.5 * len(cases)))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(cases):
        vi = vgrid[i].ravel()
        vmax = max(np.quantile(maps[s][i][vi], 0.99) for s in ("pathology", "anatomy", dcfg.reference))
        vmin = min(np.quantile(maps[s][i][vi], 0.01) for s in ("pathology", "anatomy", dcfg.reference))
        axes[i, 0].imshow(imgs[i])
        axes[i, 1].imshow(imgs[i])
        axes[i, 1].imshow(upsample(fracs[i], R), alpha=0.5, cmap="Blues")
        axes[i, 1].contour(masks[i], levels=[0.5], colors="#eb6834", linewidths=0.6)
        axes[i, 1].contour(valids[i], levels=[0.5], colors="#1baf7a", linewidths=0.8)
        hide = lambda a: np.ma.masked_where(~valids[i], upsample(a, R))  # grey = not evaluated
        for k, (title, key) in enumerate(cols):
            ax = axes[i, 2 + k]
            ax.set_facecolor("#d9d9d9")
            if key == "native":
                arr = natives["pathology"][i]
                ax.imshow(hide(arr), cmap=SEQ, vmin=np.quantile(arr[vi], 0.01), vmax=np.quantile(arr[vi], 0.99))
            elif key == "spec":
                arr = spec[i]
                ax.imshow(hide(arr), cmap=SEQ, vmin=np.quantile(arr[vi], 0.01), vmax=np.quantile(arr[vi], 0.99))
            else:  # CSR columns share one scale per case (comparable magnitudes)
                ax.imshow(hide(maps[key][i]), cmap=SEQ, vmin=vmin, vmax=vmax)
            ax.contour(masks[i], levels=[0.5], colors="white", linewidths=0.5)
            if i == 0:
                ax.set_title(title, fontsize=6)
    axes[0, 0].set_title("image", fontsize=6)
    axes[0, 1].set_title("mask (orange)\nimage area (green)", fontsize=6)
    for a_ in axes.ravel():
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
    fig.savefig(out / f"{args.dataset}_alignment_and_csr.png", dpi=150)
    plt.close(fig)
    print(f"[LOOK] {args.dataset}_alignment_and_csr.png", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    failed = [k for k, v in report.items() if isinstance(v, dict) and v.get("pass") is False]
    print("ALL AUTOMATIC CHECKS PASSED" if not failed else f"FAILED: {failed}")


if __name__ == "__main__":
    main()
