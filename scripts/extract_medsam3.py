"""MedSAM3 (SAM3 + medical LoRA, arXiv:2511.19046) in EXACTLY the format of extract_compare.py's
SAM3 run, so evaluate.py, compare_table.py and paper_fig1.py work on it unchanged.

MedSAM3 is released as a LoRA adapter for Meta's original SAM3 code (not the Hugging Face
Sam3Model), so this script runs inside the MedSAM3 repository (--medsam3-root):
  base SAM3 (facebook/sam3, sam3.pt)  +  LoRA (configs/full_lora_config.yaml)  +  LoRA weights.
Same images, padding, masks, evaluation region, 24x24 grid and prompts as every other model:
  F(x,p)  = text-fused encoder output (encoder_hidden_states, 72x72 -> 24x24 area-pooled)
  F(x)    = the same for the empty-text control ""
  native  = semantic_seg logits for p (pooled to 24x24), native0 = for ""
All prompts of one image go through ONE forward pass as separate queries (shared image backbone);
--check verifies on the first image that this equals running each prompt on its own.

Safety: the MedSAM3 loader uses load_state_dict(strict=False), which silently loads nothing if
the LoRA config does not match the weights. This script loads the file itself and stops unless
every tensor in it matches a LoRA parameter of the model by name and shape.

Usage (from ~/medsteer):
  python scripts/extract_medsam3.py --medsam3-root ~/MedSAM3 --lora medsam3_lora/<file>.pt \
      --dataset kvasir --limit 3 --out runs_test_medsam3          # smoke test
  python scripts/extract_medsam3.py --medsam3-root ~/MedSAM3 --lora ... --dataset brisc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer import baselines, residuals  # noqa: E402
from medsteer.config import config_hash, load_config  # noqa: E402
from medsteer.data import VALID_REGION, content_region, load_image, load_mask, mask_to_grid, read_manifest  # noqa: E402

R_EVAL, GRID, RES = 336, 24, 1008


def to_grid(tok: torch.Tensor, g: int = GRID) -> torch.Tensor:
    squeeze = tok.dim() == 2
    t = tok.unsqueeze(-1) if squeeze else tok
    B, N, D = t.shape
    gm = int(round(N ** 0.5))
    assert gm * gm == N, f"token count {N} is not a square grid"
    t = t.transpose(1, 2).reshape(B, D, gm, gm)
    if gm != g:
        t = F.adaptive_avg_pool2d(t, g)
    t = t.reshape(B, D, g * g).transpose(1, 2)
    return t[..., 0] if squeeze else t


def spatial_tokens(h: torch.Tensor, nq: int) -> torch.Tensor:
    """Encoder memory -> (Q, N, D); accepts (N, Q, D) (sequence-first) or (Q, N, D)."""
    if h.dim() == 3 and h.shape[1] == nq and h.shape[0] != nq:
        h = h.transpose(0, 1)
    assert h.shape[0] == nq, f"unexpected encoder output shape {tuple(h.shape)}"
    N = h.shape[1]
    s = int(N ** 0.5)
    if s * s != N:                      # leading non-spatial tokens
        h = h[:, N - s * s:]
    return h


class MedSAM3:
    def __init__(self, root: Path, lora_cfg: Path, lora_w: Path, device=None):
        sys.path.insert(0, str(root))
        if "decord" not in sys.modules:              # video reader, unused for images
            try:
                import decord  # noqa: F401
            except ImportError:
                stub = types.ModuleType("decord")
                stub.cpu = lambda *a, **k: None
                stub.VideoReader = object
                sys.modules["decord"] = stub
        import yaml
        from sam3.model_builder import build_sam3_image_model
        from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from lora_layers import LoRAConfig, apply_lora_to_model
        self._dp = (Datapoint, SAMImage, FindQueryLoaded, InferenceMetadata)
        self._collate, self._to_dev = collate_fn_api, copy_data_to_device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = build_sam3_image_model(device=self.device.type, compile=False, load_from_HF=True,
                                       bpe_path=str(root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                                       eval_mode=True)
        c = yaml.safe_load(Path(lora_cfg).read_text())["lora"]
        model = apply_lora_to_model(model, LoRAConfig(
            rank=c["rank"], alpha=c["alpha"], dropout=0.0, target_modules=c["target_modules"],
            apply_to_vision_encoder=c["apply_to_vision_encoder"], apply_to_text_encoder=c["apply_to_text_encoder"],
            apply_to_geometry_encoder=c["apply_to_geometry_encoder"], apply_to_detr_encoder=c["apply_to_detr_encoder"],
            apply_to_detr_decoder=c["apply_to_detr_decoder"], apply_to_mask_decoder=c["apply_to_mask_decoder"]))
        # ---- strict LoRA check (the original loader would silently ignore mismatches)
        sd = torch.load(lora_w, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        own = model.state_dict()
        lora_own = {k for k in own if "lora" in k.lower()}
        missing_in_model = [k for k in sd if k not in own]
        bad_shape = [k for k in sd if k in own and own[k].shape != sd[k].shape]
        not_loaded = sorted(lora_own - set(sd))
        print(f"LoRA file: {len(sd)} tensors; model LoRA params: {len(lora_own)}; "
              f"unknown names: {len(missing_in_model)}; shape mismatches: {len(bad_shape)}; "
              f"model LoRA params not in file: {len(not_loaded)}", flush=True)
        if missing_in_model or bad_shape or not_loaded or not sd:
            for name, lst in (("unknown", missing_in_model), ("shape", bad_shape), ("not in file", not_loaded)):
                if lst:
                    print(f"  {name}: {lst[:5]}{' ...' if len(lst) > 5 else ''}")
            raise SystemExit("LoRA weights do not match the LoRA config: refusing to run (would be plain SAM3).")
        model.load_state_dict(sd, strict=False)
        nz = sum(int(sd[k].abs().sum() > 0) for k in sd)
        print(f"LoRA loaded: {len(sd)}/{len(lora_own)} tensors, {nz} non-zero", flush=True)
        self.lora_info = {"file": str(lora_w), "sha256_12": hashlib.sha256(Path(lora_w).read_bytes()).hexdigest()[:12],
                          "config": str(lora_cfg), "rank": c["rank"], "alpha": c["alpha"],
                          "tensors_loaded": len(sd), "nonzero_tensors": nz}
        self.model = model.to(self.device).eval()
        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=RES, max_size=RES, square=True, consistent_transform=False),
            ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    @torch.no_grad()
    def run(self, img01: np.ndarray, texts: list[str]):
        """img01: (RES, RES, 3) float in [0,1], already padded/resized like every other model.
        -> F (Q, 576, D) text-fused encoder features, nat (Q, 576) semantic_seg logits, pooled."""
        from PIL import Image as PILImage
        Datapoint, SAMImage, FindQueryLoaded, InferenceMetadata = self._dp
        pil = PILImage.fromarray((np.clip(img01, 0, 1) * 255).round().astype(np.uint8))
        w, h = pil.size
        queries = [FindQueryLoaded(query_text=t, image_id=0, object_ids_output=[], is_exhaustive=True,
                                   query_processing_order=0,
                                   inference_metadata=InferenceMetadata(coco_image_id=i, original_image_id=i,
                                                                        original_category_id=1, original_size=[w, h],
                                                                        object_id=0, frame_index=0))
                   for i, t in enumerate(texts)]
        dp = self.transform(Datapoint(find_queries=queries, images=[SAMImage(data=pil, objects=[], size=[h, w])]))
        batch = self._to_dev(self._collate([dp], dict_key="input")["input"], self.device, non_blocking=True)
        out = self.model(batch)[-1]
        if "semantic_seg" not in out:
            raise KeyError(f"no semantic_seg in MedSAM3 output; keys: {sorted(out.keys())}")
        Q = len(texts)
        Fq = to_grid(spatial_tokens(out["encoder_hidden_states"].float(), Q))
        nat = F.adaptive_avg_pool2d(out["semantic_seg"].float(), GRID).reshape(Q, -1)
        return Fq, nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medsam3-root", required=True, help="clone of github.com/Joey-S-Liu/MedSAM3")
    ap.add_argument("--lora", required=True, help="LoRA weights file from lal-Joey/MedSAM3_v1")
    ap.add_argument("--lora-config", default=None, help="default: <root>/configs/full_lora_config.yaml")
    ap.add_argument("--config", default="configs/prompts_final2.yaml")
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs_final2_compare")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    root = Path(a.medsam3_root).expanduser().resolve()
    model = MedSAM3(root, Path(a.lora_config or root / "configs/full_lora_config.yaml"), Path(a.lora).expanduser())
    if model.device.type != "cuda":
        raise SystemExit("no GPU visible to torch - refusing to run on CPU")

    _, dsets = load_config(a.config)
    dcfg = dsets[a.dataset]
    cases = read_manifest(a.manifest, datasets=[a.dataset])
    if a.limit:
        cases = cases[: a.limit]
    out = Path(a.out) / "medsam3" / "gate1" / a.dataset
    (out / "cases").mkdir(parents=True, exist_ok=True)
    prompts = dcfg.prompts
    pids = [p.id for p in prompts]
    ref_idx = pids.index(dcfg.reference)
    texts = [""] + [p.text for p in prompts]            # query 0 = empty-text control
    meta = {"dataset": a.dataset, "n_cases": len(cases), "gate": 1.0, "resolution": R_EVAL, "grid": GRID,
            "model_resolution": RES, "backbone": "MedSAM3 (SAM3 + LoRA)", "checkpoint": "facebook/sam3 + MedSAM3 LoRA",
            "lora": model.lora_info, "config_hash": config_hash(a.config), "prompts": [p.__dict__ for p in prompts],
            "reference": dcfg.reference, "null_prompt": "", "null_condition": "empty-text control",
            "attention": False, "valid_region": VALID_REGION}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = [c for c in cases if not (out / "cases" / f"{c.case_id}.npz").exists()]
    print(f"[medsam3] {a.dataset}: {len(cases)} cases, {len(todo)} to extract, {len(prompts)} prompts", flush=True)
    t0 = time.time()
    for k, c in enumerate(todo):
        mask = load_mask(c.mask_paths, R_EVAL, c.image_path)
        frac = mask_to_grid(mask, GRID)
        valid = content_region(c.image_path, R_EVAL)
        img = load_image(c.image_path, RES)
        Fq, nat = model.run(img, texts)
        if k == 0 and not a.no_check:                    # batched queries == one query at a time?
            j = 1 + pids.index("pathology")
            F1, n1 = model.run(img, [texts[j]])
            dF_, dn_ = (Fq[j] - F1[0]).abs().max().item(), (nat[j] - n1[0]).abs().max().item()
            scale = nat[j].abs().max().item() + 1e-8
            print(f"  check: batched vs single query  max|dF|={dF_:.2e}  max|d native|={dn_:.2e} "
                  f"(rel {dn_ / scale:.1e})", flush=True)
            if dn_ / scale > 1e-2:
                raise SystemExit("batched queries differ from single-query runs - stop")
        F0, nat0 = Fq[0:1], nat[0]
        Fp_all, nat_p = Fq[1:], nat[1:]
        Fref = Fp_all[ref_idx: ref_idx + 1]
        w_les = torch.from_numpy(frac.reshape(1, -1)).to(Fq.device)
        res = {k_: np.zeros((len(prompts), GRID * GRID), np.float32) for k_ in ("csr_abs", "csr_rel", "csr_cos", "spec", "native")}
        vec = {k_: np.zeros((len(prompts), Fq.shape[-1]), np.float16) for k_ in ("dF_lesion", "dF_outside", "dF_mean", "dF_cls")}
        for j in range(len(prompts)):
            Fp = Fp_all[j: j + 1]
            dF = Fp - F0
            res["csr_abs"][j] = residuals.csr_abs(Fp, F0)[0].cpu().numpy()
            res["csr_rel"][j] = residuals.csr_rel(Fp, F0)[0].cpu().numpy()
            res["csr_cos"][j] = residuals.csr_cos(Fp, F0)[0].cpu().numpy()
            res["spec"][j] = residuals.pair_map(Fp, Fref)[0].cpu().numpy()
            res["native"][j] = nat_p[j].cpu().numpy()
            vec["dF_lesion"][j] = residuals.pooled_residual(dF, w_les)[0].cpu().numpy()
            vec["dF_outside"][j] = residuals.pooled_residual(dF, 1 - w_les)[0].cpu().numpy()
            vec["dF_mean"][j] = dF.mean(dim=1)[0].cpu().numpy()
        F0n = F0[0].cpu().numpy()
        np.savez_compressed(out / "cases" / f"{c.case_id}.npz", prompt_ids=np.array(pids),
                            f0_norm=baselines.feature_norm(F0n).astype(np.float32),
                            salience=baselines.salience(F0n).astype(np.float32),
                            native0=nat0.cpu().numpy().astype(np.float32),
                            mask=mask, frac=frac, valid=valid, valid_frac=mask_to_grid(valid, GRID), **res, **vec)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            rate = (k + 1) / (time.time() - t0)
            print(f"  {k + 1}/{len(todo)}  ({rate:.2f} cases/s, eta {(len(todo) - k - 1) / rate / 60:.1f} min)", flush=True)
    print("done:", out)


if __name__ == "__main__":
    main()
