"""Comparison models (CLIP, SAM3) in EXACTLY the format of scripts/extract.py, so that
scripts/evaluate.py and scripts/analyze.py run on them unchanged.

DINOv2 needs no run of its own: SteerViT with texts=None *is* the frozen DINOv2-Base
backbone, so attn0 / f0_norm / salience in runs/gate1 are the DINOv2 maps.

Same images, padding, masks, valid region and 24x24 grid as the SteerViT run:
masks/valid are computed at 336 px with the medsteer functions; each model sees the same
padded image at its own resolution and its maps are area-pooled onto the 24x24 grid.

  CLIP ViT-B/16 @ 384 px (24x24 tokens)
    F(x) = dense patch embeddings (MaskCLIP last layer: value path, no attention mixing)
    The image encoder never sees the text  ->  F(x,p) = F(x), CSR = 0 by construction.
    native  = cos(patch embedding, text embedding of p)
    native0 = the same for the EMPTY-TEXT control (--null-prompt, default "")
  SAM3 @ 1008 px (72x72 tokens -> 3x3 pooled to 24x24)
    F(x,p) = output of the text-image fusion encoder (DETR encoder) for prompt p
    F(x)   = the same for the EMPTY-TEXT control  ->  CSR / output shift are measured
             relative to text "", not relative to a text-free forward pass
    native = semantic_seg logits for p, native0 = for the empty-text control

  Empty-text control (feedback 2026-10-28, point 1). Neither model has a text-free path in
  transformers: Sam3Model.forward requires input_ids or text_embeds. For "" the Sam3Processor
  tokenises to <start>,<end> + padding (max_length 32, attention mask 1 on those two tokens);
  the text encoder runs on them, and the DETR fusion encoder cross-attends to the two encoded
  special tokens. The processor substitutes the string "visual" only for box-only prompts
  (text=None with input_boxes), which is not used here. CLIP: "" -> <start>,<end>, pooled at
  <end>. So this condition is an empty-text control, NOT an unprompted model.

Output: <out>/<model>/gate1/<dataset>/cases/<case_id>.npz + meta.json
("gate1" only so slurm/evaluate.sh finds it with OUT=<out>/<model>).

Usage:
  python scripts/extract_compare.py --model clip --dataset busi --out runs_compare
  python scripts/extract_compare.py --model sam3 --dataset brisc --splits test --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer import baselines, residuals  # noqa: E402
from medsteer.config import config_hash, load_config  # noqa: E402
from medsteer.data import VALID_REGION, content_region, load_image, load_mask, mask_to_grid, read_manifest  # noqa: E402

R_EVAL, GRID = 336, 24          # SteerViT evaluation geometry (mask resolution, patch grid)


def to_grid(tok: torch.Tensor, g: int = GRID) -> torch.Tensor:
    """(B, Gm*Gm, D) or (B, Gm*Gm) -> (B, g*g, D) / (B, g*g) by area pooling."""
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


def spatial_tokens(h: torch.Tensor) -> torch.Tensor:
    """Fusion-encoder output -> (B, N, D) with N a square number, whatever the layout."""
    if h.dim() == 4:                                   # (B, D, H, W) or (B, H, W, D)
        h = h.flatten(2).transpose(1, 2) if h.shape[1] > h.shape[-1] else h.flatten(1, 2)
    if h.dim() == 3 and h.shape[0] != h.shape[1] and int(h.shape[0] ** 0.5) ** 2 == h.shape[0] \
            and int(h.shape[1] ** 0.5) ** 2 != h.shape[1]:
        h = h.transpose(0, 1)                          # (N, B, D) -> (B, N, D)
    N = h.shape[1]
    if int(round(N ** 0.5)) ** 2 != N:                 # leading special tokens
        s = int(N ** 0.5)
        h = h[:, N - s * s:]
    return h


class CLIPWrap:
    name = "clip"

    def __init__(self, model_id="openai/clip-vit-base-patch16", device=None, resolution=384, **_):
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.m = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_id)
        ip = CLIPImageProcessor.from_pretrained(model_id)
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        self.resolution = resolution
        self.backbone_name = model_id
        self._F = None

    def to_tensor(self, imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def text(self, s: str) -> torch.Tensor:
        t = self.tok([s], padding=True, return_tensors="pt").to(self.device)
        e = self.m.text_projection(self.m.text_model(**t).pooler_output)
        return F.normalize(e[0], dim=-1)

    @torch.no_grad()
    def encode(self, x):
        vm = self.m.vision_model
        last = vm.encoder.layers[-1]
        grab = {}

        def pre_hook(mod, args, kwargs):              # input of the last block
            grab["h"] = args[0] if args else kwargs["hidden_states"]
        hk = last.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            o = vm(pixel_values=x, interpolate_pos_encoding=True)
        finally:
            hk.remove()
        h = grab["h"]
        d = last.self_attn.out_proj(last.self_attn.v_proj(last.layer_norm1(h)))
        d = self.m.visual_projection(vm.post_layernorm(d))
        cls = self.m.visual_projection(o.pooler_output)
        return cls, to_grid(d[:, 1:])                  # (B, 512), (B, 576, 512)

    @torch.no_grad()
    def unprompted(self, x):
        cls, Fx = self.encode(x)
        self._F = Fx
        return cls, Fx

    @torch.no_grad()
    def prompted(self, x, text):
        """-> cls, F(x,p), native map (B, N). Image features do not depend on the text."""
        cls, Fx = (None, self._F) if self._F is not None else self.encode(x)
        nat = F.normalize(Fx, dim=-1) @ self.text(text)
        return cls, Fx, nat


class BiomedCLIPWrap:
    """BiomedCLIP (Zhang et al., PMC-15M; open_clip + timm ViT-B/16, PubMedBERT text encoder).
    Same dense map as CLIPWrap (MaskCLIP value path of the last block -> final norm -> projection),
    at 384 px (timm resamples the position embeddings) so the token grid is 24x24 like CLIP."""
    name = "biomedclip"
    HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, model_id=None, device=None, resolution=384, **_):
        import open_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hub = model_id or self.HUB
        self.m, preprocess = open_clip.create_model_from_pretrained(hub)
        self.m = self.m.to(self.device).eval()
        self.tok = open_clip.get_tokenizer(hub)
        norm = [t for t in getattr(preprocess, "transforms", []) if type(t).__name__ == "Normalize"]
        mean, std = (norm[0].mean, norm[0].std) if norm else ((0.48145466, 0.4578275, 0.40821073),
                                                               (0.26862954, 0.26130258, 0.27577711))
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        trunk = self.m.visual.trunk
        try:
            trunk.set_input_size(img_size=(resolution, resolution))
        except Exception as e:           # older timm: stay at the native 224 px (14x14, pooled to 24x24)
            print(f"[biomedclip] could not resize to {resolution} px ({e}); using 224 px", flush=True)
            resolution = 224
        self.resolution = resolution
        self.backbone_name = hub
        self._F = None
        print(f"[biomedclip] input {resolution} px, mean {tuple(round(float(v), 4) for v in mean)}", flush=True)

    def to_tensor(self, imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def text(self, s: str) -> torch.Tensor:
        t = self.tok([s], context_length=256).to(self.device)
        return F.normalize(self.m.encode_text(t)[0].float(), dim=-1)

    @torch.no_grad()
    def encode(self, x):
        vis = self.m.visual
        trunk = vis.trunk
        last = trunk.blocks[-1]
        grab = {}

        def pre_hook(mod, args, kwargs):                 # input of the last block
            grab["h"] = args[0] if args else kwargs["x"]
        hk = last.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            cls = vis(x)                                 # global embedding (also triggers the hook)
        finally:
            hk.remove()
        h = grab["h"]
        D = h.shape[-1]
        v = last.attn.qkv(last.norm1(h))[..., 2 * D:]   # value path only (MaskCLIP), no attention mixing
        if hasattr(last.attn, "norm"):
            v = last.attn.norm(v)
        d = last.attn.proj(v)
        d = trunk.fc_norm(trunk.norm(d))
        d = vis.head(d)
        n_prefix = getattr(trunk, "num_prefix_tokens", 1)
        return cls.float(), to_grid(d[:, n_prefix:].float())

    @torch.no_grad()
    def unprompted(self, x):
        cls, Fx = self.encode(x)
        self._F = Fx
        return cls, Fx

    @torch.no_grad()
    def prompted(self, x, text):
        """-> cls, F(x,p), native map (B, N). Image features do not depend on the text."""
        cls, Fx = (None, self._F) if self._F is not None else self.encode(x)
        nat = F.normalize(Fx, dim=-1) @ self.text(text)
        return cls, Fx, nat


class SAM3Wrap:
    name = "sam3"

    def __init__(self, model_id="facebook/sam3", device=None, resolution=1008, null_prompt="", **_):
        from transformers import Sam3Model, Sam3Processor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.m = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.p = Sam3Processor.from_pretrained(model_id)
        ip = self.p.image_processor
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        self.resolution = resolution
        self.null_prompt = null_prompt
        self.backbone_name = model_id
        self._vis = None

    def to_tensor(self, imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def _run(self, text, B):
        t = self.p(text=[text] * B, return_tensors="pt").to(self.device)
        grab = {}

        def hook(mod, inp, out):                       # text-fused image tokens (B, 72*72, 256)
            grab["h"] = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hk = self.m.detr_encoder.register_forward_hook(hook)
        try:
            o = self.m(vision_embeds=self._vis, input_ids=t["input_ids"], attention_mask=t["attention_mask"])
        finally:
            hk.remove()
        Fp = to_grid(spatial_tokens(grab["h"].float()))
        nat = F.adaptive_avg_pool2d(o.semantic_seg.float(), GRID).reshape(B, -1)
        return None, Fp, nat

    @torch.no_grad()
    def unprompted(self, x):
        self._vis = self.m.get_vision_features(pixel_values=x)
        cls, F0, self._nat0 = self._run(self.null_prompt, x.shape[0])
        return cls, F0

    @torch.no_grad()
    def prompted(self, x, text):
        return self._run(text, x.shape[0])


MODELS = {"clip": CLIPWrap, "biomedclip": BiomedCLIPWrap, "sam3": SAM3Wrap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--config", default="configs/prompts_final2.yaml")
    ap.add_argument("--manifest", default="data/manifest_all.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--out", default="runs_compare")
    ap.add_argument("--null-prompt", default="", help='baseline text, default "" = empty-text control (CLIP/SAM3 have no text-free path)')
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    raw, dsets = load_config(args.config)
    dcfg = dsets[args.dataset]
    cases = read_manifest(args.manifest, datasets=[args.dataset], splits=args.splits)
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out) / args.model / "gate1" / args.dataset
    (out / "cases").mkdir(parents=True, exist_ok=True)

    kw = {"device": args.device, "null_prompt": args.null_prompt}
    if args.model_id:
        kw["model_id"] = args.model_id
    model = MODELS[args.model](**kw)
    bs = args.batch_size or (32 if args.model in ("clip", "biomedclip") else 4)
    prompts = dcfg.prompts
    pids = [p.id for p in prompts]
    ref_idx = pids.index(dcfg.reference)

    meta = {"dataset": args.dataset, "n_cases": len(cases), "gate": 1.0, "resolution": R_EVAL,
            "grid": GRID, "model_resolution": model.resolution, "backbone": model.backbone_name,
            "checkpoint": model.backbone_name, "config_hash": config_hash(args.config),
            "prompts": [p.__dict__ for p in prompts], "reference": dcfg.reference,
            "null_prompt": args.null_prompt,
            "null_condition": "empty-text control" if args.null_prompt == "" else "baseline text",
            "attention": False, "valid_region": VALID_REGION}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = [c for c in cases if not (out / "cases" / f"{c.case_id}.npz").exists()]
    print(f"[{args.model}] {args.dataset}: {len(cases)} cases, {len(todo)} to extract, "
          f"{len(prompts)} prompts", flush=True)
    t0 = time.time()
    for b in range(0, len(todo), bs):
        batch = todo[b: b + bs]
        img_eval = np.stack([load_image(c.image_path, R_EVAL) for c in batch])
        masks = np.stack([load_mask(c.mask_paths, R_EVAL, c.image_path) for c in batch])
        fracs = np.stack([mask_to_grid(m, GRID) for m in masks])
        valids = np.stack([content_region(c.image_path, R_EVAL) for c in batch])
        vfracs = np.stack([mask_to_grid(v, GRID) for v in valids])
        x = model.to_tensor(np.stack([load_image(c.image_path, model.resolution) for c in batch]))
        w_les = torch.from_numpy(fracs.reshape(len(batch), -1)).to(x.device)

        cls0, F0 = model.unprompted(x)
        if args.model == "sam3":
            nat0 = model._nat0
        else:
            nat0 = model.prompted(x, args.null_prompt)[2]
        _, Fref, _ = model.prompted(x, prompts[ref_idx].text)

        P, B, N, D = len(prompts), len(batch), F0.shape[1], F0.shape[2]
        res = {k: np.zeros((B, P, N), np.float32) for k in ("csr_abs", "csr_rel", "csr_cos", "spec", "native")}
        vec = {k: np.zeros((B, P, D), np.float16) for k in ("dF_lesion", "dF_outside", "dF_mean", "dF_cls")}
        for j, p in enumerate(prompts):
            clsp, Fp, nat = model.prompted(x, p.text)
            dF = Fp - F0
            res["csr_abs"][:, j] = residuals.csr_abs(Fp, F0).cpu().numpy()
            res["csr_rel"][:, j] = residuals.csr_rel(Fp, F0).cpu().numpy()
            res["csr_cos"][:, j] = residuals.csr_cos(Fp, F0).cpu().numpy()
            res["spec"][:, j] = residuals.pair_map(Fp, Fref).cpu().numpy()
            res["native"][:, j] = nat.float().cpu().numpy()
            vec["dF_lesion"][:, j] = residuals.pooled_residual(dF, w_les).cpu().numpy()
            vec["dF_outside"][:, j] = residuals.pooled_residual(dF, 1 - w_les).cpu().numpy()
            vec["dF_mean"][:, j] = dF.mean(dim=1).cpu().numpy()
            if clsp is not None and cls0 is not None:
                vec["dF_cls"][:, j] = (clsp - cls0).cpu().numpy()

        F0n = F0.float().cpu().numpy()
        nat0 = nat0.float().cpu().numpy()
        for i, c in enumerate(batch):
            arrs = {k: v[i] for k, v in res.items()} | {k: v[i] for k, v in vec.items()}
            np.savez_compressed(
                out / "cases" / f"{c.case_id}.npz", prompt_ids=np.array(pids),
                f0_norm=baselines.feature_norm(F0n[i]).astype(np.float32),
                salience=baselines.salience(F0n[i]).astype(np.float32),
                native0=nat0[i].astype(np.float32),
                mask=masks[i], frac=fracs[i], valid=valids[i], valid_frac=vfracs[i], **arrs)
        done = b + len(batch)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)}  ({rate:.2f} cases/s, eta {(len(todo) - done) / rate / 60:.1f} min)",
              flush=True)
    print("done:", out)


if __name__ == "__main__":
    main()
