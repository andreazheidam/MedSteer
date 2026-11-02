"""Dataset and preprocessing facts for the paper (feedback points 3 and 4), measured from the
files listed in the manifest, not typed by hand.

Per dataset: images, positives (non-empty mask) / negatives, patients (if recorded), splits
used, empty / missing / unreadable masks, original size range, PIL mode, dtype, channels,
intensity range, mask values. Writes CSV + LaTeX and prints both.

  python scripts/describe_data.py --manifest data/manifest_final.csv --out paper_final/data
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import read_manifest  # noqa: E402

MODALITY = {"brisc": "MRI (T1, 2D slices)", "busi": "Ultrasound (2D)",
            "kvasir": "Colonoscopy (RGB video frames)", "isic": "Dermoscopy (RGB)"}


def describe(cases):
    rows = []
    for ds in sorted({c.dataset for c in cases}):
        cs = [c for c in cases if c.dataset == ds]
        sizes, modes, dtypes, chans, lo, hi, mvals = [], Counter(), Counter(), Counter(), [], [], Counter()
        unreadable = empty = missing = pos = neg = 0
        for c in cs:
            try:
                im = Image.open(c.image_path)
                modes[im.mode] += 1
                a = np.asarray(im)
                dtypes[str(a.dtype)] += 1
                chans[1 if a.ndim == 2 else a.shape[2]] += 1
                sizes.append(a.shape[:2])
                lo.append(float(a.min())); hi.append(float(a.max()))
            except Exception:
                unreadable += 1
                continue
            if not c.mask_paths:
                missing += 1
                neg += 1 if c.is_healthy else 0
                continue
            u = None
            for mp in c.mask_paths:
                try:
                    m = np.asarray(Image.open(mp).convert("L"))
                except Exception:
                    unreadable += 1
                    continue
                mvals.update(np.unique(m).tolist()[:4])
                u = (m > (127 if m.max() > 1 else 0)) if u is None else (u | (m > (127 if m.max() > 1 else 0)))
            if u is None or not u.any():
                empty += 1
                neg += 1
            else:
                pos += 1
        h = np.array([s[0] for s in sizes]); w = np.array([s[1] for s in sizes])
        pats = {c.patient for c in cs if c.patient}
        rows.append({
            "dataset": ds, "modality": MODALITY.get(ds, "?"),
            "splits_used": ", ".join(f"{k}={v}" for k, v in sorted(Counter(c.split for c in cs).items())),
            "n_images": len(cs), "positives (non-empty mask)": pos, "negatives": neg,
            "labels": ", ".join(f"{k}={v}" for k, v in sorted(Counter(c.label for c in cs).items())),
            "patients": len(pats) if pats else "not recorded",
            "no mask file": missing, "empty mask": empty, "unreadable": unreadable,
            "height min/med/max": f"{h.min()}/{int(np.median(h))}/{h.max()}" if len(h) else "",
            "width min/med/max": f"{w.min()}/{int(np.median(w))}/{w.max()}" if len(w) else "",
            "PIL mode": ", ".join(f"{k}={v}" for k, v in modes.most_common()),
            "dtype": ", ".join(dtypes), "channels": ", ".join(f"{k}={v}" for k, v in sorted(chans.items())),
            "intensity range": f"[{min(lo):g}, {max(hi):g}]" if lo else "",
            "per-image max min/med": f"{min(hi):g}/{np.median(hi):g}" if hi else "",
            "mask values": ", ".join(str(v) for v in sorted(mvals)[:6]),
        })
    return pd.DataFrame(rows)


PREPROC = [  # common geometry -> identical for every model (medsteer/data.py)
    ("Grayscale to RGB", "single channel replicated to 3 channels (BRISC, BUSI); RGB kept"),
    ("Intensity rescaling", "none for 8-bit images; 16-bit/float images: per-image min-max to 0-255"),
    ("Clipping", "none"),
    ("Padding", "zero-padded to a square, image centred (value 0)"),
    ("Image resize", "PIL bicubic, to each model's input resolution"),
    ("Mask resize", "nearest neighbour to 336 x 336 (union of all mask files), metrics at 336 px"),
    ("Patch grid", "24 x 24 for every model (maps area-pooled to this grid)"),
    ("Evaluation region", "all pixels of the original unpadded image; only our zero padding is "
                          "excluded. Defined from image size alone (independent of the ground truth "
                          "and of intensities); dark regions and the Kvasir position overlay stay in"),
]
MODEL_NORM = [  # model-specific, as in each released implementation
    ("SteerViT / DINOv2-B/14", "336 px, 24x24 tokens", "ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) (timm config)"),
    ("CLIP ViT-B/16 (OpenAI)", "384 px, 24x24 tokens, interpolated pos. emb.", "CLIP mean (0.481, 0.458, 0.408), std (0.269, 0.261, 0.276)"),
    ("SAM3", "1008 px, 72x72 tokens, pooled 3x3 to 24x24", "mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5) (Sam3Processor)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_final.csv")
    ap.add_argument("--out", default="paper_final/data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = describe(read_manifest(a.manifest))
    d.to_csv(out / "datasets.csv", index=False)
    (out / "datasets.tex").write_text(d.set_index("dataset").T.to_latex())
    pd.DataFrame(PREPROC, columns=["step", "setting"]).to_csv(out / "preprocessing_common.csv", index=False)
    pd.DataFrame(MODEL_NORM, columns=["model", "resolution", "normalisation"]).to_csv(
        out / "preprocessing_models.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
    print(d.set_index("dataset").T.to_string())
    print(f"\nwrote {out}/datasets.csv/.tex, preprocessing_common.csv, preprocessing_models.csv")


if __name__ == "__main__":
    main()
