"""Write small fake BUSI- and BRISC-shaped folders (bright blobs = lesions) for the smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def blob(h, w, rng):
    yy, xx = np.mgrid[:h, :w]
    r = rng.uniform(0.04, 0.25) * min(h, w)
    cy, cx = rng.uniform(r, h - r), rng.uniform(r, w - r)
    return ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2


def image(h, w, rng, lesion):
    img = rng.normal(70, 20, (h, w))
    m = blob(h, w, rng) if lesion else np.zeros((h, w), bool)
    img[m] += 110
    return np.clip(img, 0, 255).astype(np.uint8), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_data")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out)
    busi = root / "Dataset_BUSI_with_GT"
    for label in ("benign", "malignant", "normal"):
        (busi / label).mkdir(parents=True, exist_ok=True)
        for i in range(1, args.n + 1):
            img, m = image(150, 190, rng, label != "normal")
            Image.fromarray(img).save(busi / label / f"{label} ({i}).png")
            Image.fromarray(m.astype(np.uint8) * 255).save(busi / label / f"{label} ({i})_mask.png")
    codes = {"glioma": "gl", "meningioma": "me", "pituitary": "pi", "no_tumor": "nt"}
    brisc = root / "brisc2025"
    k = 0
    for split in ("train", "test"):
        for label, code in codes.items():
            cdir = brisc / "classification_task" / split / label
            cdir.mkdir(parents=True, exist_ok=True)
            sdir = brisc / "segmentation_task" / split
            (sdir / "images").mkdir(parents=True, exist_ok=True)
            (sdir / "masks").mkdir(parents=True, exist_ok=True)
            for _ in range(args.n // 2):
                k += 1
                stem = f"brisc2025_{split}_{k:05d}_{code}_ax_t1"
                img, m = image(160, 160, rng, label != "no_tumor")
                Image.fromarray(img).save(cdir / f"{stem}.jpg", quality=95)
                if label != "no_tumor":
                    Image.fromarray(img).save(sdir / "images" / f"{stem}.jpg", quality=95)
                    Image.fromarray(m.astype(np.uint8) * 255).save(sdir / "masks" / f"{stem}.png")
    # Kvasir-SEG: Kvasir-SEG/{images,masks}/<id>.jpg (RGB images, JPEG masks)
    kv = root / "kvasir" / "Kvasir-SEG"
    for sub in ("images", "masks"):
        (kv / sub).mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img, m = image(180, 210, rng, True)
        if i % 2:   # black position-overlay box in the lower-left corner, as in real Kvasir
            img[125:, :55] = 0
            m[125:, :55] = False
        Image.fromarray(np.stack([img, img // 2, img // 3], -1)).save(kv / "images" / f"cju{i:04d}x.jpg", quality=95)
        Image.fromarray(m.astype(np.uint8) * 255).save(kv / "masks" / f"cju{i:04d}x.jpg", quality=95)
    # Sunnybrook: images/<group>/<pat>_rawdcm_<fff>.png + labels/<group>/<pat>gtmask<ffff>.png,
    # masks stored as 0/1 (the real files are tiny, so they may be 0/1), plus ignored .dcm files
    sb = root / "sunnybrook"
    for group, code in (("Healthy_normal", "N"), ("LV_hypertrophy", "HYP")):
        (sb / "images" / group).mkdir(parents=True, exist_ok=True)
        (sb / "labels" / group).mkdir(parents=True, exist_ok=True)
        for k in range(3):
            pat = f"SC-{code}-{k + 2:02d}"
            for frame in (20, 28, 40):
                img, m = image(256, 256, rng, True)
                Image.fromarray(img).save(sb / "images" / group / f"{pat}_rawdcm_{frame:03d}.png")
                (sb / "images" / group / f"{pat}_rawdcm_{frame:03d}.dcm").write_bytes(b"x")
                Image.fromarray(m.astype(np.uint8)).save(sb / "labels" / group / f"{pat}gtmask{frame:04d}.png")
    print("fake data ->", root)


if __name__ == "__main__":
    main()
