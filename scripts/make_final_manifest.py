"""Final manifest: BRISC / BUSI / Kvasir-SEG rows from the existing manifest + ISIC 2018 Task 1.

ISIC 2018 Task 1 (lesion boundary segmentation), test split as instructed:
  images  ISIC_<id>.jpg                 (e.g. ISIC2018_Task1-2_Test_Input/)
  masks   ISIC_<id>_segmentation.png    (e.g. ISIC2018_Task1_Test_GroundTruth/)
Every ISIC image contains a lesion (no negatives). The split is read from the folder name
(test / validation / training); folders without one are counted as test.

Usage:
  python scripts/make_final_manifest.py --isic /path/to/isic2018 \
      --base data/manifest_all.csv --out data/manifest_final.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import Case, read_manifest, summarise, write_manifest  # noqa: E402

KEEP = ("brisc", "busi", "kvasir")


def isic_split(p: Path) -> str:
    s = "/".join(p.parts).lower()
    if "test" in s:
        return "test"
    if "valid" in s:
        return "val"
    if "train" in s:
        return "train"
    return "test"


def build_isic(root: str | Path) -> list[Case]:
    root = Path(root)
    imgs, masks = {}, {}
    for p in sorted(root.rglob("ISIC_*")):
        if "__MACOSX" in p.parts or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = re.match(r"(ISIC_\d+)(_segmentation)?$", p.stem, flags=re.I)
        if not m:
            continue
        (masks if m.group(2) else imgs)[(isic_split(p), m.group(1).upper())] = p
    if not imgs:
        raise FileNotFoundError(f"no ISIC_<id>.jpg images found under {root}")
    cases = []
    for (split, iid), p in sorted(imgs.items()):
        mk = masks.get((split, iid))
        cases.append(Case(f"isic_{iid.lower()}", "isic", split, "lesion", "dermoscopy",
                          str(p.resolve()), [str(mk.resolve())] if mk else []))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic", required=True)
    ap.add_argument("--isic-split", default="test", help="ISIC split to keep (Task 1 test)")
    ap.add_argument("--base", default="data/manifest_all.csv")
    ap.add_argument("--out", default="data/manifest_final.csv")
    a = ap.parse_args()

    # BRISC: official test split only (incl. no_tumor negatives); BUSI and Kvasir-SEG have no
    # official split, so all images are used (the stored split is bookkeeping only).
    base = [c for c in read_manifest(a.base) if c.dataset in KEEP
            and not (c.dataset == "brisc" and c.split != "test")]
    isic = [c for c in build_isic(a.isic) if c.split == a.isic_split]
    nomask = [c for c in isic if not c.mask_paths]
    if nomask:
        print(f"WARNING: {len(nomask)} ISIC images without a mask, e.g. {nomask[0].image_path}")
    cases = base + isic
    write_manifest(cases, a.out)
    print(summarise(cases))
    print(f"\nISIC {a.isic_split}: {len(isic)} images, {len(isic) - len(nomask)} with masks")
    print(f"wrote {len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
