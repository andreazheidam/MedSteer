"""Build data/manifest.csv from the raw BRISC 2025 and BUSI folders and print counts.

Usage:
  python scripts/build_manifest.py --brisc /path/to/brisc2025 --busi /path/to/Dataset_BUSI_with_GT \
      --kvasir /path/to/kvasir --sunnybrook /path/to/sunnybrook
Pass every dataset you want in the same call: the manifest is rewritten each time.
Check the printed table: every lesion image you want to localise needs a mask; healthy
images (no_tumor / normal) are needed for the tumour-vs-no-tumour analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsteer.data import (build_brisc, build_busi, build_kvasir, build_sunnybrook,  # noqa: E402
                           summarise, write_manifest)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brisc", default=None)
    ap.add_argument("--busi", default=None)
    ap.add_argument("--kvasir", default=None, help="folder containing Kvasir-SEG/images, masks")
    ap.add_argument("--sunnybrook", default=None, help="folder containing images/, labels/")
    ap.add_argument("--out", default="data/manifest.csv")
    args = ap.parse_args()
    cases = []
    if args.brisc:
        cases += build_brisc(args.brisc)
    if args.busi:
        cases += build_busi(args.busi)
    if args.kvasir:
        cases += build_kvasir(args.kvasir)
    if args.sunnybrook:
        cases += build_sunnybrook(args.sunnybrook)
    if not cases:
        sys.exit("no cases found: pass at least one of --brisc --busi --kvasir --sunnybrook")
    write_manifest(cases, args.out)
    print(summarise(cases))
    missing = [c for c in cases if not c.is_healthy and not c.mask_paths]
    if missing:
        print(f"\nNOTE: {len(missing)} lesion images have no mask (used only for presence "
              f"analyses), e.g. {missing[0].image_path}")
    print(f"\nwrote {len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
