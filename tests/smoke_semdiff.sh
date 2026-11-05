#!/usr/bin/env bash
# Semantic-differential pipeline on fake data with the fake model (CPU, no downloads, < 1 min).
# Covers both run formats: extract_semdiff.py (SteerViT, with feature maps) and the native-only
# format of extract_compare.py / extract_medsam3.py (here produced by extract.py on the fake model).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export PYTHONPATH="$PWD"
CFG=configs/prompts_semdiff.yaml

python -m pytest -q tests/test_semdiff.py
python tests/make_fake_data.py --out "$TMP/data" --n 12
python scripts/build_manifest.py --kvasir "$TMP/data/kvasir" --out "$TMP/manifest.csv"
python scripts/extract_semdiff.py --fake-model --config $CFG --manifest "$TMP/manifest.csv" \
    --dataset kvasir --out "$TMP/runs"
python scripts/extract.py --fake-model --config $CFG --manifest "$TMP/manifest.csv" \
    --dataset kvasir --out "$TMP/runs/clip"          # stand-in for a native-only run
for m in steervit clip; do
  python scripts/semdiff_eval.py --run "$TMP/runs/$m/gate1/kvasir" --manifest "$TMP/manifest.csv" --workers 2
done
python scripts/semdiff_table.py --dataset kvasir --root "$TMP/runs" --out "$TMP/tables"
head -20 "$TMP/tables/summary.md"
if [ -n "${KEEP_SMOKE:-}" ]; then cp -r "$TMP" "$KEEP_SMOKE"; fi
echo "SEMDIFF SMOKE TEST OK"
