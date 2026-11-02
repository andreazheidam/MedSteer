#!/usr/bin/env bash
# End-to-end test on fake data with a fake model (CPU, no downloads, ~1 min).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export PYTHONPATH="$PWD"

python -m pytest -q tests/test_metrics.py
python tests/make_fake_data.py --out "$TMP/data" --n 12
python scripts/build_manifest.py --brisc "$TMP/data/brisc2025" \
    --busi "$TMP/data/Dataset_BUSI_with_GT" --kvasir "$TMP/data/kvasir" \
    --sunnybrook "$TMP/data/sunnybrook" --out "$TMP/manifest.csv"
python scripts/sanity.py --fake-model --manifest "$TMP/manifest.csv" --dataset busi \
    --n-cases 4 --out "$TMP/sanity"
for ds in busi brisc kvasir sunnybrook; do
  python scripts/extract.py --fake-model --manifest "$TMP/manifest.csv" --dataset "$ds" \
      --out "$TMP/runs" --attention
  python scripts/evaluate.py --run "$TMP/runs/gate1/$ds" --manifest "$TMP/manifest.csv" --workers 2
done
python scripts/analyze.py --runs "$TMP/runs/gate1/busi" "$TMP/runs/gate1/brisc" \
    "$TMP/runs/gate1/kvasir" "$TMP/runs/gate1/sunnybrook" --out "$TMP/paper"
python scripts/paper_figs.py --runs "$TMP/runs/gate1/busi" "$TMP/runs/gate1/brisc" \
    "$TMP/runs/gate1/kvasir" "$TMP/runs/gate1/sunnybrook" --manifest "$TMP/manifest.csv" --out "$TMP/paper/figs"
python scripts/panels.py --run "$TMP/runs/gate1/busi" --manifest "$TMP/manifest.csv" --out "$TMP/paper/figs"
ls "$TMP/paper/tables" "$TMP/paper/figs"
if [ -n "${KEEP_SMOKE:-}" ]; then cp -r "$TMP" "$KEEP_SMOKE"; fi
echo "SMOKE TEST OK"
