#!/usr/bin/env bash
# Fine-tuning pipeline on fake data with the trainable fake model (CPU, no downloads, ~1 min):
# train (target-only, spec hard, spec mean, head + preservation) -> extract the test split with the
# fine-tuned weights -> semdiff_eval -> semdiff_table comparing frozen vs fine-tuned.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export PYTHONPATH="$PWD"
CFG=configs/prompts_semdiff.yaml

python -m pytest -q tests/test_finetune.py
python tests/make_fake_data.py --out "$TMP/data" --n 30
python scripts/build_manifest.py --kvasir "$TMP/data/kvasir" --out "$TMP/manifest.csv"
COMMON="--fake-model --config $CFG --manifest $TMP/manifest.csv --dataset kvasir --out $TMP/ft --epochs 3 --batch-size 4 --lr 1e-2"
python scripts/finetune.py $COMMON --run-name target --loss target
python scripts/finetune.py $COMMON --run-name spec_hard --loss spec --spec-mode hard
python scripts/finetune.py $COMMON --run-name spec_mean --loss spec --spec-mode mean --controls irrelevant wrong_medical
python scripts/finetune.py $COMMON --run-name spec_head --loss spec --train-head --lambda-pres 0.1 --train-paraphrase first

python scripts/extract_semdiff.py --fake-model --config $CFG --manifest "$TMP/manifest.csv" --dataset kvasir \
    --splits test --out "$TMP/runs" --name steervit
for r in target spec_hard; do
  python scripts/extract_semdiff.py --fake-model --config $CFG --manifest "$TMP/manifest.csv" --dataset kvasir \
      --splits test --out "$TMP/runs" --name "ft_$r" --weights "$TMP/ft/$r/ckpt_best.pt"
done
for m in steervit ft_target ft_spec_hard; do
  python scripts/semdiff_eval.py --run "$TMP/runs/$m/gate1/kvasir" --manifest "$TMP/manifest.csv" --workers 2
done
python scripts/semdiff_table.py --dataset kvasir --root "$TMP/runs" --out "$TMP/tables" \
    --models steervit ft_target ft_spec_hard
grep -A6 "^## Specificity" "$TMP/tables/summary.md"
tail -n 2 "$TMP/ft/spec_hard/log.jsonl" | cut -c1-300
if [ -n "${KEEP_SMOKE:-}" ]; then cp -r "$TMP" "$KEEP_SMOKE"; fi
echo "FINETUNE SMOKE TEST OK"
