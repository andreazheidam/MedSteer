# MedSteer FINAL-2 experiment (supervisor feedback 2026-10-28)

Same datasets and pipeline as FINAL.md. New frozen prompt set `configs/prompts_final2.yaml`;
`prompts_final.yaml` and the v2 runs (`runs_final*`) stay untouched as the record of v2.

| Feedback point | Status / where |
|---|---|
| 1 SAM3 "no prompt" | It is `""` -> `<start>,<end>` tokens through the text encoder (no text-free path exists in `Sam3Model`). Documented in `extract_compare.py`, stored as `null_condition` in meta.json, labelled "empty text" in tables and figures. CLIP the same. SAM3 CSR / output shift are relative to `""`. |
| 2 drop car, report G_cat | `configs/prompts_final2.yaml`, `compare_table.py` |
| 3 wrong-medical + counterfactual | new prompt ids `wrong_medical`, `counterfactual`; G_wrong, G_cf in `compare_table.py`; extra figure columns |
| 4 G_context = target-vs-context contrast | wording in config + table header; paper text still to change |
| 5 evaluation region independent of GT | `data.content_region`: all pixels of the unpadded image, only our zero padding excluded (image size only; no GT, no intensities). Used by `extract.py` and `extract_compare.py`; `metrics.localisation` no longer adds target pixels back (`gt_outside_valid` reports any, expected 0). Old `valid_region` / `refine_valid` kept only for re-evaluating v2 runs. |
| 6 DSE = change of head logits | Checked in the official code: `get_heatmap_logits` = `lin_seg_head` (one nn.Linear) on the output patch tokens, the same tokens used for dF; we store these logits as `native`, and `native_delta = native - native0`. So DSE = h(F_p) - h(F_0) = w^T dF exactly (bias cancels). The released heatmap is softmax over patches of these logits: monotone per image (same patch ranking) but NOT additive, so DSE is defined before the softmax. `extract.py` now verifies the identity numerically every batch (printed, `meta.json: dse_check`, `final_checks.py`). |
| 7 p-values secondary | main table = median [95% CI]; p-values only in `specificity_gains_pvalues.tex` |
| 8 central conclusion | paper text |

## Run (from ~/medsteer)

```bash
git add configs/prompts_final2.yaml && git commit -m "freeze final-2 prompts"
# (step 1, manifest: unchanged, reuse data/manifest_final.csv)
python scripts/describe_data.py --manifest data/manifest_final.csv --out paper_final2/data
sbatch slurm/final.sh            # writes runs_final2/ and runs_final2_compare/ (OUT=... to change)
sbatch slurm/eval_final.sh       # after extraction
python scripts/compare_table.py  # -> paper_final2/tables
for D in brisc busi kvasir isic; do python scripts/compare_panels.py --dataset $D; done
python scripts/final_checks.py
```
5 prompts instead of 4: expect ~25% longer jobs; check that SAM3 still fits in --time=03:00:00.
Smoke test first (GPU node, ~10 min):
  python scripts/extract.py --config configs/prompts_final2.yaml --manifest data/manifest_final.csv \
      --dataset kvasir --out runs_test --limit 8 --attention
  python scripts/evaluate.py --run runs_test/gate1/kvasir --manifest data/manifest_final.csv
  -> the progress line shows "DSE check ... = 1e-6 or smaller", loc.csv has wrong_medical /
     counterfactual rows and gt_outside_valid = 0.
