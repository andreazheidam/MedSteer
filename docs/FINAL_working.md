# MedSteer final experiment (supervisor feedback 2026-10-22)

Datasets: BRISC (test), BUSI (all), Kvasir-SEG (all), ISIC 2018 Task 1 (test). Sunnybrook dropped.
Prompts: `configs/prompts_final.yaml`, exact strings, frozen. Commit it before the runs.

| Feedback point | Where |
|---|---|
| 1, 2 Exact frozen prompts, cat + car | `configs/prompts_final.yaml`; both controls in every table and figure |
| 3, 4 Preprocessing + dataset details | `scripts/describe_data.py` -> `paper_final/data/*.csv/.tex` |
| 5 Heatmap scaling statement | `compare_panels.py` row labels + `figs/CAPTION_NOTE.txt` |
| 6 Directed steering effect S = w^T dF | `native_delta` in loc.csv (already computed by evaluate.py); DSE rows in table + figures |
| 7 AUROC/AUPRC headline, CSS secondary | `compare_table.py` |
| 9 G_irrel, G_context | `compare_table.py` -> `tables/specificity_gains.*` |
| Vector figures | `compare_panels.py` writes PDF + SVG (+ PNG preview) |

## Run (from ~/medsteer)

```bash
# 0. copy these files into the repo (unzip medsteer_final.zip here); keep extract_compare.py
git add configs/prompts_final.yaml && git commit -m "freeze final prompts"   # if it is a git repo

# 1. manifest (login node)
source .venv/bin/activate
python scripts/make_final_manifest.py --isic /path/to/ISIC2018 --out data/manifest_final.csv
python scripts/describe_data.py --manifest data/manifest_final.csv --out paper_final/data

# 2. extraction: 12 GPU jobs (0-3 SteerViT, 4-7 CLIP, 8-11 SAM3)
sbatch slurm/final.sh

# 3. metrics with the unchanged evaluate.py (CPU), after step 2 finished
sbatch slurm/eval_final.sh

# 4. tables + figures
python scripts/compare_table.py --out paper_final
for D in brisc busi kvasir isic; do python scripts/compare_panels.py --dataset $D --out paper_final/figs; done
```

Main-text figure with fewer rows, e.g.:
`python scripts/compare_panels.py --dataset isic --rows sv_map sv_dse sv_csr sam_map --out paper_final/figs_main`
