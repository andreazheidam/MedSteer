# MedSteer — probing clinical steerability of visual foundation models

Given one medical image and different prompts, where and how does the **released SteerViT**
(Ruthardt et al., ECCV 2026; github.com/JonaRuthardt/SteerViT) change its visual representation?
No training: everything is measured on the frozen, released **SteerDINOv2-Base**.

**Clinical Steering Residual (CSR):** ΔF(x, p) = F(x, p) − F(x), per patch.
F(x) is SteerViT called with `texts=None`, which skips every cross-attention block and is
exactly the frozen DINOv2 backbone (checked by `scripts/sanity.py`). The steering-effect map is
Rᵢ = ‖ΔFᵢ‖₂ on the 24×24 patch grid (336 px input, 14 px patches).

Datasets: **BRISC 2025** (brain MRI: glioma / meningioma / pituitary / no tumour, with masks),
**BUSI** (breast ultrasound: benign / malignant / normal, with masks), **Kvasir-SEG**
(colonoscopy polyps) and **Sunnybrook** (cardiac MRI, left-ventricle masks, several slices per
patient: statistics are aggregated per patient).

## Setup (Hendrix)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash tests/smoke.sh          # CPU, ~1 min, fake model + fake data, no downloads
python scripts/build_manifest.py --brisc /path/to/brisc --busi /path/to/busi \
    --kvasir /path/to/kvasir --sunnybrook /path/to/sunnybrook --out data/manifest_all.csv
```
`build_manifest.py` prints counts per dataset, split and class, and how many have masks.
Check that BRISC tumour images found their masks and that no_tumor / normal images are there.
Unzip the datasets to Hendrix storage first (not inside the repo).

## Run order

| Step | Command | What it gives |
|---|---|---|
| 0. Sanity | `sbatch slurm/sanity.sh` (put a dog photo at `data/dog.jpg`) | `sanity/*/report.json` + alignment and natural-image figures. **Stop if a check fails.** |
| 1. Freeze prompts | review + commit `configs/prompts.yaml` | its hash is stored with every run |
| 2. Pilot | `python scripts/extract.py --dataset busi --limit 50 --out runs/pilot` → evaluate → look at `panels.py` | catches bugs cheaply |
| 3. Extract | `sbatch slurm/extract.sh <ds>` (BRISC: add `--splits test`) | `runs/gate1/<ds>/cases/*.npz` |
| 4. Metrics | `sbatch slurm/evaluate.sh <ds>` | `loc.csv`, `global.csv`, `direction.csv` |
| 5. Stats + figures | `sbatch slurm/evaluate.sh analyze` | `paper_gate1/tables/*.csv`, `paper_gate1/figs/*` |
| 6. Gate sweep | `sbatch slurm/gate_sweep.sh`, then steps 4-5 with `GATE=0.5` etc. | `runs/gate0.5/...`, `paper_gate0.5/...` |

Extraction is resumable (finished cases are skipped). `--gate 0.5` etc. writes to `runs/.../gate0.5/`.

## What is computed

Maps per case and prompt (`scripts/extract.py`):
`csr_abs` (main), `csr_rel` (÷‖F(x)‖, controls for feature norm), `csr_cos` (direction only),
`spec` = ‖F(x,p) − F(x, "the cat")‖ (prompt-specific steering), `native` (released linear head),
`attn` (official CLS-attention heatmap), and prompt-free baselines: feature norm, salience,
centre prior.

| Research question | Test (per image, paired) | Output table |
|---|---|---|
| RQ1 spatial | AUROC vs 0.5, AUPRC/prevalence vs 1, CSS diff vs 0; vs mask-permutation null | `rq1` |
| RQ1 vs baselines | pathology CSR vs feature-norm / salience / centre maps | `baselines` |
| RQ2 semantic | Friedman (pathology, anatomy, irrelevant) + Holm-corrected Wilcoxon pairs incl. related-wrong, nuisance, class correct vs wrong | `rq2_friedman`, `rq2_pairs` |
| RQ2 directions | cosine of lesion-pooled residuals between prompt categories | `directions`, `*_directions.pdf` |
| RQ3 contrast | image-level AUROC: lesion vs healthy from steering magnitude; "the tumor" − "healthy tissue" / "no tumor"; large − small vs GT size; class A − class B vs diagnosis | `rq3`, `rq3_size`, `rq3_class`, `class_disc` |
| RQ4 size | Spearman of AUROC / lift with log lesion area; Kruskal over size tertiles; prompt benefit vs size | `size`, `*_size.pdf` |
| Exp. 5 | native vs CSR vs attention (paired), map correlation | `native_vs_csr`, `map_corr` |
| Paraphrase | Friedman over 3 wordings of the pathology prompt | `paraphrase` |

Metric definitions (`medsteer/metrics.py`): pixel AUROC/AUPRC on the bilinearly upsampled map;
patch AUROC/AUPRC with patch = lesion if ≥50% covered (fallback: touches the lesion, so
sub-patch lesions are still scored); CSS ratio and difference; Dice with a fixed per-map Otsu
threshold (main) and best threshold (`dice_oracle`, upper bound only).

## Things to keep in mind when writing the paper

- DINOv2 without registers has high-norm artifact patches; `sanity.py` reports the correlation of
  CSR with feature norm. If it is high, lead with `csr_rel` / `spec`.
- The patch grid is 24×24, so lesions below ~0.2% of the image cover less than one patch; report
  lesion size in patches (`lesion_patches` in `loc.csv`).
- BUSI contains near-duplicate images (no train/test leakage here, but n is inflated). Images are
  zero-padded to square, then resized; grayscale is replicated to RGB.
- RoBERTa saw no clinical text in SteerViT training: plain-language prompts are primary.

## Layout

```
configs/prompts.yaml   frozen prompt sets (categories, contrast pairs)
medsteer/              data, model wrapper, residuals, baselines, metrics, stats, plotting
scripts/               build_manifest, sanity, extract, evaluate, analyze, panels
slurm/                 Hendrix job scripts
tests/                 unit tests, fake model/data, smoke.sh
paper/                 main.tex, tables/, figs/
```
