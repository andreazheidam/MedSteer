# MedSteer: characterizing medical semantic steering in vision models

Code, configuration and results for

> A. Z. Heidam and M. Mehdipour Ghazi, *Characterizing Medical Semantic Steering in Vision Models*, ISBI 2026 (submitted).

We ask whether text-conditioned localization on medical images reflects the **requested medical
concept** or only the **salient lesion**. Five models are queried with one target prompt and four
controls on four public lesion-segmentation datasets, and every map is scored against the
ground-truth mask. Nothing is trained: all models are used as released.

| Model | Role | Prompt input | Localization map |
|---|---|---|---|
| SteerViT (DINOv2-B/14) | steerable visual representation | text via gated cross-attention | released linear localization head (logits) |
| CLIP ViT-B/16 (OpenAI) | contrastive vision-language baseline | text embedding | dense patch-text cosine (MaskCLIP-style) |
| BiomedCLIP (PMC-15M) | medical contrastive vision-language model | text embedding (PubMedBERT) | dense patch-text cosine (MaskCLIP-style) |
| SAM3 (`facebook/sam3`) | concept-prompted segmentation | text | `semantic_seg` logits |
| MedSAM3 (SAM3 + medical LoRA v1) | medical concept-prompted segmentation | text | `semantic_seg` logits |

## Datasets and prompts

BRISC 2025 (brain MRI, official test split), BUSI (breast ultrasound, all images), Kvasir-SEG
(colonoscopy, all images), ISIC 2018 Task 1 (dermoscopy, official test split). The datasets are
**not** redistributed here; download them from their original sources and build the manifest
(below). Exact counts, sizes and preprocessing: `results/data/`.

Prompts are frozen in `configs/prompts_final2.yaml` (its hash is stored with every run) and sent
verbatim to every model:

| Dataset | Target | Context | Non-medical | Wrong-medical | Counterfactual |
|---|---|---|---|---|---|
| BRISC | brain tumor | brain tissue | cat | skin lesion | normal brain tissue |
| BUSI | breast lesion | breast tissue | cat | brain tumor | normal breast tissue |
| Kvasir-SEG | colon polyp | colon mucosa | cat | breast lesion | normal colon mucosa |
| ISIC | skin lesion | skin | cat | colon polyp | normal skin |

Baseline: SteerViT/DINOv2 without text input (`texts=None`); CLIP, SAM3 and MedSAM3 with the
empty string (*empty-text control*), since these models have no text-free path.

## Method in short

* **Geometry (all models):** zero-pad to square, bicubic resize to each model's input size; masks
  padded identically, nearest-neighbour resize to 336 px; all maps area-pooled to 24×24 patches.
* **Evaluation region:** all pixels of the original, unpadded image (only our padding excluded),
  computed from the image size alone, independent of the ground truth (`medsteer/data.py:content_region`).
* **Localization:** per-image pixel AUROC, AUPRC lift L = AUPRC / prevalence, pointing game.
* **Prompt specificity:** per image, against each control x separately:
  G_x = AUROC_target − AUROC_x and R_x = log2(AUPRC_target / AUPRC_x); median + bootstrap 95% CI.
* **SteerViT decomposition:** SRM = ‖ΔF‖ with ΔF = F(x,p) − F(x); DSE = wᵀΔF = h(F(x,p)) − h(F(x)),
  the prompt-induced change of the head logits. The identity is checked numerically for every
  image, patch and prompt during extraction (max. abs. error 5.7·10⁻⁶; `results/runs/*/meta.json`).

## Main results (median per image; full tables in `results/tables/`)

Target-prompt pixel AUROC:

| Map | BRISC | BUSI | Kvasir-SEG | ISIC |
|---|---|---|---|---|
| SteerViT map | 0.984 | 0.986 | 0.989 | 0.982 |
| SteerViT DSE | 0.829 | 0.872 | 0.855 | 0.718 |
| SteerViT SRM | 0.050 | 0.096 | 0.178 | 0.050 |
| CLIP map | 0.540 | 0.379 | 0.532 | 0.691 |
| BiomedCLIP map | 0.282 | 0.406 | 0.518 | 0.421 |
| SAM3 map | 0.736 | 0.977 | 0.979 | 0.898 |
| MedSAM3 map | 0.994 | 0.998 | 0.999 | 0.999 |

* High localization does not imply prompt specificity: SteerViT reaches AUROC ≥ 0.96 for nearly
  every control prompt. On AUPRC, BRISC prefers the target over all four controls, Kvasir-SEG
  mainly over the wrong-medical prompt, BUSI and ISIC show essentially none
  (`results/tables/specificity_gains_auprc_wide.csv`).
* SAM3 follows the named organ or tissue ("brain tumor" selects the brain); absent concepts fall back to the salient lesion.
* MedSAM3 localizes best but is essentially prompt-invariant (specificity gaps ≈ 0). Its training
  data likely include these public datasets.
* CLIP and BiomedCLIP do not localize the lesion (target-prompt AUROC 0.28-0.69). BiomedCLIP
     shows larger target-versus-control gaps than CLIP, but because its control maps are inverted (they highlight the background), not because the target map finds the lesion.

## Repository layout

```
medsteer/              data loading, evaluation region, metrics, SteerViT wrapper, residual maps
scripts/
  make_final_manifest.py, build_manifest.py   dataset manifest (paths, splits, masks)
  describe_data.py                            dataset and preprocessing tables
  extract.py                                  SteerViT extraction (maps, SRM, DSE check)
  extract_compare.py                          CLIP and SAM3 extraction (transformers)
  extract_medsam3.py                          MedSAM3 extraction (MedSAM3 code + LoRA)
  evaluate.py                                 per-image metrics -> loc.csv / global.csv / direction.csv
  compare_table.py                            all tables (localization, G_x, R_x, paired)
  paper_fig1.py, compare_panels.py            figures
  final_checks.py                             pre-submission checks (config hash, coverage, region, DSE)
configs/prompts_final2.yaml                   frozen prompts of the paper
slurm/                                        job scripts used on the Hendrix cluster
envs/                                         exact package versions of both environments
results/
  tables/      all result tables (.csv, .tex)
  figures/     Fig. 1 and per-dataset case figures (.pdf, .svg, .png)
  data/        dataset description and preprocessing tables
  runs/<model>/<dataset>/  meta.json (prompts, config hash, checks) and loc.csv.gz (per-image metrics)
docs/          notes from earlier project phases
```

## Reproducing

Two Python 3.11 environments (exact versions in `envs/`):

```bash
# SteerViT
python -m venv .venv && .venv/bin/pip install -r envs/requirements-steervit.txt
# CLIP, SAM3, MedSAM3 (torch built for CUDA 12.6)
python -m venv .venv-compare && .venv-compare/bin/pip install -r envs/requirements-compare.txt
```

```bash
python scripts/make_final_manifest.py --isic /path/to/ISIC2018 --base data/manifest_all.csv --out data/manifest_final.csv
sbatch slurm/final.sh                       # SteerViT (0-3), CLIP (4-7), SAM3 (8-11)
git clone https://github.com/Joey-S-Liu/MedSAM3 ~/MedSAM3       # MedSAM3 code
hf download lal-Joey/MedSAM3_v1 --local-dir medsam3_lora        # MedSAM3 LoRA weights
sbatch slurm/biomedclip.sh                  # BiomedCLIP
sbatch --export=ALL,MEDSAM3_LORA=$PWD/medsam3_lora/best_lora_weights.pt slurm/medsam3.sh
sbatch slurm/eval_final.sh                  # metrics (MedSAM3 jobs evaluate themselves)
python scripts/compare_table.py             # tables -> paper_final2/tables
python scripts/paper_fig1.py --auto         # Fig. 1
python scripts/final_checks.py
```

SAM3 requires access to the gated `facebook/sam3` checkpoint on Hugging Face.

## Status

This repository accompanies the paper and is being extended (further documentation and analyses).
