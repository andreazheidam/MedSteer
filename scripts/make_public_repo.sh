#!/bin/bash
# Build a clean, shareable copy of the project in ~/MedSteer (fresh git history, no data, no
# weights, no caches, no personal paths) from the working repository ~/medsteer.
#   bash scripts/make_public_repo.sh [target_dir]        (run from ~/medsteer, login node)
set -euo pipefail
SRC=$(pwd)
DST=${1:-$HOME/MedSteer}
[ -f "$SRC/configs/prompts_final2.yaml" ] || { echo "run from ~/medsteer"; exit 1; }
[ -e "$DST" ] && { echo "$DST exists - remove it or pass another target"; exit 1; }
mkdir -p "$DST"/{results/tables,results/figures,results/data,results/runs,envs,docs}

# ---- code and configuration
for d in medsteer scripts configs slurm tests; do if [ -d "$d" ]; then cp -r "$d" "$DST"/; fi; done
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} + ; find "$DST" -name '*.pyc' -delete
cp README_public.md "$DST/README.md"
cp requirements.txt "$DST/envs/requirements-steervit-minimal.txt"
for f in README.md FINAL.md FINAL2.md; do
  if [ -f "$f" ]; then cp "$f" "$DST/docs/${f%.md}_working.md"; fi
done

# ---- exact environments
.venv/bin/pip freeze 2>/dev/null | grep -v '^-e ' > "$DST/envs/requirements-steervit.txt" || true
.venv-compare/bin/pip freeze 2>/dev/null | grep -v '^-e ' > "$DST/envs/requirements-compare.txt" || true
{ echo "# CUDA build of torch in .venv-compare (install with --index-url https://download.pytorch.org/whl/cu126)";
  .venv-compare/bin/python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" 2>/dev/null || true
} > "$DST/envs/NOTES.txt"

# ---- results: tables, figures, dataset description
cp paper_final2/tables/*.csv paper_final2/tables/*.tex "$DST/results/tables/" 2>/dev/null || true
cp paper_final2/data/* "$DST/results/data/" 2>/dev/null || true
for d in fig1_v2 fig1_auto fig_medsam3 fig1_final figs; do
  [ -d "paper_final2/$d" ] || continue
  mkdir -p "$DST/results/figures/$d"
  cp paper_final2/$d/*.{pdf,svg,png,txt} "$DST/results/figures/$d/" 2>/dev/null || true
done

# ---- per-run metadata and per-image metrics (no feature maps, no images)
copy_run () {   # $1 = run root, $2 = model name
  for ds in brisc busi kvasir isic; do
    r="$1/$ds"; [ -d "$r" ] || continue
    o="$DST/results/runs/$2/$ds"; mkdir -p "$o"
    if [ -f "$r/meta.json" ]; then sed "s#$HOME#~#g" "$r/meta.json" > "$o/meta.json"; fi
    if [ -f "$r/loc.csv" ]; then gzip -c "$r/loc.csv" > "$o/loc.csv.gz"; fi
    if [ -f "$r/size_bins.json" ]; then cp "$r/size_bins.json" "$o/"; fi
  done
}
copy_run runs_final2/gate1 steervit
copy_run runs_final2_compare/clip/gate1 clip
copy_run runs_final2_compare/biomedclip/gate1 biomedclip
copy_run runs_final2_compare/sam3/gate1 sam3
copy_run runs_final2_compare/medsam3/gate1 medsam3

# ---- .gitignore and safety checks
cat > "$DST/.gitignore" <<'EOF'
__pycache__/
*.pyc
/.venv*/
/.hf_cache/
/data/
/runs*/
/logs/
/paper_*/
/medsam3_lora/
*.npz
*.pt
*.pth
*.safetensors
EOF
echo "== files larger than 50 MB (GitHub rejects > 100 MB):"
find "$DST" -type f -size +50M -print
echo "== possible secrets or personal paths (should be empty):"
grep -rIl --exclude=make_public_repo.sh -e "hf_[A-Za-z0-9]\{20,\}" -e "/home/${USER:-$(whoami)}" -e "ghp_" "$DST" || true

cd "$DST"
git init -q && git symbolic-ref HEAD refs/heads/main
git add -A
git commit -q -m "MedSteer: code, configuration and results for the ISBI 2026 submission"
echo
echo "built $DST: $(git ls-files | wc -l) files, $(du -sh . | cut -f1)"
echo "next: create an empty GitHub repository named MedSteer, then"
echo "  cd $DST && git remote add origin <URL> && git push -u origin main"
