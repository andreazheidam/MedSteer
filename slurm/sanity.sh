#!/bin/bash
# Phase-0 checks on a GPU (~5 min). Put any photo with a dog at data/dog.jpg first.
#   sbatch slurm/sanity.sh
#SBATCH --job-name=medsteer-sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true   # Hendrix: same Python as the venv
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
NAT=()
[ -f data/dog.jpg ] && NAT=(--natural-image data/dog.jpg --natural-prompt "the dog")
for ds in busi brisc; do
  python scripts/sanity.py --dataset $ds --out sanity/$ds "${NAT[@]}"
done
