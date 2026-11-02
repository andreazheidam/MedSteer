#!/bin/bash
# CPU only. Metrics for one dataset, or the statistics + figures over all datasets.
#   sbatch slurm/evaluate.sh kvasir                 # metrics for one dataset (gate 1)
#   GATE=0.5 sbatch slurm/evaluate.sh kvasir        # ... for another gate value
#   sbatch slurm/evaluate.sh analyze                # tables + figures, all datasets present
#SBATCH --job-name=medsteer-eval
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
TARGET=${1:?usage: sbatch slurm/evaluate.sh <dataset|analyze>}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true   # Hendrix: same Python as the venv
source .venv/bin/activate
MANIFEST=${MANIFEST:-data/manifest_all.csv}
GATE=${GATE:-1}
RUN=${OUT:-runs}/gate${GATE}
PAPER=${PAPER:-paper_gate${GATE}}

if [ "$TARGET" = "analyze" ]; then
  RUNS=$(for d in brisc busi kvasir sunnybrook; do [ -f "$RUN/$d/loc.csv" ] && echo "$RUN/$d"; done)
  python scripts/analyze.py --runs $RUNS --out "$PAPER"
  python scripts/paper_figs.py --runs $RUNS --manifest "$MANIFEST" --out "$PAPER/figs"
else
  python scripts/evaluate.py --run "$RUN/$TARGET" --manifest "$MANIFEST" \
      --workers "${SLURM_CPUS_PER_TASK:-8}"
fi
