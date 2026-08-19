#!/bin/bash
# Submit with --dependency=afterok:<ingest-job> after full ingest passes.
#SBATCH --job-name=energy-real-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
VENV="/tmp/energy-eval-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[ml]'
/usr/bin/time -v "${VENV}/bin/python" scripts/evaluate_real_market.py \
  --input /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired.csv.gz \
  --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/real-eval
