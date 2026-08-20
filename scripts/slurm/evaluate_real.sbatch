#!/bin/bash
# Submit with --dependency=afterok:<ingest-job> after full ingest passes.
#SBATCH --job-name=energy-real-eval
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# Array tasks must never share a log path. For a non-array pilot, Slurm uses its
# sentinel array-task value; the filename remains unique and auditable.
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%A_%a.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/real-eval}"
ENERGY_REGIONS="${ENERGY_REGIONS:-NSW1,QLD1,SA1,TAS1,VIC1}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  REGIONS=(NSW1 QLD1 SA1 TAS1 VIC1)
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#REGIONS[@]} )); then
    echo "unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
  fi
  ENERGY_REGIONS="${REGIONS[SLURM_ARRAY_TASK_ID]}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${ENERGY_REGIONS}}"
# Colons survive `sbatch --export`; commas are parsed by Slurm as variable separators.
DEGRADATION_COSTS="${DEGRADATION_COSTS:-0:25:50:100}"
EXTRA_ARGS=()
if [[ "${SEASONAL_BESS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--seasonal-bess)
fi
VENV="/tmp/energy-eval-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[ml]'
/usr/bin/time -v "${VENV}/bin/python" scripts/evaluate_real_market.py \
  --input /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired.csv.gz \
  --output "${OUTPUT_DIR}" \
  --regions "${ENERGY_REGIONS}" \
  --degradation-costs "${DEGRADATION_COSTS}" \
  "${EXTRA_ARGS[@]}"
