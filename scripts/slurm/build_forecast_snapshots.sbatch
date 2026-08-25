#!/bin/bash
#SBATCH --job-name=energy-forecast-snapshots
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR}"
: "${ENERGY_GIT_COMMIT:?Set exact commit}"
: "${DATA_PATH:?Set corrected market data path}"
: "${DATA_MANIFEST:?Set market manifest path}"
: "${SNAPSHOT_OUTPUT:?Set a new forecast snapshot JSONL path}"
git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="${TASK_SCRATCH}/energy-snapshots-${SLURM_JOB_ID:?}"
VENV="${TASK_SCRATCH}/energy-snapshots-venv-${SLURM_JOB_ID}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT
cd "${CODE_ROOT}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "${VENV}"
source "${VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install --no-input --retries 5 -e .
mkdir -p "$(dirname "${SNAPSHOT_OUTPUT}")"
/usr/bin/time -v python scripts/build_forecast_snapshots.py \
  --data "${DATA_PATH}" \
  --data-manifest "${DATA_MANIFEST}" \
  --days 2025-09-15,2025-12-15,2026-03-15 \
  --output "${SNAPSHOT_OUTPUT}"
