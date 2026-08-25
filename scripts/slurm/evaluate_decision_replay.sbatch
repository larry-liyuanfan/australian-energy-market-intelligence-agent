#!/bin/bash
#SBATCH --job-name=energy-decision-replay
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR}"
: "${ENERGY_GIT_COMMIT:?Set exact commit}"
: "${DATA_PATH:?Set corrected market data path}"
: "${DATA_MANIFEST:?Set market manifest path}"
: "${EVIDENCE_PATH:?Set official text evidence path}"
: "${FIGURE_RECORDS:?Set private figure records path}"
: "${OUTPUT_DIR:?Set a new output directory}"
git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="${TASK_SCRATCH}/energy-decision-${SLURM_JOB_ID:?}"
VENV="${TASK_SCRATCH}/energy-decision-venv-${SLURM_JOB_ID}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT
cd "${CODE_ROOT}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "${VENV}"
source "${VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install --no-input --retries 5 -e '.[search]'
mkdir -p "${OUTPUT_DIR}"
SNAPSHOT_ARGS=()
if [[ -n "${FORECAST_SNAPSHOTS:-}" && -f "${FORECAST_SNAPSHOTS}" ]]; then
  SNAPSHOT_ARGS=(--forecast-snapshots "${FORECAST_SNAPSHOTS}")
fi
/usr/bin/time -v python scripts/evaluate_decision_replay.py \
  --data "${DATA_PATH}" \
  --data-manifest "${DATA_MANIFEST}" \
  --evidence "${EVIDENCE_PATH}" \
  --figures "${FIGURE_RECORDS}" \
  "${SNAPSHOT_ARGS[@]}" \
  --output "${OUTPUT_DIR}"
