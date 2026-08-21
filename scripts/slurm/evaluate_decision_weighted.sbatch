#!/bin/bash
#SBATCH --job-name=energy-decision-weighted
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%A_%a.out
set -euo pipefail

SOURCE_REPO="${REPO_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent}"
ENERGY_GIT_COMMIT="${ENERGY_GIT_COMMIT:-$(git -C "${SOURCE_REPO}" rev-parse HEAD)}"
git -C "${SOURCE_REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-code-${SLURM_JOB_ID:?}"
git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
cd "${CODE_ROOT}"
export ENERGY_GIT_COMMIT

REGIONS=(NSW1 QLD1 SA1 TAS1 VIC1)
TASK_ID="${SLURM_ARRAY_TASK_ID:-2}"
if (( TASK_ID < 0 || TASK_ID >= ${#REGIONS[@]} )); then
  echo "unsupported array task: ${TASK_ID}" >&2
  exit 2
fi
REGION="${REGIONS[${TASK_ID}]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/decision-weighted}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${REGION}}"
VENV="${SLURM_TMPDIR:-/tmp}/energy-decision-venv-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}" "${CODE_ROOT}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[ml]'
/usr/bin/time -v "${VENV}/bin/python" scripts/evaluate_decision_weighted.py \
  --input /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired.csv.gz \
  --output "${OUTPUT_DIR}" \
  --regions "${REGION}"
