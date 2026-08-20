#!/bin/bash
#SBATCH --job-name=energy-ensemble-merge
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${ENERGY_GIT_COMMIT:?set exact pushed commit}"
: "${INPUT_ROOT:?set array result root}"
: "${OUTPUT_DIR:?set merged output directory}"
SOURCE_REPO="${REPO_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent}"
git -C "${SOURCE_REPO}" fetch origin feat/production-retrieval-observability
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-code-${SLURM_JOB_ID:?}"
git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${ENERGY_GIT_COMMIT}" ]]
cd "${CODE_ROOT}"
VENV="${SLURM_TMPDIR:-/tmp}/energy-merge-venv-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}" "${CODE_ROOT}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e .
"${VENV}/bin/python" scripts/merge_dispatch_ensemble.py \
  --input-root "${INPUT_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --git-sha "${ENERGY_GIT_COMMIT}"
