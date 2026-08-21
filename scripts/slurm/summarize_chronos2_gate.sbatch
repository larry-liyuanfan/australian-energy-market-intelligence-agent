#!/bin/bash
#SBATCH --job-name=energy-chronos2-gate
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the Spartan energy checkout}"
: "${ENERGY_GIT_COMMIT:?Set an exact pushed Git commit}"
: "${INPUT_ROOT:?Set the completed five-region root}"
: "${OUTPUT_DIR:?Set a new summary directory}"
: "${EXPECTED_INPUT_SHA:?Set the verified AEMO input hash}"

git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-chronos2-summary-${SLURM_JOB_ID:?}"
VENV="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-chronos2-summary-venv-${SLURM_JOB_ID}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
cd "${CODE_ROOT}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
mkdir -p "${OUTPUT_DIR}"
python scripts/summarize_chronos2_gate.py \
  --input-root "${INPUT_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --expected-git-sha "${ENERGY_GIT_COMMIT}" \
  --expected-input-sha "${EXPECTED_INPUT_SHA}"
