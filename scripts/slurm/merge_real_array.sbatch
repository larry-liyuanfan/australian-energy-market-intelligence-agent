#!/bin/bash
# Submit with --dependency=afterok:<array-job-id>.
#SBATCH --job-name=energy-real-merge
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
SOURCE_REPO="${REPO_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent}"
ENERGY_GIT_COMMIT="${ENERGY_GIT_COMMIT:-$(git -C "${SOURCE_REPO}" rev-parse HEAD)}"
git -C "${SOURCE_REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
export ENERGY_GIT_COMMIT
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-merge-code-${SLURM_JOB_ID:?}"
git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
trap 'rm -rf -- "${CODE_ROOT}"' EXIT
cd "${CODE_ROOT}"
INPUT_ROOT="${INPUT_ROOT:?set INPUT_ROOT to the array output root}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the merged artifact directory}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
PYTHONPATH=src python scripts/merge_real_market_array.py --input-root "${INPUT_ROOT}" --output "${OUTPUT_DIR}"
