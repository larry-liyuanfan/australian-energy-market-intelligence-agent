#!/bin/bash
#SBATCH --job-name=energy-history-gate
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${SUMMARY_GIT_COMMIT:?set exact summary-code commit}"
: "${EVALUATION_GIT_COMMIT:?set exact evaluation commit recorded in source metrics}"
SOURCE_METRICS="${SOURCE_METRICS:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/history-transport-e7a3cff-merged/metrics.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/history-transport-gate-${EVALUATION_GIT_COMMIT:0:7}}"
REPO="/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent"
git -C "${REPO}" cat-file -e "${SUMMARY_GIT_COMMIT}^{commit}"
git -C "${REPO}" cat-file -e "${EVALUATION_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-/tmp}/energy-history-gate-code-${SLURM_JOB_ID:?}"
VENV="${SLURM_TMPDIR:-/tmp}/energy-history-gate-venv-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT
git clone --shared --no-checkout "${REPO}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${SUMMARY_GIT_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${SUMMARY_GIT_COMMIT}" ]]
export SUMMARY_GIT_COMMIT EVALUATION_GIT_COMMIT
cd "${CODE_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install 'numpy>=2,<3'
PYTHONPATH=src "${VENV}/bin/python" scripts/summarize_history_transport.py \
  --metrics "${SOURCE_METRICS}" \
  --output "${OUTPUT_DIR}"
