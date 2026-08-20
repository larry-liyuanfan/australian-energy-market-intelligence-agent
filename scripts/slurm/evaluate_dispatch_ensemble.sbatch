#!/bin/bash
#SBATCH --job-name=energy-dispatch-ensemble
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${ENERGY_GIT_COMMIT:?set exact pushed commit}"
SOURCE_REPO="${REPO_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent}"
git -C "${SOURCE_REPO}" fetch origin feat/production-retrieval-observability
git -C "${SOURCE_REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-code-${SLURM_JOB_ID:?}"
git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${ENERGY_GIT_COMMIT}" ]]
cd "${CODE_ROOT}"
export ENERGY_GIT_COMMIT

REGION="${REGION:-SA1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/dispatch-ensemble-${REGION}-${ENERGY_GIT_COMMIT:0:7}}"
VENV="${SLURM_TMPDIR:-/tmp}/energy-ensemble-venv-${SLURM_JOB_ID:?}"
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
