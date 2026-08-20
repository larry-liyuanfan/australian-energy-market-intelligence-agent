#!/bin/bash
# Submit only after the seven-day pilot and sbatch --test-only.
#SBATCH --job-name=energy-history-year
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${ENERGY_GIT_COMMIT:?set the exact pushed commit to execute}"
REMOTE_URL="${ENERGY_REMOTE_URL:-https://github.com/larry-liyuanfan/australian-energy-market-intelligence-agent.git}"
CODE_ROOT="${SLURM_TMPDIR:-/tmp}/energy-code-${SLURM_JOB_ID:?}"
VENV="${SLURM_TMPDIR:-/tmp}/energy-venv-${SLURM_JOB_ID:?}"
OUT="${OUTPUT_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-history-12m-${ENERGY_GIT_COMMIT:0:7}}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT

git clone --filter=blob:none --no-checkout "${REMOTE_URL}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${ENERGY_GIT_COMMIT}" ]]
cd "${CODE_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
rm -rf -- "${OUT}"
/usr/bin/time -v "${VENV}/bin/python" scripts/ingest_dispatch_year.py \
  --start 2024-08-18 --end 2025-08-17 --output "${OUT}"
"${VENV}/bin/python" scripts/validate_dispatch_coverage.py \
  --data "${OUT}/dispatch_features.csv.gz" --manifest "${OUT}/run_manifest.json"
