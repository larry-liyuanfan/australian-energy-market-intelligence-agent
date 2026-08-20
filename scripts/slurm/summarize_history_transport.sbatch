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

: "${ENERGY_GIT_COMMIT:?set exact summary commit}"
SOURCE_METRICS="${SOURCE_METRICS:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/history-transport-e7a3cff-merged/metrics.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/history-transport-gate-${ENERGY_GIT_COMMIT:0:7}}"
REPO="/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent"
git -C "${REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
export ENERGY_GIT_COMMIT
cd "${REPO}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
PYTHONPATH=src python scripts/summarize_history_transport.py \
  --metrics "${SOURCE_METRICS}" \
  --output "${OUTPUT_DIR}"
