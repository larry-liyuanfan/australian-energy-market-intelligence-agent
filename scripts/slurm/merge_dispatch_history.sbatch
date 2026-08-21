#!/bin/bash
#SBATCH --job-name=energy-history-merge
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${ENERGY_GIT_COMMIT:?set exact commit}"
: "${HISTORY_INPUT:?set completed monthly MMSDM history file}"
CURRENT_INPUT="${CURRENT_INPUT:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired.csv.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-24m-${ENERGY_GIT_COMMIT:0:7}}"
REPO="/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent"
git -C "${REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
cd "${REPO}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
/usr/bin/time -v python scripts/merge_dispatch_periods.py \
  --input "${HISTORY_INPUT}" \
  --input "${CURRENT_INPUT}" \
  --start 2024-08-18 \
  --end 2026-08-17 \
  --output "${OUTPUT_DIR}"
