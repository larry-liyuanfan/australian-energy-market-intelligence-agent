#!/bin/bash
#SBATCH --job-name=energy-time-axis-v2
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail

: "${ENERGY_GIT_COMMIT:?set exact commit}"
REPO="/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent"
ROOT="/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m"
git -C "${REPO}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
cd "${REPO}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python scripts/repair_dispatch_gap.py \
  --input "${ROOT}/dispatch_features.csv.gz" \
  --output "${ROOT}/dispatch_features_repaired_v2.csv.gz" \
  --day 2026-03-10 \
  --repair-manifest "${ROOT}/repair_manifest_v2.json" \
  --final-manifest "${ROOT}/final_manifest_v2.json"
python scripts/validate_dispatch_coverage.py \
  --data "${ROOT}/dispatch_features_repaired_v2.csv.gz" \
  --manifest "${ROOT}/final_manifest_v2.json" \
  --output "${ROOT}/coverage_validation_v2.json"
