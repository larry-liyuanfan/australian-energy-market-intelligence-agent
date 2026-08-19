#!/bin/bash
# Submit with --dependency=afterok:<ingest-job>.
#SBATCH --job-name=energy-ingest-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
OUT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m
python scripts/repair_dispatch_gap.py \
  --input "${OUT}/dispatch_features.csv.gz" \
  --output "${OUT}/dispatch_features_repaired.csv.gz" \
  --day 2026-03-10
python scripts/validate_dispatch_coverage.py \
  --data "${OUT}/dispatch_features_repaired.csv.gz" --manifest "${OUT}/run_manifest.json"
