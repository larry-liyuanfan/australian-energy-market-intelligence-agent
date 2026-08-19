#!/bin/bash
#SBATCH --job-name=energy-finalize-manifest
#SBATCH --partition=cascade
#SBATCH --time=00:05:00
#SBATCH --mem=512M
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0
artifact_root=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m
python scripts/finalize_ingest_manifest.py \
  --manifest "$artifact_root/final_manifest.json" \
  --coverage "$artifact_root/coverage_validation.json" \
  --data "$artifact_root/dispatch_features_repaired.csv.gz"
