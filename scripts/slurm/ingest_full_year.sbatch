#!/bin/bash
# Submit only after pilot-derived resource update and sbatch --test-only.
#SBATCH --job-name=energy-ingest-year
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
OUT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m
rm -rf -- "${OUT}"
/usr/bin/time -v python scripts/ingest_dispatch_year.py \
  --start 2025-08-18 --end 2026-08-17 --output "${OUT}"
