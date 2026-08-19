#!/bin/bash
#SBATCH --job-name=energy-ingest-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
OUT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-pilot
rm -rf -- "${OUT}"
/usr/bin/time -v python scripts/ingest_dispatch_year.py \
  --start 2026-08-15 --end 2026-08-17 --output "${OUT}"

