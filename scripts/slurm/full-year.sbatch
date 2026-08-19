#!/bin/bash
# Submit only after measured preflight and `sbatch --test-only`.
#SBATCH --job-name=energy-full-year
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
/usr/bin/time -v python scripts/run_real_pipeline.py --months 12 --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/full-year

