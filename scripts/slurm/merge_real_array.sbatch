#!/bin/bash
# Submit with --dependency=afterok:<array-job-id>.
#SBATCH --job-name=energy-real-merge
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
INPUT_ROOT="${INPUT_ROOT:?set INPUT_ROOT to the array output root}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the merged artifact directory}"
module purge
module load Python/3.11.3
PYTHONPATH=src python scripts/merge_real_market_array.py --input-root "${INPUT_ROOT}" --output "${OUTPUT_DIR}"
