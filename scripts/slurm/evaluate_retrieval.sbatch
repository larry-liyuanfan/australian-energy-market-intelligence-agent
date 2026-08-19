#!/bin/bash
#SBATCH --job-name=energy-retrieval-eval
#SBATCH --partition=cascade
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate
python scripts/evaluate_retrieval.py \
  --evidence /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl \
  --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/retrieval-eval
