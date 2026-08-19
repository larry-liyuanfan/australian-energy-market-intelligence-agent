#!/bin/bash
#SBATCH --job-name=energy-retrieval-eval
#SBATCH --partition=sapphire
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.err

set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
VENV="/tmp/energy-retrieval-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}"' EXIT
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[search]'
"${VENV}/bin/python" scripts/evaluate_retrieval.py \
  --evidence /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl \
  --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/retrieval-eval
