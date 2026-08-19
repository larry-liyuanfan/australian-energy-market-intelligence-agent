#!/bin/bash
# Submit with afterok dependencies on ingest and evidence build.
#SBATCH --job-name=energy-agent-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
VENV="/tmp/energy-agent-eval-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[search]'
/usr/bin/time -v "${VENV}/bin/python" scripts/evaluate_agent_real.py \
  --data /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired.csv.gz \
  --data-manifest /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/final_manifest.json \
  --evidence /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl \
  --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/agent-eval
