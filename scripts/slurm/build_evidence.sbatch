#!/bin/bash
#SBATCH --job-name=energy-evidence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
VENV="/tmp/energy-evidence-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${VENV}"' EXIT
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e '.[search]'
/usr/bin/time -v "${VENV}/bin/python" scripts/build_official_evidence.py \
  --output /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index
