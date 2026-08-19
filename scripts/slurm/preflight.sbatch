#!/bin/bash
#SBATCH --job-name=energy-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out
set -euo pipefail
cd /data/gpfs/projects/punim2936/portfolio_20260820/energy-agent
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
ENERGY_VENV="${SLURM_TMPDIR:?}/energy-venv"
python -m venv "${ENERGY_VENV}"
"${ENERGY_VENV}/bin/python" -m pip install -e '.[test]'
/usr/bin/time -v "${ENERGY_VENV}/bin/python" -m pytest -q
"${ENERGY_VENV}/bin/ruff" check .
"${ENERGY_VENV}/bin/mypy" src
