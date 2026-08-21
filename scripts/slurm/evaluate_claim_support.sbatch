#!/bin/bash
#SBATCH --job-name=energy-claim-support
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=20000
#SBATCH --time=00:10:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the Spartan energy checkout}"
: "${ENERGY_GIT_COMMIT:?Set an exact pushed Git commit}"
: "${ENERGY_EVIDENCE:?Set the verified official evidence JSONL}"
: "${OUTPUT_DIR:?Set a new artifact directory}"

git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="${TASK_SCRATCH}/energy-claim-support-${SLURM_JOB_ID:?}"
VENV="${TASK_SCRATCH}/energy-claim-support-venv-${SLURM_JOB_ID}"
HF_JOB_CACHE="${TASK_SCRATCH}/energy-minicheck-hf-${SLURM_JOB_ID}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
cd "${CODE_ROOT}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}" "${HF_JOB_CACHE}"' EXIT

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "${VENV}"
source "${VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install --no-input --retries 5 --upgrade 'pip>=25.1,<26'
python -m pip install --no-input --retries 5 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1+cu121'
python -m pip install --no-input --retries 5 -e '.[factcheck]'

export HF_HOME="${HF_JOB_CACHE}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

/usr/bin/time -v python scripts/evaluate_claim_support.py \
  --evidence "${ENERGY_EVIDENCE}" \
  --output "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 8
