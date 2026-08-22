#!/bin/bash
#SBATCH --job-name=energy-claim-final
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=20000
#SBATCH --time=00:06:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR}"
: "${ENERGY_GIT_COMMIT:?Set exact commit}"
: "${ENERGY_EVIDENCE:?Set isolated evidence}"
: "${OUTPUT_DIR:?Set new output}"
git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="${TASK_SCRATCH}/energy-claim-final-${SLURM_JOB_ID:?}"
VENV="${TASK_SCRATCH}/energy-claim-final-venv-${SLURM_JOB_ID}"
HF_JOB_CACHE="${TASK_SCRATCH}/energy-minicheck-final-hf-${SLURM_JOB_ID}"
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
python -m pip install --no-input --retries 5 --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118 'torch==2.6.0+cu118'
python -m pip install --no-input --retries 5 -e '.[factcheck]'
export HF_HOME="${HF_JOB_CACHE}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/cascade"
/usr/bin/time -v python scripts/evaluate_claim_support.py \
  --evidence "${ENERGY_EVIDENCE}" --benchmark benchmarks/minicheck_claim_support_holdout_q4_2024.jsonl \
  --output "${OUTPUT_DIR}/raw" --verifier deberta-v3-large \
  --model-id lytang/MiniCheck-DeBERTa-v3-Large \
  --model-revision 2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb --device cuda --batch-size 8
python scripts/evaluate_literal_cascade.py --evidence "${ENERGY_EVIDENCE}" \
  --predictions "${OUTPUT_DIR}/raw/predictions.jsonl" --output "${OUTPUT_DIR}/cascade" \
  --threshold 0.25 --fail-on-gate
