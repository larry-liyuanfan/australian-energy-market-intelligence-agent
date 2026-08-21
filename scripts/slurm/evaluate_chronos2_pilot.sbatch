#!/bin/bash
#SBATCH --job-name=energy-chronos2-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20000
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the Spartan energy checkout}"
: "${ENERGY_GIT_COMMIT:?Set an exact pushed Git commit}"
: "${ENERGY_INPUT:?Set the verified AEMO transport file}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  : "${OUTPUT_ROOT:?Set a new array artifact root}"
  REGIONS=(NSW1 QLD1 SA1 TAS1 VIC1)
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#REGIONS[@]} )); then
    echo "invalid region array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
  fi
  export ENERGY_REGION="${REGIONS[SLURM_ARRAY_TASK_ID]}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${ENERGY_REGION}"
else
  : "${OUTPUT_DIR:?Set a new artifact directory}"
fi

git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-chronos2-${SLURM_JOB_ID:?}"
VENV="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-chronos2-venv-${SLURM_JOB_ID}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
cd "${CODE_ROOT}"
trap 'rm -rf -- "${CODE_ROOT}" "${VENV}"' EXIT

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "${VENV}"
source "${VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install --no-input --retries 5 --upgrade 'pip>=25.1,<26'
python -m pip install --no-input --retries 5 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1+cu121'
python -m pip install --no-input --retries 5 \
  --constraint scripts/chronos2-constraints.txt -e '.[ml,foundation]'

export ENERGY_GIT_COMMIT
export HF_HOME="${ARTIFACT_ROOT:-/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

python - <<'PY'
import torch
from chronos import Chronos2Pipeline

print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
assert tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 2)
assert torch.cuda.is_available()
assert Chronos2Pipeline is not None
PY

/usr/bin/time -v python scripts/evaluate_chronos2_bess.py \
  --input "${ENERGY_INPUT}" \
  --output "${OUTPUT_DIR}" \
  --region "${ENERGY_REGION:-SA1}" \
  --context-days "${CONTEXT_DAYS:-14}" \
  --test-days "${TEST_DAYS:-7}" \
  --model-id "${MODEL_ID:-amazon/chronos-2}" \
  --device cuda
