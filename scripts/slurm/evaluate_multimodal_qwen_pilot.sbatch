#!/bin/bash
#SBATCH --job-name=energy-mm-qwen-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=30000
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR}"
: "${ENERGY_GIT_COMMIT:?Set exact commit}"
: "${PAGE_RECORDS:?Set private page records}"
: "${PAGE_IMAGE_DIR:?Set private page image directory}"
: "${OUTPUT_DIR:?Set new output directory}"
QWEN_CODE_REVISION="393e2978d27852b0d0230d6994f37f9c15bed73c"
MODEL_REVISION="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL_ID="Qwen/Qwen3-VL-Embedding-2B"
git -C "${REPO_DIR}" cat-file -e "${ENERGY_GIT_COMMIT}^{commit}"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="${TASK_SCRATCH}/energy-mm-${SLURM_JOB_ID:?}"
QWEN_ROOT="${TASK_SCRATCH}/qwen3-vl-embedding-${SLURM_JOB_ID}"
VENV="${TASK_SCRATCH}/energy-mm-venv-${SLURM_JOB_ID}"
MODEL_DIR="${TASK_SCRATCH}/Qwen3-VL-Embedding-2B-${MODEL_REVISION}"
git clone --shared --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"
git clone --filter=blob:none https://github.com/QwenLM/Qwen3-VL-Embedding.git "${QWEN_ROOT}"
git -C "${QWEN_ROOT}" checkout --detach "${QWEN_CODE_REVISION}"
cd "${CODE_ROOT}"
trap 'rm -rf -- "${CODE_ROOT}" "${QWEN_ROOT}" "${VENV}"' EXIT
module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "${VENV}"
source "${VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
export HF_HOME="${TASK_SCRATCH}/qwen3-vl-hf-${SLURM_JOB_ID}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
python -m pip install --no-input --retries 5 --upgrade 'pip>=25.1,<26'
python -m pip install --no-input --retries 5 -e '.[multimodal,ml]'
python -m pip install --no-input --retries 5 -e "${QWEN_ROOT}"
mkdir -p "${HF_HOME}" "${MODEL_DIR}" "${OUTPUT_DIR}"
export MODEL_DIR
python -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id="Qwen/Qwen3-VL-Embedding-2B", revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda", local_dir=os.environ["MODEL_DIR"])'
/usr/bin/time -v python scripts/encode_qwen3_vl_pages.py \
  --records "${PAGE_RECORDS}" \
  --benchmark benchmarks/multimodal_page_retrieval_q4_2024.jsonl \
  --image-dir "${PAGE_IMAGE_DIR}" \
  --qwen-repo "${QWEN_ROOT}" \
  --model "${MODEL_DIR}" \
  --model-revision "${MODEL_REVISION}" \
  --code-revision "${QWEN_CODE_REVISION}" \
  --output "${OUTPUT_DIR}" \
  --batch-size 1 \
  --limit-pages 12 \
  --limit-queries 4
python scripts/evaluate_multimodal_retrieval.py \
  --records "${PAGE_RECORDS}" \
  --benchmark benchmarks/multimodal_page_retrieval_q4_2024.jsonl \
  --embedding-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/evaluation.json"
