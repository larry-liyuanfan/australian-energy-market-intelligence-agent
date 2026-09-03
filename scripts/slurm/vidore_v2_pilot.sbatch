#!/bin/bash
#SBATCH --job-name=energy-vidore-v2-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=50000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/vidore-pilot-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/vidore-pilot-%j.err

set -euo pipefail
: "${ENERGY_GIT_COMMIT:?Set the exact pushed commit}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/energy-visual-goal-compiler-v2
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="$TASK_SCRATCH/energy-vidore-v2-pilot-$SLURM_JOB_ID"
RUN_DIR="$ARTIFACTS/private/vidore-pilot-$SLURM_JOB_ID"
QWEN_REPO="$ENV_ROOT/Qwen3-VL-Embedding-393e2978"
MODEL_ROOT="$PROJECT/models"
export HF_HOME="$PROJECT/hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export CODE_ROOT MODEL_ROOT

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$RUN_DIR"
git -C "$REPO" cat-file -e "$ENERGY_GIT_COMMIT^{commit}"
git clone --shared --no-checkout "$REPO" "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach "$ENERGY_GIT_COMMIT"
trap 'rm -rf -- "$CODE_ROOT"' EXIT
module purge
module load GCC/13.3.0 CUDA/12.4.1
"$ENV_ROOT/venv/bin/python" -m pip install 'datasets>=3,<5' 'sentence-transformers>=5,<6' 'colpali-engine==0.3.1'
if [ ! -d "$QWEN_REPO/.git" ]; then git clone --filter=blob:none https://github.com/QwenLM/Qwen3-VL-Embedding.git "$QWEN_REPO"; fi
git -C "$QWEN_REPO" checkout --detach 393e2978d27852b0d0230d6994f37f9c15bed73c
"$ENV_ROOT/venv/bin/python" -m pip install -e "$QWEN_REPO"
export MODEL_ROOT
"$ENV_ROOT/venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
cfg=json.loads(Path(os.environ['CODE_ROOT'] + '/benchmarks/vidore_retrieval_v2.json').read_text())
for key in ('qwen_text','qwen_vl','colqwen'):
    item=cfg['models'][key]
    snapshot_download(repo_id=item['id'], revision=item['revision'], local_dir=Path(os.environ['MODEL_ROOT'])/key)
PY
cd "$CODE_ROOT"
"$ENV_ROOT/venv/bin/python" scripts/evaluate_vidore_benchmarks.py \
  --config benchmarks/vidore_retrieval_v2.json --output "$RUN_DIR/evaluation" --qwen-repo "$QWEN_REPO" \
  --qwen-vl-model "$MODEL_ROOT/qwen_vl" --qwen-text-model "$MODEL_ROOT/qwen_text" \
  --colqwen-model "$MODEL_ROOT/colqwen" --limit 8
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
