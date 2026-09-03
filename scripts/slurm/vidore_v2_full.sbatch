#!/bin/bash
#SBATCH --job-name=energy-vidore-v2-full
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --tmp=100000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/vidore-full-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/vidore-full-%j.err

set -euo pipefail
: "${ENERGY_GIT_COMMIT:?Set the exact pushed commit}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2
REPO="$PROJECT/repo"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/energy-visual-goal-compiler-v2
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="$TASK_SCRATCH/energy-vidore-v2-full-$SLURM_JOB_ID"
RUN_DIR="$ARTIFACTS/private/vidore-full-$SLURM_JOB_ID"
ENV_ROOT="$TASK_SCRATCH/energy-vidore-v2-full-env-$SLURM_JOB_ID"
QWEN_REPO="$TASK_SCRATCH/Qwen3-VL-Embedding-393e2978"
MODEL_ROOT="$TASK_SCRATCH/energy-vidore-v2-full-models-$SLURM_JOB_ID"
export HF_HOME="$TASK_SCRATCH/energy-vidore-v2-full-hf-$SLURM_JOB_ID"
export HF_HUB_DISABLE_TELEMETRY=1
export MODEL_ROOT CODE_ROOT

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$RUN_DIR"
git -C "$REPO" cat-file -e "$ENERGY_GIT_COMMIT^{commit}"
git clone --shared --no-checkout "$REPO" "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach "$ENERGY_GIT_COMMIT"
trap 'rm -rf -- "$CODE_ROOT"' EXIT
module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "$ENV_ROOT/venv"
export PIP_NO_CACHE_DIR=1
"$ENV_ROOT/venv/bin/python" -m pip install \
  "numpy==1.26.4" "pillow==10.4.0" "torch==2.8.*" "torchvision==0.23.*" "transformers==4.57.3" \
  -e "$CODE_ROOT[visual-benchmark,multimodal,ml]"
module purge
module load GCC/13.3.0 CUDA/12.4.1
PYTHON_ROOT=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3
export LD_LIBRARY_PATH="$PYTHON_ROOT/lib:${LD_LIBRARY_PATH:-}"
git clone --filter=blob:none https://github.com/QwenLM/Qwen3-VL-Embedding.git "$QWEN_REPO"
git -C "$QWEN_REPO" checkout --detach 393e2978d27852b0d0230d6994f37f9c15bed73c
"$ENV_ROOT/venv/bin/python" -m pip install "qwen-vl-utils==0.0.14"
# The pinned Qwen package's video-only OpenCV dependency requires NumPy 2, while pinned ColQwen2 requires NumPy 1.
# This image-only benchmark imports the pinned source tree directly and installs only its runtime image dependency.
"$ENV_ROOT/venv/bin/python" -m pip install --no-deps -e "$QWEN_REPO"
"$ENV_ROOT/venv/bin/python" -m pip check
"$ENV_ROOT/venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
cfg=json.loads(Path(os.environ['CODE_ROOT'] + '/benchmarks/vidore_retrieval_v2.json').read_text())
for key in ('qwen_text','qwen_vl','colqwen'):
    item=cfg['models'][key]
    snapshot_download(repo_id=item['id'], revision=item['revision'], local_dir=Path(os.environ['MODEL_ROOT'])/key)
colqwen=cfg['models']['colqwen']
base=Path(os.environ['MODEL_ROOT'])/'colqwen_base'
snapshot_download(repo_id=colqwen['base_model_id'], revision=colqwen['base_model_revision'], local_dir=base)
adapter_path=Path(os.environ['MODEL_ROOT'])/'colqwen'/'adapter_config.json'
adapter=json.loads(adapter_path.read_text())
adapter['base_model_name_or_path']=str(base)
adapter_path.write_text(json.dumps(adapter, sort_keys=True))
PY
cd "$CODE_ROOT"
"$ENV_ROOT/venv/bin/python" scripts/evaluate_vidore_benchmarks.py \
  --config benchmarks/vidore_retrieval_v2.json --output "$RUN_DIR/evaluation" --qwen-repo "$QWEN_REPO" \
  --qwen-vl-model "$MODEL_ROOT/qwen_vl" --qwen-text-model "$MODEL_ROOT/qwen_text" \
  --colqwen-model "$MODEL_ROOT/colqwen"
"$ENV_ROOT/venv/bin/python" scripts/validate_v2_artifacts.py --vidore-run "$RUN_DIR/evaluation"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
