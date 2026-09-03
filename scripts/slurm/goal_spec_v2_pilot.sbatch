#!/bin/bash
#SBATCH --job-name=energy-goal-v2-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --tmp=20000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/goal-pilot-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/goal-pilot-%j.err

set -euo pipefail
: "${ENERGY_GIT_COMMIT:?Set the exact pushed commit}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2
REPO="$PROJECT/repo"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/energy-visual-goal-compiler-v2
MARKET=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired_v2.csv.gz
MARKET_MANIFEST=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/final_manifest_v2.json
EVIDENCE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl
SNAPSHOTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/forecast-snapshots-c66e415.jsonl
LLAMA_COMMIT=7798007a29a90e3053e799394da48cf53a2f8e0f
MODEL_SHA256=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
MODEL_URL='https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/1d54a16a18cba0d8fbad4a16db801decc729e099/Qwen3-8B-Q4_K_M.gguf?download=true'
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="$TASK_SCRATCH/energy-goal-v2-pilot-$SLURM_JOB_ID"
ENV_ROOT="$TASK_SCRATCH/energy-goal-v2-pilot-env-$SLURM_JOB_ID"
LLAMA_ROOT="$TASK_SCRATCH/llama.cpp-$LLAMA_COMMIT"
SERVER="$LLAMA_ROOT/build/bin/llama-server"
MODEL_ROOT="$TASK_SCRATCH/energy-goal-v2-pilot-model-$SLURM_JOB_ID"
MODEL="$MODEL_ROOT/Qwen3-8B-Q4_K_M.gguf"
RUN_DIR="$ARTIFACTS/private/goal-pilot-$SLURM_JOB_ID"
PORT=11601

mkdir -p "$MODEL_ROOT" "$RUN_DIR" "$ENV_ROOT"
git -C "$REPO" cat-file -e "$ENERGY_GIT_COMMIT^{commit}"
git clone --shared --no-checkout "$REPO" "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach "$ENERGY_GIT_COMMIT"
if [ ! -s "$MODEL" ]; then
  curl -fL --retry 3 --retry-delay 5 -o "$MODEL.partial" "$MODEL_URL"
  mv "$MODEL.partial" "$MODEL"
fi
echo "$MODEL_SHA256  $MODEL" | sha256sum --check

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "$ENV_ROOT/venv"
export PIP_NO_CACHE_DIR=1
"$ENV_ROOT/venv/bin/python" -m pip install -e "$CODE_ROOT[test,search,workbook]"
module purge
module load GCC/13.3.0 CMake/3.31.8 CUDA/12.4.1
PYTHON_ROOT=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3
export LD_LIBRARY_PATH="$PYTHON_ROOT/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
git -C "$LLAMA_ROOT" checkout --detach "$LLAMA_COMMIT"
test "$(git -C "$LLAMA_ROOT" rev-parse HEAD)" = "$LLAMA_COMMIT"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$LLAMA_ROOT/build" --target llama-server --parallel "$SLURM_CPUS_PER_TASK"
"$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx-size 16384 \
  --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none --no-webui \
  >"$PROJECT/logs/goal-pilot-server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf -- "$CODE_ROOT"' EXIT
for _ in $(seq 1 120); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

cd "$CODE_ROOT"
COMMON=(--benchmark benchmarks/goal_spec_development_v2.jsonl --gate benchmarks/goal_spec_promotion_gate_v2.json \
  --data "$MARKET" --data-manifest "$MARKET_MANIFEST" --evidence "$EVIDENCE" --forecast-snapshots "$SNAPSHOTS" \
  --max-episodes 2)
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/direct" --system direct_tool --system-label qwen3_8b_direct_tool \
  --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-8B-Q4_K_M.gguf --seeds 101
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/goal" --system goal_spec --system-label qwen3_8b_goal_spec \
  --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-8B-Q4_K_M.gguf --seeds 101
"$ENV_ROOT/venv/bin/python" scripts/validate_v2_artifacts.py \
  --goal-run "$RUN_DIR/direct" --goal-run "$RUN_DIR/goal"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
sha256sum "$SERVER" "$MODEL" >"$RUN_DIR/runtime_sha256.txt"
