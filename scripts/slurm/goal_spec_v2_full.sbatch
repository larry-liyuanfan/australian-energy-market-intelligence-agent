#!/bin/bash
#SBATCH --job-name=energy-goal-v2-full
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --tmp=60000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/goal-full-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/goal-full-%j.err

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
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="$TASK_SCRATCH/energy-goal-v2-full-$SLURM_JOB_ID"
ENV_ROOT="$TASK_SCRATCH/energy-goal-v2-full-env-$SLURM_JOB_ID"
LLAMA_ROOT="$TASK_SCRATCH/llama.cpp-$LLAMA_COMMIT"
SERVER="$LLAMA_ROOT/build/bin/llama-server"
MODEL_ROOT="$TASK_SCRATCH/energy-goal-v2-full-models-$SLURM_JOB_ID"
MODEL8="$MODEL_ROOT/Qwen3-8B-Q4_K_M.gguf"
MODEL14="$MODEL_ROOT/Qwen3-14B-Q4_K_M.gguf"
RUN_DIR="$ARTIFACTS/private/goal-full-$SLURM_JOB_ID"
PORT=11602
server_pid=''

mkdir -p "$MODEL_ROOT" "$RUN_DIR"
git -C "$REPO" cat-file -e "$ENERGY_GIT_COMMIT^{commit}"
git clone --shared --no-checkout "$REPO" "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach "$ENERGY_GIT_COMMIT"
stage_model() {
  local path="$1" url="$2" digest="$3"
  if [ ! -s "$path" ]; then curl -fL --retry 3 --retry-delay 5 -o "$path.partial" "$url"; mv "$path.partial" "$path"; fi
  echo "$digest  $path" | sha256sum --check
}
stage_model "$MODEL8" 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/1d54a16a18cba0d8fbad4a16db801decc729e099/Qwen3-8B-Q4_K_M.gguf?download=true' d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
stage_model "$MODEL14" 'https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/530227a7d994db8eca5ab5ced2fb692b614357fd/Qwen3-14B-Q4_K_M.gguf?download=true' 500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "$ENV_ROOT/venv"
export PIP_NO_CACHE_DIR=1
"$ENV_ROOT/venv/bin/python" -m pip install -e "$CODE_ROOT[test,search,workbook]"
module purge
module load GCC/13.3.0 CMake/3.31.8 CUDA/12.4.1
PYTHON_ROOT=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3
export LD_LIBRARY_PATH="$PYTHON_ROOT/lib:${LD_LIBRARY_PATH:-}"
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
git -C "$LLAMA_ROOT" checkout --detach "$LLAMA_COMMIT"
test "$(git -C "$LLAMA_ROOT" rev-parse HEAD)" = "$LLAMA_COMMIT"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$LLAMA_ROOT/build" --target llama-server --parallel "$SLURM_CPUS_PER_TASK"
cleanup() { if [ -n "$server_pid" ]; then kill "$server_pid" 2>/dev/null || true; fi; rm -rf -- "$CODE_ROOT"; }
trap cleanup EXIT
start_server() {
  local model="$1" label="$2"
  "$SERVER" --model "$model" --host 127.0.0.1 --port "$PORT" --ctx-size 16384 \
    --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none --no-webui \
    >"$PROJECT/logs/$label-server-$SLURM_JOB_ID.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 180); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
}
stop_server() { kill "$server_pid"; wait "$server_pid" || true; server_pid=''; }

cd "$CODE_ROOT"
COMMON=(--benchmark benchmarks/goal_spec_holdout_v2.jsonl --gate benchmarks/goal_spec_promotion_gate_v2.json \
  --data "$MARKET" --data-manifest "$MARKET_MANIFEST" --evidence "$EVIDENCE" --forecast-snapshots "$SNAPSHOTS")
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/deterministic" --system deterministic --system-label deterministic
start_server "$MODEL8" qwen3-8b
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/qwen3-8b-direct" --system direct_tool --system-label qwen3_8b_direct_tool \
  --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-8B-Q4_K_M.gguf
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/qwen3-8b-goal" --system goal_spec --system-label qwen3_8b_goal_spec \
  --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-8B-Q4_K_M.gguf
stop_server
start_server "$MODEL14" qwen3-14b
"$ENV_ROOT/venv/bin/python" scripts/evaluate_goal_spec_agent.py "${COMMON[@]}" \
  --output "$RUN_DIR/qwen3-14b-goal" --system goal_spec --system-label qwen3_14b_goal_spec \
  --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-14B-Q4_K_M.gguf
stop_server
"$ENV_ROOT/venv/bin/python" scripts/aggregate_goal_spec_v2.py \
  --gate benchmarks/goal_spec_promotion_gate_v2.json --benchmark benchmarks/goal_spec_holdout_v2.jsonl \
  --run "deterministic=$RUN_DIR/deterministic" --run "qwen3_8b_direct_tool=$RUN_DIR/qwen3-8b-direct" \
  --run "qwen3_8b_goal_spec=$RUN_DIR/qwen3-8b-goal" --run "qwen3_14b_goal_spec=$RUN_DIR/qwen3-14b-goal" \
  --output "$RUN_DIR/aggregate.json"
"$ENV_ROOT/venv/bin/python" scripts/validate_v2_artifacts.py \
  --goal-run "$RUN_DIR/deterministic" --goal-run "$RUN_DIR/qwen3-8b-direct" \
  --goal-run "$RUN_DIR/qwen3-8b-goal" --goal-run "$RUN_DIR/qwen3-14b-goal" \
  --goal-aggregate "$RUN_DIR/aggregate.json"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
sha256sum "$SERVER" "$MODEL8" "$MODEL14" >"$RUN_DIR/runtime_sha256.txt"
