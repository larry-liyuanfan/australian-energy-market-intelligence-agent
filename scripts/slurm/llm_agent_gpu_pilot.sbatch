#!/bin/bash
#SBATCH --job-name=energy-llm-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --tmp=10000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/pilot-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/pilot-%j.err

set -euo pipefail
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/llm-agent-eval-20260903
MARKET=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired_v2.csv.gz
MARKET_MANIFEST=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/final_manifest_v2.json
EVIDENCE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl
SNAPSHOTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/forecast-snapshots-c66e415.jsonl
LLAMA_COMMIT=7798007a29a90e3053e799394da48cf53a2f8e0f
LLAMA_ROOT="$ENV_ROOT/llama.cpp-$LLAMA_COMMIT"
SERVER="$LLAMA_ROOT/build/bin/llama-server"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
MODEL_DIR="$TASK_SCRATCH/qwen3-8b-gguf-1d54a16-$SLURM_JOB_ID"
MODEL="$MODEL_DIR/Qwen3-8B-Q4_K_M.gguf"
MODEL_SHA256=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
MODEL_URL='https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/1d54a16a18cba0d8fbad4a16db801decc729e099/Qwen3-8B-Q4_K_M.gguf?download=true'
PORT=11571
RUN_DIR="$ARTIFACTS/private/pilot-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ENV_ROOT/xdg-cache"

mkdir -p "$RUN_DIR" "$MODEL_DIR" "$XDG_CACHE_HOME"
for path in "$MARKET" "$MARKET_MANIFEST" "$EVIDENCE" "$SNAPSHOTS"; do
  test -s "$path"
done

module purge
module load GCC/13.3.0
module load CMake/3.31.8
module load CUDA/12.4.1
PYTHON_ROOT=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3
test -s "$PYTHON_ROOT/lib/libpython3.11.so.1.0"
export LD_LIBRARY_PATH="$PYTHON_ROOT/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ! -d "$LLAMA_ROOT/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
  git -C "$LLAMA_ROOT" checkout --detach "$LLAMA_COMMIT"
fi
test "$(git -C "$LLAMA_ROOT" rev-parse HEAD)" = "$LLAMA_COMMIT"
if [ ! -x "$SERVER" ]; then
  cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
    -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON
  cmake --build "$LLAMA_ROOT/build" --target llama-server --parallel "$SLURM_CPUS_PER_TASK"
fi

curl -fL --retry 3 --retry-delay 5 -o "$MODEL.partial" "$MODEL_URL"
mv "$MODEL.partial" "$MODEL"
echo "$MODEL_SHA256  $MODEL" | sha256sum --check
sha256sum "$SERVER" "$MODEL" >"$RUN_DIR/runtime_sha256.txt"

"$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size 12288 --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none --no-webui \
  >"$PROJECT/logs/llama-cpp-pilot-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf -- "$MODEL_DIR"' EXIT
for _ in $(seq 1 90); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

cd "$REPO"
"$ENV_ROOT/venv/bin/python" scripts/smoke_llm_agent.py \
  --provider llama_cpp \
  --provider-url "http://127.0.0.1:$PORT/v1" \
  --model Qwen3-8B-Q4_K_M.gguf \
  --output "$RUN_DIR/smoke"
"$ENV_ROOT/venv/bin/python" scripts/evaluate_llm_agent.py \
  --benchmark benchmarks/llm_agent_development_v1.jsonl \
  --gate benchmarks/llm_agent_promotion_gate_v1.json \
  --output "$RUN_DIR/development" \
  --data "$MARKET" \
  --data-manifest "$MARKET_MANIFEST" \
  --evidence "$EVIDENCE" \
  --forecast-snapshots "$SNAPSHOTS" \
  --provider llama_cpp \
  --provider-url "http://127.0.0.1:$PORT/v1" \
  --model Qwen3-8B-Q4_K_M.gguf \
  --temperature 0.2 \
  --paths pure_llm constrained_hybrid \
  --memory-modes structured_state \
  --seeds 17 29
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
