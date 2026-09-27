#!/bin/bash
#SBATCH --job-name=energy-p0-holdout
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --tmp=10000
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/holdout-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/holdout-%j.err
set -euo pipefail
umask 077
: "${ENERGY_GIT_COMMIT:?exact frozen commit required}"
: "${ENERGY_PILOT_COMMIT:?exact accepted direct-tool pilot commit required}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927
ROOT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
RUN="$ROOT/planner-remediation-20260927/private/holdout-$SLURM_JOB_ID"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE="$TASK_SCRATCH/energy-code-$SLURM_JOB_ID"
MODEL="$TASK_SCRATCH/Qwen3-8B-$SLURM_JOB_ID.gguf"
MODEL_SHA=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
LLAMA_COMMIT=7798007a29a90e3053e799394da48cf53a2f8e0f
EXISTING=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/env
mkdir -p "$RUN" "$PROJECT/env/llama-runtime"
git clone --shared --no-checkout "$PROJECT/repo" "$CODE"
git -C "$CODE" checkout --detach "$ENERGY_GIT_COMMIT"
# Only journaling/export/GoalSpec compatibility may differ from the accepted
# direct-tool pilot. A planner/scorer/benchmark change needs a new pilot decision.
git -C "$CODE" diff --exit-code "$ENERGY_PILOT_COMMIT" "$ENERGY_GIT_COMMIT" -- \
 src/energy_agent ':(exclude)src/energy_agent/goal_compiler.py' \
 ':(exclude)src/energy_agent/decision_demo.py' ':(exclude)src/energy_agent/interview_demo.html' \
 'benchmarks/planner_remediation*'
test "$(git -C "$EXISTING/llama.cpp-$LLAMA_COMMIT" rev-parse HEAD)" = "$LLAMA_COMMIT"
cp -a "$EXISTING/llama.cpp-$LLAMA_COMMIT/build/bin/." "$PROJECT/env/llama-runtime/"
SERVER="$PROJECT/env/llama-runtime/llama-server"
module purge
module load GCC/13.3.0 CMake/3.31.8 CUDA/12.4.1
PYTHON_ROOT=/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3
export LD_LIBRARY_PATH="$PROJECT/env/llama-runtime:$PYTHON_ROOT/lib:${LD_LIBRARY_PATH:-}"
PYTHON="$PROJECT/env/venv/bin/python"
export PYTHONPATH="$CODE/src"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RUN/gpu.txt"
curl -fsSL --retry 2 -o "$MODEL" \
 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/1d54a16a18cba0d8fbad4a16db801decc729e099/Qwen3-8B-Q4_K_M.gguf?download=true'
echo "$MODEL_SHA  $MODEL" | sha256sum --check
sha256sum "$SERVER" "$MODEL" > "$RUN/runtime_sha256.txt"
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
PORT=11627
"$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx-size 16384 \
 --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none --no-webui > "$RUN/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
cd "$CODE"
"$PYTHON" scripts/evaluate_llm_agent.py \
 --benchmark benchmarks/planner_remediation_holdout_v3.jsonl \
 --gate benchmarks/planner_remediation_gate_v3.json --scoring-version v3 \
 --data "$ROOT/ingest-12m/dispatch_features_repaired_v2.csv.gz" \
 --data-manifest "$ROOT/ingest-12m/final_manifest_v2.json" \
 --evidence "$ROOT/claim-transport-input-905/evidence_documents.jsonl" \
 --forecast-snapshots "$ROOT/forecast-snapshots-c66e415.jsonl" \
 --provider llama_cpp --provider-url "http://127.0.0.1:$PORT/v1" --model Qwen3-8B-Q4_K_M.gguf \
 --temperature 0.2 --paths deterministic pure_llm constrained_hybrid \
 --memory-modes no_memory full_history sliding_window structured_state \
 --seeds 17 29 43 --output "$RUN/evaluation"
"$PYTHON" scripts/summarize_planner_remediation.py --run "$RUN/evaluation" --output "$RUN/public_summary.json"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P > "$RUN/slurm_usage_at_end.txt"
