#!/bin/bash
#SBATCH --job-name=energy-incremental-model
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --tmp=10000
set -euo pipefail
umask 077
: "${ENERGY_GPU_RELEASE_ID:?coordinator release required; this file must not submit itself}"
: "${ENERGY_BUNDLE:?}"
: "${ENERGY_BUNDLE_SHA:?}"
: "${ENERGY_STAGE:?pilot or full}"
BASE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/incremental-agent-20260929
ART=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
RUN="$ART/incremental-agent-20260929"
case "$ENERGY_BUNDLE" in "$BASE"/*) ;; *) exit 2;; esac
case "$ENERGY_STAGE" in pilot|full) ;; *) exit 2;; esac
if [[ "$ENERGY_STAGE" == full ]]; then
 : "${ENERGY_PILOT_RECEIPT_SHA:?reviewed pilot quality receipt required}"
 : "${ENERGY_PILOT_RECEIPT:?}"
 case "$ENERGY_PILOT_RECEIPT" in "$RUN"/*) ;; *) exit 2;; esac
 echo "$ENERGY_PILOT_RECEIPT_SHA  $ENERGY_PILOT_RECEIPT" | sha256sum --check
fi
echo "$ENERGY_BUNDLE_SHA  $ENERGY_BUNDLE" | sha256sum --check
MODEL="$BASE/Qwen3-8B-Q4_K_M.gguf"
echo "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785  $MODEL" | sha256sum --check
TASK=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-model.XXXXXX")
test -w "$TASK"
tar -xf "$ENERGY_BUNDLE" -C "$TASK"
mkdir -p "$RUN"
module purge
module load GCC/13.3.0 CMake/3.31.8 CUDA/12.4.1
ENV=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/env
PYTHON="$ENV/venv/bin/python"
export LD_LIBRARY_PATH="$ENV/llama-runtime:/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.11.3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false OMP_NUM_THREADS=6
(cd "$TASK/wheels" && sha256sum --check SHA256SUMS)
"$PYTHON" -m pip install --no-index --no-compile --ignore-installed --find-links "$TASK/wheels" --target "$TASK/deps" langgraph==1.2.12 langgraph-checkpoint-sqlite==3.1.1
export PYTHONPATH="$TASK/src:$TASK/deps"
cd "$TASK"
if [[ "$ENERGY_STAGE" == pilot ]]; then
 DATA="$ART/application-audit-20260927/job-31372698/graph_input_private.json"
 DATA_SHA=4539f94624e603fc5d44572966c0ee5c24d976417e786a4e387d0711cabfa94f
 SNAP="$ART/forecast-snapshots-c66e415.jsonl"
 SNAP_SHA=b27f3b192dc1726e9b6abfc390b655b411fe5e4f32b062d9c859a8e69e03b48d
 OPTIONS=(--development --seeds 17)
else
 "$PYTHON" -c 'import json,os; p=json.load(open(os.environ["ENERGY_PILOT_RECEIPT"])); assert p["quality_gate"] == "passed" and p["model_inference"] is True and p["source_bundle_sha256"] == os.environ["ENERGY_BUNDLE_SHA"]'
 : "${ENERGY_NOVEMBER_ARCHIVE:?}"
 : "${ENERGY_NOVEMBER_SHA:?}"
 case "$ENERGY_NOVEMBER_ARCHIVE" in "$RUN"/*) ;; *) exit 2;; esac
 echo "$ENERGY_NOVEMBER_SHA  $ENERGY_NOVEMBER_ARCHIVE" | sha256sum --check
 tar -xf "$ENERGY_NOVEMBER_ARCHIVE" -C "$TASK"
 DATA="$TASK/november/november-market.json"
 SNAP="$TASK/november/november-snapshots.jsonl"
 DATA_SHA=$("$PYTHON" -c 'import json; print(json.load(open("november/manifest.json"))["data_sha256"])')
 SNAP_SHA=$("$PYTHON" -c 'import json; print(json.load(open("november/manifest.json"))["snapshots_sha256"])')
 OPTIONS=(--consume-frozen --release-id "$ENERGY_GPU_RELEASE_ID" --scenarios benchmarks/incremental_vnext_holdout_v2_20260929.jsonl
 --scenario-sha256 9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f --seeds 17 29 43)
fi
cp "$MODEL" "$TASK/model.gguf"
MODEL_ID="energy-${SLURM_JOB_ID}-$(basename "$TASK")"
"$PYTHON" scripts/check_local_model_owner.py --port 11629
"$ENV/llama-runtime/llama-server" --help > "$TASK/runtime-help.txt"
grep -q -- '--alias' "$TASK/runtime-help.txt"
"$ENV/llama-runtime/llama-server" --model "$TASK/model.gguf" --alias "$MODEL_ID" --host 127.0.0.1 --port 11629 \
 --ctx-size 16384 --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none --no-webui > "$TASK/model-server.log" 2>&1 &
SERVER_PID=$!
finish() {
 status=$?
 kill "$SERVER_PID" 2>/dev/null || true
 # One persistent archive, including failed attempts; scratch is Slurm-owned.
 if [[ -d "$TASK/results" ]]; then
  tar -czf "$RUN/${ENERGY_STAGE}-$SLURM_JOB_ID.tar.gz" -C "$TASK" model-server.log results
 else
  tar -czf "$RUN/${ENERGY_STAGE}-$SLURM_JOB_ID.tar.gz" -C "$TASK" model-server.log
 fi
 exit "$status"
}
trap finish EXIT
for attempt in $(seq 1 90); do
 kill -0 "$SERVER_PID" || exit 3
 curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:11629/health >/dev/null && break
 sleep 1
done
"$PYTHON" scripts/check_local_model_owner.py --port 11629 --pid "$SERVER_PID" --model-id "$MODEL_ID"
"$PYTHON" scripts/run_incremental_vnext.py --model --provider-url http://127.0.0.1:11629/v1 --provider-model "$MODEL_ID" \
 --data "$DATA" --data-sha256 "$DATA_SHA" --snapshots "$SNAP" --snapshots-sha256 "$SNAP_SHA" \
 --evidence "$ART/claim-transport-input-905/evidence_documents.jsonl" \
 --evidence-sha256 9c56543061843f95b9eec85e62be5a1fec50dcc85e971197e0c50fbf00aee3cc \
 "${OPTIONS[@]}" --output "$TASK/results"
