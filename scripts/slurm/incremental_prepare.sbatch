#!/bin/bash
#SBATCH --job-name=energy-incremental-prepare
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
BASE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/incremental-agent-20260929
ART=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
: "${ENERGY_BUNDLE_SHA:?}"
: "${ENERGY_BUNDLE:?}"
case "$ENERGY_BUNDLE" in "$BASE"/*) ;; *) exit 2;; esac
echo "$ENERGY_BUNDLE_SHA  $ENERGY_BUNDLE" | sha256sum --check
TASK=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-incremental.XXXXXX")
test -w "$TASK"
tar -xf "$ENERGY_BUNDLE" -C "$TASK"
module purge
module load GCCcore/11.3.0 Python/3.11.3
PYTHON=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false OMP_NUM_THREADS=2
(cd "$TASK/wheels" && sha256sum --check SHA256SUMS)
"$PYTHON" -m pip install --no-index --no-compile --ignore-installed --find-links "$TASK/wheels" --target "$TASK/deps" langgraph==1.2.12 langgraph-checkpoint-sqlite==3.1.1
export PYTHONPATH="$TASK/src:$TASK/deps"
"$PYTHON" -c 'import langgraph.graph, langgraph.checkpoint.sqlite, scipy, pydantic; print("isolated Linux/Python graph imports passed")'
"$PYTHON" "$TASK/scripts/prepare_incremental_inputs.py" \
 --market "$ART/ingest-12m/dispatch_features_repaired_v2.csv.gz" \
 --manifest "$ART/ingest-12m/final_manifest_v2.json" \
 --template "$ART/application-audit-20260927/job-31372698/graph_input_private.json" --output "$TASK/november"
MODEL="$BASE/Qwen3-8B-Q4_K_M.gguf"
SHA=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
if [[ ! -e "$MODEL" ]]; then
 PART="$BASE/Qwen3-8B-Q4_K_M.gguf.part"
 curl --fail --location --retry 2 --continue-at - --output "$PART" \
  'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/1d54a16a18cba0d8fbad4a16db801decc729e099/Qwen3-8B-Q4_K_M.gguf?download=true'
 echo "$SHA  $PART" | sha256sum --check
 test ! -e "$MODEL"
 mv --no-clobber "$PART" "$MODEL"
fi
echo "$SHA  $MODEL" | sha256sum --check
RUN="$ART/incremental-agent-20260929"
mkdir -p "$RUN"
tar -cf "$RUN/november-inputs-$SLURM_JOB_ID.tar" -C "$TASK" november
sha256sum "$RUN/november-inputs-$SLURM_JOB_ID.tar" "$MODEL"
df -h "$BASE" "$TASK"
df -i "$BASE" "$TASK"
echo 'CPU_PREPARATION_COMPLETE_NO_MODEL_INFERENCE'
