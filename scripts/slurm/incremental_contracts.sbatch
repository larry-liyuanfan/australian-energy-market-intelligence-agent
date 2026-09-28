#!/bin/bash
#SBATCH --job-name=energy-incremental-contracts
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
: "${ENERGY_BUNDLE:?}"
: "${ENERGY_BUNDLE_SHA:?}"
BASE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/incremental-agent-20260929
RUN=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/incremental-agent-20260929
case "$ENERGY_BUNDLE" in "$BASE"/*) ;; *) exit 2;; esac
echo "$ENERGY_BUNDLE_SHA  $ENERGY_BUNDLE" | sha256sum --check
TASK=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-contracts.XXXXXX")
tar -xf "$ENERGY_BUNDLE" -C "$TASK"
module purge
module load GCCcore/11.3.0 Python/3.11.3
PYTHON=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false OMP_NUM_THREADS=2
(cd "$TASK/wheels" && sha256sum --check SHA256SUMS)
"$PYTHON" -m pip install --no-index --no-compile --ignore-installed --find-links "$TASK/wheels" --target "$TASK/deps" langgraph==1.2.12 langgraph-checkpoint-sqlite==3.1.1
export PYTHONPATH="$TASK/src:$TASK/deps"
cd "$TASK"
"$PYTHON" scripts/run_incremental_vnext.py --contract-check \
 --scenarios benchmarks/incremental_vnext_holdout_v2_20260929.jsonl \
 --scenario-sha256 9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f --output "$TASK/contracts"
mkdir -p "$RUN"
tar -czf "$RUN/contracts-$SLURM_JOB_ID.tar.gz" -C "$TASK" contracts
sha256sum "$RUN/contracts-$SLURM_JOB_ID.tar.gz"
# CUDA-linked server CLI inspection belongs to the authorised GPU preflight:
# a CPU node need not expose libcuda.so.1, even with the CUDA module loaded.
echo 'SYNTHETIC_CONTRACT_RUNNER_VERIFIED_NO_MODEL_INFERENCE'
