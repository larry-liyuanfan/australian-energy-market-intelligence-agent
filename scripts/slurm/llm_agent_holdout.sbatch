#!/bin/bash
#SBATCH --job-name=energy-llm-holdout
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/holdout-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/holdout-%j.err

set -euo pipefail
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/llm-agent-eval-20260903
MARKET=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired_v2.csv.gz
MARKET_MANIFEST=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/final_manifest_v2.json
EVIDENCE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl
SNAPSHOTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/forecast-snapshots-c66e415.jsonl
PORT=11572
RUN_DIR="$ARTIFACTS/private/holdout-$SLURM_JOB_ID"
export OLLAMA_MODELS="$ENV_ROOT/ollama-models"
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_KEEP_ALIVE=60m
export OLLAMA_TMPDIR="$ENV_ROOT/ollama-tmp"
export OLLAMA_NO_CLOUD=1
export OLLAMA_NOHISTORY=1
mkdir -p "$RUN_DIR" "$OLLAMA_TMPDIR"

for path in "$MARKET" "$MARKET_MANIFEST" "$EVIDENCE" "$SNAPSHOTS"; do
  test -s "$path"
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" serve >"$PROJECT/logs/ollama-holdout-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/api/tags" >/dev/null && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/api/tags" >/dev/null
cd "$REPO"
"$ENV_ROOT/venv/bin/python" scripts/evaluate_llm_agent.py \
  --benchmark benchmarks/llm_agent_holdout_v1.jsonl \
  --gate benchmarks/llm_agent_promotion_gate_v1.json \
  --output "$RUN_DIR" \
  --data "$MARKET" \
  --data-manifest "$MARKET_MANIFEST" \
  --evidence "$EVIDENCE" \
  --forecast-snapshots "$SNAPSHOTS" \
  --ollama-url "http://127.0.0.1:$PORT" \
  --model qwen3:4b-instruct
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES -P >"$RUN_DIR/slurm_usage.txt"
