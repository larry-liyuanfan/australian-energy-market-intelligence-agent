#!/bin/bash
#SBATCH --job-name=energy-llm-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
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
PORT=11571
export OLLAMA_MODELS="$ENV_ROOT/ollama-models"
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_TMPDIR="$ENV_ROOT/ollama-tmp"
mkdir -p "$OLLAMA_MODELS" "$OLLAMA_TMPDIR" "$ARTIFACTS/private/pilot-$SLURM_JOB_ID"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" serve >"$PROJECT/logs/ollama-pilot-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/api/tags" >/dev/null && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/api/tags" >/dev/null
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" pull qwen3:4b-instruct
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" pull qwen3:8b
cd "$REPO"
for model in qwen3:4b-instruct qwen3:8b; do
  safe_model=${model//:/-}
  "$ENV_ROOT/venv/bin/python" scripts/smoke_llm_agent.py \
    --ollama-url "http://127.0.0.1:$PORT" \
    --model "$model" \
    --output "$ARTIFACTS/private/pilot-$SLURM_JOB_ID/$safe_model-smoke"
  "$ENV_ROOT/venv/bin/python" scripts/evaluate_llm_agent.py \
    --benchmark benchmarks/llm_agent_development_v1.jsonl \
    --gate benchmarks/llm_agent_promotion_gate_v1.json \
    --output "$ARTIFACTS/private/pilot-$SLURM_JOB_ID/$safe_model-development" \
    --data "$MARKET" \
    --data-manifest "$MARKET_MANIFEST" \
    --evidence "$EVIDENCE" \
    --forecast-snapshots "$SNAPSHOTS" \
    --ollama-url "http://127.0.0.1:$PORT" \
    --model "$model" \
    --paths pure_llm constrained_hybrid \
    --memory-modes structured_state \
    --seeds 17 29
done
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" ps
