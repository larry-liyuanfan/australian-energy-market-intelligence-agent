#!/bin/bash
#SBATCH --job-name=energy-llm-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/pilot-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/pilot-%j.err

set -euo pipefail
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/llm-agent-eval-20260903
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
cd "$REPO"
"$ENV_ROOT/venv/bin/python" scripts/smoke_llm_agent.py \
  --ollama-url "http://127.0.0.1:$PORT" \
  --model qwen3:4b-instruct \
  --output "$ARTIFACTS/private/pilot-$SLURM_JOB_ID"
"$ENV_ROOT/ollama-v0.33.2/bin/ollama" ps
