#!/bin/bash
#SBATCH --job-name=energy-llm-env
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/preflight-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/preflight-%j.err

set -euo pipefail
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
OLLAMA_ROOT="$ENV_ROOT/ollama-v0.33.2"
export PIP_CACHE_DIR="$ENV_ROOT/pip-cache"

test "$(realpath "$PROJECT")" = "$PROJECT"
test -f "$REPO/benchmarks/llm_agent_promotion_gate_v1.json"
test -f "$REPO/benchmarks/llm_agent_holdout_v1.jsonl"

module purge
module load Python/3.11.3
if [ ! -x "$ENV_ROOT/venv/bin/python" ]; then
  python -m venv "$ENV_ROOT/venv"
fi
"$ENV_ROOT/venv/bin/python" -m pip install --upgrade pip
"$ENV_ROOT/venv/bin/python" -m pip install -e "$REPO[test,search,workbook]"

if [ ! -x "$OLLAMA_ROOT/bin/ollama" ]; then
  mkdir -p "$OLLAMA_ROOT"
  archive="$ENV_ROOT/ollama-linux-amd64-v0.33.2.tar.zst"
  if [ ! -f "$archive" ]; then
    curl -fL --retry 3 -o "$archive" \
      https://github.com/ollama/ollama/releases/download/v0.33.2/ollama-linux-amd64.tar.zst
  fi
  tar --zstd -xf "$archive" -C "$OLLAMA_ROOT"
fi

"$ENV_ROOT/venv/bin/python" -m pytest "$REPO/tests/test_model_agent.py" "$REPO/tests/test_llm_evaluation.py" -q
"$ENV_ROOT/venv/bin/python" -m ruff check "$REPO/src" "$REPO/scripts/evaluate_llm_agent.py" "$REPO/scripts/smoke_llm_agent.py"
"$OLLAMA_ROOT/bin/ollama" --version
sha256sum "$OLLAMA_ROOT/bin/ollama" "$REPO/benchmarks/llm_agent_holdout_v1.jsonl" \
  "$REPO/benchmarks/llm_agent_promotion_gate_v1.json"
