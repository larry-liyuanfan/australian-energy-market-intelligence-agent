#!/bin/bash
#SBATCH --job-name=energy-llm-env
#SBATCH --account=punim2936
#SBATCH --partition=cascade
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/preflight-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903/logs/preflight-%j.err

set -euo pipefail
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/llm-agent-eval-20260903
REPO="$PROJECT/repo"
ENV_ROOT="$PROJECT/env"
export PIP_CACHE_DIR="$ENV_ROOT/pip-cache"

test "$(realpath "$PROJECT")" = "$PROJECT"
test -f "$REPO/benchmarks/llm_agent_promotion_gate_v1.json"
test -f "$REPO/benchmarks/llm_agent_holdout_v1.jsonl"

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
if [ ! -x "$ENV_ROOT/venv/bin/python" ]; then
  python -m venv "$ENV_ROOT/venv"
fi
"$ENV_ROOT/venv/bin/python" -m pip install --upgrade pip
"$ENV_ROOT/venv/bin/python" -m pip install -e "$REPO[test,search,workbook]"

"$ENV_ROOT/venv/bin/python" -m pytest "$REPO/tests/test_model_agent.py" "$REPO/tests/test_llm_evaluation.py" -q
"$ENV_ROOT/venv/bin/python" -m ruff check "$REPO/src" "$REPO/scripts/evaluate_llm_agent.py" "$REPO/scripts/smoke_llm_agent.py"
sha256sum "$REPO/scripts/slurm/llm_agent_gpu_pilot.sbatch" "$REPO/benchmarks/llm_agent_holdout_v1.jsonl" \
  "$REPO/benchmarks/llm_agent_promotion_gate_v1.json"
