#!/bin/bash
#SBATCH --job-name=energy-p0-preflight
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/preflight-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/preflight-%j.err
set -euo pipefail
umask 077
: "${ENERGY_GIT_COMMIT:?exact commit required}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927
ARTIFACTS=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/planner-remediation-20260927
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE="$TASK_SCRATCH/energy-code-$SLURM_JOB_ID"
RUN="$ARTIFACTS/private/preflight-$SLURM_JOB_ID"
mkdir -p "$RUN" "$PROJECT/env"
git clone --shared --no-checkout "$PROJECT/repo" "$CODE"
git -C "$CODE" checkout --detach "$ENERGY_GIT_COMMIT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export PIP_NO_CACHE_DIR=1
python -m venv "$PROJECT/env/venv"
PYTHON="$PROJECT/env/venv/bin/python"
"$PYTHON" -m pip install "$CODE[test,search,workbook]"
cd "$CODE"
export PYTHONPATH="$CODE/src"
"$PYTHON" -m pytest -q | tee "$RUN/pytest.txt"
"$PYTHON" -m ruff check src scripts tests | tee "$RUN/ruff.txt"
"$PYTHON" -m mypy --strict src scripts | tee "$RUN/mypy.txt"
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
COMMON=(--gate benchmarks/planner_remediation_gate_v3.json --scoring-version v3 --preflight-only
  --output "$RUN/unused" --data "$ROOT/ingest-12m/dispatch_features_repaired_v2.csv.gz"
  --data-manifest "$ROOT/ingest-12m/final_manifest_v2.json")
for split in development holdout; do
  "$PYTHON" scripts/evaluate_llm_agent.py "${COMMON[@]}" \
    --benchmark "benchmarks/planner_remediation_${split}_v3.jsonl" | tee "$RUN/$split-coverage.json"
done
sha256sum benchmarks/planner_remediation* src/energy_agent/remediation.py > "$RUN/input_sha256.txt"
printf '%s\n' "$ENERGY_GIT_COMMIT" > "$RUN/git_sha.txt"
"$PYTHON" scripts/compile_demo_workbook.py --output "$RUN/q4-workbook"
"$PYTHON" scripts/export_interview_demo.py \
  --data "$ROOT/ingest-12m/dispatch_features_repaired_v2.csv.gz" \
  --data-manifest "$ROOT/ingest-12m/final_manifest_v2.json" \
  --evidence "$ROOT/claim-transport-input-905/evidence_documents.jsonl" \
  --figures "$RUN/q4-workbook/figure_manifest.jsonl" \
  --forecast-snapshots "$ROOT/forecast-snapshots-c66e415.jsonl" \
  --output "$ARTIFACTS/preview/preflight-$SLURM_JOB_ID" \
  --private-output "$RUN/demo"
