#!/bin/bash
#SBATCH --job-name=energy-p1-demo
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/demo-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927/logs/demo-%j.err
set -euo pipefail
umask 077
: "${ENERGY_GIT_COMMIT:?exact commit required}"
: "${ENERGY_FIGURES:?verified Q4 2025 figure manifest required}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/planner-remediation-20260927
ROOT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
CODE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/energy-demo-$SLURM_JOB_ID"
RUN="$ROOT/planner-remediation-20260927/private/demo-$SLURM_JOB_ID"
mkdir -p "$RUN"
git clone --shared --no-checkout "$PROJECT/repo" "$CODE"
git -C "$CODE" checkout --detach "$ENERGY_GIT_COMMIT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
PYTHON="$PROJECT/env/venv/bin/python"
export PYTHONPATH="$CODE/src"
cd "$CODE"
"$PYTHON" -m pytest tests/test_decision_demo.py tests/test_snapshots.py tests/test_workbook_evidence.py -q
"$PYTHON" -m ruff check src scripts tests
"$PYTHON" -m mypy --strict src scripts
"$PYTHON" scripts/export_interview_demo.py \
  --data "$ROOT/ingest-12m/dispatch_features_repaired_v2.csv.gz" \
  --data-manifest "$ROOT/ingest-12m/final_manifest_v2.json" \
  --evidence "$ROOT/claim-transport-input-905/evidence_documents.jsonl" \
  --figures "$ENERGY_FIGURES" \
  --forecast-snapshots "$ROOT/forecast-snapshots-c66e415.jsonl" \
  --output "$ROOT/planner-remediation-20260927/preview/demo-$SLURM_JOB_ID" \
  --private-output "$RUN/trace"
