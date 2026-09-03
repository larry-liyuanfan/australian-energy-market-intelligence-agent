#!/bin/bash
#SBATCH --job-name=energy-vg2-env
#SBATCH --account=punim2936
#SBATCH --partition=cascade
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/preflight-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2/logs/preflight-%j.err

set -euo pipefail
: "${ENERGY_GIT_COMMIT:?Set the exact pushed commit}"
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-visual-goal-compiler-v2
REPO="$PROJECT/repo"
TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CODE_ROOT="$TASK_SCRATCH/energy-vg2-preflight-$SLURM_JOB_ID"
ENV_ROOT="$TASK_SCRATCH/energy-vg2-preflight-env-$SLURM_JOB_ID"
export PIP_NO_CACHE_DIR=1
export HF_HOME="$TASK_SCRATCH/energy-vg2-preflight-hf-$SLURM_JOB_ID"
export HF_HUB_DISABLE_TELEMETRY=1

test "$(realpath "$PROJECT")" = "$PROJECT"
test -d "$REPO/.git"
git -C "$REPO" cat-file -e "$ENERGY_GIT_COMMIT^{commit}"
git clone --shared --no-checkout "$REPO" "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach "$ENERGY_GIT_COMMIT"
trap 'rm -rf -- "$CODE_ROOT"' EXIT

module purge
module load GCCcore/11.3.0 Python/3.11.3
python -m venv "$ENV_ROOT/venv"
"$ENV_ROOT/venv/bin/python" -m pip install --upgrade 'pip>=25.1,<26'
"$ENV_ROOT/venv/bin/python" -m pip install -e "$CODE_ROOT[test,search,workbook,multimodal]"

cd "$CODE_ROOT"
"$ENV_ROOT/venv/bin/python" -m pytest -q
"$ENV_ROOT/venv/bin/python" -m ruff check src tests scripts
"$ENV_ROOT/venv/bin/python" -m mypy --strict src scripts
sha256sum -c <(python - <<'PY'
import json
from pathlib import Path
freeze = json.loads(Path('benchmarks/v2_freeze_manifest.json').read_text())
for section, path_key, sha_key in (
    ('goal_spec', 'benchmark', 'benchmark_sha256'),
    ('goal_spec', 'gate', 'gate_sha256'),
    ('vidore', 'config', 'config_sha256'),
    ('vidore', 'weight_selection_source', 'weight_selection_source_sha256'),
):
    print(f"{freeze[section][sha_key]}  {freeze[section][path_key]}")
PY
)
for path in \
  /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/dispatch_features_repaired_v2.csv.gz \
  /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/ingest-12m/final_manifest_v2.json \
  /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/evidence-index/evidence_documents.jsonl \
  /data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts/forecast-snapshots-c66e415.jsonl; do
  test -s "$path"
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
