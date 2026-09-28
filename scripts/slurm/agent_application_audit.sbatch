#!/bin/bash
#SBATCH --job-name=energy-application-audit
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/application-audit-20260927/audit-%j.out
#SBATCH --error=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/application-audit-20260927/audit-%j.err
set -euo pipefail
umask 077
PROJECT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/application-audit-20260927
ROOT=/data/gpfs/projects/punim2936/portfolio_20260820/energy-artifacts
RUN="$ROOT/application-audit-20260927/job-$SLURM_JOB_ID"
mkdir -p "$RUN"
module purge
module load GCCcore/11.3.0 Python/3.11.3
python "$PROJECT/audit_agent_application.py" \
 --predictions "$ROOT/planner-remediation-20260927/private/holdout-31365531/evaluation/predictions.jsonl" \
 --benchmark "$PROJECT/planner_remediation_holdout_v3.jsonl" \
 --frozen-manifest "$PROJECT/planner_remediation_run_manifest_20260927.json" \
 --p1-full-run "$ROOT/planner-remediation-20260927/private/pilot-31365444/demo/full_run.json" \
 --p1-compact "$PROJECT/recorded_run.json" --p1-manifest "$PROJECT/p1_recorded_replay_manifest_20260927.json" \
 --market "$ROOT/ingest-12m/dispatch_features_repaired_v2.csv.gz" \
 --evidence "$ROOT/claim-transport-input-905/evidence_documents.jsonl" \
 --figures "$ROOT/planner-remediation-20260927/private/preflight-31365198/q4-workbook/figure_manifest.jsonl" \
 --workbook "$ROOT/planner-remediation-20260927/private/preflight-31365198/q4-workbook/source.xlsx" \
 --output "$RUN/application_audit.json" --graph-input-output "$RUN/graph_input_private.json"
sha256sum "$PROJECT/audit_agent_application.py" "$PROJECT/agent_application_audit.sbatch" > "$RUN/audit_code_sha256.txt"
