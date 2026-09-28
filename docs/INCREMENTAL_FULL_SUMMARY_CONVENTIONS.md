# Frozen full: one read-only summary entry

This is postprocessing preparation, **not a full evaluation result**. It does not
import or run the model, tools, optimizer, LangGraph, or frozen scorer. The running
job 31492138 and source06 are unchanged. No second job was started for this work.

After obtaining the retained private `attempts.jsonl`, run once into a new file:

```bash
python scripts/summarize_incremental_full.py \
  --attempts "$ATTEMPTS" \
  --scenarios benchmarks/incremental_vnext_holdout_v2_20260929.jsonl \
  --scenario-sha256 9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f \
  --output full-summary.json
```

The output contains totals, path/track, path/fault and path/episode groups, plus
paired per-slot diagnostics keyed by `episode:turn:seed` with all three paths.
It is a reporting skeleton until real records are supplied. Inspect private
traces for explanation; the script does not export questions, evidence text,
raw tool arguments, final answers, or nested history.

## Denominators and outcomes

- **24 episodes / 48 authored turns**, repeated over three paths and three seeds:
  432 expected records, not 432 independent tasks. All three paths call a model;
  none is a no-model baseline.
- Episode-level frozen `fault` defines the track. Natural tasks and controlled
  fault/lifecycle episodes each have 216 slots. Fault preparation rounds remain
  in the latter; checkpoint first-round pauses are planned interventions too.
  Missing slots remain in denominators; duplicate/unexpected keys and interior
  corruption fail closed. A malformed final record is explicit partial output.
- Preserve the recorded contract match without rescoring. Separately report
  completed + numerical checks, completed + state/numerical checks, contract-
  matched noncompletion, target-verified safe stops, and recovery to verified
  completion. A matched clarification/stop/pause is not completion. Unchecked
  state or numerical evidence is `null`, not a verified success.
- Planned intervention slots come from the frozen episode even when a row is
  missing. Tool fault fired, checkpoint pause observed, intervention exercised
  and target outcome verified remain separate fields. Contract-matched stops
  are not automatically labelled reasonable or semantically correct.

## Attribution and resource accounting

Initial proposals, runtime accepted/rejected proposals, initial canonical
matches, model-owned executions, system completion, tool-status counts and reuse
are distinct. A canonical match is not evidence that the model changed the plan.
Legacy owner/acceptance/reuse fields are unavailable, not inferred from its final
path. Empty-evidence rewrite/clarify/stop decisions are counted separately from
actual rewrite executions; no newly useful evidence is inferred from either.

Only current-row `calls/tool_calls`, `provenance`, `planner_attempts` and top-level
`provider_requests` are counted. `history` is never traversed. Skipped tool-call
records do not count as executed attempts. Missing provider boundaries differ
from empty boundaries; swallowed provider errors survive successful fallback.
Missing run/tool/planner traces are unmeasured, not zero events. Tool recovery is
identified from ordered same-tool failure then success, not unset attempt flags.
Known prompt/completion tokens and provider cost remain lower-bound observations
when usage/rows/the journal tail are missing. Per-field measured-row counts
retain unmeasured events. Even a clean final newline cannot prove there were no
unrecorded in-flight requests when expected slots remain missing. These counts
accompany sums. Wall-clock sums are separate from tokens/tools and are not Slurm
allocation costs. Fixed path order and shared prefix cache prohibit causal
acceleration claims; no new confidence interval or P95 claim is generated.

Final report order: completeness and episode denominators; outcome types by
track/fault; model proposal versus runtime contribution; requests/tokens/tool
costs; three trace-backed explanatory cases; limitations and replay command.
Select cases only after the full run terminates, retaining negative examples.

Verification for this small package: **11 synthetic CPU accounting fixtures**,
targeted Ruff and strict mypy passed. Fixtures cover history duplication, fault
setup/pause, unavailable legacy attribution, provider failure + fallback,
unchecked/missing values, duplicate/missing slots, safe stops and truncated
records. No model evaluation, second GPU job, full test suite or CI wait was used.
