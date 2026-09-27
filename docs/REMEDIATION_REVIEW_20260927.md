# Planner review and bounded P0/P1 remediation

Status: implementation in progress; no new real-model result or promotion yet.
This follows the 26 September career strengthening plan. Existing v1 metrics,
v2 benchmark labels and negative experiments remain unchanged.

Tool installation and readiness are recorded separately in
[environment status](ENVIRONMENT_STATUS.md). Browser review of the first
numerically valid P1 recording found irrelevant gas-market/following-quarter
figures: citation presence does not establish relevance, so that preview remains
unaccepted pending a topic/period grounding correction.

Preflight `31365151` failed the security regression suite after dated queries
were scoped too broadly. Quarter filtering now requires an explicit quarter
request, preserving daily-event retrieval and the existing injected-evidence
tests. The 24 failures remain recorded. Independent review also supplied gas
body-text, other-region/NEM and unsafe URL-scheme counterexamples; each has a
regression test. The scope screen is not described as semantic entailment.

Infrastructure record: CPU preflight `31364397` failed after two seconds because
this Spartan allocation did not export `SLURM_TMPDIR`. The replacement uses the
established `SLURM_TMPDIR` → `TMPDIR` → job-unique `/tmp` path convention. No
model request occurred. The scheduler recommended the newer `sapphire` CPU
partition, which is now explicit in the preflight script.

CPU preflight `31364416` completed in 80 seconds (482,180 KiB batch MaxRSS),
including the full suite and coverage checks for 5 development and 15 holdout
region-days. P1 preflight `31364486` passed tests/lint/types but failed at export:
official report metadata contains date-only publication values. The exporter now
treats missing time/zone precision conservatively as retrospective-only. Review
also added actual market-file digest verification before parsing; manifest claims
alone no longer establish the input hash. Both failures remain in the run record.

P1 retry `31364532` passed all calculation checks but could not serialize an event
diagnosis timestamp into the compact JSON. Diagnosis now uses the canonical
Pydantic JSON serialization; a regression test includes an actual datetime.
The failed export is not a published demonstration.

## Findings from existing artifacts and code

1. **Pilot data mismatch.** The four-turn GoalSpec pilot used 4–5 August 2025;
   its real store starts on 18 August 2025. Direct forecast/dispatch failures
   therefore combine planning error and unavailable market history. A successful
   Slurm exit and JSON artifact validator did not establish valid task coverage.
2. **Invalid GoalSpec output.** In private job `30003769`, three outputs used
   the forbidden `field_sources.bess` parent key; another selected decision replay
   with `bess=null`. One also used `official_evidence` as a modality. Every output
   ended its day at 23:59:59. That time error affects semantics but was not rejected
   by the old schema. Required outputs were also missing. The second turns invented
   2023 dates after invalid first turns discarded their user context.
3. **Correction reparsing.** Sourced state correctly replaces a region, but the
   previous direct runtime reparsed the original correction sentence and appended
   JSON as natural language. That reintroduced the old region and lost comparison
   intent. Model proposals cannot fix a wrong deterministic fallback contract.
4. **Attribution and scoring.** Old `model_proposed_calls` concatenates initial
   and replan calls; it is not strictly first-attempt planning. GoalSpec counts
   deterministic recovery as model replanning. Substring argument checks can miss
   extra regions, wrong interval endpoints and omitted economic constraints.
5. **P1 evidence/decision boundary.** Official reports are retrospective context,
   not necessarily published at decision time. Figure IDs/cell previews need to
   survive into the demonstrator. Forecast and dispatch must use the same complete
   signal; net-of-sensitivity margins must display their explicit cost assumption.

Evidence: `docs/VISUAL_GOAL_COMPILER_V2_EVALUATION.md`,
`docs/LLM_AGENT_PLANNER_MEMORY_EVALUATION.md`, private pilot job `30003769`
predictions/manifest and the source modules referenced in those reports. Only
compact summaries and hashes may be published; private row-level artifacts remain
in the Energy artifact root.

## Fixed experiment scope, before new inference

- Real Qwen3-8B Q4_K_M, pinned llama.cpp, one Iris Spartan 20 GB A100 MIG job at a
  time. No model retraining, 14B expansion or ViDoRe run.
- Four-turn, covered October development pilot diagnoses schema/interface fixes.
  GoalSpec remains experimental: schema conformity alone does not validate source
  attribution or justify making model-generated state factual memory.
- New v3 holdout: 10 episodes / 18 turns; deterministic, pure LLM and constrained
  hybrid; four memory modes; seeds 17/29/43 at temperature 0.2 for model paths.
  Prompts/dates are disjoint from the older sets. Labels are author-written, not
  independent human evaluation. Fault-injected episodes are reported separately.
- Data preflight requires all 288 interval timestamps plus prior history for every
  region/day before any model request. No zero-row market answer counts as success.
- Keep the v1 threshold values; raw path now means initial proposal, and argument
  scoring checks canonical typed values. New scores are not numerically comparable
  to v1 as an A/B lift. Report model proposal, guarded plan and execution separately.
- Deterministic recovery gets the same bounded tool-attempt budget. Model replans
  and deterministic retries remain distinct. Retry tokens and time are included.
- Run the new holdout once after the pilot decision and code freeze. Report all
  failures and uncertainty; no relabelling or tuning on its outputs.

## P1 acceptance evidence still required

One reproducible real SA1 decision replay: user request, actual model proposal,
deterministic additions, text and workbook evidence, publication-time labels,
as-of forecast, battery schedule and independent historical settlement check.
Deliver a local recorded-run demo, screenshot, two-minute narration, manifest,
resource/cost record and review handoff. The demo can replay a captured real run
without keeping a GPU or cloud service alive. It must say that it is a recorded
run, and cannot imply live model serving or a public SLA.
