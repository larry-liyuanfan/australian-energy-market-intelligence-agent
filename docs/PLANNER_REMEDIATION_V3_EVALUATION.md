# Planner/Memory v3 — completed remediation evaluation

## Decision: retain the deterministic planner

The frozen run completed, but **the model is not promoted**. Structured-state
hybrid completed **54/54** scored turns, while the model's initial complete path
was **18/54 (33.3%)** and selected-parameter accuracy **88.9%**. Deterministic
structured state completed **18/18**; both reached 100% on their memory-required
subset. The hybrid therefore added no measured task-quality benefit on this set
and increased measured P95 from **0.787 s to 22.567 s**. This is a real negative
model-selection result, not a failed delivery or a reason to tune the holdout.

The [public aggregate](../artifacts/public/planner_remediation_holdout_v3_20260927.json)
contains every path/memory/track metric. The [run manifest](../artifacts/public/planner_remediation_run_manifest_20260927.json)
binds its hashes, execution identity and resource record. The completed P1
[recorded replay](INTERVIEW_DEMO.md) is separate evidence of an attributed,
runtime-assisted product flow, not autonomous model planning.

## What was repaired and what stayed frozen

The old holdout and GoalSpec results are retained without relabelling. Diagnosis
found unavailable dates in the old GoalSpec pilot, correction text being reparsed
after sourced state was resolved, and scoring that combined initial proposals
with replans. The repair applies typed state directly, validates real interval
coverage before inference, rejects oracle substitution, and records initial
proposals, guarded calls and execution separately. Exact selected-parameter checks
replace the old substring checks. See the [diagnosis and review](REMEDIATION_REVIEW_20260927.md).

The independent GoalSpec pilot executed after its native-endpoint compatibility
repair but still achieved **0/8 task success**, 50% valid objects and field F1
0.8571. Incorrect correction-source turns and missing requested outputs remained;
it was not selected for this holdout or serving. ViDoRe was not resumed. No new
typed tool, model training, model size expansion or historical-value metric was
added to make the project appear stronger.

The v3 tasks, scorer, sampling and thresholds were fixed before inference at
`72323060ad7b2314facd82cf8064f1f0c174b308`. They were not changed after observing
results. Different tasks and scoring mean **no numerical v1-to-v3 lift claim**.

## Design, data and completed-run integrity

- Ten author-written episodes / 18 turns, with dates and prompts disjoint from
  earlier files. These are not independently human-labelled tasks.
- Three paths × four memories. Each deterministic group has 18 rows, seed 0;
  each model group has 54 rows, seeds 17/29/43 at temperature 0.2. Each contains
  the same 18 case/turn identities. Total: **504 scored rows**, not 504 unique tasks.
- Per repetition: 11 non-tool-fault turns, seven injected-fault turns, eight
  memory-required turns. The non-fault track includes prompt injection and is not
  synonymous with benign queries. Model denominators are 33/21 by track and 24
  for the memory-required subset; deterministic denominators are 11/7 and eight.
- Real official five-minute AEMO data: 525,600 rows, five regions, 18 August 2025
  to 17 August 2026. Preflight verified all 15 holdout region-days have 288 exact
  timestamps plus prior history. Official text and as-of seasonal snapshots are
  used; no workbook figures are mounted in this direct-tool holdout.
- Real Qwen3-8B Q4_K_M, pinned llama.cpp, one 20 GB A100 MIG allocation, loopback
  inference. No provider credential or mock inference substitutes for this run.
- Job **31365531**: **COMPLETED**, exit **0:0**, **01:02:12**. Final prediction
  count is 504. Prediction and metrics SHA-256 match the final manifest; local
  benchmark/gate hashes match the frozen input hashes. Twelve metric groups and
  24 track groups are present. Actual seeds and case/turn counts were checked.
- The run made **549 model requests**: 432 initial requests plus **117 replans**.
  There were 117 model-path retries and 22 deterministic retries, **139 total**;
  29 model proposals were rejected. Maximum observed tool attempts were seven
  (budget eight), and planner attempts three (initial plus at most two replans).

## Planner and memory comparison

Task success below is the post-execution composite. Initial path/parameters refer
only to the first validated model proposal, before guard completion or replanning.
Deterministic raw-model fields are not applicable. Pass@1 equals the sampled task
rate here; pass-all-3 requires all three seed samples of a case/turn to succeed.

| Path | Memory | Task successes | Initial path | Initial parameters | Recall | Pass-all-3 | P50 / P95, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic | None | 10/18 | — | — | 0% | — | 0.246 / 0.787 |
| Deterministic | Full | 18/18 | — | — | 100% | — | 0.228 / 0.801 |
| Deterministic | Sliding | 17/18 | — | — | 87.5% | — | 0.240 / 0.775 |
| Deterministic | Structured | 18/18 | — | — | 100% | — | 0.235 / 0.787 |
| Pure LLM | None | 17/54 | 50.0% | 68.2% | 0% | 27.8% | 5.580 / 21.745 |
| Pure LLM | Full | 27/54 | 57.4% | 94.3% | 100% | 44.4% | 7.404 / 21.350 |
| Pure LLM | Sliding | 24/54 | 57.4% | 91.9% | 87.5% | 38.9% | 7.383 / 22.663 |
| Pure LLM | Structured | 15/54 | 33.3% | 88.9% | 100% | 27.8% | 5.267 / 21.359 |
| Hybrid | None | 30/54 | 50.0% | 68.2% | 0% | 55.6% | 7.564 / 21.714 |
| Hybrid | Full | 54/54 | 51.9% | 94.3% | 100% | 100% | 7.853 / 17.865 |
| Hybrid | Sliding | 51/54 | 57.4% | 94.3% | 87.5% | 94.4% | 7.858 / 23.551 |
| Hybrid | Structured | 54/54 | 33.3% | 88.9% | 100% | 100% | 5.820 / 22.567 |

Full and structured state preserve sourced constraints for the runtime, but that
does **not** establish that structured context improves the model's planning:
pure LLM scored 15/54 with structured state versus 27/54 with full history.
The no-memory and sliding losses also appear in the deterministic baseline;
restoring missing user context is not a model-only benefit. Hybrid matched the
corresponding deterministic task rate in **all four memory modes**.

Descriptive 95% Wilson intervals: structured hybrid 54/54, **93.4–100%**;
deterministic structured 18/18, **82.4–100%**; pure structured 15/54,
**17.6–40.9%**. These are turn/sample intervals, not independent-task confidence
guarantees: turns share episodes, and repeated seeds share prompts. All group
intervals are in the public aggregate. Fixed serial evaluation is not a concurrent
service load test or randomised latency experiment.

## Fault-separated outcomes and recovery

| Path / memory | Non-tool-fault success | Injected-fault success | Required recovery success |
| --- | ---: | ---: | ---: |
| Deterministic / none | 5/11 | 5/7 | 2/2 |
| Deterministic / full | 11/11 | 7/7 | 2/2 |
| Deterministic / sliding | 10/11 | 7/7 | 2/2 |
| Deterministic / structured | 11/11 | 7/7 | 2/2 |
| Pure LLM / none | 11/33 | 6/21 | 2/6 |
| Pure LLM / full | 18/33 | 9/21 | 2/6 |
| Pure LLM / sliding | 15/33 | 9/21 | 2/6 |
| Pure LLM / structured | 12/33 | 3/21 | 1/6 |
| Hybrid / none | 15/33 | 15/21 | 6/6 |
| Hybrid / full | 33/33 | 21/21 | 6/6 |
| Hybrid / sliding | 30/33 | 21/21 | 6/6 |
| Hybrid / structured | 33/33 | 21/21 | 6/6 |

The suite exercises timeout, empty evidence, malicious evidence, conflicting
tool output, stale snapshots and prompt injection. Required recovery is measured
on two turns per repetition, not on every fault-labelled turn. A successful
replan alone does not repair an incomplete overall path. Deterministic recovery
is counted separately from model replanning, with the same bounded execution
budget. Results are not universal prompt-injection resistance.

Structured hybrid: initial **18/54 → guarded 54/54 → executed 54/54**, with 114
reported fallback calls across initial plans and recovery. Initial omissions were
evidence search 24, snapshot 15, event detection nine and forecast six (overlapping
counts, not unique failed tasks). Pure structured retained only 15/54 successful
executed paths; its failed rows contained 39 path failures, five failed required
recoveries and six selected-parameter mismatches, with overlaps.

The safety/check metrics must not be oversold. All paths reported zero unsafe
tool/DSL proposal errors, and all returned 100% settlement-mode consistency;
the latter is not independent numerical settlement validation. Citation-structure
rates were 100% except no-memory: 83.3% deterministic/hybrid and 72.2% pure LLM.
No-memory contamination was 5.6% in all paths (one deterministic / three model
rows); other modes reported zero on the specified forbidden-value checks.

## Steps, requests, tokens and cost

| Path / memory | Mean tool attempts | Model requests | Prompt / completion tokens | Retries |
| --- | ---: | ---: | ---: | ---: |
| Deterministic / none | 3.444 | 0 | 0 / 0 | 14 |
| Deterministic / full | 3.389 | 0 | 0 / 0 | 2 |
| Deterministic / sliding | 3.444 | 0 | 0 / 0 | 4 |
| Deterministic / structured | 3.389 | 0 | 0 / 0 | 2 |
| Pure LLM / none | 2.389 | 81 | 175,262 / 13,650 | 27 |
| Pure LLM / full | 2.444 | 60 | 156,226 / 11,796 | 6 |
| Pure LLM / sliding | 2.556 | 66 | 168,805 / 12,063 | 12 |
| Pure LLM / structured | 2.056 | 60 | 156,701 / 10,429 | 6 |
| Hybrid / none | 3.444 | 96 | 208,014 / 13,254 | 42 |
| Hybrid / full | 3.389 | 60 | 168,663 / 11,545 | 6 |
| Hybrid / sliding | 3.444 | 66 | 194,391 / 12,145 | 12 |
| Hybrid / structured | 3.389 | 60 | 166,923 / 10,424 | 6 |

Total tokens: **1,394,985 prompt + 95,306 completion**, including recovery. All
groups report **AUD 0 external-provider billing per task** because inference used
a university allocation, not a billed hosted provider. This does not mean free
compute. The job requested six CPUs / 16 GiB host RAM and one 20 GB A100 MIG,
used 11,281,388 KiB batch MaxRSS (10.759 GiB), and consumed 1.0367 MIG-allocation
hours / 6.22 allocated CPU-hours; measured aggregate CPU time was 01:01:24.
No commercial GPU tariff or whole-A100 equivalence is invented. These costs cover
this frozen run; failed pilots and the successful demo are separately preserved
in the remediation review and their manifests, not erased from project history.

## Frozen promotion gate

The designated structured-hybrid candidate passed post-execution task success,
citation structure, settlement-mode consistency, bounded recovery, region/date
recall, specified contamination and unsafe-call thresholds. It failed:

| Required condition | Observed | Decision |
| --- | ---: | --- |
| Initial complete tool path ≥90% | 18/54 = 33.3% | Fail |
| Initial selected-parameter accuracy ≥90% | 88.9% | Fail |
| Exceed deterministic memory-required task success | 24/24 versus 8/8, both 100% | Fail |

No alternative memory mode is selected after the result to bypass the frozen
candidate rule. Full-history hybrid also did not exceed its deterministic
baseline. The deterministic DAG stays authoritative; the model integration is
retained as a reproducible negative experiment. Successful execution is not
evidence that Qwen autonomously planned all stages.

## What the reported checks mean

Task success is an executed-path/selected-parameter/structural-check composite,
not answer-semantic accuracy. No-memory waives recall but retains other checks.
Parameter accuracy excludes search semantics and missing-call completeness;
those omissions have a separate path metric. Citation correctness checks HTTPS
and hash length (runtime also checks hexadecimal format), not original-document
rehashing or entailment. Settlement consistency checks field presence for the
margin basis, with no-dispatch cases passing by non-applicability. Memory recall
checks region/date values, not source-turn attribution or every remembered fact.
Contamination covers benchmark-specified forbidden values, not all possible errors.
The full [metric definitions](REMEDIATION_REVIEW_20260927.md#reading-the-frozen-v3-metrics)
are part of this report's interpretation, not a changed scorer.

Publication review corrected one diagnostic label: `failure_causes_nonexclusive`
became `failed_row_observations_nonexclusive`. For example, false recall on a
no-memory failed row is co-occurring, not a cause under its waived recall rule.
These observations are neither causal nor an exhaustive list of failed gates;
unsafe proposal errors are reported separately. The public compact records the
original compute-summary SHA and the diagnostic-only revision. Its metrics and
run manifest are unchanged; no inference, rescoring, relabelling of tasks or gate
change occurred. Four new exporter tests cover score preservation, the waived
condition, invalid inputs, source hashing and no-overwrite behaviour.

To reproduce this publication-only transformation in the current reviewed
checkout, without model inference or raw prediction access:

```bash
python scripts/summarize_planner_remediation.py \
  --relabel-summary PATH_TO_ORIGINAL_COMPUTE_SUMMARY_JSON \
  --output NEW_PUBLIC_SUMMARY_JSON
```

## P1, engineering verification and handoff

The real-Qwen SA1 15 December 2025 recording passed 13 export checks, including
independently recomputed cash flow and SoC,
and browser visual review. Two proposed stages were accepted; three were replaced
by runtime guards and diagnosis was a runtime dependency. It includes text and
117-record Q4 workbook context, as-of seasonal forecasting, SoC, independent
cash-flow/constraint checks, and explicit planned versus realised margins. Quarterly
reports are retrospective background, not day-specific causal proof or forecast
inputs. Gross values and the assumed cycling-cost sensitivity remain historical
operating proxies, excluding CAPEX/network charges/FCAS and investment returns.

CPU preflight 31365441 used a separate exact-commit checkout and passed **222
tests**, Ruff, strict mypy and all 20 development/holdout region-days. CI at
`1b7f681` passed in a fresh runner environment, including vulnerability and
high-confidence secret scans. The current [environment record](ENVIRONMENT_STATUS.md)
states which optional dependencies were **not** installed/audited; no blanket
checkpoint or transitive-license certification is implied.

After the diagnostic-only export change, local validation passed **226 tests**,
Ruff and strict mypy over 74 source files. One existing Starlette/httpx deprecation
warning was emitted; there was no test failure or unrequested dependency upgrade.
The final publication commit is also checked by the PR's fresh-runner CI.

Reproduction: use the [recorded page and Slurm template](INTERVIEW_DEMO.md), including
the exact commit variables, `direct_demo`, resource review, `sbatch --test-only`
and successful-pilot dependency. The real-data rerun requires authorised private
inputs and the pinned runtime; the committed HTML can be viewed without them.
Compact aggregates/hashes are public; raw market data, workbook, model weights and
per-turn traces remain private. See the [completion handoff](P0_P1_HANDOFF_20260927.md)
for requirement-by-requirement status. This package does not merge the PR, activate
an SG model, or edit the current resume. No further experiment is required to
complete this bounded negative-result delivery.
