# Frozen incremental Agent full run — completed, not promoted

## Result and scope

The one authorized full job **31492138** completed `0:0` in **2:00:52**.
Its 40,606,330-byte private archive has matching local/Spartan SHA-256
`7767b4d7d0465037ec2533afa72c70e0b70e2f66a4d6f08009a60d625624d11b`.
All **432 expected slots** are retained: **24 episodes / 48 authored turns**,
three model-calling paths and seeds 17/29/43. There is no no-model baseline in
this run. Labels are author-generated contracts, not independent human semantic
ratings. November dates are disjoint from the December development pilot but
share the same official AEMO source family. GPU allocation has ended; no second full job or later inference was
submitted. No runtime, prompt, label, scoring rule or original trace was changed.

[Complete compact accounting](../artifacts/public/incremental_full_20260929.json)
retains every paired episode/turn/seed slot and path/episode/fault group, including
failures and unchecked values. [Manifest](../artifacts/public/incremental_full_manifest_20260929.json)
binds inputs, runtime, original archives, reporting code and resource costs.
The independent read-only review agrees with the rewrite and failure diagnoses.

## Completion is not the same as contract matching

Natural tasks comprise 12 episodes / 24 authored turns, repeated to 72 slots per
path. Controlled fault/lifecycle episodes comprise a separate 72 slots per path,
**including their preparation rounds**. The first checkpoint-pause round is
already an intervention; it is not a natural task or completed replay.

| Track and measure | Legacy hybrid | Graph full rerun | Graph incremental |
|---|---:|---:|---:|
| Natural: completed, state + numerical checks passed | 69/72 | 69/72 | 69/72 |
| Controlled: completed, state + numerical checks passed | 51/72 | 48/72 | 48/72 |
| Controlled: recorded contract match | 57/72 | 72/72 | 72/72 |
| Controlled: contract-matched noncompletion | 6 | 24 | 24 |
| Target disposition verified / 36 planned target rounds | 18/36 | 36/36 | 36/36 |
| All tracks: runtime status `completed` | 132/144 | 117/144 | 117/144 |
| All tracks: completed and independently verified | 120/144 | 117/144 | 117/144 |
| All tracks: recorded contract match | 126/144 | 141/144 | 141/144 |

Legacy's 12 extra `completed` rows fail independent checks in four controlled
fault categories; they are not successes. Graph noncompletion includes deliberate
safe rejection and three checkpoint pauses per path. Across paths, 54 recorded
contract matches are noncompletion; only 48 are independently target-verified
safe stops. No unchecked state/numerical value is counted as verified.

Each seed gives the same aggregate outcome counts: Legacy 42/48 contract matches
and 40/48 verified completions; each Graph 47/48 and 39/48. Repeated seeds are not
independent users. These counts show observed stability, not confidence intervals
or general reasoning/production reliability.

### Controlled dispositions by frozen episode

Each entry below is verified target disposition / three second-turn slots.
The artifact separately preserves planned injection, actual firing/exercise and
target verification. Legacy exercised 33/36 targets, Graph 36/36 each; its
checkpoint lifecycle was unavailable, not a failed model recovery choice.

| Episode / intervention | Legacy | Full | Incremental |
|---|---:|---:|---:|
| 13 snapshot content; 14 evidence version | 3/3 each | 3/3 each | 3/3 each |
| 15 checkpoint correction | 0/3, unexercised | 3/3 | 3/3 |
| 16 thread isolation | 0/3 | 3/3 | 3/3 |
| 17 persistent empty evidence | 3/3 safe termination | 3/3 safe termination | 3/3 safe termination |
| 18 one controlled timeout | 3/3 retry then completion | 3/3 | 3/3 |
| 19 malicious evidence | 0/3 | 3/3 rejection | 3/3 rejection |
| 20 unsafe planner prompt | 3/3 | 3/3 | 3/3 |
| 21 settlement conflict | 0/3 | 3/3 rejection | 3/3 rejection |
| 22 missing interval | 3/3 rejection | 3/3 rejection | 3/3 rejection |
| 23 forecast signal; 24 citation hash conflict | 0/3 each | 3/3 each | 3/3 each |

There are 117 planned intervention slots including nine first-round checkpoint
slots; six actual Graph pauses are observed. The 108 second-round target slots
contain 105 exercised and 90 verified dispositions. Nine controlled timeout
cases retry successfully. Neither safe termination nor this deterministic retry
is proof of model-led evidence recovery.

## Model contribution, runtime work and request cost

| Measure | Legacy hybrid | Graph full rerun | Graph incremental |
|---|---:|---:|---:|
| Provider requests / responses | 171 / 171 | 144 / 144 | 144 / 144 |
| Prompt / completion tokens | 544,572 / 64,639 | 429,456 / 59,776 | 429,456 / 59,776 |
| Model-proposed calls, all attempts | 642 | 660 | 660 |
| Initial canonical matches | Unavailable | 135 | 135 |
| Runtime-accepted model proposals, including recovery | Unavailable | 138 | 138 |
| Runtime-rejected proposals | Unavailable | 522 | 522 |
| Actual model-owned executions | Unavailable | 141 | 117 |
| Actual system-completion executions | Unavailable | 408 | 324 |
| Status-filtered tool executions | 879 | 549 | 441 |
| Reused stages | Unavailable | 0 | 108 |
| Sum of measured row wall time, seconds | 2,527.541 | 2,331.762 | 2,329.552 |

Canonical matches demonstrate compliance with the predefined runtime contract,
not that a model designed or changed the normal plan. Accepted proposals,
executions and reuse have different denominators. Legacy's missing ownership
fields are unavailable, not zero or inferred from its final path.

The 1,869 tool records all have executed statuses: 1,761 `ok`, 30 `timeout`,
78 `error`, **zero skipped**. Consequently the recorded tool-item counts equal
status-filtered attempts in this specific archive. The 108 fewer Graph calls
belong to runtime reuse. Both Graph paths use identical tokens; fixed path order
and shared prefix cache prohibit causal speedup or model-efficiency claims.

All 459 provider requests have responses and known usage: **1,403,484 prompt +
184,191 completion = 1,587,675 tokens**, zero provider exceptions/unknown-usage
records. These independently recomputed counts match the archived runner summary.
Only current-row events are counted, never repeated `history` entries. Tool
failures, retries and fallback requests remain charged. Startup health-probe
connection errors were not evaluation requests. No final-answer entailment
audit or public-service SLA is implied by the numerical/citation checks.

## Three trace-backed cases

1. **Failure, not missing evidence:** `vnext-07 / turn2 / legacy / seed17`,
   original journal line 14. The user-sourced cycling cost is correctly 0 from
   turn 2; five official evidence records and a forecast already exist. Dispatch
   times out on all three attempts, about 14.5–14.8 seconds each. The final label
   `insufficient_evidence` is not the root cause. The same authored turn fails
   across all nine path/seed combinations; Graph has two dispatch timeouts per
   row. Recorded failure is the dispatch time budget under zero-cost settings,
   not a demonstrated retrieval-coverage or LLM parameter-generation failure.
   Exact solver internals were not diagnosed or tuned. This row costs three
   provider requests and 11,868/582 prompt/completion tokens.
2. **Real model recovery choice, no recovered evidence:** `vnext-17 / turn2 /
   full_rerun / seed17`, line 82. After a controlled empty result, the model
   removes `battery dispatch` from the scoped query and increases top_k 5→10.
   The runtime accepts and executes the rewrite; the second search is still
   `error/empty_result`, followed by safe termination. This row costs two requests
   and 7,304/538 tokens. Across both Graph paths and three seeds, all **six**
   matched rewrites are actual error-returning executions, **none skipped**.
   Evidence remains 0→0 within the injected turn; the preceding ordinary turn's
   five sources are not newly recovered evidence. No model clarify/stop choices
   occur. Claim bounded model control, not retrieval improvement.
3. **System execution versus model-labelled reuse:** `vnext-01 / turn2 /
   incremental / seed17`, line 98. The change to 1MW/3MWh reuses snapshot,
   evidence and forecast, then executes only system-owned dispatch. All nine
   checks pass. The snapshot provenance says `model_accepted` but its action is
   **reuse**, not a new model tool execution. This row costs one request and
   3,595/456 tokens; both planned and realized operating proxies are AUD 0.00.
   This demonstrates a traceable valid decision, not positive investment return.

An additional wording caveat is retained: nested dispatch `economic_boundary`
uses a generic user-supplied-sensitivity phrase even in the third case's default
cost setting. The top-level Graph answer attribution was repaired earlier; this
run is not a complete wording-provenance certification. Raw outputs are unchanged.

## Resources, reproducibility and stop condition

Executed runtime **`b9791d3` / source06**, not the later reporting commit. One
20GB MIG slice, 6 CPU, 16GiB RAM, 10,000MiB scratch; five-hour hard limit and no
automatic requeue. Actual runtime inference threadpool is 6; batch=6 is a request
setting, not a separately measured count. MaxRSS is **8,589,688KiB**; Slurm
TotalCPU displays **02:02:03**. Allocated MIG time is 7,252 seconds. GPU peak
memory and infrastructure AUD price are unavailable. Paid provider cost is zero.

Including initial failed infrastructure job 31490736 and both pilots, retained
campaign cost is **7,615 allocated MIG seconds**, 483 requests and 1,689,069 tokens.
Four read-only observer steps belong to the full allocation, not extra GPU jobs;
their resource rows are retained. Wall-time sums do not replace allocation cost.

Extract the private archive locally, then use the existing read-only entry:

```bash
python scripts/summarize_incremental_full.py \
  --attempts "$FULL_ARCHIVE_DIR/results/attempts.jsonl" \
  --scenarios benchmarks/incremental_vnext_holdout_v2_20260929.jsonl \
  --scenario-sha256 9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f \
  --output full-summary-reproduced.json
```

Reporting code is `cc13dbe`; its private and public outputs were reproduced with
identical SHA `102e33a05c5e06116ea117ab07b7ca1c7a94293a2af80bee92388767561c00f7`.
The same artifact contains 144 paired episode/turn/seed entries, each retaining
all three paths. No GPU, model or optimization is needed to replay this analysis.
The 11 small accounting fixtures were verified in the earlier reporting package;
no full-suite/CI/model rerun was performed for this terminal report.

**Not promoted:** runtime guarding/reuse is supported; superior model quality,
useful evidence recovery and controlled latency gains are not established. The
project remains historical replay with explicitly bounded operating proxies,
excluding CAPEX, fixed O&M, network fees, FCAS and investment returns. No merge,
deployment, resume/career-directory edit or subsequent GPU request is authorized
or performed by this closeout. Climate remains a separate unreleased task.
