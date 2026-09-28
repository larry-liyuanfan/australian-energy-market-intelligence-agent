# Incremental Agent — real Qwen integration pilot

Validation date: **2026-09-29**. This is the single coordinator-authorized development pilot,
not the frozen full evaluation. The executed runtime remains `6026fa9c668a01163c4fd31f1df0d82d6a88cf75`;
the reporting rules are from `f66a0b3`. No application, prompt, model or task configuration
was tuned after observing quality. This handoff does **not** authorize full execution.

## Outcome

**Real model integration was observed; model benefit was not demonstrated.** There were
12 actual provider requests and 12 responses, zero provider exceptions and zero unknown
usage records. All twelve path-turns completed with the expected sourced region/date/battery
state and nine independent numerical/citation-location checks. Corresponding historical
settlements match across the three paths. These are four linked author-written development
turns on one market day, in one seed—not twelve independent tasks or a holdout pass rate.

The graph accepts only the snapshot proposals in turns 1 and 3. Evidence, forecasting and
dispatch remain system-completed. There were **zero recovery opportunities**, so model
rewrite/clarify/stop decisions are **not exercised**. No live recovery benefit is claimed.
An answer-attribution defect remains: eight graph answers incorrectly call the default
cycling-cost sensitivity “user-supplied”. This is reported, not repaired in the recorded run.

Evidence: [compact per-turn report and manifest](../artifacts/public/incremental_live_pilot_20260929.json)
(SHA `8f3b473098dad1caa4f9729125a5791c0812999e318a93ace06e38d4daa07ecd`).
The raw attempt journal, SQLite files, source material and server logs remain private.

## Exact run and resources, including failure

| Job | Outcome | Wall time | Peak host RSS | What occurred |
| --- | --- | ---: | ---: | --- |
| 31490736 | Failed `1:0` | 18 s | 5,307,172 KiB | Model loaded; LangGraph import then failed on missing `libssl.so.1.1`; zero evaluation requests |
| 31490840 | Completed `0:0` | 173 s | 8,003,276 KiB | One same-configuration infrastructure retry; all 12 requests retained |

Both allocations: 1×`gpu:1g.20gb`, 6 CPUs, 16 GiB RAM, 10,000 MiB scratch, 30-minute limit.
Total allocated MIG time including failure: **191 seconds**; actual recorded CPU time is
13.401 + 172.798 seconds. No paid provider API was used. An infrastructure AUD price was
not supplied, so zero API spend is not zero computing cost.

One read-only observation at job elapsed ~85 seconds found **6,972 MiB** GPU memory for
PID 250136, subsequently matched to the exact job alias and model command. This is **not
peak GPU usage**; peak is unavailable. Three brief observer steps used the same allocation,
not additional GPU jobs. The server logged 32 default threads; six is the Slurm CPU allocation,
not a claim that the server created only six threads. Both pilot jobs are terminal.

The infrastructure repair only supplied existing OpenSSL 1.1 and SQLite 3.42 library paths
through the Slurm environment. The GPU module setup had omitted the older Python's SSL
dependency and otherwise selected system SQLite 3.34.1. The corrected import preflight
reported OpenSSL 1.1.1t, SQLite 3.42.0 and SciPy 1.17.1. No shared environment or driver was
installed, and no source archive was modified. One infrastructure retry was used; no quality retry.

Runtime identity:

- Bundle `energy-runtime-05.tar`: `3bc36b1575637b6d9f901735914015ca9b147dac9da6da80a4834f2ab349d45e`.
- Launcher: `c3985ab37d42a4725dd3d2448cb2417669442f1e27d7a849eff1b9a94e95cfbf`.
- Qwen3-8B Q4_K_M revision `1d54a16a18cba0d8fbad4a16db801decc729e099`, 5,027,783,488 bytes; SHA `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- `llama-server` binary SHA: `1ea585cf863473bfffb36dfd8725c7f6722b0f47d0a7cdd91e082ff3d6fc54d5`.
- Successful service alias: `energy-31490840-energy-model.0Js3ML`, numeric loopback port 11629, one slot and 16,384 context.

The server log records model loading and listening; the PID/socket/alias check passed before
the runner could issue requests. Per-request `response_received` and measured usage, plus
server inference timing records, establish actual inference rather than a configured flag.
The initial connection-refused/503 lines occur during bounded model startup, not failed
evaluation requests. Every wheel and input SHA was verified by the unchanged launcher/runner.

## Normal turns: integration and overhead

| Path | Calls in turns 1–4 | Prompt / completion tokens | Four-turn wall time | Model contribution |
| --- | --- | ---: | ---: | --- |
| Existing constrained hybrid | 6 / 6 / 5 / 5 | 15,683 / 996 | 48.328 s | Different event-inclusive runtime; no directly comparable per-call owner field |
| Graph full rerun | 4 / 4 / 4 / 4 | 15,776 / 1,233 | 54.819 s | 2 snapshot proposals accepted; 14 executions system-completed |
| Graph incremental | 4 / 1 / 4 / 1 | 15,776 / 1,233 | 50.474 s | 2 snapshot proposals accepted; 8 executions system-completed; 6 separate cache reuses |

Total: **47,235 prompt + 3,462 completion tokens**, 12 requests, no request retries or
provider failures, and no hidden unknown usage. Each graph's prospective canonical acceptance
is 2/16 planned calls. This is a strict contract diagnostic, **not general tool-use accuracy**:
an extra registered event tool is outside this minimal graph, not an unsafe tool; the
proposed search differs from the scoped canonical query; forecast proposals use 12 intervals
instead of 288; dispatch proposals use the tool's zero-cost default rather than the runtime's
50 AUD/MWh policy. Default-policy alignment is not the same as user-constraint recall.

The LLM proposes dispatch after battery/RTE changes, but the runtime replaces it under that
policy mismatch. The 4/1/4/1 execution pattern therefore demonstrates **runtime dependency
reuse**, not successful LLM optimisation of the workflow. Graph token counts are identical:
no model-token saving was observed. The earlier deterministic CPU pilot already completed
these cases with the same settlements; this integration run adds no demonstrated task benefit.

Timings are ordered samples, not P50/P95, confidence intervals or a controlled speedup.
The same llama.cpp process shares prefix-cache state across paths. Its last full-rerun turn
prefills 2,482 tokens; the last incremental turn prefills only 1, although logical prompt usage
is 4,881 in both. That cache difference confounds the wall-time comparison. Do not attribute
the difference to domain-tool reuse or claim elimination of internal forecast/MILP work.

## Source-state and economic checks

The development runner has no authored `expected` labels, so its automatic
`sourced_state_contract=true` is not an independent parameter test. A post-run, read-only
check explicitly compared all twelve records with the four user edits below, including the
retained date, dispatch parameters and source turns. A separate reviewer confirmed the same
result without rerunning the optimiser or model.

| Turn | User constraint state; date always 2025-12-15 | Planned / realised net operating proxy, AUD |
| --- | --- | ---: |
| 1 | SA1, 1 MW / 2 MWh, RTE 90% | 177.49 / 401.90 |
| 2 | SA1, 1 MW / 3 MWh, RTE 90% | 239.90 / 647.86 |
| 3 | VIC1, 1 MW / 3 MWh, RTE 90% | 234.82 / 110.11 |
| 4 | VIC1, 1 MW / 3 MWh, RTE 80% | 242.57 / 106.04 |

All corresponding settlements match across paths within 1e-5 AUD. Retained independent
checks cover power, charge/discharge exclusivity, SoC, initial/terminal SoC, energy balance,
planned cashflow, realised cashflow, forecast-signal binding and citation source location.
Source-location validity is not semantic entailment or daily causal evidence. Only sourced
state memory was used; this pilot is not a new comparison of four memory modes.

**Known defect, not a pass:** all 8 graph answers say “user-supplied” cycling cost, but none
of these turns contains a user-sourced degradation constraint. The value 50 AUD/MWh discharged
is a default sensitivity assumption. Correct attribution remains a follow-up requirement;
the original answers and metrics are not rewritten. Financial calculations are net of that
assumption, not gross margins, realised investment returns or observed degradation costs.
CAPEX, network fees, FCAS and other investment costs remain outside this historical proxy.

## Recovery ledger and promotion boundary

Recovery opportunities: **0**. Model rewrite / clarify / stop choices: **not exercised**.
Actual recovery calls, new evidence gained and recovery-only cost: **not applicable**,
not a successful zero-cost recovery. No prompt-injection or persistent-empty model trial was
added to this four-turn pilot. Passing a future persistent-empty fault contract cannot by
itself prove an evidence gain. The existing frozen full set remains unconsumed.

The defensible result is a real model/runtime integration trace and measured overhead,
plus the unchanged dependency-reuse effect. A model-value or promotion claim is unsupported.
No quality-pass receipt has been created, no full job submitted, no service deployed and no
career file changed by this project task. Coordinator review and a new explicit release
are required before any further GPU execution.

## Reproduction of the recorded configuration

The original authorization is consumed; these are reproduction instructions, not permission
to submit again. Use the pinned runtime, model, wheels and private inputs from the
[CPU manifest](../artifacts/public/incremental_cpu_manifest_20260929.json). With a fresh
authorization, set `ENERGY_STAGE=pilot`, the bundle path/hash above and
`ENERGY_GPU_RELEASE_ID`. Export the following existing-library paths alongside those variables:

```bash
: "${ENERGY_GPU_RELEASE_ID:?fresh coordinator authorization required}"
export ENERGY_GPU_RELEASE_ID ENERGY_STAGE=pilot
export ENERGY_BUNDLE=/data/gpfs/projects/punim2936/portfolio_20260820/energy-agent/incremental-agent-20260929/energy-runtime-05.tar
export ENERGY_BUNDLE_SHA=3bc36b1575637b6d9f901735914015ca9b147dac9da6da80a4834f2ab349d45e
LD_LIBRARY_PATH=/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib:/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/SQLite/3.42.0/lib
export LD_LIBRARY_PATH
# From the dedicated Energy directory; fresh authorization is required.
sbatch --test-only --time=00:30:00 --cpus-per-task=6 --mem=16G --tmp=10000 \
  --gres=gpu:1g.20gb:1 --export=ALL --output=pilot-%j.out --error=pilot-%j.err incremental_model.sbatch
# Remove only --test-only for the authorized single submission.
```

This selects the unchanged `--development --seeds 17` runner. No `--consume-frozen` or
full-stage option is used. Private archives are `pilot-31490736.tar.gz` and
`pilot-31490840.tar.gz` under the dedicated Energy artifact directory; exact hashes for
archives, attempt journal, summary and server log are in the compact report.
