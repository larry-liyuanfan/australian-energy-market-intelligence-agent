# Incremental replay: bounded repair confirmation

## Outcome

The sole authorized confirmation pilot, **31491560**, completed in 172 seconds.
All 12 path-turns completed with real Qwen responses. All eight graph answers now
attribute the unchanged AUD 50/MWh discharged cycling-cost sensitivity to a
**default assumption**, not to the user. All 108 existing independent numerical
and citation checks passed, and each planned/realized settlement matches original
job 31490840 within AUD 0.00001.

This confirms two local repairs, **not a model-quality improvement or promotion**.
The original pilot and its eight wrong attribution strings remain immutable.
No full run, SG/public deployment or resume modification was performed. No
quality-pass receipt has been created; independent review is still required.

Evidence: [pre-submission freeze](../artifacts/public/incremental_confirmation_freeze_20260929.json),
[per-turn confirmation](../artifacts/public/incremental_confirmation_20260929.json),
[run/resource manifest](../artifacts/public/incremental_confirmation_manifest_20260929.json),
and [original pilot including failure](INCREMENTAL_LIVE_PILOT_20260929.md).

## Exactly what changed

Runtime commit `b9791d3f134172c2ebba9ac9f5e8c619ec247cbb` changes only:

1. The deterministic answer template reads the sourced constraint. Default cost
   is labelled an assumption; explicit user cost includes the original source
   turn. Equal numeric values do not imply equal provenance. This answer text is
   not inserted into model memory or planner context.
2. The launcher validates `SLURM_CPUS_PER_TASK` and passes both `--threads` and
   `--threads-batch`, with matching `OMP_NUM_THREADS` and `OMP_THREAD_LIMIT`.

Actual GPU-node help supports both options. Requested limits were all 6; the
retained log reports `llama threadpool init, n_threads = 6`. Batch thread count
is recorded as a **requested setting**, not an independently observed count.
The original 32-thread default did not itself establish a CPU-allocation breach.
Login-node help was unavailable because its CUDA driver library is absent; no
model inference ran on the login node and no shared-cluster installation was made.

Model weights/revision, prompts, action policy, four development questions, seed
17, three paths, data/snapshots and frozen full protocol were unchanged. The
launcher retained the already verified OpenSSL/SQLite library-path repair.

Eight added CPU cases cover default/explicit 50/20/0, sourced overrides, cached
same-value attribution changes, isolated threads and SQLite restart/approve/correct.
All 20 graph-file cases passed locally. Runtime CI [36466915797](https://github.com/larry-liyuanfan/australian-energy-market-intelligence-agent/actions/runs/36466915797)
actually ran 257 main tests (2 skipped) and 49 optional graph tests, Ruff, strict
mypy over 89 sources, dependency audit and an installed ripgrep secret scanner.

## Real-model accounting and remaining negative result

| Path | Prompt / completion tokens | Actual tool calls by turn | Canonical proposals accepted by turn | Four-turn elapsed seconds |
|---|---:|---|---|---:|
| Legacy | 15,683 / 996 | 6, 6, 5, 5 | Not directly comparable | 48.288 |
| Full rerun | 15,776 / 1,233 | 4, 4, 4, 4 | 1, 0, 1, 0 | 54.887 |
| Incremental | 15,776 / 1,233 | 4, 1, 4, 1 | 1, 0, 1, 0 | 50.464 |

There were 12 responses from 12 requests, no evaluation request retries, no
provider exceptions and no unknown-usage records. Raw provider token counts were
independently summed and compared with the summary. Total: **50,697 tokens**.
Startup connection-refused/503 health probes were not evaluation requests.

Normal canonical planning still supplies most of the successful workflow.
Each graph accepted only two of 16 required canonical calls; this strict contract
diagnostic is not general tool-use accuracy. Reuse reduced runtime executions,
not model tokens. No recovery opportunity occurred, so no rewrite/clarify/stop
benefit was validated. Prior deterministic success and the earlier negative
Planner/Memory experiment are not superseded.

The timings above preserve costs, not a speed claim: this is one ordered sample
per turn with shared prefix caching, and both template and thread configuration
changed between pilots. They cannot support P95/SLA, confidence intervals or
causal cache/model acceleration claims.

The unchanged planned/realized amounts are respectively 177.487/401.900,
239.904/647.862, 234.820/110.114 and 242.565/106.039 AUD. These are historical net
operating proxies under assumed variable cycling cost, not gross spot margin or
investment returns. CAPEX, fixed O&M, network fees and FCAS remain excluded.

## Resources, retained failures and audit reproduction

The same allocation was used: one 20GB MIG slice, 6 CPU, 16GiB RAM, 10,000MiB
scratch and a 30-minute limit. Confirmation MaxRSS was 8,009,000KiB; TotalCPU was
172.463 seconds. No GPU peak-memory observation was collected for this job.

Including failed infrastructure job 31490736 (18 seconds, no evaluation requests),
original pilot 31490840 (173 seconds) and confirmation 31491560 (172 seconds),
allocated MIG time totals **363 seconds**, TotalCPU 358.662 seconds, with 24 model
requests and 101,394 tokens. Paid API cost is zero; infrastructure AUD cost is
unknown. No failure was removed or hidden as a free retry.

Source bundle `energy-runtime-06.tar` is SHA
`b54b2a54289f560c351c1955973c555f4d4548fc01e85e878c96d1203cc59b67`;
the new launcher has its own recorded SHA. The source-bound full gate cannot use
the old pilot as a receipt for this bundle. Private archives, full responses,
SQLite databases and model assets are not uploaded to GitHub.

After extracting the retained private confirmation archive into `$PILOT_DIR`,
the read-only, fail-closed report can be reproduced without model execution:

```bash
python scripts/audit_incremental_confirmation.py \
  --folder "$PILOT_DIR" \
  --freeze artifacts/public/incremental_confirmation_freeze_20260929.json \
  --parent-report artifacts/public/incremental_live_pilot_20260929.json \
  --output confirmation-reproduced.json
```

The auditor verifies source turns, the complete timezone-aware dispatch window,
cost and settlement mode, raw usage/seed accounting, unchanged settlement and
every observed threadpool-init value. It writes a new file, never repairs the
original model outputs, and cannot emit a promotion receipt. The auditor is a
post-run verifier, not additional code in the executed runtime bundle.
