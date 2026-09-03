# Model planner and memory evaluation

## Decision in one sentence

The real-model path is **not promoted**. On the frozen holdout, structured-state constrained hybrid reached **57/72 (79.2%)** task success, below the preregistered 85% threshold, and the model itself produced the complete required tool path on only **25/72 (34.7%)** attempts, below 90%. The deterministic DAG remains the production authority; the model integration is retained as a reproducible negative experiment and a diagnostic of where typed guards add value.

This study tests one capability boundary: whether a real language model can select and parameterise the existing eight typed tools, recover from bounded failures, and use sourced cross-turn state. It does not add tools, change the historical market backtest, or turn the service into a live trading system.

## What changed

The deterministic `DecisionCase` DAG remains the execution authority and fixed baseline. A model planner can propose only registered calls. The runtime then:

1. rejects unknown tools, SQL, Elasticsearch DSL, shell/code fields and optimisation expressions;
2. validates every argument through the canonical Pydantic model;
3. enforces the deterministic workflow, temporal window and data/version boundary;
4. executes tools with step, duplicate, timeout and replan budgets;
5. verifies citation URLs/hashes and planned-versus-realised settlement semantics;
6. records the model proposal separately from the calls actually executed.

Three paths are scored independently:

- `deterministic`: fixed DAG, no model call;
- `pure_llm`: execute only validated model proposals, exposing omissions;
- `constrained_hybrid`: use valid model proposals but fill missing or unsafe workflow stages from the deterministic DAG.

This separation prevents a perfect post-guard result from being presented as perfect model planning.

## Memory contract

Four modes run over the same multi-turn cases:

| Mode | Next-turn input | Trust boundary |
|---|---|---|
| No memory | Current user turn only | Prior constraints deliberately unavailable |
| Full history | Prior attributed user turns plus validated tool summaries | No model prose promoted to fact |
| Sliding window | Last two attributed turns and matching tool summaries | Bounded context; older raw turns evicted |
| Structured state | Latest sourced constraints plus the last eight validated tool summaries | Every constraint retains source turn/type |

Redis and trace retention are not called memory here. `ConversationMemory` must affect the next plan, remain conversation-scoped and pass correction, eviction and contamination tests.

## Frozen evaluation design

The development and holdout prompts are source-separated files. The holdout contains 13 episodes and 24 turns covering region/date correction, prior evidence reference, BESS constraint edits, multi-region comparison, user correction, prompt injection, malicious evidence, conflicting tool output, stale forecast snapshots, timeout recovery, empty-result recovery and a four-turn memory-horizon case. Dates and prompts are absent from the earlier 60-real/20-fault deterministic suite.

The promotion file was committed before the valid holdout run. Qwen3-8B is sampled at temperature 0.2 and seeds 17, 29 and 43 for each model path and memory mode. Deterministic runs use one seed. The result therefore contains 96 deterministic turn attempts and 576 base model turn attempts, plus visible replan calls where faults occur.

Metrics are task success, executed and raw-model tool path, executed and raw-model parameter accuracy, citation correctness, settlement consistency, replanning success, memory recall, state contamination, unsafe calls, steps, rejected calls, retries, tokens, P50/P95 latency and resource usage. Pass@1 and pass-all-k are calculated per path and memory mode; no retry is hidden as an initial success.

## Real runtime and provenance

- Provider: loopback OpenAI-compatible llama.cpp server on Spartan.
- Model: Qwen3-8B Q4_K_M GGUF, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- llama.cpp commit: `7798007a29a90e3053e799394da48cf53a2f8e0f`.
- Data: official AEMO market store, official evidence index and as-of forecast snapshots, each bound by manifest hash.
- Runtime: one 20GB A100 MIG allocation; model staged in Slurm scratch and server bound to `127.0.0.1`.
- External provider cost: AUD 0. The report does not convert university research allocation usage into a fictitious commercial price.

Ollama was attempted first, but its first-run state could not be created within the Iris home quota. The project did not modify existing SSH material or global home configuration; it moved to a project-scoped pinned llama.cpp build and job-local model scratch.

An initial holdout submission (`29927485`) was cancelled after 12 minutes, before any metrics or prediction file was written, because review found that temperature zero made its seed repetitions greedy rather than sampled. Its cancelled state and roughly 5.74 GiB MaxRSS remain disclosed; it is excluded from quality and pass@k results.

A second submission (`29928210`) was cancelled after 5 minutes, also before metrics were written, because the hybrid fallback still resolved accumulated structured constraints for the sliding-window mode. That would have made the execution comparison look better while violating the selected memory policy. Mode-specific visible constraints and a four-turn horizon case were added before the valid run; the cancelled job is excluded from memory metrics.

## Development result

On the six-turn development pilot repeated at two seeds, the constrained hybrid reached 12/12 post-guard task success with zero unsafe calls, while the raw model produced the complete required tool path on 4/12 attempts and averaged 0.778 parameter accuracy. Pure LLM task success was 2/12. The main failure was under-planning: the model often selected one plausible stage but omitted required upstream context, evidence or forecast stages. Prompt hardening improved date windows and BESS nesting but did not solve complete-path selection, so the result was frozen rather than repeatedly tuned on holdout.

## Frozen holdout result

The valid run used evaluation commit `5eb225d`, Slurm job `29928374`, 13 episodes, 24 turns, four memory modes and three model seeds. It produced 672 scored rows and 701 real model requests: 576 base samples plus 125 visible replans/retries. The job completed in 1:16:27 on one scheduled 20GB A100 MIG allocation; batch MaxRSS was 10,393,132 KiB (about 9.91 GiB).

| Path / memory | Attempts | Task success | Raw-model complete path | Raw-model parameters | Memory recall | Pass-all-3 | P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic / structured | 24 | 50.0% | n/a | n/a | 90.9% | n/a | 0.194 / 0.629 s |
| Pure LLM / structured | 72 | 38.9% | 38.9% | 91.0% | 90.9% | 33.3% | 6.59 / 22.22 s |
| Hybrid / no memory | 72 | 45.8% | 12.5% | 61.1% | 0% | 45.8% | 6.52 / 13.29 s |
| Hybrid / full history | 72 | 75.0% | 19.4% | 86.1% | 90.9% | 75.0% | 7.36 / 13.55 s |
| Hybrid / sliding window | 72 | 75.0% | 22.2% | 88.2% | 81.8% | 75.0% | 7.38 / 15.18 s |
| **Hybrid / structured state** | **72** | **79.2%** | **34.7%** | **91.4%** | **90.9%** | **79.2%** | **7.31 / 18.84 s** |

All hybrid structured attempts had valid citations, consistent settlement, successful bounded replanning, zero state contamination and zero unsafe tool/DSL calls. These safety/calculation successes do not cancel the task/path failures.

Across both model paths, the run consumed 1,749,385 prompt tokens and 114,393 completion tokens, with 125 retries and 24 rejected model calls. The external-provider cost field is AUD 0 because inference used a local research allocation; compute usage is not free and is reported as Slurm resources rather than converted to an invented AUD rate.

## What the comparison means

Structured state was the strongest tested memory policy, but not strong enough for promotion. On the dedicated four-turn horizon episode, constrained hybrid scored **12/12** with structured state, versus **9/12** for full history, **9/12** for the two-turn sliding window and **3/12** with no memory. Pure LLM on the same episode scored 6/12, 1/12, 1/12 and 0/12 respectively.

The final turn deliberately placed the date outside the two-turn sliding window while retaining BESS parameters in the recent window. Structured state kept the sourced date, region, power, energy and efficiency; sliding state dropped the old user constraint by policy. A validated forecast/tool summary could still expose the date indirectly to the model, so the aggregate `memory_recall` field is deliberately conservative: it measures sourced constraint-state retention, not every recoverable value in trusted tool summaries.

Full history did not dominate structured state. It used slightly fewer average prompt tokens in this small holdout (2,889 versus 2,982) but failed to consistently convert raw turns into complete plans. No-memory averaged 3,972 prompt tokens because its missing context caused 60 retries; fewer stored turns did not mean lower total cost.

## Failure analysis

The dominant model failure was **under-planning**, not unsafe planning. Qwen3-8B often emitted one locally plausible stage—such as only `optimize_battery_dispatch`, `forecast_price_risk` or `search_official_evidence`—instead of the full prerequisite workflow. Prompt hardening improved half-open dates and BESS nesting during development, but raw complete-path accuracy remained far below the frozen gate.

The 15 structured-hybrid failures were seed-stable and concentrated in five benchmark turns:

- one evidence-reference comparison lost the prior region when the new turn named only the second region;
- one region replacement was interpreted as a snapshot flow rather than a comparison;
- two forecast turns were scored against `forecast → evidence`, while the canonical product DAG orders evidence before forecast—an evaluation-specification mismatch retained rather than edited after seeing holdout;
- one combined “coverage and snapshot” prompt was ambiguous, and the deterministic intent layer selected coverage.

The holdout therefore found both model limitations and evaluator/runtime edge cases. Because the benchmark was frozen, none was relabelled post hoc and no corrected score is substituted for the preregistered result. A future v2 may clarify those prompts and repair comparison-state merging, but it requires a new untouched test source before any promotion claim.

Security results are narrower than a general robustness claim. Direct prompt injection produced no unsafe tool/DSL execution, and malicious evidence prose was quarantined from planner memory; this validates the architecture boundary, not universal prompt-injection resistance.

## Promotion decision

The constrained-hybrid structured path passed raw parameter accuracy (91.4%), citation correctness (100%), settlement consistency (100%), replanning (100%), sourced-state recall (90.9%), contamination (0%) and unsafe-call (0) thresholds. It failed task success (79.2% < 85%) and raw complete tool path (34.7% < 90%). It exceeded deterministic performance on the memory-required subset (72.7% versus 45.5%), but the preregistered rule requires every absolute threshold as well as the relative comparison.

The model planner is therefore not enabled in the SG release and no production or resume claim says that autonomous planning is promoted. The API/provider boundary, evaluation harness and fail-closed fallback remain available for future model comparison.

An unpromoted model path remains useful evidence: it shows exactly where an 8B tool-calling model helps, where it omits workflow stages, and why typed execution guards are product logic rather than demo scaffolding. Promotion failure does not affect the verified deterministic historical-replay service.

## Reproduce

Local contracts:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests scripts
.venv\Scripts\python -m mypy --strict src
```

Spartan progression:

```bash
sbatch --test-only scripts/slurm/llm_agent_runtime_preflight.sbatch
sbatch scripts/slurm/llm_agent_runtime_preflight.sbatch
sbatch --test-only scripts/slurm/llm_agent_gpu_pilot.sbatch
sbatch scripts/slurm/llm_agent_gpu_pilot.sbatch
sbatch --test-only scripts/slurm/llm_agent_holdout.sbatch
sbatch scripts/slurm/llm_agent_holdout.sbatch
```

The private run retains per-turn predictions and sourced traces. GitHub receives only a compact aggregate with exact hashes, runtime identity, Slurm usage and factual boundaries.

## Interview STAR

**Situation.** The existing 80-case Agent gate proved a deterministic tool DAG, but recruiters could reasonably ask whether a real LLM could choose tools, maintain constraints and recover from failures.

**Task.** Add real model planning without letting generation expand the trusted computing base or turning a historical replay system into an autonomous trader.

**Action.** I put a hash-pinned Qwen3-8B/llama.cpp runtime on an isolated Spartan A100 MIG job, restricted it to eight Pydantic tools, separated raw proposals from guarded execution, implemented sourced state memory and bounded replanning, and froze a 13-episode/24-turn holdout across four memory modes and three seeds. I also cancelled two early runs when review found greedy seed repetition and hidden long-memory leakage in the sliding baseline, preserving both failures rather than accepting invalid metrics.

**Result.** Structured hybrid improved the memory-required subset to 72.7% versus 45.5% deterministic and completed the four-turn stress case 12/12, with zero unsafe calls, 100% citation/settlement/replan checks and 169 local tests passing. However, overall task success was 79.2% and raw complete-path accuracy 34.7%, so the promotion gate failed and the deterministic DAG stayed in production. The lesson is that schema validation protects execution, but it does not make an 8B model a reliable workflow planner.

## Resume candidates — do not copy both by default

1. `为 NEM 历史决策回放 Agent 接入真实 Qwen3-8B 工具规划层：约束模型仅调用 8 个 Pydantic tools，确定性 runtime 负责时间/数据/引用/BESS 校验与 fail-closed 回退；在 24-turn、4 memory modes、3 seeds 的冻结留出集上实现 0 unsafe 调用与 100% 引用/结算/重规划校验。`
2. `构建可归因的跨轮 Summary/State Memory 与长记忆压力测试；structured hybrid 在四轮约束修改案例达 12/12、优于 full/sliding 的 9/12，但整体 57/72 且 raw tool path 25/72，未过预注册门槛，保留 deterministic DAG 为线上路径。`

Neither candidate implies public production SLA, general reasoning ability, live trading or investment return. The current resume is intentionally unchanged pending a separate tailoring decision.
