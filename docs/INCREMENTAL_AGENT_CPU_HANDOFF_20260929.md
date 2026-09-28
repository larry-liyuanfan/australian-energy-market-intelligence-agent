# Incremental decision Agent — CPU-ready handoff

Validation date: **2026-09-29**. Base `4333f2862ff1f06a277c621f4bc3733c2dc1272e`, exclusive branch `codex/energy-visual-goal-compiler-v2`, existing draft PR12. Resolve the delivery SHA with `git log -1 --format=%H -- docs/INCREMENTAL_AGENT_CPU_HANDOFF_20260929.md`. No main push, merge, SG deployment, release change or career-file write.

## Outcome and exact claim

The new opt-in entry connects the **existing Qwen-compatible `TurnPlanner`, sourced conversation state, registered domain tools, LangGraph and SQLite**. Its real-data CPU path now reaches BESS dispatch and historical settlement, unlike the previous market-only graph. Actual Qwen inference through this new graph is **not yet verified**: there were no new model requests, GPU jobs or paid API calls in this package.

There are two distinct capabilities, not one inflated claim:

- Normal replay is a dependency-compiled, deterministic four-tool workflow. A provider can propose these calls; exact schema/scope matches are attributed to the model, everything supplied by the system is labelled system completion. Normal-path proposals do not freely choose the workflow.
- **Empty-evidence recovery is model-controlled.** After observing an empty search, the model may choose a restricted market-only query rewrite, ask for clarification, or stop. Each choice changes the next executable action. Invalid choices explicitly fall back to the same-scope runtime retry. Counterfactual tests demonstrate this control using labelled test doubles; they do not establish real Qwen decision quality.

The old `EnergyAgent`, model runtime and market-only graph remain available. The only backward-compatible provider changes are an optional completion-token cap (unused by old defaults) and availability metadata for reported usage. Default API/deployment behaviour is unchanged.

## Dependency and attribution contract

```text
source-bound user region/date/battery constraints
        ↓ canonical domain contract ← attributed model proposal
market facts → official evidence
      └─────→ as-of forecast → BESS dispatch
market prices + fixed schedule → historical settlement
all verified outputs + citations → bounded template answer
```

Four tools are used from the existing registry of eight; no new tool schema or model SQL/DSL/optimisation expression is accepted. This is a one-region, one-complete-day **text-evidence** vertical slice, not a second general-purpose Agent. It does not migrate the earlier chart/figure router or all event-diagnosis tools into the graph.

Each cache key binds the implementation version, validated complete arguments, upstream keys and content versions. Market versions include actual rows plus numeric source records, not merely a mutable data label. Evidence versions include document content. Forecast versions include the complete selected snapshot, arrays, full model/data SHA and cutoff, not its truncated ID alone. Dispatch additionally binds the full BatterySpec, forecast dependency, objective, mode and cycling-cost sensitivity. Settlement binds dispatch/market dependencies and values; the answer binds verified outputs/citations. Settlement and answer are regenerated, not separate model tools or independently claimed cache hits.

Only the latest entry per stage is retained. Each use records cache key, dependencies, original computation turn, current user-constraint sources, action (`reuse`/`execute`), attribution and invalidation reason. Reused results are revalidated, including their stored result digest. Battery-only changes preserve the first three keys; region/date changes invalidate the corresponding upstream/downstream keys. Full result and planner-attempt records remain private in SQLite/JSON.

The existing dispatch tool internally reads the forecast snapshot and computes a perfect-foresight comparison for regret. Avoiding a forecast *tool call* does not imply removing every internal forecast read or optimisation. The four-turn pilot still reoptimises dispatch in all four turns. No reduction in MILP solves or GPU/model tokens is claimed.

The supported correction grammar is explicit paired `MW/MWh`, efficiency/RTE percent, degradation sensitivity, region and date. Unsupported single-power/capacity or SoC edits fail closed instead of silently retaining old values. Arbitrary natural-language constraint extraction has not been solved.

## Model decision point and budgets

The graph uses the existing `LlamaCppPlanner` through a numeric-loopback-only CLI, with environment proxies and redirects disabled. It sends only registered tool schemas. The new entry caps completions at 512 tokens and provider timeout at 45 seconds, with at most one initial request plus one empty-evidence decision per graph turn. Usage, rejected proposals and provider failures are retained; unavailable token usage is not recorded as measured zero. A boundary recorder preserves provider attempts even when the legacy runtime catches an exception and falls back. Flat legacy and nested graph token fields are normalized, with unknown usage and incomplete accounting reported separately.

The rewrite removes the battery-dispatch terms while retaining the exact region, day and quarter, and requests up to ten results. The validator accepts only that pre-authorised alternative. It does not permit relaxing report period, inventing a region, changing the BESS contract or returning arbitrary query code. Clarify/stop are parsed from an exact small JSON decision, never promoted to sourced facts. Retrieved text is not fed back as instructions.

The actual retriever first requests at least 100 candidates for a scoped query, then applies topic/region/quarter checks. Thus increasing `top_k` alone is not a recovery algorithm. On the real retained index, a genuinely missing **Q4 2024** report query returned zero results both before and after the rewrite. That negative observation is published. Mock “empty once, then success” tests measure control flow only; there is **no observed real recovery gain**.

Timeouts are same-argument retries, not replanning. Each tool has at most two attempts; graph recursion is bounded. Thread-based timeout remains cooperative rather than a hard kill. Provider calls interrupted outside a known approval checkpoint are not automatically replayed: the public invocation wrapper requires manual request/usage accounting, avoiding an unreported second model request.

## Persistence and correction

The CLI can pause after a verified stage, exit, then resume the same SQLite thread. `approve`, `cancel` and `correct` are distinct actions. A correction preserves already validated cache entries and starts a new sourced user turn; the interrupted turn's calls/proposals are retained in history. Another thread does not inherit those constraints.

The in-progress execution policy (mode, provider/model identity, seed and reuse policy) is checkpoint-bound; silently changing it on resume is rejected. Data/evidence/forecast versions are checked on execution and again before publishing the final answer, including after a pause at the last dispatch stage. A crash during a read-only tool can still cause recomputation; this is not distributed exactly-once execution or multi-user authentication.

A real two-process check on SA1 2025-12-15 paused after the forecast with three calls, then resumed with a change to 1 MW / 3 MWh. It completed with **one new dispatch call and three reused upstream results**. This was deterministic CPU execution, not a live Qwen demonstration.

## Four real-data pilot turns and independent comparison

Input: 576 official AEMO records, SA1/VIC1 on **one** day (2025-12-15), existing as-of forecast snapshots and official text records. Four author-written cases, not four independent dates or held-out customer tasks:

1. SA1: 1 MW / 2 MWh initial replay.
2. Same region/date: change to 1 MW / 3 MWh.
3. Same date/battery: change region to VIC1.
4. Same region/date/size: change efficiency to 80%.

| Path | Actual tool calls by turn | What differs |
| --- | --- | --- |
| Existing deterministic planner/completion | 6, 6, 5, 5 | Includes event/diagnosis work outside the new minimal slice |
| New graph, complete rerun | 4, 4, 4, 4 | Like-for-like four-stage no-cache baseline |
| New graph, incremental | **4, 1, 4, 1** | Six upstream tool calls avoided across battery-only changes |
| Real Qwen full-rerun/incremental | **Not run** | Waiting for resource release; no mock results substituted |

All twelve path/turn observations passed independent power, SoC, energy-balance and cashflow checks, and produced equal planned/actual settlement values across the three paths. The evaluation imports the earlier standalone `audit_schedule`, not production `verify_result` or the MILP solver as its oracle. Source ID/URL/hash/excerpt location is separately checked against mounted sources; that is not human semantic entailment or a causal explanation of daily prices.

The [public pilot](../artifacts/public/incremental_cpu_pilot_20260929.json) retains each measured wall time and result. These are single ordered samples including graph checkpointing, full-input fingerprints and independent validation, excluding common initial input loading and Python startup. They are **not P50/P95 or a statistically controlled speedup**. In this pilot the incremental graph was still slower than the legacy deterministic path. The like-for-like benefit is fewer tool calls, not established end-to-end or model-quality superiority.

`versions()` deliberately recomputes full in-memory content hashes on execution, favouring conservative correctness in this bounded prototype. That can dominate at a million rows; only 576 rows were measured. Moving to immutable artifact-level digests with explicit reload/version handles is a remaining engineering task, not a claimed scalability achievement. Do not run this implementation over the full annual store and assume the small-pilot latency holds.

All money remains a historical operating proxy. `planned_margin_aud`/`realized_margin_aud` in the tool are **net of an assumed variable cycling-cost sensitivity**, not gross revenue. CAPEX, fixed O&M, network fees, FCAS and actual investment return remain excluded. Later official reports are retrospective context, not as-of forecasting inputs.

## Frozen vNext and what remains unmeasured

[Scenario v2](../benchmarks/incremental_vnext_holdout_v2_20260929.jsonl) and [protocol](../benchmarks/incremental_vnext_protocol_20260929.json) freeze **24 episodes / 48 turns** before model consumption. The retained [v1-to-v2 amendment](../benchmarks/incremental_vnext_contract_revision_20260929.json) replaces ambiguous second-turn prose with explicit values/status labels before any new model output; v1 remains available. November 18/19 scenarios are disjoint from this package's January synthetic development and December real pilot. Labels are author-generated contracts over the same AEMO source family, **not independent user annotations or independent-source validation**.

Groups cover battery/RTE/cost changes, region/date edits, versions, checkpoint correction, thread isolation, empty evidence, timeout, malicious evidence/unsafe suggestions, and numerical/citation conflicts. The first twelve normal episodes and twelve controlled fault/lifecycle episodes must be reported separately. Required metrics separate initial model proposals, accepted fields, recovery choices, graph completion, final contract correctness and independently checked financial/citation outputs. All attempts, unknown usage, tokens, wall times and per-seed results must remain visible. Cache savings belong to the runtime, not the model.

The real-model holdout is **not consumed** and has no model pass rate. The full runner now traverses all 24 episodes, 48 turns and three paths using labelled synthetic mechanics inputs. November official inputs are separately ready: 2,304 rows over four regions/two days, plus eight seasonal-conformal snapshots, each built using 8,640 earlier intervals. Snapshots were generated retrospectively with an as-of training cutoff; their construction date is not claimed to precede the historical decision.

CPU contract traversal is protocol QA, not a model comparison. The graph paths each matched 48/48 authored status/state contracts, versus 43/48 legacy. One legacy mismatch is a status-taxonomy difference for isolated context, not a five-case demonstrated quality lift. The other four injected cases expose accepted malicious snippets, wrong cashflow, wrong signal and wrong citation hash in the legacy path. These are controlled diagnostics, not natural market error rates.

`task_contract_success` and `intervention_score` are separate. Tool faults require an observed injection plus the expected stage/reason; versions and lifecycle changes have their own reachability records. Generic `failed` cannot verify a target fault. Eleven of twelve second-turn interventions are exercised/verified for each graph; the prompt-injection model case is not exercised without a successful real provider response. Legacy does not implement the checkpoint lifecycle; its strict isolation diagnostic additionally requires no context-free tool calls. See the compact CPU contract artifact for exact denominators. These numbers cannot be called LLM recovery rates.

No new end-to-end model task score, token saving, pass@1/pass^k, stability interval or production SLA exists yet. The earlier frozen Qwen negative result is unchanged.

### Required real-model recovery report — not run

The eventual authorized model handoff must separate **normal integration/overhead**
from **recovery decisions**. Matching the canonical four-step plan only establishes
that proposals can be accepted and executed under the fixed runtime contract; it
does not demonstrate an improved decision. Likewise, the frozen `empty_evidence`
intervention always returns empty results. A successful abstention or failure
contract on that case cannot establish evidence-recovery gain. Do not change the
fault or add tasks to manufacture a positive result.

For each recovery opportunity, report this ledger from retained attempts and tool
outputs, with one row per path/episode/turn/seed and additional rows for retries:

| Required field | Reporting rule |
| --- | --- |
| Opportunity and provenance | Episode, turn, seed, path, source/data versions; distinguish deliberately persistent-empty injection from a naturally missing report. |
| Model observation and choice | Record successful response, provider failure or invalid response separately; then parsed rewrite / clarify / stop / invalid choice. A failed request followed by fallback is not an observed model decision. |
| Actual next action and owner | Show the executed tool name and validated query/arguments, or no further tool call for clarification/stop; distinguish accepted model choice from deterministic fallback and same-argument retry. |
| New evidence | Record evidence counts before/after, newly returned source IDs and source validation outcome. For clarify/stop, mark retrieval not attempted, not successful recovery. For a rewrite with zero new sources, report zero evidence gain. New IDs alone do not establish answer usefulness or semantic support. |
| Calls and attempts | Show all provider requests, successful responses, tool attempts, retries and fallbacks, including failed/rejected attempts. Separate initial planning from recovery; also report whole-turn totals. |
| Tokens, time and cost | Show prompt/completion tokens for every measured request, unknown-usage counts, model latency and whole-turn wall time. Report recovery-only wall time only if separately recorded; otherwise mark unavailable, not an invented sum of component times. Label partial totals as lower bounds; report infrastructure price as unavailable if no price basis exists. |
| Outcome and comparator | Keep authored task-status match, exercised target intervention, independent citation/settlement checks and evidence gained as separate fields. Compare the observed deterministic next action and its costs on the same case; never infer a counterfactual outcome from an unexecuted choice. |

Aggregate rewrite/clarify/stop/invalid/provider-failure counts by path and seed.
State both denominators: all recovery opportunities and successful model responses.
Do not drop requests that failed before parsing or combine normal turns with the
recovery subset to inflate success. Report paired measured costs where available;
runtime cache reuse and fewer domain-tool calls remain runtime effects, not model
reasoning improvements. Normal turns should be titled **integration and overhead**,
not model benefit.

The report must allow conclusions such as **valid recovery choice, no new evidence**,
**safe runtime fallback after model failure**, or **no benefit over deterministic
execution at greater cost**. A model-benefit claim requires additional evidence
already measured within the authorized protocol, not merely a passing contract.
All real-model rows and aggregates remain **not run** in this CPU handoff. This
reporting clarification changes no frozen labels, code, model, holdout or CPU result.

Evidence: [CPU contract audit](../artifacts/public/incremental_contract_cpu_20260929.json),
[run/installation manifest](../artifacts/public/incremental_cpu_manifest_20260929.json) and
[Linux wheel hashes](../requirements/agent-graph-linux-py311-wheels.json). The manifest records
the three CPU jobs including the failed post-check, exact private-input/model hashes and
declared wheel licenses. Runtime archives and raw attempt journals remain private.

The first delivery CI (`36446118448`) caught a Linux `/proc/PID/fd` enumeration
race: a descriptor closed before `readlink`, rejecting even the test's own
listener. The follow-up reads each link once and tolerates only a vanished
descriptor; permission failures, absent live listener ownership and wrong model
identity still reject. A portable regression and the actual Linux ownership
test cover that distinction. Inspect the latest delivery SHA's CI rather than
claiming the initial run passed.

## Reproduce and demonstrate

Use the already pinned optional graph environment from the previous handoff locally. For Spartan, 41 hash-pinned Linux CPython 3.11 wheels (13,679,421 bytes) were prepared and installed in node-local scratch, with successful LangGraph/SQLite/SciPy/Pydantic imports. No shared environment was changed. Private input hashes are in the new run manifest. `run_incremental_replay.py --help` documents the CLI; `evaluate_incremental_pilot.py --help` documents the three-path CPU comparison.

```powershell
$env:PYTHONPATH = 'src'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$py = 'artifacts/private/langgraph-env/Scripts/python.exe'
# $inputArgs contains the three authorised private paths and their SHA-256 values.
# --data / --data-sha256: the official compact subset
# --evidence / --evidence-sha256: official text JSONL
# --snapshots / --snapshots-sha256: existing forecast JSONL
& $py scripts/run_incremental_replay.py @inputArgs --checkpoint artifacts/private/example.sqlite --thread analyst-A --question 'Replay SA1 2025-12-15 BESS 1MW/2MWh with official evidence.' --pause-after forecast_price_risk --output artifacts/private/paused.json
& $py scripts/run_incremental_replay.py @inputArgs --checkpoint artifacts/private/example.sqlite --thread analyst-A --resume correct --correction 'Use BESS 1MW/3MWh instead.' --output artifacts/private/resumed.json
& $py scripts/evaluate_incremental_pilot.py @inputArgs --private-run artifacts/private/fresh-pilot --output artifacts/private/fresh-pilot.json
```

For a subsequently authorised real-model pilot, use the same CLI with `--provider llama_cpp --provider-url http://127.0.0.1:11629/v1` on both pause and resume. Omission of the provider selects deterministic mode; a resumed in-progress hybrid run cannot silently switch modes. Never call a provider against a paid/public endpoint. Existing response paths are not overwritten. Full tool outputs, SQLite and source materials remain private.

The code uses actual `StateGraph` nodes/conditional edges/checkpointing, following official [graph](https://docs.langchain.com/oss/python/langgraph/graph-api), [interrupt](https://docs.langchain.com/oss/python/langgraph/interrupts) and [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) contracts. It does not imply a separate LangChain application or MCP server.

The complete runner has three explicit modes:

```bash
# Synthetic mechanics only: no model request, no official-data score.
python scripts/run_incremental_vnext.py --contract-check \
  --scenarios benchmarks/incremental_vnext_holdout_v2_20260929.jsonl \
  --scenario-sha256 9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f \
  --output artifacts/private/new-contract-check
# For a released live pilot, add the six private input path/hash flags:
python scripts/run_incremental_vnext.py --development --model --seeds 17 \
  --provider-url http://127.0.0.1:11629/v1 --provider-model "$MODEL_ID" \
  "${INPUT_FLAGS[@]}" --output "$PRIVATE_NEW_RUN"
# Full execution additionally requires --consume-frozen --release-id,
# the frozen file/hash above and --seeds 17 29 43 (no --development).
```

## Installed/prepared tools and remaining resource release

At initial read-only inspection, project space was 37% used but only 627/500,000 inodes were free. A later authorized preparation check found 1,242 free. CPU job **31484156 completed 0:0 in 78 seconds**, with peak RSS 141,236 KiB: isolated dependencies, November inputs and the pinned model download were verified. The post-job snapshot had 1,233 free inodes and 291 GiB free; these are shared-project observations, not exclusive attribution. No account registration or private provider key was needed. No Trip/Climate/FLARE job or directory was modified.

CPU job **31484355 completed 0:0 in 80 seconds**, peak RSS 1,723,208 KiB, traversing the synthetic 24-episode contract. After accounting/attribution fixes, job **31484808** completed the revised 144-row traversal and archived every result with no runner exceptions, then **failed 127:0** at an added CUDA-linked `llama-server --help` check on a CPU node (`libcuda.so.1` unavailable). Its overall Slurm result is not green. The unnecessary CPU/GPU-driver check was removed; server CLI/alias and model identity are mandatory in the later authorized GPU preflight. No driver was installed on the shared cluster and no GPU job was submitted to work around this expected CPU limitation.

Code and wheels use one archive per revision; dependencies, model working copy and SQLite files are unpacked in Slurm scratch. Results use one compressed private archive per job. The fixed project directory is `energy-agent/incremental-agent-20260929`; November and run archives are under `energy-artifacts/incremental-agent-20260929`. The preparation script now writes directly to that artifact directory, matching the full-run input contract. The first produced November archive was moved there without changing its verified hash.

Proposed release sequence, subject to the coordinator confirming Trip's GPU terminal state and healthy storage:

1. CPU prerequisites are ready: exact private input/code hashes, unique complete five-minute November windows, 30-day as-of history and dependency imports. Check fresh space/inode counts before GPU release; do not delete other projects to obtain inodes.
2. One live integration pilot: `punim2936`, `gpu-a100-mig`, `gpu:1g.20gb:1`, 6 CPUs, 16 GiB RAM, 30-minute limit, 10 GB node-local scratch. Estimate **0.1–0.3 MIG-hours**, not a reservation or observed cost. Run `sbatch --test-only` only after resource release, then submit one job. No stacked competition jobs.
3. Only after successful model/graph/source/accounting checks, size the full run from measured pilot throughput. Provisional 24×2×3 paths×3 seeds is 432 initial turns; potential recovery increases requests. Rough **0.8–1.5 MIG-hours** is an estimate based on the previous 3,732-second / 549-request job, not measured vNext throughput. Full run remains unapproved/unsubmitted.

Model: pinned `Qwen/Qwen3-8B-GGUF` revision `1d54a16a18cba0d8fbad4a16db801decc729e099`, file `Qwen3-8B-Q4_K_M.gguf`, SHA `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, **5,027,783,488 bytes**, is now actually cached and fully SHA-verified in the Energy project directory. Only this file was downloaded, not another model. Existing Energy-only runtime `energy-agent/planner-remediation-20260927/env/llama-runtime/llama-server` exists; its successful model load remains to be verified on the authorized GPU node. Reuse llama.cpp commit `7798007a29a90e3053e799394da48cf53a2f8e0f`.

The delivered Slurm files are `incremental_prepare.sbatch`, `incremental_contracts.sbatch` and `incremental_model.sbatch`. The GPU file requires a coordinator-issued `ENERGY_GPU_RELEASE_ID` and exact bundle SHA, does not submit itself, and refuses full execution without a reviewed pilot receipt binding the same source bundle. Run `sbatch --test-only` before each authorized submission. Start only one pilot; derive full walltime/memory from it. The server startup refuses occupied ports, verifies the launched PID owns the loopback listening socket and checks a unique model alias via `/v1/models`; the client uses the same alias. Existing listeners are never terminated.

Keep fixed Energy project/artifact roots and node-local model/dependency scratch. The proposed output allowance of <100 MiB/<30 persistent files per pilot excludes the explicitly authorized 5.03 GB model cache and is a budget, not measured pilot use. Package validation dates follow the task's Sydney clock; remote Slurm scheduling timestamps differed, so job IDs, elapsed times and recorded raw logs are preserved without silently normalizing that discrepancy.

## Personal contribution / interview STAR

**S/T:** 历史回放已经有真实 Qwen 规划实验和独立 LangGraph 演示，但二者尚未贯通；用户仅改电池参数时还会重复执行全部查询。我负责把模型建议、来源状态、领域计算和持久化接到一个可审计入口，并验证增量更新的价值。

**A:** 我建立按参数、数据与模型内容版本绑定的依赖键，保留模型提出、系统补齐和实际执行三层记录；实现跨进程恢复与纠错，以及功率、SoC、预测信号和结算校验。发现模型仅作旁路提议仍过浅后，增加空证据时由模型选择受限改写、澄清或终止的决策点，并用反事实测试验证下一动作确实改变。非法建议显式回退，超时重试不包装成重规划。

**R:** 四个真实官方数据 CPU 回合的结算与原确定性路径一致；同构图的工具调用从每轮4次变为4/1/4/1。但新增校验与持久化使整体仍慢于原路径，真实缺报告查询也未因改写恢复。我保留这些负结果，只把已验证的依赖重算与恢复写为成果。已补齐 Linux 环境、November 真数据和144行合成合同遍历；真实 Qwen 在新图中的决策质量、48轮真实模型评测及规模性能仍待资源释放和验证。

This STAR is a project handoff, not an instruction to update the current resume. The defensible personal contribution is the integration/decision/cache/verification design and its measured CPU behaviour, not authoring AEMO data, LangGraph, Qwen or claiming a new model-quality improvement.

Review-only resume candidate (not applied): 将来源约束、LangGraph 持久化与 BESS 历史结算贯通为可审计增量回放；在4个真实数据开发回合中保持独立经济校验一致，将同构全量执行的4/4/4/4次工具调用降为4/1/4/1，并明确保留未提升延迟与检索未恢复的负结果。
