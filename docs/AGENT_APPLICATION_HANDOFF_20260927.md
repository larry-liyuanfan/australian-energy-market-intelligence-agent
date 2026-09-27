# Energy Agent application handoff — 2026-09-27 package

## Delivered outcome and ownership

This package makes the analyst workflow visible and adds **actual, opt-in LangGraph execution**. It does not promote the experimental model planner, rerun the frozen model benchmark, change the default API, deploy to SG, or edit career materials. It builds on `d204c8ea823e81c2c507f03e1110618547c70798`; the publication commit is the commit containing this file (resolve with `git log -1 --format=%H -- docs/AGENT_APPLICATION_HANDOFF_20260927.md`).

The [self-contained application demo](demos/agent-application-20260927/index.html) presents two deliberately separate tracks:

1. **Recorded Qwen/BESS application:** SA1 on 2025-12-15, sourced user constraints, actual model proposals, runtime corrections, official text/workbook previews, as-of forecast, fixed battery schedule and historical settlement. The original P1 recording is unchanged. The new explanatory answer is a deterministic rendering of checked archived outputs, **not a retained or newly generated LLM answer**.
2. **New live local graph execution:** 576 official AEMO rows, SA1/VIC1 on that day; three completed business turns, six separate CLI process invocations, two cross-process interrupt/resume cycles and 13/13 bounded checks. No model requests, tokens or GPU jobs. The HTML displays the resulting records, not a live backend.

My added application code owns the graph adapter, provenance-preserving state restoration, independent result verification, archival business checker, CLI harness and presentation. It reuses the project's existing eight-tool registry, deterministic planner, source data and conversation-memory implementation. It does not claim authorship of LangGraph, AEMO material or the previously generated model outputs.

## Framework usage register

| Component | Actual use in this package | Boundary |
| --- | --- | --- |
| LangGraph 1.2.12 | `StateGraph(GraphState)`, conditional edges, dynamic `interrupt`, `Command(resume=...)` | Real execution, not an import-only integration |
| SQLite checkpointer 3.1.1 | `SqliteSaver`, private persistent state keyed by `thread_id` | Local restart/resume; not distributed durability or user authentication |
| Existing typed runtime | Pydantic inputs, registered tools, dataset/source/result checks | Model never writes SQL, ES DSL or optimisation expressions |
| Existing memory | User-sourced constraints and verified tool summaries participate in the next turn | Redis tracing alone is not called memory |
| `langchain-core` 1.6.5 | `RunnableConfig` and LangGraph dependency | Not a separate LangChain chain/application |
| MCP | No Energy MCP server/client added | No MCP implementation claim |
| Qwen3-8B | Earlier recorded P1 and frozen v3 experiment only | Not called by the new graph |

Installed into an ignored, project-local Python 3.13 environment, not the default environment or SG. The `graph` extra pins the two direct graph packages; [exact environment constraints](../requirements/agent-graph-py313.lock) record the resolved versions. This is not a hash-enforced universal lock. External LangSmith tracing is disabled. An audit found vulnerabilities in the environment's bootstrap pip 26.0.1; only this isolated environment was upgraded to pip 26.2.1 and its audit then reported no known vulnerabilities (editable project excluded).

Design follows official [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [interrupt/resume](https://docs.langchain.com/oss/python/langgraph/interrupts) and [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) documentation. In particular, the approval node has no tool side effects before `interrupt`, because that node restarts when resumed.

## Executable graph, not a renamed old loop

```text
normalise sourced state → deterministic plan → approval / interrupt
                                                 ↓ approve
                  bounded retry ← verify ← execute registered tool
                                      ↓ valid / more tools
                                   advance ─────────┘
                                      ↓ last valid result
                           deterministic cited answer → checkpoint
```

LangGraph owns node transitions, conditional retry and checkpoint boundaries. Existing components own domain planning and tool calculations. The adapter validates market snapshot/comparison/coverage workflows; the real-data demonstration exercises snapshot and comparison, **not coverage or a LangGraph BESS workflow**. Full battery replay remains in the original runtime.

The initial SA1 snapshot was paused before any tools and resumed in another process. A subsequent “Correction: use VIC1; keep the date” retained the date with `source_turn=1`, changed the region with `source_turn=2`, and did not duplicate a user turn on resume. An independent thread compared the two regions; a new thread with no date did not inherit another thread's date. All three completed results matched the original deterministic runtime on identical inputs.

Verification checks result/tool identity, official citation ID/URL/hash, exact snapshot source values and timestamp, and independently recomputed comparison counts/means/maxima. Dataset-version mismatch fails both on a new turn and after a restart. Synthetic tests cover timeout/empty-result budgets, cancellation, wrong thread, forged tool identity and forged market values. At most two attempts per tool and a graph recursion limit of 40 are allowed. A slow thread-based tool may still delay shutdown: **the timeout is cooperative, not a hard wall-clock/cancellation guarantee**. Only bounded local read-only tools are supported. This is not exactly-once trading execution.

## What was independently checked

The CPU-only archival checker does not import the original benchmark scorer, model or battery optimiser. Successful Spartan job **31372698** used one CPU, requested 1 GiB / three minutes, completed in three seconds, and reported MaxRSS 35,696 KiB. No computation ran on the login node.

| Evidence | Checked denominator | Result and limit |
| --- | --- | --- |
| Frozen v3 retained records | 504 sampled rows / 18 authored turns | Structured request-scope audit only; original scores are unchanged |
| Final answer semantics for those rows | **0 retained answers** | Cannot retrospectively verify 18 complete answers, citations or cashflows |
| P1 business replay | 1 case / 288 intervals / 21 checks | 21/21: source/signal/schedule identity, constraints, market facts and cashflows |
| P1 citation location | 5 source records | 1 exact visible text excerpt, 3 figure previews, 1 link-only record; not 5 excerpt checks |
| New graph execution | 3 completed turns / 13 checks | 13/13 functional checks; not a task-quality benchmark or SLA |

The P1 constrained 1 MW / 2 MWh schedule has 90% round-trip efficiency, 10–90% SoC and 50% initial/terminal SoC. Independently recomputing all 288 intervals gives planned gross margin **AUD 253.381832**, and settlement of that same schedule at actual historical prices **AUD 477.794477**. Maximum energy-balance residual is 2.22e-16 MWh. A separately assumed AUD 50/MWh discharged sensitivity subtracts AUD 75.894664; it is not an observed degradation cost. These are historical operating proxies, excluding CAPEX, fixed O&M, network charges, FCAS and investment returns. The checker does not re-solve optimality or oracle regret.

Q4 evidence was published after the market day and is retrospective quarterly context, not input to the as-of forecast or proof of daily causality. The five-word text excerpt cannot establish a day-specific explanation. Figure previews locate in the retained workbook records; original PDF page numbers and spreadsheet coordinates are absent and are not invented. Source location does not prove semantic entailment.

Two small audit attempts are retained rather than hidden: 31372559 failed on the wrong figure-manifest hash; the authoritative P1 input was located and used. Job 31372651 then exposed omitted optional battery fields in archived calls; the independent checker resolved and tested existing `BatterySpec` defaults. Job 31372698 passed after both corrections. Old dependencies were already terminal; rejected `afterok` submissions did not execute. Original model jobs and their outputs were not modified.

## Quality, latency and resource trade-offs

Keep the frozen v3 result: structured hybrid completed 54/54 sampled turns but the model's initial complete tool path was correct on only 18/54; the deterministic structured-memory baseline already completed 18/18. The separate archival request-scope audit uses a different, narrower definition and is **not** a model-quality improvement. No model promotion is justified.

Existing structured hybrid P50/P95 was 5.820/22.567 seconds versus deterministic 0.235/0.787 seconds. The complete frozen experiment issued 549 model requests, including 117 replans, with 1,394,985 prompt and 95,306 completion tokens. It reported 139 tool retries, including 22 deterministic retries. The original job consumed 3,732 seconds on one 20-GB A100 MIG allocation with six CPUs (1.0367 MIG-hours / 6.22 allocated CPU-hours); zero external provider billing does not mean free compute.

The separate P1 sample took 18.384 seconds and 2,423 prompt / 510 completion tokens, with two model-accepted stages and four runtime-owned stages. Its forecast was an offline `seasonal_split_conformal` snapshot, not online LightGBM training. The new graph's six CLI invocations each took approximately 1.23–1.28 seconds **including Python startup**; these few deterministic runs cannot be compared as a model latency improvement, P95 estimate or production SLA.

Local verification: 23 targeted tests passed, Ruff passed, strict mypy passed for the five added Python source files; the secret scan completed. The existing CI suite remains, with a separate optional-graph smoke job. No local full-suite rerun was represented as completed. The publication CI run and exact commit are recorded in the delivery/PR; later results must not silently amend frozen reports.

## Reproduce within the evidence boundary

Open `docs/demos/agent-application-20260927/index.html` directly for the dependency-free archived walkthrough. For actual graph execution, obtain the authorised private subset; no model or GPU is needed. Public fixtures support contracts, **not real-market claims**. The private subset SHA is `4539f94624e603fc5d44572966c0ee5c24d976417e786a4e387d0711cabfa94f`, derived from archive SHA `9025d32d1d949dfc1e23329384428241a37e00131d306d3fe6478a898b209567`. It is not committed to GitHub.

```powershell
python -m venv artifacts/private/graph-repro-env
$py = 'artifacts/private/graph-repro-env/Scripts/python.exe'
& $py -m pip install pip==26.2.1
& $py -m pip install -c requirements/agent-graph-py313.lock -e '.[graph,test]'
$env:PYTHONPATH = 'src'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
& $py -m pytest tests/test_graph_runtime.py tests/test_application_audit.py tests/test_application_demo.py
$data = 'artifacts/private/graph_input_private.json' # authorised input, not a bundled fixture
$sha = '4539f94624e603fc5d44572966c0ee5c24d976417e786a4e387d0711cabfa94f'
& $py scripts/run_langgraph_market.py --data $data --data-sha256 $sha --checkpoint artifacts/private/repro.sqlite --thread analyst-A --question 'Show the SA1 market snapshot for 2025-12-15 with official evidence.' --pause --output artifacts/private/repro-paused.json
& $py scripts/run_langgraph_market.py --data $data --data-sha256 $sha --checkpoint artifacts/private/repro.sqlite --thread analyst-A --resume approve --output artifacts/private/repro-resumed.json
& $py scripts/run_langgraph_market.py --data $data --data-sha256 $sha --checkpoint artifacts/private/repro.sqlite --thread analyst-A --question 'Correction: use VIC1; keep the date.' --output artifacts/private/repro-corrected.json
```

Use new output/checkpoint paths for a fresh run; the CLI refuses to overwrite response files. `scripts/evaluate_langgraph_demo.py --help` documents the six-process harness. `scripts/audit_agent_application.py --help` documents archival inputs; the companion Slurm script fixes the authorised Energy-only locations. The public manifest binds code and output hashes. Large sources, SQLite checkpoints, full responses and environments remain private. This is reproducible code with input prerequisites, not a zero-setup public real-data service.

## Unknowns and next decisions

There is no measured user adoption, analyst feedback or public production SLA. New graph state is not multi-user authenticated or distributed; there is no hard-kill timeout and no complete LangGraph migration of forecast/BESS. New graph answers are deterministic templates. The frozen model experiment does not measure general reasoning ability; retained rows cannot support a retrospective semantic-answer score. These gaps are explicit, not reasons to create more benchmark samples in this bounded package. SG and the original API were not redeployed or newly health-certified.

## Interview story (about 90 seconds, Chinese)

我把项目定位为能源分析师的历史决策回放：问某区域某天发生了什么，并区分当时能够制定的电池计划和事后实际价格结算。难点不是让模型调用更多工具，而是不能把模型建议、运行时纠错和真实收益混为一谈。我保留八个强类型工具和确定性执行层，给用户约束记录来源轮次，官方文本与图表记录来源和 hash，预测快照带训练截止时间，再校验调度约束和结算口径。真实 Qwen 实验里，带结构化记忆的混合路径完成了 54 次采样，但模型最初完整路径只有 18 次正确，确定性基线也能完成，所以没有强行上线模型。我随后把市场查询接入真正的 LangGraph，用 SQLite 实现跨进程暂停恢复、用户纠错及会话隔离，并在真实 AEMO 数据上验证。对一个完整 BESS 案例，我独立重算 288 个时段，计划毛收益约 253 澳元、历史结算约 478 澳元；这只是排除资本和网络等成本的历史运营代理。我的取舍是把可解释的业务闭环做实，而不是用框架名或成功率掩盖运行时兜底。

## Resume candidates — review only, do not update the current resume

- 构建面向能源分析师的历史决策回放 Agent，贯通真实 NEM 市场数据、官方文本/图表证据、无泄漏预测快照与受约束 BESS 历史结算；将市场查询接入 LangGraph，以持久化 checkpoint、带来源的跨轮约束及结果校验支持暂停恢复、用户纠错和会话隔离。
- 对真实 Qwen 工具规划与四种记忆策略开展确定性、纯模型、受约束混合路径对照，区分模型初始路径正确性与运行时补救；混合结构化记忆完成 54/54 次采样但未优于确定性基线，保留负实验结论，并独立复核单案例 288 时段的电池约束及计划/历史结算口径。

Numbers above are bounded by the linked archived evidence; neither sentence claims autonomous trading, investment returns or online deployment of the new framework.
