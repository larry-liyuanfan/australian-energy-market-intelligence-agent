# Engineering decisions

## ADR-001 — Keep the deterministic DecisionCase DAG as the execution authority

Status: accepted.

The model may select only the eight registered typed tools and propose JSON arguments. Pydantic validation, temporal boundaries, dataset versions, citation checks, tool execution, BESS optimisation and historical settlement remain deterministic. Unknown tools and SQL, Elasticsearch DSL, shell commands, code, or optimisation expressions are rejected before execution.

Rationale: this isolates the capability being tested—planning—without widening the trusted computing base or changing the historical-replay product boundary.

## ADR-002 — Compare three paths without hiding fallback

Status: accepted.

- `deterministic`: fixed DecisionCase DAG.
- `pure_llm`: validated model calls are executed, but missing calls are not silently added.
- `constrained_hybrid`: model calls are filtered to the required DAG; missing or unsafe calls fail closed to deterministic calls.

Every result reports rejected calls, fallbacks, retries, steps, tokens and latency. Model-proposed parameter accuracy is scored separately from post-guard execution accuracy.

## ADR-003 — Memory is sourced decision state, not a cache label

Status: accepted.

The four evaluated modes are no memory, complete history, sliding window and structured state. Structured memory contains only constraints deterministically extracted from attributed user turns and compact summaries of validated tool outputs. Retrieved prose, model prose and error text never become facts. State is conversation-scoped, bounded and used in the next planning decision.

## ADR-004 — Holdout and promotion are preregistered

Status: accepted.

The model-development cases and holdout live in separate files. The holdout uses dates and prompts absent from the original 60-case Decision Replay suite. Thresholds, seeds and the rule requiring hybrid to exceed the deterministic memory subset were recorded in `benchmarks/llm_agent_promotion_gate_v1.json` before holdout execution.

## ADR-005 — Real runtime first, negative result allowed

Status: accepted.

The first provider is a real local `qwen3:4b-instruct` runtime through Ollama, with an OpenAI-compatible Model Studio adapter retained when credentials exist. Mock planners are used only in unit tests. The LLM path is promoted only if every preregistered threshold passes; otherwise it remains a reproducible negative experiment.
