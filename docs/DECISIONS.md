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

The model-development cases and holdout live in separate files. The holdout uses dates and prompts absent from the original 60-case Decision Replay suite. Thresholds, a non-zero sampling temperature, seeds and the rule requiring hybrid to exceed the deterministic memory subset were recorded in `benchmarks/llm_agent_promotion_gate_v1.json` before the valid holdout execution.

The first holdout submission was cancelled before metrics were written when a code review found that the llama.cpp adapter still used greedy temperature zero. Repeating seed labels under greedy decoding would not test sampling stability. The cancelled job and resource usage remain in the report; only the subsequent non-zero-temperature run can make pass@1/pass-all-k claims.

## ADR-005 — Real runtime first, negative result allowed

Status: accepted.

The verified provider is a real local Qwen3-8B Q4_K_M GGUF runtime through a pinned llama.cpp server on a Spartan A100 MIG allocation. The server is loopback-only, the model is staged in job scratch, and both runtime binaries and inputs are hash checked. The OpenAI-compatible Model Studio adapter remains available when credentials exist; mock planners are used only in unit tests. The LLM path is promoted only if every preregistered threshold passes; otherwise it remains a reproducible negative experiment.

The initial Ollama route was rejected after the Iris home quota prevented its first-run state from being created. No existing SSH material or home-directory configuration was changed to force it through. A project-scoped llama.cpp build and Slurm scratch model resolved that infrastructure problem without weakening isolation.
