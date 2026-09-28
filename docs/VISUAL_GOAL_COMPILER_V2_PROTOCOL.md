# Visual retrieval and GoalSpec compiler v2 protocol

## Decision boundary

This work is an offline evaluation of two candidate capabilities for the historical Australian Energy Market Decision
Replay Agent. It does not enable a public or Singapore model planner, live forecasting, automatic bidding, trading, or
investment advice. BESS settlement remains a backward historical operating proxy under the existing exclusions.

The frozen input contract is `benchmarks/v2_freeze_manifest.json`. A result is ineligible if its benchmark, gate,
dataset, model, configuration, code, or output hashes do not match the run manifest.

## Workflow A — public visual-document retrieval

`ViDoReCorpus` adapts three pinned public retrieval tasks without changing their query-to-page relevance relation:

- DocVQA (`vidore/docvqa_test_subsampled` plus its pinned Tesseract OCR companion);
- InfoVQA (`vidore/infovqa_test_subsampled` plus its pinned Tesseract OCR companion);
- TAT-DQA (`vidore/tatdqa_test` plus its pinned Tesseract OCR companion).

The frozen comparison contains six observable runs:

1. OCR BM25;
2. Qwen3-Embedding-0.6B over OCR text;
3. equal-weight OCR BM25/Qwen3-text RRF, which is the text baseline;
4. Qwen3-VL-Embedding-2B single-vector retrieval;
5. `vidore/colqwen2-v1.0` multi-vector MaxSim late interaction;
6. equal-weight text/Qwen3-VL/ColQwen RRF.

Fusion weights come only from the existing AEMO Q4 2024 development route. No ViDoRe query, label, score, or slice is
used to select a weight. Each task reports nDCG@5, Recall@1/5, MRR, indexing throughput, query P50/P95, serialized index
bytes, peak CUDA memory, and available table/chart/infographic/text-heavy slices. A visual or fused path is positive
only when its paired-bootstrap nDCG@5 difference over the frozen OCR text baseline has a lower 95% bound above zero on
at least two of the three tasks. Otherwise it remains an offline negative gate.

## Workflow B — model GoalSpec, deterministic compiler

`GoalSpec` is a strict Pydantic contract. It contains only intent and constraints: regions, comparison mode, a
timezone-aware half-open time range, requested outputs, evidence modality, optional BESS power/energy/round-trip
efficiency, current source turn, per-field source turns, and explicit correction records. Tool names, SQL,
Elasticsearch DSL, shell/code/file fields, and optimiser expressions are outside the schema and fail closed.

`GoalSpecCompiler` is deterministic. It expands a validated goal into the existing `DecisionCase` and registered-tool
DAG. The model never chooses a tool in this path. The evaluation stores three separate objects for every attempt:

- raw model GoalSpec JSON;
- validated GoalSpec and compiled DAG;
- executed calls, results, citations, settlement checks, retries, and trace identity.

The direct-tool Qwen3-8B path remains as an unmodified control. The frozen v2 set has 18 episodes and 36 turns whose
dates and wording do not occur in either v1 file. It covers cross-turn correction, two-region comparison, visual
evidence, snapshot, BESS constraints, timeout, empty retrieval, direct injection, malicious evidence, tool-result
contamination, forecast and coverage. Known v1 region merge, replacement/comparison, DAG-order and compound-intent
failures appear only in `goal_spec_development_v2.jsonl`; frozen v2 labels are not edited after execution.

The four systems are the existing deterministic DAG, Qwen3-8B direct-tool planning, Qwen3-8B GoalSpec and Qwen3-14B
GoalSpec. Every model system uses the three frozen non-zero-temperature seeds. The candidate must meet every threshold:
required-field F1 at least 90%, compiled task success at least 85%, citation/settlement/replanning at least 95%, memory
recall at least 90%, and exactly zero unsafe fields/calls and state contamination. Passing permits only a subsequent SG
loopback gate; failing retains the deterministic SG planner.

## Spartan execution and publication

Every job performs an exact-commit detached checkout under job scratch. Runtime environments, Hugging Face caches,
models and rendered public benchmark pages remain in job scratch. Per-query prompts, rankings, embeddings, GoalSpecs,
predictions and traces are written only below the task-specific Spartan `private/` output tree. GitHub receives compact
aggregates, hashes, resource records, factual boundaries and negative gates.

The required progression is `sbatch --test-only` → clean-room CPU preflight → small development/public-data pilot →
one frozen full run → aggregate/schema validation. Infrastructure failures may be corrected without inspecting test
quality; quality failures do not reopen the frozen benchmark or gate.
