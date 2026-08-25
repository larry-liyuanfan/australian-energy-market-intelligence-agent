# Historical Decision-Replay Agent system card

## Intended use

- Retrospective NEM price-event investigation.
- Official evidence and figure discovery.
- Historical, as-of BESS schedule replay and operating-proxy analysis.
- Demonstration of typed Agent orchestration, provenance and release gates.

## Out of scope

- Live market operations, automatic bidding or dispatch control.
- Investment advice, ROI, asset valuation or bankable revenue forecasts.
- Causal attribution from market associations.
- Unrestricted SQL, Elasticsearch DSL, filesystem access or optimisation expressions.

## Serving modes

| Capability | Verified default | Optional |
|---|---|---|
| Planning | Deterministic typed DAG | Model Studio registered-tool planner |
| Forecast | Versioned offline snapshot; seasonal fallback | Other snapshot producers with hashes |
| Text evidence | ES BM25 + local hybrid | Local-only fallback |
| Figure evidence | Precompiled workbook records | PDF page / hosted visual reranker |
| Trace | In-process LRU + Redis when configured | — |

## Failure policy

- Incomplete historical settlement windows fail closed.
- Forecast snapshots with a training cutoff after the requested day are rejected at load time.
- Empty visual evidence gets one text fallback; empty text evidence may get one figure/page escalation.
- Missing required tools produce `insufficient_evidence`.
- Citation hashes and the economic boundary are explicit verification fields.
- Linux service and Slurm runs keep per-tool thread timeouts. Windows clean-room
  runs serialize the native SciPy/HiGHS MILP call on the caller thread because
  upstream Windows builds have shown access violations in short-lived worker threads.

## Known limitations

- Current report relevance labels are small and author-curated.
- Model Studio live planning and provider cost require credentials and a separate evaluation.
- The promoted SG loopback release serves the 905-chunk text index, 130 private-mounted workbook figures and 15 hash-aligned forecast snapshots; optional PDF/Qwen page reranking remains offline.
- Historical market operating proxies exclude material costs and market products described in the README boundary.
