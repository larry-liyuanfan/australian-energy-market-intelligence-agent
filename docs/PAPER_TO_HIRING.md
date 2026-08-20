# Paper-to-hiring implementation map

This note records why each research idea was selected, what was implemented, and
what the evidence does **not** establish. It prevents paper names from becoming
unsupported resume keywords.

## Sampled Australian hiring signals

The sample was checked on 20 August 2026 and is deliberately small. It supports
project prioritisation, not a claim about the entire labour market.

| Current role sample | Repeated signal used here | Project evidence target |
|---|---|---|
| [Powerline — Senior Software Engineer, Australia](https://jobs.ashbyhq.com/powerline/4c721307-bb99-4cff-87ce-9d7d5ad25cfb) | Real-time energy-market pipelines, battery optimisation, ML integration, and agentic applications | One traceable market-data → forecast → constrained dispatch → realised-value chain |
| [Kogan — ML Engineer](https://jobs.lever.co/kogan/5dc7a8ea-a5c0-48e1-bfae-273bba001c87) | Practical forecasting, reliable ML services, RAG/agents, monitoring, CI/CD, and measurable business impact | Time-split baselines, fine-grained Agent/RAG evaluation, deployed observability, and bounded economic metrics |
| [Canva — Senior MLE, Research Optimisation](https://www.lifeatcanva.com/en/jobs/6000000001162285/senior-machine-learning-engineer-research-optimisation/) | Turn research into tested, reusable, observable, cost-aware production systems | Method-inspired implementations with typed interfaces, tests, manifests, resource records, and explicit reproduction boundaries |
| [MYOB — Senior AI Engineer](https://jobs.lever.co/myob-2/ac143cf9-be7f-4919-9297-7dfefdabf150) | End-to-end AI ownership, reliability/safety, cloud deployment, monitoring, and business alignment | FastAPI/Elasticsearch/Redis/Prometheus deployment plus failure recovery and decision-quality gates |

The common hiring message is therefore not “used an LLM framework.” It is:
research literacy, end-to-end ownership, decision-aware evaluation, production
reliability, and honest business boundaries.

## Selected papers and implementation decisions

### Adaptive Conformal Inference under distribution shift

- Primary source: Gibbs and Candès, [Adaptive Conformal Inference Under
  Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html),
  NeurIPS 2021.
- Hiring-relevant gap: fixed calibration produced materially different 90%
  coverage across NEM region-season folds.
- Implemented: a leakage-safe online miscoverage controller. Each interval is
  emitted before its test label is observed; the observed miss then updates the
  next alpha, while a bounded residual window responds to regime change.
- Evaluation: fixed split-conformal remains as an ablation. Coverage, width,
  alpha range, region, season, code SHA, and data hash are retained.
- Boundary: this is an ACI-inspired engineering implementation, not a verbatim
  reproduction of the paper's experiments or theoretical guarantee.

### Smart Predict-then-Optimize

- Primary source: Elmachtoub and Grigas, [Smart “Predict, then
  Optimize”](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922),
  *Management Science* 68(1).
- Hiring-relevant gap: MAE alone does not show whether a forecast creates a
  better constrained BESS decision.
- Implemented: for every out-of-time region-season fold and cycling-cost
  scenario, persistence and LightGBM both drive the same 1 MW/2 MWh MILP and are
  settled on actual prices. The evaluation records MAE winner, net-margin
  winner, ranking agreement, and oracle regret.
- Boundary: the models are **not** trained with SPO+ loss. This is a
  decision-focused evaluation inspired by SPO, not decision-focused training.

### Scenario-based CVaR optimisation

- Primary source: Rockafellar and Uryasev, [Optimization of Conditional
  Value-at-Risk](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf),
  *The Journal of Risk* 2(3), 2000.
- Implemented: a scenario-based BESS MILP that maximises expected operating
  margin plus a configurable lower-tail CVaR term while retaining the physical
  SoC and mutually exclusive charge/discharge constraints.
- Implemented evaluation path: ten price scenarios are formed from complete-day
  residual paths. The first half of the preceding calibration window supplies
  the scenario bank; the second half chooses among three risk aversions using a
  tail objective plus a mean-margin guardrail, with point dispatch as fallback.
  The selected policy is then settled on unseen prices against risk-neutral and
  oracle baselines.
- Current evidence: deterministic golden cases verify feasibility, exact solver
  completion, and the expected tail-protection/mean-margin trade-off. The
  real-market result remains outside career materials until its full seasonal
  Slurm artifact passes the data/SHA/resource gates.

### ALCE and RAGChecker

- Primary sources: Gao et al., [Enabling Large Language Models to Generate Text
  with Citations](https://aclanthology.org/2023.emnlp-main.398/), EMNLP 2023;
  Ru et al., [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024 Datasets and Benchmarks Track.
- Hiring-relevant gap: the earlier “citation completeness” metric passed when an
  answer had any citation, even if another generated claim had none.
- Implemented: every deterministic answer claim carries explicit evidence IDs.
  Offline evaluation now separates claim-level citation completeness from
  citation-ID validity, while retrieval source routing remains a separate MRR /
  Recall@5 benchmark.
- Boundary: ID validity and provenance association do not prove semantic
  entailment. Passage-level human labels or a separately validated entailment
  judge are still required before claiming citation correctness.

### Tool-agent reliability and untrusted evidence

- Primary sources: Yao et al., [τ-bench](https://arxiv.org/abs/2406.12045), ICLR
  2025; Debenedetti et al., [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024 Datasets and Benchmarks Track.
- Existing implementation retained: typed tool schemas, bounded parallel calls,
  timeout/retry, loop detection, empty-result recovery, trace persistence, and
  deterministic planner fallback. Retrieved text is data and cannot introduce
  SQL, Elasticsearch DSL, or optimisation expressions.
- Next gate, not a completed result: repeat-trial `pass^k` and adversarial
  untrusted-evidence fixtures. They remain outside resume claims until executed.

## Evidence gate for career materials

Only results that have a real-data artifact, run manifest, data hash, code SHA,
resource record, and passing CI may enter the evidence ledger. A paper citation
by itself is never project evidence. Fixture-only safety tests and planned
semantic entailment evaluation must stay labelled as such.

For Slurm arrays, every task also receives a distinct `%A_%a` log. Metrics and
manifests remain region-sharded, so concurrent stdout cannot obscure which
resource record produced a result.
