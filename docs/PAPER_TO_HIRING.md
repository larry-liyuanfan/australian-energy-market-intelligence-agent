# Paper-to-hiring implementation map

This note records why each research idea was selected, what was implemented, and
what the evidence does **not** establish. It prevents paper names from becoming
unsupported resume keywords.

## Sampled Australian hiring signals

The sample was rechecked on 21 August 2026 and is deliberately small. It supports
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

### Chronos-2 universal forecasting

- Primary source: Ansari et al., [Chronos-2: From Univariate to Universal
  Forecasting](https://arxiv.org/abs/2510.15821), 2025; official
  [implementation](https://github.com/amazon-science/chronos-forecasting).
- Hiring-relevant question: does a 120M zero-shot time-series foundation model
  justify its GPU and dependency cost on five-minute NEM prices, and does
  forecast quality transport into the constrained battery decision?
- Implemented: an official-pipeline adapter with typed aligned windows, strict
  UTC timestamp normalization, lazy optional dependencies and exact runtime
  manifests. Rolling origins use 14 context days and a 288-step horizon.
  Cross-learning is disabled because later origins contain outcomes that would
  be future information for earlier origins if the batch shared state.
- Evaluation: a seven-day SA1 pilot was followed by a declared five-region,
  28-day gate against persistence and LightGBM through the same BESS MILP. The
  paired economic interval uses seven-day circular moving blocks to retain
  within-week dependence and same-day cross-region pairing.
- Verified result: aggregate delta was positive, but only 2/5 regions improved;
  the moving-block 95% interval crossed zero. Weighted MAE regressed 12.85% and
  raw q10–q90 coverage was 74.33%. Four of five conditions failed, so the model
  was not promoted.
- Failure remediation: on a frozen, earlier non-overlapping SA1 development
  window, past market plus known calendar covariates improved Chronos-2 MAE from
  49.67 to 47.73. Preceding-only split conformal calibration reached 84.77%
  coverage. Same-information LightGBM still beat the candidate's BESS proxy by
  AUD 1,031.39, and the paired moving-block interval crossed zero; the declared
  five-region expansion therefore stopped.
- Production evidence: five isolated A100 MIG tasks completed in 1:49–2:05;
  model evaluation used about 0.74 GiB peak CUDA memory and 25–27 seconds, while
  complete environment provisioning drove job MaxRSS to about 7.68 GiB. A
  missing-NumPy summary failure led to pinned runtime provisioning and separate
  evaluation/summary SHAs.
- Boundary: this is a backward extension partly overlapping the inspected pilot,
  not a prospective or untouched test. The covariate remediation moves backward
  in market time and is development evidence only; it does not establish
  cross-region transport, fine-tuning value or a positive foundation-model gain.

### 2026 electricity-price TSFM evidence

- Primary source: Bui et al., [Empirical evaluation of Time Series Foundation
  Models for Day-ahead and Imbalance Electricity Price Forecasting in
  Belgium](https://arxiv.org/abs/2605.17045), 2026.
- Relevant finding: Chronos-2 was the strongest tested TSFM in ARX mode and beat
  the best comparison ensemble on day-ahead MAE, but lost on most imbalance
  horizons and remained weak under extreme market conditions.
- Project consequence: keep the domain baseline, covariate ablation, price-event
  slices and BESS settlement gate. A positive aggregate from a volatile-price
  foundation model cannot override cross-region consistency, interval coverage
  or decision-value uncertainty.
- Boundary: this repository does not reproduce the Belgian experiment and does
  not use that paper as evidence for the NEM result; it is post-design external
  context consistent with the already measured stop decision.

### Dependence-aware economic uncertainty

- Method source: circular block bootstrap methods preserve local time dependence
  by resampling contiguous blocks rather than individual observations; see the
  method discussion in [Bootstrapping Generalization Error Bounds for Time
  Series](https://link.springer.com/article/10.1007/s13171-026-00452-x).
- Implemented: the Chronos gate first averages five paired regional deltas on
  each calendar day, then resamples seven-day circular blocks. This retains
  cross-region shocks on a day and local serial structure within each block.
- Boundary: 28 days still provide only four weekly blocks and limited price
  regimes. The interval is a stability diagnostic, not a universal performance
  guarantee or a substitute for a later prospective window.

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
  The selected policy is evaluated only at the predeclared 50 AUD/MWh cycling
  cost and settled on unseen prices against risk-neutral and oracle baselines;
  the wider cost grid remains a risk-neutral sensitivity analysis.
- Verified real-market result: across 20 region-season folds, the nested gate
  retained point dispatch 17 times and selected a CVaR candidate three times.
  All three selected candidates worsened unseen realised tail margin; the
  five-region annualised mean moved from AUD 41,048.15 to 41,011.24/MW-year.
  The risk-improvement gate therefore failed and no positive CVaR claim enters
  career materials. Feasibility, calibration-overfit diagnosis and safe
  fallback remain valid engineering evidence.

### Optimiser-informed loss proxy

- Motivation: the verified 9/20 MAE wins versus 17/20 BESS-value wins show that
  uniform point-error weighting is misaligned with the constrained dispatch
  decision.
- Implemented pilot: perfect-foresight schedules from training days only create
  bounded sample weights for charge/discharge intervals. A separate preceding
  calibration window chooses between the baseline and weighted model using mean
  realised margin plus a tail guardrail; unseen seasonal prices are used only
  for settlement.
- SA1 result: the raw candidate added AUD 180.20 over 112 test days, but worsened
  daily CVaR05 from -9.36 to -12.45 AUD. The binary selector therefore retained
  the baseline in 4/4 folds.
- Follow-up gate: before reading the five-region result, a fixed convex grid and
  release rule were declared (positive aggregate, at least 3/5 positive regions,
  and every regional CVaR05 above a 10% tail floor). SA1 selected 0.25 in two
  seasons and improved 1.69%, but the 560-region-day run fell 3.54%, only 2/5
  regions improved, and the paired region-season bootstrap interval crossed
  zero. The cross-region gate rejected the pilot signal.
- Boundary: this is a tested optimiser-informed regression proxy, not an SPO+
  implementation or a positive revenue/risk result.

### ALCE and RAGChecker

- Primary sources: Gao et al., [Enabling Large Language Models to Generate Text
  with Citations](https://aclanthology.org/2023.emnlp-main.398/), EMNLP 2023;
  Ru et al., [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024 Datasets and Benchmarks Track; Stammbach and Neumann,
  [Improving Evidence Retrieval with Claim-Evidence Entailment](https://aclanthology.org/2021.ranlp-1.174/),
  RANLP 2021.
- Hiring-relevant gap: the earlier “citation completeness” metric passed when an
  answer had any citation, even if another generated claim had none.
- Implemented: every deterministic answer claim carries explicit evidence IDs.
  Offline evaluation separates claim-level citation completeness from
  citation-ID validity. A new 20-claim author-curated exact-support development set then
  tests passage retrieval separately from report routing. It exposed 64-D LSA-only
  Recall@5 0.50 and RRF 0.70; numeric-preserving hybrid reranking reached 1.00
  Recall@5 and MRR 0.80 while retaining source-routing MRR/Recall@5 1.00/1.00.
- Boundary: the lexical/numeric features were selected on this same small set,
  so bootstrap intervals do not turn it into untouched-holdout evidence. ID
  validity, term-consistent support labels and passage retrieval do
  not prove answer-level semantic entailment. Independent human-blind labels or
  a separately validated entailment judge remain required before claiming
  citation correctness.

### MiniCheck: sentence-level grounded fact checking

- Paper: Tang, Laban and Durrett, “MiniCheck: Efficient Fact-Checking of LLMs
  on Grounding Documents,” EMNLP 2024.
- Paper idea: score whether a grounding document supports each generated
  sentence with a specialised sub-1B verifier instead of treating retrieval or
  citation presence as factuality.
- Project translation: a pinned MiniCheck Flan-T5 verifier is evaluated on 20
  official energy passages paired with 20 controlled counterfactuals spanning
  numeric, direction, temporal, entity and quantifier/negation errors. The
  threshold and paired-bootstrap gate are frozen before GPU execution.
- Result: counterfactual rejection reached 90%, but supported-claim recall was
  45%; balanced accuracy was 67.5% with a paired-bootstrap 95% interval of
  57.5%–77.5%. Three of four promotion conditions failed, so the verifier stays
  offline and no citation-correctness claim is promoted.
- Hiring signal: separates retrieval evaluation, answer attribution and semantic
  verification, while recording model revision and keeping the verifier outside
  typed-tool planning.
- Boundary: the challenge set is author-written and non-blind. Even a passing
  result is not independent annotation evidence or a guarantee of live-answer
  factuality.

### Tool-agent reliability and untrusted evidence

- Primary sources: Yao et al., [τ-bench](https://arxiv.org/abs/2406.12045), ICLR
  2025; Debenedetti et al., [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024 Datasets and Benchmarks Track; Zhan et al.,
  [InjecAgent](https://arxiv.org/abs/2403.02691), ACL Findings 2024.
- Existing implementation retained: typed tool schemas, bounded parallel calls,
  timeout/retry, loop detection, empty-result recovery, trace persistence, and
  deterministic planner fallback. Retrieved text is data and cannot introduce
  SQL, Elasticsearch DSL, or optimisation expressions.
- Implemented: 24 attacks span instruction override, secret exfiltration, tool
  substitution, SQL/DSL injection, optimisation-expression injection, citation
  spoofing, fake role boundaries and split instructions; eight benign controls
  measure overblocking. Five deterministic repetitions yield 120/120 attack
  plan-integrity passes, zero unsafe tool actions/marker leakage and 40/40
  benign successes. The zero-failure Wilson 95% upper bound remains 3.10%.
- Boundary: retrieved text never enters the deterministic planner, so `pass^5`
  is a regression invariant rather than stochastic model reliability. This is
  not the AgentDojo/InjecAgent benchmark, live-model behaviour or evidence of
  broad prompt-injection robustness; human policy grading remains future work.

## Evidence gate for career materials

Only results that have a real-data artifact, run manifest, data hash, code SHA,
resource record, and passing CI may enter the evidence ledger. A paper citation
by itself is never project evidence. Fixture-only safety tests and planned
semantic entailment evaluation must stay labelled as such.

A failed paper-driven gate may enter the evidence ledger as an engineering stop
result, but it cannot enter resume bullets as a positive accuracy, safety or
economic lift. The Chronos-2 and MiniCheck runs are retained on exactly that
basis.

For Slurm arrays, every task also receives a distinct `%A_%a` log. Metrics and
manifests remain region-sharded, so concurrent stdout cannot obscure which
resource record produced a result.

The evaluation job checks out one requested commit into a detached job-local
Git worktree and exports that same SHA into its run manifest. This is a tested invariant, added
after a queued legacy run completed its metrics but observed a newer shared
checkout at manifest time. That legacy chain is retained as a provenance-gate
failure rather than repaired into a publishable result.
