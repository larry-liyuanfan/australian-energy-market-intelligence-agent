# Interview defence and code evidence map

## 90-second story

This project turns an LLM market assistant into an auditable decision system
over official Australian NEM data. I built eight typed tools for market
snapshots, region/period comparison, event diagnosis, official-evidence search,
price-risk forecasting, constrained battery dispatch and data-coverage
explanation. The Agent cannot emit arbitrary SQL, Elasticsearch DSL or optimiser
expressions; a state machine validates arguments, budgets calls, retries bounded
failures and records traceable evidence and calculations.

The data gate covers 18 August 2024 through 17 August 2026: 730 complete days,
five NEM regions and 1,051,200 five-minute rows with no duplicate or missing
region/timestamp keys. An interval-ending day-boundary defect in the first build
created five duplicate keys and one ten-minute gap per region; the validator
failed closed, and I repaired only the affected official days from AEMO monthly
MMSDM archives rather than interpolating. The official-document index contains
735 chunks; fused retrieval reached MRR 0.967 and Recall@5 1.00 on the fixed
20-query set, while the Elasticsearch-only path remained a measured negative
baseline.

For the business decision, persistence and LightGBM forecasts drive the same
1 MW/2 MWh battery MILP and are settled only on future realised prices. Across
40 rolling folds and 1,120 region-days, LightGBM won MAE in only 15 folds, but
forecast-driven dispatch beat the threshold rule in 39. At a predeclared
50 AUD/MWh discharged-energy cost, all five regions had positive historical
net-operating proxies; the mean was AUD 84,792/MW-year versus AUD 23,279 for the
rule baseline, with a paired-fold bootstrap interval of AUD 3,079.62–6,912.10.
Mean oracle capture was 57.08%, so the result is useful but far from perfect
foresight.

I also ran the risk idea to a negative conclusion. Seven of 40 folds selected a
non-point scenario-CVaR policy, and mean proxy rose 1.65%, but mean fold CVaR05
worsened by AUD 3.62 with a 95% interval of -10.60 to -0.003; only one of five
regions improved. The predeclared gate failed, so I published the stop result
instead of claiming risk reduction. All monetary values are backward historical
operating proxies: they exclude CAPEX, fixed O&M, network charges, FCAS,
asset-specific degradation and investment return.

## Deep-dive questions

1. Why does interval-ending market data create a day-boundary trap?
2. How did the coverage validator distinguish a missing official row from a duplicate key?
3. Why repair from monthly MMSDM archives instead of interpolating five-minute prices?
4. Which hashes and manifests prove the repaired v2 time axis?
5. How do rolling folds prevent future information from entering forecast features?
6. Why compare persistence, seasonal and LightGBM baselines before a more complex model?
7. Why can MAE lose while dispatch value wins?
8. Where is realised price allowed to enter the pipeline, and where is it prohibited?
9. Which SoC, power, energy, efficiency and terminal-state constraints enter the BESS MILP?
10. Why compare against both a threshold rule and a perfect-foresight oracle?
11. What does oracle capture measure, and why is it not an ROI metric?
12. Why annualise by evaluated days rather than claim a realised calendar-year return?
13. What is paired in the fold bootstrap, and what uncertainty remains outside it?
14. Why report equivalent full cycles and positive-day share alongside margin?
15. How are complete-day residual scenarios built without mixing calibration and evaluation?
16. How does the lower-tail CVaR linearisation interact with expected margin?
17. Why did a +1.65% mean proxy not pass the scenario-CVaR promotion gate?
18. What does only 1/5 improved regions reveal about transport across regimes?
19. Why is a published negative gate stronger evidence than silently retuning risk aversion?
20. How are typed tools safer than free-form ReAct or arbitrary Elasticsearch DSL?
21. How are numerical market evidence and explanatory official-document evidence routed separately?
22. Why did the Elasticsearch-only retrieval path underperform fused retrieval?
23. What do the 100 Agent tasks measure, and which cases are deliberately fault-injected?
24. How are logical task success and raw physical tool attempts reported separately?
25. Which trace, cache and Prometheus fields make a wrong answer replayable?
26. What remains unverified because live Model Studio credentials were not used?
27. Which assumptions prevent the historical operating proxy from becoming an investment recommendation?
28. What evidence would be required before adding FCAS, degradation or prospective bidding claims?

## Code evidence map

| Question area | Code or artifact to open |
|---|---|
| Typed tool schemas and constrained Agent state | `src/energy_agent/schemas.py`, `src/energy_agent/tools.py`, `src/energy_agent/agent.py` |
| AEMO ingestion and v2 time-axis repair | `src/energy_agent/market.py`, `scripts/ingest_dispatch_mmsdm.py`, `scripts/repair_dispatch_gap.py`, `scripts/validate_dispatch_coverage.py` |
| Leakage-safe forecasts and conformal intervals | `src/energy_agent/forecast.py`, `scripts/evaluate_real_market.py` |
| BESS constraints and settlement | `src/energy_agent/battery.py` |
| Rolling evaluation and metric definitions | `src/energy_agent/evaluation.py`, `src/energy_agent/metrics.py` |
| Two-year decision transport gate | `scripts/summarize_history_transport.py`, `docs/HISTORICAL_TRANSPORT_GATE.md`, `artifacts/public/historical_transport_gate_20260821.json` |
| Scenario-CVaR negative gate | `scripts/summarize_risk_transport.py`, `docs/RISK_TRANSPORT_GATE.md`, `artifacts/public/risk_transport_gate_20260821.json` |
| Official-document hybrid retrieval | `src/energy_agent/evidence.py`, `scripts/build_official_evidence.py`, `scripts/evaluate_retrieval.py` |
| Provider boundary and deterministic fallback | `src/energy_agent/providers.py` |
| FastAPI, traces and metrics | `src/energy_agent/api.py`, `scripts/evaluate_service.py` |
| Agent task evaluation | `scripts/evaluate_agent_real.py`, `src/energy_agent/evaluation.py` |
| Reproducible Spartan execution | `scripts/slurm/`, public compact artifacts and run manifests |
| Executable regression evidence | `tests/` |

## Claim boundary

- The two-year result is backward historical transport, not a prospective live trial.
- The monetary metric is a gross/net-operating proxy under an explicit cycling-cost assumption, not accounting profit or ROI.
- The failed scenario-CVaR gate prohibits a claim of lower tail risk.
- Loopback service evaluation is not a public-internet SLA.
- Live Model Studio generation/cost remains unverified; deterministic planning and provider adapters are implemented and tested.
