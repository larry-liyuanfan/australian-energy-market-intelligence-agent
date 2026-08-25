# Case study: one auditable SA1 BESS decision replay

## Decision question

For 15 December 2025 in SA1, explain observed price events, retrieve the most relevant AEMO report and chart evidence, create a forecast using only earlier information, schedule a standard battery, and report how that fixed schedule would have settled historically.

The intended user is an energy analyst or BESS strategy engineer reviewing a past operating decision. The output is not an instruction for a live asset.

## Why a DAG is necessary

The work cannot be represented faithfully as one parallel tool batch. Event diagnosis needs the interval selected by event detection. Historical settlement must occur only after the forecast-driven schedule is fixed. Evidence verification needs the exact claims and calculations produced upstream.

The versioned `DecisionCase` therefore moves through:

1. intent and NEM-time constraints;
2. structured market context;
3. event detection and association-only diagnosis;
4. official evidence retrieval;
5. as-of forecast lookup or declared seasonal fallback;
6. constrained BESS optimisation;
7. actual-price settlement for a completed day;
8. citation, calculation and economic-boundary verification.

## What the answer contains

- Event interval, RRP and pre/event/post demand, available generation and interchange summaries.
- Official source URL, publication/retrieval time, page or figure identity, modality and hashes.
- Forecast point/interval arrays, training cutoff, model/data hashes or fallback reason.
- Charge, discharge and SoC arrays under 1 MW / 2 MWh, 90% round-trip efficiency, 10–90% SoC and 50% initial/terminal SoC.
- Planned forecast-signal margin, realised historical settlement, oracle regret and explicit economic exclusions.
- A trace showing stage order, retries, alternate-modality recovery, missing tools and verification status.

## Hiring signal

The project is designed to demonstrate one coherent capability: turning multimodal retrieval, time-series ML, constrained optimisation and reliable Agent orchestration into a defensible decision workflow. Individual paper experiments remain useful engineering evidence, but they do not replace the end-to-end product or its release gate.
