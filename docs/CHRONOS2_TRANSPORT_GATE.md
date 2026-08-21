# Chronos-2 five-region transport stop gate

The official 120M-parameter Chronos-2 model was added as a zero-shot forecasting
challenger, not as a replacement selected by model popularity. The adapter uses
the official `Chronos2Pipeline`, 14 days of price context and a 288-interval
day-ahead horizon. Every forecast origin is chronological. Cross-learning is
disabled because later rolling origins contain outcomes that are future data for
earlier origins; allowing information sharing inside the batch would invalidate
the comparison.

The follow-up gate was declared after a seven-day SA1 pilot and before reading
the five-region aggregate. It covers 21 July–17 August 2026, 28 days in each of
the five NEM regions (140 region-days, 40,320 intervals). Persistence, LightGBM
and Chronos-2 drive the same 1 MW/2 MWh daily battery MILP and are settled on
actual prices. Each region is compared with its better persistence/LightGBM
economic baseline. Promotion required all of:

1. positive aggregate operating-proxy delta;
2. a positive 95% lower bound from a paired seven-day circular moving-block
   bootstrap over cross-region mean daily deltas;
3. at least three of five positive regions;
4. weighted MAE no more than 110% of the region-wise best baseline; and
5. raw q10–q90 coverage between 75% and 85%.

## Verified stop result

The aggregate economic delta was positive at **AUD 387.86**, but only SA1 and
TAS1 were positive. NSW1, QLD1 and VIC1 were negative. The paired moving-block
mean daily delta was AUD 2.77 per region with a 95% interval of **-9.03 to
15.63**, so the economic signal was not stable.

Chronos-2 weighted MAE was **28.62** versus **25.36** for the region-wise best
baseline, a ratio of **1.1285**. Raw q10–q90 coverage was **74.33%**; SA1 fell to
63.06%. Four of five promotion conditions failed and `promotion_pass=false`.
The model is therefore retained as a reproducible negative foundation-model
baseline, not promoted as a forecasting or BESS-value improvement.

Corrected evaluation array `29479941` completed five tasks at `dd0459f`; that
exact code explicitly sets `cross_learning=False`. Each full job took 1:48–2:03
including isolated dependency setup, with batch MaxRSS about 7.68 GiB. Model
evaluation itself took 25.21–26.60 seconds and peaked at 794,805,760 CUDA bytes.
Summary job `29479949` ran from the same exact SHA and completed in 10 seconds at
201,920 KiB MaxRSS. An earlier array bound to `d6cc4b6` produced the same numbers,
but its manifest could not prove the explicit cross-learning setting and is
excluded from published provenance. The compact evidence is
[`artifacts/public/chronos2_transport_gate_20260821.json`](../artifacts/public/chronos2_transport_gate_20260821.json).

This extension partly overlaps the inspected SA1 pilot and contains only one
recent 28-day regime, so it is not prospective or an untouched final test. The
moving-block interval preserves within-week and same-day cross-region dependence
but cannot create missing regimes. Raw model intervals were not conformalised.
Economics remain historical spot-market operating proxies at 50 AUD/MWh
discharged and exclude CAPEX, fixed OPEX, FCAS, network fees, taxes, financing,
asset-specific degradation and investment return.
