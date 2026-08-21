# Chronos-2 covariate development gate

This protocol was frozen after the univariate Chronos-2 five-region stop result
and before running the covariate candidate. It is a remediation/development
experiment, not a new claim of independent confirmation.

## Fixed scope

- Region pilot: SA1 only.
- Test window: 23 June–20 July 2026, the 28 complete days immediately preceding
  and non-overlapping with the 21 July–17 August univariate gate.
- Interval calibration: the preceding 14 complete days only.
- Forecast: 14 context days, 288 five-minute intervals day-ahead.
- Candidate: official `amazon/chronos-2` with past-only total demand, available
  generation and net interchange plus known-future calendar sin/cos features.
- Comparators: persistence, LightGBM with the same past market information and
  calendar features, and univariate Chronos-2 on the identical origins.
- Chronos cross-learning: disabled for every rolling origin.
- BESS and value boundary: the same 1 MW/2 MWh daily MILP and 50 AUD/MWh
  discharged-energy cost; historical operating proxy only.

The candidate may expand to a five-region development run only if it beats the
best comparator's aggregate operating proxy, its seven-day paired circular
moving-block 95% lower bound is positive, MAE is no worse than 110% of the best
comparator, and leakage-safe conformal q10–q90 coverage lies between 75% and
85%. Failure of any condition stops expansion.

Even a pass cannot enter the resume as a model lift. The model and feature design
were chosen after inspecting a later Chronos window, and this test moves backward
in market time. A positive result would only justify freezing the design for a
future, newly arrived prospective window. Raw and conformal intervals, all
comparators, runtime, memory, code/data hashes and negative conditions must be
retained.
