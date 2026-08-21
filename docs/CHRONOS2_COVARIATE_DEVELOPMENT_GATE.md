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

## Verified result

Spartan job `29481739` completed at exact commit `735d0d1` over 8,064 test
intervals. Covariate Chronos-2 improved MAE to **47.73**, versus **49.67** for
univariate Chronos-2, **61.24** for same-information LightGBM and **105.12** for
persistence. Raw q10–q90 coverage was 81.92%. Preceding-only split conformal
calibration increased coverage to **84.77%** while widening the mean interval
from 170.57 to 179.54 AUD/MWh.

The downstream decision did not improve. LightGBM produced an AUD 1,444.61
28-day net-operating proxy; covariate Chronos-2 produced AUD 413.22 and
univariate Chronos-2 AUD 173.19. The candidate delta against the best comparator
was **-AUD 1,031.39**. Its paired seven-day circular moving-block mean daily
delta was -AUD 36.84 with a 95% interval of **-151.65 to 32.86**. The MAE and
coverage conditions passed, but both economic conditions failed. Per the frozen
protocol, the five-region expansion was not submitted.

The complete job took 2:40 including isolated environment provisioning. Model
evaluation took 56.70 seconds, used 1,542,636 KiB CPU MaxRSS and peaked at
5,074,494,464 CUDA bytes. Exact hashes and the stop decision are in
[`artifacts/public/chronos2_covariate_development_stop_20260821.json`](../artifacts/public/chronos2_covariate_development_stop_20260821.json).
