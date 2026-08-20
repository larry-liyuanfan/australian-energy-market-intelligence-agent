# Historical transport gate

The original release evaluates 18 August 2025 to 17 August 2026.  Results from
that period have already informed engineering decisions, so another model tuned
against the same labels cannot be presented as a new untouched test.

The next experiment therefore adds the earlier period 18 August 2024 to
17 August 2025 as a **backward temporal transport check**.  It answers whether
the already frozen point-forecast/dispatch protocol behaves similarly in a
different market regime.  It is not a prospective trial and it cannot erase the
negative five-region ensemble gate on the original period.

Before downloading that year, the acceptance rules are frozen as follows:

1. official daily AEMO coverage must be complete after any documented MMSDM
   repair; no interpolation or synthetic fill is permitted;
2. all five regions and all complete 28-day seasonal folds are reported;
3. the LightGBM-driven dispatch must beat the threshold-rule dispatch on net
   operating-margin proxy in at least 60% of region-season folds;
4. at least four of five regions must have positive annualised net operating
   margin at the fixed 50 AUD/MWh discharged-energy cost;
5. point MAE, interval coverage/width, net proxy, equivalent full cycles,
   oracle regret and lower-tail daily margin are reported together;
6. a pass may be described only as historical transport robustness.  A fail is
   retained as a stop result, and no further tuning on this period is promoted.

The economic metric remains a historical spot-market operating proxy.  It
excludes CAPEX, network charges, FCAS, taxes, financing and a complete physical
degradation model, and is not investment return.

## Verified result

The exact-SHA chain completed on Spartan on 21 August 2026. The earlier and
current periods each contain 525,600 official rows; the joined two-year table
contains 1,051,200 rows with zero duplicate region/timestamp keys and zero
five-minute gaps. Evaluation SHA `e79b2b7` produced eight 28-day folds per
region (40 total); summary SHA `248cd62` evaluated the frozen rules above.

- LightGBM-driven dispatch beat the threshold rule in 39/40 folds (97.5%).
- All five regions had positive annualised net-operating proxies at the fixed
  50 AUD/MWh discharged-energy cost.
- Mean annualised proxy was AUD 84,792/MW-year versus AUD 23,279/MW-year for
  the threshold rule; the paired-fold mean improvement was AUD 4,718.80 with a
  95% bootstrap interval of AUD 3,079.62–6,912.10.
- Point MAE won only 15/40 folds, preserving the decision-focused distinction
  between forecast error and downstream dispatch value.
- Regional daily CVaR05 remained negative (-155.05 to -8.36 AUD), so the gate
  supports historical transport robustness, not a low-risk or investment claim.

The compact artifact records Slurm IDs, code/data hashes and the superseded-v1
boundary: `artifacts/public/historical_transport_gate_20260821.json`.
