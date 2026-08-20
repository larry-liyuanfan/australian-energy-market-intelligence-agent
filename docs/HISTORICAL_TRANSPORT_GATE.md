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
