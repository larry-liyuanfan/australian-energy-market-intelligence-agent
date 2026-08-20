# Two-year risk/return transport gate

The corrected 18 August 2024–17 August 2026 evaluation already settled a
scenario-CVaR dispatch policy alongside the point-forecast MILP in every one of
the 40 region-season folds. Each fold used only its preceding calibration
window: the first half supplied complete-day residual scenarios and the second
half selected risk aversion 0.25, 0.5, 1.0, or the point fallback. Test prices
were used only to settle the frozen choice.

Before inspecting the corrected two-year aggregate risk result, the promotion
gate is frozen as follows:

1. the paired-bootstrap 95% lower bound of the mean fold-level daily CVaR05
   lift must be above AUD 0;
2. at least three of five regions must improve aggregate daily CVaR05;
3. the five-region mean annualised net-operating proxy must retain at least 95%
   of point-dispatch value;
4. every single region must retain at least 90% of its point-dispatch value;
5. all 40 folds, all selections, margin deltas, tail deltas, and failures are
   reported; point fallback is counted as no tail improvement, not a win.

This is an aggregation of an already-completed **backward historical transport
check**, not a prospective trial or a fresh untouched test. Passing may support
the narrower statement that a calibration-only risk policy improved historical
lower-tail operating proxy while preserving most mean proxy. It cannot support
an ROI, investment, or future-performance claim. The proxy excludes CAPEX,
fixed O&M, network charges, FCAS, taxes, financing, and complete degradation.

## Verified result

Spartan job `29464324` completed the exact-SHA summary at commit `919d3c4`
against evaluation commit `e79b2b7`. The source contained all 40 corrected
two-year folds; seven selected a non-point CVaR policy.

- The risk-aware policy increased the five-region mean annualised proxy from
  AUD 84,792 to AUD 86,191/MW-year (+1.65%), but the mean fold-level daily
  CVaR05 lift was **-AUD 3.62** with a 95% paired-bootstrap interval of
  **-AUD 10.60 to -AUD 0.003**.
- Only QLD1 improved aggregate daily CVaR05. NSW1 and TAS1 worsened, while SA1
  and VIC1 selected point dispatch in every fold.
- TAS1 retained only 88.18% of point-dispatch mean value, below the 90%
  single-region floor.
- The promotion gate therefore failed. The most likely engineering diagnosis
  is that deterministic calibration-residual resampling did not transport the
  price-spike tail well enough across regimes; the higher mean is not evidence
  of lower risk.

The failed gate is retained as a stop result in
`artifacts/public/risk_transport_gate_20260821.json`. No positive CVaR or
low-risk claim enters the resume.
