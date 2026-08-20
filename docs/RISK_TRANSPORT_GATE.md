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

The verified result is appended only after the exact-SHA Spartan summary job
finishes. A failed gate remains public as a stop result.
