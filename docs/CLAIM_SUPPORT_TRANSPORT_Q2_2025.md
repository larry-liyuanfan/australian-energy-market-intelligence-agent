# Source-disjoint claim-support transport gate

## Decision question

The first fixed MiniCheck Flan-T5 gate rejected 90% of controlled
counterfactuals but recalled only 45% of supported energy claims. This follow-up
does not retune that failed test. It asks whether the official MiniCheck
DeBERTa-v3-Large alternative transports to a previously unused AEMO report at
the published 0.5 threshold.

## Frozen protocol

- Candidate: `lytang/MiniCheck-DeBERTa-v3-Large` at immutable Hugging Face
  revision `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb`.
- Implementation reference: official MiniCheck repository commit
  `b58b9fa69acbd1015ec970fa65dd752413a053d2`.
- Threshold: `support_probability > 0.5`; no holdout calibration.
- Holdout source: AEMO Quarterly Energy Dynamics Q2 2025, which is disjoint
  from every passage in the original 20-pair development challenge set.
- Holdout: 14 official passage facts, each paired with one controlled
  counterfactual across numeric, temporal, direction, entity and
  entity-number-binding perturbations.
- Statistics: the same balanced accuracy, class recalls, AUROC, Brier/ECE and
  5,000 paired-bootstrap balanced-accuracy interval used by the first gate.
- Promotion criteria: balanced accuracy, supported recall and counterfactual
  rejection each at least 0.85, with bootstrap lower bound at least 0.75.

The author-written labels are not blind human annotations. A pass would permit
only a bounded source-transport claim for this controlled set; a failure remains
a stop result. Neither outcome authorizes insertion into the online answer path
without an additional live-answer evaluation and latency/resource gate.

## Paper-to-system translation

MiniCheck contributes a sentence-level fact-checking contract and multiple
official lightweight checkpoints. ARES contributes the stronger evaluation
lesson: keep a labeled validation resource separate from the evaluation
population and report uncertainty instead of a single point estimate. Granite
Guardian is retained as a future, larger-model comparison rather than silently
substituted into the 20GB-MIG hot path.

## Primary references

- MiniCheck official implementation: <https://github.com/Liyan06/MiniCheck>
- MiniCheck paper (EMNLP 2024): <https://aclanthology.org/2024.emnlp-main.499/>
- ARES official implementation: <https://github.com/stanford-futuredata/ARES>
- ARES paper (NAACL 2024): <https://aclanthology.org/2024.naacl-long.20/>
- Granite Guardian official implementation: <https://github.com/ibm-granite/granite-guardian>

## Sequential development and final result

The fixed DeBERTa `0.5` Q2 2025 run (job `29491647`) improved supported
recall to 92.9% but rejected only 64.3% of counterfactuals, so it failed. Q2
was then reclassified as development data. A literal/entity/direction
consistency layer and three-state supported/unsupported/abstain policy were
developed on Q2 and Q1 2025. The threshold was calibrated to `0.25`; both
development sources passed, but those numbers are not transport evidence.

The complete cascade was frozen at `c8a194b` before constructing the final
Q4 2024 source-disjoint benchmark. Final job `29491950` completed inference and
wrote all artifacts, then exited non-zero as designed because
`--fail-on-gate` failed. It achieved 85.7% balanced accuracy with abstention as
miss, 71.4% supported recall, 100% counterfactual rejection, 92.9% coverage and
92.3% selective accuracy; the paired-bootstrap interval was 75.0%–96.4%.
Supported recall and selective accuracy missed their frozen gates. The cascade
is not promoted online and is not a positive resume claim. Exact resource,
source, benchmark and output hashes are in
`artifacts/public/claim_support_transport_stop_20260822.json`.
