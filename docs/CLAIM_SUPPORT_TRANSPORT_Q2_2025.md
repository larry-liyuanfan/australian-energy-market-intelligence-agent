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
