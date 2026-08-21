# MiniCheck claim-support gate

## Why this gate exists

Citation IDs and passage retrieval do not establish that a cited passage
semantically supports a generated claim. Following MiniCheck (Tang, Laban and
Durrett, EMNLP 2024), this gate treats sentence-level grounded fact checking as
a separate model task. The verifier is not allowed to select tools or create
optimisation expressions.

## Frozen design

- Model: `lytang/MiniCheck-Flan-T5-Large` (770M parameters).
- Hugging Face revision: `a496016e7b493686ed6e1c52250b9b9d39b0dcb2`.
- Decision threshold: support probability greater than `0.5`.
- Scoring contract: softmax over the checkpoint's published first-decoder-token
  IDs `3` (unsupported) and `209` (supported).
- Challenge set: 20 official AEMO/AER passage-and-claim pairs, each containing
  one supported claim and one author-written controlled counterfactual.
- Counterfactual families: numeric, direction, temporal, entity, and
  quantifier/negation.
- Uncertainty: 5,000 paired bootstrap replicates resampling the 20 facts, so a
  fact's supported and counterfactual examples remain together.

## Promotion gate

The fixed verifier must meet all four criteria without threshold tuning:

1. balanced accuracy at least 0.85;
2. supported-claim recall at least 0.85;
3. counterfactual rejection recall at least 0.85; and
4. paired-bootstrap 95% balanced-accuracy lower bound at least 0.75.

The gate result will be published whether it passes or fails. A pass permits a
claim about this fixed, controlled challenge set only. It does not establish
independent human agreement, live-answer factuality, cross-domain calibration,
or a production SLA.

## Primary references

- Tang, Laban and Durrett, “MiniCheck: Efficient Fact-Checking of LLMs on
  Grounding Documents,” EMNLP 2024: <https://aclanthology.org/2024.emnlp-main.499/>
- Official implementation: <https://github.com/Liyan06/MiniCheck>
