# Frozen-feature Q2 2025 passage holdout

## Why this gate exists

The original 20-claim passage set found that exact numbers and local lexical
terms were being diluted among adjacent report chunks. Those same 20 claims were
then used to select the deterministic numeric/lexical rerank features, so their
positive score is development evidence rather than generalisation evidence.

Before evaluating another report, the retrieval implementation was frozen at
commit `1fd514fcf406f36c2848632b613bcd208e004779`. The holdout runner fails if
`src/energy_agent/evidence.py` differs from that SHA. No retrieval weights or
features were changed after reading the holdout result.

## Data and labels

- Previously unused source: AEMO *Quarterly Energy Dynamics Q2 2025*.
- Official PDF: <https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q2-2025.pdf>.
- Source SHA-256: `50fdfc6fbbf64d526a9087aa7a25b2917fd2140d916c10a98d15627763462c1c`.
- Corpus after adding the report: six reports, 905 chunks.
- Holdout: 14 exact-support queries covering demand, generation, batteries,
  spot prices, gas and WEM outcomes.
- The holdout source is disjoint from the five-report development set.

Labels are author-curated and checked for required-term consistency. They are
not human-blind judgments and do not establish semantic entailment or answer
correctness.

## Frozen result

| Stage | MRR | Recall@5 | Top-1 gold |
|---|---:|---:|---:|
| BM25 | 0.7143 | 1.0000 | 0.4286 |
| 64-D LSA | 0.0560 | 0.2143 | 0.0000 |
| Unweighted RRF | 0.3095 | 0.6429 | 0.0714 |
| Deterministic feature rerank | **0.7500** | **1.0000** | **0.5000** |

The predeclared gate required label consistency 1.0, rerank Recall@5 at least
0.80 and MRR at least 0.60; it passed. The 2,000-sample bootstrap interval for
rerank MRR was 0.6071–0.8929.

This supports retaining BM25 plus the auditable numeric/lexical feature reranker.
It also confirms that the 64-D LSA channel is a weak baseline on exact numeric
passages; it must not be presented as a neural embedding or as the source of the
quality gain. After this frozen-feature gate passed, the sixth report was
promoted and the deployed SG index was reverified at 905/905 chunks, so
this evaluation remains an offline release gate while deployment health is
recorded separately in
[`sg_evidence_905_deployment_20260821.json`](../artifacts/public/sg_evidence_905_deployment_20260821.json).

Compact public evidence:
[`artifacts/public/passage_holdout_q2_2025_20260821.json`](../artifacts/public/passage_holdout_q2_2025_20260821.json).
