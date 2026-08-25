# Experiment catalogue

This catalogue preserves negative and superseded results without allowing them to compete with the historical decision-replay product narrative.

## Promoted product evidence

| Artifact | Role |
|---|---|
| `historical_transport_gate_20260821.json` | Corrected two-year, five-region decision-value gate |
| `qed_workbook_figure_routing_holdout_20260822.json` | Source-disjoint workbook figure routing |
| `passage_holdout_q2_2025_20260821.json` | Frozen-feature text passage transport |
| `sg_evidence_905_deployment_20260821.json` | Versioned 905-chunk SG text deployment |
| `production_observability_20260820.json` | Service/metrics evidence |
| `trace_retention_alerting_20260820.json` | Trace capacity and alert-rule evidence |
| `degradation_sensitivity_20260820.json` | User-supplied discharged-energy-cost sensitivity |
| `seasonal_sensitivity_20260820.json` | Region/season stability evidence |

## Rejected but informative

| Artifact | Stop reason |
|---|---|
| `risk_transport_gate_20260821.json` | Mean increased but unseen lower-tail value regressed |
| `decision_weighted_sa1_gate_20260821.json` | Raw gain failed tail guardrail |
| `dispatch_ensemble_gate_20260821.json` | SA1 signal did not transport across five regions |
| `chronos2_transport_gate_20260821.json` | MAE/coverage/cross-region gate failed |
| `chronos2_covariate_development_stop_20260821.json` | MAE improved but BESS value lost to LightGBM |
| `minicheck_claim_support_stop_20260821.json` | Supported-claim recall was insufficient |
| `claim_support_transport_stop_20260822.json` | Counterfactual rejection passed; support recall did not |
| `multimodal_qwen_q4_2024_20260822.json` | Fusion regression fixed but did not beat text |

## Supporting transport, not the online mainline

| Artifact | Boundary |
|---|---|
| `multimodal_qwen_q1_2025_transport_20260822.json` | Same-author PDF page-routing transport; offline Qwen adapter |
| `paper_driven_evaluation_20260821.json` | Method-inspired evaluation and exact-SHA execution |
| `evidence_security_gate_20260821.json` | Development retrieval/security contracts, not live-LLM robustness |

## Superseded or preflight evidence

| Artifact | Status |
|---|---|
| `release_evidence_20260820.json` | One-year v1 release summary; retained for provenance |
| `sg_deployment_20260820.json` | Earlier 735-chunk SG release |
| `aemo_dispatch_preflight_20260817.json` | Data-source feasibility only |
| `spartan_preflight_29432568.json` | Resource preflight only |
| `spartan_ingest_pilot_29432806.json` | Ingest pilot only |

## Fixture/contract only

Deterministic prompt-injection, timeout, empty-result and synthetic market fixtures prove bounded schemas and failure behaviour. They are reported separately from real-window task quality and never used as market-accuracy evidence.
