# Multimodal evidence and agent architecture

## Why this belongs in the energy agent

AEMO and AER reports contain decision-relevant evidence in charts, tables, maps and page layout, not only in extracted text. A production answer such as “show the regional negative-price pattern” should therefore retrieve the relevant page image, preserve the source page and asset hash, and still combine it with structured five-minute market calculations. The multimodal path extends the existing typed tool instead of creating an unconstrained vision side channel.

## Research-to-system mapping

| Source | Reused idea | Energy implementation |
|---|---|---|
| [ColPali](https://github.com/illuin-tech/colpali) at inspected commit `c23838d` and the ICLR 2025 paper | Treat a rendered document page as the retrieval unit; score query tokens against visual patches with late interaction | `maxsim_late_interaction` and `LateInteractionPageIndex` accept a variable number of page-patch vectors and compute mean query-token MaxSim |
| [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) code revision `393e2978` | Use a dual-tower multimodal encoder for efficient first-stage recall and keep cross-modal reranking separate | The pinned 2B model is an offline Spartan adapter; its single vector is consumed by the same visual-index interface as one token |
| [Docling](https://github.com/docling-project/docling) at inspected commit `e1cb2b2` | Preserve a unified, structured document representation rather than flattening every page to text | `PageRecord` carries source/page identity, layout modality, dimensions and independent source/page hashes; the builder can later be replaced by a Docling adapter without changing the tool contract |
| [Magentic-One](https://arxiv.org/abs/2411.04468) | Route work through an orchestrator with explicit progress and recovery rather than letting a model call arbitrary backends | The bounded state machine detects chart/table intent, selects the registered evidence tool and records modality/rank components in the trace |

Repository names are design references, not claimed reproductions of their published benchmark results.

## Data plane

1. `build_multimodal_pages.py` renders each official PDF page at controlled DPI.
2. Each page gets an immutable `asset_id`, source SHA-256, rendered-image SHA-256, page number, dimensions and deterministic coarse modality.
3. Full page images and extracted page text stay under `artifacts/private/` locally or the fixed Spartan artifact directory.
4. `encode_qwen3_vl_pages.py` runs only as an offline GPU batch and writes dense page/query arrays plus an exact-hash manifest.
5. `LateInteractionPageIndex` validates finite, non-zero, dimension-aligned embeddings before retrieval.
6. `MultimodalEvidenceIndex` fuses page-text and visual rankings with weighted RRF, numeric coverage and an explicit modality-match feature.
7. The API returns only page/source provenance, hashes, modality and score components; it never exposes private filesystem paths.

## Agent control plane

The existing `search_official_evidence` input now has two bounded fields:

- `retrieval_mode`: `hybrid_rerank` or `multimodal_fusion`.
- `preferred_modality`: `auto`, `text`, `visual`, `chart` or `table`.

The deterministic planner routes chart/figure/plot and table requests to multimodal fusion. A future live planner must still emit the same Pydantic schema. The tool registry checks whether a multimodal backend is configured and otherwise falls back to the existing text index; no user-supplied SQL, Elasticsearch DSL, file path or model expression is accepted.

## Current verified scope

- Real AEMO Q4 2024 PDF: 87 pages rendered at 144 DPI from source SHA `b900ab9a...d914`.
- Private page assets: 19,692,032 bytes; page records and diagnostics have separate exact hashes.
- Public author-curated benchmark: 14 chart/table page-retrieval queries with expected page IDs.
- Unit/contract coverage includes MaxSim behavior, modality routing, weighted fusion, page-level provenance and prevention of private-path exposure.

Neural retrieval metrics are reported only after the pinned Spartan pilot/full run completes. The 14 labels are author-curated from official captions, not a blind human relevance set.

## Industrial extension points

- Replace the deterministic page classifier with a Docling layout/table adapter while retaining `PageRecord`.
- Store dual-encoder page vectors in Elasticsearch or a vector store; keep ColPali patch tensors in an efficient multi-vector service rather than flattening them.
- Add a Qwen3-VL reranker only over the bounded top-k candidate pages and record GPU/model cost per query.
- Add a chart-grounding tool that returns figure bounding boxes and structured series, then cross-checks extracted values against five-minute market aggregates before answer synthesis.
- Version visual indexes behind an alias and use the same rollback/health-check pattern as the deployed 905-chunk text index.
