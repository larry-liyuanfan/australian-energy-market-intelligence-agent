from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from .evidence import HybridEvidenceIndex, OfficialChunk, tokens

PageModality = Literal["text", "page_image", "chart", "table", "mixed"]
ModalityPreference = Literal["auto", "text", "visual", "chart", "table"]

VISUAL_INTENT = re.compile(
    r"\b(chart|figure|plot|graph|diagram|visual|screenshot)\b|图表|图中|曲线",
    re.IGNORECASE,
)
TABLE_INTENT = re.compile(r"\b(table|tabular|rows?|columns?)\b|表格|表中", re.IGNORECASE)


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    source_id: str
    title: str
    text: str
    url: str
    published_at: str
    retrieved_at: str
    source_sha256: str
    page_number: int
    modality: PageModality
    asset_id: str
    asset_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.page_number < 1 or self.width < 1 or self.height < 1:
            raise ValueError("page geometry and page number must be positive")
        for digest in (self.source_sha256, self.asset_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("SHA-256 values must be lowercase hexadecimal")

    def to_chunk(self) -> OfficialChunk:
        return OfficialChunk(
            chunk_id=self.page_id,
            source_id=self.source_id,
            title=self.title,
            text=self.text,
            url=self.url,
            published_at=self.published_at,
            retrieved_at=self.retrieved_at,
            sha256=self.source_sha256,
            modality=self.modality,
            page_number=self.page_number,
            asset_id=self.asset_id,
            asset_sha256=self.asset_sha256,
        )


class QueryEncoder(Protocol):
    def __call__(self, query: str) -> np.ndarray: ...


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty vector or matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding rows must have non-zero norm")
    normalised: np.ndarray = matrix / norms
    return normalised


def maxsim_late_interaction(query_tokens: np.ndarray, document_patches: np.ndarray) -> float:
    """ColBERT/ColPali-style mean MaxSim from query tokens to visual page patches."""

    query = _normalise_rows(query_tokens)
    document = _normalise_rows(document_patches)
    if query.shape[1] != document.shape[1]:
        raise ValueError("query and page embeddings must share a dimension")
    return float(np.max(query @ document.T, axis=1).mean())


class LateInteractionPageIndex:
    """Read-only visual index; heavyweight page encoding remains an offline batch job."""

    backend = "visual_maxsim_late_interaction"

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        query_encoder: QueryEncoder | None = None,
        model_id: str = "precomputed",
        model_revision: str = "unknown",
    ) -> None:
        if not embeddings:
            raise ValueError("visual embeddings must not be empty")
        self.embeddings = {page_id: _normalise_rows(value) for page_id, value in embeddings.items()}
        dimensions = {value.shape[1] for value in self.embeddings.values()}
        if len(dimensions) != 1:
            raise ValueError("all page embeddings must share a dimension")
        self.query_encoder = query_encoder
        self.model_id = model_id
        self.model_revision = model_revision

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 20) -> list[dict[str, float | str | int]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ranked = sorted(
            (
                (maxsim_late_interaction(query_embedding, patches), page_id)
                for page_id, patches in self.embeddings.items()
            ),
            reverse=True,
        )
        return [
            {"page_id": page_id, "visual_score": score, "visual_rank": rank}
            for rank, (score, page_id) in enumerate(ranked[:top_k], start=1)
        ]

    def search(self, query: str, top_k: int = 20) -> list[dict[str, float | str | int]]:
        if self.query_encoder is None:
            raise RuntimeError("visual query encoder is not configured")
        return self.search_by_embedding(self.query_encoder(query), top_k)


def infer_modality_preference(query: str, requested: ModalityPreference = "auto") -> ModalityPreference:
    if requested != "auto":
        return requested
    if TABLE_INTENT.search(query):
        return "table"
    if VISUAL_INTENT.search(query):
        return "chart"
    return "auto"


def classify_page_modality(*, image_blocks: int, drawing_objects: int, table_count: int, numeric_ratio: float) -> PageModality:
    if table_count > 0 and (drawing_objects >= 8 or image_blocks >= 2):
        return "mixed"
    if table_count > 0:
        return "table"
    if drawing_objects >= 8 or (image_blocks >= 2 and numeric_ratio >= 0.03):
        return "chart"
    if image_blocks > 0:
        return "mixed"
    return "text"


class MultimodalEvidenceIndex:
    """Page-level text/visual retrieval with weighted RRF and modality-aware reranking."""

    backend = "page_text_hybrid+visual_maxsim+weighted_rrf"

    def __init__(
        self,
        pages: list[PageRecord],
        visual_index: LateInteractionPageIndex | None = None,
        dense_dimensions: int = 64,
    ) -> None:
        if not pages:
            raise ValueError("pages must not be empty")
        if len({page.page_id for page in pages}) != len(pages):
            raise ValueError("page ids must be unique")
        self.pages = {page.page_id: page for page in pages}
        self.documents = [page.to_chunk() for page in pages]
        self.text_index = HybridEvidenceIndex(self.documents, dense_dimensions=dense_dimensions)
        self.visual_index = visual_index
        if visual_index is not None:
            unknown = set(visual_index.embeddings) - self.pages.keys()
            if unknown:
                raise ValueError(f"visual index references unknown pages: {sorted(unknown)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["bm25", "dense", "rrf", "hybrid_rerank"] = "hybrid_rerank",
    ) -> list[dict[str, Any]]:
        return self.text_index.search(query, top_k=top_k, mode=mode)

    def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        preferred_modality: ModalityPreference = "auto",
    ) -> list[dict[str, Any]]:
        preference = infer_modality_preference(query, preferred_modality)
        candidate_k = min(len(self.pages), max(20, top_k * 5))
        text_hits = self.text_index.search(query, top_k=candidate_k, mode="hybrid_rerank")
        text_ranks = {str(hit["chunk_id"]): int(hit["rank"]) for hit in text_hits}
        text_scores = {str(hit["chunk_id"]): float(hit["score"]) for hit in text_hits}
        visual_hits = self.visual_index.search(query, candidate_k) if self.visual_index is not None else []
        visual_ranks = {str(hit["page_id"]): int(hit["visual_rank"]) for hit in visual_hits}
        visual_scores = {str(hit["page_id"]): float(hit["visual_score"]) for hit in visual_hits}
        candidate_ids = set(text_ranks) | set(visual_ranks)
        query_terms = set(tokens(query))
        numeric_terms = {term for term in query_terms if any(character.isdigit() for character in term)}
        scored: list[tuple[float, str, dict[str, float]]] = []
        for page_id in candidate_ids:
            page = self.pages[page_id]
            text_rrf = 1 / (60 + text_ranks[page_id]) if page_id in text_ranks else 0.0
            visual_rrf = 1 / (60 + visual_ranks[page_id]) if page_id in visual_ranks else 0.0
            page_terms = set(tokens(page.text))
            numeric_coverage = len(numeric_terms & page_terms) / max(1, len(numeric_terms))
            modality_match = float(
                (preference == "chart" and page.modality == "chart")
                or (preference == "table" and page.modality == "table")
                or (preference == "visual" and page.modality != "text")
            )
            # Rank evidence is the primary signal.  Numeric and modality
            # metadata are deterministic tie-breakers only: their old additive
            # bonuses were larger than an RRF rank step and could demote a page
            # ranked first by both retrievers.
            # Modality intent decides whether the more expensive visual channel
            # runs; it does not grant that channel an uncalibrated score boost.
            # Equal-weight RRF is the safe default until a source-disjoint route
            # calibration set demonstrates a stable alternative.
            score = text_rrf + visual_rrf
            scored.append(
                (
                    score,
                    page_id,
                    {
                        "text_rrf": text_rrf,
                        "visual_rrf": visual_rrf,
                        "text_score": text_scores.get(page_id, 0.0),
                        "visual_score": visual_scores.get(page_id, 0.0),
                        "numeric_coverage": numeric_coverage,
                        "modality_match": modality_match,
                    },
                )
            )
        output: list[dict[str, Any]] = []
        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                item[2]["numeric_coverage"],
                item[2]["modality_match"],
                item[1],
            ),
            reverse=True,
        )
        for rank, (score, page_id, component_scores) in enumerate(ranked[:top_k], start=1):
            page = self.pages[page_id]
            output.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": page.page_id,
                    "source_id": page.source_id,
                    "title": page.title,
                    "text": page.text,
                    "url": page.url,
                    "published_at": page.published_at,
                    "retrieved_at": page.retrieved_at,
                    "sha256": page.source_sha256,
                    "evidence_type": "explanatory",
                    "modality": page.modality,
                    "page_number": page.page_number,
                    "asset_id": page.asset_id,
                    "asset_sha256": page.asset_sha256,
                    "component_scores": component_scores,
                    "visual_model_id": self.visual_index.model_id if self.visual_index else None,
                    "visual_model_revision": self.visual_index.model_revision if self.visual_index else None,
                    "visual_available": self.visual_index is not None,
                }
            )
        return output


def load_page_records(path: Path) -> list[PageRecord]:
    return [PageRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def embedding_manifest_sha256(page_ids: list[str], model_id: str, model_revision: str) -> str:
    payload = json.dumps(
        {"page_ids": page_ids, "model_id": model_id, "model_revision": model_revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def reciprocal_rank_fusion_bound(result_count: int, channels: int = 2, k: int = 60) -> float:
    if result_count < 1 or channels < 1 or k < 1:
        raise ValueError("RRF parameters must be positive")
    return channels / (k + 1)
