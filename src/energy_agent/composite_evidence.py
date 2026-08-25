from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from .evidence import EvidenceIndex, OfficialChunk


class CompositeEvidenceIndex:
    """Route bounded evidence requests across text, figure and optional page indexes."""

    backend = "composite_text+figure+optional_visual"

    def __init__(
        self,
        text_index: EvidenceIndex,
        figure_index: EvidenceIndex | None = None,
        page_index: EvidenceIndex | None = None,
    ) -> None:
        self.text_index = text_index
        self.figure_index = figure_index
        self.page_index = page_index
        seen: set[str] = set()
        documents: list[OfficialChunk] = []
        for index in (text_index, figure_index, page_index):
            if index is None:
                continue
            for document in index.documents:
                if document.chunk_id not in seen:
                    documents.append(document)
                    seen.add(document.chunk_id)
        self.documents = documents

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["bm25", "dense", "rrf", "hybrid_rerank"] = "hybrid_rerank",
    ) -> list[dict[str, Any]]:
        return self.text_index.search(query, top_k=top_k, mode=mode)

    @staticmethod
    def _rrf(channels: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
        scores: defaultdict[str, float] = defaultdict(float)
        records: dict[str, dict[str, Any]] = {}
        components: defaultdict[str, dict[str, float]] = defaultdict(dict)
        for channel_index, hits in enumerate(channels, start=1):
            for rank, hit in enumerate(hits, start=1):
                chunk_id = str(hit["chunk_id"])
                scores[chunk_id] += 1 / (60 + rank)
                records[chunk_id] = hit
                components[chunk_id][f"channel_{channel_index}_rrf"] = 1 / (60 + rank)
        output: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(sorted(scores, key=lambda item: (scores[item], item), reverse=True)[:top_k], 1):
            output.append(
                {
                    **records[chunk_id],
                    "rank": rank,
                    "score": scores[chunk_id],
                    "component_scores": {
                        **records[chunk_id].get("component_scores", {}),
                        **components[chunk_id],
                    },
                }
            )
        return output

    def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        preferred_modality: Literal["auto", "text", "visual", "chart", "table"] = "auto",
    ) -> list[dict[str, Any]]:
        if preferred_modality == "text":
            return self.search(query, top_k)
        channels: list[list[dict[str, Any]]] = []
        if preferred_modality in {"chart", "table", "visual"} and self.figure_index is not None:
            channels.append(self.figure_index.search(query, max(top_k * 3, 10)))
        if preferred_modality in {"chart", "table", "visual"} and self.page_index is not None:
            page_search = getattr(self.page_index, "search_multimodal", None)
            if callable(page_search):
                channels.append(page_search(query, max(top_k * 3, 10), preferred_modality))
        channels.append(self.text_index.search(query, max(top_k * 3, 10)))
        return self._rrf(channels, top_k)
