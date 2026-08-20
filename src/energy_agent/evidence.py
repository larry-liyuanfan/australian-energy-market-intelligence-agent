from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

TOKEN = re.compile(r"[a-z0-9$%]+")


@dataclass(frozen=True)
class OfficialChunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    url: str
    published_at: str
    retrieved_at: str
    sha256: str
    evidence_type: str = "explanatory"


class EvidenceIndex(Protocol):
    documents: list[OfficialChunk]
    backend: str

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["bm25", "dense", "rrf", "hybrid_rerank"] = "hybrid_rerank",
    ) -> list[dict[str, Any]]: ...


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridEvidenceIndex:
    """BM25 + dense LSA + RRF + deterministic rerank over official chunks."""

    def __init__(self, documents: list[OfficialChunk], dense_dimensions: int = 64) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = documents
        self.backend = "local_hybrid"
        self.tokenized = [tokens(f"{doc.title} {doc.text}") for doc in documents]
        self.term_counts = [Counter(row) for row in self.tokenized]
        self.lengths = np.asarray([len(row) for row in self.tokenized], dtype=float)
        self.average_length = float(np.mean(self.lengths)) or 1.0
        document_frequency: Counter[str] = Counter()
        for row in self.tokenized:
            document_frequency.update(set(row))
        self.idf = {
            term: math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }
        self.vectorizer: Any = None
        self.reducer: Any = None
        self.dense: np.ndarray | None = None
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000)
            sparse = self.vectorizer.fit_transform([f"{doc.title} {doc.text}" for doc in documents])
            dimensions = min(dense_dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
            if dimensions >= 2:
                self.reducer = TruncatedSVD(n_components=dimensions, random_state=20260820)
                self.dense = normalize(self.reducer.fit_transform(sparse))
        except ImportError:
            self.dense = None

    def _bm25(self, query: str) -> np.ndarray:
        query_terms = tokens(query)
        scores = np.zeros(len(self.documents))
        for index, counts in enumerate(self.term_counts):
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                denominator = frequency + 1.2 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += self.idf.get(term, 0.0) * frequency * 2.2 / denominator
        return scores

    def _dense(self, query: str) -> np.ndarray:
        if self.dense is None or self.vectorizer is None or self.reducer is None:
            return np.zeros(len(self.documents))
        from sklearn.preprocessing import normalize

        query_vector = normalize(self.reducer.transform(self.vectorizer.transform([query])))[0]
        return np.asarray(self.dense @ query_vector, dtype=float)

    @staticmethod
    def _ranks(scores: np.ndarray) -> dict[int, int]:
        return {index: rank for rank, index in enumerate(np.argsort(-scores), start=1)}

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["bm25", "dense", "rrf", "hybrid_rerank"] = "hybrid_rerank",
        lexical_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        bm25 = (
            np.asarray([lexical_scores.get(doc.chunk_id, 0.0) for doc in self.documents], dtype=float)
            if lexical_scores is not None
            else self._bm25(query)
        )
        dense = self._dense(query)
        bm25_ranks, dense_ranks = self._ranks(bm25), self._ranks(dense)
        query_terms = set(tokens(query))
        candidates: list[tuple[float, int]] = []
        for index, doc in enumerate(self.documents):
            rrf = 1 / (60 + bm25_ranks[index]) + 1 / (60 + dense_ranks[index])
            title_overlap = len(query_terms & set(tokens(doc.title))) / max(1, len(query_terms))
            exact_boost = 0.02 if query.lower() in doc.text.lower() else 0.0
            if mode == "bm25":
                score = float(bm25[index])
            elif mode == "dense":
                score = float(dense[index])
            elif mode == "rrf":
                score = rrf
            else:
                score = rrf + 0.03 * title_overlap + exact_boost
            candidates.append((score, index))
        output = []
        for rank, (score, index) in enumerate(sorted(candidates, reverse=True)[:top_k], start=1):
            doc = self.documents[index]
            output.append(
                {
                    "rank": rank,
                    "score": score,
                    "bm25_score": float(bm25[index]),
                    "dense_score": float(dense[index]),
                    **doc.__dict__,
                }
            )
        return output


class ElasticsearchHybridEvidenceIndex:
    """Fixed-query Elasticsearch BM25 fused with the local dense/RRF/rerank path."""

    alias = "energy-official-evidence"
    backend = "elasticsearch_bm25+local_dense_rrf_rerank"

    def __init__(self, local: HybridEvidenceIndex, client: Any) -> None:
        self.local = local
        self.documents = local.documents
        self.client = client
        digest = hashlib.sha256(
            "\n".join(f"{doc.chunk_id}:{doc.sha256}" for doc in self.documents).encode()
        ).hexdigest()[:12]
        self.index_name = f"energy-official-evidence-{digest}"
        self.indexed_documents = 0

    def ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
                mappings={
                    "dynamic": "strict",
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "url": {"type": "keyword", "index": False},
                        "published_at": {"type": "keyword", "index": False},
                        "retrieved_at": {"type": "keyword", "index": False},
                        "sha256": {"type": "keyword"},
                        "evidence_type": {"type": "keyword"},
                    },
                },
            )
            operations: list[dict[str, object]] = []
            for doc in self.documents:
                operations.extend(
                    [
                        {"index": {"_index": self.index_name, "_id": doc.chunk_id}},
                        doc.__dict__,
                    ]
                )
            response = self.client.bulk(operations=operations, refresh=True)
            if response.get("errors"):
                raise RuntimeError("Elasticsearch bulk indexing reported errors")
        aliases = self.client.indices.get_alias(name=self.alias) if self.client.indices.exists_alias(name=self.alias) else {}
        actions: list[dict[str, dict[str, str]]] = []
        for old_index in aliases:
            if old_index != self.index_name:
                actions.append({"remove": {"index": old_index, "alias": self.alias}})
        if self.index_name not in aliases:
            actions.append({"add": {"index": self.index_name, "alias": self.alias}})
        if actions:
            self.client.indices.update_aliases(actions=actions)
        self.indexed_documents = int(self.client.count(index=self.alias)["count"])
        if self.indexed_documents != len(self.documents):
            raise RuntimeError(
                f"Elasticsearch index count mismatch: {self.indexed_documents} != {len(self.documents)}"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["bm25", "dense", "rrf", "hybrid_rerank"] = "hybrid_rerank",
    ) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.alias,
            size=min(100, max(20, top_k * 10)),
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "text"],
                    "type": "best_fields",
                }
            },
            source=["chunk_id"],
        )
        scores = {
            str(hit["_source"]["chunk_id"]): float(hit.get("_score") or 0.0)
            for hit in response["hits"]["hits"]
        }
        return self.local.search(query, top_k, mode, lexical_scores=scores)


def load_official_chunks(path: Path) -> list[OfficialChunk]:
    return [OfficialChunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
