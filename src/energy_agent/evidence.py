from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridEvidenceIndex:
    """BM25 + dense LSA + RRF + deterministic rerank over official chunks."""

    def __init__(self, documents: list[OfficialChunk], dense_dimensions: int = 64) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = documents
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

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        bm25 = self._bm25(query)
        dense = self._dense(query)
        bm25_ranks, dense_ranks = self._ranks(bm25), self._ranks(dense)
        query_terms = set(tokens(query))
        candidates: list[tuple[float, int]] = []
        for index, doc in enumerate(self.documents):
            rrf = 1 / (60 + bm25_ranks[index]) + 1 / (60 + dense_ranks[index])
            title_overlap = len(query_terms & set(tokens(doc.title))) / max(1, len(query_terms))
            exact_boost = 0.02 if query.lower() in doc.text.lower() else 0.0
            candidates.append((rrf + 0.03 * title_overlap + exact_boost, index))
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


def load_official_chunks(path: Path) -> list[OfficialChunk]:
    return [OfficialChunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
