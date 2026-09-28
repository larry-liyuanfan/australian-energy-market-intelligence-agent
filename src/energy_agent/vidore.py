from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .multimodal import maxsim_late_interaction

DocumentSlice = Literal["table", "chart", "infographic", "text-heavy"]


@dataclass(frozen=True)
class ViDoReDocument:
    document_id: str
    image_filename: str
    ocr_text: str
    slice: DocumentSlice
    image: object


@dataclass(frozen=True)
class ViDoReQuery:
    query_id: str
    text: str
    relevant_document_ids: tuple[str, ...]
    slice: DocumentSlice


@dataclass(frozen=True)
class ViDoReCorpus:
    task_id: str
    dataset_revision: str
    documents: tuple[ViDoReDocument, ...]
    queries: tuple[ViDoReQuery, ...]

    @property
    def data_sha256(self) -> str:
        payload = {
            "task_id": self.task_id,
            "dataset_revision": self.dataset_revision,
            "documents": [
                {
                    "document_id": item.document_id,
                    "image_filename": item.image_filename,
                    "ocr_text_sha256": hashlib.sha256(item.ocr_text.encode()).hexdigest(),
                    "slice": item.slice,
                }
                for item in self.documents
            ],
            "queries": [item.__dict__ for item in self.queries],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _plain_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(_plain_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_plain_text(item) for item in value.values())
    return ""


def infer_document_slice(task_id: str, text: str) -> DocumentSlice:
    lowered = text.lower()
    if "infovqa" in task_id.lower():
        return "infographic"
    if "tatdqa" in task_id.lower() or sum(token in lowered for token in ("table", "total", "year ended")) >= 2:
        return "table"
    if sum(token in lowered for token in ("chart", "graph", "axis", "plot", "figure")) >= 2:
        return "chart"
    return "text-heavy"


def adapt_vidore_rows(task_id: str, dataset_revision: str, rows: Iterable[dict[str, Any]]) -> ViDoReCorpus:
    documents: dict[str, ViDoReDocument] = {}
    query_rows: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        filename = str(row.get("image_filename") or row.get("page") or f"row-{index}")
        document_id = hashlib.sha256(f"{task_id}:{filename}".encode()).hexdigest()[:24]
        text = _plain_text(row.get("ocr")) or _plain_text(row.get("text_description"))
        slice_name = infer_document_slice(task_id, text)
        documents.setdefault(
            document_id,
            ViDoReDocument(
                document_id=document_id,
                image_filename=filename,
                ocr_text=text,
                slice=slice_name,
                image=row.get("image"),
            ),
        )
        query = str(row.get("query", "")).strip()
        if not query:
            raise ValueError(f"{task_id} row {index} has no query")
        query_id = str(row.get("questionId") or row.get("query_id") or f"{task_id}-{index:05d}")
        query_rows.append((query_id, query, document_id))
    queries = tuple(
        ViDoReQuery(
            query_id=query_id,
            text=query,
            relevant_document_ids=(document_id,),
            slice=documents[document_id].slice,
        )
        for query_id, query, document_id in query_rows
    )
    if not documents or not queries:
        raise ValueError(f"{task_id} must contain documents and queries")
    return ViDoReCorpus(
        task_id=task_id,
        dataset_revision=dataset_revision,
        documents=tuple(documents.values()),
        queries=queries,
    )


def single_vector_ranking(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
) -> dict[str, list[str]]:
    query = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if query.ndim != 2 or documents.ndim != 2 or query.shape[1] != documents.shape[1]:
        raise ValueError("single-vector query/document embeddings must share a two-dimensional shape")
    if query.shape[0] != len(query_ids) or documents.shape[0] != len(document_ids):
        raise ValueError("single-vector ids must match embedding rows")
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    documents /= np.maximum(np.linalg.norm(documents, axis=1, keepdims=True), 1e-12)
    scores = query @ documents.T
    return {
        query_id: [document_ids[index] for index in np.argsort(-scores[row_index], kind="stable")]
        for row_index, query_id in enumerate(query_ids)
    }


def late_interaction_ranking(
    query_embeddings: dict[str, np.ndarray],
    document_embeddings: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    return {
        query_id: [
            document_id
            for _score, document_id in sorted(
                (
                    (maxsim_late_interaction(query_value, document_value), document_id)
                    for document_id, document_value in document_embeddings.items()
                ),
                key=lambda item: (-item[0], item[1]),
            )
        ]
        for query_id, query_value in query_embeddings.items()
    }


def weighted_rrf(
    runs: dict[str, dict[str, list[str]]],
    weights: dict[str, float],
    *,
    k: int = 60,
) -> dict[str, list[str]]:
    if set(runs) != set(weights) or any(value < 0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("fusion runs and non-negative weights must have identical non-empty channels")
    query_ids = set.intersection(*(set(run) for run in runs.values()))
    output: dict[str, list[str]] = {}
    for query_id in sorted(query_ids):
        scores: dict[str, float] = defaultdict(float)
        for channel, run in runs.items():
            for rank, document_id in enumerate(run[query_id], start=1):
                scores[document_id] += weights[channel] / (k + rank)
        output[query_id] = [
            document_id for document_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
    return output


def rank_of(ranking: list[str], relevant: tuple[str, ...]) -> int | None:
    relevant_set = set(relevant)
    return next((rank for rank, document_id in enumerate(ranking, start=1) if document_id in relevant_set), None)


def retrieval_metrics(corpus: ViDoReCorpus, run: dict[str, list[str]]) -> dict[str, float]:
    ranks = [rank_of(run.get(query.query_id, []), query.relevant_document_ids) for query in corpus.queries]
    reciprocal = [0.0 if rank is None else 1 / rank for rank in ranks]
    ndcg = [0.0 if rank is None or rank > 5 else 1 / math.log2(rank + 1) for rank in ranks]
    return {
        "queries": float(len(ranks)),
        "ndcg_at_5": float(np.mean(ndcg)),
        "recall_at_1": float(np.mean([rank == 1 for rank in ranks])),
        "recall_at_5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
        "mrr": float(np.mean(reciprocal)),
    }


def sliced_retrieval_metrics(corpus: ViDoReCorpus, run: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for slice_name in ("table", "chart", "infographic", "text-heavy"):
        queries = tuple(item for item in corpus.queries if item.slice == slice_name)
        if queries:
            subset = ViDoReCorpus(
                task_id=corpus.task_id,
                dataset_revision=corpus.dataset_revision,
                documents=corpus.documents,
                queries=queries,
            )
            output[slice_name] = retrieval_metrics(subset, run)
    return output


def paired_bootstrap_ndcg_difference(
    corpus: ViDoReCorpus,
    challenger: dict[str, list[str]],
    baseline: dict[str, list[str]],
    *,
    seed: int = 20260903,
    samples: int = 10_000,
) -> dict[str, float | list[float] | bool]:
    differences: list[float] = []
    for query in corpus.queries:
        challenger_rank = rank_of(challenger.get(query.query_id, []), query.relevant_document_ids)
        baseline_rank = rank_of(baseline.get(query.query_id, []), query.relevant_document_ids)
        challenger_score = 0.0 if challenger_rank is None or challenger_rank > 5 else 1 / math.log2(challenger_rank + 1)
        baseline_score = 0.0 if baseline_rank is None or baseline_rank > 5 else 1 / math.log2(baseline_rank + 1)
        differences.append(challenger_score - baseline_score)
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=np.float64)
    bootstraps = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        bootstraps[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    interval = np.quantile(bootstraps, [0.025, 0.975])
    return {
        "mean_ndcg_at_5_difference": float(np.mean(values)),
        "paired_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        "significant_improvement": bool(interval[0] > 0),
        "samples": samples,
        "seed": seed,
    }


def promotion_decision(task_results: dict[str, dict[str, Any]], candidate: str) -> dict[str, Any]:
    significant = [
        task_id
        for task_id, result in task_results.items()
        if bool(result["significance_vs_ocr_text"][candidate]["significant_improvement"])
    ]
    return {
        "candidate": candidate,
        "significant_tasks": significant,
        "tasks_total": len(task_results),
        "required_tasks": 2,
        "promotion_pass": len(significant) >= 2,
        "rule": "At least 2 of 3 frozen ViDoRe tasks must show a paired-bootstrap nDCG@5 improvement over OCR text.",
    }
