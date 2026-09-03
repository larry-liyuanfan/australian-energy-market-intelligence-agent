from __future__ import annotations

import numpy as np

from energy_agent.vidore import (
    adapt_vidore_rows,
    paired_bootstrap_ndcg_difference,
    promotion_decision,
    retrieval_metrics,
    single_vector_ranking,
    weighted_rrf,
)


def corpus() -> object:
    return adapt_vidore_rows(
        "infovqa",
        "a" * 40,
        [
            {"questionId": "q1", "query": "solar output", "image_filename": "a.png", "ocr": "solar output"},
            {"questionId": "q2", "query": "battery table", "image_filename": "b.png", "ocr": "battery table"},
        ],
    )


def test_vidore_adapter_deduplicates_documents_and_preserves_qrels() -> None:
    adapted = adapt_vidore_rows(
        "docvqa",
        "b" * 40,
        [
            {"questionId": "q1", "query": "first", "image_filename": "same.png", "ocr": "one"},
            {"questionId": "q2", "query": "second", "image_filename": "same.png", "ocr": "one"},
        ],
    )
    assert len(adapted.documents) == 1
    assert len(adapted.queries) == 2
    assert adapted.queries[0].relevant_document_ids == adapted.queries[1].relevant_document_ids
    assert len(adapted.data_sha256) == 64


def test_single_vector_and_weighted_rrf_are_deterministic() -> None:
    dense = single_vector_ranking(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ["q1", "q2"],
        ["d1", "d2"],
    )
    assert dense == {"q1": ["d1", "d2"], "q2": ["d2", "d1"]}
    fused = weighted_rrf({"one": dense, "two": dense}, {"one": 1.0, "two": 2.0})
    assert fused == dense


def test_metrics_bootstrap_and_two_of_three_promotion_rule() -> None:
    adapted = corpus()
    query_ids = [query.query_id for query in adapted.queries]  # type: ignore[attr-defined]
    relevant = [query.relevant_document_ids[0] for query in adapted.queries]  # type: ignore[attr-defined]
    perfect = {query_id: [document_id] for query_id, document_id in zip(query_ids, relevant, strict=True)}
    reversed_run = {query_ids[0]: [relevant[1], relevant[0]], query_ids[1]: [relevant[0], relevant[1]]}
    assert retrieval_metrics(adapted, perfect)["ndcg_at_5"] == 1.0  # type: ignore[arg-type]
    significance = paired_bootstrap_ndcg_difference(adapted, perfect, reversed_run, samples=500)  # type: ignore[arg-type]
    assert significance["mean_ndcg_at_5_difference"] > 0
    task_result = {
        "significance_vs_ocr_text": {
            "fusion": {"significant_improvement": True},
        }
    }
    decision = promotion_decision(
        {
            "a": task_result,
            "b": task_result,
            "c": {"significance_vs_ocr_text": {"fusion": {"significant_improvement": False}}},
        },
        "fusion",
    )
    assert decision["promotion_pass"] is True
