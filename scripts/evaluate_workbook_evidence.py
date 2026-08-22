from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, cast
from urllib.request import Request, urlopen

from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks
from energy_agent.workbook_evidence import FigureEvidenceIndex, extract_figure_evidence

SOURCE_ID = "aemo-qed-q2-2026"
PUBLISHED_AT = "2026-07-28"
WORKBOOK_URL = (
    "https://www.aemo.com.au/-/media/files/major-publications/qed/2026/"
    "qed-q2-2026-databook.xlsx?hash=6562E99E34079FBB9C10CD3CD4A34A4F&"
    "rev=d278ba1633a34414b464163292eddbb2&sc_lang=en"
)
FIGURE_LABEL = re.compile(r"Figure\s+(\d+)", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 energy-agent-research/0.1"})
    with urlopen(request, timeout=180) as response:  # fixed official AEMO origin
        return cast(bytes, response.read())


def _reciprocal_rank(items: list[Any], relevant: Callable[[Any], bool]) -> float:
    return next((1.0 / rank for rank, item in enumerate(items, start=1) if relevant(item)), 0.0)


def _recall(items: list[Any], relevant: Callable[[Any], bool], k: int) -> float:
    return float(any(relevant(item) for item in items[:k]))


def _interval(values: list[float], *, iterations: int = 5_000, seed: int = 20260822) -> list[float]:
    if not values:
        return [0.0, 0.0]
    random = Random(seed)
    samples = [mean(random.choice(values) for _ in values) for _ in range(iterations)]
    samples.sort()
    return [samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--text-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    payload = args.workbook.read_bytes() if args.workbook else _download(WORKBOOK_URL)
    retrieved_at = datetime.now(UTC).isoformat()
    figures = extract_figure_evidence(
        payload,
        source_id=SOURCE_ID,
        url=WORKBOOK_URL,
        published_at=PUBLISHED_AT,
        retrieved_at=retrieved_at,
    )
    figure_index = FigureEvidenceIndex(figures)
    text_documents = load_official_chunks(args.text_evidence)
    text_index = HybridEvidenceIndex(text_documents)
    benchmark = _load_jsonl(args.benchmark)

    predictions: list[dict[str, Any]] = []
    text_rr: list[float] = []
    figure_rr: list[float] = []
    text_r5: list[float] = []
    figure_r5: list[float] = []
    for item in benchmark:
        query = str(item["query"])
        expected = int(item["figure_number"])
        text_hits = text_index.search(query, top_k=20, mode="hybrid_rerank")
        figure_hits = figure_index.search(query, top_k=5)

        def relevant_text(hit: dict[str, Any], _expected: int = expected) -> bool:
            labels = [int(value) for value in FIGURE_LABEL.findall(str(hit["text"]))]
            return _expected in labels and len(labels) <= 4

        def relevant_figure(hit: dict[str, Any], _expected: int = expected) -> bool:
            return int(hit["figure_number"]) == _expected

        current_text_rr = _reciprocal_rank(text_hits, relevant_text)
        current_figure_rr = _reciprocal_rank(figure_hits, relevant_figure)
        text_rr.append(current_text_rr)
        figure_rr.append(current_figure_rr)
        text_r5.append(_recall(text_hits, relevant_text, 5))
        figure_r5.append(_recall(figure_hits, relevant_figure, 5))
        predictions.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "expected_figure": f"Figure {expected}",
                "text_rr": current_text_rr,
                "figure_rr": current_figure_rr,
                "text_top5": [hit["chunk_id"] for hit in text_hits[:5]],
                "figure_top5": [hit["figure_id"] for hit in figure_hits],
            }
        )

    differences = [candidate - baseline for candidate, baseline in zip(figure_rr, text_rr, strict=True)]
    text_mrr = mean(text_rr)
    text_recall_at_5 = mean(text_r5)
    figure_mrr = mean(figure_rr)
    figure_recall_at_5 = mean(figure_r5)
    metrics: dict[str, Any] = {
        "evaluation_role": "author_curated_q2_2026_figure_routing_development",
        "queries": len(benchmark),
        "text_chunk_baseline": {"mrr": text_mrr, "recall_at_5": text_recall_at_5},
        "workbook_figure_router": {"mrr": figure_mrr, "recall_at_5": figure_recall_at_5},
        "paired_mrr_delta": mean(differences),
        "paired_bootstrap_mrr_delta_95": _interval(differences),
    }
    gate = {
        "criteria": "figure Recall@5 >= 0.90 and MRR >= text baseline on the author-curated development set",
        "promotion_pass": figure_recall_at_5 >= 0.90 and figure_mrr >= text_mrr,
    }
    figures_path = args.output / "figure_manifest.jsonl"
    figures_path.write_text(
        "".join(json.dumps(figure.public_dict(), ensure_ascii=False, default=str) + "\n" for figure in figures),
        encoding="utf-8",
    )
    predictions_path = args.output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps({"metrics": metrics, "gate": gate}, indent=2), encoding="utf-8")
    manifest = {
        "created_at": retrieved_at,
        "workbook_source": {"source_id": SOURCE_ID, "published_at": PUBLISHED_AT, "url": WORKBOOK_URL},
        "workbook_sha256": hashlib.sha256(payload).hexdigest(),
        "workbook_bytes": len(payload),
        "figures": len(figures),
        "figures_with_images": sum(figure.image_count > 0 for figure in figures),
        "image_objects": sum(figure.image_count for figure in figures),
        "preview_cells": sum(figure.preview_cell_count for figure in figures),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "text_evidence_sha256": hashlib.sha256(args.text_evidence.read_bytes()).hexdigest(),
        "figure_manifest_sha256": hashlib.sha256(figures_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "retrieval": FigureEvidenceIndex.backend,
        "evidence_boundary": (
            "Author-curated figure-routing development labels over one official workbook; "
            "not blind QA, VLM reasoning, OCR accuracy or independent answer correctness."
        ),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gate": gate, "manifest": manifest}, indent=2))
    if args.fail_on_gate and not gate["promotion_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
