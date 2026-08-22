from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.multimodal import LateInteractionPageIndex, MultimodalEvidenceIndex, load_page_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_of(hits: list[dict[str, Any]], expected: set[str]) -> int | None:
    return next((rank for rank, hit in enumerate(hits, start=1) if str(hit["chunk_id"]) in expected), None)


def metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        "mrr": float(np.mean([0.0 if rank is None else 1 / rank for rank in ranks])),
        "recall_at_1": float(np.mean([rank == 1 for rank in ranks])),
        "recall_at_5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pages = load_page_records(args.records)
    benchmark = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((args.embedding_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = json.loads((args.embedding_dir / "ids.json").read_text(encoding="utf-8"))
    page_path = args.embedding_dir / "page_embeddings.npy"
    query_path = args.embedding_dir / "query_embeddings.npy"
    checks = {
        args.records: "records_sha256",
        args.benchmark: "benchmark_sha256",
        page_path: "page_embeddings_sha256",
        query_path: "query_embeddings_sha256",
        args.embedding_dir / "ids.json": "ids_sha256",
    }
    for path, key in checks.items():
        if sha256(path) != manifest[key]:
            raise ValueError(f"hash mismatch for {path}")
    page_embeddings = np.load(page_path, allow_pickle=False)
    query_embeddings = np.load(query_path, allow_pickle=False)
    if page_embeddings.shape[0] != len(ids["page_ids"]) or query_embeddings.shape[0] != len(ids["query_ids"]):
        raise ValueError("embedding row count does not match ids")
    page_by_id = {page.page_id: page for page in pages}
    pages = [page_by_id[page_id] for page_id in ids["page_ids"]]
    selected_query_ids = set(ids["query_ids"])
    benchmark = [row for row in benchmark if row["query_id"] in selected_query_ids]
    query_by_id = {query_id: query_embeddings[index] for index, query_id in enumerate(ids["query_ids"])}
    query_text_to_id = {str(row["query"]): str(row["query_id"]) for row in benchmark}
    visual = LateInteractionPageIndex(
        {page_id: page_embeddings[index] for index, page_id in enumerate(ids["page_ids"])},
        query_encoder=lambda query: query_by_id[query_text_to_id[query]],
        model_id=str(manifest["model_id"]),
        model_revision=str(manifest["model_revision"]),
    )
    index = MultimodalEvidenceIndex(pages, visual)
    details: list[dict[str, Any]] = []
    text_ranks: list[int | None] = []
    visual_ranks: list[int | None] = []
    fusion_ranks: list[int | None] = []
    for row in benchmark:
        query = str(row["query"])
        expected = set(row["expected_page_ids"])
        text_hits = index.search(query, top_k=len(pages))
        visual_hits_raw = visual.search(query, top_k=len(pages))
        visual_hits = [{"chunk_id": hit["page_id"]} for hit in visual_hits_raw]
        fusion_hits = index.search_multimodal(query, top_k=len(pages), preferred_modality=row["preferred_modality"])
        text_rank = rank_of(text_hits, expected)
        visual_rank = rank_of(visual_hits, expected)
        fusion_rank = rank_of(fusion_hits, expected)
        text_ranks.append(text_rank)
        visual_ranks.append(visual_rank)
        fusion_ranks.append(fusion_rank)
        details.append(
            {
                "query_id": row["query_id"],
                "expected_page_ids": sorted(expected),
                "text_rank": text_rank,
                "visual_rank": visual_rank,
                "fusion_rank": fusion_rank,
                "fusion_top5": [hit["chunk_id"] for hit in fusion_hits[:5]],
            }
        )
    rng = np.random.default_rng(20260822)
    text_rr = np.asarray([0 if rank is None else 1 / rank for rank in text_ranks])
    fusion_rr = np.asarray([0 if rank is None else 1 / rank for rank in fusion_ranks])
    differences = []
    for _ in range(5_000):
        sample = rng.integers(0, len(benchmark), len(benchmark))
        differences.append(float(np.mean(fusion_rr[sample] - text_rr[sample])))
    output = {
        "benchmark_examples": len(benchmark),
        "label_boundary": "Author-curated page labels from AEMO figure/table captions; not blind human relevance judgments.",
        "text": metrics(text_ranks),
        "visual": metrics(visual_ranks),
        "fusion": metrics(fusion_ranks),
        "fusion_minus_text_mrr_bootstrap_95_interval": [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ],
        "model": {
            "id": manifest["model_id"],
            "revision": manifest["model_revision"],
            "code_revision": manifest["code_revision"],
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
