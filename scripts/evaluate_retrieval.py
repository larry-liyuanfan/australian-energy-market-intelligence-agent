from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks

TASKS = [
    ("Q2 2026 South Australia cold still conditions transmission constraints", "aemo-qed-q2-2026"),
    ("Q2 2026 battery price setting dispatch intervals", "aemo-qed-q2-2026"),
    ("Q2 2026 NEM wholesale electricity prices", "aemo-qed-q2-2026"),
    ("June 2026 South Australia restricted imports", "aemo-qed-q2-2026"),
    ("Quarterly Energy Dynamics April June 2026", "aemo-qed-q2-2026"),
    ("Q1 2026 NEM demand wholesale prices", "aemo-qed-q1-2026"),
    ("Quarterly Energy Dynamics January March 2026", "aemo-qed-q1-2026"),
    ("Q1 2026 renewable generation and battery output", "aemo-qed-q1-2026"),
    ("summer 2026 operational demand NEM", "aemo-qed-q1-2026"),
    ("Q1 2026 interconnector flows", "aemo-qed-q1-2026"),
    ("Q4 2025 NEM wholesale electricity price", "aemo-qed-q4-2025"),
    ("Quarterly Energy Dynamics October December 2025", "aemo-qed-q4-2025"),
    ("Q4 2025 battery and renewable output", "aemo-qed-q4-2025"),
    ("Q3 2025 NEM wholesale electricity price", "aemo-qed-q3-2025"),
    ("Quarterly Energy Dynamics July September 2025", "aemo-qed-q3-2025"),
    ("Q3 2025 electricity demand generation mix", "aemo-qed-q3-2025"),
    ("significant electricity prices January March 2026", "aer-significant-q1-2026"),
    ("AER significant prices Tasmania January 2026", "aer-significant-q1-2026"),
    ("AER high price events first quarter 2026", "aer-significant-q1-2026"),
    ("market price cap events January to March 2026 regulator", "aer-significant-q1-2026"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = HybridEvidenceIndex(load_official_chunks(args.evidence))
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "scope": "20-query curated official-source routing benchmark; source-level relevance, not passage factuality"
    }
    for mode in ("bm25", "dense", "rrf", "hybrid_rerank"):
        reciprocal_ranks: list[float] = []
        recalls: list[bool] = []
        for task_index, (query, expected) in enumerate(TASKS, start=1):
            hits = index.search(query, top_k=5, mode=mode)
            sources = [str(hit["source_id"]) for hit in hits]
            reciprocal_rank = next((1 / (rank + 1) for rank, source in enumerate(sources) if source == expected), 0.0)
            reciprocal_ranks.append(reciprocal_rank)
            recalls.append(expected in sources)
            rows.append(
                {
                    "task_id": f"retrieval-{task_index:03d}",
                    "query": query,
                    "expected_source": expected,
                    "mode": mode,
                    "retrieved_sources": sources,
                    "reciprocal_rank": reciprocal_rank,
                }
            )
        metrics[mode] = {
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "recall_at_5": sum(recalls) / len(recalls),
            "mrr_bootstrap_95_interval": [
                float(value)
                for value in np.quantile(
                    np.mean(
                        np.random.default_rng(20260820).choice(
                            np.asarray(reciprocal_ranks), size=(2_000, len(reciprocal_ranks)), replace=True
                        ),
                        axis=1,
                    ),
                    [0.025, 0.975],
                )
            ],
        }
    results_path = args.output / "predictions.jsonl"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "tasks": len(TASKS),
        "label_boundary": "Queries and expected official reports are curated source-routing labels; no passage-level human judgments.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
