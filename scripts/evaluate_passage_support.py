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

from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks, tokens


def required_terms_present(text: str, required_terms: list[str]) -> bool:
    document_tokens = set(tokens(text))
    return all(set(tokens(term)).issubset(document_tokens) for term in required_terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/official_passage_support.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line]
    documents = load_official_chunks(args.evidence)
    by_id = {document.chunk_id: document for document in documents}
    index = HybridEvidenceIndex(documents)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "scope": (
            "Twenty author-curated exact-support labels over five official reports. "
            "This is passage retrieval plus label-consistency evaluation, not independent blind annotation "
            "or an LLM semantic-entailment judge."
        ),
        "tasks": len(tasks),
    }
    label_checks = []
    for task in tasks:
        expected = [by_id[chunk_id] for chunk_id in task["expected_chunks"]]
        label_checks.append(
            any(required_terms_present(document.text, task["required_terms"]) for document in expected)
        )
    metrics["gold_label_term_consistency"] = sum(label_checks) / len(label_checks)
    for mode in ("bm25", "dense", "rrf", "hybrid_rerank"):
        reciprocal_ranks: list[float] = []
        recalls: list[bool] = []
        top1_support: list[bool] = []
        for task in tasks:
            hits = index.search(task["query"], top_k=5, mode=mode)
            retrieved = [str(hit["chunk_id"]) for hit in hits]
            gold = set(task["expected_chunks"])
            rank = next((position for position, chunk_id in enumerate(retrieved, start=1) if chunk_id in gold), None)
            reciprocal_rank = 1 / rank if rank is not None else 0.0
            recalled = rank is not None
            reciprocal_ranks.append(reciprocal_rank)
            recalls.append(recalled)
            top1_support.append(bool(retrieved and retrieved[0] in gold))
            rows.append(
                {
                    "task_id": task["task_id"],
                    "mode": mode,
                    "expected_chunks": task["expected_chunks"],
                    "retrieved_chunks": retrieved,
                    "reciprocal_rank": reciprocal_rank,
                    "recall_at_5": recalled,
                    "top1_gold_support": top1_support[-1],
                }
            )
        rng = np.random.default_rng(20260821)
        bootstrap = np.mean(
            rng.choice(np.asarray(reciprocal_ranks), size=(2_000, len(reciprocal_ranks)), replace=True), axis=1
        )
        metrics[mode] = {
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "recall_at_5": sum(recalls) / len(recalls),
            "top1_gold_support_rate": sum(top1_support) / len(top1_support),
            "mrr_bootstrap_95_interval": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        }
    predictions_path = args.output / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "documents": len(documents),
        "tasks": len(tasks),
        "label_boundary": "Author-curated exact-support records; no independent human-blind or LLM entailment claim.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gate_pass = metrics["gold_label_term_consistency"] == 1.0 and metrics["hybrid_rerank"]["recall_at_5"] >= 0.95
    (args.output / "gate.json").write_text(
        json.dumps(
            {
                "gate_pass": gate_pass,
                "criteria": "gold label term consistency = 1.0 and hybrid passage Recall@5 >= 0.95",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.fail_on_gate and not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
