from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.claim_verification import literal_consistency, selective_cascade_metrics
from energy_agent.evidence import load_official_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    documents = {document.chunk_id: document.text for document in load_official_chunks(args.evidence)}
    missing = sorted({str(row["evidence_chunk_id"]) for row in rows} - documents.keys())
    if missing:
        raise ValueError(f"predictions reference missing chunks: {missing}")

    evaluated: list[dict[str, Any]] = []
    consistencies: list[bool] = []
    for row in rows:
        consistency = literal_consistency(documents[str(row["evidence_chunk_id"])], str(row["claim"]))
        probability = float(row["support_probability"])
        decision = "unsupported" if not consistency.consistent else "supported" if probability > 0.5 else "abstain"
        consistencies.append(consistency.consistent)
        evaluated.append(
            {
                **row,
                "literal_consistency": consistency.__dict__,
                "cascade_decision": decision,
            }
        )
    metrics = selective_cascade_metrics(
        [int(row["label"]) for row in rows],
        [float(row["support_probability"]) for row in rows],
        consistencies,
    )
    pair_ids = sorted({str(row["pair_id"]) for row in evaluated})
    pair_rows = {pair_id: [row for row in evaluated if row["pair_id"] == pair_id] for pair_id in pair_ids}
    rng = np.random.default_rng(20260822)
    bootstrap = []
    for _ in range(5_000):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sample = [row for pair_id in sampled for row in pair_rows[str(pair_id)]]
        sample_metrics = selective_cascade_metrics(
            [int(row["label"]) for row in sample],
            [float(row["support_probability"]) for row in sample],
            [bool(row["literal_consistency"]["consistent"]) for row in sample],
        )
        bootstrap.append(sample_metrics["balanced_accuracy_with_abstention_as_miss"])
    metrics["paired_bootstrap_balanced_accuracy_95_interval"] = [
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]
    gate = {
        "balanced_accuracy_gte_0_85": metrics["balanced_accuracy_with_abstention_as_miss"] >= 0.85,
        "support_recall_gte_0_85": metrics["support_recall"] >= 0.85,
        "counterfactual_rejection_gte_0_85": metrics["counterfactual_rejection_recall"] >= 0.85,
        "coverage_gte_0_80": metrics["coverage"] >= 0.80,
        "selective_accuracy_gte_0_95": metrics["selective_accuracy"] >= 0.95,
        "bootstrap_lower_gte_0_75": metrics["paired_bootstrap_balanced_accuracy_95_interval"][0] >= 0.75,
    }
    gate["promotion_pass"] = all(gate.values())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evaluated), encoding="utf-8"
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
