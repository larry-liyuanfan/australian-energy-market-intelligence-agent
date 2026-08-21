from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.claim_verification import MiniCheckFlanVerifier, binary_metrics
from energy_agent.evidence import load_official_chunks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/minicheck_claim_support.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-id", default="lytang/MiniCheck-Flan-T5-Large")
    parser.add_argument("--model-revision", default="a496016e7b493686ed6e1c52250b9b9d39b0dcb2")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line]
    documents = {document.chunk_id: document for document in load_official_chunks(args.evidence)}
    missing = sorted({str(row["evidence_chunk_id"]) for row in rows} - documents.keys())
    if missing:
        raise ValueError(f"benchmark references missing chunks: {missing}")
    verifier = MiniCheckFlanVerifier(model_id=args.model_id, revision=args.model_revision, device=args.device)
    started = time.perf_counter()
    scores = verifier.score(
        [documents[str(row["evidence_chunk_id"])].text for row in rows],
        [str(row["claim"]) for row in rows],
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started
    predictions: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        predictions.append(
            {
                **row,
                "support_probability": score.support_probability,
                "predicted_label": int(score.predicted_supported),
                "correct": int(score.predicted_supported) == int(row["label"]),
            }
        )
    labels = [int(row["label"]) for row in rows]
    probabilities = [score.support_probability for score in scores]
    metrics: dict[str, Any] = binary_metrics(labels, probabilities)
    metrics["examples"] = len(rows)
    metrics["pairs"] = len({str(row["pair_id"]) for row in rows})
    metrics["elapsed_seconds"] = elapsed
    negative_types = sorted({str(row["perturbation_type"]) for row in rows if int(row["label"]) == 0})
    metrics["rejection_by_perturbation"] = {
        kind: sum(
            int(row["predicted_label"]) == 0
            for row in predictions
            if int(row["label"]) == 0 and row["perturbation_type"] == kind
        )
        / sum(int(row["label"]) == 0 and row["perturbation_type"] == kind for row in predictions)
        for kind in negative_types
    }
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    pair_rows = {pair_id: [row for row in predictions if row["pair_id"] == pair_id] for pair_id in pair_ids}
    rng = np.random.default_rng(20260821)
    bootstrap = []
    for _ in range(5_000):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sample = [row for pair_id in sampled for row in pair_rows[str(pair_id)]]
        bootstrap.append(
            binary_metrics(
                [int(row["label"]) for row in sample],
                [float(row["support_probability"]) for row in sample],
            )["balanced_accuracy"]
        )
    metrics["paired_bootstrap_balanced_accuracy_95_interval"] = [
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]
    gate = {
        "balanced_accuracy_gte_0_85": metrics["balanced_accuracy"] >= 0.85,
        "support_recall_gte_0_85": metrics["support_recall"] >= 0.85,
        "counterfactual_rejection_gte_0_85": metrics["counterfactual_rejection_recall"] >= 0.85,
        "bootstrap_lower_gte_0_75": metrics["paired_bootstrap_balanced_accuracy_95_interval"][0] >= 0.75,
    }
    gate["promotion_pass"] = all(gate.values())
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    manifest = {
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "threshold": 0.5,
        "evidence_sha256": sha256(args.evidence),
        "benchmark_sha256": sha256(args.benchmark),
        "predictions_sha256": sha256(predictions_path),
        "label_boundary": (
            "Author-written, non-blind controlled counterfactuals over official passages; this measures a pinned "
            "verifier on an energy-domain challenge set, not independent human agreement or live-answer factuality."
        ),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.fail_on_gate and not gate["promotion_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
