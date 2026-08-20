from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[position]


def task_questions(repeats: int) -> list[str]:
    templates = (
        "Explain {region} data coverage",
        "Detect {region} price spike events 2026-01-15",
        "Forecast {region} price risk 2026-02-10",
        "Optimize {region} BESS battery dispatch 2026-02-11",
    )
    return [
        template.format(region=region)
        for _ in range(repeats)
        for region in ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
        for template in templates
    ]


def post(base_url: str, question: str, timeout: float) -> dict[str, Any]:
    body = json.dumps({"question": question}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/agent/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return {
                "http_ok": response.status == 200,
                "task_ok": payload.get("status") == "completed",
                "citation_ok": bool(payload.get("citations")),
                "tool_ok": all(call.get("status") == "ok" for call in payload.get("tool_calls", [])),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "trace_id": payload.get("trace_id"),
                "error": None,
            }
    except Exception as exc:
        return {
            "http_ok": False,
            "task_ok": False,
            "citation_ok": False,
            "tool_ok": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "trace_id": None,
            "error": type(exc).__name__,
        }


def ratio(results: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(result[key]) for result in results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded loopback service-level evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--repeats", type=int, default=5, help="5 produces 100 requests")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=35.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    questions = task_questions(args.repeats)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(post, args.base_url, question, args.timeout_seconds) for question in questions]
        results = [future.result() for future in as_completed(futures)]
    latencies = [float(result["latency_ms"]) for result in results]
    metrics = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "http_success_rate": ratio(results, "http_ok"),
        "task_success_rate": ratio(results, "task_ok"),
        "citation_complete_rate": ratio(results, "citation_ok"),
        "raw_tool_success_rate": ratio(results, "tool_ok"),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "http_success_rate_gte_0_99": metrics["http_success_rate"] >= 0.99,
        "task_success_rate_gte_0_95": metrics["task_success_rate"] >= 0.95,
        "citation_complete_rate_gte_0_95": metrics["citation_complete_rate"] >= 0.95,
        "raw_tool_success_rate_gte_0_95": metrics["raw_tool_success_rate"] >= 0.95,
        "p95_latency_lte_10s": metrics["latency_ms"]["p95"] <= 10_000,
    }
    artifact = {
        "scope": "single-host Docker loopback; not a public-internet SLA or quality benchmark",
        "generated_at": datetime.now(UTC).isoformat(),
        "task_mix": "5 regions x coverage/event/forecast/BESS x repeats",
        "metrics": metrics,
        "gates": gates,
        "error_types": dict(
            sorted(
                {
                    error: sum(result["error"] == error for result in results)
                    for error in {result["error"] for result in results if result["error"]}
                }.items()
            )
        ),
        "trace_ids": [result["trace_id"] for result in results if result["trace_id"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    if args.fail_on_gate and not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
