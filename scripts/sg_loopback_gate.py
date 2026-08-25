from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, cast

CASES = (
    ("event", "Detect and explain price events in SA1 on 2025-12-15"),
    ("figure", "Show the official SA1 price chart around 2025-12-15"),
    ("replay", "Replay a 1 MW / 2 MWh BESS dispatch for SA1 on 2025-12-15"),
)


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return cast(dict[str, Any], json.load(response))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the loopback decision-replay deployment")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 3:
        raise ValueError("requests must be at least three")
    health = request_json(f"{args.base_url}/healthz")
    checks: list[bool] = []
    latencies: list[float] = []
    workflow_counts: dict[str, int] = {}
    for index in range(args.requests):
        case, question = CASES[index % len(CASES)]
        started = time.perf_counter()
        response = request_json(
            f"{args.base_url}/api/agent/query",
            {"question": question, "max_tool_calls": 8},
        )
        latencies.append((time.perf_counter() - started) * 1000)
        workflow = str(response.get("workflow_type"))
        workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
        citations = response.get("citations", [])
        valid = (
            response.get("status") == "completed"
            and bool(citations)
            and bool(response.get("verification", {}).get("required_tools_satisfied"))
            and request_json(
                f"{args.base_url}/api/agent/traces/{response['trace_id']}"
            ).get("trace_id")
            == response["trace_id"]
        )
        if case == "figure":
            valid = valid and any(
                citation.get("modality") in {"chart", "table", "mixed"}
                and citation.get("asset_id")
                and citation.get("asset_sha256")
                for citation in citations
            )
        if case == "replay":
            valid = valid and (
                response.get("forecast", {}).get("forecast_source") == "offline_snapshot"
                and response.get("dispatch", {}).get("forecast_source") == "offline_snapshot"
                and response.get("forecast", {}).get("forecast_snapshot_id")
                == response.get("dispatch", {}).get("forecast_snapshot_id")
                and response.get("dispatch", {}).get("margin_basis")
                == "historical_actual_settlement_after_as_of_schedule"
            )
        checks.append(bool(valid))
    result = {
        "schema_version": "sg-loopback-decision-replay-v1",
        "requests": args.requests,
        "success_rate": sum(checks) / len(checks),
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "max_latency_ms": max(latencies),
        "workflow_counts": workflow_counts,
        "health": health,
        "promotion_pass": all(checks) and percentile(latencies, 0.95) < 2000,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["promotion_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
