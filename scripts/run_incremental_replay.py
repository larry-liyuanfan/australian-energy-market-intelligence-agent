"""Private local CLI for persisted replay. No implicit model or paid endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.sqlite import SqliteSaver

from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks
from energy_agent.graph_runtime import registry_from_subset
from energy_agent.incremental_graph import build_incremental_graph, invoke_replay
from energy_agent.loopback import loopback_transport
from energy_agent.providers import LlamaCppPlanner
from energy_agent.snapshots import load_forecast_snapshots
from energy_agent.tools import ToolRegistry


def registry(args: Any) -> ToolRegistry:
    base = registry_from_subset(args.data, args.data_sha256)
    for path, expected in ((args.evidence, args.evidence_sha256), (args.snapshots, args.snapshots_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("private input hash mismatch")
    return ToolRegistry(base.store, HybridEvidenceIndex(load_official_chunks(args.evidence)),
                        load_forecast_snapshots(args.snapshots, expected_data_sha256=base.store.evidence[0].sha256))


def inputs(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    for name in ("data", "evidence", "snapshots"):
        parser.add_argument(f"--{name}", type=Path, required=required)
        parser.add_argument(f"--{name}-sha256", required=required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--question")
    parser.add_argument("--resume", choices=["approve", "cancel", "correct"])
    parser.add_argument("--correction")
    parser.add_argument("--pause-after")
    parser.add_argument("--full-rerun", action="store_true")
    parser.add_argument("--provider", choices=["none", "llama_cpp"], default="none")
    parser.add_argument("--provider-url", default="http://127.0.0.1:11629/v1")
    parser.add_argument("--provider-model", default="Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; choose a new private response path")
    base = registry(args)
    url = urlsplit(args.provider_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"} or url.username or url.password:
        raise ValueError("only a loopback model server is permitted")
    planner = LlamaCppPlanner(base_url=args.provider_url, model=args.provider_model, max_tokens=512,
                             timeout_seconds=45, transport=loopback_transport) if args.provider == "llama_cpp" else None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(args.checkpoint)) as saver:
        graph = build_incremental_graph(base, saver, planner=planner, mode="hybrid" if planner else "deterministic",
                                        incremental=not args.full_rerun, seed=args.seed)
        result = invoke_replay(graph, args.thread, question=args.question,
                               resume={"action": args.resume, "question": args.correction} if args.resume else None,
                               pause_after=args.pause_after)
        result["interrupts"] = [i.value for i in result.pop("__interrupt__", ())]
    result["execution_mode"] = "loopback_provider_requested" if result["execution_policy"]["mode"] == "hybrid" else "deterministic_no_model"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "calls": len(result["calls"]),
                      "reuse": sum(p["action"] == "reuse" for p in result["provenance"]),
                      "planner_attempts": len(result["planner_attempts"]), "interrupts": len(result["interrupts"])}))


if __name__ == "__main__":
    main()
