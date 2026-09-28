"""Opt-in local CLI: real tools and SQLite checkpoints, deterministic planning."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# This offline CLI never exports checkpoints or traces to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.sqlite import SqliteSaver

from energy_agent.graph_runtime import build_graph, invoke_graph, registry_from_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Private two-region official AEMO subset")
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Private local SQLite path")
    parser.add_argument("--thread", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--resume", choices=["approve", "cancel"])
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New private response path; never overwrite")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an earlier response")
    registry = registry_from_subset(args.data, args.data_sha256)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(args.checkpoint)) as saver:
        graph = build_graph(registry, saver)
        result = invoke_graph(graph, args.thread, question=args.question, resume=args.resume, pause=args.pause)
        interrupts = result.pop("__interrupt__", ())
        result["interrupts"] = [item.value for item in interrupts]
        result["checkpoint_count"] = sum(1 for _ in graph.get_state_history({"configurable": {"thread_id": args.thread}}))
    result["execution_mode"] = "live_local_langgraph_deterministic_tools_no_model_inference"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "interrupts": len(result["interrupts"]),
                      "tool_attempts": len(result["calls"]), "thread": args.thread,
                      "checkpoint_count": result["checkpoint_count"]}))


if __name__ == "__main__":
    main()
