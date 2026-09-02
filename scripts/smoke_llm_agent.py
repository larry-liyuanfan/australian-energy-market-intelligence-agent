from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from energy_agent.market import fixture_store
from energy_agent.model_agent import AgentPath, MemoryMode, ModelDrivenAgent
from energy_agent.providers import LlamaCppPlanner, OllamaPlanner
from energy_agent.tools import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-model tool-call smoke test")
    parser.add_argument("--provider", choices=("ollama", "llama_cpp"), default="ollama")
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    planner = (
        OllamaPlanner(model=args.model, base_url=args.provider_url, temperature=0.0)
        if args.provider == "ollama"
        else LlamaCppPlanner(model=args.model, base_url=args.provider_url, temperature=0.0)
    )
    run = ModelDrivenAgent(
        ToolRegistry(fixture_store()),
        planner,
        timeout_seconds=10,
    ).run_turn(
        "Diagnose the SA1 price event on 2025-01-02 and find official evidence.",
        conversation_id="real-model-preflight",
        path=AgentPath.constrained_hybrid,
        memory_mode=MemoryMode.structured_state,
        seed=17,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    result_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "llm-agent-runtime-smoke-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "provider": run.metrics.provider,
        "model": run.metrics.model,
        "seed": run.metrics.seed,
        "prompt_tokens": run.metrics.prompt_tokens,
        "completion_tokens": run.metrics.completion_tokens,
        "planner_latency_ms": run.metrics.planner_latency_ms,
        "rejected_model_calls": run.metrics.rejected_model_calls,
        "unsafe_tool_or_dsl_calls": run.metrics.unsafe_tool_or_dsl_calls,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "fixture_only": True,
        "real_model_runtime": run.metrics.provider in {"ollama_local", "llama_cpp_local"},
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
