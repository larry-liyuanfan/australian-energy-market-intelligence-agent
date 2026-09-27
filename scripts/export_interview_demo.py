from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from evaluate_llm_agent import build_registry

from energy_agent.decision_demo import build_demo_bundle
from energy_agent.model_agent import AgentPath, MemoryMode, ModelDrivenAgent
from energy_agent.providers import LlamaCppPlanner
from energy_agent.remediation import validate_market_windows

QUESTION = (
    "Explain what happened in SA1 on 2025-12-15 with official text and chart evidence, "
    "then replay a 1MW/2MWh BESS with 90% round-trip efficiency using only earlier prices "
    "for the forecast and plan. Report the actual historical settlement separately."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one auditable real historical decision replay")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--forecast-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--provider-url")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    registry = build_registry(args)
    validate_market_windows(
        [{"turns": [{"expected": {"region": "SA1", "date": "2025-12-15"}}]}], registry.store
    )
    planner = LlamaCppPlanner(args.model, args.provider_url, temperature=0.2) if args.provider_url else None
    agent = ModelDrivenAgent(registry, planner, timeout_seconds=10)
    run = agent.run_turn(
        QUESTION, conversation_id="p1-recorded-sa1-20251215",
        path=AgentPath.constrained_hybrid if planner else AgentPath.deterministic,
        memory_mode=MemoryMode.structured_state, seed=args.seed,
    )
    args.private_output.mkdir(parents=True, exist_ok=False)
    (args.private_output / "full_run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    bundle = build_demo_bundle(run, registry.store, QUESTION)
    bundle["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    input_hashes = {}
    for name in ("data", "data_manifest", "evidence", "figures", "forecast_snapshots"):
        with getattr(args, name).open("rb") as handle:
            input_hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    bundle["input_hashes"] = input_hashes
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recorded_run.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    template = Path(__file__).parents[1] / "src/energy_agent/interview_demo.html"
    # The JSON is inert data; escape '<' so retrieved metadata cannot end its script element.
    embedded = json.dumps(bundle, ensure_ascii=True).replace("<", "\\u003c")
    (args.output / "index.html").write_text(template.read_text(encoding="utf-8").replace("__RECORDED_RUN__", embedded), encoding="utf-8")
    print(json.dumps({"verification": bundle["verification"], "trace_id": run.trace_id, "mode": bundle["mode"]}))


if __name__ == "__main__":
    main()
