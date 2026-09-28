from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from energy_agent.v2_artifacts import (
    validate_goal_aggregate,
    validate_goal_run_directory,
    validate_vidore_run_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GoalSpec/ViDoRe v2 artifact schemas and hash links")
    parser.add_argument("--goal-run", type=Path, action="append", default=[])
    parser.add_argument("--goal-aggregate", type=Path)
    parser.add_argument("--vidore-run", type=Path)
    args = parser.parse_args()
    if not args.goal_run and args.goal_aggregate is None and args.vidore_run is None:
        parser.error("provide at least one artifact path")
    goal_runs: list[dict[str, Any]] = []
    output: dict[str, object] = {"goal_runs": goal_runs}
    for path in args.goal_run:
        goal_runs.append(validate_goal_run_directory(path))
    if args.goal_aggregate:
        output["goal_aggregate"] = validate_goal_aggregate(args.goal_aggregate)
    if args.vidore_run:
        output["vidore_run"] = validate_vidore_run_directory(args.vidore_run)
    print(json.dumps({"valid": True, "validated": sorted(output)}, indent=2))


if __name__ == "__main__":
    main()
