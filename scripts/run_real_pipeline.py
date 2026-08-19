from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed real-data pipeline entrypoint")
    parser.add_argument("--months", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Deliberately fail closed until daily ingest coverage and evidence hashes are present.
    manifest = {
        "status": "blocked",
        "reason": "daily official ingest/coverage gate not yet satisfied",
        "months_requested": args.months,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    raise SystemExit("coverage gate failed; no synthetic substitution permitted")


if __name__ == "__main__":
    main()
