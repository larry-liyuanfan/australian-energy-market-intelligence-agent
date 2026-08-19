from __future__ import annotations

import argparse
import json
from pathlib import Path

from .market import download_dispatch_day, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-dispatch-day")
    fetch.add_argument("day", help="YYYYMMDD")
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch-dispatch-day":
        manifest = download_dispatch_day(args.day, args.output)
        manifest_path = args.output.with_suffix(".manifest.json")
        write_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2))
