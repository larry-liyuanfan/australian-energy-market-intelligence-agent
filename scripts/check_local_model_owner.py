"""Linux-only fail-closed ownership check for an explicitly started local model."""
from __future__ import annotations

import argparse
import json
import os
import socket
import urllib.request
from pathlib import Path


def check_owner(pid: int, port: int, model_id: str) -> None:
    os.kill(pid, 0)
    inodes = {os.readlink(fd)[8:-1] for fd in Path(f"/proc/{pid}/fd").iterdir()
              if os.readlink(fd).startswith("socket:[")}
    expected = f"0100007F:{port:04X}"
    listeners = [line.split() for line in Path("/proc/net/tcp").read_text().splitlines()[1:]]
    if not any(row[1] == expected and row[3] == "0A" and row[9] in inodes for row in listeners):
        raise ValueError("local_listener_is_not_owned_by_launched_pid")
    # Disable inherited HTTP proxies for this exact loopback identity check.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
        models = json.load(response)
    if [item.get("id") for item in models.get("data", [])] != [model_id]:
        raise ValueError("model_service_identity_differs")
    os.kill(pid, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.pid is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", args.port))
    else:
        if not args.model_id:
            raise ValueError("model-id required")
        check_owner(args.pid, args.port, args.model_id)


if __name__ == "__main__":
    main()
