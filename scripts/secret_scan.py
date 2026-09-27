"""Fail-closed wrapper for the existing bounded, high-confidence rg scan."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PATTERN = (
    r"(-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{20,}|"
    r"gh[opsu]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,})"
)


def scan(root: Path) -> int:
    try:
        result = subprocess.run(
            [
                "rg", "--no-config", "--files-with-matches", "--hidden",
                "--glob", "!.git/**", "--glob", "!.venv/**", "--", PATTERN, str(root),
            ],
            # Only the status is used. Keep bytes so non-UTF-8 filenames cannot
            # raise decoding errors or leak captured content via a traceback.
            capture_output=True, check=False, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("Secret scan ERROR: scanner timed out; result is not clean.")
        return 2
    except OSError as error:
        # Never print exception text or scanner stdout/stderr: those can contain
        # credential values, including in filenames or malformed-input errors.
        print(f"Secret scan ERROR: scanner could not execute ({type(error).__name__}); result is not clean.")
        return 2
    if result.returncode == 1:
        print("High-confidence secret scan completed: no matches.")
        return 0
    if result.returncode == 0:
        print("Secret scan BLOCKED: high-confidence pattern detected; inspect locally without logging secret values.")
        return 1
    print(f"Secret scan ERROR: rg exited {result.returncode}; result is not clean.")
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    raise SystemExit(scan(args.root))


if __name__ == "__main__":
    main()
