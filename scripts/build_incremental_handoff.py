"""Regenerate compact CPU-only handoff artifacts, never promote synthetic scores."""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(args.contracts.read_text())
    if not contract["contract_fixture_only"] or contract.get("model_requested", True):
        raise ValueError("this artifact builder is only for the retained CPU mechanics run")
    groups = {}
    for path in ("legacy", "full_rerun", "incremental"):
        rows = [r for r in contract["rows"] if r["path"] == path]
        groups[path] = {"turn_rows": len(rows), "authored_task_contract_matches": sum(r["score"]["task_contract_success"] for r in rows),
            "runner_exceptions": sum(bool(r.get("failure")) for r in rows),
            "intervention_rows": [{"episode_id": r["episode_id"], "fault": r["controlled_fault"],
                                   **r["intervention_score"]} for r in rows if r["intervention_score"]]}
    compact = {"schema_version": "incremental-cpu-contract-audit-v1", "validation_date": "2026-09-29",
        "data_track": "synthetic_contract_only", "model_requests": 0, "is_model_quality_result": False,
        "source_summary_sha256": sha(args.contracts), "job": "31484808",
        "job_terminal": "FAILED 127:0 after completed traversal: CUDA-linked --help on CPU lacked libcuda.so.1",
        "source_archive_sha256": "728a7f95bac1ea796bd9630cace1d8e686e40ef677cee7c200e8322f16f250e2",
        "groups": groups, "caveats": [
            "All 144 path-turns retained; a completed traversal is not an overall successful Slurm job.",
            "43/48 legacy includes one status taxonomy mismatch; not a five-task natural/model quality improvement.",
            "Target intervention scoring requires reachability and expected reason, separately from task status.",
            "Model prompt injection is not exercised on CPU. Legacy has no checkpoint correction lifecycle.",
            "Post-run fixes only narrow model-response attribution and disable proxy/redirects; CPU paths unchanged."]}
    licenses = []
    for path in sorted(args.wheels.glob("*.whl")):
        with zipfile.ZipFile(path) as wheel:
            meta = email.message_from_bytes(wheel.read(next(n for n in wheel.namelist() if n.endswith("METADATA"))))
        license_value: Any = meta.get("License-Expression") or meta.get("License") or [
            v for v in meta.get_all("Classifier", []) if v.startswith("License ::")]
        if not license_value:
            raise ValueError(f"unreviewed license metadata: {path.name}")
        licenses.append({"package": meta["Name"], "version": meta["Version"], "declared_license": license_value})
    patterns = ["src/energy_agent/incremental*.py", "src/energy_agent/loopback.py", "src/energy_agent/providers.py",
                "scripts/*incremental*.py", "scripts/check_local_model_owner.py", "scripts/slurm/incremental*.sbatch",
                "tests/test_incremental*.py", "benchmarks/incremental*", "requirements/agent-graph-linux-py311-wheels.json",
                "docs/INCREMENTAL_AGENT_CPU_HANDOFF_20260929.md", "artifacts/public/incremental_cpu_pilot_20260929.json",
                ".github/workflows/quality.yml", ".gitattributes"]
    paths = sorted({p for pattern in patterns for p in root.glob(pattern)})
    manifest = {"schema_version": "incremental-agent-cpu-manifest-v1", "validation_date": "2026-09-29",
        "date_basis": "task Sydney date; remote scheduler clock differs, keep raw job IDs/elapsed logs",
        "git_base": "4333f2862ff1f06a277c621f4bc3733c2dc1272e",
        "delivery_sha_command": "git log -1 --format=%H -- artifacts/public/incremental_cpu_manifest_20260929.json",
        "new_model_inference": False, "gpu_jobs_submitted": 0, "paid_api_calls": 0,
        "new_live_model_metrics": None, "cpu_rows_are_not_llm_pass_rate": True,
        "runtime_bundle": {"file": args.bundle.name, "sha256": sha(args.bundle)},
        "revised_contract_bundle": {"file": "energy-runtime-03.tar", "sha256": "ca7922f39e10095232f01be8e6697a585df040b9cdd141ec668c69fddaca70c2"},
        "model_cache": {"revision": "1d54a16a18cba0d8fbad4a16db801decc729e099", "file": "Qwen3-8B-Q4_K_M.gguf",
            "bytes": 5027783488, "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
            "full_file_hash_verified": True, "new_graph_model_load_verified": False},
        "private_inputs": {
            "annual_parent": "9025d32d1d949dfc1e23329384428241a37e00131d306d3fe6478a898b209567",
            "december_market": "4539f94624e603fc5d44572966c0ee5c24d976417e786a4e387d0711cabfa94f",
            "official_text": "9c56543061843f95b9eec85e62be5a1fec50dcc85e971197e0c50fbf00aee3cc",
            "december_snapshots": "b27f3b192dc1726e9b6abfc390b655b411fe5e4f32b062d9c859a8e69e03b48d",
            "november_archive": "793e781d5758629f91ebb72491ea903683094c00abd5bf4c97c1623e148e3fa6",
            "november_market": "76547ea72e398391faccda799ad8aaaac3b9163374ce7d4a946dc7483795c4f2",
            "november_snapshots": "ab4563e7d7c1f1e859f0acaa3432759bcb3403e7cca37314849f5b7f63be7fbb"},
        "cpu_jobs": [
            {"job": "31484156", "state": "COMPLETED", "exit": "0:0", "seconds": 78, "peak_rss_kib": 141236, "role": "dependencies_inputs_model_download"},
            {"job": "31484355", "state": "COMPLETED", "exit": "0:0", "seconds": 80, "peak_rss_kib": 1723208, "role": "initial_synthetic_contracts"},
            {"job": "31484808", "state": "FAILED", "exit": "127:0", "seconds": 89, "peak_rss_kib": 1964220, "role": "revised_contracts_completed_then_invalid_cpu_cuda_help_check"}],
        "allocation_per_cpu_job": {"cpus": 2, "memory_gib": 4}, "infrastructure_cost_aud": None,
        "cost_boundary": "No paid provider requests; allocated CPU time is reported, university infrastructure price not supplied.",
        "license_review": {"scope": "41 optional Linux wheel metadata declarations, wheels remain private",
                           "licenses": licenses, "notes": "Certifi/orjson include MPL-2.0; no binary redistribution in GitHub. Not a legal opinion or whole-base-environment audit."},
        "checks": {"real_pilot_path_turns": 12, "real_pilot_unique_days": 1,
                   "synthetic_contract_path_turns": 144, "ruff": "passed", "mypy": "89 sources passed",
                   "secret_scan": "executed; no matches", "github_ci": "inspect delivery commit run, not older green badges"},
        "files": [{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha(p)} for p in paths]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, report in (("incremental_contract_cpu_20260929.json", compact), ("incremental_cpu_manifest_20260929.json", manifest)):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"manifest_files": len(paths), "license_declarations": len(licenses), "public_outputs": 2}))


if __name__ == "__main__":
    main()
