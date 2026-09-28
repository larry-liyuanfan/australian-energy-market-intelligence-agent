"""Compile the official workbook needed by the recorded replay, on a compute node."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from energy_agent.evidence_scope import official_evidence_url
from energy_agent.workbook_evidence import extract_figure_evidence

URL = (
    "https://www.aemo.com.au/-/media/files/major-publications/qed/2025/"
    "qed-q4-2025-databook.xlsx?rev=72b25ad0d0c04d4d8c1740914a392c67&sc_lang=en"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace an existing workbook compilation")
    request = Request(URL, headers={"User-Agent": "energy-agent-research/0.1"})
    with urlopen(request, timeout=60) as response:
        if not official_evidence_url(response.url):
            raise ValueError("official workbook redirected outside the approved origin")
        payload = response.read(10_000_001)
    if len(payload) > 10_000_000 or not payload.startswith(b"PK"):
        raise ValueError("not a bounded XLSX payload")
    retrieved = datetime.now(UTC).isoformat()
    figures = extract_figure_evidence(
        payload, source_id="aemo-qed-q4-2025", url=URL,
        published_at="2026-01-29", retrieved_at=retrieved,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.xlsx").write_bytes(payload)
    records = "".join(json.dumps(figure.public_dict(), ensure_ascii=False) + "\n" for figure in figures)
    (args.output / "figure_manifest.jsonl").write_text(records, encoding="utf-8")
    manifest = {
        "url": URL, "publication_date": "2026-01-29", "publication_date_precision": "date_only",
        "retrieved_at": retrieved, "source_period": "Q4 2025", "bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(), "figures": len(figures),
        "figure_manifest_sha256": hashlib.sha256(records.encode()).hexdigest(),
        "usage_boundary": "Official source retained privately; no relicensing claim or public raw-workbook redistribution.",
        "publication_evidence": "https://www.aemo.com.au/energy-systems/major-publications/quarterly-energy-dynamics-qed",
        "role": "P1 development context, not a new retrieval holdout or day-specific causal explanation",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
