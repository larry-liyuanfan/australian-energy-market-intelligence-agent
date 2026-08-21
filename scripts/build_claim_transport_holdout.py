from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from build_official_evidence import chunks
from pypdf import PdfReader

from energy_agent.evidence import OfficialChunk

SOURCE_ID = "aemo-qed-q1-2025"
TITLE = "Quarterly Energy Dynamics Q1 2025"
PUBLISHED_AT = "2025-05-07"
URL = "https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q1-2025.pdf"
EXPECTED_SHA256 = "8ad4aa67ac5c5d72281371e5c658eb6866892f59d1177fa0fbf1596a3efc3534"
FROZEN_CASCADE_SHA = "41d0489981291f8b3c6ceb3727da5d6ab7bb9ee5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = args.pdf.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Q1 2025 PDF hash mismatch: {digest}")
    reader = PdfReader(args.pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    source_chunks = chunks(text)
    retrieved_at = datetime.now(UTC).isoformat()
    documents = [
        OfficialChunk(
            chunk_id=f"{SOURCE_ID}-{index:04d}",
            source_id=SOURCE_ID,
            title=TITLE,
            text=chunk,
            url=URL,
            published_at=PUBLISHED_AT,
            retrieved_at=retrieved_at,
            sha256=digest,
        )
        for index, chunk in enumerate(source_chunks)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output / "evidence_documents.jsonl"
    evidence_path.write_text("".join(json.dumps(row.__dict__) + "\n" for row in documents), encoding="utf-8")
    manifest = {
        "source_id": SOURCE_ID,
        "title": TITLE,
        "url": URL,
        "published_at": PUBLISHED_AT,
        "retrieved_at": retrieved_at,
        "source_sha256": digest,
        "bytes": len(payload),
        "pages": len(reader.pages),
        "chunks": len(documents),
        "chunk_size": 1400,
        "chunk_overlap": 200,
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "frozen_cascade_sha": FROZEN_CASCADE_SHA,
        "usage_boundary": "Official AEMO source; extracted text retained as a private research artifact.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
