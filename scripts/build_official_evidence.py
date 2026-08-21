from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pypdf import PdfReader

from energy_agent.evidence import HybridEvidenceIndex, OfficialChunk

SOURCES = [
    (
        "aemo-qed-q2-2026",
        "Quarterly Energy Dynamics Q2 2026",
        "2026-07-28",
        "https://www.aemo.com.au/-/media/files/major-publications/qed/2026/qed-q2-2026.pdf",
    ),
    (
        "aemo-qed-q1-2026",
        "Quarterly Energy Dynamics Q1 2026",
        "2026-04-30",
        "https://www.aemo.com.au/-/media/files/major-publications/qed/2026/qed-q1-2026.pdf",
    ),
    (
        "aemo-qed-q4-2025",
        "Quarterly Energy Dynamics Q4 2025",
        "2026-01-29",
        "https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q4-2025.pdf",
    ),
    (
        "aemo-qed-q3-2025",
        "Quarterly Energy Dynamics Q3 2025",
        "2025-10-30",
        "https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q3-2025.pdf",
    ),
    (
        "aemo-qed-q2-2025",
        "Quarterly Energy Dynamics Q2 2025",
        "2025-07-31",
        "https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q2-2025.pdf",
    ),
    (
        "aer-significant-q1-2026",
        "Significant electricity prices January to March 2026",
        "2026-05-28",
        "https://www.aer.gov.au/system/files/2026-05/Significant%20electricity%20prices%20for%20January%20to%20March%202026.pdf",
    ),
]


def chunks(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    clean = " ".join(text.split())
    return [
        clean[start : start + size]
        for start in range(0, len(clean), size - overlap)
        if len(clean[start : start + size]) >= 200
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    documents: list[OfficialChunk] = []
    provenance: list[dict[str, Any]] = []
    for source_id, title, published_at, url in SOURCES:
        request = Request(url, headers={"User-Agent": "energy-agent-research/0.1"})
        with urlopen(request, timeout=120) as response:  # fixed official origins
            payload = response.read()
        retrieved_at = datetime.now(UTC).isoformat()
        digest = hashlib.sha256(payload).hexdigest()
        reader = PdfReader(io.BytesIO(payload))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        source_chunks = chunks(text)
        for chunk_index, chunk in enumerate(source_chunks):
            documents.append(
                OfficialChunk(
                    f"{source_id}-{chunk_index:04d}",
                    source_id,
                    title,
                    chunk,
                    url,
                    published_at,
                    retrieved_at,
                    digest,
                )
            )
        provenance.append(
            {
                "source_id": source_id,
                "title": title,
                "url": url,
                "published_at": published_at,
                "retrieved_at": retrieved_at,
                "sha256": digest,
                "bytes": len(payload),
                "pages": len(reader.pages),
                "chunks": len(source_chunks),
                "usage_boundary": "Official source copyright and terms retained; extracted text used for research indexing only.",
            }
        )
    docs_path = args.output / "evidence_documents.jsonl"
    docs_path.write_text("".join(json.dumps(doc.__dict__) + "\n" for doc in documents), encoding="utf-8")
    provenance_path = args.output / "source_provenance.jsonl"
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
    evidence_index = HybridEvidenceIndex(documents)
    smoke_queries = {
        "21 22 June 2026 cold still transmission constraints restricted imports South Australia": "aemo-qed-q2-2026",
        "significant prices Tasmania January 2026": "aer-significant-q1-2026",
        "battery price setting dispatch intervals": "aemo-qed-q2-2026",
    }
    outcomes: list[dict[str, Any]] = []
    for query, expected in smoke_queries.items():
        hits = evidence_index.search(query, top_k=5)
        outcomes.append(
            {
                "query": query,
                "expected_source": expected,
                "retrieved_sources": [hit["source_id"] for hit in hits],
                "reciprocal_rank": next((1 / hit["rank"] for hit in hits if hit["source_id"] == expected), 0),
            }
        )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "sources": len(provenance),
        "chunks": len(documents),
        "documents_sha256": hashlib.sha256(docs_path.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "retrieval": "BM25 + 64-D TF-IDF/TruncatedSVD LSA + RRF + deterministic feature rerank",
        "smoke_mrr": sum(float(row["reciprocal_rank"]) for row in outcomes) / len(outcomes),
        "smoke_queries": outcomes,
        "scope_boundary": "Smoke queries validate plumbing only; not the 100-task Agent evaluation.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
