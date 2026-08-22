from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from energy_agent.multimodal import PageRecord, classify_page_modality


def numeric_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    return sum(character.isdigit() for character in compact) / max(1, len(compact))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PDF into provenance-safe page evidence artifacts.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--retrieved-at", default=None)
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()
    if args.dpi < 72 or args.dpi > 300:
        raise ValueError("dpi must be between 72 and 300")

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Install the multimodal extra: pip install -e '.[multimodal]'") from exc

    payload = args.pdf.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    retrieved_at = args.retrieved_at or datetime.now(UTC).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "pages"
    image_dir.mkdir(exist_ok=True)
    pages: list[PageRecord] = []
    diagnostics: list[dict[str, Any]] = []
    document: Any = pymupdf.open(stream=payload, filetype="pdf")  # type: ignore[no-untyped-call]
    zoom = args.dpi / 72
    for page_index in range(len(document)):
        page: Any = document[page_index]
        page_number = page_index + 1
        matrix: Any = pymupdf.Matrix(zoom, zoom)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")
        asset_sha256 = hashlib.sha256(image_bytes).hexdigest()
        asset_id = f"{args.source_id}-page-{page_number:04d}"
        image_path = image_dir / f"{asset_id}.png"
        image_path.write_bytes(image_bytes)
        text = " ".join(page.get_text("text").split())
        page_dict = page.get_text("dict")
        image_blocks = sum(1 for block in page_dict.get("blocks", []) if block.get("type") == 1)
        drawing_objects = len(page.get_drawings())
        try:
            table_count = len(page.find_tables().tables)
        except (AttributeError, TypeError):
            table_count = 0
        modality = classify_page_modality(
            image_blocks=image_blocks,
            drawing_objects=drawing_objects,
            table_count=table_count,
            numeric_ratio=numeric_ratio(text),
        )
        if len(text) < 50 and image_blocks > 0:
            modality = "page_image"
        pages.append(
            PageRecord(
                page_id=asset_id,
                source_id=args.source_id,
                title=args.title,
                text=text,
                url=args.url,
                published_at=args.published_at,
                retrieved_at=retrieved_at,
                source_sha256=source_sha256,
                page_number=page_number,
                modality=modality,
                asset_id=asset_id,
                asset_sha256=asset_sha256,
                width=pixmap.width,
                height=pixmap.height,
            )
        )
        diagnostics.append(
            {
                "page_id": asset_id,
                "page_number": page_number,
                "modality": modality,
                "image_blocks": image_blocks,
                "drawing_objects": drawing_objects,
                "table_count": table_count,
                "numeric_ratio": numeric_ratio(text),
                "text_characters": len(text),
            }
        )
    records_path = args.output / "page_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(page.__dict__, ensure_ascii=False) + "\n" for page in pages),
        encoding="utf-8",
    )
    diagnostics_path = args.output / "page_diagnostics.jsonl"
    diagnostics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in diagnostics),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_id": args.source_id,
        "source_sha256": source_sha256,
        "pages": len(pages),
        "dpi": args.dpi,
        "page_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "page_diagnostics_sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        "asset_bytes": sum((image_dir / f"{page.asset_id}.png").stat().st_size for page in pages),
        "modality_counts": {
            modality: sum(page.modality == modality for page in pages)
            for modality in ("text", "page_image", "chart", "table", "mixed")
        },
        "usage_boundary": "Rendered official-report pages remain private/local or on Spartan; only hashes and aggregate metrics may be published.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
