from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.multimodal import load_page_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batched(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Qwen3-VL page/query embedding batch.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--qwen-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError("batch size must be between 1 and 16")

    import torch
    from PIL import Image

    sys.path.insert(0, str(args.qwen_repo.resolve()))
    module = importlib.import_module("src.models.qwen3_vl_embedding")
    embedder_class = module.Qwen3VLEmbedder
    started = time.perf_counter()
    embedder = embedder_class(
        model_name_or_path=str(args.model),
        max_length=8192,
        max_frames=1,
        fps=1,
        torch_dtype=torch.bfloat16,
    )
    pages = load_page_records(args.records)
    benchmark = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line]
    if args.limit_pages is not None:
        if args.limit_pages < 1:
            raise ValueError("limit-pages must be positive")
        pages = pages[: args.limit_pages]
    if args.limit_queries is not None:
        if args.limit_queries < 1:
            raise ValueError("limit-queries must be positive")
        benchmark = benchmark[: args.limit_queries]
    available_pages = {page.page_id for page in pages}
    missing_labels = sorted(
        {
            str(page_id)
            for row in benchmark
            for page_id in row["expected_page_ids"]
            if str(page_id) not in available_pages
        }
    )
    if missing_labels:
        raise ValueError(f"benchmark labels are absent from selected pages: {missing_labels}")
    page_inputs: list[dict[str, Any]] = []
    for page in pages:
        image_path = args.image_dir / f"{page.asset_id}.png"
        if sha256(image_path) != page.asset_sha256:
            raise ValueError(f"page image hash mismatch: {page.asset_id}")
        page_inputs.append({"image": Image.open(image_path).convert("RGB")})
    query_inputs = [
        {
            "text": str(row["query"]),
            "instruction": "Retrieve the official energy-market report page that answers this question.",
        }
        for row in benchmark
    ]

    def encode(inputs: list[dict[str, Any]]) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for batch in batched(inputs, args.batch_size):
                outputs.append(embedder.process(batch).detach().float().cpu().numpy())
        array: np.ndarray = np.concatenate(outputs, axis=0)
        return array

    page_embeddings = encode(page_inputs)
    query_embeddings = encode(query_inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    page_path = args.output / "page_embeddings.npy"
    query_path = args.output / "query_embeddings.npy"
    np.save(page_path, page_embeddings, allow_pickle=False)
    np.save(query_path, query_embeddings, allow_pickle=False)
    ids = {
        "page_ids": [page.page_id for page in pages],
        "query_ids": [str(row["query_id"]) for row in benchmark],
    }
    ids_path = args.output / "ids.json"
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "code_revision": args.code_revision,
        "embedding_family": "Qwen3-VL dual-tower single-vector; consumed by the generic MaxSim interface as one token",
        "pages": len(pages),
        "queries": len(benchmark),
        "embedding_dimension": int(page_embeddings.shape[1]),
        "records_sha256": sha256(args.records),
        "benchmark_sha256": sha256(args.benchmark),
        "page_embeddings_sha256": sha256(page_path),
        "query_embeddings_sha256": sha256(query_path),
        "ids_sha256": sha256(ids_path),
        "runtime_seconds": time.perf_counter() - started,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
