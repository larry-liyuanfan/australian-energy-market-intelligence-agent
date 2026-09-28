from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from energy_agent.evidence import HybridEvidenceIndex, OfficialChunk
from energy_agent.vidore import (
    ViDoReCorpus,
    adapt_vidore_rows,
    late_interaction_ranking,
    paired_bootstrap_ndcg_difference,
    promotion_decision,
    retrieval_metrics,
    single_vector_ranking,
    sliced_retrieval_metrics,
    weighted_rrf,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def timed_rankings(
    query_ids: list[str],
    search: Callable[[str], list[str]],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query_id in query_ids:
        started = time.perf_counter()
        run[query_id] = search(query_id)
        latencies.append((time.perf_counter() - started) * 1000)
    return run, {"p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95)}


def bm25_run(corpus: ViDoReCorpus) -> tuple[dict[str, list[str]], dict[str, float]]:
    fallback = "document image with no supplied OCR text"
    chunks = [
        OfficialChunk(
            chunk_id=document.document_id,
            source_id=corpus.task_id,
            title=document.image_filename,
            text=document.ocr_text or fallback,
            url="https://huggingface.co/datasets/vidore",
            published_at="2024-01-01T00:00:00+00:00",
            retrieved_at="2026-09-03T00:00:00+00:00",
            sha256="0" * 64,
        )
        for document in corpus.documents
    ]
    index = HybridEvidenceIndex(chunks)
    query_by_id = {query.query_id: query.text for query in corpus.queries}
    return timed_rankings(
        list(query_by_id),
        lambda query_id: [
            str(hit["chunk_id"]) for hit in index.search(query_by_id[query_id], top_k=len(chunks), mode="bm25")
        ],
    )


def encode_qwen_text(
    corpora: list[ViDoReCorpus], model_path: Path, batch_size: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    torch = importlib.import_module("torch")
    sentence_transformers = importlib.import_module("sentence_transformers")
    sentence_transformer = sentence_transformers.SentenceTransformer

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = sentence_transformer(
        str(model_path),
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda",
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    encoded = 0
    for corpus in corpora:
        documents = [document.ocr_text or "document image with no supplied OCR text" for document in corpus.documents]
        queries = [query.text for query in corpus.queries]
        document_vectors = np.asarray(
            model.encode(documents, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32,
        )
        query_vectors = np.asarray(
            model.encode(
                queries,
                batch_size=batch_size,
                normalize_embeddings=True,
                prompt="Instruct: Retrieve the document page that answers the query\nQuery:",
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        output[corpus.task_id] = {"documents": document_vectors, "queries": query_vectors}
        encoded += len(documents)
    elapsed = time.perf_counter() - started
    profile = {
        "index_seconds": elapsed,
        "index_documents_per_second": encoded / elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return output, profile


def encode_qwen_vl(
    corpora: list[ViDoReCorpus], qwen_repo: Path, model_path: Path, batch_size: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    torch = importlib.import_module("torch")

    sys.path.insert(0, str(qwen_repo.resolve()))
    embedder_class = importlib.import_module("src.models.qwen3_vl_embedding").Qwen3VLEmbedder
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embedder = embedder_class(
        model_name_or_path=str(model_path),
        max_length=8192,
        max_frames=1,
        fps=1,
        torch_dtype=torch.bfloat16,
    )

    def encode(inputs: list[dict[str, Any]]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start : start + batch_size]
                chunks.append(embedder.process(batch).detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    output: dict[str, dict[str, np.ndarray]] = {}
    encoded = 0
    for corpus in corpora:
        document_vectors = encode(
            [{"image": cast(Any, document.image).convert("RGB")} for document in corpus.documents]
        )
        query_vectors = encode(
            [
                {
                    "text": query.text,
                    "instruction": "Retrieve the document page that answers this question.",
                }
                for query in corpus.queries
            ]
        )
        output[corpus.task_id] = {"documents": document_vectors, "queries": query_vectors}
        encoded += len(corpus.documents)
    elapsed = time.perf_counter() - started
    profile = {
        "index_seconds": elapsed,
        "index_documents_per_second": encoded / elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return output, profile


def encode_colqwen(
    corpora: list[ViDoReCorpus], model_path: Path, batch_size: int
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    torch = importlib.import_module("torch")
    colpali_models = importlib.import_module("colpali_engine.models")
    colqwen_model = colpali_models.ColQwen2
    colqwen_processor = colpali_models.ColQwen2Processor

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = colqwen_model.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    processor = colqwen_processor.from_pretrained(str(model_path), local_files_only=True)

    def encode_images(images: list[object]) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = processor.process_images(images[start : start + batch_size]).to(model.device)
                embeddings = model(**batch)
                values.extend(item.detach().float().cpu().numpy() for item in embeddings)
        return values

    def encode_queries(queries: list[str]) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                batch = processor.process_queries(queries[start : start + batch_size]).to(model.device)
                embeddings = model(**batch)
                values.extend(item.detach().float().cpu().numpy() for item in embeddings)
        return values

    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    encoded = 0
    for corpus in corpora:
        document_values = encode_images([cast(Any, document.image).convert("RGB") for document in corpus.documents])
        query_values = encode_queries([query.text for query in corpus.queries])
        output[corpus.task_id] = {
            "documents": {
                document.document_id: document_values[index] for index, document in enumerate(corpus.documents)
            },
            "queries": {query.query_id: query_values[index] for index, query in enumerate(corpus.queries)},
        }
        encoded += len(corpus.documents)
    elapsed = time.perf_counter() - started
    profile = {
        "index_seconds": elapsed,
        "index_documents_per_second": encoded / elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return output, profile


def save_vectors(output: Path, task_id: str, name: str, values: dict[str, np.ndarray] | np.ndarray) -> int:
    path = output / "cache" / task_id
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{name}.npz"
    if isinstance(values, dict):
        save_compressed = cast(Any, np.savez_compressed)
        save_compressed(target, **{f"v{index}": value for index, value in enumerate(values.values())})
    else:
        np.savez_compressed(target, values=values)
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen public ViDoRe retrieval comparison")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-repo", type=Path, required=True)
    parser.add_argument("--qwen-vl-model", type=Path, required=True)
    parser.add_argument("--qwen-text-model", type=Path, required=True)
    parser.add_argument("--colqwen-model", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    load_dataset = importlib.import_module("datasets").load_dataset

    corpora: list[ViDoReCorpus] = []
    for task in config["tasks"]:
        dataset = load_dataset(task["ocr_dataset_id"], revision=task["ocr_dataset_revision"], split="test")
        if args.limit:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        corpora.append(adapt_vidore_rows(task["task_id"], task["ocr_dataset_revision"], dataset))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "dataset_manifest.json").write_text(
        json.dumps(
            {
                corpus.task_id: {
                    "dataset_revision": corpus.dataset_revision,
                    "documents": len(corpus.documents),
                    "queries": len(corpus.queries),
                    "data_sha256": corpus.data_sha256,
                }
                for corpus in corpora
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    qwen_text, text_profile = encode_qwen_text(corpora, args.qwen_text_model, config["batch_sizes"]["text"])
    qwen_vl, qwen_vl_profile = encode_qwen_vl(
        corpora, args.qwen_repo, args.qwen_vl_model, config["batch_sizes"]["qwen_vl"]
    )
    colqwen, colqwen_profile = encode_colqwen(corpora, args.colqwen_model, config["batch_sizes"]["colqwen"])
    task_results: dict[str, dict[str, Any]] = {}
    predictions = args.output / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for corpus in corpora:
            document_ids = [item.document_id for item in corpus.documents]
            query_ids = [item.query_id for item in corpus.queries]
            bm25, bm25_latency = bm25_run(corpus)
            text_dense_raw = single_vector_ranking(
                qwen_text[corpus.task_id]["queries"],
                qwen_text[corpus.task_id]["documents"],
                query_ids,
                document_ids,
            )
            text_dense, text_dense_latency = timed_rankings(query_ids, text_dense_raw.__getitem__)
            text_baseline = weighted_rrf({"bm25": bm25, "qwen_text": text_dense}, config["fusion_weights"]["ocr_text"])
            qwen_vl_raw = single_vector_ranking(
                qwen_vl[corpus.task_id]["queries"],
                qwen_vl[corpus.task_id]["documents"],
                query_ids,
                document_ids,
            )
            qwen_visual, qwen_latency = timed_rankings(query_ids, qwen_vl_raw.__getitem__)
            colqwen_raw = late_interaction_ranking(
                colqwen[corpus.task_id]["queries"], colqwen[corpus.task_id]["documents"]
            )
            colqwen_run, colqwen_latency = timed_rankings(query_ids, colqwen_raw.__getitem__)
            fusion = weighted_rrf(
                {
                    "ocr_text": text_baseline,
                    "qwen_vl": qwen_visual,
                    "colqwen": colqwen_run,
                },
                config["fusion_weights"]["text_visual"],
            )
            runs = {
                "ocr_bm25": bm25,
                "qwen_text": text_dense,
                "ocr_text": text_baseline,
                "qwen_vl": qwen_visual,
                "colqwen": colqwen_run,
                "fusion": fusion,
            }
            index_bytes = {
                "qwen_text": save_vectors(
                    args.output, corpus.task_id, "qwen_text_documents", qwen_text[corpus.task_id]["documents"]
                ),
                "qwen_vl": save_vectors(
                    args.output, corpus.task_id, "qwen_vl_documents", qwen_vl[corpus.task_id]["documents"]
                ),
                "colqwen": save_vectors(
                    args.output, corpus.task_id, "colqwen_documents", colqwen[corpus.task_id]["documents"]
                ),
            }
            result = {
                "metrics": {name: retrieval_metrics(corpus, run) for name, run in runs.items()},
                "slices": {name: sliced_retrieval_metrics(corpus, run) for name, run in runs.items()},
                "query_latency": {
                    "ocr_bm25": bm25_latency,
                    "qwen_text": text_dense_latency,
                    "qwen_vl": qwen_latency,
                    "colqwen": colqwen_latency,
                },
                "index_bytes": index_bytes,
                "significance_vs_ocr_text": {
                    name: paired_bootstrap_ndcg_difference(corpus, run, text_baseline)
                    for name, run in {"qwen_vl": qwen_visual, "colqwen": colqwen_run, "fusion": fusion}.items()
                },
            }
            task_results[corpus.task_id] = result
            query_by_id = {query.query_id: query for query in corpus.queries}
            for query_id in query_ids:
                handle.write(
                    json.dumps(
                        {
                            "task_id": corpus.task_id,
                            "query_id": query_id,
                            "query": query_by_id[query_id].text,
                            "relevant_document_ids": query_by_id[query_id].relevant_document_ids,
                            "rankings": {name: run[query_id][:10] for name, run in runs.items()},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    metrics = {
        "schema_version": "vidore-retrieval-evaluation-v2",
        "tasks": task_results,
        "promotion": {
            candidate: promotion_decision(task_results, candidate) for candidate in ("qwen_vl", "colqwen", "fusion")
        },
        "resource_profile": {
            "qwen_text": text_profile,
            "qwen_vl": qwen_vl_profile,
            "colqwen": colqwen_profile,
        },
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": "vidore-retrieval-run-manifest-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "config_sha256": sha256(args.config),
        "dataset_manifest_sha256": sha256(args.output / "dataset_manifest.json"),
        "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(predictions),
        "models": config["models"],
        "environment": {
            key: __import__("os").environ.get(key)
            for key in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
            if __import__("os").environ.get(key)
        },
        "boundaries": [
            "Public document-page retrieval benchmark; it does not measure answer correctness.",
            "Fusion weights were frozen from the existing AEMO development task before ViDoRe execution.",
            "No ViDoRe query or label participates in weight selection.",
        ],
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics["promotion"], indent=2))


if __name__ == "__main__":
    main()
