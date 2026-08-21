from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

MINICHECK_NEGATIVE_TOKEN_ID = 3
MINICHECK_POSITIVE_TOKEN_ID = 209


@dataclass(frozen=True)
class ClaimSupportScore:
    support_probability: float
    predicted_supported: bool


class MiniCheckFlanVerifier:
    """Pinned MiniCheck Flan-T5 adapter for sentence-level grounded fact checking.

    The adapter implements the public MiniCheck first-token scoring contract while
    keeping the dependency optional. It never participates in tool planning.
    """

    def __init__(
        self,
        *,
        model_id: str = "lytang/MiniCheck-Flan-T5-Large",
        revision: str = "a496016e7b493686ed6e1c52250b9b9d39b0dcb2",
        device: str = "cpu",
        max_length: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("Install the factcheck optional dependencies") from exc
        self._torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.model_id = model_id
        self.revision = revision
        self.negative_token_id = MINICHECK_NEGATIVE_TOKEN_ID
        self.positive_token_id = MINICHECK_POSITIVE_TOKEN_ID
        vocabulary_size = int(self.model.config.vocab_size)
        if self.positive_token_id >= vocabulary_size:
            raise RuntimeError("MiniCheck label token IDs exceed the pinned model vocabulary")

    def score(self, documents: list[str], claims: list[str], *, batch_size: int = 8) -> list[ClaimSupportScore]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results: list[ClaimSupportScore] = []
        separator = str(self.tokenizer.eos_token)
        for start in range(0, len(documents), batch_size):
            texts = [
                f"predict: {document}{separator}{claim}"
                for document, claim in zip(
                    documents[start : start + batch_size], claims[start : start + batch_size], strict=True
                )
            ]
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            decoder_start = int(self.model.config.decoder_start_token_id)
            decoder_input_ids = self._torch.full(
                (len(texts), 1), decoder_start, dtype=self._torch.long, device=self.device
            )
            with self._torch.inference_mode():
                logits = self.model(**encoded, decoder_input_ids=decoder_input_ids).logits[:, 0, :]
                label_logits = logits[:, [self.negative_token_id, self.positive_token_id]]
                probabilities = self._torch.softmax(label_logits, dim=-1)[:, 1].detach().cpu().numpy()
            results.extend(
                ClaimSupportScore(float(probability), bool(probability > 0.5)) for probability in probabilities
            )
        return results


class MiniCheckDebertaVerifier:
    """Pinned official MiniCheck DeBERTa adapter for a source-disjoint transport gate.

    This is the second sub-1B verifier exposed by the official MiniCheck project.
    Its checkpoint, revision, 2,048-token contract and 0.5 threshold are frozen
    before the Q2 2025 source-disjoint holdout is scored.
    """

    def __init__(
        self,
        *,
        model_id: str = "lytang/MiniCheck-DeBERTa-v3-Large",
        revision: str = "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
        device: str = "cpu",
        max_length: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("Install the factcheck optional dependencies") from exc
        self._torch = torch
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            num_labels=2,
            finetuning_task="text-classification",
        )
        config.problem_type = "single_label_classification"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            config=config,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.model_id = model_id
        self.revision = revision

    def score(self, documents: list[str], claims: list[str], *, batch_size: int = 8) -> list[ClaimSupportScore]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results: list[ClaimSupportScore] = []
        separator = str(self.tokenizer.eos_token)
        for start in range(0, len(documents), batch_size):
            texts = [
                f"{document}{separator}{claim}"
                for document, claim in zip(
                    documents[start : start + batch_size], claims[start : start + batch_size], strict=True
                )
            ]
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            results.extend(
                ClaimSupportScore(float(probability), bool(probability > 0.5)) for probability in probabilities
            )
        return results


def binary_metrics(labels: list[int], probabilities: list[float], *, bins: int = 5) -> dict[str, float]:
    """Dependency-light classification, ranking and calibration diagnostics."""
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and equal length")
    if set(labels) - {0, 1}:
        raise ValueError("labels must be binary")
    predicted = [int(value > 0.5) for value in probabilities]
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positive = sum(label == prediction == 1 for label, prediction in zip(labels, predicted, strict=True))
    true_negative = sum(label == prediction == 0 for label, prediction in zip(labels, predicted, strict=True))
    support_recall = true_positive / positives if positives else 0.0
    rejection_recall = true_negative / negatives if negatives else 0.0
    order = np.argsort(np.asarray(probabilities))
    ranks = np.empty(len(labels), dtype=float)
    ranks[order] = np.arange(1, len(labels) + 1)
    positive_rank_sum = float(ranks[np.asarray(labels) == 1].sum())
    auroc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    probabilities_array = np.asarray(probabilities, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    brier = float(np.mean((probabilities_array - labels_array) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in pairwise(edges):
        selected = (probabilities_array >= lower) & (
            probabilities_array <= upper if upper == 1.0 else probabilities_array < upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(labels_array[selected].mean()) - float(probabilities_array[selected].mean())
            )
    return {
        "accuracy": sum(label == prediction for label, prediction in zip(labels, predicted, strict=True)) / len(labels),
        "balanced_accuracy": (support_recall + rejection_recall) / 2,
        "support_recall": support_recall,
        "counterfactual_rejection_recall": rejection_recall,
        "auroc": auroc,
        "brier_score": brier,
        "ece_5_bin": ece,
    }
