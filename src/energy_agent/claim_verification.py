from __future__ import annotations

import re
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


@dataclass(frozen=True)
class LiteralConsistency:
    consistent: bool
    missing_literals: tuple[str, ...]
    direction_conflict: bool
    missing_entities: tuple[str, ...]
    missing_bindings: tuple[str, ...]


LITERAL = re.compile(r"(?<![a-z0-9])(?:q[1-4]|\$?\d[\d,]*(?:\.\d+)?%?)(?![a-z0-9])", re.IGNORECASE)
CLAUSE_SPLIT = re.compile(r"(?<!\d),(?!\d)|[.;•\n]+|\b(?:but|while|whereas)\b", re.IGNORECASE)
UP_CUE = re.compile(
    r"\b(?:up|above|higher|increase(?:d|s|ing)?|rose|rise|rises|rising|grew|grow(?:s|ing|th)?|uplift)\b"
)
DOWN_CUE = re.compile(
    r"\b(?:down|below|lower|decrease(?:d|s|ing)?|fell|fall(?:s|ing)?|reduce(?:d|s|ing)?|reduction|"
    r"decline(?:d|s|ing)?|drop(?:ped|s|ping)?)\b"
)
REGION_ENTITIES = {
    "new south wales": re.compile(r"\bnew south wales\b"),
    "south australia": re.compile(r"\bsouth australia(?:n)?\b"),
    "victoria": re.compile(r"\bvictori(?:a|an)\b"),
    "queensland": re.compile(r"\bqueensland\b"),
    "tasmania": re.compile(r"\btasmani(?:a|an)\b"),
}
TECHNOLOGY = re.compile(r"\b(batter(?:y|ies)|gas(?: facilities)?|coal|wind|solar)\b", re.IGNORECASE)
LEXICAL_TOKEN = re.compile(r"[a-z]+")
LEXICAL_STOP = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "while", "with"}
)


def _normalise_literal(value: str) -> str:
    return value.lower().replace("$", "").replace(",", "")


def _direction_groups(text: str) -> set[str]:
    lowered = text.lower()
    groups: set[str] = set()
    if UP_CUE.search(lowered):
        groups.add("up")
    if DOWN_CUE.search(lowered):
        groups.add("down")
    return groups


def _entity_number_bindings(text: str) -> set[str]:
    bindings: set[str] = set()
    for match in TECHNOLOGY.finditer(text):
        suffix = text[match.end() : match.end() + 100]
        number = LITERAL.search(suffix)
        if number is None:
            continue
        entity = match.group(1).lower()
        entity = "battery" if entity.startswith("batter") else "gas" if entity.startswith("gas") else entity
        bindings.add(f"{entity}:{_normalise_literal(number.group(0))}")
    return bindings


def _region_number_bindings(text: str) -> set[str]:
    lowered = text.lower()
    bindings: set[str] = set()
    for entity, pattern in REGION_ENTITIES.items():
        for match in pattern.finditer(lowered):
            number = LITERAL.search(lowered[match.end() : match.end() + 300])
            if number is not None:
                bindings.add(f"{entity}:{_normalise_literal(number.group(0))}")
    return bindings


def literal_consistency(document: str, claim: str) -> LiteralConsistency:
    """Conservative literal/relationship check used only as a verifier cascade.

    It does not claim semantic entailment. It checks that claim literals occur in
    the passage, that direction cues agree in the clause with the strongest
    numeric overlap, and that repeated technology-number bindings are preserved.
    """
    document_lower = document.lower()
    claim_literals = tuple(dict.fromkeys(_normalise_literal(value) for value in LITERAL.findall(claim)))
    document_literals = {_normalise_literal(value) for value in LITERAL.findall(document)}
    missing_literals = tuple(value for value in claim_literals if value not in document_literals)

    direction_conflict = False
    document_clauses = [clause for clause in CLAUSE_SPLIT.split(document_lower) if clause.strip()]
    for claim_clause in (clause for clause in CLAUSE_SPLIT.split(claim.lower()) if clause.strip()):
        claim_directions = _direction_groups(claim_clause)
        clause_literals = tuple(_normalise_literal(value) for value in LITERAL.findall(claim_clause))
        if not claim_directions or not clause_literals:
            continue
        claim_words = {word for word in LEXICAL_TOKEN.findall(claim_clause) if word not in LEXICAL_STOP}
        scored = [
            (
                sum(value in {_normalise_literal(item) for item in LITERAL.findall(clause)} for value in clause_literals),
                len(claim_words & set(LEXICAL_TOKEN.findall(clause))),
                clause,
            )
            for clause in document_clauses
        ]
        best_key = max(((numeric, lexical) for numeric, lexical, _ in scored), default=(0, 0))
        best_directions = set().union(
            *(
                _direction_groups(clause)
                for numeric, lexical, clause in scored
                if (numeric, lexical) == best_key and numeric > 0
            )
        )
        if best_directions and claim_directions.isdisjoint(best_directions):
            direction_conflict = True
            break

    claim_entities = tuple(entity for entity, pattern in REGION_ENTITIES.items() if pattern.search(claim.lower()))
    missing_entities = tuple(
        entity for entity in claim_entities if not REGION_ENTITIES[entity].search(document_lower)
    )

    claim_bindings = _entity_number_bindings(claim)
    document_bindings = _entity_number_bindings(document)
    binding_entities = {binding.split(":", maxsplit=1)[0] for binding in claim_bindings}
    missing_bindings = (
        tuple(sorted(claim_bindings - document_bindings))
        if {"battery", "gas"}.issubset(binding_entities)
        else ()
    )
    claim_region_bindings = _region_number_bindings(claim)
    document_region_bindings = _region_number_bindings(document)
    missing_bindings += tuple(sorted(claim_region_bindings - document_region_bindings))
    return LiteralConsistency(
        consistent=not (missing_literals or direction_conflict or missing_entities or missing_bindings),
        missing_literals=missing_literals,
        direction_conflict=direction_conflict,
        missing_entities=missing_entities,
        missing_bindings=missing_bindings,
    )


def selective_cascade_metrics(
    labels: list[int], probabilities: list[float], consistencies: list[bool], *, threshold: float = 0.5
) -> dict[str, float]:
    """Metrics for support / unsupported / abstain decisions at an explicit threshold."""
    if not labels or not (len(labels) == len(probabilities) == len(consistencies)):
        raise ValueError("labels, probabilities and consistencies must be non-empty and equal length")
    decisions = [
        "unsupported" if not consistent else "supported" if probability > threshold else "abstain"
        for probability, consistent in zip(probabilities, consistencies, strict=True)
    ]
    positives = sum(labels)
    negatives = len(labels) - positives
    supported_true_positives = sum(
        label == 1 and decision == "supported" for label, decision in zip(labels, decisions, strict=True)
    )
    rejected_true_negatives = sum(
        label == 0 and decision == "unsupported" for label, decision in zip(labels, decisions, strict=True)
    )
    handled = sum(decision != "abstain" for decision in decisions)
    handled_correct = sum(
        (label == 1 and decision == "supported") or (label == 0 and decision == "unsupported")
        for label, decision in zip(labels, decisions, strict=True)
    )
    support_recall = supported_true_positives / positives if positives else 0.0
    rejection_recall = rejected_true_negatives / negatives if negatives else 0.0
    return {
        "balanced_accuracy_with_abstention_as_miss": (support_recall + rejection_recall) / 2,
        "support_recall": support_recall,
        "counterfactual_rejection_recall": rejection_recall,
        "coverage": handled / len(labels),
        "selective_accuracy": handled_correct / handled if handled else 0.0,
        "abstention_rate": 1 - handled / len(labels),
    }


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
