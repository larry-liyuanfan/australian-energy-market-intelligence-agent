from typing import Any

from energy_agent.evidence import ElasticsearchHybridEvidenceIndex, HybridEvidenceIndex, OfficialChunk


class FakeIndices:
    def __init__(self) -> None:
        self.created: set[str] = set()
        self.aliases: dict[str, dict[str, object]] = {}

    def exists(self, index: str) -> bool:
        return index in self.created

    def create(self, index: str, settings: object, mappings: object) -> None:
        del settings, mappings
        self.created.add(index)

    def get_alias(self, name: str, **_: object) -> dict[str, dict[str, object]]:
        return self.aliases.get(name, {})

    def exists_alias(self, name: str) -> bool:
        return name in self.aliases

    def update_aliases(self, actions: list[dict[str, dict[str, str]]]) -> None:
        for action in actions:
            if add := action.get("add"):
                self.aliases.setdefault(add["alias"], {})[add["index"]] = {}
            if remove := action.get("remove"):
                self.aliases.setdefault(remove["alias"], {}).pop(remove["index"], None)


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.documents: dict[str, dict[str, Any]] = {}

    def bulk(self, operations: list[dict[str, Any]], refresh: bool) -> dict[str, bool]:
        assert refresh
        for position in range(0, len(operations), 2):
            self.documents[str(operations[position]["index"]["_id"])] = operations[position + 1]
        return {"errors": False}

    def search(self, **_: object) -> dict[str, object]:
        relevant = self.documents["b"]
        return {"hits": {"hits": [{"_score": 12.0, "_source": {"chunk_id": relevant["chunk_id"]}}]}}

    def count(self, **_: object) -> dict[str, int]:
        return {"count": len(self.documents)}


def test_hybrid_evidence_ranks_relevant_official_chunk() -> None:
    common = {"published_at": "2026-01-01", "retrieved_at": "2026-08-20", "sha256": "0" * 64}
    documents = [
        OfficialChunk(
            "a",
            "qed",
            "Battery price setting",
            "Grid batteries charged at noon and discharged at the evening peak.",
            "https://aemo.com.au/a",
            **common,
        ),
        OfficialChunk(
            "b",
            "event",
            "Network constraint event",
            "A transmission outage restricted imports into South Australia.",
            "https://aer.gov.au/b",
            **common,
        ),
    ]
    index = HybridEvidenceIndex(documents, dense_dimensions=2)
    assert index.search("South Australia transmission imports", top_k=1)[0]["source_id"] == "event"


def test_hybrid_rerank_preserves_exact_numeric_passage_evidence() -> None:
    common = {"published_at": "2026-01-01", "retrieved_at": "2026-08-20", "sha256": "0" * 64}
    documents = [
        OfficialChunk(
            "summary",
            "qed",
            "Quarterly Energy Dynamics Q2 2026",
            "Wholesale spot prices averaged $74/MWh, down 47% year on year.",
            "https://aemo.com.au/summary",
            **common,
        ),
        OfficialChunk(
            "adjacent",
            "qed",
            "Quarterly Energy Dynamics Q2 2026",
            "Wholesale prices and battery output changed during the quarter.",
            "https://aemo.com.au/adjacent",
            **common,
        ),
    ]
    index = HybridEvidenceIndex(documents, dense_dimensions=2)
    hit = index.search("Q2 2026 wholesale spot prices $74 MWh down 47%", top_k=1)[0]
    assert hit["chunk_id"] == "summary"
    assert hit["numeric_coverage"] > 0


def test_elasticsearch_bm25_is_indexed_and_fused_without_user_dsl() -> None:
    common = {"published_at": "2026-01-01", "retrieved_at": "2026-08-20", "sha256": "0" * 64}
    documents = [
        OfficialChunk("a", "qed", "Battery", "Evening peak", "https://aemo.com.au/a", **common),
        OfficialChunk("b", "event", "Constraint", "Imports restricted", "https://aer.gov.au/b", **common),
    ]
    client = FakeElasticsearch()
    index = ElasticsearchHybridEvidenceIndex(HybridEvidenceIndex(documents, dense_dimensions=2), client)
    index.ensure_index()
    index.ensure_index()
    result = index.search("transmission constraint", top_k=1)
    assert len(client.documents) == 2
    assert index.indexed_documents == 2
    assert index.index_name in client.indices.aliases[index.alias]
    assert result[0]["chunk_id"] == "b"


def test_external_bm25_results_are_source_diversified() -> None:
    common = {"published_at": "2026-01-01", "retrieved_at": "2026-08-20", "sha256": "0" * 64}
    documents = [
        OfficialChunk(str(index), "dominant", "Dominant", "price", f"https://a/{index}", **common)
        for index in range(4)
    ] + [OfficialChunk("other", "other", "Other", "constraint", "https://b", **common)]
    index = HybridEvidenceIndex(documents, dense_dimensions=2)
    scores = {str(position): 10.0 - position for position in range(4)} | {"other": 1.0}
    hits = index.search("price", top_k=3, mode="bm25", lexical_scores=scores, max_per_source=2)
    assert [hit["source_id"] for hit in hits] == ["dominant", "dominant", "other"]
