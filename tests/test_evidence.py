from energy_agent.evidence import HybridEvidenceIndex, OfficialChunk


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
