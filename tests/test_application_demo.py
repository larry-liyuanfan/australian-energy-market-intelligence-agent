from __future__ import annotations

import json
from pathlib import Path

from scripts.render_agent_application import answer_blocks


def test_new_answer_preserves_business_units_and_recording_boundaries() -> None:
    root = Path(__file__).parents[1]
    compact = json.loads((root / "docs/demos/model-replay-20260927/recorded_run.json").read_text())
    blocks = answer_blocks(compact)
    text = " ".join(b["text"] for b in blocks)
    assert "253.38" in text and "477.79" in text and "75.89" in text
    assert "144/288" in text and "-906.68" in text
    assert "after this market day" in text and "does not establish" in text
    assert "not an observed asset cost" in text and "investment return" in text
    available = {c["evidence_id"] for c in compact["citations"]} | {"aemo-dispatch-12m", "calculation:schedule"}
    assert all(set(block["evidence_ids"]) <= available for block in blocks)
