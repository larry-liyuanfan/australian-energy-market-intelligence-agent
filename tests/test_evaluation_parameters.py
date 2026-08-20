from __future__ import annotations

import pytest

from energy_agent.evaluation import parse_degradation_costs


@pytest.mark.parametrize("separator", [",", ":", ";"])
def test_degradation_costs_accept_cli_and_slurm_safe_separators(separator: str) -> None:
    assert parse_degradation_costs(separator.join(("0", "25", "100"))) == (0.0, 25.0, 100.0)


def test_degradation_costs_reject_empty_or_negative_values() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_degradation_costs("")
    with pytest.raises(ValueError, match="non-negative"):
        parse_degradation_costs("0:-1")
