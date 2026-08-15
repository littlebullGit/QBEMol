"""Validate the normalized result data distributed with QBEMol."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "data" / "results"


def load_result(filename: str) -> dict[str, Any]:
    """Load one publication result and require a JSON object."""

    payload = json.loads((RESULTS_DIRECTORY / filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {filename}")
    return payload


def iter_numbers(value: Any) -> Iterator[float]:
    """Yield every numeric value nested within a JSON-compatible value."""

    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from iter_numbers(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            yield from iter_numbers(nested_value)


@pytest.mark.parametrize("filename", ("h4.json", "f2.json", "h6.json", "n_butane.json"))
def test_result_files_are_finite_and_provenance_bound(filename: str) -> None:
    """Require each public result to use the schema and contain finite data."""

    payload = load_result(filename)

    assert payload["schema"] == "qbemol.paper-results.v1"
    assert payload["provenance"]["source_artifacts"]
    assert all(
        artifact["sha256"].startswith("sha256:")
        for artifact in payload["provenance"]["source_artifacts"]
    )
    assert all(math.isfinite(number) for number in iter_numbers(payload))


def test_h4_and_f2_match_the_accepted_endpoints() -> None:
    """Protect the accepted local H4 control and F2 rescue endpoints."""

    h4 = load_result("h4.json")["results"]
    f2 = load_result("f2.json")["results"]

    assert h4["fci_be"]["energy_ha"] == pytest.approx(-2.1663874444936675)
    assert h4["greedy_adapt_vqe_be"]["error_mha_vs_fci_be"] == pytest.approx(
        0.24257732255472675
    )
    assert f2["greedy_adapt_vqe_be"]["error_mha_vs_fci_be"] == pytest.approx(
        71.37077679780646
    )
    assert f2["lookahead_adapt_vqe_be"]["error_mha_vs_fci_be"] == pytest.approx(
        0.0035463336303109827
    )
    event = f2["first_decisive_selector_event"]
    assert event["selected_operator_index"] == 22
    assert event["selected_raw_gradient_rank"] == 5


def test_h6_preserves_the_mixed_topology_classification() -> None:
    """Prevent the H6 mixed-topology run from being presented as fixed topology."""

    payload = load_result("h6.json")

    assert payload["validation"]["status"] == (
        "invalid_for_fixed_four_fragment_distance_comparison"
    )
    assert {
        distance: geometry["resolved_fragment_count"]
        for distance, geometry in payload["geometries"].items()
    } == {"1.0": 4, "1.5": 6, "2.0": 6}


def test_n_butane_errors_match_the_exported_energies() -> None:
    """Check n-butane error signs and accepted lookahead comparisons."""

    geometries = load_result("n_butane.json")["geometries"]
    for geometry in geometries.values():
        reference = geometry["fci_be_energy_ha"]
        assert geometry["error_mha_vs_fci_be"]["greedy"] == pytest.approx(
            1000.0 * (geometry["greedy_energy_ha"] - reference)
        )
        assert geometry["error_mha_vs_fci_be"]["lookahead"] == pytest.approx(
            1000.0 * (geometry["lookahead_energy_ha"] - reference)
        )

    assert geometries["1.30"]["lookahead_error_change_mha_vs_greedy"] > 0.0
    assert geometries["1.54"]["lookahead_error_change_mha_vs_greedy"] < 0.0
    assert geometries["2.10"]["lookahead_error_change_mha_vs_greedy"] < 0.0
