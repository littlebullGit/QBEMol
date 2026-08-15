from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pyscf import gto

from scripts import f2_qbe_vqe as f2
from scripts import h4_qbe_vqe as h4
from scripts import validate_adapt_implementations as adapt_validation


@pytest.mark.parametrize(
    (
        "module",
        "system_name",
        "be_order",
        "expected_selector",
        "expected_coordinates",
    ),
    (
        (f2, "f2", 1, "lookahead", [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]]),
        (
            h4,
            "h4",
            2,
            "greedy",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
        ),
    ),
)
def test_molecule_script_runs_generic_qbe_vqe(
    module: ModuleType,
    system_name: str,
    be_order: int,
    expected_selector: str,
    expected_coordinates: list[list[float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require each molecule script to invoke only the generic QBE-VQE runner."""

    calls: list[list[str]] = []

    def capture_run(arguments: list[str]) -> int:
        """Capture one delegated generic-runner invocation."""

        calls.append(arguments)
        return 0

    assert callable(getattr(module, "run_qbe_vqe", None))
    assert not any("reference" in name.lower() for name in vars(module))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "run_qbe_vqe", capture_run)

    assert module.main() == 0
    assert len(calls) == 1

    arguments = calls[0]
    geometry_path = Path(arguments[0])
    output_dir = Path(arguments[arguments.index("--output-dir") + 1])
    molecule = gto.M(atom=str(geometry_path), basis="sto-3g", unit="angstrom")

    assert geometry_path == output_dir / f"{system_name}.xyz"
    assert arguments[arguments.index("--be-order") + 1] == str(be_order)
    selector = (
        arguments[arguments.index("--selector") + 1]
        if "--selector" in arguments
        else "greedy"
    )
    assert selector == expected_selector
    np.testing.assert_allclose(
        molecule.atom_coords(unit="angstrom"), expected_coordinates, atol=1e-12
    )


def test_adapt_validation_uses_publication_tolerance() -> None:
    """Keep the separate implementation validator's boundary behavior covered."""

    results = [
        {"difference_from_reference_mha": adapt_validation.MATCH_TOLERANCE_MHA / 2}
    ]
    assert adapt_validation.evaluate_results(results)

    results[0]["difference_from_reference_mha"] = adapt_validation.MATCH_TOLERANCE_MHA
    assert not adapt_validation.evaluate_results(results)
