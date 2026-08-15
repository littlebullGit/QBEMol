"""Focused non-long tests for the standalone H6 production CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "h6_qbe_vqe.py"


def _load_script_module():
    """Load the H6 production script as a Python module."""

    spec = importlib.util.spec_from_file_location("h6_qbe_vqe", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_records_h6_production_controls(tmp_path: Path) -> None:
    """Dry-run should expose the H6 solver controls without result targets."""

    module = _load_script_module()
    exit_code = module.main(
        [
            "--dry-run",
            "--skip-fci-be",
            "--spacings",
            "1.0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "result.json").read_text())

    assert exit_code == 0
    assert payload["status"] == "NOT_RUN"
    assert payload["production_controls"] == {
        "basis": "sto-3g",
        "be_order": 2,
        "fragmentation": "chemgen",
        "h_treatment": "treat_H_like_heavy_atom",
        "ansatz": "adapt_fast",
        "estimator": "direct_sv",
        "optimizer": "SLSQP",
        "stage1_max_iter": 200,
        "stage2_max_iter": 200,
        "stage3_max_iter": 200,
        "stage1_energy_tol": 1e-10,
        "stage2_energy_tol": 1e-10,
        "stage3_energy_tol": 1e-10,
        "adapt_gradient_threshold": 1e-3,
        "adapt_eigenvalue_threshold": 0.0,
        "adapt_max_iterations": 50,
        "adapt_check_cyclicity": False,
        "max_restarts": 1,
        "random_seed": 281,
        "warm_start": False,
        "direct_sv_use_exact_jacobian": False,
    }
    assert [cell["case"] for cell in payload["cells"]] == ["greedy", "lookahead"]
    assert "reference" not in payload


def test_vqe_arguments_use_fastadapt_direct_statevector(tmp_path: Path) -> None:
    """Compute mode should instantiate the accepted fragment-solver arguments."""

    module = _load_script_module()
    args = module.build_vqe_args(tmp_path / "hamiltonians")

    assert args.ansatz_type == "adapt_fast"
    assert args.estimator_type == "direct_sv"
    assert args.adapt_max_iterations == 50
    assert args.adapt_check_cyclicity is False
    assert args.direct_sv_use_exact_jacobian is False
    assert (args.stage1_max_iter, args.stage2_max_iter, args.stage3_max_iter) == (
        200,
        200,
        200,
    )
