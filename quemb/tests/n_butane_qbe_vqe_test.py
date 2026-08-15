"""Focused non-long tests for the standalone n-butane production CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "n-butane_qbe_vqe.py"


def _load_script_module():
    """Load the hyphenated script file as a Python module."""

    spec = importlib.util.spec_from_file_location("n_butane_qbe_vqe", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_writes_plan_and_selected_geometries(tmp_path: Path) -> None:
    """Dry-run should stay pure-Python and emit a launch plan."""

    module = _load_script_module()
    exit_code = module.main(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--distances",
            "1.30",
            "2.10",
            "--selectors",
            "greedy",
        ]
    )

    payload = json.loads((tmp_path / "dry_run.json").read_text())
    assert exit_code == 0
    assert payload["schema"] == "qbemol.n-butane-qbe-vqe.dry-run.v1"
    assert payload["selected_distances_angstrom"] == [
        pytest.approx(1.30),
        pytest.approx(2.10),
    ]
    assert payload["selected_selectors"] == ["greedy"]
    assert payload["generated_sector_pool_sentinel"] == "__generated_ovp_ceo_pool__"
    assert all(str(tmp_path) in item["planned_xyz"] for item in payload["geometries"])


def test_default_plan_is_the_full_production_scan(tmp_path: Path) -> None:
    """The default command should plan all three distances and selectors."""

    module = _load_script_module()
    exit_code = module.main(["--dry-run", "--output-dir", str(tmp_path)])
    payload = json.loads((tmp_path / "dry_run.json").read_text())

    assert exit_code == 0
    assert payload["selected_distances_angstrom"] == [1.30, 1.54, 2.10]
    assert payload["selected_selectors"] == ["greedy", "lookahead"]
    assert payload["source_sha256"]


def test_compute_arguments_use_the_generated_fixed_sector_pool(tmp_path: Path) -> None:
    """Compute mode should bind the W12 sector pool and accepted solver controls."""

    module = _load_script_module()
    modules = module._load_compute_modules()
    args = module.build_vqe_args(
        modules,
        tmp_path,
        module.SELECTOR_LABELS["lookahead"],
        verbose=0,
    )

    assert modules["generated_sector_pool"] == module.GENERATED_SECTOR_POOL
    assert args.ansatz_type == "adapt_sector"
    assert args.estimator_type == "direct_sv"
    assert args.adapt_sector_selector_policy == "always_top5_energy"
    assert args.adapt_max_iterations == 20
    assert args.adapt_check_cyclicity is False
    assert args.direct_sv_use_exact_jacobian is False
