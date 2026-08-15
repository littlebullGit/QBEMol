import json
from pathlib import Path

import pytest

import scripts.run_qbe_vqe as runner
from scripts.run_qbe_vqe import (
    build_molecule,
    build_parser,
    build_selector_config,
    build_vqe_args,
    validate_args,
)


def parse_args(*args: str):
    return build_parser().parse_args(["molecule.xyz", *args])


def test_default_configuration_uses_fast_adapt():
    args = parse_args()

    validate_args(args)

    assert args.be_order == 2
    assert args.ansatz == "adapt_fast"
    assert args.selector == "greedy"


def test_lookahead_requires_fast_adapt():
    args = parse_args("--selector", "lookahead", "--ansatz", "uccsd")

    with pytest.raises(ValueError, match="requires --ansatz adapt_fast"):
        validate_args(args)


def test_direct_statevector_rejects_non_fast_adapt():
    args = parse_args("--estimator", "direct_sv", "--ansatz", "adapt_matrix_free")

    with pytest.raises(ValueError, match="supports --ansatz uccsd or"):
        validate_args(args)


def test_solver_and_selector_configuration(tmp_path: Path):
    args = parse_args(
        "--selector",
        "lookahead",
        "--adapt-iterations",
        "12",
        "--lookahead-top-k",
        "3",
    )

    vqe_args = build_vqe_args(args, tmp_path / "hamiltonians")
    selector = build_selector_config(args)

    assert vqe_args.ansatz_type == "adapt_fast"
    assert vqe_args.adapt_max_iterations == 12
    assert vqe_args.hamiltonian_dir == str(tmp_path / "hamiltonians")
    assert selector.top_k == 3


def test_build_molecule_from_xyz(tmp_path: Path):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    args = parse_args()
    args.xyz = xyz

    molecule = build_molecule(args)

    assert molecule.natm == 2
    assert molecule.nelectron == 2


def test_run_couples_be1_to_vqe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    output_dir = tmp_path / "result"
    args = build_parser().parse_args(
        [str(xyz), "--be-order", "1", "--output-dir", str(output_dir)]
    )
    optimize_args = {}

    class FakeMeanField:
        converged = True
        e_tot = -1.0

        def kernel(self):
            return self.e_tot

    class FakeEmbedding:
        ebe_tot = -1.1
        Fobjs = [object()]

        def optimize(self, **kwargs):
            optimize_args.update(kwargs)

    monkeypatch.setattr(runner.scf, "RHF", lambda molecule: FakeMeanField())
    monkeypatch.setattr(runner, "fragmentate", lambda **kwargs: object())
    monkeypatch.setattr(runner, "BE", lambda mean_field, fragments: FakeEmbedding())
    monkeypatch.setattr(runner, "clear_ansatz_cache", lambda: None)
    monkeypatch.setattr(runner, "reset_vqe_state", lambda: None)

    result = runner.run(args)

    assert optimize_args["solver"] == "VQE"
    assert optimize_args["only_chem"] is True
    assert result["energy_hartree"] == pytest.approx(-1.1)
    assert json.loads((output_dir / "result.json").read_text())["qbe"]["order"] == 1
