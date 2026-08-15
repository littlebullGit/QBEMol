import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from attrs import evolve

from quemb.molbe import vqe_solver


def test_vqe_state_stage_transitions():
    state = vqe_solver.VQEState()
    args = vqe_solver.VQE_ArgsUser()
    assert state.get_stage(1.0, args) == 1
    state.update_be_iteration(-1.0)
    assert state.get_be_energy_change() == pytest.approx(1.0)
    stage = state.get_stage(state.get_be_energy_change(), args)
    assert stage == 1
    state.update_be_iteration(-0.995)
    assert state.get_be_energy_change() == pytest.approx(0.005)
    stage = state.get_stage(state.get_be_energy_change(), args)
    assert stage == 2
    state.update_be_iteration(-0.9946)
    assert state.get_be_energy_change() == pytest.approx(0.0004)
    stage = state.get_stage(state.get_be_energy_change(), args)
    assert stage == 3


def test_vqe_state_get_vqe_params():
    state = vqe_solver.VQEState()
    args = vqe_solver.VQE_ArgsUser()
    params = state.get_vqe_params(1, args)
    assert params == (args.stage1_max_iter, args.stage1_energy_tol)
    params = state.get_vqe_params(2, args)
    assert params == (args.stage2_max_iter, args.stage2_energy_tol)
    params = state.get_vqe_params(3, args)
    assert params == (args.stage3_max_iter, args.stage3_energy_tol)


def test_vqe_state_adaptive_disabled():
    state = vqe_solver.VQEState()
    args = evolve(vqe_solver.VQE_ArgsUser(), adaptive_convergence=False)
    stage = state.get_stage(0.0, args)
    assert stage == 2


def test_full_canonicalization_rejects_frozen_core():
    """Keep the paper orbital basis out of reduced active-space runs."""

    with pytest.raises(ValueError, match="only without frozen-core"):
        vqe_solver.VQE_ArgsUser(
            orbital_canonicalization="full",
            frozen_core="manual",
            frozen_core_num_orbitals=1,
        )


def test_parse_fcidump_requires_qiskit(monkeypatch):
    monkeypatch.setattr(vqe_solver, "QISKIT_AVAILABLE", False)
    with pytest.raises(ImportError):
        vqe_solver.parse_fcidump_hamiltonian(Path("dummy"))


def test_compute_rdms_requires_qiskit(monkeypatch):
    monkeypatch.setattr(vqe_solver, "QISKIT_AVAILABLE", False)
    with pytest.raises(ImportError):
        vqe_solver.compute_rdms_from_statevector(object(), 1, 1)


def test_solve_vqe_requires_qiskit(monkeypatch):
    monkeypatch.setattr(vqe_solver, "QISKIT_AVAILABLE", False)
    frag = SimpleNamespace(dname="frag0")
    with pytest.raises(ImportError):
        vqe_solver.solve_vqe(object(), frag, vqe_solver.VQE_ArgsUser())
