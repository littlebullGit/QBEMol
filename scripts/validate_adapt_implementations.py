#!/usr/bin/env python
"""Reproduce the H4/F2 FastAdaptVQE and MatrixFreeAdaptVQE validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.vqe_solver import VQE_ArgsUser, clear_ansatz_cache, reset_vqe_state

REFERENCE_ENERGIES = {
    "h4": {
        "adapt_fast": -2.1661448638,
        "adapt_matrix_free": -2.1661448675,
        "fci_be": -2.1663874445,
    },
    "f2": {
        "adapt_fast": -195.9728739586,
        "adapt_matrix_free": -195.9728724736,
        "fci_be": -196.0442447354,
    },
}
MATCH_TOLERANCE_MHA = 0.5
RESULTS_DIR = Path(
    os.environ.get("ADAPT_VALIDATION_RESULTS_DIR", "results/adapt_validation")
)


@dataclass(frozen=True)
class Benchmark:
    molecule: Callable[[], gto.Mole]
    be_order: int
    only_chem: bool
    adapt_iterations: int


def create_h4() -> gto.Mole:
    return gto.M(
        atom=[("H", (float(index), 0.0, 0.0)) for index in range(4)],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="angstrom",
    )


def create_f2() -> gto.Mole:
    return gto.M(
        atom=[("F", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 1.42))],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="angstrom",
    )


BENCHMARKS = {
    "h4": Benchmark(create_h4, be_order=2, only_chem=False, adapt_iterations=20),
    "f2": Benchmark(create_f2, be_order=1, only_chem=True, adapt_iterations=50),
}


def build_vqe_args(
    ansatz: str,
    adapt_iterations: int,
    hamiltonian_dir: Path,
) -> VQE_ArgsUser:
    """Build one full-space VQE configuration for a validation case."""

    return VQE_ArgsUser(
        estimator_type="aer_exact",
        ansatz_type=ansatz,
        frozen_core="none",
        orbital_canonicalization="full",
        ucc_generalized=False,
        ucc_reps=1,
        ucc_preserve_spin=True,
        adapt_gradient_threshold=1e-3,
        adapt_eigenvalue_threshold=1e-5,
        adapt_max_iterations=adapt_iterations,
        stage1_max_iter=100,
        stage2_max_iter=100,
        stage3_max_iter=100,
        optimizer_name="SLSQP",
        max_restarts=1,
        random_seed=42,
        verbose=1,
        warm_start=False,
        hamiltonian_dir=str(hamiltonian_dir),
    )


def run_case(system_name: str, ansatz: str) -> dict[str, float | str]:
    benchmark = BENCHMARKS[system_name]
    molecule = benchmark.molecule()
    mean_field = scf.RHF(molecule)
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"{system_name} RHF did not converge")

    clear_ansatz_cache()
    reset_vqe_state()
    hamiltonian_dir = RESULTS_DIR / "hamiltonians" / system_name / ansatz
    hamiltonian_dir.mkdir(parents=True, exist_ok=True)
    embedding = BE(
        mean_field,
        fragmentate(
            mol=molecule,
            frag_type="autogen",
            n_BE=benchmark.be_order,
        ),
    )
    started = perf_counter()
    embedding.optimize(
        solver="VQE",
        solver_args=build_vqe_args(
            ansatz,
            benchmark.adapt_iterations,
            hamiltonian_dir,
        ),
        conv_tol=1e-6,
        max_iter=50,
        only_chem=benchmark.only_chem,
    )
    energy = float(embedding.ebe_tot)
    difference = 1000.0 * abs(energy - REFERENCE_ENERGIES[system_name][ansatz])
    return {
        "system": system_name,
        "ansatz": ansatz,
        "energy_hartree": energy,
        "difference_from_reference_mha": difference,
        "runtime_seconds": perf_counter() - started,
    }


def evaluate_results(results: list[dict[str, float | str]]) -> bool:
    return all(
        float(result["difference_from_reference_mha"]) < MATCH_TOLERANCE_MHA
        for result in results
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(system, ansatz)
        for system in BENCHMARKS
        for ansatz in ("adapt_fast", "adapt_matrix_free")
    ]
    passed = evaluate_results(results)
    output = {
        "reference_energies": REFERENCE_ENERGIES,
        "tolerance_mha": MATCH_TOLERANCE_MHA,
        "results": results,
        "passed": passed,
    }
    output_path = RESULTS_DIR / "result.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"ADAPT implementation validation: {'PASS' if passed else 'FAIL'}")
    print(f"Result: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
