#!/usr/bin/env python
"""Run coupled quantum bootstrap embedding with a VQE fragment solver."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.vqe_solver import VQE_ArgsUser, clear_ansatz_cache, reset_vqe_state

if __package__:
    from .lookahead_utils import SelectorConfig, patched_fast_adapt
else:
    from lookahead_utils import SelectorConfig, patched_fast_adapt


ANSATZ_CHOICES = ("uccsd", "adapt", "adapt_fast", "adapt_matrix_free")
ESTIMATOR_CHOICES = ("statevector", "aer_exact", "direct_sv", "backend")
FRAGMENTATION_CHOICES = ("autogen", "chemgen", "graphgen")
OPTIMIZER_CHOICES = ("SLSQP", "COBYLA", "L_BFGS_B", "SPSA")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a closed-shell molecular QBE-VQE calculation from XYZ geometry."
    )
    parser.add_argument("xyz", type=Path, help="XYZ geometry file")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--unit", choices=("angstrom", "bohr"), default="angstrom")
    parser.add_argument("--be-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--fragmentation", choices=FRAGMENTATION_CHOICES, default="autogen"
    )
    parser.add_argument(
        "--only-chemical-potential",
        action="store_true",
        help="Disable density matching; always enabled for BE1",
    )
    parser.add_argument("--ansatz", choices=ANSATZ_CHOICES, default="adapt_fast")
    parser.add_argument("--estimator", choices=ESTIMATOR_CHOICES, default="aer_exact")
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default="SLSQP")
    parser.add_argument("--selector", choices=("greedy", "lookahead"), default="greedy")
    parser.add_argument("--be-iterations", type=int, default=50)
    parser.add_argument("--be-convergence", type=float, default=1e-6)
    parser.add_argument("--vqe-iterations", type=int, default=100)
    parser.add_argument("--adapt-iterations", type=int, default=20)
    parser.add_argument("--gradient-threshold", type=float, default=1e-3)
    parser.add_argument("--energy-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--frozen-core", choices=("none", "auto", "manual"), default="none"
    )
    parser.add_argument("--frozen-core-orbitals", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookahead-top-k", type=int, default=5)
    parser.add_argument("--lookahead-probe-iterations", type=int, default=40)
    parser.add_argument("--lookahead-start-iteration", type=int, default=5)
    parser.add_argument("--lookahead-recent-window", type=int, default=10)
    parser.add_argument("--lookahead-min-overlap", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verbose", type=int, choices=range(4), default=1)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.spin != 0:
        raise ValueError("QBEMol's molecular VQE workflow currently requires spin=0")
    if args.selector == "lookahead" and args.ansatz != "adapt_fast":
        raise ValueError("lookahead selection requires --ansatz adapt_fast")
    if args.estimator == "direct_sv" and args.ansatz not in {"uccsd", "adapt_fast"}:
        raise ValueError("direct_sv supports --ansatz uccsd or --ansatz adapt_fast")
    if args.frozen_core == "manual" and args.frozen_core_orbitals < 1:
        raise ValueError("manual frozen core requires --frozen-core-orbitals >= 1")
    positive_values = {
        "be-iterations": args.be_iterations,
        "be-convergence": args.be_convergence,
        "vqe-iterations": args.vqe_iterations,
        "adapt-iterations": args.adapt_iterations,
        "gradient-threshold": args.gradient_threshold,
        "energy-threshold": args.energy_threshold,
        "threads": args.threads,
        "lookahead-top-k": args.lookahead_top_k,
        "lookahead-probe-iterations": args.lookahead_probe_iterations,
        "lookahead-start-iteration": args.lookahead_start_iteration,
        "lookahead-recent-window": args.lookahead_recent_window,
        "lookahead-min-overlap": args.lookahead_min_overlap,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")


def build_molecule(args: argparse.Namespace) -> gto.Mole:
    xyz = args.xyz.expanduser().resolve()
    if not xyz.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz}")
    return gto.M(
        atom=str(xyz),
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
        verbose=args.verbose,
    )


def build_vqe_args(args: argparse.Namespace, hamiltonian_dir: Path) -> VQE_ArgsUser:
    return VQE_ArgsUser(
        hamiltonian_dir=str(hamiltonian_dir),
        ansatz_type=args.ansatz,
        estimator_type=args.estimator,
        optimizer_name=args.optimizer,
        adapt_gradient_threshold=args.gradient_threshold,
        adapt_eigenvalue_threshold=args.energy_threshold,
        adapt_max_iterations=args.adapt_iterations,
        adapt_check_cyclicity=True,
        adapt_cyclicity_action="skip",
        stage1_max_iter=args.vqe_iterations,
        stage2_max_iter=args.vqe_iterations,
        stage3_max_iter=args.vqe_iterations,
        frozen_core=args.frozen_core,
        frozen_core_num_orbitals=args.frozen_core_orbitals,
        aer_max_parallel_threads=args.threads,
        max_restarts=1,
        random_seed=args.seed,
        warm_start=False,
        verbose=args.verbose,
    )


def build_selector_config(args: argparse.Namespace) -> SelectorConfig:
    return SelectorConfig(
        top_k=args.lookahead_top_k,
        probe_maxiter=args.lookahead_probe_iterations,
        probe_backend="direct_sv",
        probe_aer_threads=args.threads,
        generic_min_iteration=args.lookahead_start_iteration,
        generic_recent_window=args.lookahead_recent_window,
        generic_min_overlap=args.lookahead_min_overlap,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    molecule = build_molecule(args)
    output_dir = (args.output_dir or Path("results") / args.xyz.stem).resolve()
    hamiltonian_dir = output_dir / "hamiltonians"
    hamiltonian_dir.mkdir(parents=True, exist_ok=True)

    mean_field = scf.RHF(molecule)
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("RHF did not converge")

    clear_ansatz_cache()
    reset_vqe_state()
    fragments = fragmentate(
        mol=molecule,
        frag_type=args.fragmentation,
        n_BE=args.be_order,
    )
    embedding = BE(mean_field, fragments)
    vqe_args = build_vqe_args(args, hamiltonian_dir)
    only_chem = args.only_chemical_potential or args.be_order == 1

    guided_class = None
    selector_context = nullcontext(None)
    if args.selector == "lookahead":
        selector_context = patched_fast_adapt(
            config=build_selector_config(args),
            case_name=args.xyz.stem,
            selector_mode="lookahead_generic",
            excitation_labels=[],
            log_prefix="QBELookahead",
        )

    started = perf_counter()
    with selector_context as guided_class:
        embedding.optimize(
            solver="VQE",
            solver_args=vqe_args,
            conv_tol=args.be_convergence,
            max_iter=args.be_iterations,
            only_chem=only_chem,
        )
    runtime = perf_counter() - started

    result: dict[str, Any] = {
        "energy_hartree": float(embedding.ebe_tot),
        "hartree_fock_energy": float(mean_field.e_tot),
        "runtime_seconds": runtime,
        "fragment_count": len(embedding.Fobjs),
        "input": {
            "geometry": str(args.xyz.expanduser().resolve()),
            "basis": args.basis,
            "charge": args.charge,
            "spin": args.spin,
            "unit": args.unit,
        },
        "qbe": {
            "order": args.be_order,
            "fragmentation": args.fragmentation,
            "only_chemical_potential": only_chem,
            "max_iterations": args.be_iterations,
            "convergence_tolerance": args.be_convergence,
        },
        "vqe": {
            "ansatz": args.ansatz,
            "estimator": args.estimator,
            "optimizer": args.optimizer,
            "max_iterations": args.vqe_iterations,
            "adapt_max_iterations": args.adapt_iterations,
            "gradient_threshold": args.gradient_threshold,
            "energy_threshold": args.energy_threshold,
            "frozen_core": args.frozen_core,
            "frozen_core_orbitals": args.frozen_core_orbitals,
            "threads": args.threads,
            "seed": args.seed,
        },
        "selector": args.selector,
    }
    if guided_class is not None:
        result["lookahead"] = {
            "top_k": args.lookahead_top_k,
            "probe_max_iterations": args.lookahead_probe_iterations,
            "start_iteration": args.lookahead_start_iteration,
            "recent_window": args.lookahead_recent_window,
            "min_overlap": args.lookahead_min_overlap,
            "solves": list(guided_class.solve_records),
        }

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"QBE-VQE energy: {embedding.ebe_tot:.12f} Ha")
    print(f"Result: {result_path}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.xyz.expanduser().is_file():
        parser.error(f"XYZ file not found: {args.xyz.expanduser().resolve()}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
