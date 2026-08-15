"""Run the H6 QBE-VQE scan over 1.0, 1.5, and 2.0 Angstroms.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyscf import fci, gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.ceo_manifest import load_ceo_manifest_for_system
from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.fast_adapt_vqe import (
    ExactSparseSVSolver,
    FastAdaptVQE,
    _SelectionDecision,
    summarize_fastadapt_timings,
)
from quemb.molbe.vqe_solver import (
    VQE_ArgsUser,
    _adapt_selector_factory_context,
    _exact_sparse_ceo_manifest_context,
    clear_ansatz_cache,
    reset_vqe_state,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "h6_qbe_vqe"
MANIFEST_REGISTRY = (
    PROJECT_ROOT / "data" / "h6_manifests" / "h6_be_fragment_pool_registry.json"
)
BE_MAX_ITERATIONS = 50
CASE_POLICIES = {
    "greedy": "greedy_gradient",
    "lookahead": "always_top5_energy",
}


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one local file."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI for the standalone H6 compute script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
        help="H-H spacings to run in Angstrom. Default: 1.0 1.5 2.0",
    )
    parser.add_argument(
        "--skip-fci-be",
        action="store_true",
        help="Skip the FCI-BE control cells and run only greedy/lookahead CEO-VQE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for per-cell artifacts and the aggregate result.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the exact compute plan without launching long calculations.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid spacings or a missing local manifest registry."""

    if any(spacing <= 0.0 for spacing in args.spacings):
        raise ValueError("H6 spacings must be positive")
    if len(set(args.spacings)) != len(args.spacings):
        raise ValueError("H6 spacings must not repeat")
    if not MANIFEST_REGISTRY.is_file():
        raise FileNotFoundError(
            f"Local manifest registry not found: {MANIFEST_REGISTRY}"
        )


def _case_output_path(output_dir: Path, spacing: float, case_name: str) -> Path:
    """Resolve the JSON artifact path for one spacing/case pair."""

    return output_dir / f"spacing_{spacing:.1f}" / case_name / "result.json"


def _manifest_summary() -> dict[str, Any]:
    """Summarize the local CEO manifest registry used by the fresh run."""

    registry = {
        "path": str(MANIFEST_REGISTRY),
        "sha256": _sha256(MANIFEST_REGISTRY),
        "resolved": {},
    }
    for n_qubits, n_electrons in ((4, 2), (8, 4), (12, 6)):
        manifest = load_ceo_manifest_for_system(
            MANIFEST_REGISTRY,
            n_qubits=n_qubits,
            n_electrons=n_electrons,
        )
        registry["resolved"][f"{n_qubits}q_{n_electrons}e"] = {
            "manifest_digest": manifest.digest,
            "pool_digest": manifest.pool_digest,
            "operator_count": len(manifest.operators),
        }
    return registry


def _plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the exact standalone compute plan without running it."""

    cases = ["greedy", "lookahead"]
    if not args.skip_fci_be:
        cases = ["fci_be", *cases]
    return {
        "status": "NOT_RUN",
        "mode": "dry_run",
        "project_root": str(PROJECT_ROOT),
        "manifest": _manifest_summary(),
        "production_controls": {
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
        },
        "cells": [
            {
                "spacing_angstrom": spacing,
                "case": case_name,
                "output_path": str(
                    _case_output_path(args.output_dir, spacing, case_name)
                ),
                "selector_policy": CASE_POLICIES.get(case_name),
            }
            for spacing in args.spacings
            for case_name in cases
        ],
    }


def create_h6(spacing: float) -> gto.Mole:
    """Create the linear H6 STO-3G molecule for one spacing."""

    return gto.M(
        atom=[("H", (index * spacing, 0.0, 0.0)) for index in range(6)],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="angstrom",
    )


def build_fragments(molecule: gto.Mole) -> list[Any]:
    """Build the BE2 `chemgen` fragmentation used in the H6 scan."""

    return fragmentate(
        mol=molecule,
        frag_type="chemgen",
        n_BE=2,
        frozen_core=False,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )


def build_mean_field(spacing: float) -> scf.hf.RHF:
    """Construct and solve the RHF reference for one H6 geometry."""

    mean_field = scf.RHF(create_h6(spacing))
    mean_field.conv_tol = 1e-12
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"H6 RHF did not converge at {spacing:.1f} Angstrom")
    return mean_field


def build_vqe_args(hamiltonian_dir: Path) -> VQE_ArgsUser:
    """Construct the VQE fragment-solver settings for the H6 run."""

    hamiltonian_dir.mkdir(parents=True, exist_ok=True)
    return VQE_ArgsUser(
        hamiltonian_dir=str(hamiltonian_dir),
        ansatz_type="adapt_fast",
        estimator_type="direct_sv",
        optimizer_name="SLSQP",
        stage1_max_iter=200,
        stage2_max_iter=200,
        stage3_max_iter=200,
        stage1_energy_tol=1e-10,
        stage2_energy_tol=1e-10,
        stage3_energy_tol=1e-10,
        adapt_gradient_threshold=1e-3,
        adapt_eigenvalue_threshold=0.0,
        adapt_max_iterations=50,
        adapt_check_cyclicity=False,
        max_restarts=1,
        random_seed=281,
        warm_start=False,
        verbose=2,
        direct_sv_use_exact_jacobian=False,
    )


@contextmanager
def _capture_be_optimization() -> Iterator[dict[str, Any]]:
    """Capture BE outer-loop convergence data for one embedding run."""

    import quemb.molbe.mbe as mbe_module

    captured: dict[str, Any] = {}
    original = mbe_module.BEOPT.optimize

    def wrapped(optimizer, *call_args, **call_kwargs):
        started = perf_counter()
        try:
            return original(optimizer, *call_args, **call_kwargs)
        finally:
            final_error = float(optimizer.err)
            tolerance = float(optimizer.conv_tol)
            captured.update(
                {
                    "iterations_completed": int(optimizer.iter),
                    "maximum_iterations": int(optimizer.max_space),
                    "final_density_error": final_error,
                    "convergence_tolerance": tolerance,
                    "converged": final_error < tolerance,
                    "optimize_wall_s": perf_counter() - started,
                    "final_potentials": [float(value) for value in optimizer.pot],
                }
            )

    mbe_module.BEOPT.optimize = wrapped
    try:
        yield captured
    finally:
        mbe_module.BEOPT.optimize = original


def _finite_energy(value: object) -> float | None:
    """Return a finite float or `None` for a non-finite energy-like value."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        as_float = float(value)
        if np.isfinite(as_float):
            return as_float
    return None


def _extract_current_theta(theta: Sequence[float]) -> list[float]:
    """Return the current ADAPT parameter vector as plain Python floats."""

    return [float(value) for value in np.asarray(theta, dtype=float).reshape(-1)]


def _build_exact_sparse_shortlist(
    *,
    abs_gradients: Sequence[float],
    sorted_candidates: Sequence[int],
    prev_op_indices: Sequence[int],
    gradient_threshold: float,
    top_k: int,
    check_cyclicity: bool,
    cyclicity_fn: Callable[[list[int]], bool],
) -> tuple[list[int], list[int], str | None]:
    """Build the deterministic top-k shortlist used by the production path."""

    shortlist: list[int] = []
    skipped_cyclic: list[int] = []
    termination_reason: str | None = None
    for candidate_idx in sorted_candidates:
        candidate_grad = float(abs_gradients[candidate_idx])
        if candidate_grad < gradient_threshold:
            termination_reason = "converged"
            break
        if check_cyclicity and cyclicity_fn(
            list(prev_op_indices) + [int(candidate_idx)]
        ):
            skipped_cyclic.append(int(candidate_idx))
            continue
        shortlist.append(int(candidate_idx))
        if len(shortlist) >= top_k:
            break
    return shortlist, skipped_cyclic, termination_reason


def _select_probe_winner(
    probe_records: Sequence[dict[str, Any]],
    *,
    energy_tie_tolerance: float,
) -> dict[str, Any] | None:
    """Pick the lowest relaxed-energy probe with deterministic tie-breaking."""

    successful = [
        record
        for record in probe_records
        if bool(record.get("optimizer_success"))
        and _finite_energy(record.get("relaxed_energy")) is not None
    ]
    if not successful:
        return None
    minimum_energy = min(float(record["relaxed_energy"]) for record in successful)
    tied = [
        record
        for record in successful
        if float(record["relaxed_energy"]) <= minimum_energy + energy_tie_tolerance
    ]
    tied.sort(
        key=lambda record: (
            int(record.get("gradient_rank", 10**9)),
            int(record.get("candidate_idx", 10**9)),
        )
    )
    return tied[0]


def _serialize_optimizer_result(optimizer_result: object) -> dict[str, Any]:
    """Normalize solver optimizer metadata into one JSON-safe dictionary."""

    if optimizer_result is None:
        return {
            "success": True,
            "status": 0,
            "message": "no_parameters",
            "nfev": 1,
            "nit": 0,
        }
    return {
        "success": bool(getattr(optimizer_result, "success", False)),
        "status": int(getattr(optimizer_result, "status", -1)),
        "message": str(getattr(optimizer_result, "message", "")),
        "nfev": int(getattr(optimizer_result, "nfev", 0) or 0),
        "nit": int(getattr(optimizer_result, "nit", 0) or 0),
    }


def _probe_exact_sparse_candidate(
    *,
    initial_state,
    prefix_indices: Sequence[int],
    candidate_idx: int,
    excitation_pool: Sequence[Any],
    theta_prefix: Sequence[float],
    operator,
    h_sparse,
    maxiter: int,
    ftol: float,
) -> dict[str, Any]:
    """Run one exact-sparse candidate relaxation for the always-top5 selector."""

    candidate_theta0 = _extract_current_theta(theta_prefix) + [0.0]
    operator_sequence = [
        excitation_pool[idx] for idx in [*prefix_indices, candidate_idx]
    ]
    probe_t0 = perf_counter()
    solver = ExactSparseSVSolver(
        initial_state=initial_state,
        optimizer_maxiter=maxiter,
        optimizer_ftol=ftol,
        initial_point=np.asarray(candidate_theta0, dtype=float),
        verbose=0,
    )
    solver.set_operator_sequence(operator_sequence)
    raw_result = solver.compute_minimum_eigenvalue(operator)
    elapsed_s = perf_counter() - probe_t0

    optimal_point = np.asarray(raw_result.optimal_point, dtype=float).reshape(-1)
    relaxed_energy = _finite_energy(getattr(raw_result, "eigenvalue", None))
    recomputed_state = solver.statevector_for(optimal_point)
    recomputed_energy = float(
        np.vdot(recomputed_state, h_sparse @ recomputed_state).real
    )
    energy_residual = (
        None if relaxed_energy is None else abs(relaxed_energy - recomputed_energy)
    )
    optimizer_payload = _serialize_optimizer_result(
        getattr(raw_result, "optimizer_result", None)
    )
    solver_breakdown = getattr(solver, "last_timing_breakdown", None)

    return {
        "candidate_idx": int(candidate_idx),
        "prefix_indices": [int(idx) for idx in prefix_indices],
        "path_indices": [int(idx) for idx in [*prefix_indices, candidate_idx]],
        "initial_point": candidate_theta0,
        "optimal_point": [float(value) for value in optimal_point.tolist()],
        "relaxed_energy": relaxed_energy,
        "recomputed_energy": float(recomputed_energy),
        "energy_residual": None if energy_residual is None else float(energy_residual),
        "optimizer_success": bool(optimizer_payload["success"]),
        "optimizer_status": int(optimizer_payload["status"]),
        "optimizer_message": str(optimizer_payload["message"]),
        "optimizer_nfev": int(optimizer_payload["nfev"]),
        "optimizer_nit": int(optimizer_payload["nit"]),
        "backend": (
            str(solver_breakdown.get("backend", "exact_sparse"))
            if isinstance(solver_breakdown, dict)
            else "exact_sparse"
        ),
        "elapsed_s": float(elapsed_s),
    }


def probe_exact_sparse_candidates(
    *,
    initial_state,
    prefix_indices: Sequence[int],
    candidate_indices: Sequence[int],
    excitation_pool: Sequence[Any],
    theta_prefix: Sequence[float],
    operator,
    h_sparse,
    maxiter: int,
    ftol: float,
    abs_gradients: Sequence[float],
    sorted_candidates: Sequence[int],
) -> list[dict[str, Any]]:
    """Probe one shortlist with the accepted exact-sparse candidate solver."""

    rank_map = {
        int(candidate_idx): int(rank + 1)
        for rank, candidate_idx in enumerate(sorted_candidates)
    }
    records: list[dict[str, Any]] = []
    for candidate_idx in candidate_indices:
        probe_started = perf_counter()
        try:
            record = _probe_exact_sparse_candidate(
                initial_state=initial_state,
                prefix_indices=prefix_indices,
                candidate_idx=int(candidate_idx),
                excitation_pool=excitation_pool,
                theta_prefix=theta_prefix,
                operator=operator,
                h_sparse=h_sparse,
                maxiter=maxiter,
                ftol=ftol,
            )
        except Exception as exc:
            record = {
                "candidate_idx": int(candidate_idx),
                "prefix_indices": [int(idx) for idx in prefix_indices],
                "path_indices": [int(idx) for idx in [*prefix_indices, candidate_idx]],
                "initial_point": _extract_current_theta(theta_prefix) + [0.0],
                "optimal_point": None,
                "relaxed_energy": None,
                "recomputed_energy": None,
                "energy_residual": None,
                "optimizer_success": False,
                "optimizer_status": -1,
                "optimizer_message": f"{type(exc).__name__}: {exc}",
                "optimizer_nfev": 0,
                "optimizer_nit": 0,
                "backend": "exact_sparse",
                "elapsed_s": float(perf_counter() - probe_started),
            }
        record["abs_gradient"] = float(abs_gradients[candidate_idx])
        record["gradient_rank"] = int(rank_map[candidate_idx])
        records.append(record)
    return records


def _merge_selection_diagnostics(
    current: object,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Merge one selector event into the existing ADAPT diagnostics payload."""

    merged = dict(current or {}) if isinstance(current, dict) else {}
    merged["exact_sparse_selector"] = event
    return merged


def _make_recording_adapt_class(
    base_class: type[FastAdaptVQE],
    *,
    selector_policy: str,
) -> type[FastAdaptVQE]:
    """Create a recording ADAPT class with optional exact sparse top-5 selection."""

    class RecordingAdapt(base_class):
        solve_records: list[dict[str, Any]] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.selector_events: list[dict[str, Any]] = []
            self._exact_sparse_h_cache: dict[int, Any] = {}
            self._selection_operator = None

        def compute_minimum_eigenvalue(self, operator, aux_operators=None):
            self.selector_events = []
            self._exact_sparse_h_cache = {}
            self._selection_operator = operator
            result = super().compute_minimum_eigenvalue(
                operator,
                aux_operators=aux_operators,
            )
            result.selector_events = list(self.selector_events)
            result.timing_summary = summarize_fastadapt_timings(
                setup_timings=result.setup_timings,
                iteration_timings=result.iteration_timing_history,
                selector_events=self.selector_events,
            )
            solve_record = {
                "solve_index": len(type(self).solve_records) + 1,
                "selector_policy": selector_policy,
                "n_operators": len(self._selected_operator_indices),
                "selected_indices": [
                    int(idx) for idx in self._selected_operator_indices
                ],
                "selector_events": list(self.selector_events),
                "iteration_gradient_history": list(result.iteration_gradient_history),
                "iteration_timing_history": list(result.iteration_timing_history),
                "selection_diagnostic_history": list(
                    getattr(result, "selection_diagnostic_history", [])
                ),
                "setup_timings": dict(result.setup_timings),
                "timing_summary": dict(result.timing_summary),
                "final_fragment_energy": (
                    float(result.eigenvalue.real)
                    if getattr(result, "eigenvalue", None) is not None
                    else None
                ),
                "pool_provenance": dict(getattr(self, "pool_provenance", {}) or {}),
            }
            type(self).solve_records.append(solve_record)
            return result

        def _selection_decision(
            self,
            *,
            iteration: int,
            gradients: list[float],
            abs_gradients: list[float],
            sorted_candidates: list[int],
            prev_op_indices: list[int],
            theta: list[float],
        ) -> _SelectionDecision:
            base_decision = super()._selection_decision(
                iteration=iteration,
                gradients=gradients,
                abs_gradients=abs_gradients,
                sorted_candidates=sorted_candidates,
                prev_op_indices=prev_op_indices,
                theta=theta,
            )
            if selector_policy != "always_top5_energy":
                return base_decision
            if (
                base_decision.selected_idx is None
                or not self._uses_exact_sparse_evolution()
            ):
                return base_decision
            operator = self._selection_operator
            if operator is None:
                return base_decision

            shortlist, skipped_cyclic, termination_reason = (
                _build_exact_sparse_shortlist(
                    abs_gradients=abs_gradients,
                    sorted_candidates=sorted_candidates,
                    prev_op_indices=prev_op_indices,
                    gradient_threshold=float(self.gradient_threshold),
                    top_k=5,
                    check_cyclicity=bool(self.check_cyclicity),
                    cyclicity_fn=self._check_cyclicity,
                )
            )
            if not shortlist:
                return base_decision

            h_cache_key = id(operator)
            if h_cache_key not in self._exact_sparse_h_cache:
                self._exact_sparse_h_cache[h_cache_key] = operator.to_matrix(
                    sparse=True
                ).tocsr()
            h_sparse = self._exact_sparse_h_cache[h_cache_key]
            current_theta = _extract_current_theta(theta)
            event_t0 = perf_counter()
            probe_records = probe_exact_sparse_candidates(
                initial_state=self._initial_state,
                prefix_indices=prev_op_indices,
                candidate_indices=shortlist,
                excitation_pool=self._excitation_pool,
                theta_prefix=current_theta,
                operator=operator,
                h_sparse=h_sparse,
                maxiter=200,
                ftol=1e-10,
                abs_gradients=abs_gradients,
                sorted_candidates=sorted_candidates,
            )
            winner = _select_probe_winner(
                probe_records,
                energy_tie_tolerance=1e-10,
            )
            selected_idx = int(base_decision.selected_idx)
            selected_initial_point = (
                None
                if base_decision.initial_point is None
                else np.asarray(base_decision.initial_point, dtype=float).reshape(-1)
            )
            selected_source = "greedy_fallback"
            selected_rank = int(
                next(
                    rank + 1
                    for rank, idx in enumerate(sorted_candidates)
                    if idx == selected_idx
                )
            )
            if winner is not None:
                selected_idx = int(winner["candidate_idx"])
                selected_initial_point = np.asarray(
                    winner["optimal_point"],
                    dtype=float,
                ).reshape(-1)
                selected_source = "energy_probe"
                selected_rank = int(winner["gradient_rank"])

            event = {
                "iteration": int(iteration),
                "policy": selector_policy,
                "prefix_before": [int(idx) for idx in prev_op_indices],
                "theta_before": current_theta,
                "shortlist": probe_records,
                "shortlist_indices": [int(idx) for idx in shortlist],
                "shortlist_count": int(len(shortlist)),
                "skipped_cyclic": [int(idx) for idx in skipped_cyclic],
                "termination_reason_before_probe": termination_reason,
                "base_selected_idx": int(base_decision.selected_idx),
                "selected_idx": int(selected_idx),
                "selected_rank": int(selected_rank),
                "selected_source": selected_source,
                "total_probe_nfev": sum(
                    int(record.get("optimizer_nfev", 0) or 0)
                    for record in probe_records
                ),
                "total_probe_nit": sum(
                    int(record.get("optimizer_nit", 0) or 0) for record in probe_records
                ),
                "total_probe_elapsed_s": sum(
                    float(record.get("elapsed_s", 0.0) or 0.0)
                    for record in probe_records
                ),
                "selection_elapsed_s": float(perf_counter() - event_t0),
            }
            self.selector_events.append(event)
            return dataclass_replace(
                base_decision,
                selected_idx=int(selected_idx),
                initial_point=selected_initial_point,
                diagnostics=_merge_selection_diagnostics(
                    getattr(base_decision, "diagnostics", None),
                    event,
                ),
                guided_selection=(winner is not None),
            )

    RecordingAdapt.__name__ = f"H6Recording{base_class.__name__}"
    return RecordingAdapt


def _selector_factory_for_policy(
    policy: str,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Return a local selector factory plus its created class holder."""

    holder: dict[str, Any] = {}

    def factory(adapt_class, adapt_kwargs):
        recording_class = holder.get("class")
        if recording_class is None:
            recording_class = _make_recording_adapt_class(
                adapt_class,
                selector_policy=policy,
            )
            recording_class.solve_records = []
            holder["class"] = recording_class
        return recording_class(**adapt_kwargs)

    return factory, holder


def run_full_system_fci(mean_field: scf.hf.RHF) -> float:
    """Compute the full-system FCI energy for one spacing."""

    energy, _ = fci.FCI(mean_field).kernel()
    return float(energy)


def run_fci_be(mean_field: scf.hf.RHF) -> tuple[float, dict[str, Any]]:
    """Run the FCI-BE control cell for one geometry."""

    clear_ansatz_cache()
    reset_vqe_state()
    embedding = BE(mean_field, build_fragments(mean_field.mol))
    with _capture_be_optimization() as be_optimization:
        embedding.optimize(
            solver="FCI",
            conv_tol=1e-6,
            max_iter=BE_MAX_ITERATIONS,
            only_chem=False,
        )
    return float(embedding.ebe_tot), be_optimization


def _path_statistics(solve_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize operator-path and selector behavior across fragment solves."""

    selector_events = [
        event for solve in solve_records for event in solve.get("selector_events", [])
    ]
    rank_distribution = Counter(
        str(int(event["selected_rank"]))
        for event in selector_events
        if event.get("selected_rank") is not None
    )
    return {
        "fragment_solver_call_count": len(solve_records),
        "operator_counts": [
            int(solve.get("n_operators", 0)) for solve in solve_records
        ],
        "selected_rank_distribution": dict(sorted(rank_distribution.items())),
        "probe_count": sum(
            len(event.get("shortlist", [])) for event in selector_events
        ),
        "selector_event_count": len(selector_events),
        "probe_nfev_total": sum(
            int(event.get("total_probe_nfev", 0) or 0) for event in selector_events
        ),
        "probe_nit_total": sum(
            int(event.get("total_probe_nit", 0) or 0) for event in selector_events
        ),
        "probe_elapsed_s": sum(
            float(event.get("total_probe_elapsed_s", 0.0) or 0.0)
            for event in selector_events
        ),
        "selection_elapsed_s": sum(
            float(event.get("selection_elapsed_s", 0.0) or 0.0)
            for event in selector_events
        ),
    }


def run_case(
    mean_field: scf.hf.RHF,
    *,
    spacing: float,
    case_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one fresh CEO-manifest VQE case and persist its artifact."""

    selector_policy = CASE_POLICIES[case_name]
    clear_ansatz_cache()
    reset_vqe_state()
    case_dir = output_dir / f"spacing_{spacing:.1f}" / case_name
    hamiltonian_dir = case_dir / "hamiltonians"
    embedding = BE(mean_field, build_fragments(mean_field.mol))
    selector_factory, holder = _selector_factory_for_policy(selector_policy)
    started = perf_counter()
    with _capture_be_optimization() as be_optimization:
        with _exact_sparse_ceo_manifest_context(MANIFEST_REGISTRY):
            with _adapt_selector_factory_context(selector_factory):
                embedding.optimize(
                    solver="VQE",
                    solver_args=build_vqe_args(hamiltonian_dir),
                    conv_tol=1e-6,
                    max_iter=BE_MAX_ITERATIONS,
                    only_chem=False,
                )
    runtime_s = perf_counter() - started
    recording_class = holder.get("class")
    solve_records = (
        list(recording_class.solve_records) if recording_class is not None else []
    )
    result = {
        "case": case_name,
        "selector_policy": selector_policy,
        "spacing_angstrom": float(spacing),
        "energy_hartree": float(embedding.ebe_tot),
        "runtime_seconds": runtime_s,
        "be_optimization": be_optimization,
        "solve_records": solve_records,
        "path_statistics": _path_statistics(solve_records),
    }
    _write_json(_case_output_path(output_dir, spacing, case_name), result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the requested cells, and emit fresh results."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        output_path = args.output_dir / "result.json"
        _write_json(output_path, _plan_payload(args))
        print(f"H6 production dry-run plan written to {output_path}")
        return 0

    include_fci_be = not args.skip_fci_be
    manifest_summary = _manifest_summary()
    aggregate_results: dict[str, dict[str, Any]] = {}
    for spacing in args.spacings:
        spacing_key = f"{spacing:.1f}"
        mean_field = build_mean_field(spacing)
        full_system_fci = run_full_system_fci(mean_field)
        spacing_results: dict[str, Any] = {
            "full_system_fci": {
                "energy_hartree": full_system_fci,
            }
        }
        if include_fci_be:
            fci_be_energy, fci_be_opt = run_fci_be(mean_field)
            fci_be_result = {
                "case": "fci_be",
                "spacing_angstrom": float(spacing),
                "energy_hartree": fci_be_energy,
                "runtime_seconds": fci_be_opt.get("optimize_wall_s"),
                "be_optimization": fci_be_opt,
            }
            spacing_results["fci_be"] = fci_be_result
            _write_json(
                _case_output_path(args.output_dir, spacing, "fci_be"), fci_be_result
            )
        for case_name in ("greedy", "lookahead"):
            case_result = run_case(
                mean_field,
                spacing=spacing,
                case_name=case_name,
                output_dir=args.output_dir,
            )
            spacing_results[case_name] = case_result
        aggregate_results[spacing_key] = spacing_results

    payload = {
        "status": "COMPLETED",
        "project_root": str(PROJECT_ROOT),
        "manifest": manifest_summary,
        "results": aggregate_results,
    }
    output_path = args.output_dir / "result.json"
    _write_json(output_path, payload)
    print("H6 CEO-manifest QBE-VQE run completed")
    print(f"Result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
