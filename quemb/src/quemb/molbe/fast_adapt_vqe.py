"""
Fast ADAPT-VQE implementation with direct statevector gradient computation.

Usage:
    from quemb.molbe.fast_adapt_vqe import FastAdaptVQE
    fast_adapt = FastAdaptVQE(
        solver=base_vqe,
        operators=operator_pool,
        initial_state=hf_circuit,
        evolution=LieTrotter(),
        gradient_threshold=1e-3,
        max_iterations=30,
    )
    result = fast_adapt.compute_minimum_eigenvalue(hamiltonian)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import evolved_operator_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_algorithms.minimum_eigensolvers.adapt_vqe import (
    AdaptVQEResult,
    TerminationCriterion,
)
from qiskit_algorithms.minimum_eigensolvers.vqe import VQE
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse.linalg import expm_multiply


class _DirectSVResult:
    """Minimal result object compatible with QiskitVQE result expectations."""

    def __init__(self, eigenvalue, optimal_point, cost_function_evals):
        self.eigenvalue = eigenvalue
        self.optimal_point = optimal_point
        self.optimal_value = eigenvalue
        self.cost_function_evals = cost_function_evals
        self.optimal_parameters = None
        self.aux_operators_evaluated = None


@dataclass(slots=True)
class _SelectionDecision:
    """Internal selection hook payload shared by ADAPT backends."""

    selected_idx: int | None = None
    initial_point: Sequence[float] | np.ndarray | None = None
    diagnostics: dict[str, object] | None = None
    termination_reason: str | None = None
    termination_criterion: TerminationCriterion | None = None
    guided_selection: bool = False


class DirectSVSolver:
    """Lightweight VQE solver using direct Statevector + sparse H evaluation.

    Replaces Qiskit's QiskitVQE for the inner optimization loop in FastAdaptVQE.
    Uses scipy.optimize.minimize(SLSQP) with the energy computed as:

        E = <psi|H|psi> = vdot(psi, H_sparse @ psi)

    This avoids Qiskit estimator overhead (Pauli decomposition, circuit
    compilation) and is typically 1.5-2x faster for 12-16 qubit fragments.
    """

    def __init__(
        self,
        *,
        optimizer_maxiter: int = 200,
        optimizer_ftol: float = 1e-10,
        ansatz: QuantumCircuit | None = None,
        initial_point: np.ndarray | None = None,
        verbose: int = 0,
        use_exact_jacobian: bool = True,
    ) -> None:
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.optimizer_maxiter = optimizer_maxiter
        self.optimizer_ftol = optimizer_ftol
        self.verbose = verbose
        self.use_exact_jacobian = use_exact_jacobian
        self._h_sparse_cache = None
        self.last_timing_breakdown: dict | None = None
        self.timing_history: list[dict] = []
        self._reverse_gradient = ReverseEstimatorGradient()
        from qiskit.primitives import StatevectorEstimator

        self.estimator = StatevectorEstimator()

    def compute_minimum_eigenvalue(self, operator, aux_operators=None):
        """Run SLSQP optimization with direct statevector energy evaluation."""
        t0 = perf_counter()

        if self._h_sparse_cache is None:
            self._h_sparse_cache = operator.to_matrix(sparse=True)
        h_sparse = self._h_sparse_cache

        ansatz = self.ansatz
        param_list = list(ansatz.parameters)
        n_params = len(param_list)
        if self.initial_point is None:
            x0 = np.zeros(n_params, dtype=float)
        else:
            raw_x0 = np.asarray(self.initial_point, dtype=float).ravel()
            x0 = np.zeros(n_params, dtype=float)
            x0[: min(n_params, raw_x0.size)] = raw_x0[:n_params]

        nfev_counter = [0]
        njev_counter = [0]
        parameter_bind_total_s = 0.0
        statevector_total_s = 0.0
        expectation_total_s = 0.0
        cost_function_total_s = 0.0
        jacobian_total_s = 0.0
        cached_grad: dict[str, np.ndarray | tuple[float, ...] | None] = {
            "theta": None,
            "value": None,
        }

        def objective(theta):
            nonlocal parameter_bind_total_s, statevector_total_s
            nonlocal expectation_total_s, cost_function_total_s
            eval_t0 = perf_counter()
            nfev_counter[0] += 1
            bind_t0 = perf_counter()
            bound = ansatz.assign_parameters(dict(zip(param_list, theta)))
            parameter_bind_total_s += perf_counter() - bind_t0
            sv_t0 = perf_counter()
            psi = Statevector(bound).data
            statevector_total_s += perf_counter() - sv_t0
            expect_t0 = perf_counter()
            energy = np.vdot(psi, h_sparse @ psi).real
            expectation_total_s += perf_counter() - expect_t0
            cost_function_total_s += perf_counter() - eval_t0
            return energy

        def exact_jacobian(theta):
            nonlocal jacobian_total_s
            theta_key = tuple(float(x) for x in np.asarray(theta, dtype=float))
            if cached_grad["theta"] == theta_key and cached_grad["value"] is not None:
                return np.asarray(cached_grad["value"], dtype=float)

            njev_counter[0] += 1
            jac_t0 = perf_counter()
            job = self._reverse_gradient.run(
                [ansatz],
                [operator],
                [list(theta_key)],
                [param_list],
            )
            gradient = np.asarray(job.result().gradients[0], dtype=float)
            jacobian_total_s += perf_counter() - jac_t0
            cached_grad["theta"] = theta_key
            cached_grad["value"] = gradient.copy()
            return gradient

        jacobian_fn = None
        if n_params > 0 and self.use_exact_jacobian:
            try:
                exact_jacobian(x0)
                jacobian_fn = exact_jacobian
            except Exception as exc:
                if self.verbose >= 1:
                    print(
                        "    [DirectSVSolver] ReverseEstimatorGradient unavailable, "
                        f"falling back to objective-only SLSQP: {exc}"
                    )
                cached_grad["theta"] = None
                cached_grad["value"] = None
                jacobian_total_s = 0.0
                njev_counter[0] = 0

        optimizer_result = None
        if n_params == 0:
            energy = objective(np.array([], dtype=float))
            result = _DirectSVResult(
                eigenvalue=energy,
                optimal_point=np.array([], dtype=float),
                cost_function_evals=nfev_counter[0],
            )
            result.optimal_circuit = ansatz
        else:
            optimizer_result = scipy_minimize(
                objective,
                x0,
                method="SLSQP",
                jac=jacobian_fn,
                options={
                    "maxiter": self.optimizer_maxiter,
                    "ftol": self.optimizer_ftol,
                },
            )
            result = _DirectSVResult(
                eigenvalue=optimizer_result.fun,
                optimal_point=optimizer_result.x,
                cost_function_evals=int(
                    getattr(optimizer_result, "nfev", nfev_counter[0])
                ),
            )
            result.optimal_circuit = ansatz.assign_parameters(
                dict(zip(param_list, optimizer_result.x))
            )
        result.optimizer_result = optimizer_result

        elapsed = perf_counter() - t0
        self.last_timing_breakdown = {
            "total_s": elapsed,
            "nfev": int(
                getattr(
                    getattr(result, "optimizer_result", None), "nfev", nfev_counter[0]
                )
            ),
            "njev": int(
                getattr(
                    getattr(result, "optimizer_result", None), "njev", njev_counter[0]
                )
            ),
            "backend": "direct_sv",
            "used_exact_jacobian": bool(jacobian_fn is not None),
            "gradient_backend": (
                "reverse_estimator" if jacobian_fn is not None else "none"
            ),
            "optimizer_success": bool(getattr(optimizer_result, "success", True)),
            "optimizer_status": int(getattr(optimizer_result, "status", 0)),
            "optimizer_message": str(
                getattr(optimizer_result, "message", "no_parameters")
            ),
            "optimizer_nit": int(getattr(optimizer_result, "nit", 0)),
            "cost_function_total_s": float(cost_function_total_s),
            "jacobian_total_s": float(jacobian_total_s),
            "parameter_bind_total_s": float(parameter_bind_total_s),
            "statevector_total_s": float(statevector_total_s),
            "expectation_total_s": float(expectation_total_s),
            "scipy_overhead_s": float(
                max(0.0, elapsed - cost_function_total_s - jacobian_total_s)
            ),
        }
        self.timing_history.append(self.last_timing_breakdown)

        return result


class ExactSparseSVSolver:
    r"""Exact selected-operator statevector solver for ADAPT-VQE.

    Unlike :class:`DirectSVSolver`, this solver does not build or simulate a
    Qiskit evolved-operator circuit.  It applies the selected Hermitian
    generators sequentially,

    .. math::

        |\psi(\theta)\rangle =
        \prod_j \exp(-i\theta_j G_j)|\mathrm{HF}\rangle,

    with :func:`scipy.sparse.linalg.expm_multiply`.  The class deliberately
    uses objective-only SLSQP in the first implementation slice: Qiskit's
    reverse circuit gradient is not the derivative of this custom exact
    evolution path.
    """

    uses_exact_sparse_evolution = True

    def __init__(
        self,
        *,
        initial_state: QuantumCircuit | np.ndarray,
        optimizer_maxiter: int = 200,
        optimizer_ftol: float = 1e-10,
        initial_point: np.ndarray | None = None,
        verbose: int = 0,
    ) -> None:
        if isinstance(initial_state, QuantumCircuit):
            initial_vector = Statevector(initial_state).data
            self.ansatz: QuantumCircuit | None = initial_state
        else:
            initial_vector = np.asarray(initial_state, dtype=np.complex128)
            self.ansatz = None

        initial_vector = np.asarray(initial_vector, dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(initial_vector))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("initial_state must be a finite, nonzero statevector")

        self._initial_statevector = initial_vector / norm
        self.initial_point = initial_point
        self.optimizer_maxiter = int(optimizer_maxiter)
        self.optimizer_ftol = float(optimizer_ftol)
        self.verbose = int(verbose)
        self._operator_sequence: list[BaseOperator] = []
        self._operator_matrix_cache: dict[int, tuple[BaseOperator, object]] = {}
        self._operator_matrices: list[object] = []
        self._h_sparse_cache = None
        self._h_operator_cache = None
        self.last_statevector = self._initial_statevector.copy()
        self.last_timing_breakdown: dict[str, float | int | str | bool] | None = None
        self.timing_history: list[dict[str, float | int | str | bool]] = []

        from qiskit.primitives import StatevectorEstimator

        self.estimator = StatevectorEstimator()

    def set_operator_sequence(self, operators: Sequence[BaseOperator]) -> None:
        """Set the ordered Hermitian generators used by the exact ansatz."""
        self._operator_sequence = list(operators)
        matrices: list[object] = []
        for operator in self._operator_sequence:
            cache_key = id(operator)
            cached = self._operator_matrix_cache.get(cache_key)
            if cached is None or cached[0] is not operator:
                matrix = operator.to_matrix(sparse=True).tocsr()
                self._operator_matrix_cache[cache_key] = (
                    operator,
                    matrix,
                )
            else:
                matrix = cached[1]
            matrices.append(matrix)
        self._operator_matrices = matrices

    def statevector_for(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the exact state for the current ordered generator sequence."""
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != len(self._operator_matrices):
            raise ValueError(
                "Exact sparse ansatz parameter count mismatch: "
                f"got {values.size}, expected {len(self._operator_matrices)}"
            )

        state = self._initial_statevector.copy()
        for angle, generator in zip(values, self._operator_matrices):
            state = expm_multiply((-1j * float(angle)) * generator, state)
        return np.asarray(state, dtype=np.complex128)

    def compute_minimum_eigenvalue(self, operator, aux_operators=None):
        """Optimize the current exact sparse ansatz with objective-only SLSQP."""
        del aux_operators
        t0 = perf_counter()

        if self._h_sparse_cache is None or self._h_operator_cache is not operator:
            self._h_sparse_cache = operator.to_matrix(sparse=True).tocsr()
            self._h_operator_cache = operator
        h_sparse = self._h_sparse_cache

        n_params = len(self._operator_matrices)
        if self.initial_point is None:
            x0 = np.zeros(n_params, dtype=float)
        else:
            raw_x0 = np.asarray(self.initial_point, dtype=float).reshape(-1)
            x0 = np.zeros(n_params, dtype=float)
            x0[: min(n_params, raw_x0.size)] = raw_x0[:n_params]

        nfev_counter = 0
        evolution_total_s = 0.0
        expectation_total_s = 0.0

        def objective(theta):
            nonlocal nfev_counter, evolution_total_s, expectation_total_s
            nfev_counter += 1
            evolution_start = perf_counter()
            state = self.statevector_for(theta)
            evolution_total_s += perf_counter() - evolution_start
            expectation_start = perf_counter()
            energy = float(np.vdot(state, h_sparse @ state).real)
            expectation_total_s += perf_counter() - expectation_start
            return energy

        if n_params == 0:
            optimal_point = np.array([], dtype=float)
            eigenvalue = objective(optimal_point)
            optimizer_result = None
        else:
            optimizer_result = scipy_minimize(
                objective,
                x0,
                method="SLSQP",
                jac=None,
                options={
                    "maxiter": self.optimizer_maxiter,
                    "ftol": self.optimizer_ftol,
                },
            )
            optimal_point = np.asarray(optimizer_result.x, dtype=float)
            eigenvalue = float(optimizer_result.fun)

        self.last_statevector = self.statevector_for(optimal_point)
        result = _DirectSVResult(
            eigenvalue=eigenvalue,
            optimal_point=optimal_point,
            cost_function_evals=int(
                getattr(optimizer_result, "nfev", nfev_counter)
                if optimizer_result is not None
                else nfev_counter
            ),
        )
        result.optimal_circuit = None
        result.optimal_state = self.last_statevector.copy()
        result.optimizer_result = optimizer_result

        elapsed = perf_counter() - t0
        self.last_timing_breakdown = {
            "total_s": float(elapsed),
            "nfev": int(result.cost_function_evals),
            "njev": 0,
            "backend": "exact_sparse",
            "used_exact_jacobian": False,
            "gradient_backend": "none",
            "optimizer_success": bool(getattr(optimizer_result, "success", True)),
            "optimizer_status": int(getattr(optimizer_result, "status", 0)),
            "optimizer_message": str(
                getattr(optimizer_result, "message", "no_parameters")
            ),
            "optimizer_nit": int(getattr(optimizer_result, "nit", 0)),
            "evolution_total_s": float(evolution_total_s),
            "expectation_total_s": float(expectation_total_s),
            "scipy_overhead_s": float(
                max(0.0, elapsed - evolution_total_s - expectation_total_s)
            ),
        }
        self.timing_history.append(dict(self.last_timing_breakdown))
        return result


def summarize_fastadapt_timings(
    *,
    setup_timings: dict[str, float],
    iteration_timings: list[dict[str, object]],
    selector_events: list[dict[str, object]] | None = None,
) -> dict[str, float | int | str | bool]:
    """Aggregate a fragment solve into a compact timing summary."""
    total_gradient = sum(
        float(item.get("gradient_time_s", 0.0)) for item in iteration_timings
    )
    total_selection = sum(
        float(item.get("selection_time_s", 0.0)) for item in iteration_timings
    )
    total_vqe = sum(float(item.get("vqe_time_s", 0.0)) for item in iteration_timings)
    total_setup_ham = float(setup_timings.get("hamiltonian_to_sparse_s", 0.0))
    total_setup_pool = float(setup_timings.get("pool_to_sparse_s", 0.0))
    total_probe = 0.0
    total_probe_work = 0.0
    total_probe_nfev = 0
    total_probe_nit = 0
    total_probe_calls = 0
    total_shortlist_candidates = 0
    total_refined_candidates = 0
    total_coarse_scan_evals = 0
    n_selector_events = len(selector_events) if selector_events else 0
    if selector_events:
        total_probe = sum(
            float(
                event.get(
                    "total_probe_wall_s",
                    event.get(
                        "total_probe_elapsed_s",
                        event.get("full_probe_elapsed_sum_s", 0.0),
                    ),
                )
            )
            for event in selector_events
        )
        total_probe_work = sum(
            float(
                event.get(
                    "total_probe_elapsed_s",
                    event.get("full_probe_elapsed_sum_s", 0.0),
                )
            )
            for event in selector_events
        )
        total_probe_nfev = sum(
            int(event.get("total_probe_nfev", 0) or 0) for event in selector_events
        )
        total_probe_nit = sum(
            int(event.get("total_probe_nit", 0) or 0) for event in selector_events
        )
        total_probe_calls = sum(
            int(
                event.get(
                    "full_probe_call_count",
                    event.get("refined_candidate_count", 0),
                )
                or 0
            )
            for event in selector_events
        )
        total_shortlist_candidates = sum(
            int(
                event.get(
                    "shortlist_count",
                    (
                        len(event.get("shortlist", []))
                        if isinstance(event.get("shortlist", []), list)
                        else 0
                    ),
                )
                or 0
            )
            for event in selector_events
        )
        total_refined_candidates = sum(
            int(event.get("refined_candidate_count", 0) or 0)
            for event in selector_events
        )
        total_coarse_scan_evals = sum(
            int(event.get("coarse_scan_n_evals", 0) or 0) for event in selector_events
        )

    accounted_phases = {
        "hamiltonian_to_sparse": total_setup_ham,
        "pool_to_sparse": total_setup_pool,
        "gradient_eval": total_gradient,
        "selection": total_selection,
        "inner_vqe": total_vqe,
        "probe": total_probe,
    }
    dominant_phase, dominant_time = max(
        accounted_phases.items(), key=lambda item: item[1]
    )
    n_iterations = len(iteration_timings)
    n_vqe_steps = sum(
        1 for item in iteration_timings if float(item.get("vqe_time_s", 0.0)) > 0.0
    )
    total_main_vqe_nfev = sum(
        int(item.get("vqe_cost_function_evals", 0) or 0) for item in iteration_timings
    )
    optimizer_reports = [
        item["inner_vqe_breakdown"]
        for item in iteration_timings
        if isinstance(item.get("inner_vqe_breakdown"), dict)
        and "optimizer_success" in item["inner_vqe_breakdown"]
    ]
    optimizer_failure_count = sum(
        not bool(report["optimizer_success"]) for report in optimizer_reports
    )
    total_vqe_like_calls = int(n_vqe_steps + total_probe_calls)

    return {
        "hamiltonian_to_sparse_s": total_setup_ham,
        "pool_to_sparse_s": total_setup_pool,
        "gradient_eval_total_s": total_gradient,
        "selection_total_s": total_selection,
        "inner_vqe_total_s": total_vqe,
        "probe_total_s": total_probe,
        "probe_work_total_s": total_probe_work,
        "total_accounted_s": sum(accounted_phases.values()),
        "n_iterations": int(n_iterations),
        "n_vqe_steps": int(n_vqe_steps),
        "n_selector_events": int(n_selector_events),
        "main_vqe_step_count": int(n_vqe_steps),
        "main_vqe_nfev_total": int(total_main_vqe_nfev),
        "main_vqe_optimizer_report_count": len(optimizer_reports),
        "main_vqe_optimizer_failure_count": int(optimizer_failure_count),
        "main_vqe_optimizers_all_successful": (
            optimizer_failure_count == 0 and len(optimizer_reports) == n_vqe_steps
        ),
        "probe_call_count": int(total_probe_calls),
        "probe_nfev_total": int(total_probe_nfev),
        "probe_nit_total": int(total_probe_nit),
        "shortlist_candidate_count": int(total_shortlist_candidates),
        "refined_candidate_count": int(total_refined_candidates),
        "coarse_scan_n_evals_total": int(total_coarse_scan_evals),
        "vqe_like_call_count": int(total_vqe_like_calls),
        "avg_gradient_eval_s": (total_gradient / n_iterations) if n_iterations else 0.0,
        "avg_selection_s": (total_selection / n_iterations) if n_iterations else 0.0,
        "avg_inner_vqe_s": (total_vqe / n_vqe_steps) if n_vqe_steps else 0.0,
        "avg_probe_elapsed_s": (
            (total_probe / n_selector_events) if n_selector_events else 0.0
        ),
        "avg_probe_work_s": (
            (total_probe_work / n_selector_events) if n_selector_events else 0.0
        ),
        "avg_probe_calls_per_selector": (
            (total_probe_calls / n_selector_events) if n_selector_events else 0.0
        ),
        "max_inner_vqe_s": max(
            (float(item.get("vqe_time_s", 0.0)) for item in iteration_timings),
            default=0.0,
        ),
        "dominant_phase": dominant_phase,
        "dominant_phase_s": float(dominant_time),
    }


class FastAdaptVQEResult(AdaptVQEResult):
    """AdaptVQEResult extended with FastAdaptVQE timing and gradient diagnostics.

    Avoids monkey-patching dynamic attributes onto the upstream
    ``AdaptVQEResult`` by declaring them as typed instance attributes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.iteration_gradient_history: list[dict[str, object]] = []
        self.iteration_timing_history: list[dict[str, object]] = []
        self.selection_diagnostic_history: list[dict[str, object]] = []
        self.setup_timings: dict[str, float] = {}
        self.timing_summary: dict[str, float | int | str | bool] = {}


class FastAdaptVQE:
    """ADAPT-VQE with direct statevector gradient computation.

    Instead of building symbolic commutators [H, A_i] as SparsePauliOp products
    (which scales as O(n_ham_terms * n_op_terms) per operator), this computes
    gradients via sparse matrix-vector products:

        grad_i = <psi| [H, A_i] |psi>

    where H and A_i are converted to scipy sparse matrices once, and gradients
    are computed as:

        hpsi = H @ psi
        apsi_i = A_i @ psi
        a_hpsi_i = A_i @ hpsi
        grad_i = 1j * (vdot(hpsi, apsi_i) - vdot(psi, a_hpsi_i))

    Each step is O(nnz * 2^n) instead of O(n_terms^2) symbolic algebra.

    Parameters
    ----------
    solver : VQE
        Qiskit VQE instance for inner optimization.
    gradient_threshold : float
        Convergence threshold for max gradient.
    eigenvalue_threshold : float
        Convergence threshold for eigenvalue change.
    max_iterations : int or None
        Maximum ADAPT iterations.
    operators : list of SparsePauliOp
        Operator pool (UCC singles+doubles).
    initial_state : QuantumCircuit
        HF initial state circuit.
    evolution : object
        Evolution synthesis (e.g. LieTrotter).
    flatten : bool
        Whether to flatten evolved operator ansatz.
    verbose : int
        Verbosity level.
    gradient_tie_tolerance : float
        Absolute tolerance used to register equal-magnitude gradients.
        Candidates inside one registered tie are ranked by pool index so
        floating-point reduction order cannot change the selected operator.
    record_full_gradient_vector : bool
        Persist every pool gradient in each iteration diagnostic. Disabled by
        default because large production pools can make result objects large.
    """

    def __init__(
        self,
        solver: VQE,
        *,
        gradient_threshold: float = 1e-3,
        eigenvalue_threshold: float = 1e-5,
        max_iterations: int | None = 20,
        operators: Sequence[BaseOperator],
        initial_state: QuantumCircuit,
        evolution=None,
        flatten: bool = True,
        verbose: int = 0,
        check_cyclicity: bool = True,
        cyclicity_action: str = "skip",
        gradient_log_top_k: int = 5,
        tracked_operator_indices: Sequence[int] = (),
        gradient_tie_tolerance: float = 1e-10,
        record_full_gradient_vector: bool = False,
    ) -> None:
        self.solver = solver
        self.gradient_threshold = gradient_threshold
        self.eigenvalue_threshold = eigenvalue_threshold
        self.max_iterations = max_iterations
        self._excitation_pool: list[BaseOperator] = list(operators)
        self._initial_state = initial_state
        self._evolution = evolution
        self._flatten = flatten
        self.verbose = verbose
        self._excitation_list: list[BaseOperator] = []
        self._selected_operator_indices: list[int] = []
        self.check_cyclicity = check_cyclicity
        self.cyclicity_action = cyclicity_action
        self.gradient_log_top_k = max(0, int(gradient_log_top_k))
        self.gradient_tie_tolerance = float(gradient_tie_tolerance)
        if self.gradient_tie_tolerance < 0.0:
            raise ValueError("gradient_tie_tolerance must be non-negative")
        self.record_full_gradient_vector = bool(record_full_gradient_vector)
        self.tracked_operator_indices = tuple(
            int(idx) for idx in tracked_operator_indices
        )
        if any(
            idx < 0 or idx >= len(self._excitation_pool)
            for idx in self.tracked_operator_indices
        ):
            raise ValueError(
                "tracked_operator_indices must reference valid excitation-pool indices"
            )
        self._iteration_gradient_history: list[dict[str, object]] = []
        self._iteration_timing_history: list[dict[str, object]] = []
        self._selection_diagnostic_history: list[dict[str, object]] = []

    @staticmethod
    def _get_statevector(
        circuit: QuantumCircuit, theta: list[float] | None
    ) -> np.ndarray:
        """Get statevector numpy array from a (possibly parameterized) circuit."""
        if theta and circuit.num_parameters > 0:
            bound = circuit.assign_parameters(dict(zip(circuit.parameters, theta)))
            return Statevector(bound).data
        return Statevector(circuit).data

    def _uses_exact_sparse_evolution(self) -> bool:
        """Whether the inner solver owns an exact sparse operator sequence."""
        return bool(
            getattr(self.solver, "uses_exact_sparse_evolution", False)
            and hasattr(self.solver, "set_operator_sequence")
            and hasattr(self.solver, "statevector_for")
        )

    def _set_selected_operator_sequence(self) -> None:
        """Update either the exact-state solver or the legacy circuit ansatz."""
        if self._uses_exact_sparse_evolution():
            self.solver.set_operator_sequence(self._excitation_list)
            return
        self.solver.ansatz = self._build_ansatz()

    def _get_current_statevector(self, theta: Sequence[float]) -> np.ndarray:
        """Evaluate the selected ansatz using its configured evolution semantics."""
        if self._uses_exact_sparse_evolution():
            return self.solver.statevector_for(theta)
        return self._get_statevector(
            self.solver.ansatz,
            list(theta) if len(theta) else None,
        )

    def _compute_gradients_fast(
        self,
        psi: np.ndarray,
        H_sparse,
        op_matrices: list,
    ) -> list[float]:
        """Compute all ADAPT gradients via sparse matrix-vector products.

        For each pool operator A_i, the ADAPT gradient is:

            grad_i = <psi| i*[H, A_i] |psi>
                   = i * (<psi|H A_i|psi> - <psi|A_i H|psi>)
                   = i * (hpsi^dag @ apsi_i - psi^dag @ A_i @ hpsi)

        where hpsi = H|psi> is computed once and reused.

        Returns list of real gradient values matching Qiskit's convention.
        """
        hpsi = H_sparse @ psi

        gradients = []
        for A_mat in op_matrices:
            apsi = A_mat @ psi
            a_hpsi = A_mat @ hpsi

            term1 = np.vdot(hpsi, apsi)
            term2 = np.vdot(psi, a_hpsi)

            prefactor = -1j if self._uses_exact_sparse_evolution() else 1j
            grad = prefactor * (term1 - term2)
            gradients.append(grad.real)

        return gradients

    @staticmethod
    def _check_cyclicity(indices: list[int]) -> bool:
        """Check for repeating sequences in operator selection indices."""
        if len(indices) > 1 and indices[-2] == indices[-1]:
            return True
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        return cycle_regex.search(" ".join(map(str, indices))) is not None

    def _guided_selection(
        self,
        *,
        iteration: int,
        gradients: list[float],
        abs_gradients: list[float],
        sorted_candidates: list[int],
        prev_op_indices: list[int],
    ) -> int | None:
        """Optionally override the next operator choice.

        Subclasses can return an excitation-pool index to force a specific
        operator choice for an experiment. Returning ``None`` preserves the
        standard ADAPT ranking + cyclicity behavior.
        """
        return None

    def _selection_log_prefix(self) -> str:
        """Prefix used by the shared selection helper for verbose logs."""
        return "FastADAPT"

    @staticmethod
    def _serialize_selection_payload(value: object) -> object:
        """Convert selection-hook diagnostics into plain Python containers."""
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, (complex, np.complexfloating)):
            return {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
            }
        if isinstance(value, np.ndarray):
            return [
                FastAdaptVQE._serialize_selection_payload(item)
                for item in value.tolist()
            ]
        if isinstance(value, dict):
            return {
                str(key): FastAdaptVQE._serialize_selection_payload(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [FastAdaptVQE._serialize_selection_payload(item) for item in value]
        return str(value)

    def _default_selection_decision(
        self,
        *,
        iteration: int,
        gradients: list[float],
        abs_gradients: list[float],
        sorted_candidates: list[int],
        prev_op_indices: list[int],
    ) -> _SelectionDecision:
        """Legacy guided-selection + greedy threshold/cyclicity fallback."""
        guided_idx = self._guided_selection(
            iteration=iteration,
            gradients=gradients,
            abs_gradients=abs_gradients,
            sorted_candidates=sorted_candidates,
            prev_op_indices=prev_op_indices.copy(),
        )
        if guided_idx is not None:
            guided_idx = int(guided_idx)
            return _SelectionDecision(
                selected_idx=guided_idx,
                guided_selection=True,
            )

        termination_reason = None
        selected_idx = None
        for candidate_idx in sorted_candidates:
            candidate_grad = abs_gradients[candidate_idx]

            if candidate_grad < self.gradient_threshold:
                if iteration == 1 and not self._uses_exact_sparse_evolution():
                    raise AlgorithmError(
                        "All gradients below threshold in first iteration. "
                        "Try tighter threshold or different operator pool."
                    )
                termination_reason = "converged"
                break

            if self.check_cyclicity:
                trial_indices = prev_op_indices + [candidate_idx]
                if self._check_cyclicity(trial_indices):
                    if self.cyclicity_action == "stop":
                        if self.verbose >= 1:
                            print(
                                f"  [{self._selection_log_prefix()}] CYCLICITY "
                                f"at idx={candidate_idx}, stopping"
                            )
                        termination_reason = "cyclicity"
                        break
                    if self.verbose >= 1:
                        print(
                            f"  [{self._selection_log_prefix()}] CYCLICITY at "
                            f"idx={candidate_idx} (|grad|={candidate_grad:.6e}), "
                            "skipping"
                        )
                    continue

            selected_idx = candidate_idx
            break

        return _SelectionDecision(
            selected_idx=selected_idx,
            termination_reason=termination_reason,
        )

    def _selection_decision(
        self,
        *,
        iteration: int,
        gradients: list[float],
        abs_gradients: list[float],
        sorted_candidates: list[int],
        prev_op_indices: list[int],
        theta: Sequence[float],
    ) -> _SelectionDecision:
        """Protected selection seam for lookahead-style overrides."""
        del theta
        return self._default_selection_decision(
            iteration=iteration,
            gradients=gradients,
            abs_gradients=abs_gradients,
            sorted_candidates=sorted_candidates,
            prev_op_indices=prev_op_indices,
        )

    def _normalize_selection_decision(
        self,
        decision: _SelectionDecision | int | np.integer,
        *,
        theta: Sequence[float],
        pool_size: int,
    ) -> _SelectionDecision:
        """Validate hook output and normalize optional probe metadata."""
        if isinstance(decision, (int, np.integer)):
            normalized = _SelectionDecision(selected_idx=int(decision))
        elif isinstance(decision, _SelectionDecision):
            normalized = decision
        else:
            raise AlgorithmError(
                "Selection hook must return _SelectionDecision or an integer "
                f"pool index, got {type(decision).__name__}"
            )

        selected_idx = normalized.selected_idx
        if selected_idx is not None:
            selected_idx = int(selected_idx)
            if not 0 <= selected_idx < pool_size:
                raise AlgorithmError(
                    "Selection hook returned invalid excitation index "
                    f"{selected_idx}"
                )

        if selected_idx is not None and (
            normalized.termination_reason is not None
            or normalized.termination_criterion is not None
        ):
            raise AlgorithmError(
                "Selection hook cannot both select an operator and terminate"
            )

        if selected_idx is None:
            if normalized.termination_reason is None:
                if normalized.termination_criterion == TerminationCriterion.CONVERGED:
                    termination_reason = "converged"
                elif normalized.termination_criterion is not None:
                    termination_reason = "cyclicity"
                else:
                    raise AlgorithmError(
                        "Selection hook must select an operator or provide a "
                        "handled termination"
                    )
            else:
                termination_reason = str(normalized.termination_reason)
            termination_criterion = (
                normalized.termination_criterion
                if normalized.termination_criterion is not None
                else (
                    TerminationCriterion.CONVERGED
                    if termination_reason == "converged"
                    else TerminationCriterion.CYCLICITY
                )
            )
        else:
            termination_reason = None
            termination_criterion = None

        initial_point = None
        if normalized.initial_point is not None:
            if selected_idx is None:
                raise AlgorithmError(
                    "Selection hook provided an initial point without "
                    "selecting an operator"
                )
            initial_point = np.asarray(
                normalized.initial_point,
                dtype=float,
            ).reshape(-1)
            expected_size = len(theta) + 1
            if initial_point.size != expected_size:
                raise AlgorithmError(
                    "Selection hook returned an initial point with invalid "
                    f"shape: got {initial_point.size}, expected {expected_size}"
                )

        diagnostics = None
        if normalized.diagnostics is not None:
            serialized = self._serialize_selection_payload(normalized.diagnostics)
            if not isinstance(serialized, dict):
                raise AlgorithmError(
                    "Selection hook diagnostics must serialize to a dictionary"
                )
            diagnostics = dict(serialized)

        return _SelectionDecision(
            selected_idx=selected_idx,
            initial_point=initial_point,
            diagnostics=diagnostics,
            termination_reason=termination_reason,
            termination_criterion=termination_criterion,
            guided_selection=bool(normalized.guided_selection),
        )

    def _record_selection_decision(
        self,
        *,
        iteration: int,
        iteration_timing: dict[str, object],
        iteration_diagnostic: dict[str, object],
        decision: _SelectionDecision,
    ) -> None:
        """Persist serializable selection metadata onto result histories."""
        iteration_timing["guided_selection"] = bool(decision.guided_selection)
        if decision.initial_point is not None:
            iteration_timing["selection_initial_point"] = [
                float(value) for value in decision.initial_point.tolist()
            ]
        if decision.diagnostics is not None:
            selection_diagnostics = dict(decision.diagnostics)
            iteration_timing["selection_diagnostics"] = selection_diagnostics
            iteration_diagnostic["selection_diagnostics"] = selection_diagnostics
            self._selection_diagnostic_history.append(
                {
                    "iteration": int(iteration),
                    "diagnostics": selection_diagnostics,
                }
            )

    def _rank_candidates_with_ties(
        self,
        abs_gradients: Sequence[float],
    ) -> list[int]:
        """Rank gradients deterministically inside registered tie groups.

        Backends can differ by a few last-place bits even when two gradients
        are physically tied. Ranking directly by those values makes ADAPT pick
        a different operator on each backend. Preserve descending-gradient
        ordering between non-tied groups, but use the stable pool index inside
        every group whose values are within ``gradient_tie_tolerance`` of that
        group's maximum.
        """
        by_value = sorted(
            range(len(abs_gradients)),
            key=lambda idx: (-float(abs_gradients[idx]), idx),
        )
        ranked: list[int] = []
        group_start = 0
        while group_start < len(by_value):
            group_max = float(abs_gradients[by_value[group_start]])
            group_end = group_start + 1
            while group_end < len(by_value):
                candidate = by_value[group_end]
                if (
                    group_max - float(abs_gradients[candidate])
                    > self.gradient_tie_tolerance
                ):
                    break
                group_end += 1
            ranked.extend(sorted(by_value[group_start:group_end]))
            group_start = group_end
        return ranked

    def _after_vqe_iteration(
        self,
        *,
        iteration: int,
        operator: SparsePauliOp,
        prev_op_indices: list[int],
        theta: list[float],
        raw_vqe_result,
    ) -> None:
        """Optional hook after the inner VQE step for an ADAPT iteration.

        Subclasses can override this to capture plateau states or other
        diagnostics from the real solver trajectory without rewriting the full
        ADAPT loop.
        """
        del iteration, operator, prev_op_indices, theta, raw_vqe_result

    def _make_iteration_gradient_diagnostic(
        self,
        *,
        iteration: int,
        gradients: list[float],
        abs_gradients: list[float],
        sorted_candidates: list[int],
    ) -> dict[str, object]:
        """Build serializable gradient diagnostics for the current ADAPT step."""
        diagnostic: dict[str, object] = {"iteration": int(iteration)}
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_candidates)}
        if sorted_candidates:
            top_value = max(abs_gradients)
            diagnostic["top_tie_indices"] = [
                int(idx)
                for idx, value in enumerate(abs_gradients)
                if top_value - value <= self.gradient_tie_tolerance
            ]
            diagnostic["gradient_tie_tolerance"] = self.gradient_tie_tolerance
        if self.record_full_gradient_vector:
            diagnostic["gradients"] = [float(gradient) for gradient in gradients]

        if self.gradient_log_top_k > 0:
            top_entries: list[dict[str, object]] = []
            for rank, idx in enumerate(
                sorted_candidates[: self.gradient_log_top_k], start=1
            ):
                top_entries.append(
                    {
                        "rank": int(rank),
                        "idx": int(idx),
                        "gradient": float(gradients[idx]),
                        "abs_gradient": float(abs_gradients[idx]),
                    }
                )
            diagnostic["top_gradients"] = top_entries

        if self.tracked_operator_indices:
            tracked_entries: list[dict[str, object]] = []
            for idx in self.tracked_operator_indices:
                tracked_entries.append(
                    {
                        "idx": int(idx),
                        "rank": int(rank_map[idx]),
                        "gradient": float(gradients[idx]),
                        "abs_gradient": float(abs_gradients[idx]),
                    }
                )
            diagnostic["tracked_gradients"] = tracked_entries

        return diagnostic

    def _log_iteration_gradient_diagnostic(
        self,
        *,
        prefix: str,
        gradient_time_s: float,
        abs_gradients: list[float],
        sorted_candidates: list[int],
        diagnostic: dict[str, object],
    ) -> None:
        """Emit a human-readable gradient summary for the current ADAPT step."""
        top_idx = sorted_candidates[0]
        print(
            f"  [{prefix}] Gradient eval: {gradient_time_s:.2f}s, "
            f"top|grad|={abs_gradients[top_idx]:.6e} at idx {top_idx}"
        )

        for entry in diagnostic.get("top_gradients", []):
            print(
                "    Top {rank}: idx={idx}, grad={gradient:+.6e}, |grad|={abs_gradient:.6e}".format(
                    **entry
                )
            )

        for entry in diagnostic.get("tracked_gradients", []):
            print(
                "    Track idx={idx}: rank={rank}, grad={gradient:+.6e}, |grad|={abs_gradient:.6e}".format(
                    **entry
                )
            )

    def _finalize_iteration_gradient_diagnostic(
        self,
        diagnostic: dict[str, object],
        *,
        gradients: list[float],
        abs_gradients: list[float],
        sorted_candidates: list[int],
        selected_idx: int | None = None,
        termination_reason: str | None = None,
    ) -> None:
        """Attach selection outcome to the gradient diagnostics and persist it."""
        if selected_idx is not None:
            diagnostic["selected_idx"] = int(selected_idx)
            diagnostic["selected_rank"] = int(sorted_candidates.index(selected_idx) + 1)
            diagnostic["selected_gradient"] = float(gradients[selected_idx])
            diagnostic["selected_abs_gradient"] = float(abs_gradients[selected_idx])
        if termination_reason is not None:
            diagnostic["termination_reason"] = termination_reason
        self._iteration_gradient_history.append(diagnostic)

    def _build_ansatz(self) -> QuantumCircuit:
        """Build ansatz from current excitation list."""
        if not self._excitation_list:
            return self._initial_state
        return self._initial_state.compose(
            evolved_operator_ansatz(
                operators=self._excitation_list,
                reps=1,
                evolution=self._evolution,
                flatten=self._flatten,
            )
        )

    def compute_minimum_eigenvalue(
        self,
        operator: SparsePauliOp,
        aux_operators=None,
    ) -> AdaptVQEResult:
        """Run ADAPT-VQE with fast statevector gradients.

        Parameters
        ----------
        operator : SparsePauliOp
            Qubit Hamiltonian.
        aux_operators : optional
            Additional operators to evaluate at the end.

        Returns
        -------
        AdaptVQEResult
            Compatible with Qiskit's AdaptVQEResult.
        """
        if self.verbose >= 1:
            print("  [FastADAPT] Converting Hamiltonian to sparse matrix...")
        t0 = perf_counter()
        H_sparse = operator.to_matrix(sparse=True)
        t_ham = perf_counter() - t0
        if self.verbose >= 1:
            print(
                f"  [FastADAPT] Hamiltonian: {H_sparse.shape}, nnz={H_sparse.nnz}, took {t_ham:.1f}s"
            )

        if self.verbose >= 1:
            print(
                f"  [FastADAPT] Converting {len(self._excitation_pool)} pool operators to sparse matrices..."
            )
        t0 = perf_counter()
        op_matrices = [op.to_matrix(sparse=True) for op in self._excitation_pool]
        t_ops = perf_counter() - t0
        if self.verbose >= 1:
            print(f"  [FastADAPT] Pool conversion took {t_ops:.1f}s")

        self.solver.ansatz = self._initial_state
        theta: list[float] = []
        self._excitation_list = []
        self._selected_operator_indices = []
        self._set_selected_operator_sequence()
        self._iteration_gradient_history = []
        self._iteration_timing_history = []
        self._selection_diagnostic_history = []
        if hasattr(self.solver, "timing_history"):
            self.solver.timing_history = []
        if hasattr(self.solver, "last_timing_breakdown"):
            self.solver.last_timing_breakdown = None
        history: list[complex] = []
        prev_op_indices: list[int] = []
        raw_vqe_result = None
        termination = TerminationCriterion.MAXIMUM
        max_grad_val = 0.0
        max_idx = 0
        iteration = 0
        setup_timings = {
            "hamiltonian_to_sparse_s": float(t_ham),
            "pool_to_sparse_s": float(t_ops),
        }

        max_iter = self.max_iterations if self.max_iterations is not None else 999

        for iteration in range(1, max_iter + 1):
            if self.verbose >= 1:
                print(f"\n  [FastADAPT] --- Iteration {iteration}/{max_iter} ---")

            iteration_timing: dict[str, object] = {"iteration": int(iteration)}

            t_grad_start = perf_counter()
            psi = self._get_current_statevector(theta)
            grads = self._compute_gradients_fast(psi, H_sparse, op_matrices)
            t_grad = perf_counter() - t_grad_start
            iteration_timing["gradient_time_s"] = float(t_grad)

            abs_grads = [abs(g) for g in grads]
            sorted_candidates = self._rank_candidates_with_ties(abs_grads)
            iteration_diagnostic = self._make_iteration_gradient_diagnostic(
                iteration=iteration,
                gradients=grads,
                abs_gradients=abs_grads,
                sorted_candidates=sorted_candidates,
            )

            if self.verbose >= 1:
                self._log_iteration_gradient_diagnostic(
                    prefix="FastADAPT",
                    gradient_time_s=t_grad,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                    diagnostic=iteration_diagnostic,
                )

            selected_idx = None
            t_select_start = perf_counter()
            selection_decision = self._normalize_selection_decision(
                self._selection_decision(
                    iteration=iteration,
                    gradients=grads,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                    prev_op_indices=prev_op_indices.copy(),
                    theta=theta.copy(),
                ),
                theta=theta,
                pool_size=len(abs_grads),
            )
            self._record_selection_decision(
                iteration=iteration,
                iteration_timing=iteration_timing,
                iteration_diagnostic=iteration_diagnostic,
                decision=selection_decision,
            )
            selected_idx = selection_decision.selected_idx
            iteration_timing["selection_time_s"] = float(
                perf_counter() - t_select_start
            )

            if selected_idx is None:
                if sorted_candidates:
                    max_idx = sorted_candidates[0]
                    max_grad_val = abs_grads[max_idx]
                self._finalize_iteration_gradient_diagnostic(
                    iteration_diagnostic,
                    gradients=grads,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                    termination_reason=selection_decision.termination_reason,
                )
                if (
                    selection_decision.termination_criterion
                    == TerminationCriterion.CONVERGED
                ):
                    if self.verbose >= 1:
                        print(
                            f"  [FastADAPT] CONVERGED: all gradients < {self.gradient_threshold}"
                        )
                    termination = selection_decision.termination_criterion
                else:
                    if self.verbose >= 1:
                        print(
                            "  [FastADAPT] CYCLICITY: all above-threshold operators are cyclic"
                        )
                    termination = (
                        selection_decision.termination_criterion
                        or TerminationCriterion.CYCLICITY
                    )
                iteration_timing["termination_reason"] = (
                    selection_decision.termination_reason
                )
                iteration_timing["total_iteration_s"] = float(
                    iteration_timing.get("gradient_time_s", 0.0)
                ) + float(iteration_timing.get("selection_time_s", 0.0))
                self._iteration_timing_history.append(iteration_timing)
                break

            max_idx = selected_idx
            max_grad_val = abs_grads[max_idx]
            iteration_timing["selected_idx"] = int(max_idx)
            if self.verbose >= 1 and selection_decision.guided_selection:
                print(
                    f"  [FastADAPT] Guided selection forced idx={max_idx} "
                    f"(|grad|={abs_grads[max_idx]:.6e})"
                )
            self._finalize_iteration_gradient_diagnostic(
                iteration_diagnostic,
                gradients=grads,
                abs_gradients=abs_grads,
                sorted_candidates=sorted_candidates,
                selected_idx=max_idx,
            )
            prev_op_indices.append(max_idx)
            self._selected_operator_indices = prev_op_indices.copy()

            if self.verbose >= 1 and max_idx != sorted_candidates[0]:
                print(
                    f"  [FastADAPT] Selected idx={max_idx} (|grad|={max_grad_val:.6e}) "
                    f"after skipping cyclic candidates"
                )

            self._excitation_list.append(self._excitation_pool[max_idx])
            theta.append(0.0)

            self._set_selected_operator_sequence()
            if selection_decision.initial_point is not None:
                self.solver.initial_point = selection_decision.initial_point.copy()
            else:
                self.solver.initial_point = np.array(theta)

            t_vqe_start = perf_counter()
            prev_raw_vqe_result = raw_vqe_result
            raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
            self._after_vqe_iteration(
                iteration=iteration,
                operator=operator,
                prev_op_indices=prev_op_indices.copy(),
                theta=theta.copy(),
                raw_vqe_result=raw_vqe_result,
            )
            t_vqe = perf_counter() - t_vqe_start
            iteration_timing["vqe_time_s"] = float(t_vqe)
            iteration_timing["vqe_cost_function_evals"] = int(
                raw_vqe_result.cost_function_evals
            )
            solver_breakdown = getattr(self.solver, "last_timing_breakdown", None)
            if isinstance(solver_breakdown, dict):
                iteration_timing["inner_vqe_breakdown"] = {
                    key: (
                        bool(value)
                        if isinstance(value, (bool, np.bool_))
                        else (
                            float(value)
                            if isinstance(value, (int, float, np.floating))
                            else int(value) if isinstance(value, np.integer) else value
                        )
                    )
                    for key, value in solver_breakdown.items()
                }

            if self.verbose >= 1:
                print(
                    f"  [FastADAPT] VQE: {t_vqe:.1f}s, E={raw_vqe_result.eigenvalue.real:.10f}, "
                    f"evals={raw_vqe_result.cost_function_evals}"
                )

            if iteration > 1:
                eigenvalue_diff = abs(raw_vqe_result.eigenvalue - history[-1])
                if eigenvalue_diff < self.eigenvalue_threshold:
                    if self.verbose >= 1:
                        print(
                            f"  [FastADAPT] CONVERGED: eigenvalue change "
                            f"{eigenvalue_diff:.2e} < {self.eigenvalue_threshold}"
                        )
                    self._excitation_list.pop()
                    theta.pop()
                    prev_op_indices.pop()
                    self._selected_operator_indices = prev_op_indices.copy()
                    self._set_selected_operator_sequence()
                    self.solver.initial_point = np.array(theta)
                    raw_vqe_result = prev_raw_vqe_result
                    termination = TerminationCriterion.CONVERGED
                    iteration_timing["termination_reason"] = "eigenvalue_converged"
                    iteration_timing["total_iteration_s"] = (
                        float(iteration_timing.get("gradient_time_s", 0.0))
                        + float(iteration_timing.get("selection_time_s", 0.0))
                        + float(iteration_timing.get("vqe_time_s", 0.0))
                    )
                    self._iteration_timing_history.append(iteration_timing)
                    break

            iteration_timing["total_iteration_s"] = (
                float(iteration_timing.get("gradient_time_s", 0.0))
                + float(iteration_timing.get("selection_time_s", 0.0))
                + float(iteration_timing.get("vqe_time_s", 0.0))
            )
            self._iteration_timing_history.append(iteration_timing)
            history.append(raw_vqe_result.eigenvalue)
        else:
            termination = TerminationCriterion.MAXIMUM
            if self.verbose >= 1:
                print(f"  [FastADAPT] Reached max iterations ({max_iter})")

        if termination == TerminationCriterion.MAXIMUM:
            final_screen_start = perf_counter()
            final_psi = self._get_current_statevector(theta)
            final_grads = self._compute_gradients_fast(
                final_psi,
                H_sparse,
                op_matrices,
            )
            final_gradient_time = perf_counter() - final_screen_start
            final_abs_grads = [abs(gradient) for gradient in final_grads]
            final_sorted_candidates = self._rank_candidates_with_ties(final_abs_grads)
            if final_sorted_candidates:
                max_idx = final_sorted_candidates[0]
                max_grad_val = max(final_abs_grads)
            else:
                max_idx = 0
                max_grad_val = 0.0
            final_diagnostic = self._make_iteration_gradient_diagnostic(
                iteration=iteration + 1,
                gradients=final_grads,
                abs_gradients=final_abs_grads,
                sorted_candidates=final_sorted_candidates,
            )
            self._finalize_iteration_gradient_diagnostic(
                final_diagnostic,
                gradients=final_grads,
                abs_gradients=final_abs_grads,
                sorted_candidates=final_sorted_candidates,
                termination_reason="maximum_post_optimization_screen",
            )
            self._iteration_timing_history.append(
                {
                    "iteration": int(iteration + 1),
                    "gradient_time_s": float(final_gradient_time),
                    "selection_time_s": 0.0,
                    "vqe_time_s": 0.0,
                    "total_iteration_s": float(final_gradient_time),
                    "termination_reason": ("maximum_post_optimization_screen"),
                    "final_screening": True,
                }
            )

        result = FastAdaptVQEResult()
        if raw_vqe_result is None and self._uses_exact_sparse_evolution():
            optimal_state = self._get_current_statevector(theta)
            raw_vqe_result = _DirectSVResult(
                eigenvalue=float(
                    np.vdot(
                        optimal_state,
                        H_sparse @ optimal_state,
                    ).real
                ),
                optimal_point=np.asarray(theta, dtype=float),
                cost_function_evals=0,
            )
            raw_vqe_result.optimal_state = np.asarray(
                optimal_state,
                dtype=np.complex128,
            ).copy()
            raw_vqe_result.optimal_circuit = None
        if raw_vqe_result is not None:
            result.combine(raw_vqe_result)
            optimal_state = getattr(raw_vqe_result, "optimal_state", None)
            if optimal_state is not None:
                result.optimal_state = np.asarray(
                    optimal_state, dtype=np.complex128
                ).copy()
                result.optimal_circuit = None
        result.num_iterations = iteration
        result.final_max_gradient = float(max_grad_val)
        result.termination_criterion = termination
        result.eigenvalue_history = history
        result.iteration_gradient_history = list(self._iteration_gradient_history)
        result.iteration_timing_history = list(self._iteration_timing_history)
        result.selection_diagnostic_history = list(self._selection_diagnostic_history)
        result.setup_timings = dict(setup_timings)
        result.timing_summary = summarize_fastadapt_timings(
            setup_timings=setup_timings,
            iteration_timings=self._iteration_timing_history,
            selector_events=[],
        )

        if self.verbose >= 1:
            print(
                f"\n  [FastADAPT] Final: {iteration} iterations, "
                f"{len(self._excitation_list)} operators, "
                f"E={result.eigenvalue.real if result.eigenvalue else 'N/A'}"
            )
            print(
                "  [FastADAPT] Timing summary: "
                f"setup={result.timing_summary['hamiltonian_to_sparse_s'] + result.timing_summary['pool_to_sparse_s']:.1f}s, "
                f"grad={result.timing_summary['gradient_eval_total_s']:.1f}s, "
                f"select={result.timing_summary['selection_total_s']:.1f}s, "
                f"vqe={result.timing_summary['inner_vqe_total_s']:.1f}s, "
                f"dominant={result.timing_summary['dominant_phase']}"
            )

        if aux_operators is not None:
            optimal_state = getattr(result, "optimal_state", None)
            if optimal_state is not None:
                state = np.asarray(optimal_state, dtype=np.complex128)

                def exact_aux_value(aux_operator):
                    matrix = aux_operator.to_matrix(sparse=True)
                    value = complex(np.vdot(state, matrix @ state))
                    return (value, {"backend": "exact_sparse"})

                if isinstance(aux_operators, dict):
                    result.aux_operators_evaluated = {
                        key: exact_aux_value(aux_operator)
                        for key, aux_operator in aux_operators.items()
                    }
                else:
                    result.aux_operators_evaluated = [
                        exact_aux_value(aux_operator) for aux_operator in aux_operators
                    ]
            else:
                from qiskit_algorithms.observables_evaluator import (
                    estimate_observables,
                )

                result.aux_operators_evaluated = estimate_observables(
                    self.solver.estimator,
                    self.solver.ansatz,
                    aux_operators,
                    result.optimal_point,
                )

        return result
