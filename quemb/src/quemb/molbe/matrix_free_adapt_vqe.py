"""Matrix-free ADAPT-VQE: identical algorithm to FastAdaptVQE but without
materializing any sparse matrices. Computes H|ψ⟩ and A_i|ψ⟩ by applying
Pauli strings directly to the statevector via bit operations.

Usage:
    Drop-in replacement for FastAdaptVQE. Same constructor, same interface.

    from quemb.molbe.matrix_free_adapt_vqe import MatrixFreeAdaptVQE
    adapt_vqe = MatrixFreeAdaptVQE(
        solver=base_vqe,
        operators=operator_pool,
        initial_state=hf_circuit,
        evolution=LieTrotter(),
        gradient_threshold=1e-3,
        max_iterations=30,
    )
    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers.adapt_vqe import (
    AdaptVQEResult,
    TerminationCriterion,
)

from quemb.molbe.fast_adapt_vqe import (
    _DirectSVResult,
    FastAdaptVQE,
    FastAdaptVQEResult,
    summarize_fastadapt_timings,
)


def _precompute_pauli_masks(op: SparsePauliOp) -> list[tuple[complex, int, int]]:
    """Precompute (coeff*global_phase, flip_mask, z_mask) for each Pauli term.

    This avoids re-parsing string labels on every gradient evaluation.
    Qiskit uses big-endian labelling: label[0] is the highest qubit.
    We reverse to get little-endian (bit j ↔ qubit j).
    """
    masks: list[tuple[complex, int, int]] = []
    labels = op.paulis.to_labels()
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        flip_mask = 0
        z_mask = 0
        y_count = 0
        for j, p in enumerate(reversed(label)):
            if p == "X":
                flip_mask |= 1 << j
            elif p == "Y":
                flip_mask |= 1 << j
                z_mask |= 1 << j
                y_count += 1
            elif p == "Z":
                z_mask |= 1 << j
        global_phase = (-1j) ** (y_count % 4)
        masks.append((complex(coeff * global_phase), flip_mask, z_mask))
    return masks


def _apply_pauli_masks(
    masks: list[tuple[complex, int, int]],
    psi: np.ndarray,
    arange: np.ndarray,
) -> np.ndarray:
    """Apply a SparsePauliOp (given as precomputed masks) to a statevector.

    For each Pauli term with coefficient c, flip_mask, z_mask:
        result += c * (-1)^popcount(i & z_mask) * psi[i ^ flip_mask]

    The parity (popcount mod 2) is computed via XOR-fold — 5 bitwise ops,
    fully vectorised over the 2^n basis states.

    Parameters
    ----------
    masks : list of (coeff_with_phase, flip_mask, z_mask)
        Precomputed by _precompute_pauli_masks.
    psi : ndarray, shape (2^n,), complex128
        Input statevector.
    arange : ndarray, shape (2^n,), intp
        Pre-allocated np.arange(2^n) to avoid repeated allocation.

    Returns
    -------
    result : ndarray, shape (2^n,), complex128
    """
    N = len(psi)
    result = np.zeros(N, dtype=np.complex128)

    for coeff_phase, flip_mask, z_mask in masks:
        flipped = arange ^ flip_mask

        if z_mask:
            masked = arange & z_mask
            temp = masked ^ (masked >> 16)
            temp = temp ^ (temp >> 8)
            temp = temp ^ (temp >> 4)
            temp = temp ^ (temp >> 2)
            temp = temp ^ (temp >> 1)
            sign = np.where(temp & 1, -1.0, 1.0)
            result += coeff_phase * sign * psi[flipped]
        else:
            result += coeff_phase * psi[flipped]

    return result


class MatrixFreeAdaptVQE(FastAdaptVQE):
    """ADAPT-VQE with matrix-free gradient computation.

    Identical algorithm and interface to FastAdaptVQE. The only difference
    is that H|ψ⟩ and A_i|ψ⟩ are computed by iterating over Pauli terms
    instead of sparse matrix-vector products.

    Memory: O(2^n) — just statevectors, no stored matrices.
    Speed:  Gradient eval ~5-10x slower, total runtime ~10-15% slower.
    """

    def _selection_log_prefix(self) -> str:
        return "MatrixFree"

    @staticmethod
    def _compute_gradients_matrix_free(
        psi: np.ndarray,
        h_masks: list[tuple[complex, int, int]],
        pool_masks: list[list[tuple[complex, int, int]]],
        arange: np.ndarray,
        *,
        commutator_prefactor: complex = 1j,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        """Compute ADAPT gradients without any stored matrices.

        For each pool operator A_i:
            grad_i = Re[ i * (<Hψ|A_i ψ> - <ψ|A_i Hψ>) ]

        where Hψ and A_i|ψ⟩ are computed by applying Pauli terms directly.

        Parameters
        ----------
        psi : statevector, shape (2^n,)
        h_masks : precomputed masks for the Hamiltonian
        pool_masks : list of precomputed masks, one per pool operator
        arange : pre-allocated np.arange(2^n)
        progress_callback : callable, optional
            Called after each completed pool operator as
            ``progress_callback(completed, total)``. The default is ``None``
            and leaves the numerical path unchanged. This seam is used by
            long-running Brown validation jobs to persist progress and memory
            telemetry without retaining intermediate vectors.
        """
        hpsi = _apply_pauli_masks(h_masks, psi, arange)

        gradients = []
        total = len(pool_masks)
        for index, op_masks in enumerate(pool_masks):
            apsi = _apply_pauli_masks(op_masks, psi, arange)
            a_hpsi = _apply_pauli_masks(op_masks, hpsi, arange)

            term1 = np.vdot(hpsi, apsi)
            term2 = np.vdot(psi, a_hpsi)

            grad = commutator_prefactor * (term1 - term2)
            gradients.append(grad.real)
            if progress_callback is not None:
                progress_callback(index + 1, total)

        return gradients

    def compute_minimum_eigenvalue(
        self,
        operator: SparsePauliOp,
        aux_operators=None,
    ) -> AdaptVQEResult:
        """Run ADAPT-VQE with matrix-free gradients.

        Same algorithm as FastAdaptVQE.compute_minimum_eigenvalue, but
        replaces sparse matrix construction with precomputed Pauli masks.

        Parameters
        ----------
        operator : SparsePauliOp
            Qubit Hamiltonian.
        aux_operators : optional
            Additional operators to evaluate at the end.
        """
        N = 2**operator.num_qubits
        arange = np.arange(N, dtype=np.intp)

        if self.verbose >= 1:
            print(
                f"  [MatrixFree] Precomputing Hamiltonian masks "
                f"({len(operator)} Pauli terms, {operator.num_qubits} qubits)..."
            )
        t0 = perf_counter()
        h_masks = _precompute_pauli_masks(operator)
        t_ham = perf_counter() - t0
        if self.verbose >= 1:
            print(
                f"  [MatrixFree] Hamiltonian masks: "
                f"{len(h_masks)} terms, took {t_ham:.2f}s"
            )

        if self.verbose >= 1:
            print(
                f"  [MatrixFree] Precomputing {len(self._excitation_pool)} "
                f"pool operator masks..."
            )
        t0 = perf_counter()
        pool_masks = [_precompute_pauli_masks(op) for op in self._excitation_pool]
        t_ops = perf_counter() - t0
        if self.verbose >= 1:
            avg_terms = (
                sum(len(m) for m in pool_masks) / len(pool_masks) if pool_masks else 0
            )
            print(
                f"  [MatrixFree] Pool masks: {len(pool_masks)} operators, "
                f"avg {avg_terms:.1f} terms/op, took {t_ops:.2f}s"
            )
            est_sparse_gb = (
                N * len(h_masks) * 20 / 1e9
                + N * sum(len(m) for m in pool_masks) * 20 / 1e9
            )
            print(
                f"  [MatrixFree] Estimated sparse matrix memory avoided: "
                f"~{est_sparse_gb:.1f} GB"
            )

        self.solver.ansatz = self._initial_state
        theta: list[float] = []
        self._excitation_list = []
        self._selected_operator_indices = []
        self._iteration_gradient_history = []
        self._iteration_timing_history = []
        self._selection_diagnostic_history = []
        self._set_selected_operator_sequence()
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

        max_iter = self.max_iterations if self.max_iterations is not None else 999
        setup_timings = {
            "hamiltonian_to_sparse_s": float(t_ham),
            "pool_to_sparse_s": float(t_ops),
        }

        for iteration in range(1, max_iter + 1):
            if self.verbose >= 1:
                print(f"\n  [MatrixFree] --- Iteration {iteration}/{max_iter} ---")

            iteration_timing: dict[str, object] = {"iteration": int(iteration)}

            t_grad_start = perf_counter()
            psi = self._get_current_statevector(theta)
            grads = self._compute_gradients_matrix_free(
                psi,
                h_masks,
                pool_masks,
                arange,
                commutator_prefactor=(
                    -1j if self._uses_exact_sparse_evolution() else 1j
                ),
            )
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
                    prefix="MatrixFree",
                    gradient_time_s=t_grad,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                    diagnostic=iteration_diagnostic,
                )

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
                            f"  [MatrixFree] CONVERGED: all gradients < {self.gradient_threshold}"
                        )
                    termination = selection_decision.termination_criterion
                else:
                    if self.verbose >= 1:
                        print(
                            "  [MatrixFree] CYCLICITY: all above-threshold operators are cyclic"
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
                    f"  [MatrixFree] Guided selection forced idx={max_idx} "
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
                    f"  [MatrixFree] Selected idx={max_idx} (|grad|={max_grad_val:.6e}) "
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
                iteration_timing["inner_vqe_breakdown"] = dict(solver_breakdown)

            if self.verbose >= 1:
                print(
                    f"  [MatrixFree] VQE: {t_vqe:.1f}s, "
                    f"E={raw_vqe_result.eigenvalue.real:.10f}, "
                    f"evals={raw_vqe_result.cost_function_evals}"
                )

            if iteration > 1:
                eigenvalue_diff = abs(raw_vqe_result.eigenvalue - history[-1])
                if eigenvalue_diff < self.eigenvalue_threshold:
                    if self.verbose >= 1:
                        print(
                            f"  [MatrixFree] CONVERGED: eigenvalue change "
                            f"{eigenvalue_diff:.2e} "
                            f"< {self.eigenvalue_threshold}"
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
                print(f"  [MatrixFree] Reached max iterations ({max_iter})")

        if termination == TerminationCriterion.MAXIMUM:
            final_screen_start = perf_counter()
            final_psi = self._get_current_statevector(theta)
            final_grads = self._compute_gradients_matrix_free(
                final_psi,
                h_masks,
                pool_masks,
                arange,
                commutator_prefactor=(
                    -1j if self._uses_exact_sparse_evolution() else 1j
                ),
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
                        _apply_pauli_masks(
                            h_masks,
                            optimal_state,
                            arange,
                        ),
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
        result.iteration_gradient_history = list(self._iteration_gradient_history)
        result.iteration_timing_history = list(self._iteration_timing_history)
        result.selection_diagnostic_history = list(self._selection_diagnostic_history)
        result.eigenvalue_history = history
        result.setup_timings = dict(setup_timings)
        result.timing_summary = summarize_fastadapt_timings(
            setup_timings=setup_timings,
            iteration_timings=self._iteration_timing_history,
            selector_events=[],
        )

        if self.verbose >= 1:
            print(
                f"\n  [MatrixFree] Final: {iteration} iterations, "
                f"{len(self._excitation_list)} operators, "
                f"E={result.eigenvalue.real if result.eigenvalue else 'N/A'}"
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
