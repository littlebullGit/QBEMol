"""Reusable experiment-side utilities for ADAPT lookahead selectors."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np
from qiskit.circuit.library import evolved_operator_ansatz
from qiskit_algorithms.exceptions import AlgorithmError

try:
    from qiskit_algorithms.gradients import ReverseEstimatorGradient
except Exception:  
    ReverseEstimatorGradient = None

from qiskit_algorithms.minimum_eigensolvers.adapt_vqe import (
    AdaptVQEResult,
    TerminationCriterion,
)
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.optimize import minimize, minimize_scalar

from quemb.molbe import BE
import quemb.molbe.fast_adapt_vqe as fast_adapt_module
from quemb.molbe.fast_adapt_vqe import FastAdaptVQE
from quemb.molbe.vqe_solver import get_vqe_fragment_observables

try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
except Exception:  
    AerEstimatorV2 = None


TriggerCyclicityFn = Callable[[list[int]], bool]
FastAdaptVQEResult = getattr(fast_adapt_module, "FastAdaptVQEResult", AdaptVQEResult)
summarize_fastadapt_timings = cast(
    Any, getattr(fast_adapt_module, "summarize_fastadapt_timings")
)


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    top_k: int = 5
    family_top_n: int = 10
    probe_maxiter: int = 40
    probe_ftol: float = 1e-10
    probe_progress_interval_s: float = 60.0
    probe_backend: str = "direct_sv"
    probe_aer_threads: int = 4
    probe_aer_precision: str = "double"
    staged_probe: bool = False
    refine_top_m: int = 3
    coarse_scan_steps: tuple[float, ...] = ()
    parallel_probes: bool = False
    parallel_workers: int = 4
    analytical_grad: bool = False
    generic_min_iteration: int = 5
    generic_recent_window: int = 10
    generic_min_overlap: int = 3


_SINGLE_EXCITATION_RE = re.compile(r"^(\d+)([ab])->(\d+)([ab])$")
_DOUBLE_EXCITATION_RE = re.compile(
    r"^(\d+)([ab]),(\d+)([ab])->(\d+)([ab]),(\d+)([ab])$"
)


def parse_excitation_label(label: str) -> dict[str, Any]:
    single_match = _SINGLE_EXCITATION_RE.match(label)
    if single_match:
        occ, occ_spin, vir, vir_spin = single_match.groups()
        return {
            "kind": "single",
            "occupied_orbitals": [int(occ)],
            "virtual_orbitals": [int(vir)],
            "spin_channels": [f"{occ_spin}->{vir_spin}"],
            "diagonal_channel": (int(occ), int(vir)),
            "coupled_channels": [(int(occ), int(vir))],
        }

    double_match = _DOUBLE_EXCITATION_RE.match(label)
    if not double_match:
        return {
            "kind": "other",
            "occupied_orbitals": [],
            "virtual_orbitals": [],
            "spin_channels": [],
            "diagonal_channel": None,
            "coupled_channels": [],
        }

    occ1, spin1, occ2, spin2, vir1, vir_spin1, vir2, vir_spin2 = double_match.groups()
    o1 = int(occ1)
    o2 = int(occ2)
    v1 = int(vir1)
    v2 = int(vir2)
    coupled_channels = [(o1, v1), (o2, v2)]
    diagonal_channel = None
    kind = "mixed_double"
    if o1 == o2 and v1 == v2 and spin1 != spin2 and vir_spin1 != vir_spin2:
        diagonal_channel = (o1, v1)
        kind = "pair_double"

    return {
        "kind": kind,
        "occupied_orbitals": [o1, o2],
        "virtual_orbitals": [v1, v2],
        "spin_channels": [f"{spin1}->{vir_spin1}", f"{spin2}->{vir_spin2}"],
        "diagonal_channel": diagonal_channel,
        "coupled_channels": coupled_channels,
    }


def build_diagonal_channel_catalog(
    excitation_labels: list[str],
) -> dict[tuple[int, int], dict[str, int | None]]:
    catalog: dict[tuple[int, int], dict[str, int | None]] = {}
    for idx, label in enumerate(excitation_labels):
        parsed = parse_excitation_label(label)
        if parsed["kind"] == "single":
            occ = parsed["occupied_orbitals"][0]
            vir = parsed["virtual_orbitals"][0]
            key = (occ, vir)
            entry = catalog.setdefault(
                key,
                {
                    "alpha_single_idx": None,
                    "beta_single_idx": None,
                    "pair_double_idx": None,
                },
            )
            spin_channel = parsed["spin_channels"][0]
            if spin_channel == "a->a":
                entry["alpha_single_idx"] = idx
            elif spin_channel == "b->b":
                entry["beta_single_idx"] = idx
        elif parsed["kind"] == "pair_double":
            key = cast(tuple[int, int], parsed["diagonal_channel"])
            entry = catalog.setdefault(
                key,
                {
                    "alpha_single_idx": None,
                    "beta_single_idx": None,
                    "pair_double_idx": None,
                },
            )
            entry["pair_double_idx"] = idx
    return catalog


def normalize_initial_point(
    initial_point: list[float] | np.ndarray,
    n_params: int,
) -> np.ndarray:
    """Resize an initial point to the ansatz parameter dimension."""
    arr = np.asarray(initial_point, dtype=float).reshape(-1)
    if arr.size == n_params:
        return arr.copy()
    if arr.size > n_params:
        return arr[:n_params].copy()
    out = np.zeros(n_params, dtype=float)
    out[: arr.size] = arr
    return out


def build_generic_repeat_pressure_context(
    *,
    iteration: int,
    abs_grads: list[float],
    sorted_candidates: list[int],
    prev_op_indices: list[int],
    gradient_threshold: float,
    top_k: int,
    min_iteration: int,
    recent_window: int,
    min_overlap: int,
    check_cyclicity: bool,
    cyclicity_fn: TriggerCyclicityFn,
) -> dict[str, Any]:
    """Build the generic recent-repeat / cyclic-pressure trigger context."""
    top_candidates: list[int] = []
    for candidate_idx in sorted_candidates:
        if abs_grads[candidate_idx] < gradient_threshold:
            break
        top_candidates.append(candidate_idx)
        if len(top_candidates) >= top_k:
            break

    recent_indices = prev_op_indices[-recent_window:]
    recent_set = set(recent_indices)
    repeated_in_top = [idx for idx in top_candidates if idx in recent_set]
    cyclic_in_top = [
        idx
        for idx in top_candidates
        if check_cyclicity and cyclicity_fn(prev_op_indices + [idx])
    ]
    top1_idx = top_candidates[0] if top_candidates else None
    top1_in_recent = top1_idx in recent_set if top1_idx is not None else False

    should_activate = (
        iteration >= min_iteration
        and len(prev_op_indices) >= min_overlap
        and top1_in_recent
        and (len(repeated_in_top) >= min_overlap or len(cyclic_in_top) > 0)
    )

    return {
        "mode": "generic_repeat_pressure",
        "iteration": int(iteration),
        "top_candidates": [int(idx) for idx in top_candidates],
        "recent_window": [int(idx) for idx in recent_indices],
        "top1_idx": None if top1_idx is None else int(top1_idx),
        "top1_in_recent_window": bool(top1_in_recent),
        "recent_overlap": [int(idx) for idx in repeated_in_top],
        "recent_overlap_count": int(len(repeated_in_top)),
        "cyclic_candidates": [int(idx) for idx in cyclic_in_top],
        "cyclic_candidate_count": int(len(cyclic_in_top)),
        "min_iteration": int(min_iteration),
        "recent_window_size": int(recent_window),
        "min_recent_overlap": int(min_overlap),
        "should_activate": bool(should_activate),
    }


def build_reference_excitation_map(norb: int, nelec: int) -> list[str]:
    n_alpha = (nelec + 1) // 2
    n_beta = nelec - n_alpha
    mapper = JordanWignerMapper()
    hf_state = HartreeFock(norb, (n_alpha, n_beta), mapper)
    ucc = UCC(
        num_spatial_orbitals=norb,
        num_particles=(n_alpha, n_beta),
        excitations="sd",
        qubit_mapper=mapper,
        initial_state=hf_state,
        generalized=False,
        reps=1,
        preserve_spin=True,
    )

    def spin_orb_label(idx: int) -> str:
        spin = "a" if idx < norb else "b"
        return f"{idx % norb}{spin}"

    labels: list[str] = []
    for occupied, virtual in ucc.excitation_list or []:
        occ_label = ",".join(spin_orb_label(i) for i in occupied)
        vir_label = ",".join(spin_orb_label(i) for i in virtual)
        labels.append(f"{occ_label}->{vir_label}")
    return labels


def safe_excitation_label(idx: int, excitation_labels: list[str]) -> str:
    if 0 <= idx < len(excitation_labels):
        return excitation_labels[idx]
    return f"pool_idx_{idx}"


def safe_excitation_labels(
    indices: list[int], excitation_labels: list[str]
) -> list[str]:
    return [safe_excitation_label(idx, excitation_labels) for idx in indices]


def _probe_log(
    *,
    log_prefix: str,
    backend_name: str,
    candidate_idx: int | None,
    candidate_label: str | None,
    message: str,
) -> None:
    """Emit a per-probe progress line that survives ProcessPool workers."""
    ident_parts = [f"backend={backend_name}"]
    if candidate_idx is not None:
        ident_parts.append(f"idx={candidate_idx}")
    if candidate_label:
        ident_parts.append(f"exc={candidate_label}")
    ident = " ".join(ident_parts)
    print(f"  [{log_prefix}] Probe[{ident}] {message}", flush=True)


def _timing_value(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _complex_eigenvalue(value: object) -> complex:
    if value is None:
        raise RuntimeError("FastAdapt VQE returned no eigenvalue")
    return complex(cast(Any, value))


def collect_final_fragment_phase_timings(mybe: BE) -> list[dict[str, Any]]:
    observables = cast(dict[str, Any], get_vqe_fragment_observables())
    records: list[dict[str, Any]] = []
    for frag_idx, frag in enumerate(mybe.Fobjs):
        payload = cast(dict[str, Any], observables.get(str(frag.dname), {}))
        timings_raw = payload.get("timings", {})
        timings: dict[str, float] = {}
        if isinstance(timings_raw, dict):
            for key, value in timings_raw.items():
                timing = _timing_value(value)
                if timing is not None:
                    timings[key] = timing
        record: dict[str, Any] = {
            "fragment_index": int(frag_idx),
            "fragment_name": str(frag.dname),
            "timings": timings,
        }
        energy = _timing_value(payload.get("energy"))
        if energy is not None:
            record["energy"] = energy
        records.append(record)
    return records


def aggregate_final_fragment_phase_timings(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    for record in records:
        timings = cast(dict[str, Any], record.get("timings") or {})
        for key, value in timings.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if totals:
        dominant_phase, dominant_value = max(totals.items(), key=lambda item: item[1])
    else:
        dominant_phase, dominant_value = "none", 0.0
    return {
        "phase_totals_s": totals,
        "dominant_phase": dominant_phase,
        "dominant_phase_s": float(dominant_value),
    }


def summarize_selector_events(
    selector_events: list[dict[str, Any]],
) -> dict[str, int | float]:
    """Aggregate lookahead selector-event counters and timings."""
    totals: dict[str, int | float] = {
        "selector_event_count": 0,
        "shortlist_candidate_count": 0,
        "refined_candidate_count": 0,
        "probe_call_count": 0,
        "probe_nfev_total": 0,
        "probe_nit_total": 0,
        "coarse_scan_n_evals_total": 0,
        "coarse_scan_elapsed_s": 0.0,
        "probe_elapsed_s": 0.0,
        "probe_wall_s": 0.0,
        "selection_overhead_s": 0.0,
    }
    for event in selector_events:
        shortlist_raw = event.get("shortlist")
        shortlist_count = event.get("shortlist_count")
        if shortlist_count is None and isinstance(shortlist_raw, list):
            shortlist_count = len(shortlist_raw)
        refined_count = event.get("refined_candidate_count")
        if refined_count is None:
            refined_count = event.get("full_probe_call_count", 0)
        probe_call_count = event.get("full_probe_call_count", refined_count or 0)

        totals["selector_event_count"] += 1
        totals["shortlist_candidate_count"] += int(shortlist_count or 0)
        totals["refined_candidate_count"] += int(refined_count or 0)
        totals["probe_call_count"] += int(probe_call_count or 0)
        totals["probe_nfev_total"] += int(event.get("total_probe_nfev", 0) or 0)
        totals["probe_nit_total"] += int(event.get("total_probe_nit", 0) or 0)
        totals["coarse_scan_n_evals_total"] += int(
            event.get("coarse_scan_n_evals", 0) or 0
        )
        totals["coarse_scan_elapsed_s"] += float(
            event.get("coarse_scan_elapsed_s", 0.0) or 0.0
        )
        totals["probe_elapsed_s"] += float(
            event.get(
                "total_probe_elapsed_s",
                event.get("full_probe_elapsed_sum_s", 0.0),
            )
            or 0.0
        )
        totals["probe_wall_s"] += float(
            event.get(
                "total_probe_wall_s",
                event.get(
                    "total_probe_elapsed_s",
                    event.get("full_probe_elapsed_sum_s", 0.0),
                ),
            )
            or 0.0
        )
        totals["selection_overhead_s"] += float(
            event.get("selection_overhead_s", 0.0) or 0.0
        )
    return totals


def aggregate_solve_timing_summaries(
    solve_records: list[dict[str, Any]],
    runtime_s: float,
) -> dict[str, Any]:
    totals = {
        "hamiltonian_to_sparse_s": 0.0,
        "pool_to_sparse_s": 0.0,
        "gradient_eval_total_s": 0.0,
        "selection_total_s": 0.0,
        "inner_vqe_total_s": 0.0,
        "probe_total_s": 0.0,
        "probe_work_total_s": 0.0,
        "total_accounted_s": 0.0,
    }
    count_totals = {
        "n_selector_events": 0,
        "probe_call_count": 0,
        "probe_nfev_total": 0,
        "probe_nit_total": 0,
        "main_vqe_step_count": 0,
        "main_vqe_nfev_total": 0,
        "vqe_like_call_count": 0,
        "coarse_scan_n_evals_total": 0,
        "shortlist_candidate_count": 0,
        "refined_candidate_count": 0,
    }
    for record in solve_records:
        summary = record.get("timing_summary", {})
        if not isinstance(summary, dict):
            continue
        for key in totals:
            totals[key] += float(summary.get(key, 0.0))
        count_totals["n_selector_events"] += int(summary.get("n_selector_events", 0))
        count_totals["probe_call_count"] += int(summary.get("probe_call_count", 0))
        count_totals["probe_nfev_total"] += int(summary.get("probe_nfev_total", 0))
        count_totals["probe_nit_total"] += int(summary.get("probe_nit_total", 0))
        count_totals["main_vqe_step_count"] += int(
            summary.get("main_vqe_step_count", summary.get("n_vqe_steps", 0))
        )
        count_totals["main_vqe_nfev_total"] += int(
            summary.get("main_vqe_nfev_total", 0)
        )
        count_totals["vqe_like_call_count"] += int(
            summary.get("vqe_like_call_count", 0)
        )
        count_totals["coarse_scan_n_evals_total"] += int(
            summary.get("coarse_scan_n_evals_total", 0)
        )
        count_totals["shortlist_candidate_count"] += int(
            summary.get("shortlist_candidate_count", 0)
        )
        count_totals["refined_candidate_count"] += int(
            summary.get("refined_candidate_count", 0)
        )
    accounted_phases = {
        "hamiltonian_to_sparse": totals["hamiltonian_to_sparse_s"],
        "pool_to_sparse": totals["pool_to_sparse_s"],
        "gradient_eval": totals["gradient_eval_total_s"],
        "selection": totals["selection_total_s"],
        "inner_vqe": totals["inner_vqe_total_s"],
        "probe": totals["probe_total_s"],
    }
    dominant_phase, dominant_value = max(
        accounted_phases.items(), key=lambda item: item[1]
    )
    return {
        **totals,
        **count_totals,
        "runtime_s": float(runtime_s),
        "be_overhead_s": float(runtime_s - totals["total_accounted_s"]),
        "dominant_phase": dominant_phase,
        "dominant_phase_s": float(dominant_value),
    }


def make_lookahead_fast_adapt(
    config: SelectorConfig,
    case_name: str,
    selector_mode: str,
    excitation_labels: list[str],
    log_prefix: str = "EthaneLookahead",
) -> type[FastAdaptVQE]:
    class LookaheadFastAdaptVQE(FastAdaptVQE):
        solve_records: list[dict[str, Any]] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.selector_events: list[dict[str, Any]] = []
            self._last_result = None
            self._generic_lookahead_activated = False
            self._probe_backend = None
            self._probe_new_param_seed_cache: dict[int, float] = {}
            self._diagonal_channel_catalog = build_diagonal_channel_catalog(
                excitation_labels
            )

        def _build_trial_ansatz(self, trial_indices: list[int]):
            if not trial_indices:
                return self._initial_state
            trial_ops = [self._excitation_pool[idx] for idx in trial_indices]
            return self._initial_state.compose(
                evolved_operator_ansatz(
                    operators=trial_ops,
                    reps=1,
                    evolution=self._evolution,
                    flatten=self._flatten,
                )
            )

        def _generic_trigger_context(
            self,
            *,
            iteration: int,
            abs_grads: list[float],
            sorted_candidates: list[int],
            prev_op_indices: list[int],
        ) -> dict[str, Any]:
            return build_generic_repeat_pressure_context(
                iteration=iteration,
                abs_grads=abs_grads,
                sorted_candidates=sorted_candidates,
                prev_op_indices=prev_op_indices,
                gradient_threshold=self.gradient_threshold,
                top_k=config.top_k,
                min_iteration=config.generic_min_iteration,
                recent_window=config.generic_recent_window,
                min_overlap=config.generic_min_overlap,
                check_cyclicity=self.check_cyclicity,
                cyclicity_fn=self._check_cyclicity,
            )

        def _standard_select_candidate(
            self,
            *,
            iteration: int,
            abs_grads: list[float],
            sorted_candidates: list[int],
            prev_op_indices: list[int],
        ) -> tuple[int | None, str | None]:
            termination_reason = None
            for candidate_idx in sorted_candidates:
                candidate_grad = abs_grads[candidate_idx]
                if candidate_grad < self.gradient_threshold:
                    if iteration == 1:
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
                                    f"  [{log_prefix}] CYCLICITY at idx={candidate_idx}, stopping"
                                )
                            termination_reason = "cyclicity"
                            break
                        if self.verbose >= 1:
                            print(
                                f"  [{log_prefix}] CYCLICITY at idx={candidate_idx} "
                                f"(|grad|={candidate_grad:.6e}), skipping"
                            )
                        continue

                return candidate_idx, None

            return None, termination_reason

        def _lookahead_select_candidate(
            self,
            *,
            iteration: int,
            grads: list[float],
            abs_grads: list[float],
            sorted_candidates: list[int],
            prev_op_indices: list[int],
            theta: list[float],
            operator,
            h_sparse,
            trigger_context: dict[str, Any],
        ) -> tuple[int | None, list[float] | None, str | None]:
            select_t0 = perf_counter()
            shortlist: list[int] = []
            skipped_cyclic: list[int] = []
            termination_reason = None

            for candidate_idx in sorted_candidates:
                candidate_grad = abs_grads[candidate_idx]
                if candidate_grad < self.gradient_threshold:
                    if iteration == 1 and not shortlist:
                        raise AlgorithmError(
                            "All gradients below threshold in first iteration. "
                            "Try tighter threshold or different operator pool."
                        )
                    termination_reason = "converged"
                    break

                if self.check_cyclicity and self._check_cyclicity(
                    prev_op_indices + [candidate_idx]
                ):
                    skipped_cyclic.append(candidate_idx)
                    continue

                shortlist.append(candidate_idx)
                if len(shortlist) >= config.top_k:
                    break

            if not shortlist:
                return None, None, termination_reason or "cyclicity"

            event: dict[str, Any] = {
                "iteration": int(iteration),
                "trigger_mode": "generic_repeat_pressure",
                "shortlist_top_k": config.top_k,
                "shortlist": [],
                "shortlist_count": 0,
                "skipped_cyclic": [int(idx) for idx in skipped_cyclic],
                "trigger_context": trigger_context,
            }

            trial_ansatz_map: dict[int, Any] = {}
            entry_map: dict[int, dict[str, Any]] = {}
            rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_candidates)}
            coarse_tasks: list[dict[str, Any]] = []
            refine_initial_points: dict[int, np.ndarray] = {}
            total_probe_nfev = 0
            total_probe_nit = 0
            full_probe_elapsed_sum = 0.0

            staged_probe_active = (
                config.staged_probe
                and len(shortlist) > config.refine_top_m
                and len(config.coarse_scan_steps) > 0
            )

            for candidate_idx in shortlist:
                trial_indices = prev_op_indices + [candidate_idx]
                trial_ansatz = self._build_trial_ansatz(trial_indices)
                candidate_entry = {
                    "idx": int(candidate_idx),
                    "rank": int(rank_map[candidate_idx]),
                    "gradient": float(grads[candidate_idx]),
                    "abs_gradient": float(abs_grads[candidate_idx]),
                    "excitation": safe_excitation_label(
                        candidate_idx, excitation_labels
                    ),
                    "n_params": int(len(list(trial_ansatz.parameters))),
                    "refined": not staged_probe_active,
                }
                event["shortlist"].append(candidate_entry)
                entry_map[candidate_idx] = candidate_entry
                trial_ansatz_map[candidate_idx] = trial_ansatz

                if staged_probe_active:
                    n_params = len(list(trial_ansatz.parameters))
                    direction = -1.0 if grads[candidate_idx] >= 0 else 1.0
                    trial_thetas = [
                        normalize_initial_point(
                            theta + [direction * step],
                            n_params,
                        )
                        for step in config.coarse_scan_steps
                    ]
                    coarse_tasks.append(
                        {
                            "candidate_idx": candidate_idx,
                            "ansatz": trial_ansatz,
                            "operator": operator,
                            "h_sparse": h_sparse,
                            "trial_thetas": trial_thetas,
                        }
                    )

            if staged_probe_active:
                probe_backend = self._probe_backend
                if probe_backend is None:
                    raise RuntimeError("Probe backend not initialized")
                coarse_scan = probe_backend.coarse_scan_many(tasks=coarse_tasks)
                event["staged_probe_active"] = True
                event["coarse_scan_backend"] = str(coarse_scan["backend"])
                event["coarse_scan_batched"] = bool(coarse_scan["batched"])
                event["coarse_scan_elapsed_s"] = float(coarse_scan["elapsed_s"])
                event["coarse_scan_n_evals"] = int(coarse_scan["n_evals"])
                event["coarse_scan_steps"] = list(config.coarse_scan_steps)
                event["refine_top_m"] = int(config.refine_top_m)

                coarse_ranked: list[tuple[float, int]] = []
                for task, coarse_result in zip(coarse_tasks, coarse_scan["results"]):
                    candidate_idx = int(task["candidate_idx"])
                    candidate_entry = entry_map[candidate_idx]
                    best_idx = int(coarse_result["best_index"])
                    best_theta = np.asarray(task["trial_thetas"][best_idx], dtype=float)
                    refine_initial_points[candidate_idx] = best_theta
                    candidate_entry["coarse_scan_energies"] = [
                        float(val) for val in coarse_result["energies"]
                    ]
                    candidate_entry["coarse_best_energy"] = float(
                        coarse_result["best_energy"]
                    )
                    candidate_entry["coarse_best_step"] = float(
                        config.coarse_scan_steps[best_idx]
                    )
                    coarse_ranked.append(
                        (float(coarse_result["best_energy"]), candidate_idx)
                    )

                coarse_ranked.sort(key=lambda item: (item[0], rank_map[item[1]]))
                refined_candidates = [
                    candidate_idx
                    for _, candidate_idx in coarse_ranked[: config.refine_top_m]
                ]
                event["refine_candidates"] = [int(idx) for idx in refined_candidates]
            else:
                event["staged_probe_active"] = False
                event["refine_top_m"] = int(len(shortlist))
                event["coarse_scan_elapsed_s"] = 0.0
                event["coarse_scan_n_evals"] = 0
                event["coarse_scan_steps"] = []
                refined_candidates = shortlist
                for candidate_idx in shortlist:
                    trial_ansatz = trial_ansatz_map[candidate_idx]
                    n_params = len(list(trial_ansatz.parameters))
                    seed_alpha = float(
                        self._probe_new_param_seed_cache.get(candidate_idx, 0.0)
                    )
                    refine_initial_points[candidate_idx] = normalize_initial_point(
                        theta + [seed_alpha], n_params
                    )
                    entry_map[candidate_idx]["probe_seed_alpha"] = float(seed_alpha)
            event["shortlist_count"] = int(len(shortlist))
            event["refined_candidate_count"] = int(len(refined_candidates))

            best_candidate = None
            best_probe = None
            probe_backend = self._probe_backend
            if probe_backend is None:
                raise RuntimeError("Probe backend not initialized")

            probe_tasks = [
                {
                    "candidate_idx": candidate_idx,
                    "candidate_label": safe_excitation_label(
                        candidate_idx, excitation_labels
                    ),
                    "ansatz": trial_ansatz_map[candidate_idx],
                    "operator": operator,
                    "h_sparse": h_sparse,
                    "initial_point": refine_initial_points[candidate_idx],
                    "maxiter": config.probe_maxiter,
                    "ftol": config.probe_ftol,
                    "progress_interval_s": config.probe_progress_interval_s,
                    "log_prefix": log_prefix,
                    "use_analytical_grad": config.analytical_grad,
                }
                for candidate_idx in refined_candidates
            ]
            event["full_probe_call_count"] = int(len(probe_tasks))

            use_parallel = config.parallel_probes and hasattr(
                probe_backend, "probe_parallel"
            )
            if use_parallel:
                probe_batch_t0 = perf_counter()
                shortlist_desc = ", ".join(
                    f"{task['candidate_idx']}:{task['candidate_label']}"
                    for task in probe_tasks
                )
                print(
                    f"  [{log_prefix}] Running {len(probe_tasks)} probes in parallel"
                    f" (workers={config.parallel_workers},"
                    f" analytical_grad={config.analytical_grad})"
                )
                print(f"  [{log_prefix}] Probe shortlist: {shortlist_desc}")
                parallel_results = probe_backend.probe_parallel(
                    probe_tasks=probe_tasks,
                    max_workers=config.parallel_workers,
                )
                total_par_nfev = sum(int(r["nfev"]) for r in parallel_results)
                total_par_elapsed = sum(float(r["elapsed_s"]) for r in parallel_results)
                probe_batch = {
                    "results": parallel_results,
                    "elapsed_s": total_par_elapsed,
                    "wall_s": perf_counter() - probe_batch_t0,
                    "n_evals": total_par_nfev,
                    "batched": True,
                    "backend": str(getattr(probe_backend, "name", "probe"))
                    + "+parallel",
                }
            elif callable(getattr(probe_backend, "probe_many", None)):
                probe_batch_t0 = perf_counter()
                probe_batch = probe_backend.probe_many(tasks=probe_tasks)
                probe_batch.setdefault("wall_s", perf_counter() - probe_batch_t0)
            else:
                probe_batch_t0 = perf_counter()
                fallback_results = []
                total_fallback_nfev = 0
                fallback_elapsed_s = 0.0
                for task in probe_tasks:
                    probe = probe_backend.probe(
                        ansatz=task["ansatz"],
                        operator=task["operator"],
                        h_sparse=task["h_sparse"],
                        initial_point=task["initial_point"],
                        maxiter=task["maxiter"],
                        ftol=task["ftol"],
                    )
                    fallback_results.append(probe)
                    total_fallback_nfev += int(probe.get("nfev", 0))
                    fallback_elapsed_s += float(probe.get("elapsed_s", 0.0))
                probe_batch = {
                    "results": fallback_results,
                    "elapsed_s": fallback_elapsed_s,
                    "wall_s": perf_counter() - probe_batch_t0,
                    "n_evals": int(total_fallback_nfev),
                    "batched": False,
                    "backend": str(getattr(probe_backend, "name", "probe")),
                }

            event["full_probe_backend"] = str(probe_batch["backend"])
            event["full_probe_batched"] = bool(probe_batch["batched"])
            event["full_probe_n_evals"] = int(probe_batch["n_evals"])
            full_probe_elapsed_sum = float(probe_batch["elapsed_s"])
            full_probe_wall_s = float(
                probe_batch.get("wall_s", probe_batch.get("elapsed_s", 0.0))
            )
            if len(probe_batch["results"]) != len(refined_candidates):
                raise RuntimeError(
                    "probe_many returned a result count that does not match the refined shortlist"
                )

            for candidate_idx, probe in zip(refined_candidates, probe_batch["results"]):
                candidate_entry = entry_map[candidate_idx]
                candidate_entry["refined"] = True
                candidate_entry["probe_energy"] = float(probe["energy"])
                candidate_entry["probe_success"] = bool(probe["success"])
                candidate_entry["probe_nit"] = int(probe["nit"])
                candidate_entry["probe_nfev"] = int(probe["nfev"])
                if "njev" in probe:
                    candidate_entry["probe_njev"] = int(probe["njev"])
                if "n_unique_evals" in probe:
                    candidate_entry["probe_unique_evals"] = int(probe["n_unique_evals"])
                if "cache_hits" in probe:
                    candidate_entry["probe_cache_hits"] = int(probe["cache_hits"])
                if "gradient_cache_hits" in probe:
                    candidate_entry["probe_gradient_cache_hits"] = int(
                        probe["gradient_cache_hits"]
                    )
                if "used_exact_jacobian" in probe:
                    candidate_entry["probe_used_exact_jacobian"] = bool(
                        probe["used_exact_jacobian"]
                    )
                if "jacobian_total_s" in probe:
                    candidate_entry["probe_jacobian_total_s"] = float(
                        probe["jacobian_total_s"]
                    )
                candidate_entry["probe_message"] = str(probe["message"])
                candidate_entry["probe_backend"] = str(probe["backend"])
                candidate_entry["probe_elapsed_s"] = float(probe["elapsed_s"])
                optimal_point = np.asarray(probe["optimal_point"], dtype=float)
                if optimal_point.size:
                    self._probe_new_param_seed_cache[candidate_idx] = float(
                        optimal_point[-1]
                    )
                total_probe_nfev += int(probe["nfev"])
                total_probe_nit += int(probe["nit"])

                if best_probe is None:
                    best_candidate = candidate_idx
                    best_probe = probe
                    continue

                energy_delta = float(probe["energy"]) - float(best_probe["energy"])
                if energy_delta < -1e-12:
                    best_candidate = candidate_idx
                    best_probe = probe
                elif (
                    abs(energy_delta) <= 1e-12
                    and best_candidate is not None
                    and rank_map[candidate_idx] < rank_map[best_candidate]
                ):
                    best_candidate = candidate_idx
                    best_probe = probe

            if best_candidate is None or best_probe is None:
                return None, None, termination_reason or "cyclicity"

            event["selected_idx"] = int(best_candidate)
            event["selected_rank"] = int(rank_map[best_candidate])
            event["selected_excitation"] = safe_excitation_label(
                best_candidate, excitation_labels
            )
            event["selected_probe_energy"] = float(best_probe["energy"])
            event["total_probe_nfev"] = int(total_probe_nfev)
            event["total_probe_nit"] = int(total_probe_nit)
            event["full_probe_elapsed_sum_s"] = float(full_probe_elapsed_sum)
            event["full_probe_wall_s"] = float(full_probe_wall_s)
            event["total_probe_elapsed_s"] = float(
                event.get("coarse_scan_elapsed_s", 0.0) + full_probe_elapsed_sum
            )
            event["total_probe_wall_s"] = float(
                event.get("coarse_scan_elapsed_s", 0.0) + full_probe_wall_s
            )
            event["selection_elapsed_s"] = float(perf_counter() - select_t0)
            event["selection_overhead_s"] = float(
                max(
                    0.0,
                    float(event["selection_elapsed_s"])
                    - float(event["total_probe_wall_s"]),
                )
            )
            self.selector_events.append(event)

            return (
                best_candidate,
                np.asarray(best_probe["optimal_point"], dtype=float).tolist(),
                None,
            )

        def _family_select_candidate(
            self,
            *,
            iteration: int,
            grads: list[float],
            abs_grads: list[float],
            sorted_candidates: list[int],
            prev_op_indices: list[int],
            trigger_context: dict[str, Any],
        ) -> tuple[int | None, list[float] | None, str | None]:
            shortlist: list[int] = []
            skipped_cyclic: list[int] = []
            termination_reason = None

            for candidate_idx in sorted_candidates:
                candidate_grad = abs_grads[candidate_idx]
                if candidate_grad < self.gradient_threshold:
                    if iteration == 1 and not shortlist:
                        raise AlgorithmError(
                            "All gradients below threshold in first iteration. "
                            "Try tighter threshold or different operator pool."
                        )
                    termination_reason = "converged"
                    break

                if self.check_cyclicity and self._check_cyclicity(
                    prev_op_indices + [candidate_idx]
                ):
                    skipped_cyclic.append(candidate_idx)
                    continue

                shortlist.append(candidate_idx)
                if len(shortlist) >= config.family_top_n:
                    break

            if not shortlist:
                return None, None, termination_reason or "cyclicity"

            recent_window = prev_op_indices[-config.generic_recent_window :]
            channel_state: dict[tuple[int, int], dict[str, Any]] = {}

            def ensure_channel_state(channel: tuple[int, int]) -> dict[str, Any]:
                if channel not in channel_state:
                    catalog_entry = self._diagonal_channel_catalog.get(channel, {})
                    channel_state[channel] = {
                        "alpha_seen": False,
                        "beta_seen": False,
                        "pair_seen": False,
                        "history_touch_count": 0,
                        "recent_touch_count": 0,
                        "recent_diagonal_member_count": 0,
                        "recent_mixed_touch_count": 0,
                        "top_touch_count": 0,
                        "top_diagonal_member_count": 0,
                        "top_mixed_touch_count": 0,
                        "top_max_abs_grad": 0.0,
                        "alpha_single_idx": catalog_entry.get("alpha_single_idx"),
                        "beta_single_idx": catalog_entry.get("beta_single_idx"),
                        "pair_double_idx": catalog_entry.get("pair_double_idx"),
                    }
                return channel_state[channel]

            def register_observation(
                *,
                idx: int,
                source: str,
                abs_grad: float | None = None,
            ) -> None:
                label = safe_excitation_label(idx, excitation_labels)
                parsed = parse_excitation_label(label)
                for channel in parsed["coupled_channels"]:
                    entry = ensure_channel_state(channel)
                    entry["history_touch_count"] += 1 if source == "history" else 0
                    if source == "recent":
                        entry["recent_touch_count"] += 1
                    elif source == "top":
                        entry["top_touch_count"] += 1
                        if abs_grad is not None:
                            entry["top_max_abs_grad"] = max(
                                float(entry["top_max_abs_grad"]),
                                float(abs_grad),
                            )

                    if parsed["kind"] == "mixed_double":
                        if source == "recent":
                            entry["recent_mixed_touch_count"] += 1
                        elif source == "top":
                            entry["top_mixed_touch_count"] += 1

                diagonal_channel = parsed.get("diagonal_channel")
                if diagonal_channel is None:
                    return
                entry = ensure_channel_state(diagonal_channel)
                if source == "recent":
                    entry["recent_diagonal_member_count"] += 1
                elif source == "top":
                    entry["top_diagonal_member_count"] += 1

                if parsed["kind"] == "single":
                    spin_channel = parsed["spin_channels"][0]
                    if spin_channel == "a->a":
                        entry["alpha_seen"] = True
                    elif spin_channel == "b->b":
                        entry["beta_seen"] = True
                elif parsed["kind"] == "pair_double":
                    entry["pair_seen"] = True

            for idx in prev_op_indices:
                register_observation(idx=idx, source="history")
            for idx in recent_window:
                register_observation(idx=idx, source="recent")
            for idx in shortlist:
                register_observation(idx=idx, source="top", abs_grad=abs_grads[idx])

            event: dict[str, Any] = {
                "iteration": int(iteration),
                "trigger_mode": "family_channel_topn",
                "shortlist_top_n": int(config.family_top_n),
                "shortlist": [],
                "shortlist_count": int(len(shortlist)),
                "skipped_cyclic": [int(idx) for idx in skipped_cyclic],
                "trigger_context": trigger_context,
                "injected_candidates": [],
                "family_channel_scores": [],
            }

            shortlist_set = set(shortlist)
            for candidate_idx in shortlist:
                event["shortlist"].append(
                    {
                        "idx": int(candidate_idx),
                        "rank": int(shortlist.index(candidate_idx) + 1),
                        "gradient": float(grads[candidate_idx]),
                        "abs_gradient": float(abs_grads[candidate_idx]),
                        "excitation": safe_excitation_label(
                            candidate_idx, excitation_labels
                        ),
                    }
                )

            completion_candidates: list[
                tuple[tuple[Any, ...], int, tuple[int, int], str]
            ] = []
            for channel, state in channel_state.items():
                pair_double_idx = state.get("pair_double_idx")
                if pair_double_idx is None or bool(state["pair_seen"]):
                    continue
                if state["alpha_seen"] and state["beta_seen"]:
                    status = "activated_incomplete"
                    state_priority = 2
                elif state["alpha_seen"] or state["beta_seen"]:
                    status = "seeded_incomplete"
                    state_priority = 1
                else:
                    continue

                candidate_idx = int(pair_double_idx)
                if self.check_cyclicity and self._check_cyclicity(
                    prev_op_indices + [candidate_idx]
                ):
                    continue

                in_top_n = candidate_idx in shortlist_set
                if not in_top_n:
                    event["injected_candidates"].append(
                        {
                            "idx": int(candidate_idx),
                            "excitation": safe_excitation_label(
                                candidate_idx, excitation_labels
                            ),
                            "channel": f"{channel[0]}->{channel[1]}",
                            "reason": status,
                        }
                    )
                channel_score = (
                    state_priority,
                    int(state["recent_diagonal_member_count"])
                    + int(state["recent_touch_count"]),
                    int(state["top_diagonal_member_count"])
                    + int(state["top_touch_count"]),
                    float(state["top_max_abs_grad"]),
                    1 if in_top_n else 0,
                )
                event["family_channel_scores"].append(
                    {
                        "channel": f"{channel[0]}->{channel[1]}",
                        "status": status,
                        "candidate_idx": int(candidate_idx),
                        "candidate_excitation": safe_excitation_label(
                            candidate_idx, excitation_labels
                        ),
                        "score": [
                            int(channel_score[0]),
                            int(channel_score[1]),
                            int(channel_score[2]),
                            float(channel_score[3]),
                            int(channel_score[4]),
                        ],
                        "alpha_seen": bool(state["alpha_seen"]),
                        "beta_seen": bool(state["beta_seen"]),
                        "pair_seen": bool(state["pair_seen"]),
                        "recent_touch_count": int(state["recent_touch_count"]),
                        "recent_diagonal_member_count": int(
                            state["recent_diagonal_member_count"]
                        ),
                        "top_touch_count": int(state["top_touch_count"]),
                        "top_diagonal_member_count": int(
                            state["top_diagonal_member_count"]
                        ),
                        "top_max_abs_grad": float(state["top_max_abs_grad"]),
                        "in_top_n": bool(in_top_n),
                    }
                )
                completion_candidates.append(
                    (channel_score, candidate_idx, channel, status)
                )

            if completion_candidates:
                completion_candidates.sort(
                    key=lambda item: (
                        item[0][0],
                        item[0][1],
                        item[0][2],
                        item[0][3],
                        item[0][4],
                        -item[1],
                    ),
                    reverse=True,
                )
                _, selected_idx, selected_channel, selected_status = (
                    completion_candidates[0]
                )
                event["selected_idx"] = int(selected_idx)
                event["selected_rank"] = int(
                    next(
                        (
                            rank + 1
                            for rank, idx in enumerate(sorted_candidates)
                            if idx == selected_idx
                        ),
                        len(sorted_candidates) + 1,
                    )
                )
                event["selected_excitation"] = safe_excitation_label(
                    selected_idx, excitation_labels
                )
                event["selection_reason"] = "family_pair_completion"
                event["selected_channel"] = (
                    f"{selected_channel[0]}->{selected_channel[1]}"
                )
                event["selected_channel_status"] = selected_status
                event["selection_elapsed_s"] = 0.0
                self.selector_events.append(event)
                return selected_idx, None, None

            fallback_idx, fallback_reason = self._standard_select_candidate(
                iteration=iteration,
                abs_grads=abs_grads,
                sorted_candidates=sorted_candidates,
                prev_op_indices=prev_op_indices,
            )
            if fallback_idx is None:
                return None, None, fallback_reason

            event["selected_idx"] = int(fallback_idx)
            event["selected_rank"] = int(
                next(
                    (
                        rank + 1
                        for rank, idx in enumerate(sorted_candidates)
                        if idx == fallback_idx
                    ),
                    len(sorted_candidates) + 1,
                )
            )
            event["selected_excitation"] = safe_excitation_label(
                fallback_idx, excitation_labels
            )
            event["selection_reason"] = "fallback_top_gradient"
            event["selected_channel"] = None
            event["selected_channel_status"] = None
            event["selection_elapsed_s"] = 0.0
            self.selector_events.append(event)
            return fallback_idx, None, None

        def compute_minimum_eigenvalue(self, operator, aux_operators=None):
            if self.verbose >= 1:
                print(f"  [{log_prefix}] Converting Hamiltonian to sparse matrix...")
            t0 = perf_counter()
            h_sparse = cast(Any, operator).to_matrix(sparse=True)
            h_nnz = (
                int(h_sparse.nnz)
                if hasattr(h_sparse, "nnz")
                else int(np.count_nonzero(h_sparse.toarray()))
            )
            t_ham = perf_counter() - t0
            if self.verbose >= 1:
                print(
                    f"  [{log_prefix}] Hamiltonian: {h_sparse.shape}, "
                    f"nnz={h_nnz}, took {t_ham:.1f}s"
                )

            if self.verbose >= 1:
                print(
                    f"  [{log_prefix}] Converting {len(self._excitation_pool)} pool operators "
                    "to sparse matrices..."
                )
            t0 = perf_counter()
            op_matrices = [
                cast(Any, op).to_matrix(sparse=True) for op in self._excitation_pool
            ]
            t_ops = perf_counter() - t0
            if self.verbose >= 1:
                print(f"  [{log_prefix}] Pool conversion took {t_ops:.1f}s")

            self.solver.ansatz = self._initial_state
            theta: list[float] = []
            self._excitation_list = []
            self._selected_operator_indices = []
            self._iteration_gradient_history = []
            self._iteration_timing_history = []
            self.selector_events = []
            self._generic_lookahead_activated = False
            self._probe_new_param_seed_cache = {}
            self._probe_backend = create_probe_backend(
                backend_name=config.probe_backend,
                aer_max_parallel_threads=config.probe_aer_threads,
                aer_precision=config.probe_aer_precision,
            )
            if hasattr(self.solver, "timing_history"):
                cast(Any, self.solver).timing_history = []
            if hasattr(self.solver, "last_timing_breakdown"):
                cast(Any, self.solver).last_timing_breakdown = None
            if self.verbose >= 1:
                print(f"  [{log_prefix}] Probe backend: {self._probe_backend.name}")
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
                    print(
                        f"\n  [{log_prefix}] --- Iteration {iteration}/{max_iter} ---"
                    )

                iteration_timing: dict[str, Any] = {"iteration": int(iteration)}

                t_grad_start = perf_counter()
                psi = self._get_statevector(
                    self.solver.ansatz, theta if theta else None
                )
                grads = self._compute_gradients_fast(psi, h_sparse, op_matrices)
                t_grad = perf_counter() - t_grad_start
                iteration_timing["gradient_time_s"] = float(t_grad)

                abs_grads = [abs(g) for g in grads]
                sorted_candidates = sorted(
                    range(len(abs_grads)), key=lambda i: abs_grads[i], reverse=True
                )
                iteration_diagnostic = self._make_iteration_gradient_diagnostic(
                    iteration=iteration,
                    gradients=grads,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                )

                if self.verbose >= 1:
                    self._log_iteration_gradient_diagnostic(
                        prefix=log_prefix,
                        gradient_time_s=t_grad,
                        abs_gradients=abs_grads,
                        sorted_candidates=sorted_candidates,
                        diagnostic=iteration_diagnostic,
                    )

                generic_trigger_context = self._generic_trigger_context(
                    iteration=iteration,
                    abs_grads=abs_grads,
                    sorted_candidates=sorted_candidates,
                    prev_op_indices=prev_op_indices,
                )
                if (
                    selector_mode in {"lookahead_generic", "family_generic"}
                    and not self._generic_lookahead_activated
                    and generic_trigger_context["should_activate"]
                ):
                    self._generic_lookahead_activated = True
                    if self.verbose >= 1:
                        print(
                            f"  [{log_prefix}] Generic lookahead trigger activated: "
                            f"recent_overlap={generic_trigger_context['recent_overlap_count']}, "
                            f"cyclic_topk={generic_trigger_context['cyclic_candidate_count']}, "
                            f"recent_window={generic_trigger_context['recent_window']}"
                        )

                lookahead_active = (
                    selector_mode in {"lookahead_generic", "family_generic"}
                    and self._generic_lookahead_activated
                )
                iteration_timing["lookahead_active"] = bool(lookahead_active)
                t_select_start = perf_counter()
                if lookahead_active:
                    if selector_mode == "family_generic":
                        (
                            selected_idx,
                            selected_initial_point,
                            termination_reason,
                        ) = self._family_select_candidate(
                            iteration=iteration,
                            grads=grads,
                            abs_grads=abs_grads,
                            sorted_candidates=sorted_candidates,
                            prev_op_indices=prev_op_indices,
                            trigger_context=generic_trigger_context,
                        )
                    else:
                        (
                            selected_idx,
                            selected_initial_point,
                            termination_reason,
                        ) = self._lookahead_select_candidate(
                            iteration=iteration,
                            grads=grads,
                            abs_grads=abs_grads,
                            sorted_candidates=sorted_candidates,
                            prev_op_indices=prev_op_indices,
                            theta=theta,
                            operator=operator,
                            h_sparse=h_sparse,
                            trigger_context=generic_trigger_context,
                        )
                    if selected_idx is not None and self.verbose >= 1:
                        print(
                            f"  [{log_prefix}] Lookahead selected idx={selected_idx} "
                            f"(|grad|={abs_grads[selected_idx]:.6e}) from top-{config.top_k}"
                        )
                else:
                    selected_initial_point = None
                    selected_idx, termination_reason = self._standard_select_candidate(
                        iteration=iteration,
                        abs_grads=abs_grads,
                        sorted_candidates=sorted_candidates,
                        prev_op_indices=prev_op_indices,
                    )
                selection_wall_s = float(perf_counter() - t_select_start)
                probe_time_s = 0.0
                probe_work_s = 0.0
                if lookahead_active and self.selector_events:
                    last_event = self.selector_events[-1]
                    if last_event.get("iteration") == iteration:
                        probe_time_s = float(
                            last_event.get(
                                "total_probe_wall_s",
                                last_event.get(
                                    "total_probe_elapsed_s",
                                    last_event.get("full_probe_elapsed_sum_s", 0.0),
                                ),
                            )
                        )
                        probe_work_s = float(
                            last_event.get(
                                "total_probe_elapsed_s",
                                last_event.get("full_probe_elapsed_sum_s", 0.0),
                            )
                        )
                iteration_timing["selection_wall_s"] = selection_wall_s
                iteration_timing["probe_time_s"] = probe_time_s
                iteration_timing["probe_work_s"] = probe_work_s
                iteration_timing["selection_time_s"] = max(
                    0.0, selection_wall_s - probe_time_s
                )
                if lookahead_active and self.selector_events:
                    last_event = self.selector_events[-1]
                    if last_event.get("iteration") == iteration and self.verbose >= 1:
                        print(
                            f"  [{log_prefix}] Probe batch: "
                            f"shortlist={int(last_event.get('shortlist_count', 0))}, "
                            f"refined={int(last_event.get('refined_candidate_count', 0))}, "
                            f"probe_calls={int(last_event.get('full_probe_call_count', 0))}, "
                            f"probe_nfev={int(last_event.get('total_probe_nfev', 0))}, "
                            f"probe_wall={float(last_event.get('total_probe_wall_s', 0.0)):.1f}s, "
                            f"probe_sum={float(last_event.get('total_probe_elapsed_s', 0.0)):.1f}s, "
                            f"selection_only={float(iteration_timing['selection_time_s']):.1f}s"
                        )

                if selected_idx is None:
                    self._finalize_iteration_gradient_diagnostic(
                        iteration_diagnostic,
                        gradients=grads,
                        abs_gradients=abs_grads,
                        sorted_candidates=sorted_candidates,
                        termination_reason=termination_reason or "cyclicity",
                    )
                    if termination_reason == "converged":
                        termination = TerminationCriterion.CONVERGED
                    else:
                        termination = TerminationCriterion.CYCLICITY
                    iteration_timing["termination_reason"] = (
                        termination_reason or "cyclicity"
                    )
                    iteration_timing["total_iteration_s"] = float(
                        iteration_timing.get("gradient_time_s", 0.0)
                    ) + float(iteration_timing.get("selection_time_s", 0.0))
                    self._iteration_timing_history.append(iteration_timing)
                    break

                max_idx = selected_idx
                if max_idx is None:
                    raise RuntimeError("No ADAPT operator selected")
                max_grad_val = abs_grads[max_idx]
                iteration_timing["selected_idx"] = int(max_idx)
                self._finalize_iteration_gradient_diagnostic(
                    iteration_diagnostic,
                    gradients=grads,
                    abs_gradients=abs_grads,
                    sorted_candidates=sorted_candidates,
                    selected_idx=max_idx,
                )
                prev_op_indices.append(max_idx)
                self._selected_operator_indices = prev_op_indices.copy()

                self._excitation_list.append(self._excitation_pool[max_idx])
                if selected_initial_point is None:
                    theta.append(0.0)
                else:
                    theta = list(selected_initial_point)

                self.solver.ansatz = self._build_ansatz()
                self.solver.initial_point = np.array(theta, dtype=float)

                t_vqe_start = perf_counter()
                prev_raw_vqe_result = raw_vqe_result
                raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
                optimal_point = raw_vqe_result.optimal_point
                if optimal_point is None:
                    raise RuntimeError("FastAdapt VQE returned no optimal point")
                theta = optimal_point.tolist()
                t_vqe = perf_counter() - t_vqe_start
                iteration_timing["vqe_time_s"] = float(t_vqe)
                iteration_timing["vqe_cost_function_evals"] = int(
                    raw_vqe_result.cost_function_evals or 0
                )
                solver_breakdown = getattr(self.solver, "last_timing_breakdown", None)
                if isinstance(solver_breakdown, dict):
                    iteration_timing["inner_vqe_breakdown"] = {
                        key: (
                            float(value)
                            if isinstance(value, (int, float, np.integer, np.floating))
                            else value
                        )
                        for key, value in solver_breakdown.items()
                    }

                if self.selector_events:
                    last_event = self.selector_events[-1]
                    if last_event.get("iteration") == iteration:
                        last_event["actual_post_vqe_energy"] = float(
                            _complex_eigenvalue(raw_vqe_result.eigenvalue).real
                        )

                if self.verbose >= 1:
                    print(
                        f"  [{log_prefix}] VQE: {t_vqe:.1f}s, "
                        f"E={_complex_eigenvalue(raw_vqe_result.eigenvalue).real:.10f}, "
                        f"evals={raw_vqe_result.cost_function_evals}"
                    )

                if iteration > 1:
                    eigenvalue_diff = abs(
                        _complex_eigenvalue(raw_vqe_result.eigenvalue) - history[-1]
                    )
                    if eigenvalue_diff < self.eigenvalue_threshold:
                        self._excitation_list.pop()
                        theta.pop()
                        self.solver.ansatz = self._build_ansatz()
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
                if self.verbose >= 1:
                    selector_totals = summarize_selector_events(self.selector_events)
                    main_vqe_step_count = sum(
                        1
                        for item in self._iteration_timing_history
                        if float(item.get("vqe_time_s", 0.0)) > 0.0
                    )
                    main_vqe_nfev_total = sum(
                        int(item.get("vqe_cost_function_evals", 0) or 0)
                        for item in self._iteration_timing_history
                    )
                    main_vqe_total_s = sum(
                        float(item.get("vqe_time_s", 0.0))
                        for item in self._iteration_timing_history
                    )
                    grad_total_s = sum(
                        float(item.get("gradient_time_s", 0.0))
                        for item in self._iteration_timing_history
                    )
                    print(
                        f"  [{log_prefix}] Bottleneck snapshot: "
                        f"main_vqe_calls={main_vqe_step_count}, "
                        f"probe_calls={int(selector_totals['probe_call_count'])}, "
                        f"total_vqe_like_calls={main_vqe_step_count + int(selector_totals['probe_call_count'])}, "
                        f"main_vqe_nfev={main_vqe_nfev_total}, "
                        f"probe_nfev={int(selector_totals['probe_nfev_total'])}, "
                        f"main_vqe_time={main_vqe_total_s:.1f}s, "
                        f"probe_wall={float(selector_totals['probe_wall_s']):.1f}s, "
                        f"probe_work={float(selector_totals['probe_elapsed_s']):.1f}s, "
                        f"grad_time={grad_total_s:.1f}s"
                    )
                history.append(_complex_eigenvalue(raw_vqe_result.eigenvalue))
            else:
                termination = TerminationCriterion.MAXIMUM

            result = cast(Any, FastAdaptVQEResult())
            if raw_vqe_result is not None:
                result.combine(raw_vqe_result)
            result.num_iterations = iteration
            result.final_max_gradient = max_grad_val
            result.termination_criterion = termination
            result.eigenvalue_history = history
            result.iteration_gradient_history = list(self._iteration_gradient_history)
            result.iteration_timing_history = list(self._iteration_timing_history)
            result.selector_events = list(self.selector_events)
            result.setup_timings = dict(setup_timings)
            result.timing_summary = summarize_fastadapt_timings(
                setup_timings=setup_timings,
                iteration_timings=self._iteration_timing_history,
                selector_events=self.selector_events,
            )
            self._last_result = result

            solve_record = {
                "solve_index": len(type(self).solve_records) + 1,
                "case": case_name,
                "selector_mode": selector_mode,
                "n_operators": len(self._selected_operator_indices),
                "selected_indices": list(self._selected_operator_indices),
                "selected_excitations": safe_excitation_labels(
                    list(self._selected_operator_indices), excitation_labels
                ),
                "selector_events": list(self.selector_events),
                "iteration_gradient_history": list(self._iteration_gradient_history),
                "iteration_timing_history": list(self._iteration_timing_history),
                "setup_timings": dict(setup_timings),
                "timing_summary": dict(result.timing_summary),
                "termination": str(result.termination_criterion),
                "final_fragment_energy": (
                    float(_complex_eigenvalue(result.eigenvalue).real)
                    if getattr(result, "eigenvalue", None) is not None
                    else None
                ),
            }
            type(self).solve_records.append(solve_record)
            if self.verbose >= 1:
                print(
                    f"  [{log_prefix}] Timing summary: "
                    f"setup={result.timing_summary['hamiltonian_to_sparse_s'] + result.timing_summary['pool_to_sparse_s']:.1f}s, "
                    f"grad={result.timing_summary['gradient_eval_total_s']:.1f}s, "
                    f"select={result.timing_summary['selection_total_s']:.1f}s, "
                    f"vqe={result.timing_summary['inner_vqe_total_s']:.1f}s, "
                    f"probe_wall={result.timing_summary['probe_total_s']:.1f}s, "
                    f"probe_work={result.timing_summary.get('probe_work_total_s', 0.0):.1f}s, "
                    f"dominant={result.timing_summary['dominant_phase']}"
                )

            return result

    LookaheadFastAdaptVQE.solve_records = []
    return LookaheadFastAdaptVQE


@contextmanager
def patched_fast_adapt(
    config: SelectorConfig,
    case_name: str,
    selector_mode: str,
    excitation_labels: list[str],
    log_prefix: str = "EthaneLookahead",
):
    original = fast_adapt_module.FastAdaptVQE
    guided_cls = make_lookahead_fast_adapt(
        config=config,
        case_name=case_name,
        selector_mode=selector_mode,
        excitation_labels=excitation_labels,
        log_prefix=log_prefix,
    )
    fast_adapt_module.FastAdaptVQE = guided_cls
    try:
        yield guided_cls
    finally:
        fast_adapt_module.FastAdaptVQE = original


@dataclass(slots=True)
class ProbeBackendConfig:
    """Probe backend settings shared by experiment harnesses."""

    name: str
    aer_device: str = "CPU"
    aer_max_parallel_threads: int = 4
    aer_precision: str = "double"


class BaseProbeBackend:
    """Common interface for shortlist probe backends."""

    def __init__(self, config: ProbeBackendConfig):
        self.config = config
        self.name = config.name

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def coarse_scan_many(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def probe_many(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        t0 = perf_counter()
        results: list[dict[str, Any]] = []
        total_nfev = 0
        for task in tasks:
            probe = self.probe(
                ansatz=task["ansatz"],
                operator=task["operator"],
                h_sparse=task["h_sparse"],
                initial_point=task["initial_point"],
                maxiter=task["maxiter"],
                ftol=task["ftol"],
            )
            results.append(probe)
            total_nfev += int(probe.get("nfev", 0))
        return {
            "results": results,
            "elapsed_s": perf_counter() - t0,
            "n_evals": int(total_nfev),
            "batched": False,
            "backend": self.name,
        }


class _DirectSVProbeEvaluator:
    """Exact statevector probe objective with per-task memoization."""

    def __init__(self, *, ansatz, h_sparse):
        self.ansatz = ansatz
        self.h_sparse = h_sparse
        self.param_list = list(ansatz.parameters)
        self.energy_cache: dict[tuple[float, ...], float] = {}
        self.unique_eval_count = 0
        self.cache_hits = 0

    @staticmethod
    def theta_key(theta: np.ndarray) -> tuple[float, ...]:
        return tuple(float(x) for x in np.asarray(theta, dtype=float).reshape(-1))

    def energy(self, theta: np.ndarray) -> float:
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        theta_key = self.theta_key(theta_arr)
        if theta_key in self.energy_cache:
            self.cache_hits += 1
            return float(self.energy_cache[theta_key])

        if theta_arr.size:
            bound = self.ansatz.assign_parameters(dict(zip(self.param_list, theta_arr)))
        else:
            bound = self.ansatz
        psi = Statevector(bound).data
        energy = float(np.vdot(psi, self.h_sparse @ psi).real)
        self.energy_cache[theta_key] = energy
        self.unique_eval_count += 1
        return energy


def _build_aer_estimator_options(
    *,
    config: ProbeBackendConfig,
    run_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    precision = str(config.aer_precision).lower()
    if precision not in {"single", "double"}:
        raise ValueError(
            f"Invalid Aer precision '{config.aer_precision}'. "
            "Use 'single' or 'double'."
        )

    backend_options: dict[str, Any] = {
        "method": "statevector",
        "device": config.aer_device.upper(),
        "precision": precision,
    }
    if config.aer_device.upper() != "GPU":
        backend_options["max_parallel_threads"] = config.aer_max_parallel_threads

    options: dict[str, Any] = {
        "default_precision": 0.0,
        "backend_options": backend_options,
    }
    if run_options:
        options["run_options"] = run_options
    return options


def _sv_energy(ansatz, h_sparse, theta: np.ndarray) -> float:
    """Evaluate energy via Statevector + sparse H. Top-level for pickling."""
    evaluator = _DirectSVProbeEvaluator(ansatz=ansatz, h_sparse=h_sparse)
    return evaluator.energy(theta)


def _parameter_shift_gradient(ansatz, h_sparse, theta: np.ndarray) -> np.ndarray:
    """Compute analytical gradient via parameter-shift rule."""
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += np.pi / 2
        theta_minus = theta.copy()
        theta_minus[i] -= np.pi / 2
        grad[i] = (
            _sv_energy(ansatz, h_sparse, theta_plus)
            - _sv_energy(ansatz, h_sparse, theta_minus)
        ) / 2.0
    return grad


def _run_single_one_d_probe(
    *,
    ansatz,
    h_sparse,
    initial_point,
    backend_name: str,
    alpha_lower: float,
    alpha_upper: float,
    xatol: float,
    maxiter: int,
) -> dict[str, Any]:
    """Run a frozen-background 1D search on the newly appended parameter."""
    param_list = list(ansatz.parameters)
    n_params = len(param_list)
    theta0 = normalize_initial_point(initial_point, n_params)
    t0 = perf_counter()

    if n_params == 0:
        energy = _sv_energy(ansatz, h_sparse, np.array([], dtype=float))
        return {
            "energy": energy,
            "optimal_point": np.array([], dtype=float),
            "success": True,
            "nit": 0,
            "nfev": 0,
            "message": "zero-parameter ansatz",
            "backend": backend_name,
            "elapsed_s": perf_counter() - t0,
        }

    eval_count = 0
    best_energy = float("inf")
    best_alpha = float(theta0[-1])

    def objective(alpha: float) -> float:
        nonlocal eval_count, best_energy, best_alpha
        trial = theta0.copy()
        trial[-1] = float(alpha)
        eval_count += 1
        energy = _sv_energy(ansatz, h_sparse, trial)
        if energy < best_energy:
            best_energy = energy
            best_alpha = float(alpha)
        return energy

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(alpha_lower, alpha_upper),
        options={"xatol": xatol, "maxiter": maxiter},
    )

    if bool(getattr(result, "success", False)):
        alpha = float(result.x)
        energy = float(result.fun)
    else:
        alpha = best_alpha
        energy = best_energy

    optimal_theta = theta0.copy()
    optimal_theta[-1] = alpha
    return {
        "energy": energy,
        "optimal_point": optimal_theta,
        "success": bool(getattr(result, "success", False)),
        "nit": int(getattr(result, "nit", 1)),
        "nfev": int(getattr(result, "nfev", eval_count)),
        "message": str(getattr(result, "message", "")),
        "backend": backend_name,
        "elapsed_s": perf_counter() - t0,
    }


def _quadratic_local_prediction(
    *,
    energy_minus: float,
    energy_zero: float,
    energy_plus: float,
    delta: float,
    alpha_cap: float,
) -> tuple[float, float, bool]:
    """Predict the local minimum of a 1D quadratic fit around alpha=0."""
    gradient_fd = (energy_plus - energy_minus) / (2.0 * delta)
    curvature = (energy_plus - 2.0 * energy_zero + energy_minus) / (delta**2)
    if curvature > 1e-10:
        alpha = float(np.clip(-gradient_fd / curvature, -alpha_cap, alpha_cap))
        energy = float(
            energy_zero + gradient_fd * alpha + 0.5 * curvature * alpha * alpha
        )
        return energy, alpha, True

    best_sample = min(
        (
            (energy_minus, -delta),
            (energy_zero, 0.0),
            (energy_plus, delta),
        ),
        key=lambda item: item[0],
    )
    return float(best_sample[0]), float(best_sample[1]), False


def _run_single_quadratic_probe(
    *,
    ansatz,
    h_sparse,
    initial_point,
    backend_name: str,
    delta: float,
    alpha_cap: float,
) -> dict[str, Any]:
    """Run a local quadratic surrogate on the newly appended parameter."""
    param_list = list(ansatz.parameters)
    n_params = len(param_list)
    theta0 = normalize_initial_point(initial_point, n_params)
    t0 = perf_counter()

    if n_params == 0:
        energy = _sv_energy(ansatz, h_sparse, np.array([], dtype=float))
        return {
            "energy": energy,
            "optimal_point": np.array([], dtype=float),
            "success": True,
            "nit": 0,
            "nfev": 0,
            "message": "zero-parameter ansatz",
            "backend": backend_name,
            "elapsed_s": perf_counter() - t0,
        }

    theta_minus = theta0.copy()
    theta_plus = theta0.copy()
    theta_minus[-1] = -delta
    theta_plus[-1] = delta

    energy_zero = _sv_energy(ansatz, h_sparse, theta0)
    energy_minus = _sv_energy(ansatz, h_sparse, theta_minus)
    energy_plus = _sv_energy(ansatz, h_sparse, theta_plus)
    predicted_energy, predicted_alpha, convex = _quadratic_local_prediction(
        energy_minus=energy_minus,
        energy_zero=energy_zero,
        energy_plus=energy_plus,
        delta=delta,
        alpha_cap=alpha_cap,
    )

    optimal_theta = theta0.copy()
    optimal_theta[-1] = predicted_alpha
    return {
        "energy": predicted_energy,
        "optimal_point": optimal_theta,
        "success": True,
        "nit": 1,
        "nfev": 3,
        "message": (
            "quadratic local fit"
            if convex
            else "quadratic local fit fell back to best sample"
        ),
        "backend": backend_name,
        "elapsed_s": perf_counter() - t0,
    }


def _run_single_probe(
    *,
    ansatz,
    h_sparse,
    initial_point,
    maxiter: int,
    ftol: float,
    backend_name: str,
    use_analytical_grad: bool = False,
) -> dict[str, Any]:
    """Run a single SLSQP probe. Top-level function for ProcessPoolExecutor."""
    evaluator = _DirectSVProbeEvaluator(ansatz=ansatz, h_sparse=h_sparse)
    n_params = len(evaluator.param_list)
    theta0 = normalize_initial_point(initial_point, n_params)
    t0 = perf_counter()

    if n_params == 0:
        energy = evaluator.energy(np.array([], dtype=float))
        return {
            "energy": energy,
            "optimal_point": np.array([], dtype=float),
            "success": True,
            "nit": 0,
            "nfev": 0,
            "message": "zero-parameter ansatz",
            "backend": backend_name,
            "elapsed_s": perf_counter() - t0,
        }

    eval_count = 0
    best_energy = float("inf")
    best_theta = theta0.copy()

    def cost_fn(theta: np.ndarray) -> float:
        nonlocal eval_count, best_energy, best_theta
        theta = np.asarray(theta, dtype=float)
        eval_count += 1
        energy = evaluator.energy(theta)
        if energy < best_energy:
            best_energy = energy
            best_theta = theta.copy()
        return energy

    jac = None
    if use_analytical_grad:

        def jac(theta: np.ndarray) -> np.ndarray:
            return _parameter_shift_gradient(
                ansatz, h_sparse, np.asarray(theta, dtype=float)
            )

    result = minimize(
        cost_fn,
        theta0,
        method="SLSQP",
        jac=jac,
        bounds=[(-np.pi, np.pi)] * n_params,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    if result.success:
        optimal_theta = np.asarray(result.x, dtype=float)
        optimal_energy = float(result.fun)
    else:
        optimal_theta = best_theta
        optimal_energy = best_energy

    suffix = "+grad" if use_analytical_grad else ""
    return {
        "energy": optimal_energy,
        "optimal_point": optimal_theta,
        "success": bool(result.success),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", eval_count)),
        "n_unique_evals": int(evaluator.unique_eval_count),
        "cache_hits": int(evaluator.cache_hits),
        "message": str(getattr(result, "message", "")),
        "backend": f"{backend_name}{suffix}",
        "elapsed_s": perf_counter() - t0,
    }


def _run_single_probe_exact_jac(
    *,
    ansatz,
    operator,
    h_sparse,
    initial_point,
    maxiter: int,
    ftol: float,
    backend_name: str,
    candidate_idx: int | None = None,
    candidate_label: str | None = None,
    log_prefix: str = "LookaheadADAPT",
    progress_interval_s: float = 60.0,
) -> dict[str, Any]:
    """Run a full direct-SV probe with exact reverse-mode Jacobians."""
    evaluator = _DirectSVProbeEvaluator(ansatz=ansatz, h_sparse=h_sparse)
    n_params = len(evaluator.param_list)
    theta0 = normalize_initial_point(initial_point, n_params)
    t0 = perf_counter()
    last_progress_log_t = t0

    def emit_probe_log(message: str, *, force: bool = False) -> None:
        nonlocal last_progress_log_t
        now = perf_counter()
        if not force and now - last_progress_log_t < progress_interval_s:
            return
        _probe_log(
            log_prefix=log_prefix,
            backend_name=backend_name,
            candidate_idx=candidate_idx,
            candidate_label=candidate_label,
            message=message,
        )
        last_progress_log_t = now

    emit_probe_log(
        (
            f"start n_params={n_params}, maxiter={maxiter}, ftol={ftol:.0e}, "
            f"exact_jacobian={'yes' if ReverseEstimatorGradient is not None else 'no'}"
        ),
        force=True,
    )

    if n_params == 0:
        energy = evaluator.energy(np.array([], dtype=float))
        emit_probe_log("done zero-parameter ansatz", force=True)
        return {
            "energy": energy,
            "optimal_point": np.array([], dtype=float),
            "success": True,
            "nit": 0,
            "nfev": 0,
            "njev": 0,
            "n_unique_evals": int(evaluator.unique_eval_count),
            "cache_hits": int(evaluator.cache_hits),
            "used_exact_jacobian": False,
            "jacobian_total_s": 0.0,
            "message": "zero-parameter ansatz",
            "backend": backend_name,
            "elapsed_s": perf_counter() - t0,
        }

    eval_count = 0
    best_energy = float("inf")
    best_theta = theta0.copy()
    grad_cache: dict[tuple[float, ...], np.ndarray] = {}
    grad_cache_hits = 0
    jacobian_total_s = 0.0
    njev_counter = 0
    jacobian_fn = None
    callback_iterations = 0

    def cost_fn(theta: np.ndarray) -> float:
        nonlocal eval_count, best_energy, best_theta
        theta = np.asarray(theta, dtype=float)
        eval_count += 1
        energy = evaluator.energy(theta)
        if energy < best_energy:
            best_energy = energy
            best_theta = theta.copy()
        if eval_count == 1:
            emit_probe_log(
                (
                    f"first-energy elapsed={perf_counter() - t0:.1f}s "
                    f"energy={energy:.10f}"
                ),
                force=True,
            )
        else:
            emit_probe_log(
                (
                    f"heartbeat stage=energy elapsed={perf_counter() - t0:.1f}s "
                    f"nfev={eval_count} njev={njev_counter} best={best_energy:.10f} "
                    f"unique={evaluator.unique_eval_count} e_cache_hits={evaluator.cache_hits}"
                )
            )
        return energy

    if ReverseEstimatorGradient is not None:
        reverse_gradient = ReverseEstimatorGradient()

        def exact_jacobian(theta: np.ndarray) -> np.ndarray:
            nonlocal grad_cache_hits, jacobian_total_s, njev_counter
            theta_key = evaluator.theta_key(theta)
            cached = grad_cache.get(theta_key)
            if cached is not None:
                grad_cache_hits += 1
                return np.asarray(cached, dtype=float)

            jac_call_idx = njev_counter + 1
            emit_probe_log(
                (
                    f"jac-start call={jac_call_idx} elapsed={perf_counter() - t0:.1f}s "
                    f"nfev={eval_count}"
                ),
                force=(jac_call_idx == 1),
            )
            jac_t0 = perf_counter()
            stop_heartbeat: Event | None = None
            heartbeat_thread: Thread | None = None
            if progress_interval_s > 0.0:
                stop_heartbeat = Event()

                def jac_heartbeat() -> None:
                    assert stop_heartbeat is not None
                    while not stop_heartbeat.wait(progress_interval_s):
                        emit_probe_log(
                            (
                                f"jac-wait call={jac_call_idx} "
                                f"jac_elapsed={perf_counter() - jac_t0:.1f}s "
                                f"elapsed={perf_counter() - t0:.1f}s "
                                f"nfev={eval_count} njev={njev_counter} "
                                f"unique={evaluator.unique_eval_count} "
                                f"e_cache_hits={evaluator.cache_hits}"
                            ),
                            force=True,
                        )

                heartbeat_thread = Thread(
                    target=jac_heartbeat,
                    name=(
                        "probe-jac-"
                        f"{candidate_idx if candidate_idx is not None else 'na'}"
                    ),
                    daemon=True,
                )
                heartbeat_thread.start()

            try:
                job = reverse_gradient.run(
                    [ansatz],
                    [operator],
                    [list(theta_key)],
                    [evaluator.param_list],
                )
                gradient = np.asarray(job.result().gradients[0], dtype=float)
            finally:
                if stop_heartbeat is not None:
                    stop_heartbeat.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)
            jac_elapsed_s = perf_counter() - jac_t0
            jacobian_total_s += jac_elapsed_s
            njev_counter += 1
            grad_cache[theta_key] = gradient.copy()
            if jac_call_idx == 1 or jac_elapsed_s >= 30.0:
                emit_probe_log(
                    (
                        f"jac-done call={jac_call_idx} jac_s={jac_elapsed_s:.1f}s "
                        f"grad_norm={np.linalg.norm(gradient):.6e}"
                    ),
                    force=True,
                )
            return gradient

        try:
            exact_jacobian(theta0)
            jacobian_fn = exact_jacobian
            emit_probe_log("jac warmup ok", force=True)
        except Exception:  # pragma: no cover - defensive fallback
            grad_cache.clear()
            grad_cache_hits = 0
            jacobian_total_s = 0.0
            njev_counter = 0
            emit_probe_log(
                "jac warmup failed; falling back to objective-only SLSQP", force=True
            )

    def callback(theta: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        theta = np.asarray(theta, dtype=float)
        emit_probe_log(
            (
                f"iter={callback_iterations} elapsed={perf_counter() - t0:.1f}s "
                f"nfev={eval_count} njev={njev_counter} best={best_energy:.10f} "
                f"tail={theta[-1]:+.4f}"
            ),
            force=(callback_iterations == 1),
        )

    result = minimize(
        cost_fn,
        theta0,
        method="SLSQP",
        jac=jacobian_fn,
        callback=callback,
        bounds=[(-np.pi, np.pi)] * n_params,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    if result.success:
        optimal_theta = np.asarray(result.x, dtype=float)
        optimal_energy = float(result.fun)
    else:
        optimal_theta = best_theta
        optimal_energy = best_energy

    emit_probe_log(
        (
            f"done success={bool(result.success)} nit={int(getattr(result, 'nit', 0))} "
            f"nfev={int(getattr(result, 'nfev', eval_count))} "
            f"njev={int(getattr(result, 'njev', njev_counter))} "
            f"unique={evaluator.unique_eval_count} e_cache_hits={evaluator.cache_hits} "
            f"g_cache_hits={grad_cache_hits} jac_total_s={jacobian_total_s:.1f}s "
            f"best={optimal_energy:.10f} elapsed={perf_counter() - t0:.1f}s"
        ),
        force=True,
    )

    return {
        "energy": optimal_energy,
        "optimal_point": optimal_theta,
        "success": bool(result.success),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", eval_count)),
        "njev": int(getattr(result, "njev", njev_counter)),
        "n_unique_evals": int(evaluator.unique_eval_count),
        "cache_hits": int(evaluator.cache_hits),
        "gradient_cache_hits": int(grad_cache_hits),
        "used_exact_jacobian": bool(jacobian_fn is not None),
        "jacobian_total_s": float(jacobian_total_s),
        "message": str(getattr(result, "message", "")),
        "backend": backend_name,
        "elapsed_s": perf_counter() - t0,
    }


class DirectStatevectorProbeBackend(BaseProbeBackend):
    """Exact CPU statevector probe used by the current proof-of-principle runs."""

    @staticmethod
    def _evaluate_energy(
        *,
        ansatz,
        h_sparse,
        theta: np.ndarray,
    ) -> float:
        param_list = list(ansatz.parameters)
        if theta.size:
            bound = ansatz.assign_parameters(dict(zip(param_list, theta)))
        else:
            bound = ansatz
        psi = Statevector(bound).data
        return float(np.vdot(psi, h_sparse @ psi).real)

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        del operator
        return _run_single_probe(
            ansatz=ansatz,
            h_sparse=h_sparse,
            initial_point=initial_point,
            maxiter=maxiter,
            ftol=ftol,
            backend_name=self.name,
            use_analytical_grad=False,
        )

    def probe_parallel(
        self,
        *,
        probe_tasks: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """Run multiple probes in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: list[dict[str, Any] | None] = [None] * len(probe_tasks)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {}
            for i, task in enumerate(probe_tasks):
                fut = pool.submit(
                    _run_single_probe,
                    ansatz=task["ansatz"],
                    h_sparse=task["h_sparse"],
                    initial_point=task["initial_point"],
                    maxiter=task["maxiter"],
                    ftol=task["ftol"],
                    backend_name=self.name,
                    use_analytical_grad=task.get("use_analytical_grad", False),
                )
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

        return results  # type: ignore[return-value]

    def coarse_scan_many(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        t0 = perf_counter()
        results: list[dict[str, Any]] = []
        total_evals = 0
        for task in tasks:
            energies: list[float] = []
            for theta in task["trial_thetas"]:
                energies.append(
                    self._evaluate_energy(
                        ansatz=task["ansatz"],
                        h_sparse=task["h_sparse"],
                        theta=np.asarray(theta, dtype=float),
                    )
                )
                total_evals += 1
            results.append(
                {
                    "energies": energies,
                    "best_energy": float(min(energies)),
                    "best_index": int(np.argmin(energies)),
                }
            )
        return {
            "results": results,
            "elapsed_s": perf_counter() - t0,
            "n_evals": total_evals,
            "batched": False,
            "backend": self.name,
        }


class DirectStatevectorExactJacProbeBackend(DirectStatevectorProbeBackend):
    """Exact CPU statevector probe with the full objective and exact Jacobians."""

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
        candidate_idx: int | None = None,
        candidate_label: str | None = None,
        log_prefix: str = "LookaheadADAPT",
        progress_interval_s: float = 60.0,
    ) -> dict[str, Any]:
        return _run_single_probe_exact_jac(
            ansatz=ansatz,
            operator=operator,
            h_sparse=h_sparse,
            initial_point=initial_point,
            maxiter=maxiter,
            ftol=ftol,
            backend_name=self.name,
            candidate_idx=candidate_idx,
            candidate_label=candidate_label,
            log_prefix=log_prefix,
            progress_interval_s=progress_interval_s,
        )

    def probe_parallel(
        self,
        *,
        probe_tasks: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """Run multiple exact-Jacobian probes in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: list[dict[str, Any] | None] = [None] * len(probe_tasks)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {}
            for i, task in enumerate(probe_tasks):
                fut = pool.submit(
                    _run_single_probe_exact_jac,
                    ansatz=task["ansatz"],
                    operator=task["operator"],
                    h_sparse=task["h_sparse"],
                    initial_point=task["initial_point"],
                    maxiter=task["maxiter"],
                    ftol=task["ftol"],
                    backend_name=self.name,
                    candidate_idx=task.get("candidate_idx"),
                    candidate_label=task.get("candidate_label"),
                    log_prefix=task.get("log_prefix", "LookaheadADAPT"),
                    progress_interval_s=float(task.get("progress_interval_s", 60.0)),
                )
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

        return results  # type: ignore[return-value]


class AerExactProbeBackend(BaseProbeBackend):
    """Exact AerEstimatorV2 probe backend for CPU or GPU statevector scoring."""

    def __init__(self, config: ProbeBackendConfig):
        super().__init__(config)
        if AerEstimatorV2 is None:
            raise RuntimeError("qiskit_aer.primitives.EstimatorV2 is not available")
        self.estimator = AerEstimatorV2(
            options=_build_aer_estimator_options(config=config)
        )

    def _evaluate_energy(self, ansatz, operator, theta: np.ndarray) -> float:
        if theta.size == 0:
            job = self.estimator.run([(ansatz, operator)])
        else:
            job = self.estimator.run([(ansatz, operator, [theta.tolist()])])
        result = job.result()
        evs = np.asarray(result[0].data.evs, dtype=float).reshape(-1)
        return float(evs[0])

    def _evaluate_many_energies(
        self,
        pubs: list[tuple[Any, Any, list[list[float]] | None]],
    ) -> list[float]:
        job = self.estimator.run(pubs)
        result = job.result()
        energies: list[float] = []
        for pub_result in result:
            evs = np.asarray(pub_result.data.evs, dtype=float).reshape(-1)
            energies.append(float(evs[0]))
        return energies

    def _evaluate_energy_batches(
        self,
        pubs: list[tuple[Any, Any, list[list[float]] | None]],
    ) -> list[list[float]]:
        job = self.estimator.run(pubs)
        result = job.result()
        grouped: list[list[float]] = []
        for pub_result in result:
            evs = np.asarray(pub_result.data.evs, dtype=float).reshape(-1)
            grouped.append([float(ev) for ev in evs])
        return grouped

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        del h_sparse
        param_list = list(ansatz.parameters)
        n_params = len(param_list)
        theta0 = normalize_initial_point(initial_point, n_params)
        t0 = perf_counter()

        if n_params == 0:
            energy = self._evaluate_energy(ansatz, operator, np.array([], dtype=float))
            return {
                "energy": energy,
                "optimal_point": np.array([], dtype=float),
                "success": True,
                "nit": 0,
                "nfev": 0,
                "message": "zero-parameter ansatz",
                "backend": self.name,
                "elapsed_s": perf_counter() - t0,
            }

        eval_count = 0
        best_energy = float("inf")
        best_theta = theta0.copy()

        def cost_fn(theta: np.ndarray) -> float:
            nonlocal eval_count, best_energy, best_theta
            theta = np.asarray(theta, dtype=float)
            eval_count += 1
            energy = self._evaluate_energy(ansatz, operator, theta)
            if energy < best_energy:
                best_energy = energy
                best_theta = theta.copy()
            return energy

        result = minimize(
            cost_fn,
            theta0,
            method="SLSQP",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": maxiter, "ftol": ftol},
        )

        if result.success:
            optimal_theta = np.asarray(result.x, dtype=float)
            optimal_energy = float(result.fun)
        else:
            optimal_theta = best_theta
            optimal_energy = best_energy

        return {
            "energy": optimal_energy,
            "optimal_point": optimal_theta,
            "success": bool(result.success),
            "nit": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", eval_count)),
            "message": str(getattr(result, "message", "")),
            "backend": self.name,
            "elapsed_s": perf_counter() - t0,
        }

    def coarse_scan_many(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        t0 = perf_counter()
        pubs: list[tuple[Any, Any, list[list[float]] | None]] = []
        layout: list[tuple[int, int]] = []

        for task_idx, task in enumerate(tasks):
            for trial_idx, theta in enumerate(task["trial_thetas"]):
                theta_arr = np.asarray(theta, dtype=float).reshape(-1)
                if theta_arr.size == 0:
                    pubs.append((task["ansatz"], task["operator"], None))
                else:
                    pubs.append(
                        (task["ansatz"], task["operator"], [theta_arr.tolist()])
                    )
                layout.append((task_idx, trial_idx))

        energies = self._evaluate_many_energies(pubs)
        grouped: list[list[float] | None] = [None] * len(tasks)
        for task_idx, trial_idx in layout:
            if grouped[task_idx] is None:
                grouped[task_idx] = [0.0] * len(tasks[task_idx]["trial_thetas"])
        for energy, (task_idx, trial_idx) in zip(energies, layout):
            grouped[task_idx][trial_idx] = float(energy)

        results: list[dict[str, Any]] = []
        for task_energies in grouped:
            if task_energies is None:
                raise RuntimeError("coarse_scan_many: task produced no energy results")
            results.append(
                {
                    "energies": task_energies,
                    "best_energy": float(min(task_energies)),
                    "best_index": int(np.argmin(task_energies)),
                }
            )

        return {
            "results": results,
            "elapsed_s": perf_counter() - t0,
            "n_evals": len(pubs),
            "batched": True,
            "backend": self.name,
        }


class BatchedAerExactProbeBackend(AerExactProbeBackend):
    """Full-probe batched Aer backend for shortlist refinement."""

    initial_step = 0.2
    step_shrink = 0.5
    improvement_tol = 1e-12

    def __init__(self, config: ProbeBackendConfig):
        BaseProbeBackend.__init__(self, config)
        if AerEstimatorV2 is None:
            raise RuntimeError("qiskit_aer.primitives.EstimatorV2 is not available")
        run_options: dict[str, Any] = {
            "runtime_parameter_bind_enable": True,
        }
        if config.aer_device.upper() == "GPU":
            run_options["batched_shots_gpu"] = True
        self.estimator = AerEstimatorV2(
            options=_build_aer_estimator_options(
                config=config,
                run_options=run_options,
            )
        )

    @staticmethod
    def _clip_theta(theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, -np.pi, np.pi)

    def _build_round_candidates(
        self,
        theta: np.ndarray,
        step: float,
    ) -> list[np.ndarray]:
        proposals: list[np.ndarray] = [theta.copy()]
        for param_idx in range(theta.size):
            plus = theta.copy()
            minus = theta.copy()
            plus[param_idx] += step
            minus[param_idx] -= step
            proposals.append(self._clip_theta(plus))
            proposals.append(self._clip_theta(minus))
        return proposals

    def probe_many(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        t0 = perf_counter()
        if not tasks:
            return {
                "results": [],
                "elapsed_s": 0.0,
                "n_evals": 0,
                "batched": True,
                "backend": self.name,
            }

        states: list[dict[str, Any]] = []
        zero_param_results: dict[int, dict[str, Any]] = {}
        initial_pubs: list[tuple[Any, Any, list[list[float]] | None]] = []
        initial_state_indices: list[int] = []
        total_nfev = 0

        for task_idx, task in enumerate(tasks):
            ansatz = task["ansatz"]
            n_params = len(list(ansatz.parameters))
            theta0 = normalize_initial_point(task["initial_point"], n_params)
            if n_params == 0:
                energy = self._evaluate_energy(
                    ansatz,
                    task["operator"],
                    np.array([], dtype=float),
                )
                zero_param_results[task_idx] = {
                    "energy": energy,
                    "optimal_point": np.array([], dtype=float),
                    "success": True,
                    "nit": 0,
                    "nfev": 0,
                    "message": "zero-parameter ansatz",
                    "backend": self.name,
                    "elapsed_s": 0.0,
                }
                continue

            state = {
                "task_idx": task_idx,
                "ansatz": ansatz,
                "operator": task["operator"],
                "theta": theta0.copy(),
                "best_theta": theta0.copy(),
                "best_energy": float("inf"),
                "step": float(self.initial_step),
                "nit": 0,
                "nfev": 0,
                "success": False,
            }
            states.append(state)
            initial_pubs.append((ansatz, task["operator"], [theta0.tolist()]))
            initial_state_indices.append(len(states) - 1)

        if initial_pubs:
            initial_groups = self._evaluate_energy_batches(initial_pubs)
            total_nfev += len(initial_pubs)
            for state_idx, energies in zip(initial_state_indices, initial_groups):
                state = states[state_idx]
                state["best_energy"] = float(energies[0])

        max_rounds = max(int(task["maxiter"]) for task in tasks) if tasks else 0
        ftol = min(float(task["ftol"]) for task in tasks) if tasks else 0.0

        for _ in range(max_rounds):
            active_specs: list[tuple[int, list[np.ndarray]]] = []
            pubs: list[tuple[Any, Any, list[list[float]] | None]] = []
            for state_idx, state in enumerate(states):
                if float(state["step"]) < ftol:
                    state["success"] = True
                    continue
                state["nit"] = int(state["nit"]) + 1
                proposals = self._build_round_candidates(
                    np.asarray(state["theta"], dtype=float),
                    float(state["step"]),
                )
                active_specs.append((state_idx, proposals))
                pubs.append(
                    (
                        state["ansatz"],
                        state["operator"],
                        [proposal.tolist() for proposal in proposals],
                    )
                )

            if not pubs:
                break

            grouped_energies = self._evaluate_energy_batches(pubs)
            total_nfev += sum(len(proposals) for _, proposals in active_specs)

            for (state_idx, proposals), energies in zip(active_specs, grouped_energies):
                state = states[state_idx]
                state["nfev"] = int(state["nfev"]) + len(proposals)
                best_idx = int(np.argmin(energies))
                best_energy = float(energies[best_idx])
                current_best = float(state["best_energy"])
                if best_energy < current_best - self.improvement_tol:
                    best_theta = np.asarray(proposals[best_idx], dtype=float)
                    state["theta"] = best_theta.copy()
                    state["best_theta"] = best_theta.copy()
                    state["best_energy"] = best_energy
                    continue

                state["step"] = float(state["step"]) * self.step_shrink
                if float(state["step"]) < ftol:
                    state["success"] = True

        results: list[dict[str, Any]] = []
        state_map = {int(state["task_idx"]): state for state in states}
        elapsed_s = perf_counter() - t0
        for task_idx in range(len(tasks)):
            if task_idx in zero_param_results:
                result = dict(zero_param_results[task_idx])
                result["elapsed_s"] = elapsed_s
                results.append(result)
                continue
            state = state_map[task_idx]
            results.append(
                {
                    "energy": float(state["best_energy"]),
                    "optimal_point": np.asarray(state["best_theta"], dtype=float),
                    "success": bool(state["success"]),
                    "nit": int(state["nit"]),
                    "nfev": int(state["nfev"]),
                    "message": (
                        "batched coordinate search converged"
                        if state["success"]
                        else "batched coordinate search maxiter"
                    ),
                    "backend": self.name,
                    "elapsed_s": elapsed_s,
                }
            )

        return {
            "results": results,
            "elapsed_s": elapsed_s,
            "n_evals": int(total_nfev),
            "batched": True,
            "backend": self.name,
        }


class GridScanProbeBackend(DirectStatevectorProbeBackend):
    """Fast probe that evaluates a grid of values for the new parameter only.

    Instead of running full SLSQP optimization over all parameters, this
    backend fixes existing parameters at their current values and evaluates
    energy at a small grid of trial values for the last (newly appended)
    parameter.  This is ~20-60x cheaper than SLSQP for ranking purposes.
    """

    GRID = np.array([-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0])

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        del operator  # not needed for direct SV
        param_list = list(ansatz.parameters)
        n_params = len(param_list)
        theta0 = normalize_initial_point(initial_point, n_params)
        t0 = perf_counter()

        if n_params == 0:
            energy = self._evaluate_energy(
                ansatz=ansatz, h_sparse=h_sparse, theta=np.array([], dtype=float)
            )
            return {
                "energy": energy,
                "optimal_point": np.array([], dtype=float),
                "success": True,
                "nit": 0,
                "nfev": 0,
                "message": "zero-parameter ansatz",
                "backend": self.name,
                "elapsed_s": perf_counter() - t0,
            }

        best_energy = float("inf")
        best_theta = theta0.copy()
        nfev = 0

        for grid_val in self.GRID:
            trial = theta0.copy()
            trial[-1] = grid_val
            energy = self._evaluate_energy(
                ansatz=ansatz, h_sparse=h_sparse, theta=trial
            )
            nfev += 1
            if energy < best_energy:
                best_energy = energy
                best_theta = trial.copy()

        return {
            "energy": best_energy,
            "optimal_point": best_theta,
            "success": True,
            "nit": 1,
            "nfev": nfev,
            "message": f"grid scan {len(self.GRID)} points",
            "backend": self.name,
            "elapsed_s": perf_counter() - t0,
        }


class OneDRelaxProbeBackend(DirectStatevectorProbeBackend):
    """Relax only the newly appended parameter and keep prior params frozen."""

    alpha_lower = -float(np.pi)
    alpha_upper = float(np.pi)
    xatol = 1e-3
    scalar_maxiter = 200

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        del operator, maxiter, ftol
        return _run_single_one_d_probe(
            ansatz=ansatz,
            h_sparse=h_sparse,
            initial_point=initial_point,
            backend_name=self.name,
            alpha_lower=self.alpha_lower,
            alpha_upper=self.alpha_upper,
            xatol=self.xatol,
            maxiter=self.scalar_maxiter,
        )

    def probe_parallel(
        self,
        *,
        probe_tasks: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: list[dict[str, Any] | None] = [None] * len(probe_tasks)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {}
            for i, task in enumerate(probe_tasks):
                fut = pool.submit(
                    _run_single_one_d_probe,
                    ansatz=task["ansatz"],
                    h_sparse=task["h_sparse"],
                    initial_point=task["initial_point"],
                    backend_name=self.name,
                    alpha_lower=self.alpha_lower,
                    alpha_upper=self.alpha_upper,
                    xatol=self.xatol,
                    maxiter=self.scalar_maxiter,
                )
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

        return results  # type: ignore[return-value]


class QuadraticLocalProbeBackend(DirectStatevectorProbeBackend):
    """Rank candidates with a local quadratic model on the new parameter."""

    delta = 0.1
    alpha_cap = 0.5

    def probe(
        self,
        *,
        ansatz,
        operator,
        h_sparse,
        initial_point: list[float] | np.ndarray,
        maxiter: int,
        ftol: float,
    ) -> dict[str, Any]:
        del operator, maxiter, ftol
        return _run_single_quadratic_probe(
            ansatz=ansatz,
            h_sparse=h_sparse,
            initial_point=initial_point,
            backend_name=self.name,
            delta=self.delta,
            alpha_cap=self.alpha_cap,
        )

    def probe_parallel(
        self,
        *,
        probe_tasks: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: list[dict[str, Any] | None] = [None] * len(probe_tasks)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {}
            for i, task in enumerate(probe_tasks):
                fut = pool.submit(
                    _run_single_quadratic_probe,
                    ansatz=task["ansatz"],
                    h_sparse=task["h_sparse"],
                    initial_point=task["initial_point"],
                    backend_name=self.name,
                    delta=self.delta,
                    alpha_cap=self.alpha_cap,
                )
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

        return results  # type: ignore[return-value]


def create_probe_backend(
    *,
    backend_name: str,
    aer_max_parallel_threads: int = 4,
    aer_precision: str = "double",
) -> BaseProbeBackend:
    """Construct a probe backend by name."""
    name = backend_name.strip().lower()
    if name == "direct_sv":
        return DirectStatevectorProbeBackend(ProbeBackendConfig(name="direct_sv"))
    if name in {"direct_sv_exact_jac", "full_direct_sv_exact_jac"}:
        return DirectStatevectorExactJacProbeBackend(ProbeBackendConfig(name=name))
    if name == "grid_scan":
        return GridScanProbeBackend(ProbeBackendConfig(name="grid_scan"))
    if name == "one_d_relax":
        return OneDRelaxProbeBackend(ProbeBackendConfig(name="one_d_relax"))
    if name == "quadratic_local":
        return QuadraticLocalProbeBackend(ProbeBackendConfig(name="quadratic_local"))
    if name == "aer_exact_cpu":
        return AerExactProbeBackend(
            ProbeBackendConfig(
                name="aer_exact_cpu",
                aer_device="CPU",
                aer_max_parallel_threads=aer_max_parallel_threads,
                aer_precision=aer_precision,
            )
        )
    if name == "aer_exact_gpu":
        return AerExactProbeBackend(
            ProbeBackendConfig(
                name="aer_exact_gpu",
                aer_device="GPU",
                aer_max_parallel_threads=aer_max_parallel_threads,
                aer_precision=aer_precision,
            )
        )
    if name == "aer_exact_cpu_batch":
        return BatchedAerExactProbeBackend(
            ProbeBackendConfig(
                name="aer_exact_cpu_batch",
                aer_device="CPU",
                aer_max_parallel_threads=aer_max_parallel_threads,
                aer_precision=aer_precision,
            )
        )
    if name == "aer_exact_gpu_batch":
        return BatchedAerExactProbeBackend(
            ProbeBackendConfig(
                name="aer_exact_gpu_batch",
                aer_device="GPU",
                aer_max_parallel_threads=aer_max_parallel_threads,
                aer_precision=aer_precision,
            )
        )
    raise ValueError(
        f"Unknown probe backend '{backend_name}'. "
        "Use direct_sv, direct_sv_exact_jac, full_direct_sv_exact_jac, "
        "grid_scan, one_d_relax, quadratic_local, "
        "aer_exact_cpu, aer_exact_gpu, aer_exact_cpu_batch, or aer_exact_gpu_batch."
    )
