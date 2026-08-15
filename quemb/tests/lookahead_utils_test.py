import math

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scripts.lookahead_utils import (
    AerExactProbeBackend,
    BatchedAerExactProbeBackend,
    DirectStatevectorProbeBackend,
    ProbeBackendConfig,
    SelectorConfig,
    aggregate_solve_timing_summaries,
    patched_fast_adapt,
)
from quemb.molbe.fast_adapt_vqe import summarize_fastadapt_timings


def test_aer_exact_coarse_scan_many_matches_direct_statevector():
    pytest.importorskip("qiskit_aer.primitives")

    theta = Parameter("theta")
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta, 0)
    operator = SparsePauliOp("Z")
    h_sparse = operator.to_matrix(sparse=True)

    tasks = [
        {
            "candidate_idx": 0,
            "ansatz": ansatz,
            "operator": operator,
            "h_sparse": h_sparse,
            "trial_thetas": [
                np.asarray([0.0], dtype=float),
                np.asarray([math.pi], dtype=float),
            ],
        },
        {
            "candidate_idx": 1,
            "ansatz": ansatz,
            "operator": operator,
            "h_sparse": h_sparse,
            "trial_thetas": [
                np.asarray([math.pi / 2], dtype=float),
                np.asarray([0.0], dtype=float),
                np.asarray([math.pi], dtype=float),
            ],
        },
    ]

    direct = DirectStatevectorProbeBackend(ProbeBackendConfig(name="direct_sv"))
    aer = AerExactProbeBackend(
        ProbeBackendConfig(
            name="aer_exact_cpu",
            aer_device="CPU",
            aer_max_parallel_threads=1,
        )
    )

    direct_scan = direct.coarse_scan_many(tasks=tasks)
    aer_scan = aer.coarse_scan_many(tasks=tasks)

    assert aer_scan["batched"] is True
    assert aer_scan["backend"] == "aer_exact_cpu"
    assert aer_scan["n_evals"] == 5
    assert direct_scan["n_evals"] == 5
    assert len(aer_scan["results"]) == len(direct_scan["results"]) == 2

    for aer_result, direct_result in zip(aer_scan["results"], direct_scan["results"]):
        np.testing.assert_allclose(
            aer_result["energies"],
            direct_result["energies"],
            atol=1e-10,
        )
        assert aer_result["best_index"] == direct_result["best_index"]
        assert aer_result["best_energy"] == pytest.approx(
            direct_result["best_energy"],
            abs=1e-10,
        )


def test_staged_probe_refines_only_coarse_top_m_candidates():
    class DummyAnsatz:
        def __init__(self, candidate_idx: int, n_params: int):
            self.candidate_idx = candidate_idx
            self.parameters = tuple(range(n_params))

    class FakeProbeBackend:
        name = "fake_batched"

        def __init__(self):
            self.refined_candidates: list[int] = []
            self.refine_initial_points: list[np.ndarray] = []

        def coarse_scan_many(self, *, tasks):
            coarse_energies = {
                0: [-0.70, -0.68, -0.65],
                1: [-1.20, -1.10, -1.00],
                2: [-0.40, -0.35, -0.30],
            }
            results = []
            for task in tasks:
                energies = coarse_energies[int(task["candidate_idx"])]
                results.append(
                    {
                        "energies": energies,
                        "best_energy": float(min(energies)),
                        "best_index": int(np.argmin(energies)),
                    }
                )
            return {
                "results": results,
                "elapsed_s": 0.05,
                "n_evals": sum(len(task["trial_thetas"]) for task in tasks),
                "batched": True,
                "backend": self.name,
            }

        def probe(
            self,
            *,
            ansatz,
            operator,
            h_sparse,
            initial_point,
            maxiter,
            ftol,
        ):
            del operator, h_sparse, maxiter, ftol
            theta = np.asarray(initial_point, dtype=float)
            self.refined_candidates.append(int(ansatz.candidate_idx))
            self.refine_initial_points.append(theta.copy())
            refined_energies = {0: -1.30, 1: -1.50, 2: -0.80}
            return {
                "energy": refined_energies[int(ansatz.candidate_idx)],
                "optimal_point": theta + 0.01,
                "success": True,
                "nit": 1,
                "nfev": 2,
                "message": "ok",
                "backend": self.name,
                "elapsed_s": 0.02,
            }

    config = SelectorConfig(
        top_k=3,
        probe_maxiter=5,
        probe_ftol=1e-6,
        staged_probe=True,
        refine_top_m=2,
        coarse_scan_steps=(0.05, 0.10, 0.20),
    )
    fake_backend = FakeProbeBackend()

    with patched_fast_adapt(
        config=config,
        case_name="unit_test",
        selector_mode="lookahead_generic",
        excitation_labels=["op0", "op1", "op2"],
        log_prefix="UnitTestLookahead",
    ) as guided_cls:
        solver = object.__new__(guided_cls)
        solver.gradient_threshold = 1e-3
        solver.check_cyclicity = False
        solver.cyclicity_action = "stop"
        solver.verbose = 0
        solver.selector_events = []
        solver._probe_backend = fake_backend
        solver._probe_new_param_seed_cache = {}
        solver._check_cyclicity = lambda _: False
        solver._build_trial_ansatz = lambda trial_indices: DummyAnsatz(
            candidate_idx=int(trial_indices[-1]),
            n_params=len(trial_indices),
        )

        selected_idx, optimal_point, termination_reason = (
            solver._lookahead_select_candidate(
                iteration=6,
                grads=[0.40, -0.30, 0.20],
                abs_grads=[0.40, 0.30, 0.20],
                sorted_candidates=[0, 1, 2],
                prev_op_indices=[9],
                theta=[0.12],
                operator=object(),
                h_sparse=object(),
                trigger_context={"should_activate": True},
            )
        )

    assert termination_reason is None
    assert selected_idx == 1
    np.testing.assert_allclose(optimal_point, np.asarray([0.13, 0.06]), atol=1e-12)
    assert fake_backend.refined_candidates == [1, 0]

    event = solver.selector_events[-1]
    assert event["staged_probe_active"] is True
    assert event["refine_candidates"] == [1, 0]
    assert event["coarse_scan_backend"] == "fake_batched"
    assert event["coarse_scan_batched"] is True
    assert event["coarse_scan_n_evals"] == 9
    assert event["coarse_scan_elapsed_s"] == pytest.approx(0.05)
    assert event["full_probe_elapsed_sum_s"] == pytest.approx(0.04)
    assert event["total_probe_elapsed_s"] == pytest.approx(0.09)
    assert event["selected_idx"] == 1
    assert event["selected_probe_energy"] == pytest.approx(-1.50)

    shortlist_by_idx = {entry["idx"]: entry for entry in event["shortlist"]}
    assert shortlist_by_idx[1]["refined"] is True
    assert shortlist_by_idx[0]["refined"] is True
    assert shortlist_by_idx[2]["refined"] is False
    assert shortlist_by_idx[2]["coarse_best_energy"] == pytest.approx(-0.40)


def test_timing_summaries_account_for_probe_time():
    timing_summary = summarize_fastadapt_timings(
        setup_timings={
            "hamiltonian_to_sparse_s": 1.0,
            "pool_to_sparse_s": 2.0,
        },
        iteration_timings=[
            {
                "gradient_time_s": 3.0,
                "selection_time_s": 4.0,
                "vqe_time_s": 5.0,
            }
        ],
        selector_events=[
            {
                "full_probe_elapsed_sum_s": 7.0,
                "coarse_scan_elapsed_s": 6.0,
                "total_probe_elapsed_s": 13.0,
            }
        ],
    )

    assert timing_summary["probe_total_s"] == pytest.approx(13.0)
    assert timing_summary["total_accounted_s"] == pytest.approx(28.0)
    assert timing_summary["dominant_phase"] == "probe"
    assert timing_summary["dominant_phase_s"] == pytest.approx(13.0)

    aggregate = aggregate_solve_timing_summaries(
        solve_records=[{"timing_summary": timing_summary}],
        runtime_s=31.0,
    )

    assert aggregate["probe_total_s"] == pytest.approx(13.0)
    assert aggregate["total_accounted_s"] == pytest.approx(28.0)
    assert aggregate["be_overhead_s"] == pytest.approx(3.0)
    assert aggregate["dominant_phase"] == "probe"


def test_selector_uses_probe_many_when_available():
    class DummyAnsatz:
        def __init__(self, candidate_idx: int, n_params: int):
            self.candidate_idx = candidate_idx
            self.parameters = tuple(range(n_params))

    class FakeBatchProbeBackend:
        name = "fake_full_batch"

        def probe_many(self, *, tasks):
            results = []
            for task in tasks:
                candidate_idx = int(task["candidate_idx"])
                theta = np.asarray(task["initial_point"], dtype=float)
                energy = {0: -1.00, 1: -1.25, 2: -1.40}[candidate_idx]
                results.append(
                    {
                        "energy": energy,
                        "optimal_point": theta + 0.02,
                        "success": True,
                        "nit": 3,
                        "nfev": 9,
                        "message": "batched ok",
                        "backend": self.name,
                        "elapsed_s": 0.07,
                    }
                )
            return {
                "results": results,
                "elapsed_s": 0.21,
                "n_evals": 27,
                "batched": True,
                "backend": self.name,
            }

    config = SelectorConfig(
        top_k=3,
        probe_maxiter=5,
        probe_ftol=1e-6,
        staged_probe=False,
    )
    fake_backend = FakeBatchProbeBackend()

    with patched_fast_adapt(
        config=config,
        case_name="unit_test",
        selector_mode="lookahead_generic",
        excitation_labels=["op0", "op1", "op2"],
        log_prefix="UnitTestLookahead",
    ) as guided_cls:
        solver = object.__new__(guided_cls)
        solver.gradient_threshold = 1e-3
        solver.check_cyclicity = False
        solver.cyclicity_action = "stop"
        solver.verbose = 0
        solver.selector_events = []
        solver._probe_backend = fake_backend
        solver._probe_new_param_seed_cache = {}
        solver._check_cyclicity = lambda _: False
        solver._build_trial_ansatz = lambda trial_indices: DummyAnsatz(
            candidate_idx=int(trial_indices[-1]),
            n_params=len(trial_indices),
        )

        selected_idx, optimal_point, termination_reason = (
            solver._lookahead_select_candidate(
                iteration=6,
                grads=[0.40, -0.30, 0.20],
                abs_grads=[0.40, 0.30, 0.20],
                sorted_candidates=[0, 1, 2],
                prev_op_indices=[9],
                theta=[0.12],
                operator=object(),
                h_sparse=object(),
                trigger_context={"should_activate": True},
            )
        )

    assert termination_reason is None
    assert selected_idx == 2
    np.testing.assert_allclose(optimal_point, np.asarray([0.14, 0.02]), atol=1e-12)

    event = solver.selector_events[-1]
    assert event["staged_probe_active"] is False
    assert event["full_probe_backend"] == "fake_full_batch"
    assert event["full_probe_batched"] is True
    assert event["full_probe_n_evals"] == 27
    assert event["full_probe_elapsed_sum_s"] == pytest.approx(0.21)
    assert event["selected_probe_energy"] == pytest.approx(-1.40)


def test_aer_exact_backend_forwards_precision_to_estimator(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEstimator:
        def __init__(self, *, options):
            captured["options"] = options

    monkeypatch.setattr("scripts.lookahead_utils.AerEstimatorV2", FakeEstimator)

    backend = AerExactProbeBackend(
        ProbeBackendConfig(
            name="aer_exact_gpu",
            aer_device="GPU",
            aer_max_parallel_threads=7,
            aer_precision="single",
        )
    )

    assert backend.name == "aer_exact_gpu"
    assert captured["options"] == {
        "default_precision": 0.0,
        "backend_options": {
            "method": "statevector",
            "device": "GPU",
            "precision": "single",
        },
    }


def test_batched_aer_backend_enables_runtime_bind_options(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEstimator:
        def __init__(self, *, options):
            captured["options"] = options

    monkeypatch.setattr("scripts.lookahead_utils.AerEstimatorV2", FakeEstimator)

    backend = BatchedAerExactProbeBackend(
        ProbeBackendConfig(
            name="aer_exact_gpu_batch",
            aer_device="GPU",
            aer_max_parallel_threads=7,
            aer_precision="single",
        )
    )

    assert backend.name == "aer_exact_gpu_batch"
    assert captured["options"] == {
        "default_precision": 0.0,
        "backend_options": {
            "method": "statevector",
            "device": "GPU",
            "precision": "single",
        },
        "run_options": {
            "runtime_parameter_bind_enable": True,
            "batched_shots_gpu": True,
        },
    }


def test_batched_aer_probe_many_improves_simple_tasks():
    pytest.importorskip("qiskit_aer.primitives")

    theta = Parameter("theta")
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta, 0)
    operator = SparsePauliOp("Z")

    backend = BatchedAerExactProbeBackend(
        ProbeBackendConfig(
            name="aer_exact_cpu_batch",
            aer_device="CPU",
            aer_max_parallel_threads=1,
        )
    )

    initial_points = [
        np.asarray([0.0], dtype=float),
        np.asarray([0.2], dtype=float),
    ]
    initial_energies = [math.cos(float(point[0])) for point in initial_points]
    batch = backend.probe_many(
        tasks=[
            {
                "candidate_idx": idx,
                "ansatz": ansatz,
                "operator": operator,
                "h_sparse": operator.to_matrix(sparse=True),
                "initial_point": point,
                "maxiter": 8,
                "ftol": 1e-4,
            }
            for idx, point in enumerate(initial_points)
        ]
    )

    assert batch["batched"] is True
    assert batch["backend"] == "aer_exact_cpu_batch"
    assert batch["n_evals"] > 0
    assert len(batch["results"]) == 2

    for result, initial_energy in zip(batch["results"], initial_energies):
        assert result["energy"] < initial_energy - 1e-6
        assert result["success"] in {True, False}
        assert result["nfev"] > 0
