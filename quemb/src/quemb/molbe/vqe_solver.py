# Author(s): Derek Peng (VQE implementation)
"""
VQE (Variational Quantum Eigensolver) solver for Bootstrap Embedding.

This module implements a VQE solver with:
- UCCSD ansatz using Qiskit
- Adaptive 3-stage convergence strategy
- FCIDUMP Hamiltonian file parsing
- Jordan-Wigner fermion-to-qubit mapping
- Statevector simulation for exact RDM calculation
- Warm-start capability for BE iterations
- Optional diagnostics for iteration-level convergence traces
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Sequence
from warnings import warn

import h5py
import numpy as np
from attrs import Factory, define, field, validators
from numpy import ndarray
from pyscf import ao2mo
from pyscf.scf.hf import RHF
from pyscf.tools import fcidump

from quemb.molbe.ceo_manifest import canonical_digest
from quemb.molbe.pfrag import Frags
from quemb.shared.typing import Matrix

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_algorithms import VQE as QiskitVQE
    from qiskit_algorithms import AdaptVQE
    from qiskit_algorithms.exceptions import AlgorithmError
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA
    from qiskit_nature.second_q.circuit.library import UCC
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp

    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    # Create dummy types for type hints when Qiskit is not available
    SparsePauliOp = Any  # type: ignore
    Statevector = Any  # type: ignore
    QuantumCircuit = Any  # type: ignore
    ElectronicEnergy = Any  # type: ignore
    ElectronicIntegrals = Any  # type: ignore

    warn(
        f"Qiskit not available. VQE solver will not work. "
        f"Install with: pip install qiskit qiskit-nature qiskit-algorithms\n"
        f"Error: {e}"
    )


_ACTIVE_SPACE_SELECTION = "scf_block_canonical_frontier"
_GENERATED_SECTOR_POOL = "__generated_ovp_ceo_pool__"
_BLOCK_CANONICALIZATION = "occupied_virtual"
_FULL_CANONICALIZATION = "full"


def _validate_nonnegative_count(
    _instance: object, attribute: Any, value: int
) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{attribute.name} must be a non-negative integer")


def _validate_boundary_gap(
    _instance: object, attribute: Any, value: float
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{attribute.name} must be finite and non-negative")


@define(frozen=True, slots=True)
class ActiveSpaceSpec:
    """Deterministic frontier active-space selection in the SCF block basis."""

    frozen_occupied_orbitals: Final[int] = field(
        default=0,
        validator=_validate_nonnegative_count,
    )
    discarded_virtual_orbitals: Final[int] = field(
        default=0,
        validator=_validate_nonnegative_count,
    )
    selection: Final[Literal["scf_block_canonical_frontier"]] = field(
        default=_ACTIVE_SPACE_SELECTION,
        validator=validators.in_([_ACTIVE_SPACE_SELECTION]),
    )
    minimum_boundary_gap_ha: Final[float] = field(
        default=0.0,
        validator=_validate_boundary_gap,
    )


@define(frozen=True)
class VQE_ArgsUser:
    """
    User-facing VQE configuration arguments.

    Parameters
    ----------
    hamiltonian_dir : str
        Directory containing fragment Hamiltonians in FCIDUMP format.
        Files should be named like: h10_be2f0, h10_be2f1, etc.
        Default: 'store_h10_files/be2/'

    orbital_canonicalization : {"occupied_virtual", "full"}
        Orbital basis used for VQE. The default preserves the SCF occupied
        projector. ``full`` reproduces the full-space basis used for the
        published H4/F2 calculations and cannot be combined with frozen core.

    adaptive_convergence : bool
        Enable adaptive 3-stage convergence strategy.
        If True, VQE convergence tightens as BE converges.
        If False, use fixed convergence parameters.
        Default: True

    # Stage 1: Early BE iterations (BE iter 1-3)
    stage1_max_iter : int
        Maximum VQE iterations for stage 1. Default: 50
    stage1_energy_tol : float
        Energy convergence tolerance for stage 1. Default: 1e-3
    stage1_be_threshold : float
        BE energy change threshold to exit stage 1. Default: 1e-2

    # Stage 2: Middle BE iterations (BE iter 4-7)
    stage2_max_iter : int
        Maximum VQE iterations for stage 2. Default: 100
    stage2_energy_tol : float
        Energy convergence tolerance for stage 2. Default: 1e-4
    stage2_be_threshold : float
        BE energy change threshold to exit stage 2. Default: 1e-3

    # Stage 3: Final BE iterations (BE iter 8+)
    stage3_max_iter : int
        Maximum VQE iterations for stage 3. Default: 200
    stage3_energy_tol : float
        Energy convergence tolerance for stage 3. Default: 1e-6

    # Estimator settings (Phase 13 discovery - critical for accuracy!)
    estimator_type : Literal["statevector", "aer_exact", "backend"]
        Which Qiskit estimator to use. Default: "aer_exact"
        - "statevector": StatevectorEstimator for EXACT expectation values (Python/NumPy)
        - "aer_exact": AerEstimatorV2 with precision=0 for EXACT values (C++/OpenMP, 2-3x faster) (RECOMMENDED)
        - "backend": BackendEstimatorV2 with AerSimulator (adds ~15 mHa sampling noise, for hardware compatibility)
        See Phase 13 in RESEARCH_JOURNAL.md - BackendEstimatorV2 was found to add
        ~15 mHa noise even with statevector backend, causing VQE to report incorrect
        energies (including below FCI). StatevectorEstimator and AerEstimatorV2(precision=0) give exact results.
    backend_estimator_precision : float | None
        Precision for BackendEstimatorV2 (only used when estimator_type='backend').
        Default Qiskit value is 0.015625 (~15.6 mHa). Set to 1e-6 for higher precision.
        None means use Qiskit default. Default: None
    aer_max_parallel_threads : int
        Maximum parallel threads for AerEstimatorV2 (only used when estimator_type='aer_exact').
        Default: 4. Set higher for multi-core systems.

    # Optimizer settings
    optimizer_name : str
        Optimizer to use ('SPSA', 'COBYLA', 'L_BFGS_B', 'SLSQP'). Default: 'SPSA'
    cobyla_rhobeg : float
        Initial step size for COBYLA. Default: 0.1
    cobyla_rhoend : float
        Final step size for COBYLA. Default: 1e-6

    max_restarts : int
        Number of optimizer runs to attempt (including the first run). Each
        additional run starts from a random initial point; the lowest-energy
        solution is selected. Default: 3
    restart_energy_tol : float
        Skip further restarts when the improvement of the best energy falls
        below this threshold. Default: 1e-6
    random_seed : int | None
        Seed for random initial points. Default: None (use entropy)

    # Warm start settings
    warm_start : bool
        Use previous VQE parameters as initial guess.
        Significantly speeds up convergence in later BE iterations.
        Default: True

    verbose : int
        Verbosity level (0-3). Default: 0
        0: Silent
        1: BE iteration info
        2: VQE convergence info
        3: Full debug output
    show_progress_bar : bool
        Display a textual progress bar during optimizer evaluations. Default: False
    track_iteration_history : bool
        Record optimizer iteration diagnostics for later inspection.
        Default: False
    track_density_matrices : bool
        Compute intermediate one-particle density matrices at callback checkpoints.
        Default: False
    density_sample_interval : int
        Interval (in optimizer evaluations) between density matrix captures when
        ``track_density_matrices`` is enabled. Default: 1

    # Best-selection on iteration limit
    # NOTE: This feature is REDUNDANT for scipy-based optimizers (COBYLA, L_BFGS_B,
    # SLSQP) because scipy already returns the minimum value found during optimization,
    # not the last iteration's value. Kept for potential use with other optimizers.
    # See Experiment 13 in TRACKING.md and Phase 12 in RESEARCH_JOURNAL.md.
    use_best_on_limit : bool
        When VQE hits max iterations without converging, use the best (lowest
        energy) parameters seen during optimization instead of the final parameters.
        NOTE: Redundant for scipy optimizers - they already track and return the best.
        Default: True
    best_select_min_iter : int
        Skip the first N iterations when tracking best parameters (burn-in period).
        Early iterations may have unstable parameters. Default: 5
    """

    hamiltonian_dir: Final[str] = "store_h10_files/be2/"

    # Adaptive convergence
    adaptive_convergence: Final[bool] = True

    # Stage 1 parameters
    stage1_max_iter: Final[int] = 50
    stage1_energy_tol: Final[float] = 1e-3
    stage1_be_threshold: Final[float] = 1e-2

    # Stage 2 parameters
    stage2_max_iter: Final[int] = 100
    stage2_energy_tol: Final[float] = 1e-4
    stage2_be_threshold: Final[float] = 1e-3

    # Stage 3 parameters
    stage3_max_iter: Final[int] = 200
    stage3_energy_tol: Final[float] = 1e-6

    # Ansatz selection (Phase 24: ADAPT-VQE support)
    # 'uccsd': Fixed UCCSD ansatz (all singles and doubles excitations)
    # 'adapt': ADAPT-VQE (iteratively grows ansatz, avoids local minima)
    # 'adapt_fast': ADAPT-VQE with fast statevector gradients (10-100x faster gradient eval)
    # ADAPT-VQE achieved 0.086 mHa vs UCCSD's 16.87 mHa on H4 (196x improvement)
    # See scripts/test_adapt_vqe_h4_be_full.py for validation
    ansatz_type: Final[
        Literal["uccsd", "adapt", "adapt_fast", "adapt_matrix_free", "adapt_sector"]
    ] = field(
        default="uccsd",
        validator=validators.in_(
            ["uccsd", "adapt", "adapt_fast", "adapt_matrix_free", "adapt_sector"]
        ),
    )

    # ADAPT-VQE parameters (only used when ansatz_type='adapt')
    adapt_gradient_threshold: Final[float] = 1e-3  # Stop when all gradients below this
    adapt_eigenvalue_threshold: Final[float] = 1e-5  # Energy convergence threshold
    adapt_max_iterations: Final[int] = 20  # Max ADAPT iterations (operators added)
    adapt_check_cyclicity: Final[bool] = True  # Check for repeating operator sequences (disable for F₂-like plateau breakthrough)
    adapt_cyclicity_action: Final[str] = "skip"  # "stop" = halt on cycle (old behavior), "skip" = try next operator (new)
    adapt_gradient_log_top_k: Final[int] = field(
        default=5, validator=validators.ge(0)
    )  # Record/log top-k gradients at each ADAPT step
    adapt_tracked_operator_indices: Final[tuple[int, ...]] = field(
        factory=tuple,
        converter=tuple,
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(tuple),
        ),
    )  # Extra pool indices to log every ADAPT step (e.g. op 22 in F₂)
    adapt_sector_selector_policy: Final[
        Literal["greedy_gradient", "always_top5_energy"]
    ] = field(
        default="greedy_gradient",
        validator=validators.in_(["greedy_gradient", "always_top5_energy"]),
    )
    adapt_sector_selector_top_k: Final[int] = field(
        default=5, validator=validators.ge(1)
    )
    adapt_sector_probe_max_iterations: Final[int] = field(
        default=200, validator=validators.ge(1)
    )
    adapt_sector_probe_ftol: Final[float] = field(
        default=1.0e-10, validator=validators.gt(0.0)
    )
    adapt_sector_energy_tie_tolerance: Final[float] = field(
        default=1.0e-10, validator=validators.ge(0.0)
    )

    # k-UpCCGSD (generalized UCC) parameters
    # Generalized UCC allows excitations between ALL orbital pairs, not just occ→virt.
    # This can capture strong correlation effects that standard UCC misses (e.g., σ-σ* in F₂).
    # When generalized=True, the operator pool includes ALL orbital excitations.
    # The 'reps' parameter creates k repetitions of the ansatz (k-UpCCGSD).
    # Reference: Lee et al. (2019) "Generalized UCC ansatz"
    ucc_generalized: Final[bool] = False  # If True, use generalized excitations (any→any)
    ucc_reps: Final[int] = 1  # Number of ansatz repetitions (k in k-UpCCGSD)
    ucc_preserve_spin: Final[bool] = True  # If True, preserve spin symmetry in excitations

    # Frozen core approximation (reduces operator pool and qubits)
    # Core electrons (e.g., 1s orbitals) don't participate in chemistry and can be frozen.
    # This significantly speeds up ADAPT-VQE by reducing the operator pool size.
    # Example: N₂ with 8 orbitals → freeze 1 core → 315 operators → 204 operators (35% fewer)
    frozen_core: Final[Literal["none", "auto", "manual"]] = field(
        default="none",
        validator=validators.in_(["none", "auto", "manual"]),
    )
    # 'none': No frozen orbitals (default, backward compatible)
    # 'auto': Automatically freeze orbitals with occupation > frozen_core_threshold
    # 'manual': Freeze exactly frozen_core_num_orbitals orbitals

    frozen_core_threshold: Final[float] = 1.98  # Occupation threshold for auto mode
    frozen_core_num_orbitals: Final[int] = 0  # Number of orbitals to freeze in manual mode
    # Shared production contract for matched occupied freeze + virtual discard.
    # When absent, frozen_core_num_orbitals remains a backward-compatible
    # shorthand with no virtual discard.
    active_space: Final[ActiveSpaceSpec | None] = None

    # The published H4/F2 calculations diagonalized the complete effective
    # one-electron Hamiltonian. Production active-space calculations preserve
    # the SCF occupied projector by canonicalizing occupied and virtual blocks.
    orbital_canonicalization: Final[Literal["occupied_virtual", "full"]] = field(
        default=_BLOCK_CANONICALIZATION,
        kw_only=True,
        validator=validators.in_(
            [
                _BLOCK_CANONICALIZATION,
                _FULL_CANONICALIZATION,
            ]
        ),
    )

    # Optimizer selection
    optimizer_name: Final[str] = (
        "SPSA"  # Changed from COBYLA: SPSA reduces fragment asymmetry by 49%
    )

    # COBYLA optimizer
    cobyla_rhobeg: Final[float] = 0.1
    cobyla_rhoend: Final[float] = 1e-6

    # Restart settings
    max_restarts: Final[int] = 3
    restart_energy_tol: Final[float] = 1e-6
    random_seed: Final[int | None] = None
    # Parallel restarts: run multiple restarts concurrently using ThreadPoolExecutor
    # Set to 1 for sequential restarts (default), >1 for parallel execution
    # Parallel restarts can significantly speed up VQE when multiple CPU cores are available
    parallel_restarts: Final[int] = 1

    # Warm start
    warm_start: Final[bool] = True

    # Verbosity
    verbose: Final[int] = 0
    # Diagnostics / tracing
    show_progress_bar: Final[bool] = False
    track_iteration_history: Final[bool] = False
    track_density_matrices: Final[bool] = False
    density_sample_interval: Final[int] = 1

    # Best-selection on iteration limit
    # When VQE hits max iterations without converging, use the best (lowest energy)
    # parameters seen during optimization instead of the final parameters.
    # This helps when the optimizer "bounces" and final value is worse than earlier.
    use_best_on_limit: Final[bool] = True
    best_select_min_iter: Final[int] = 5  # Skip first N iterations (burn-in)

    # Estimator selection (Phase 13 discovery, Phase 23 enhancement)
    # 'statevector': StatevectorEstimator for EXACT expectation values (Python/NumPy, slower)
    # 'aer_exact': AerEstimatorV2 with precision=0 for EXACT values (C++/OpenMP, 2-3x faster, RECOMMENDED)
    # 'backend': BackendEstimatorV2 + AerSimulator (adds ~15 mHa noise by default, for hardware compatibility)
    # See Phase 13 in RESEARCH_JOURNAL.md for details on why this matters.
    # NOTE: Invalid values will raise ValueError at construction time (not silently fallback!)
    estimator_type: Final[Literal["statevector", "aer_exact", "backend", "direct_sv"]] = field(
        default="aer_exact",
        validator=validators.in_(["statevector", "aer_exact", "backend", "direct_sv"]),
    )

    # BackendEstimatorV2 precision (only used when estimator_type='backend')
    # Default Qiskit value is 0.015625 (~15.6 mHa noise). Set lower for more precision.
    # Only relevant when estimator_type='backend'.
    backend_estimator_precision: Final[float | None] = None  # None = use Qiskit default

    # AerEstimatorV2 parallel threads (only used when estimator_type='aer_exact')
    # Higher values can speed up larger circuits. Default: 4
    aer_max_parallel_threads: Final[int] = 4

    # Floating-point precision for Aer exact simulation
    # GPU throughput is often materially better in single precision.
    aer_precision: Final[Literal["single", "double"]] = field(
        default="double",
        validator=validators.in_(["single", "double"]),
    )

    # Direct statevector inner-solver options
    # When True, use an exact reverse-mode Jacobian inside DirectSVSolver instead
    # of letting SciPy SLSQP approximate gradients numerically.
    direct_sv_use_exact_jacobian: Final[bool] = True

    # Debug: HF configuration analysis
    # When True, exhaustively check all HF configurations to see if aufbau is optimal.
    # This is O(C(norb, nsocc)) and prints verbose output. Useful for debugging
    # but slows down production runs. Default: False (disabled)
    debug_config_analysis: Final[bool] = False

    # Transpilation optimization level (0-3) using generate_preset_pass_manager()
    # Benchmarked on 24-qubit UCCSD (1715 parameters):
    #   0: 0.4s,  625k gates (fastest transpile, most gates)
    #   1: 1.7s,  367k gates (RECOMMENDED - best gate reduction, 41% fewer)
    #   2: 4.5s,  414k gates (slower, more gates than level 1)
    #   3: 6.7s,  424k gates (slowest, more gates than level 1)
    # Level 1 is optimal: only 1.3s more than level 0, but 41% fewer gates.
    # Fewer gates = faster VQE iterations, which dominates total runtime.
    transpile_optimization_level: Final[int] = 1

    def __attrs_post_init__(self) -> None:
        """Validate active-space and orbital-canonicalization compatibility."""

        has_reduction = False
        if self.active_space is not None:
            if not isinstance(self.active_space, ActiveSpaceSpec):
                raise TypeError("active_space must be an ActiveSpaceSpec or None")
            if (
                self.frozen_core_num_orbitals not in (
                    0,
                    self.active_space.frozen_occupied_orbitals,
                )
            ):
                raise ValueError(
                    "frozen_core_num_orbitals conflicts with active_space"
                )
            has_reduction = (
                self.active_space.frozen_occupied_orbitals > 0
                or self.active_space.discarded_virtual_orbitals > 0
            )
        if has_reduction and self.frozen_core != "manual":
            raise ValueError(
                "A nontrivial active_space requires frozen_core='manual'"
            )
        if (
            self.orbital_canonicalization == _FULL_CANONICALIZATION
            and self.frozen_core != "none"
        ):
            raise ValueError(
                "Full orbital canonicalization is supported only without "
                "frozen-core reduction"
            )


@define(frozen=True, slots=True)
class CanonicalFragmentHamiltonian:
    """Effective fragment Hamiltonian in the deterministic canonical basis.

    Production active-space calculations diagonalize the effective one-electron
    Hamiltonian separately inside the SCF occupied and virtual subspaces. The
    full-space paper compatibility mode diagonalizes it without that partition.
    The object is the single source of transformed integrals for one solve.
    """

    h1: ndarray
    h2: ndarray
    rotation: ndarray
    norb: int
    nelec: int
    occupied_count: int
    orbital_energies: tuple[float, ...]
    occupied_projector_leakage: float
    digest: str


@define(frozen=True, slots=True)
class ActiveSpaceHamiltonian:
    """Closed-shell SCF-frontier projection of a canonical Hamiltonian."""

    h1: ndarray
    h2: ndarray
    norb_full: int
    nelec_full: int
    norb_active: int
    nelec_active: int
    selection: str
    frozen_indices: tuple[int, ...]
    discarded_virtual_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    frozen_energy: float
    frozen_boundary_gap_ha: float | None
    discarded_virtual_boundary_gap_ha: float | None
    minimum_boundary_gap_ha: float
    orbital_energies: tuple[float, ...]
    occupied_projector_leakage: float
    full_hamiltonian_digest: str
    active_hamiltonian_digest: str
    total_hamiltonian_digest: str

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-compatible record suitable for paper evidence."""

        return {
            "schema": "quemb.active-space-provenance/v1",
            "norb_full": self.norb_full,
            "nelec_full": self.nelec_full,
            "norb_active": self.norb_active,
            "nelec_active": self.nelec_active,
            "selection": self.selection,
            "frozen_indices": list(self.frozen_indices),
            "discarded_virtual_indices": list(
                self.discarded_virtual_indices
            ),
            "frozen_occupied_orbitals": len(self.frozen_indices),
            "discarded_virtual_orbitals": len(
                self.discarded_virtual_indices
            ),
            "active_indices": list(self.active_indices),
            "frozen_energy_ha": self.frozen_energy,
            "optimizer_objective": "H_active",
            "reported_energy": "expectation(H_active) + frozen_energy_ha",
            "frozen_boundary_gap_ha": self.frozen_boundary_gap_ha,
            "discarded_virtual_boundary_gap_ha": (
                self.discarded_virtual_boundary_gap_ha
            ),
            "minimum_boundary_gap_ha": self.minimum_boundary_gap_ha,
            "canonical_orbital_energies_ha": list(self.orbital_energies),
            "occupied_projector_leakage": self.occupied_projector_leakage,
            "full_hamiltonian_digest": self.full_hamiltonian_digest,
            "active_hamiltonian_digest": self.active_hamiltonian_digest,
            "total_hamiltonian_digest": self.total_hamiltonian_digest,
        }


@define(frozen=True, slots=True)
class AOCharacterContext:
    """Immutable fragment-level inputs required for AO-character diagnostics."""

    schema: str
    overlap: tuple[tuple[float, ...], ...]
    ao_labels: tuple[str, ...]
    ao_atom_indices: tuple[int, ...]
    atom_labels: tuple[str, ...]
    element_labels: tuple[str, ...]
    global_ao_metadata_digest: str
    digest: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-safe payload copy."""

        return {
            "schema": self.schema,
            "overlap": [list(row) for row in self.overlap],
            "ao_labels": list(self.ao_labels),
            "ao_atom_indices": list(self.ao_atom_indices),
            "atom_labels": list(self.atom_labels),
            "element_labels": list(self.element_labels),
            "global_ao_metadata_digest": self.global_ao_metadata_digest,
            "digest": self.digest,
        }


@define(frozen=True, slots=True)
class AOCharacterDiagnostic:
    """Immutable AO-character payload for one canonical fragment basis."""

    schema: str
    fragment_name: str
    norb: int
    nao: int
    atom_count: int
    element_labels: tuple[str, ...]
    residual_tolerance: float
    max_orthonormality_residual: float
    max_population_normalization_residual: float
    ao_labels: tuple[str, ...]
    ao_atom_indices: tuple[int, ...]
    atom_labels: tuple[str, ...]
    orbital_roles: tuple[str, ...]
    frozen_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    discarded_indices: tuple[int, ...]
    overlap: tuple[tuple[float, ...], ...]
    lowdin_sqrt_overlap: tuple[tuple[float, ...], ...]
    c_ao: tuple[tuple[float, ...], ...]
    lowdin_coefficients: tuple[tuple[float, ...], ...]
    ao_weights_by_orbital: tuple[tuple[float, ...], ...]
    atom_weights_by_orbital: tuple[tuple[float, ...], ...]
    element_weights_by_orbital: tuple[tuple[float, ...], ...]
    role_subspace_populations: tuple[tuple[str, dict[str, Any]], ...]
    overlap_digest: str
    c_ao_digest: str
    lowdin_sqrt_overlap_digest: str
    lowdin_coefficients_digest: str
    role_digest: str
    ao_metadata_digest: str
    ao_weights_digest: str
    atom_weights_digest: str
    element_weights_digest: str
    digest: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-safe payload copy."""

        return {
            "schema": self.schema,
            "fragment_name": self.fragment_name,
            "norb": self.norb,
            "nao": self.nao,
            "atom_count": self.atom_count,
            "element_labels": list(self.element_labels),
            "residual_tolerance": self.residual_tolerance,
            "max_orthonormality_residual": self.max_orthonormality_residual,
            "max_population_normalization_residual": (
                self.max_population_normalization_residual
            ),
            "ao_labels": list(self.ao_labels),
            "ao_atom_indices": list(self.ao_atom_indices),
            "atom_labels": list(self.atom_labels),
            "orbital_roles": list(self.orbital_roles),
            "role_indices": {
                "frozen": list(self.frozen_indices),
                "active": list(self.active_indices),
                "discarded": list(self.discarded_indices),
            },
            "overlap": [list(row) for row in self.overlap],
            "lowdin_sqrt_overlap": [
                list(row) for row in self.lowdin_sqrt_overlap
            ],
            "c_ao": [list(row) for row in self.c_ao],
            "lowdin_coefficients": [
                list(row) for row in self.lowdin_coefficients
            ],
            "ao_weights_by_orbital": [
                list(row) for row in self.ao_weights_by_orbital
            ],
            "atom_weights_by_orbital": [
                list(row) for row in self.atom_weights_by_orbital
            ],
            "element_weights_by_orbital": [
                list(row) for row in self.element_weights_by_orbital
            ],
            "role_subspace_populations": {
                role: deepcopy(payload)
                for role, payload in self.role_subspace_populations
            },
            "digests": {
                "overlap": self.overlap_digest,
                "c_ao": self.c_ao_digest,
                "lowdin_sqrt_overlap": self.lowdin_sqrt_overlap_digest,
                "lowdin_coefficients": self.lowdin_coefficients_digest,
                "roles": self.role_digest,
                "ao_metadata": self.ao_metadata_digest,
                "ao_weights": self.ao_weights_digest,
                "atom_weights": self.atom_weights_digest,
                "element_weights": self.element_weights_digest,
            },
            "digest": self.digest,
        }


# Global cache for transpiled UCCSD ansatzes
# Key: (norb, nelec, optimization_level)
# Value: transpiled QuantumCircuit
_ansatz_cache: dict[tuple[int, int, int], Any] = {}


def clear_ansatz_cache() -> None:
    """Clear the transpiled ansatz cache."""
    global _ansatz_cache
    _ansatz_cache.clear()


def reset_vqe_state() -> None:
    """
    Reset the global VQE state.

    Call this when switching between different molecules to prevent
    state contamination (e.g., warm-start parameters from previous molecule,
    BE iteration history, fragment energies/RDMs).

    Usage:
        from quemb.molbe.vqe_solver import reset_vqe_state, clear_ansatz_cache

        # Before starting a new molecule calculation:
        clear_ansatz_cache()  # Clear cached UCCSD ansatz
        reset_vqe_state()     # Clear VQE state (params, history, RDMs)
    """
    global _vqe_state
    _vqe_state.fragment_params.clear()
    _vqe_state.be_energy_history.clear()
    _vqe_state.current_be_iter = 0
    _vqe_state.fragment_iteration_history.clear()
    _vqe_state.fragment_energies.clear()
    _vqe_state.fragment_rdm1.clear()
    _vqe_state.fragment_rdm2.clear()
    _vqe_state.fragment_timings.clear()


def get_cached_ansatz(
    norb: int, nelec: int, optimization_level: int, verbose: int = 0
) -> Any:
    """
    Get a transpiled UCCSD ansatz from cache, or build and cache it.

    This avoids repeated transpilation across BE iterations, which is the
    main performance bottleneck for large systems (e.g., 24 qubits).

    Parameters
    ----------
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons
    optimization_level : int
        Transpilation optimization level (0-3)
    verbose : int
        Verbosity level

    Returns
    -------
    ansatz : QuantumCircuit
        Transpiled UCCSD ansatz circuit
    """
    global _ansatz_cache

    cache_key = (norb, nelec, optimization_level)

    if cache_key in _ansatz_cache:
        if verbose >= 1:
            print(f"  Using cached transpiled ansatz (norb={norb}, nelec={nelec})")
            sys.stdout.flush()
        return _ansatz_cache[cache_key]

    # Build and transpile
    if verbose >= 1:
        print(f"  Building UCCSD ansatz for {norb} orbitals, {nelec} electrons...")
        sys.stdout.flush()

    ansatz = build_uccsd_ansatz(norb, nelec)

    if verbose >= 2:
        print(f"  UCCSD ansatz built: {ansatz.num_parameters} parameters")
        sys.stdout.flush()

    if verbose >= 1:
        print(
            f"  Transpiling ansatz (optimization_level={optimization_level}, "
            f"this may take several minutes for large systems)..."
        )
        sys.stdout.flush()

    # Use generate_preset_pass_manager instead of transpile() to avoid segfault
    # with large UCCSD circuits (observed with Qiskit 2.3.0 / qiskit-aer 0.17.2)
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        basis_gates=["cx", "rz", "sx", "x"],
    )
    ansatz = pm.run(ansatz)

    if verbose >= 1:
        print(
            f"  Ansatz transpiled: {ansatz.num_qubits} qubits, depth={ansatz.depth()}"
        )
        print(f"  Caching transpiled ansatz for reuse in subsequent BE iterations")
        sys.stdout.flush()

    # Cache for reuse
    _ansatz_cache[cache_key] = ansatz

    return ansatz


class VQEState:
    """
    Global state for VQE solver across BE iterations.

    Stores warm-start parameters and BE convergence history.
    """

    def __init__(self):
        self.fragment_params: dict[str, ndarray] = {}  # frag_name -> optimal parameters
        self.be_energy_history: list[float] = []  # BE iteration energies
        self.current_be_iter: int = 0
        self.fragment_iteration_history: dict[str, list[dict[str, Any]]] = {}
        self.fragment_energies: dict[str, float] = {}
        self.fragment_rdm1: dict[str, ndarray] = {}
        self.fragment_rdm2: dict[str, ndarray] = {}
        self.fragment_timings: dict[str, dict[str, float]] = {}

    def get_stage(self, be_energy_change: float, args: VQE_ArgsUser) -> int:
        """Determine VQE convergence stage based on BE convergence."""
        if not args.adaptive_convergence:
            return 2  # Use stage 2 (medium) as default

        if be_energy_change < args.stage2_be_threshold:
            return 3  # Tight convergence
        elif be_energy_change < args.stage1_be_threshold:
            return 2  # Medium convergence
        else:
            return 1  # Loose convergence

    def get_vqe_params(self, stage: int, args: VQE_ArgsUser) -> tuple[int, float]:
        """Get VQE max_iter and energy_tol for given stage."""
        if stage == 1:
            return args.stage1_max_iter, args.stage1_energy_tol
        elif stage == 2:
            return args.stage2_max_iter, args.stage2_energy_tol
        else:  # stage == 3
            return args.stage3_max_iter, args.stage3_energy_tol

    def update_be_iteration(self, be_energy: float):
        """Update BE iteration counter and energy history."""
        self.be_energy_history.append(be_energy)
        self.current_be_iter += 1

    def get_be_energy_change(self) -> float:
        """Calculate BE energy change from previous iteration."""
        if len(self.be_energy_history) < 2:
            return 1.0  # Large value for first iteration
        return abs(self.be_energy_history[-1] - self.be_energy_history[-2])


# Global VQE state (persists across BE iterations)
_vqe_state = VQEState()

# Private, process-local ADAPT selector-factory seam used by the H6 BE worker.
# The default remains None so existing greedy class construction is unchanged.
_AdaptSelectorFactory = Callable[[type[Any], dict[str, Any]], Any]
_adapt_selector_factory_var: ContextVar[_AdaptSelectorFactory | None] = ContextVar(
    "_adapt_selector_factory_var",
    default=None,
)
_exact_sparse_ceo_manifest_var: ContextVar[str | None] = ContextVar(
    "_exact_sparse_ceo_manifest_var",
    default=None,
)

# Global callback registry for VQE result tracking
# Callbacks receive: (frag_name: str, energy: float, rdm1: ndarray, rdm2: ndarray)
_vqe_result_callbacks: list[Callable[[str, float, ndarray, ndarray], None]] = []


def _get_adapt_selector_factory() -> _AdaptSelectorFactory | None:
    """Return the active private ADAPT selector-factory override, if any."""
    return _adapt_selector_factory_var.get()


def _get_exact_sparse_ceo_manifest() -> str | None:
    """Return the active private CEO manifest path, if any."""
    return _exact_sparse_ceo_manifest_var.get()


@contextmanager
def _adapt_selector_factory_context(factory: _AdaptSelectorFactory | None):
    """Temporarily install a private ADAPT selector-factory override."""
    if factory is not None and not callable(factory):
        raise TypeError("ADAPT selector factory must be callable or None")
    token = _adapt_selector_factory_var.set(factory)
    try:
        yield
    finally:
        _adapt_selector_factory_var.reset(token)


@contextmanager
def _exact_sparse_ceo_manifest_context(manifest_path: str | Path | None):
    """Temporarily install a private CEO exact-sparse pool manifest path."""
    manifest_str = None if manifest_path is None else str(manifest_path)
    token = _exact_sparse_ceo_manifest_var.set(manifest_str)
    try:
        yield
    finally:
        _exact_sparse_ceo_manifest_var.reset(token)


def _validate_adapt_selector_factory_usage(
    vqe_args: VQE_ArgsUser, *, use_adapt: bool
) -> _AdaptSelectorFactory | None:
    """Reject private selector-factory use outside the exact-sparse CEO path."""
    selector_factory = _get_adapt_selector_factory()
    if selector_factory is None:
        return None

    private_manifest = _get_exact_sparse_ceo_manifest()
    active_spec = resolve_active_space_spec(vqe_args)
    has_explicit_reduction = (
        active_spec.frozen_occupied_orbitals > 0
        or active_spec.discarded_virtual_orbitals > 0
    )
    downfolding_is_explicit = (
        vqe_args.frozen_core == "none" and not has_explicit_reduction
    ) or (
        vqe_args.frozen_core == "manual" and has_explicit_reduction
    )
    supports_private_selector_factory = (
        use_adapt
        and private_manifest is not None
        and vqe_args.ansatz_type in ("adapt_fast", "adapt_matrix_free")
        and vqe_args.estimator_type == "direct_sv"
        and downfolding_is_explicit
    )
    if supports_private_selector_factory:
        return selector_factory

    raise ValueError(
        "Private ADAPT selector factory requires a private CEO manifest "
        "context with ansatz_type 'adapt_fast' or 'adapt_matrix_free', "
        "estimator_type='direct_sv', and either frozen_core='none' or a "
        "nonempty explicit manual ActiveSpaceSpec."
    )


def _validate_private_ceo_manifest_downfolding(
    manifest: Any,
    vqe_args: VQE_ArgsUser,
) -> None:
    """Reject Hamiltonian-backed CEO artifacts for a projected Hamiltonian."""

    if vqe_args.frozen_core != "manual":
        return
    if getattr(manifest, "artifact_kind", None) != "pool_only":
        raise ValueError(
            "Manual downfolding with a private CEO manifest requires a "
            "pool-only artifact resolved by active-space size and filling"
        )
    if getattr(manifest, "hamiltonian_digest", None) is not None:
        raise ValueError(
            "A pool-only CEO artifact used with manual downfolding must not "
            "pin a source Hamiltonian"
        )


def _build_adapt_vqe(
    adapt_class: type[Any],
    *,
    adapt_kwargs: dict[str, Any],
    pool_provenance: dict[str, Any],
    selector_factory: _AdaptSelectorFactory | None,
) -> Any:
    """Instantiate the selected ADAPT backend, optionally through a wrapper."""
    if selector_factory is None:
        adapt_vqe = adapt_class(**adapt_kwargs)
    else:
        adapt_vqe = selector_factory(adapt_class, dict(adapt_kwargs))
        if adapt_vqe is None:
            raise TypeError(
                "ADAPT selector factory must return an ADAPT-VQE instance"
            )
    adapt_vqe.pool_provenance = dict(pool_provenance)
    return adapt_vqe


def register_vqe_callback(
    callback: Callable[[str, float, ndarray, ndarray], None],
) -> None:
    """
    Register a callback to be called after each VQE fragment solve.

    The callback receives:
    - frag_name: str - fragment name
    - energy: float - fragment energy
    - rdm1: ndarray - 1-RDM in fragment MO basis
    - rdm2: ndarray - 2-RDM in fragment MO basis

    This is useful for tracking best results across BE iterations.
    """
    _vqe_result_callbacks.append(callback)


def clear_vqe_callbacks() -> None:
    """Clear all registered VQE callbacks."""
    _vqe_result_callbacks.clear()


def _hamiltonian_digest(h1: ndarray, h2: ndarray, nelec: int) -> str:
    """Hash an integral Hamiltonian without depending on text formatting."""

    digest = sha256()
    digest.update(b"quemb.spatial-hamiltonian/v1\0")
    digest.update(int(nelec).to_bytes(8, byteorder="little", signed=True))
    for label, values in ((b"h1", h1), (b"h2", h2)):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(label)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


_AO_CHARACTER_SCHEMA = "quemb.butane.001b.ao-character/v1"
_AO_CHARACTER_CONTEXT_SCHEMA = "quemb.butane.001b.ao-character-context/v1"
_AO_CHARACTER_CONTEXT_ATTR = "ao_character_context"
_AO_CHARACTER_EVIDENCE_ATTR = "ao_character_evidence"
_AO_CHARACTER_LATEST_ATTR = "ao_character_latest"


def _coerce_real_matrix(
    name: str,
    values: Any,
    *,
    shape: tuple[int, ...] | None = None,
    symmetric: bool = False,
) -> ndarray:
    """Return a finite real matrix, accepting only negligible imaginary parts."""

    array = np.asarray(values)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, found {array.shape}")
    if np.iscomplexobj(array):
        imaginary = np.asarray(np.imag(array), dtype=float)
        if imaginary.size and float(np.max(np.abs(imaginary))) > 1.0e-12:
            raise ValueError(f"{name} must be real-valued")
        array = np.real(array)
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if symmetric and not np.allclose(array, array.T, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return array


def _coerce_index_tuple(
    name: str,
    values: Sequence[int] | tuple[int, ...],
    *,
    upper_bound: int,
) -> tuple[int, ...]:
    """Validate an integer index sequence."""

    result: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"{name} must contain only integers")
        index = int(raw)
        if index < 0 or index >= upper_bound:
            raise ValueError(
                f"{name} must contain indices in [0, {upper_bound}), found {index}"
            )
        result.append(index)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique indices")
    return tuple(result)


def _coerce_index_sequence(
    name: str,
    values: Sequence[int] | tuple[int, ...],
    *,
    upper_bound: int,
) -> tuple[int, ...]:
    """Validate an integer index sequence that may contain duplicates."""

    result: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"{name} must contain only integers")
        index = int(raw)
        if index < 0 or index >= upper_bound:
            raise ValueError(
                f"{name} must contain indices in [0, {upper_bound}), found {index}"
            )
        result.append(index)
    return tuple(result)


def _coerce_label_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    """Validate a non-empty sequence of non-empty strings."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of labels")
    result: list[str] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise ValueError(f"{name}[{index}] must be a string")
        label = raw.strip()
        if not label:
            raise ValueError(f"{name}[{index}] must be non-empty")
        result.append(label)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return tuple(result)


def _matrix_to_tuple(matrix: ndarray) -> tuple[tuple[float, ...], ...]:
    """Freeze a numeric matrix into JSON-safe tuples."""

    contiguous = np.asarray(matrix, dtype=float)
    return tuple(
        tuple(float(value) for value in contiguous[row_index].tolist())
        for row_index in range(contiguous.shape[0])
    )


def _extract_element_label(atom_label: str) -> str:
    """Derive an element symbol from an explicit atom label."""

    match = re.match(r"^\s*([A-Za-z]+)", atom_label)
    if match is None:
        raise ValueError(
            f"atom_labels entries must begin with an element symbol, found {atom_label!r}"
        )
    symbol = match.group(1)
    return symbol[0].upper() + symbol[1:].lower()


def _element_labels_from_atom_labels(
    atom_labels: tuple[str, ...],
) -> tuple[str, ...]:
    """Derive ordered unique element labels from ordered atom labels."""

    element_labels: list[str] = []
    for label in atom_labels:
        element = _extract_element_label(label)
        if element not in element_labels:
            element_labels.append(element)
    return tuple(element_labels)


def _ao_metadata_digest(
    *,
    ao_labels: tuple[str, ...],
    ao_atom_indices: tuple[int, ...],
    atom_labels: tuple[str, ...],
    element_labels: tuple[str, ...],
) -> str:
    """Hash the ordered AO/atom/element metadata contract."""

    return canonical_digest(
        {
            "ao_labels": list(ao_labels),
            "ao_atom_indices": list(ao_atom_indices),
            "atom_labels": list(atom_labels),
            "element_labels": list(element_labels),
        }
    )


def build_ao_character_context(
    *,
    overlap: ndarray,
    ao_labels: Sequence[str],
    ao_atom_indices: Sequence[int],
    atom_labels: Sequence[str],
) -> AOCharacterContext:
    """Build a validated immutable AO-character context for later solves."""

    ao_labels_tuple = _coerce_label_tuple("ao_labels", ao_labels)
    atom_labels_tuple = _coerce_label_tuple("atom_labels", atom_labels)
    ao_atom_indices_tuple = _coerce_index_sequence(
        "ao_atom_indices",
        ao_atom_indices,
        upper_bound=len(atom_labels_tuple),
    )
    nao = len(ao_labels_tuple)
    if len(ao_atom_indices_tuple) != nao:
        raise ValueError("ao_atom_indices must have the same length as ao_labels")
    overlap_matrix = _coerce_real_matrix(
        "overlap",
        overlap,
        shape=(nao, nao),
        symmetric=True,
    )
    _symmetric_matrix_square_root(overlap_matrix)
    element_labels = _element_labels_from_atom_labels(atom_labels_tuple)
    ao_metadata_digest = _ao_metadata_digest(
        ao_labels=ao_labels_tuple,
        ao_atom_indices=ao_atom_indices_tuple,
        atom_labels=atom_labels_tuple,
        element_labels=element_labels,
    )
    payload = {
        "schema": _AO_CHARACTER_CONTEXT_SCHEMA,
        "overlap": [list(row) for row in _matrix_to_tuple(overlap_matrix)],
        "ao_labels": list(ao_labels_tuple),
        "ao_atom_indices": list(ao_atom_indices_tuple),
        "atom_labels": list(atom_labels_tuple),
        "element_labels": list(element_labels),
        "global_ao_metadata_digest": ao_metadata_digest,
    }
    return AOCharacterContext(
        schema=_AO_CHARACTER_CONTEXT_SCHEMA,
        overlap=_matrix_to_tuple(overlap_matrix),
        ao_labels=ao_labels_tuple,
        ao_atom_indices=ao_atom_indices_tuple,
        atom_labels=atom_labels_tuple,
        element_labels=element_labels,
        global_ao_metadata_digest=ao_metadata_digest,
        digest=canonical_digest(payload),
    )


def _coerce_ao_character_context(context: Any) -> AOCharacterContext:
    """Validate a fragment AO-character context attribute."""

    if isinstance(context, AOCharacterContext):
        candidate = context
    elif isinstance(context, dict):
        required = {
            "schema",
            "overlap",
            "ao_labels",
            "ao_atom_indices",
            "atom_labels",
            "element_labels",
            "global_ao_metadata_digest",
            "digest",
        }
        if set(context) != required:
            raise ValueError(
                "ao_character_context must contain exactly "
                f"{sorted(required)!r}"
            )
        ao_labels_tuple = _coerce_label_tuple("ao_labels", context["ao_labels"])
        atom_labels_tuple = _coerce_label_tuple(
            "atom_labels", context["atom_labels"]
        )
        ao_atom_indices_tuple = _coerce_index_sequence(
            "ao_atom_indices",
            context["ao_atom_indices"],
            upper_bound=len(atom_labels_tuple),
        )
        if len(ao_atom_indices_tuple) != len(ao_labels_tuple):
            raise ValueError(
                "ao_character_context ao_atom_indices must have the same length as ao_labels"
            )
        overlap_matrix = _coerce_real_matrix(
            "ao_character_context overlap",
            context["overlap"],
            shape=(len(ao_labels_tuple), len(ao_labels_tuple)),
            symmetric=True,
        )
        element_labels_tuple = _coerce_label_tuple(
            "element_labels", context["element_labels"]
        )
        candidate = AOCharacterContext(
            schema=str(context["schema"]),
            overlap=_matrix_to_tuple(overlap_matrix),
            ao_labels=ao_labels_tuple,
            ao_atom_indices=ao_atom_indices_tuple,
            atom_labels=atom_labels_tuple,
            element_labels=element_labels_tuple,
            global_ao_metadata_digest=str(
                context["global_ao_metadata_digest"]
            ),
            digest=str(context["digest"]),
        )
    else:
        raise ValueError(
            "ao_character_context must be an AOCharacterContext or mapping"
        )

    if candidate.schema != _AO_CHARACTER_CONTEXT_SCHEMA:
        raise ValueError(
            "ao_character_context schema must equal "
            f"{_AO_CHARACTER_CONTEXT_SCHEMA!r}"
        )
    expected_element_labels = _element_labels_from_atom_labels(
        candidate.atom_labels
    )
    if candidate.element_labels != expected_element_labels:
        raise ValueError(
            "ao_character_context element_labels do not match atom_labels"
        )
    expected_ao_metadata_digest = _ao_metadata_digest(
        ao_labels=candidate.ao_labels,
        ao_atom_indices=candidate.ao_atom_indices,
        atom_labels=candidate.atom_labels,
        element_labels=candidate.element_labels,
    )
    if candidate.global_ao_metadata_digest != expected_ao_metadata_digest:
        raise ValueError(
            "ao_character_context global_ao_metadata_digest is stale"
        )
    _symmetric_matrix_square_root(np.asarray(candidate.overlap, dtype=float))
    payload = {
        "schema": candidate.schema,
        "overlap": [list(row) for row in candidate.overlap],
        "ao_labels": list(candidate.ao_labels),
        "ao_atom_indices": list(candidate.ao_atom_indices),
        "atom_labels": list(candidate.atom_labels),
        "element_labels": list(candidate.element_labels),
        "global_ao_metadata_digest": candidate.global_ao_metadata_digest,
    }
    if candidate.digest != canonical_digest(payload):
        raise ValueError("ao_character_context digest mismatch")
    return candidate


def set_fragment_ao_character_context(
    fragment: Any, context: AOCharacterContext | dict[str, Any]
) -> AOCharacterContext:
    """Install a validated AO-character context on a fragment-like object."""

    validated = _coerce_ao_character_context(context)
    setattr(fragment, _AO_CHARACTER_CONTEXT_ATTR, validated)
    return validated


def _symmetric_matrix_square_root(overlap: ndarray) -> ndarray:
    """Return the deterministic symmetric square root of an SPD overlap."""

    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("overlap eigenvalues must be finite")
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue <= 0.0:
        raise ValueError(
            "overlap must be strictly positive definite; "
            f"minimum eigenvalue={min_eigenvalue:.12g}"
        )
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _validate_role_partition(
    *,
    canonical: CanonicalFragmentHamiltonian,
    active: ActiveSpaceHamiltonian,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    """Validate and materialize the frozen/active/discarded role partition."""

    if active.norb_full != canonical.norb:
        raise ValueError("active-space role partition does not match canonical norb")
    frozen_indices = _coerce_index_tuple(
        "frozen_indices", active.frozen_indices, upper_bound=canonical.norb
    )
    active_indices = _coerce_index_tuple(
        "active_indices", active.active_indices, upper_bound=canonical.norb
    )
    discarded_indices = _coerce_index_tuple(
        "discarded_virtual_indices",
        active.discarded_virtual_indices,
        upper_bound=canonical.norb,
    )
    covered = set(frozen_indices) | set(active_indices) | set(discarded_indices)
    if len(covered) != (
        len(frozen_indices) + len(active_indices) + len(discarded_indices)
    ):
        raise ValueError("role indices must be pairwise disjoint")
    expected = set(range(canonical.norb))
    if covered != expected:
        raise ValueError("role indices must cover every canonical orbital exactly once")
    orbital_roles = [""] * canonical.norb
    for index in frozen_indices:
        orbital_roles[index] = "frozen"
    for index in active_indices:
        orbital_roles[index] = "active"
    for index in discarded_indices:
        orbital_roles[index] = "discarded"
    return (
        frozen_indices,
        active_indices,
        discarded_indices,
        tuple(orbital_roles),
    )


def build_ao_character_diagnostic(
    canonical: CanonicalFragmentHamiltonian,
    active: ActiveSpaceHamiltonian,
    fragment: Any,
    *,
    overlap: ndarray,
    ao_labels: Sequence[str],
    ao_atom_indices: Sequence[int],
    atom_labels: Sequence[str],
    residual_tolerance: float = 1.0e-10,
) -> AOCharacterDiagnostic:
    """Build a symmetric-Lowdin AO-character diagnostic for canonical orbitals."""

    if (
        isinstance(residual_tolerance, bool)
        or not isinstance(
            residual_tolerance, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(residual_tolerance)
        or residual_tolerance < 0.0
    ):
        raise ValueError("residual_tolerance must be finite and non-negative")
    if not isinstance(canonical, CanonicalFragmentHamiltonian):
        raise TypeError("canonical must be a CanonicalFragmentHamiltonian")
    if not isinstance(active, ActiveSpaceHamiltonian):
        raise TypeError("active must be an ActiveSpaceHamiltonian")
    if not hasattr(fragment, "TA") or not hasattr(fragment, "mo_coeffs"):
        raise ValueError("fragment must provide TA and mo_coeffs")
    if active.full_hamiltonian_digest != canonical.digest:
        raise ValueError(
            "active full_hamiltonian_digest must match canonical.digest"
        )

    ao_labels_tuple = _coerce_label_tuple("ao_labels", ao_labels)
    atom_labels_tuple = _coerce_label_tuple("atom_labels", atom_labels)
    ao_atom_indices_tuple = _coerce_index_sequence(
        "ao_atom_indices",
        ao_atom_indices,
        upper_bound=len(atom_labels_tuple),
    )
    nao = len(ao_labels_tuple)
    if len(ao_atom_indices_tuple) != nao:
        raise ValueError("ao_atom_indices must have the same length as ao_labels")

    overlap_matrix = _coerce_real_matrix(
        "overlap",
        overlap,
        shape=(nao, nao),
        symmetric=True,
    )
    ta = _coerce_real_matrix("fragment.TA", fragment.TA)
    mo_coeffs = _coerce_real_matrix("fragment.mo_coeffs", fragment.mo_coeffs)
    rotation = _coerce_real_matrix(
        "canonical.rotation",
        canonical.rotation,
        shape=(canonical.norb, canonical.norb),
    )
    if ta.ndim != 2:
        raise ValueError("fragment.TA must be a rank-2 matrix")
    if mo_coeffs.ndim != 2:
        raise ValueError("fragment.mo_coeffs must be a rank-2 matrix")
    if ta.shape[0] != nao:
        raise ValueError(
            "fragment.TA row count must match the AO overlap/label dimension"
        )
    if mo_coeffs.shape[1] != canonical.norb:
        raise ValueError(
            "fragment.mo_coeffs column count must match canonical.norb"
        )
    if ta.shape[1] != mo_coeffs.shape[0]:
        raise ValueError(
            "fragment.TA column count must match fragment.mo_coeffs row count"
        )

    (
        frozen_indices,
        active_indices,
        discarded_indices,
        orbital_roles,
    ) = _validate_role_partition(canonical=canonical, active=active)

    lowdin_sqrt_overlap = _symmetric_matrix_square_root(overlap_matrix)
    c_ao = ta @ mo_coeffs @ rotation
    c_ao = _coerce_real_matrix("C_AO", c_ao, shape=(nao, canonical.norb))
    lowdin_coefficients = lowdin_sqrt_overlap @ c_ao
    lowdin_coefficients = _coerce_real_matrix(
        "lowdin_coefficients",
        lowdin_coefficients,
        shape=(nao, canonical.norb),
    )

    orthonormality = lowdin_coefficients.T @ lowdin_coefficients
    max_orthonormality_residual = float(
        np.max(np.abs(orthonormality - np.eye(canonical.norb)))
    )
    if max_orthonormality_residual > float(residual_tolerance):
        raise ValueError(
            "Lowdin orthonormality residual exceeds tolerance: "
            f"{max_orthonormality_residual:.3e} > {float(residual_tolerance):.3e}"
        )

    ao_weights = np.square(np.abs(lowdin_coefficients)).T
    atom_weights = np.zeros((canonical.norb, len(atom_labels_tuple)), dtype=float)
    for ao_index, atom_index in enumerate(ao_atom_indices_tuple):
        atom_weights[:, atom_index] += ao_weights[:, ao_index]

    element_labels_list: list[str] = []
    atom_to_element: list[int] = []
    for label in atom_labels_tuple:
        element = _extract_element_label(label)
        try:
            element_index = element_labels_list.index(element)
        except ValueError:
            element_labels_list.append(element)
            element_index = len(element_labels_list) - 1
        atom_to_element.append(element_index)
    element_labels = tuple(element_labels_list)
    element_weights = np.zeros((canonical.norb, len(element_labels)), dtype=float)
    for atom_index, element_index in enumerate(atom_to_element):
        element_weights[:, element_index] += atom_weights[:, atom_index]

    per_orbital_residuals = [
        np.abs(np.sum(ao_weights, axis=1) - 1.0),
        np.abs(np.sum(atom_weights, axis=1) - 1.0),
        np.abs(np.sum(element_weights, axis=1) - 1.0),
    ]
    role_subspace_populations: list[tuple[str, dict[str, Any]]] = []
    for role_name, indices in (
        ("frozen", frozen_indices),
        ("active", active_indices),
        ("discarded", discarded_indices),
    ):
        role_ao = (
            np.sum(ao_weights[list(indices)], axis=0)
            if indices
            else np.zeros(nao)
        )
        role_atom = (
            np.sum(atom_weights[list(indices)], axis=0)
            if indices
            else np.zeros(len(atom_labels_tuple))
        )
        role_element = (
            np.sum(element_weights[list(indices)], axis=0)
            if indices
            else np.zeros(len(element_labels))
        )
        expected_total = float(len(indices))
        role_total_residual = max(
            abs(float(np.sum(role_ao)) - expected_total),
            abs(float(np.sum(role_atom)) - expected_total),
            abs(float(np.sum(role_element)) - expected_total),
        )
        per_orbital_residuals.extend(
            [np.asarray([role_total_residual])]
        )
        role_subspace_populations.append(
            (
                role_name,
                {
                    "orbital_indices": list(indices),
                    "orbital_count": len(indices),
                    "ao_totals": [float(value) for value in role_ao.tolist()],
                    "atom_totals": [float(value) for value in role_atom.tolist()],
                    "element_totals": [
                        float(value) for value in role_element.tolist()
                    ],
                    "total_population": expected_total,
                },
            )
        )

    max_population_normalization_residual = float(
        max(
            float(np.max(residuals)) if residuals.size else 0.0
            for residuals in per_orbital_residuals
        )
    )
    if max_population_normalization_residual > float(residual_tolerance):
        raise ValueError(
            "AO/atom/element population normalization residual exceeds tolerance: "
            f"{max_population_normalization_residual:.3e} > "
            f"{float(residual_tolerance):.3e}"
        )

    overlap_payload = _matrix_to_tuple(overlap_matrix)
    lowdin_sqrt_overlap_payload = _matrix_to_tuple(lowdin_sqrt_overlap)
    c_ao_payload = _matrix_to_tuple(c_ao)
    lowdin_coefficients_payload = _matrix_to_tuple(lowdin_coefficients)
    ao_weights_payload = _matrix_to_tuple(ao_weights)
    atom_weights_payload = _matrix_to_tuple(atom_weights)
    element_weights_payload = _matrix_to_tuple(element_weights)
    role_payload = {
        "orbital_roles": list(orbital_roles),
        "frozen_indices": list(frozen_indices),
        "active_indices": list(active_indices),
        "discarded_indices": list(discarded_indices),
    }
    ao_metadata_digest = _ao_metadata_digest(
        ao_labels=ao_labels_tuple,
        ao_atom_indices=ao_atom_indices_tuple,
        atom_labels=atom_labels_tuple,
        element_labels=element_labels,
    )
    top_level_payload = {
        "schema": _AO_CHARACTER_SCHEMA,
        "fragment_name": str(getattr(fragment, "dname", "unknown")),
        "norb": canonical.norb,
        "nao": nao,
        "atom_count": len(atom_labels_tuple),
        "element_labels": list(element_labels),
        "residual_tolerance": float(residual_tolerance),
        "max_orthonormality_residual": max_orthonormality_residual,
        "max_population_normalization_residual": (
            max_population_normalization_residual
        ),
        "ao_labels": list(ao_labels_tuple),
        "ao_atom_indices": list(ao_atom_indices_tuple),
        "atom_labels": list(atom_labels_tuple),
        "orbital_roles": list(orbital_roles),
        "role_indices": {
            "frozen": list(frozen_indices),
            "active": list(active_indices),
            "discarded": list(discarded_indices),
        },
        "overlap": [list(row) for row in overlap_payload],
        "lowdin_sqrt_overlap": [
            list(row) for row in lowdin_sqrt_overlap_payload
        ],
        "c_ao": [list(row) for row in c_ao_payload],
        "lowdin_coefficients": [
            list(row) for row in lowdin_coefficients_payload
        ],
        "ao_weights_by_orbital": [list(row) for row in ao_weights_payload],
        "atom_weights_by_orbital": [
            list(row) for row in atom_weights_payload
        ],
        "element_weights_by_orbital": [
            list(row) for row in element_weights_payload
        ],
        "role_subspace_populations": {
            role: deepcopy(payload) for role, payload in role_subspace_populations
        },
        "digests": {
            "overlap": canonical_digest(overlap_payload),
            "c_ao": canonical_digest(c_ao_payload),
            "lowdin_sqrt_overlap": canonical_digest(
                lowdin_sqrt_overlap_payload
            ),
            "lowdin_coefficients": canonical_digest(
                lowdin_coefficients_payload
            ),
            "roles": canonical_digest(role_payload),
            "ao_metadata": ao_metadata_digest,
            "ao_weights": canonical_digest(ao_weights_payload),
            "atom_weights": canonical_digest(atom_weights_payload),
            "element_weights": canonical_digest(element_weights_payload),
        },
    }
    return AOCharacterDiagnostic(
        schema=_AO_CHARACTER_SCHEMA,
        fragment_name=str(getattr(fragment, "dname", "unknown")),
        norb=canonical.norb,
        nao=nao,
        atom_count=len(atom_labels_tuple),
        element_labels=element_labels,
        residual_tolerance=float(residual_tolerance),
        max_orthonormality_residual=max_orthonormality_residual,
        max_population_normalization_residual=(
            max_population_normalization_residual
        ),
        ao_labels=ao_labels_tuple,
        ao_atom_indices=ao_atom_indices_tuple,
        atom_labels=atom_labels_tuple,
        orbital_roles=orbital_roles,
        frozen_indices=frozen_indices,
        active_indices=active_indices,
        discarded_indices=discarded_indices,
        overlap=overlap_payload,
        lowdin_sqrt_overlap=lowdin_sqrt_overlap_payload,
        c_ao=c_ao_payload,
        lowdin_coefficients=lowdin_coefficients_payload,
        ao_weights_by_orbital=ao_weights_payload,
        atom_weights_by_orbital=atom_weights_payload,
        element_weights_by_orbital=element_weights_payload,
        role_subspace_populations=tuple(role_subspace_populations),
        overlap_digest=top_level_payload["digests"]["overlap"],
        c_ao_digest=top_level_payload["digests"]["c_ao"],
        lowdin_sqrt_overlap_digest=top_level_payload["digests"][
            "lowdin_sqrt_overlap"
        ],
        lowdin_coefficients_digest=top_level_payload["digests"][
            "lowdin_coefficients"
        ],
        role_digest=top_level_payload["digests"]["roles"],
        ao_metadata_digest=top_level_payload["digests"]["ao_metadata"],
        ao_weights_digest=top_level_payload["digests"]["ao_weights"],
        atom_weights_digest=top_level_payload["digests"]["atom_weights"],
        element_weights_digest=top_level_payload["digests"][
            "element_weights"
        ],
        digest=canonical_digest(top_level_payload),
    )


def record_fragment_ao_character_diagnostic(
    fragment: Any,
    canonical: CanonicalFragmentHamiltonian,
    active: ActiveSpaceHamiltonian,
) -> AOCharacterDiagnostic | None:
    """Consume fragment AO context, append immutable evidence, and summarize it."""

    raw_context = getattr(fragment, _AO_CHARACTER_CONTEXT_ATTR, None)
    if raw_context is None:
        return None
    context = _coerce_ao_character_context(raw_context)
    diagnostic = build_ao_character_diagnostic(
        canonical,
        active,
        fragment,
        overlap=np.asarray(context.overlap, dtype=float),
        ao_labels=context.ao_labels,
        ao_atom_indices=context.ao_atom_indices,
        atom_labels=context.atom_labels,
    )
    if diagnostic.ao_metadata_digest != context.global_ao_metadata_digest:
        raise ValueError(
            "ao_character_context global AO metadata is stale for this diagnostic"
        )

    existing = getattr(fragment, _AO_CHARACTER_EVIDENCE_ATTR, ())
    if not isinstance(existing, tuple) or any(
        not isinstance(item, AOCharacterDiagnostic) for item in existing
    ):
        raise ValueError(
            "fragment ao_character_evidence must be a tuple of AOCharacterDiagnostic"
        )
    updated = existing + (diagnostic,)
    setattr(fragment, _AO_CHARACTER_EVIDENCE_ATTR, updated)
    setattr(fragment, _AO_CHARACTER_LATEST_ATTR, diagnostic)

    provenance = getattr(fragment, "active_space_provenance", None)
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("fragment active_space_provenance must be a dict")
        provenance["ao_character_schema"] = diagnostic.schema
        provenance["ao_character_digest"] = diagnostic.digest
        provenance["ao_character_metadata_digest"] = (
            diagnostic.ao_metadata_digest
        )
        provenance["ao_character_context_digest"] = context.digest
        provenance["ao_character_evidence_count"] = len(updated)
    return diagnostic


def _active_space_total_digest(
    *,
    full_digest: str,
    active_digest: str,
    frozen_indices: tuple[int, ...],
    discarded_virtual_indices: tuple[int, ...],
    frozen_energy: float,
    selection: str,
    minimum_boundary_gap_ha: float,
) -> str:
    """Hash the active objective together with its frozen constant."""

    digest = sha256()
    digest.update(b"quemb.active-space-total-hamiltonian/v1\0")
    digest.update(selection.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        np.asarray([minimum_boundary_gap_ha], dtype="<f8").tobytes()
    )
    digest.update(b"\0")
    digest.update(full_digest.encode("ascii"))
    digest.update(b"\0")
    digest.update(active_digest.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        np.asarray(
            [len(frozen_indices), len(discarded_virtual_indices)],
            dtype="<i8",
        ).tobytes()
    )
    digest.update(b"\0")
    digest.update(
        np.asarray(frozen_indices, dtype="<i8").tobytes(order="C")
    )
    digest.update(b"\0")
    digest.update(
        np.asarray(discarded_virtual_indices, dtype="<i8").tobytes(order="C")
    )
    digest.update(b"\0")
    digest.update(np.asarray([frozen_energy], dtype="<f8").tobytes())
    return f"sha256:{digest.hexdigest()}"


def build_canonical_fragment_hamiltonian(
    frag: Frags,
    orbital_canonicalization: Literal["occupied_virtual", "full"] = (
        _BLOCK_CANONICALIZATION
    ),
) -> CanonicalFragmentHamiltonian:
    """Construct the current fragment Hamiltonian in one canonical basis.

    Both FCI and VQE call this function.  Consequently an occupied-orbital
    projection cannot silently use the SCF orbital ordering in one solver and
    the effective-Hamiltonian ordering in the other. ``occupied_virtual``
    preserves the SCF occupied projector. ``full`` reproduces the orbital basis
    used by the published H4/F2 calculations and is restricted to full-space
    VQE calculations.
    """

    with h5py.File(frag.eri_file, "r") as integral_file:
        eri = integral_file[frag.dname][()]
    eri_ao = ao2mo.restore(1, eri, frag.nao)

    if not hasattr(frag, "_effective_h1e"):
        raise RuntimeError(
            "SCF must be run before constructing the canonical Hamiltonian"
        )
    if frag.mo_coeffs is None:
        raise RuntimeError("Fragment MO coefficients are not available")

    coefficients = np.asarray(frag.mo_coeffs)
    h1_mo = np.einsum(
        "ip,ij,jq->pq",
        coefficients,
        frag._effective_h1e,
        coefficients,
        optimize=True,
    )
    h2_mo = np.einsum(
        "ip,jq,kr,ls,ijkl->pqrs",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        eri_ao,
        optimize=True,
    )

    norb = int(h1_mo.shape[0])
    occupied_count = int(frag.nsocc)
    if occupied_count <= 0 or occupied_count >= norb:
        raise ValueError(
            "Occupied/virtual block canonicalization requires at least one "
            "orbital in each block"
        )

    if orbital_canonicalization == _BLOCK_CANONICALIZATION:
        occupied_energies, occupied_rotation = np.linalg.eigh(
            h1_mo[:occupied_count, :occupied_count]
        )
        occupied_order = np.argsort(occupied_energies, kind="stable")
        occupied_energies = occupied_energies[occupied_order]
        occupied_rotation = occupied_rotation[:, occupied_order]
        virtual_energies, virtual_rotation = np.linalg.eigh(
            h1_mo[occupied_count:, occupied_count:]
        )
        virtual_order = np.argsort(virtual_energies, kind="stable")
        virtual_energies = virtual_energies[virtual_order]
        virtual_rotation = virtual_rotation[:, virtual_order]

        rotation = np.zeros_like(h1_mo)
        rotation[:occupied_count, :occupied_count] = occupied_rotation
        rotation[occupied_count:, occupied_count:] = virtual_rotation
        orbital_energies = np.concatenate((occupied_energies, virtual_energies))
    elif orbital_canonicalization == _FULL_CANONICALIZATION:
        orbital_energies, rotation = np.linalg.eigh(h1_mo)
        orbital_order = np.argsort(orbital_energies, kind="stable")
        orbital_energies = orbital_energies[orbital_order]
        rotation = rotation[:, orbital_order]
    else:
        raise ValueError(
            "orbital_canonicalization must be 'occupied_virtual' or 'full'"
        )

    # Eigenvectors are defined only up to a sign. Fix each phase so hashes and
    # FCIDUMPs remain stable across otherwise equivalent runs.
    for column in range(norb):
        pivot = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[pivot, column] < 0.0:
            rotation[:, column] *= -1.0

    h1 = rotation.T @ h1_mo @ rotation
    h2 = np.einsum(
        "ip,jq,kr,ls,ijkl->pqrs",
        rotation,
        rotation,
        rotation,
        rotation,
        h2_mo,
        optimize=True,
    )
    nelec = 2 * int(frag.nsocc)
    occupied_projector_leakage = float(
        np.linalg.norm(rotation[occupied_count:, :occupied_count])
    )
    if (
        orbital_canonicalization == _BLOCK_CANONICALIZATION
        and occupied_projector_leakage > 1.0e-12
    ):
        raise RuntimeError(
            "Occupied/virtual block canonicalization leaked outside the SCF "
            f"occupied projector: {occupied_projector_leakage:.3e}"
        )
    return CanonicalFragmentHamiltonian(
        h1=h1,
        h2=h2,
        rotation=rotation,
        norb=norb,
        nelec=nelec,
        occupied_count=occupied_count,
        orbital_energies=tuple(float(value) for value in orbital_energies),
        occupied_projector_leakage=occupied_projector_leakage,
        digest=_hamiltonian_digest(h1, h2, nelec),
    )


def resolve_active_space_spec(arguments: Any) -> ActiveSpaceSpec:
    """Resolve a solver argument object to one unambiguous active-space spec."""

    explicit = getattr(arguments, "active_space", None)
    legacy_count = getattr(arguments, "frozen_core_num_orbitals", 0)
    if explicit is None:
        return ActiveSpaceSpec(frozen_occupied_orbitals=legacy_count)
    if not isinstance(explicit, ActiveSpaceSpec):
        raise TypeError("active_space must be an ActiveSpaceSpec or None")
    if legacy_count not in (0, explicit.frozen_occupied_orbitals):
        raise ValueError("frozen_core_num_orbitals conflicts with active_space")
    return explicit


def build_active_space_hamiltonian(
    canonical: CanonicalFragmentHamiltonian,
    active_space: int | ActiveSpaceSpec,
) -> ActiveSpaceHamiltonian:
    """Retain the SCF-block frontier selected by ``active_space``.

    An integer remains a backward-compatible shorthand for freezing that many
    lowest occupied orbitals while retaining every virtual orbital.
    """

    if isinstance(active_space, ActiveSpaceSpec):
        spec = active_space
    elif isinstance(active_space, bool) or not isinstance(
        active_space, (int, np.integer)
    ):
        raise TypeError("active_space must be an integer or ActiveSpaceSpec")
    else:
        spec = ActiveSpaceSpec(frozen_occupied_orbitals=int(active_space))
    frozen_count = spec.frozen_occupied_orbitals
    discarded_virtual_count = spec.discarded_virtual_orbitals
    occupied_count = canonical.occupied_count
    if canonical.h1.shape != (canonical.norb, canonical.norb):
        raise ValueError("Canonical h1 dimension does not match norb")
    if canonical.h2.shape != (canonical.norb,) * 4:
        raise ValueError("Canonical h2 dimension does not match norb")
    if canonical.rotation.shape != (canonical.norb, canonical.norb):
        raise ValueError("Canonical rotation dimension does not match norb")
    if len(canonical.orbital_energies) != canonical.norb:
        raise ValueError("Canonical orbital-energy count does not match norb")
    if canonical.nelec != 2 * occupied_count:
        raise ValueError(
            "Canonical closed-shell electron count does not match occupied_count"
        )
    virtual_count = canonical.norb - occupied_count
    if frozen_count < 0:
        raise ValueError("frozen_core_num_orbitals must be non-negative")
    if frozen_count >= occupied_count and frozen_count != 0:
        raise ValueError(
            "frozen_core_num_orbitals must leave at least one occupied "
            "orbital in the active space"
        )
    if frozen_count > canonical.norb:
        raise ValueError("Cannot freeze more orbitals than the full space contains")
    if discarded_virtual_count >= virtual_count and discarded_virtual_count != 0:
        raise ValueError(
            "discarded_virtual_orbitals must leave at least one virtual "
            "orbital in the active space"
        )

    frozen_indices = tuple(range(frozen_count))
    discarded_virtual_indices = tuple(
        range(canonical.norb - discarded_virtual_count, canonical.norb)
    )
    active_indices = tuple(
        index
        for index in range(canonical.norb)
        if index not in frozen_indices
        and index not in discarded_virtual_indices
    )
    if frozen_indices:
        h1_intermediate, h2_intermediate, nelec, frozen_energy = apply_frozen_core(
            canonical.h1,
            canonical.h2,
            canonical.nelec,
            list(frozen_indices),
        )
    else:
        h1_intermediate = canonical.h1
        h2_intermediate = canonical.h2
        nelec = canonical.nelec
        frozen_energy = 0.0

    intermediate_indices = tuple(
        index for index in range(canonical.norb) if index not in frozen_indices
    )
    retained_positions = tuple(
        position
        for position, original_index in enumerate(intermediate_indices)
        if original_index not in discarded_virtual_indices
    )
    h1 = h1_intermediate[np.ix_(retained_positions, retained_positions)]
    retained_grid = np.ix_(
        retained_positions,
        retained_positions,
        retained_positions,
        retained_positions,
    )
    h2 = h2_intermediate[retained_grid]

    active_digest = _hamiltonian_digest(h1, h2, nelec)
    boundary_gap = None
    if 0 < frozen_count < occupied_count:
        boundary_gap = float(
            canonical.orbital_energies[frozen_count]
            - canonical.orbital_energies[frozen_count - 1]
        )
        if boundary_gap <= spec.minimum_boundary_gap_ha:
            raise ValueError(
                "Frozen/retained occupied boundary gap "
                f"{boundary_gap:.12g} Ha does not exceed required minimum "
                f"{spec.minimum_boundary_gap_ha:.12g} Ha"
            )
    virtual_boundary_gap = None
    if 0 < discarded_virtual_count < virtual_count:
        first_discarded = canonical.norb - discarded_virtual_count
        virtual_boundary_gap = float(
            canonical.orbital_energies[first_discarded]
            - canonical.orbital_energies[first_discarded - 1]
        )
        if virtual_boundary_gap <= spec.minimum_boundary_gap_ha:
            raise ValueError(
                "Retained/discarded virtual boundary gap "
                f"{virtual_boundary_gap:.12g} Ha does not exceed required "
                f"minimum {spec.minimum_boundary_gap_ha:.12g} Ha"
            )

    return ActiveSpaceHamiltonian(
        h1=h1,
        h2=h2,
        norb_full=canonical.norb,
        nelec_full=canonical.nelec,
        norb_active=int(h1.shape[0]),
        nelec_active=int(nelec),
        selection=spec.selection,
        frozen_indices=frozen_indices,
        discarded_virtual_indices=discarded_virtual_indices,
        active_indices=active_indices,
        frozen_energy=float(frozen_energy),
        frozen_boundary_gap_ha=boundary_gap,
        discarded_virtual_boundary_gap_ha=virtual_boundary_gap,
        minimum_boundary_gap_ha=float(spec.minimum_boundary_gap_ha),
        orbital_energies=canonical.orbital_energies,
        occupied_projector_leakage=canonical.occupied_projector_leakage,
        full_hamiltonian_digest=canonical.digest,
        active_hamiltonian_digest=active_digest,
        total_hamiltonian_digest=_active_space_total_digest(
            full_digest=canonical.digest,
            active_digest=active_digest,
            frozen_indices=frozen_indices,
            discarded_virtual_indices=discarded_virtual_indices,
            frozen_energy=float(frozen_energy),
            selection=spec.selection,
            minimum_boundary_gap_ha=float(spec.minimum_boundary_gap_ha),
        ),
    )


def build_vqe_active_space_hamiltonian(
    canonical: CanonicalFragmentHamiltonian,
    vqe_args: VQE_ArgsUser,
) -> ActiveSpaceHamiltonian:
    """Return the explicit manual active space accepted by the VQE path."""

    if vqe_args.frozen_core != "manual":
        raise ValueError(
            "Matched VQE active-space construction requires frozen_core='manual'"
        )
    return build_active_space_hamiltonian(
        canonical,
        resolve_active_space_spec(vqe_args),
    )


def _map_active_space_hamiltonian(active: ActiveSpaceHamiltonian) -> SparsePauliOp:
    """Map active spatial integrals to the blocked-spin JW Hamiltonian."""

    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")
    electronic_integrals = ElectronicIntegrals.from_raw_integrals(
        h1_a=active.h1,
        h2_aa=active.h2,
        h1_b=active.h1,
        h2_bb=active.h2,
        h2_ba=np.einsum("pqrs->qprs", active.h2),
    )
    qubit_op = JordanWignerMapper().map(
        ElectronicEnergy(electronic_integrals).second_q_op()
    )
    if hasattr(qubit_op, "simplify"):
        qubit_op = qubit_op.simplify(atol=1e-12)
    if hasattr(qubit_op, "coeffs"):
        coefficients = np.real_if_close(qubit_op.coeffs, tol=1e-9)
        residual = (
            float(np.max(np.abs(np.imag(coefficients))))
            if coefficients.size
            else 0.0
        )
        if residual > 0.0:
            warn(
                "Projected qubit Hamiltonian coefficients to reals; "
                f"residual imaginary magnitude={residual:.2e}"
            )
        qubit_op = SparsePauliOp(qubit_op.paulis, np.real(coefficients))
    return qubit_op


def parse_fcidump_hamiltonian(filepath: Path) -> tuple[SparsePauliOp, int, int, float]:
    """
    Parse FCIDUMP-format Hamiltonian file.

    Parameters
    ----------
    filepath : Path
        Path to FCIDUMP file

    Returns
    -------
    hamiltonian : SparsePauliOp
        Qubit Hamiltonian in Pauli operator form
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons
    core_energy : float
        Core/nuclear energy shift (not included in the qubit Hamiltonian)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    fc_data = fcidump.read(str(filepath))

    norb = int(fc_data["NORB"])
    nelec = int(fc_data["NELEC"])
    core_energy = float(fc_data["ECORE"])

    h1 = np.asarray(fc_data["H1"], dtype=float)
    # Restore the full two-electron tensor in chemists' notation <pq|rs>
    h2 = ao2mo.restore(1, fc_data["H2"], norb)

    # Build fermionic Hamiltonian using Qiskit Nature helpers (handles spin expansion)
    electronic_integrals = ElectronicIntegrals.from_raw_integrals(
        h1_a=h1,
        h2_aa=h2,
        h1_b=h1,
        h2_bb=h2,
        h2_ba=np.einsum("pqrs->qprs", h2),
    )
    electronic_energy = ElectronicEnergy(electronic_integrals)
    fermionic_op = electronic_energy.second_q_op()

    # Map to qubits using Jordan-Wigner
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    # Remove negligible imaginary parts introduced by numerical noise
    if hasattr(qubit_op, "simplify"):
        qubit_op = qubit_op.simplify(atol=1e-12)

    # Ensure Hermiticity by projecting coefficients onto the reals
    if hasattr(qubit_op, "coeffs"):
        coerced_coeffs = np.real_if_close(qubit_op.coeffs, tol=1e-9)
        max_imag = (
            float(np.max(np.abs(np.imag(coerced_coeffs))))
            if coerced_coeffs.size
            else 0.0
        )
        if max_imag > 0.0:
            warn(
                "Projected qubit Hamiltonian coefficients to reals; "
                f"residual imaginary magnitude={max_imag:.2e}"
            )
        qubit_op = SparsePauliOp(qubit_op.paulis, np.real(coerced_coeffs))

    return qubit_op, norb, nelec, core_energy


def parse_fcidump_hamiltonian_with_frozen_core(
    filepath: Path,
    frozen_core: str = "none",
    frozen_core_threshold: float = 1.98,
    frozen_core_num_orbitals: int = 0,
    verbose: int = 0,
) -> tuple[SparsePauliOp, int, int, float, list[int], int]:
    """
    Parse FCIDUMP-format Hamiltonian file with optional frozen core approximation.

    Parameters
    ----------
    filepath : Path
        Path to FCIDUMP file
    frozen_core : str
        Frozen core mode: 'none', 'auto', or 'manual'
    frozen_core_threshold : float
        Occupation threshold for auto mode (default 1.98)
    frozen_core_num_orbitals : int
        Number of orbitals to freeze in manual mode
    verbose : int
        Verbosity level

    Returns
    -------
    hamiltonian : SparsePauliOp
        Qubit Hamiltonian in Pauli operator form (active space only if frozen)
    norb_active : int
        Number of active spatial orbitals
    nelec_active : int
        Number of active electrons
    total_core_energy : float
        Core energy including frozen orbital contribution
    frozen_indices : list[int]
        Indices of frozen orbitals (in original basis)
    norb_full : int
        Total number of orbitals including frozen
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    # Read FCIDUMP file
    fc_data = fcidump.read(str(filepath))

    norb_full = int(fc_data["NORB"])
    nelec = int(fc_data["NELEC"])
    core_energy = float(fc_data["ECORE"])

    h1 = np.asarray(fc_data["H1"], dtype=float)
    h2 = ao2mo.restore(1, fc_data["H2"], norb_full)

    # Determine frozen orbitals
    frozen_indices: list[int] = []

    if frozen_core == "none":
        pass  # No frozen orbitals
    elif frozen_core == "auto":
        # Auto-detect based on diagonal occupation proxy (h1 eigenvalues)
        # Lower energy orbitals are more likely to be core
        # We freeze orbitals whose HF occupation would be ~2.0
        # For canonical basis, diagonal h1 elements are orbital energies
        h1_diag = np.diag(h1)
        sorted_indices = np.argsort(h1_diag)  # Sort by energy (lowest first)

        # Heuristic: freeze orbitals that are significantly lower in energy
        # than the HOMO-LUMO region
        n_occ = nelec // 2
        if n_occ > 1:
            # Check energy gap between orbitals
            # Freeze if orbital energy is much lower than HOMO
            homo_energy = h1_diag[sorted_indices[n_occ - 1]]
            for i in range(n_occ):
                orbital_idx = sorted_indices[i]
                orbital_energy = h1_diag[orbital_idx]
                # Freeze if energy gap to HOMO is > 10 Ha (core-valence gap)
                if homo_energy - orbital_energy > 10.0:
                    frozen_indices.append(orbital_idx)
                else:
                    break  # Stop at first valence orbital

        if verbose >= 1 and frozen_indices:
            print(f"  Auto-detected frozen orbitals: {frozen_indices}")
    elif frozen_core == "manual":
        # Freeze the first N orbitals (lowest energy in canonical basis)
        if frozen_core_num_orbitals > 0:
            h1_diag = np.diag(h1)
            sorted_indices = np.argsort(h1_diag)
            frozen_indices = list(sorted_indices[:frozen_core_num_orbitals])

        if verbose >= 1 and frozen_indices:
            print(f"  Manually frozen orbitals: {frozen_indices}")
    else:
        raise ValueError(f"Invalid frozen_core mode: {frozen_core}")

    # Apply frozen core transformation if we have frozen orbitals
    if frozen_indices:
        h1_active, h2_active, nelec_active, frozen_energy = apply_frozen_core(
            h1, h2, nelec, frozen_indices, verbose
        )
        total_core_energy = core_energy + frozen_energy
        norb_active = h1_active.shape[0]

        if verbose >= 1:
            print(f"  Frozen core: {len(frozen_indices)} orbitals frozen")
            print(f"  Active space: {norb_active} orbitals, {nelec_active} electrons")
            print(f"  Qubits reduced: {2*norb_full} -> {2*norb_active}")
    else:
        h1_active = h1
        h2_active = h2
        nelec_active = nelec
        norb_active = norb_full
        total_core_energy = core_energy

    # Build fermionic Hamiltonian for active space
    electronic_integrals = ElectronicIntegrals.from_raw_integrals(
        h1_a=h1_active,
        h2_aa=h2_active,
        h1_b=h1_active,
        h2_bb=h2_active,
        h2_ba=np.einsum("pqrs->qprs", h2_active),
    )
    electronic_energy = ElectronicEnergy(electronic_integrals)
    fermionic_op = electronic_energy.second_q_op()

    # Map to qubits using Jordan-Wigner
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    # Clean up numerical noise
    if hasattr(qubit_op, "simplify"):
        qubit_op = qubit_op.simplify(atol=1e-12)

    if hasattr(qubit_op, "coeffs"):
        coerced_coeffs = np.real_if_close(qubit_op.coeffs, tol=1e-9)
        qubit_op = SparsePauliOp(qubit_op.paulis, np.real(coerced_coeffs))

    return qubit_op, norb_active, nelec_active, total_core_energy, frozen_indices, norb_full


def apply_frozen_core(
    h1e: ndarray,
    h2e: ndarray,
    nelec: int,
    frozen_indices: list[int],
    verbose: int = 0,
) -> tuple[ndarray, ndarray, int, float]:
    """
    Apply frozen core approximation to reduce the active space.

    Frozen orbitals are assumed to be doubly occupied. Their contribution
    to the energy is computed and added to the core energy, and the
    Hamiltonian is reduced to the active space only.

    Parameters
    ----------
    h1e : ndarray
        One-electron integrals (norb, norb)
    h2e : ndarray
        Two-electron integrals in chemist notation (norb, norb, norb, norb)
    nelec : int
        Total number of electrons
    frozen_indices : list[int]
        Indices of orbitals to freeze (must be occupied)
    verbose : int
        Verbosity level

    Returns
    -------
    h1e_active : ndarray
        One-electron integrals for active space
    h2e_active : ndarray
        Two-electron integrals for active space
    nelec_active : int
        Number of electrons in active space
    frozen_energy : float
        Energy contribution from frozen orbitals
    """
    h1e = np.asarray(h1e)
    h2e = np.asarray(h2e)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError(f"h1e must be square; got shape {h1e.shape}")
    norb = h1e.shape[0]
    if h2e.shape != (norb, norb, norb, norb):
        raise ValueError(
            "h2e must have shape (norb, norb, norb, norb); "
            f"got {h2e.shape} for norb={norb}"
        )
    if type(nelec) is not int or nelec < 0 or nelec > 2 * norb or nelec % 2:
        raise ValueError("nelec must be an even integer in [0, 2*norb]")
    if any(type(index) is not int for index in frozen_indices):
        raise TypeError("Frozen orbital indices must be integers")
    if len(set(frozen_indices)) != len(frozen_indices):
        raise ValueError("Frozen orbital indices must be unique")
    n_frozen = len(frozen_indices)
    if 2 * n_frozen > nelec:
        raise ValueError("Cannot freeze more doubly occupied orbitals than electrons")

    if n_frozen == 0:
        return h1e, h2e, nelec, 0.0

    # Validate frozen indices
    frozen_indices = sorted(frozen_indices)
    if any(i < 0 or i >= norb for i in frozen_indices):
        raise ValueError(f"Frozen indices {frozen_indices} out of range [0, {norb})")

    # Active orbital indices (all orbitals not frozen)
    active_indices = [i for i in range(norb) if i not in frozen_indices]
    n_active = len(active_indices)

    if verbose >= 2:
        print(f"\n  Frozen core approximation:")
        print(f"    Frozen orbitals: {frozen_indices} ({n_frozen} orbitals, {2*n_frozen} electrons)")
        print(f"    Active orbitals: {active_indices} ({n_active} orbitals)")

    # Compute frozen core energy
    # E_frozen = sum_i [2*h1[i,i] + sum_j (2*J[i,j] - K[i,j])]
    # where i,j are frozen orbitals
    frozen_energy = 0.0

    # One-electron contribution from frozen orbitals (factor of 2 for spin)
    for i in frozen_indices:
        frozen_energy += 2.0 * h1e[i, i]

    # Two-electron contribution from frozen-frozen interactions
    for i in frozen_indices:
        for j in frozen_indices:
            # Coulomb: 2 * (ii|jj) for closed shell
            # Exchange: -(ij|ji)
            J_ij = h2e[i, i, j, j]  # Coulomb integral
            K_ij = h2e[i, j, j, i]  # Exchange integral
            frozen_energy += 2.0 * J_ij - K_ij

    # Effective one-electron operator for active space
    # h1_eff[p,q] = h1[p,q] + sum_i [2*(pq|ii) - (pi|iq)]
    # where i are frozen orbitals, p,q are active orbitals
    result_dtype = np.result_type(h1e.dtype, h2e.dtype)
    h1e_active = np.zeros((n_active, n_active), dtype=result_dtype)

    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            h1e_active[p_idx, q_idx] = h1e[p, q]

            # Add frozen core contribution to effective one-electron operator
            for i in frozen_indices:
                # Coulomb: 2*(pq|ii)
                J_pq_i = h2e[p, q, i, i]
                # Exchange: -(pi|iq)
                K_pq_i = h2e[p, i, i, q]
                h1e_active[p_idx, q_idx] += 2.0 * J_pq_i - K_pq_i

    # Extract active-space two-electron integrals
    h2e_active = np.zeros(
        (n_active, n_active, n_active, n_active),
        dtype=result_dtype,
    )
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            for r_idx, r in enumerate(active_indices):
                for s_idx, s in enumerate(active_indices):
                    h2e_active[p_idx, q_idx, r_idx, s_idx] = h2e[p, q, r, s]

    # Active electrons = total - frozen (2 electrons per frozen orbital)
    nelec_active = nelec - 2 * n_frozen

    if verbose >= 2:
        print(f"    Active electrons: {nelec_active}")
        print(f"    Frozen core energy: {frozen_energy:.8f} Ha")

    frozen_energy = np.real_if_close(frozen_energy, tol=1.0e8)
    if np.iscomplexobj(frozen_energy):
        raise ValueError("Frozen-core energy has a non-negligible imaginary part")
    return h1e_active, h2e_active, nelec_active, float(frozen_energy)


def validate_rdm_pair(
    rdm1: ndarray,
    rdm2: ndarray,
    *,
    label: str,
    atol: float = 1.0e-6,
) -> int:
    """Validate shapes, particle trace, and PySCF chemist contraction."""

    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError(f"{label} rdm1 must be square; got {rdm1.shape}")
    norb = rdm1.shape[0]
    if rdm2.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"{label} rdm2 has shape {rdm2.shape}; expected {(norb,) * 4}"
        )
    trace_value = np.real_if_close(np.trace(rdm1), tol=1.0e8)
    if np.iscomplexobj(trace_value):
        raise ValueError(f"{label} rdm1 trace is not real: {trace_value}")
    electron_count = int(round(float(trace_value)))
    if electron_count < 0 or abs(float(trace_value) - electron_count) > atol:
        raise ValueError(
            f"{label} rdm1 trace {float(trace_value):.12g} is not an integer "
            "particle count"
        )
    if electron_count > 1:
        contracted = np.einsum("pqrr->pq", rdm2) / (electron_count - 1)
        error = float(np.max(np.abs(contracted - rdm1)))
        if error > atol:
            raise ValueError(
                f"{label} rdm2 contraction error {error:.3e} exceeds {atol:.1e}"
            )
    return electron_count


def expand_rdm_from_active_space(
    rdm1_active: ndarray,
    rdm2_active: ndarray,
    frozen_indices: list[int],
    norb_full: int,
    discarded_virtual_indices: list[int] | None = None,
) -> tuple[ndarray, ndarray]:
    """
    Expand RDMs from active space back to full orbital space.

    Frozen orbitals are set to occupation 2.0 (doubly occupied).  Their
    disconnected Coulomb and exchange terms are restored using the PySCF
    spin-summed chemist convention

    ``D2[p,q,r,s] = D1[p,q] D1[r,s] - 1/2 D1[p,s] D1[r,q]``

    whenever at least one factor belongs to the frozen determinant.

    Parameters
    ----------
    rdm1_active : ndarray
        1-RDM in active space (n_active, n_active)
    rdm2_active : ndarray
        2-RDM in active space (n_active, n_active, n_active, n_active)
    frozen_indices : list[int]
        Indices of frozen orbitals in full space
    norb_full : int
        Total number of orbitals including frozen
    discarded_virtual_indices : list[int] or None
        Canonical virtual orbitals omitted from the active Hamiltonian.  Their
        restored 1- and 2-RDM rows/columns remain exactly zero.

    Returns
    -------
    rdm1_full : ndarray
        1-RDM in full orbital space
    rdm2_full : ndarray
        2-RDM in full orbital space
    """
    rdm1_active = np.asarray(rdm1_active)
    rdm2_active = np.asarray(rdm2_active)
    if type(norb_full) is not int or norb_full < 0:
        raise ValueError("norb_full must be a non-negative integer")
    if any(type(index) is not int for index in frozen_indices):
        raise TypeError("Frozen orbital indices must be integers")
    discarded_virtual_indices = (
        []
        if discarded_virtual_indices is None
        else list(discarded_virtual_indices)
    )
    if any(type(index) is not int for index in discarded_virtual_indices):
        raise TypeError("Discarded virtual orbital indices must be integers")
    if len(set(frozen_indices)) != len(frozen_indices):
        raise ValueError("Frozen orbital indices must be unique")
    if len(set(discarded_virtual_indices)) != len(discarded_virtual_indices):
        raise ValueError("Discarded virtual orbital indices must be unique")
    if set(frozen_indices) & set(discarded_virtual_indices):
        raise ValueError("Frozen and discarded orbital indices must be disjoint")
    if any(index < 0 or index >= norb_full for index in frozen_indices):
        raise ValueError(
            f"Frozen indices {frozen_indices} out of range [0, {norb_full})"
        )
    if any(
        index < 0 or index >= norb_full
        for index in discarded_virtual_indices
    ):
        raise ValueError(
            "Discarded virtual indices out of range "
            f"[0, {norb_full}): {discarded_virtual_indices}"
        )
    n_frozen = len(frozen_indices)
    expected_active = (
        norb_full - n_frozen - len(discarded_virtual_indices)
    )
    if rdm1_active.shape != (expected_active, expected_active):
        raise ValueError(
            f"Active rdm1 has shape {rdm1_active.shape}; expected "
            f"{(expected_active, expected_active)}"
        )
    validate_rdm_pair(rdm1_active, rdm2_active, label="active-space")
    if n_frozen == 0 and not discarded_virtual_indices:
        return rdm1_active, rdm2_active

    frozen_indices = sorted(frozen_indices)
    discarded_virtual_indices = sorted(discarded_virtual_indices)
    active_indices = [
        index
        for index in range(norb_full)
        if index not in frozen_indices
        and index not in discarded_virtual_indices
    ]
    # Initialize full RDMs
    result_dtype = np.result_type(rdm1_active.dtype, rdm2_active.dtype)
    rdm1_full = np.zeros((norb_full, norb_full), dtype=result_dtype)
    rdm2_full = np.zeros(
        (norb_full, norb_full, norb_full, norb_full),
        dtype=result_dtype,
    )

    # Fill frozen orbital contributions to 1-RDM (occupation = 2)
    for i in frozen_indices:
        rdm1_full[i, i] = 2.0

    # Fill active space 1-RDM
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            rdm1_full[p, q] = rdm1_active[p_idx, q_idx]

    # Fill frozen-frozen 2-RDM contributions.  Use += because the Coulomb and
    # exchange slots coincide when i == j, giving 4 - 2 = 2 as required for
    # a doubly occupied spatial orbital.
    for i in frozen_indices:
        for j in frozen_indices:
            rdm2_full[i, i, j, j] += 4.0
            rdm2_full[i, j, j, i] -= 2.0

    # Fill frozen-active cross terms in 2-RDM
    # rdm2[i,i,p,q] = 2 * rdm1[p,q] (frozen i interacts with active p,q)
    # rdm2[i,p,q,i] = -rdm1[q,p] and
    # rdm2[p,i,i,q] = -rdm1[p,q] (exchange)
    for i in frozen_indices:
        for p_idx, p in enumerate(active_indices):
            for q_idx, q in enumerate(active_indices):
                rdm2_full[i, i, p, q] += 2.0 * rdm1_active[p_idx, q_idx]
                rdm2_full[p, q, i, i] += 2.0 * rdm1_active[p_idx, q_idx]
                rdm2_full[i, p, q, i] -= rdm1_active[q_idx, p_idx]
                rdm2_full[p, i, i, q] -= rdm1_active[p_idx, q_idx]

    # Fill active-active 2-RDM
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            for r_idx, r in enumerate(active_indices):
                for s_idx, s in enumerate(active_indices):
                    rdm2_full[p, q, r, s] = rdm2_active[p_idx, q_idx, r_idx, s_idx]

    validate_rdm_pair(rdm1_full, rdm2_full, label="expanded full-space")
    return rdm1_full, rdm2_full


def build_uccsd_ansatz(norb: int, nelec: int) -> QuantumCircuit:
    """
    Build UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons

    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized UCCSD circuit
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

    mapper = JordanWignerMapper()

    # For BE fragments: spin configuration depends on electron count
    # Odd nelec: use balanced (gives doublet S=1/2)
    # Even nelec: use balanced (gives singlet S=0)
    # Note: balanced means n_alpha = (nelec+1)//2, n_beta = nelec - n_alpha
    n_alpha = (nelec + 1) // 2
    n_beta = nelec - n_alpha
    num_particles = (n_alpha, n_beta)

    # Initial state: Hartree-Fock
    # Qiskit uses BLOCKED spin-orbital ordering (matching FCI/PySCF):
    #   qubits 0..norb-1 = alpha spin-orbitals
    #   qubits norb..2*norb-1 = beta spin-orbitals
    # HartreeFock occupies the lowest n_alpha alpha and n_beta beta orbitals
    hf_state = HartreeFock(norb, num_particles, mapper)

    # UCCSD ansatz
    ansatz = UCCSD(
        num_spatial_orbitals=norb,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_state,
    )

    return ansatz


# def compute_rdm1_from_statevector(
#     statevector: Statevector,
#     norb: int,
#     nelec: int,  # noqa: ARG001 - kept for signature parity with full RDM helper
# ) -> ndarray:
#     """
#     Compute 1-RDM from a VQE statevector.

#     Parameters
#     ----------
#     statevector : Statevector
#         Optimized or intermediate VQE statevector.
#     norb : int
#         Number of spatial orbitals.
#     nelec : int
#         Number of electrons (unused but retained for future extensions).

#     Returns
#     -------
#     ndarray
#         One-particle reduced density matrix (real-valued).
#     """

#     # TODO: print out statevector to see if correct or not [DONE]
#     print(f"\n{'='*80}")
#     print(f"DEBUG compute_rdm1_from_statevector - Statevector Info")
#     print(f"{'='*80}")
#     print(f"Statevector dimension: {len(statevector.data)}")
#     print(f"Number of qubits: {int(np.log2(len(statevector.data)))}")
#     print(f"Statevector norm: {np.linalg.norm(statevector.data):.10f}")
#     # Print largest amplitude components
#     amplitudes = np.abs(statevector.data)**2
#     sorted_indices = np.argsort(amplitudes)[::-1]
#     print(f"Top 5 basis states by probability:")
#     for i in range(min(5, len(sorted_indices))):
#         idx = sorted_indices[i]
#         prob = amplitudes[idx]
#         if prob > 1e-10:
#             binary = format(idx, f'0{int(np.log2(len(statevector.data)))}b')
#             print(f"  |{binary}> : prob={prob:.6f}, amplitude={statevector.data[idx]:.6f}")
#     print(f"{'='*80}\n")

#     if not QISKIT_AVAILABLE:
#         raise ImportError("Qiskit is required for VQE solver")

#     nqubits = 2 * norb
#     mapper = JordanWignerMapper()
#     rdm1 = np.zeros((norb, norb), dtype=complex)

#     for p in range(norb):
#         for q in range(norb):
#             value = 0.0 + 0.0j

#             # TODO: verify similarity with FCI spin [DONE]
#             # FCI uses same spin summation: rdm1[p,q] = sum over spin of <a+_{p,spin} a_{q,spin}>
#             for spin in (0, 1):  # 0 -> alpha, 1 -> beta

#                 # TODO: print out p,q,spin [DONE]
#                 op_str = f"+_{2 * p + spin} -_{2 * q + spin}"
#                 fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)
#                 pauli_op = mapper.map(fermionic_op)

#                 # TODO: print out value added here [DONE]
#                 # TODO: print state vector of pauli_op [DONE]
#                 exp_val = statevector.expectation_value(pauli_op)
#                 value += exp_val

#                 # Debug: print detailed info for diagonal elements (occupations)
#                 if p == q:
#                     print(f"  RDM1[{p},{q}] spin={spin}: op={op_str}, exp_val={exp_val:.6f}")

#             # Print final value for diagonal (occupation) elements
#             if p == q:
#                 print(f"  RDM1[{p},{q}] TOTAL = {value.real:.6f}")
#             rdm1[p, q] = value

#     return rdm1.real


def compute_rdms_from_statevector(
    statevector: Statevector, norb: int, nelec: int
) -> tuple[ndarray, ndarray]:
    """
    Compute 1-RDM and 2-RDM from VQE statevector.

    Uses Jordan-Wigner mapping to compute fermionic RDMs.

    Parameters
    ----------
    statevector : Statevector
        Optimized VQE statevector
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons

    Returns
    -------
    rdm1 : ndarray (norb, norb)
        One-particle reduced density matrix
    rdm2 : ndarray (norb, norb, norb, norb)
        Two-particle reduced density matrix
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    # TODO: print out statevector info for FCI comparison
    # finished adding
    print(f"\n{'=' * 80}")
    print(f"DEBUG compute_rdms_from_statevector - Statevector Info")
    print(f"{'=' * 80}")
    print(f"Statevector dimension: {len(statevector.data)}")
    nqubits_sv = int(np.log2(len(statevector.data)))
    print(f"Number of qubits: {nqubits_sv}")
    print(f"Statevector norm: {np.linalg.norm(statevector.data):.10f}")

    # Print largest amplitude components (basis states)
    amplitudes = np.abs(statevector.data) ** 2
    sorted_indices = np.argsort(amplitudes)[::-1]
    print(f"\nTop 10 basis states by probability:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = amplitudes[idx]
        if prob > 1e-10:
            binary = format(idx, f"0{nqubits_sv}b")
            # Interpret binary string as qubit occupations
            # In Jordan-Wigner: qubit i=1 means orbital i occupied
            print(
                f"  |{binary}> : prob={prob:.6f}, amplitude={statevector.data[idx]:.6f}"
            )

    # Interpret the dominant configuration
    dominant_idx = sorted_indices[0]
    dominant_binary = format(dominant_idx, f"0{nqubits_sv}b")
    print(f"\nDominant configuration analysis:")
    print(f"  Binary: |{dominant_binary}>")
    print(f"  Qubit occupations (0=empty, 1=occupied):")
    # BLOCKED ordering: qubits 0..norb-1 = alpha, qubits norb..2*norb-1 = beta
    norb_sv = nqubits_sv // 2
    for q in range(nqubits_sv):
        occ = dominant_binary[nqubits_sv - 1 - q]  # Reverse for qubit ordering
        spin = "alpha" if q < norb_sv else "beta"
        spatial_orb = q if q < norb_sv else q - norb_sv
        print(f"    Qubit {q} (spatial orb {spatial_orb}, {spin}): {occ}")
    print(f"{'=' * 80}\n")

    nqubits = 2 * norb

    # Initialize RDMs
    rdm1 = np.zeros((norb, norb), dtype=complex)
    rdm2 = np.zeros((norb, norb, norb, norb), dtype=complex)
    mapper = JordanWignerMapper()
    spin_labels = (0, 1)
    spin_configs = (
        (0, 0, 0, 0),  # αα
        (0, 1, 1, 0),  # αβ
        (1, 0, 0, 1),  # βα
        (1, 1, 1, 1),  # ββ
    )

    # 1-RDM: <a+_p a_q>
    # Debug: print diagonal elements (occupations) with per-spin breakdown
    # BLOCKED spin-orbital ordering (matching Qiskit's Hamiltonian and FCI):
    #   alpha spin-orbital index = p (for spatial orbital p)
    #   beta spin-orbital index = p + norb (for spatial orbital p)
    print(f"\n--- 1-RDM diagonal elements (occupations) ---")
    print(
        f"Using BLOCKED spin-orbital ordering: alpha=0..{norb - 1}, beta={norb}..{2 * norb - 1}"
    )
    print(
        f"FCI uses same spin summation: rdm1[p,q] = sum over spin of <a+_{{p,spin}} a_{{q,spin}}>"
    )
    for p in range(norb):
        for q in range(norb):
            value = 0.0 + 0.0j
            for spin in spin_labels:
                # BLOCKED ordering: spin_orbital = p + spin * norb
                spin_orb_p = p + spin * norb
                spin_orb_q = q + spin * norb
                op_str = f"+_{spin_orb_p} -_{spin_orb_q}"
                fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)
                pauli_op = mapper.map(fermionic_op)
                exp_val = statevector.expectation_value(pauli_op)
                value += exp_val

                # Debug: print detailed info for diagonal elements (occupations)
                if p == q:
                    spin_name = "alpha" if spin == 0 else "beta"
                    print(
                        f"  RDM1[{p},{q}] spin={spin_name}: op={op_str}, exp_val={exp_val:.6f}"
                    )

            # Print final value for diagonal (occupation) elements
            if p == q:
                print(f"  RDM1[{p},{q}] TOTAL = {value.real:.6f}")
            rdm1[p, q] = value
    print(f"--- End 1-RDM diagonal debug ---\n")

    print(f"--- 1-RDM full matrix (spin-summed) ---")
    print(np.array2string(rdm1.real, precision=6, suppress_small=True))
    print(f"--- End 1-RDM full matrix ---\n")

    # NOTE: Using BLOCKED spin-orbital indexing to match Qiskit's Hamiltonian and FCI
    # (qubits 0..norb-1 = alpha, norb..2*norb-1 = beta)

    # 2-RDM in CHEMIST CONVENTION
    # Target: rdm2[a,b,c,d] = <a†_a a†_c a_d a_b>  (same as PySCF FCI)
    #
    # The loop computes:  val = <a†_p a†_q a_s a_r>   for each (p,q,r,s)
    # But stores it at:   rdm2[p, r, q, s] = val     (q and r swapped in index)
    #
    # Proof: reading rdm2[a,b,c,d] gets the value from the iteration where
    #   p=a, r=b, q=c, s=d  →  val = <a†_a a†_c a_d a_b>  ✓ chemist convention
    #
    # BLOCKED ordering: spin_orbital = spatial_orbital + spin * norb
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    value = 0.0 + 0.0j
                    for spin_p, spin_q, spin_s, spin_r in spin_configs:
                        # BLOCKED ordering for each index
                        so_p = p + spin_p * norb
                        so_q = q + spin_q * norb
                        so_s = s + spin_s * norb
                        so_r = r + spin_r * norb
                        # Compute <a+_p a+_q a_s a_r> (original operator, works with spin_configs)
                        op_str = f"+_{so_p} +_{so_q} -_{so_s} -_{so_r}"
                        fermionic_op = FermionicOp(
                            {op_str: 1.0}, num_spin_orbitals=nqubits
                        )
                        pauli_op = mapper.map(fermionic_op)
                        value += statevector.expectation_value(pauli_op)
                    # Swap q,r in storage index → chemist convention (see proof above)
                    rdm2[p, r, q, s] = value

    # Verify RDM consistency using CHEMIST convention: sum_r rdm2[p,q,r,r] = (N-1)*rdm1[p,q]
    # This is a fundamental N-representability condition
    nelec_check = int(round(np.trace(rdm1).real))
    trace_error = abs(float(np.trace(rdm1).real) - nelec)
    if trace_error > 1e-6 or nelec_check != nelec:
        raise ValueError(
            "Statevector RDM1 particle trace mismatch: "
            f"trace={np.trace(rdm1).real:.12g}, expected={nelec}"
        )
    if nelec_check > 1:
        # CHEMIST contraction: pqrr->pq (sum over last two indices when equal)
        rdm1_from_rdm2 = np.einsum("pqrr->pq", rdm2) / (nelec_check - 1)
        contraction_error = np.max(np.abs(rdm1 - rdm1_from_rdm2))
        print(
            f"\n--- RDM consistency check (CHEMIST convention, BEFORE basis transform) ---"
        )
        print(f"Statevector norm: {np.linalg.norm(statevector.data):.10f}")
        print(f"RDM1 trace: {np.trace(rdm1).real:.6f} (expected: {nelec_check})")
        print(
            f"Contraction check (chemist): max |rdm1 - sum_r rdm2[p,q,r,r]/(N-1)| = {contraction_error:.2e}"
        )
        if contraction_error > 1e-6:
            raise ValueError(
                "Statevector RDM2 violates the chemist contraction relation: "
                f"max error={contraction_error:.3e}"
            )
        else:
            print("  OK: RDM2 satisfies chemist contraction relation")
        print(f"--- End RDM consistency check ---\n")

    # r = len(rdm1) // 2
    # l = len(rdm1) // 2 - 1

    # temp = deepcopy(rdm2[r, :, :, :])
    # rdm2[r, :, :, :] = rdm2[l, :, :, :]
    # rdm2[l, :, :, :] = temp

    # temp = deepcopy(rdm2[:, r, :, :])
    # rdm2[:, r, :, :] = rdm2[:, l, :, :]
    # rdm2[:, l, :, :] = temp

    # temp = deepcopy(rdm2[:, :, r, :])
    # rdm2[:, :, r, :] = rdm2[:, :, l, :]
    # rdm2[:, :, l, :] = temp

    # temp = deepcopy(rdm2[:, :, :, r])
    # rdm2[:, :, :, r] = rdm2[:, :, :, l]
    # rdm2[:, :, :, l] = temp

    # n = len(rdm1)
    # for i in range(0, r):
    #     ii = n - i - 1

    #     temp = deepcopy(rdm2[i, l : r + 1, :, :])
    #     rdm2[i, l : r + 1, :, :] = rdm2[ii, l : r + 1, :, :]
    #     rdm2[ii, l : r + 1, :, :] = temp

    #     temp = deepcopy(rdm2[l : r + 1, i, :, :])
    #     rdm2[l : r + 1, i, :, :] = rdm2[l : r + 1, ii, :, :]
    #     rdm2[l : r + 1, ii, :, :] = temp

    #     temp = deepcopy(rdm2[:, :, i, l : r + 1])
    # NOTE: No permutation needed for RDM2 either - we use correct BLOCKED spin-orbital indexing
    # The old [0,2,1,3] permutation was a band-aid for INTERLEAVED vs BLOCKED mismatch
    # NOTE: Contraction sanity check already done above (lines 605-634) - no need to duplicate

    # Convert to real (imaginary parts should be negligible)
    rdm1 = rdm1.real
    rdm2 = rdm2.real

    print(f"--- 2-RDM (spin-summed) as (pq,rs) supermatrix ---")
    rdm2_mat = rdm2.reshape(norb * norb, norb * norb)
    print(f"rdm2 shape: {rdm2.shape} (supermatrix shape: {rdm2_mat.shape})")
    if norb <= 6:
        print(np.array2string(rdm2_mat, precision=6, suppress_small=True))
    else:
        nprint = min(10, norb * norb)
        print(
            np.array2string(
                rdm2_mat[:nprint, :nprint],
                precision=6,
                suppress_small=True,
            )
        )
    print(f"--- End 2-RDM print ---\n")

    return rdm1, rdm2


def regenerate_fcidump_with_heff(
    frag: Frags,
    output_dir: str | Path,
    vqe_args: VQE_ArgsUser | None = None,
) -> Path:
    """
    Regenerate FCIDUMP file with current effective Hamiltonian in canonical MO basis.

    CRITICAL: VQE requires orthonormal orbitals for Jordan-Wigner mapping.
    The embedding AO basis is NOT orthonormal (AO overlap matrix has off-diagonal
    elements), so we MUST transform to fragment MO basis before creating FCIDUMP.

    By default, orbitals are canonicalized separately inside the SCF occupied
    and virtual subspaces, preserving the occupied projector. The explicit
    ``full`` mode diagonalizes the complete one-electron Hamiltonian to
    reproduce the published full-space H4/F2 calculations.

    This function:
    1. Loads current effective Hamiltonian from fragment (in AO basis)
    2. Transforms h1e and h2e from embedding AO basis to fragment MO basis
    3. Canonicalizes h1e_mo using the requested orbital-basis mode
    4. Rotates h2e to match canonical orbital ordering
    5. Writes FCIDUMP with canonically ordered orbitals for VQE

    Parameters
    ----------
    frag : Frags
        Fragment object with _effective_h1e stored from recent SCF
    output_dir : str or Path
        Directory to write updated FCIDUMP file
    vqe_args : VQE_ArgsUser or None, optional
        VQE configuration arguments, including the canonicalization mode. If
        ``debug_config_analysis`` is true, performs exhaustive HF configuration
        analysis. Default: None, which uses occupied/virtual block mode.

    Returns
    -------
    Path
        Path to the regenerated FCIDUMP file

    Notes
    -----
    The Hamiltonian is extracted from frag._effective_h1e (embedding AO basis),
    which is stored in pfrag.py:285 when SCF is called.

    The transformation to canonical MO basis:
    1. h1e_mo = C^T @ h1e_ao @ C
    2. h2e_mo = C_ip C_jq C_kr C_ls h2e_ao[ijkl]
    3. Diagonalize either the occupied/virtual blocks or the complete h1e_mo
    4. Sort the resulting eigenvalues in the selected mode
    5. h1e_canonical = U.T @ h1e_mo @ U
    6. h2e_canonical = U_ip U_jq U_kr U_ls h2e_mo[ijkl]
    7. Retain U for the RDM back-transformation; leave mo_coeffs unchanged

    This ensures VQE:
    1. Uses orthonormal orbitals (required for Jordan-Wigner)
    2. The default mode keeps the HF state in the SCF occupied subspace
    3. Sees chemical potential updates across BE iterations
    4. Returns RDMs in correct basis for energy assembly
    """
    orbital_canonicalization = (
        vqe_args.orbital_canonicalization
        if vqe_args is not None
        else _BLOCK_CANONICALIZATION
    )
    canonical = build_canonical_fragment_hamiltonian(
        frag,
        orbital_canonicalization=orbital_canonicalization,
    )
    frag._canonical_fragment_hamiltonian = canonical
    h1e = canonical.h1
    h2e = canonical.h2
    h1e_canonical = canonical.h1
    h2e_canonical = canonical.h2
    mo_rotation_sorted = canonical.rotation
    mo_energies_sorted = np.diag(canonical.h1)

    # CRITICAL FIX FOR BE ASSEMBLY:
    # DO NOT update frag.mo_coeffs! The TA matrix was computed with the original
    # mo_coeffs during fragment initialization. If we change mo_coeffs here, we
    # create a basis mismatch during RDM assembly:
    #   rdm_AO = TA @ Pc @ (mo_coeffs @ rdm_MO @ mo_coeffs.T) @ TA.T
    # where TA expects original mo_coeffs but gets canonical mo_coeffs.
    #
    # Instead, we:
    # 1. Keep frag.mo_coeffs = C (original, unchanged)
    # 2. Store rotation matrix U for later RDM transformation
    # 3. VQE computes RDMs in canonical basis
    # 4. After VQE, transform RDMs back: rdm_original = U @ rdm_canonical @ U.T
    # 5. BE assembly uses original mo_coeffs → correct transformation!
    #
    # This allows VQE to work in canonical basis (preventing excited states)
    # while maintaining BE assembly consistency.

    # Store rotation matrix for RDM basis transformation
    frag.canonical_rotation = mo_rotation_sorted  # U matrix

    # Get number of orbitals
    norb = canonical.norb
    nelec = canonical.nelec

    # Keep original MO coefficients (DO NOT UPDATE)
    # frag.mo_coeffs remains C (original from SCF)

    # ========== DEBUG: Configuration search for FCI vs VQE comparison ==========
    # TODO: Print all HF configuration energies for comparison with FCI [DONE]

    # ========== DEBUG: Configuration analysis (optional, off by default) ==========
    # This exhaustively checks all HF configurations to see if aufbau is optimal.
    # Controlled by vqe_args.debug_config_analysis flag.
    if vqe_args is not None and vqe_args.debug_config_analysis:
        print(f"\n{'=' * 80}")
        print(f"DEBUG vqe_solver.py - CONFIGURATION ANALYSIS for fragment {frag.dname}")
        print(f"{'=' * 80}")
        print(f"Number of orbitals: {norb}")
        print(f"Number of electrons: {nelec}")
        print(f"Canonical h1e eigenvalues (MO energies): {mo_energies_sorted}")

        # Compute HF energy for all possible orbital configurations
        from itertools import combinations

        nsocc = nelec // 2
        all_configs = list(combinations(range(norb), nsocc))

        print(f"\nTesting {len(all_configs)} possible HF configurations...")

        config_energies = []
        for config in all_configs:
            config_list = list(config)
            E_hf = 0.0

            # One-electron contribution (factor of 2 for spin)
            for i in config_list:
                E_hf += 2.0 * h1e_canonical[i, i]

            # Two-electron contribution (Coulomb and Exchange)
            for i in config_list:
                for j in config_list:
                    J_ij = h2e_canonical[i, i, j, j]  # Coulomb
                    K_ij = h2e_canonical[i, j, j, i]  # Exchange
                    E_hf += 2 * J_ij - K_ij

            config_energies.append((E_hf, config_list))

        # Sort by energy
        config_energies.sort(key=lambda x: x[0])

        aufbau_orbitals = list(range(nsocc))
        best_config = config_energies[0][1]
        best_energy = config_energies[0][0]

        print(f"\nALL configurations ranked by HF energy:")
        for idx, (E, conf) in enumerate(config_energies):
            marker = " <- SELECTED (lowest)" if conf == best_config else ""
            aufbau_marker = " (aufbau)" if conf == aufbau_orbitals else ""
            print(f"  {idx + 1}. {conf}: E_HF = {E:.8f} Ha{marker}{aufbau_marker}")

        # Print energy gaps
        if len(config_energies) > 1:
            print(f"\nEnergy gaps from lowest configuration:")
            E_lowest = config_energies[0][0]
            for idx, (E, conf) in enumerate(config_energies[1:], start=2):
                gap = E - E_lowest
                print(
                    f"  Config {idx} - Config 1: {gap:.8f} Ha ({gap * 27.2114:.4f} eV)"
                )

        # Check if aufbau matches best config
        if best_config != aufbau_orbitals:
            print(f"\n{'*' * 60}")
            print(f"WARNING: Best HF config differs from aufbau!")
            print(f"  Aufbau fills: {aufbau_orbitals}")
            print(f"  Best config:  {best_config}")
            print(f"  VQE will use aufbau initial state (orbitals 0-{nsocc - 1})")
            print(f"  This may cause VQE to converge to wrong state!")
            print(f"{'*' * 60}")
        else:
            print(f"\nAufbau matches best config - good!")

        print(f"{'=' * 80}\n")
    # ========== END: Configuration analysis ==========

    # Write to FCIDUMP file with unique name to avoid race conditions
    output_path = Path(output_dir)
    output_file = output_path / f"h10_{frag.dname}_current"

    # DEBUG: Print what we're about to write to FCIDUMP
    print(f"\n{'=' * 80}")
    print(f"DEBUG vqe_solver.py - Writing FCIDUMP in canonical MO basis")
    print(f"{'=' * 80}")
    print(f"Basis: Canonical fragment MO (orbitals sorted by energy)")
    print(f"h1e_canonical.shape: {h1e.shape}")
    print(f"MO energies (sorted): {mo_energies_sorted}")
    print(f"h1e_canonical diagonal: {np.diag(h1e)}")
    print(
        "Occupied-projector leakage: "
        f"{canonical.occupied_projector_leakage:.3e}"
    )
    print(f"h2e_canonical.shape: {h2e.shape}")
    print(f"norb (MO basis): {frag.mo_coeffs.shape[1]}")
    print(f"nelec: {2 * frag.nsocc}")
    print(f"Orbitals 0-{frag.nsocc - 1} will be occupied in HF initial state")
    print(f"")
    print(f"NOTE: frag.mo_coeffs is NOT updated (kept as original basis)")
    print(f"      VQE RDMs will be transformed back to original basis after solving")
    print(f"      This maintains BE assembly consistency with TA matrix")
    print(f"{'=' * 80}\n")

    # Write FCIDUMP in canonical MO basis (orbitals sorted by energy)
    norb_mo = frag.mo_coeffs.shape[1]
    fcidump.from_integrals(
        str(output_file),
        h1e,
        h2e,
        norb_mo,  # Number of MO orbitals
        2 * frag.nsocc,  # Number of electrons
        ms=0,  # Total spin
    )

    return output_file


def _solve_sector_adapt_active(
    active: ActiveSpaceHamiltonian,
    vqe_args: VQE_ArgsUser,
    manifest_path: str | Path | None,
) -> tuple[Any, ndarray, ndarray, dict[str, Any]]:
    """Solve one active Hamiltonian with native fixed-sector CEO-ADAPT."""

    from pyscf.fci import direct_spin1

    from quemb.molbe.sector_adapt_vqe import (
        SectorSelectorConfig,
        descriptor_from_manifest,
        fixed_sector_dimension,
        generate_compact_ovp_ceo_pool,
        sector_state_to_pyscf_ci,
        solve_sector_adapt,
    )

    if vqe_args.estimator_type != "direct_sv":
        raise ValueError(
            "adapt_sector requires estimator_type='direct_sv' as the exact "
            "deterministic estimator contract"
        )
    if _get_adapt_selector_factory() is not None:
        raise ValueError(
            "adapt_sector selection must use its native fixed-sector policy"
        )

    n_alpha = (active.nelec_active + 1) // 2
    n_beta = active.nelec_active - n_alpha
    expected_hf_bitstring = tuple(
        int(
            qubit < n_alpha
            or active.norb_active <= qubit < active.norb_active + n_beta
        )
        for qubit in range(2 * active.norb_active)
    )
    manifest = None
    if manifest_path not in {None, _GENERATED_SECTOR_POOL}:
        from quemb.molbe.ceo_manifest import load_ceo_manifest_for_system

        manifest = load_ceo_manifest_for_system(
            manifest_path,
            n_qubits=2 * active.norb_active,
            n_electrons=active.nelec_active,
        )
        _validate_private_ceo_manifest_downfolding(manifest, vqe_args)
        if tuple(int(bit) for bit in manifest.hf_bitstring) != expected_hf_bitstring:
            raise ValueError(
                "CEO manifest HF reference differs from the fixed sector"
            )
        pool = tuple(descriptor_from_manifest(item) for item in manifest.operators)
        pool_provenance = {
            "pool_type": "manifest",
            "manifest_digest": manifest.digest,
            "pool_digest": manifest.pool_digest,
            "pool_size": len(pool),
        }
    else:
        pool = generate_compact_ovp_ceo_pool(2 * active.norb_active)
        pool_provenance = {
            "pool_type": "generated_ovp_ceo",
            "pool_digest": canonical_digest([item.payload() for item in pool]),
            "pool_size": len(pool),
        }
    determinant_count = fixed_sector_dimension(
        active.norb_active, n_alpha, n_beta
    )

    last_progress: dict[str, int] = {}

    def progress(stage: str, completed: int, total: int) -> None:
        if vqe_args.verbose < 2:
            return
        previous = last_progress.get(stage, 0)
        if completed == total or completed - previous >= 256:
            print(
                f"    [fixed-sector] {stage}: {completed}/{total}",
                flush=True,
            )
            last_progress[stage] = completed

    started = perf_counter()
    result = solve_sector_adapt(
        active.h1,
        active.h2,
        n_alpha=n_alpha,
        n_beta=n_beta,
        pool=pool,
        gradient_threshold=vqe_args.adapt_gradient_threshold,
        eigenvalue_threshold=vqe_args.adapt_eigenvalue_threshold,
        max_adapt_iterations=vqe_args.adapt_max_iterations,
        optimizer_max_iterations=max(
            vqe_args.stage1_max_iter,
            vqe_args.stage2_max_iter,
            vqe_args.stage3_max_iter,
        ),
        optimizer_ftol=min(
            vqe_args.stage1_energy_tol,
            vqe_args.stage2_energy_tol,
            vqe_args.stage3_energy_tol,
        ),
        gradient_log_top_k=vqe_args.adapt_gradient_log_top_k,
        selector_config=SectorSelectorConfig(
            policy=vqe_args.adapt_sector_selector_policy,
            top_k=vqe_args.adapt_sector_selector_top_k,
            probe_max_iterations=(
                vqe_args.adapt_sector_probe_max_iterations
            ),
            probe_ftol=vqe_args.adapt_sector_probe_ftol,
            energy_tie_tolerance=(
                vqe_args.adapt_sector_energy_tie_tolerance
            ),
        ),
        progress_callback=progress,
    )
    ci = sector_state_to_pyscf_ci(result.state)
    rdm_started = perf_counter()
    rdm1, rdm2 = direct_spin1.make_rdm12(
        ci,
        active.norb_active,
        (n_alpha, n_beta),
    )
    rdm1 = np.asarray(rdm1, dtype=np.float64)
    rdm2 = np.asarray(rdm2, dtype=np.float64)
    validate_rdm_pair(rdm1, rdm2, label="fixed-sector active-space")
    rdm_energy = float(
        np.einsum("pq,qp", active.h1, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", active.h2, rdm2)
    )
    rdm_energy_residual = abs(rdm_energy - result.energy)
    if rdm_energy_residual > 1.0e-9:
        raise ValueError(
            "Fixed-sector RDM energy identity failed: "
            f"residual={rdm_energy_residual:.3e} Ha"
        )
    evidence = {
        **result.evidence(),
        "manifest_digest": None if manifest is None else manifest.digest,
        "pool_digest": pool_provenance["pool_digest"],
        "pool_size": len(pool),
        "pool_provenance": pool_provenance,
        "active_hamiltonian_digest": active.active_hamiltonian_digest,
        "active_space_total_digest": active.total_hamiltonian_digest,
        "active_energy_ha": result.energy,
        "reported_energy_ha": result.energy + active.frozen_energy,
        "frozen_energy_ha": active.frozen_energy,
        "rdm1_trace": float(np.trace(rdm1)),
        "rdm_energy_ha": rdm_energy,
        "rdm_energy_residual_ha": rdm_energy_residual,
        "wall_s": perf_counter() - started,
        "rdm_wall_s": perf_counter() - rdm_started,
        "fixed_sector_bytes_per_real_vector": determinant_count * 8,
    }
    return result, rdm1, rdm2, evidence


def _finalize_sector_adapt_fragment(
    *,
    frag: Frags,
    vqe_args: VQE_ArgsUser,
    result: Any,
    rdm1_active: ndarray,
    rdm2_active: ndarray,
    evidence: dict[str, Any],
    frozen_indices: Sequence[int],
    discarded_virtual_indices: Sequence[int],
    norb_full: int,
    phase_timings: dict[str, float],
) -> tuple[Matrix, Matrix]:
    """Expand, rotate, validate, record, and return a sector-ADAPT RDM pair."""

    global _vqe_state

    rdm1_canonical = np.asarray(rdm1_active, dtype=np.float64)
    rdm2_canonical = np.asarray(rdm2_active, dtype=np.float64)
    if frozen_indices or discarded_virtual_indices:
        rdm1_canonical, rdm2_canonical = expand_rdm_from_active_space(
            rdm1_canonical,
            rdm2_canonical,
            frozen_indices,
            norb_full,
            discarded_virtual_indices,
        )
    rotation = getattr(frag, "canonical_rotation", None)
    if rotation is None:
        raise RuntimeError("Fixed-sector VQE requires the canonical fragment rotation")
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (norb_full, norb_full):
        raise ValueError(
            "Canonical rotation and expanded fixed-sector RDM dimensions differ"
        )
    rdm1 = rotation @ rdm1_canonical @ rotation.T
    rdm2 = np.einsum(
        "ip,jq,kr,ls,pqrs->ijkl",
        rotation,
        rotation,
        rotation,
        rotation,
        rdm2_canonical,
        optimize=True,
    )
    validate_rdm_pair(rdm1, rdm2, label="fixed-sector fragment-basis")

    frag_name = str(frag.dname)
    total_energy = float(evidence["reported_energy_ha"])
    frag.active_space_provenance["active_objective_energy_ha"] = float(
        evidence["active_energy_ha"]
    )
    frag.active_space_provenance["reported_total_energy_ha"] = total_energy
    phase_timings.update(
        {
            "optimizer": float(evidence["wall_s"] - evidence["rdm_wall_s"]),
            "statevector_and_rdms": float(evidence["rdm_wall_s"]),
            "optimizer_iterations": float(result.optimizer_iterations),
            "total": float(evidence["wall_s"]),
        }
    )
    rdm1_bytes = np.ascontiguousarray(rdm1, dtype="<f8").tobytes(order="C")
    evidence = {
        **evidence,
        "fragment_name": frag_name,
        "fragment_rdm1_digest": f"sha256:{sha256(rdm1_bytes).hexdigest()}",
        "phase_timings": dict(phase_timings),
    }
    prior_evidence = list(getattr(frag, "sector_adapt_evidence", []))
    prior_evidence.append(evidence)
    frag.sector_adapt_evidence = prior_evidence
    prior_states = list(getattr(frag, "sector_adapt_states", []))
    prior_states.append(result.state)
    frag.sector_adapt_states = prior_states

    parameters = np.asarray(result.parameters, dtype=np.float64)
    if vqe_args.warm_start:
        _vqe_state.fragment_params[frag_name] = parameters
    _vqe_state.fragment_timings[frag_name] = dict(phase_timings)
    if vqe_args.track_iteration_history:
        _vqe_state.fragment_iteration_history[frag_name] = [
            dict(item) for item in result.gradient_history
        ]
    else:
        _vqe_state.fragment_iteration_history.pop(frag_name, None)
    _vqe_state.fragment_energies[frag_name] = total_energy
    _vqe_state.fragment_rdm1[frag_name] = rdm1
    _vqe_state.fragment_rdm2[frag_name] = rdm2
    for callback in _vqe_result_callbacks:
        try:
            callback(frag_name, total_energy, rdm1, rdm2)
        except Exception as exc:
            if vqe_args.verbose >= 1:
                print(f"  Warning: VQE callback failed: {exc}")
    return rdm1, rdm2


def solve_vqe(
    mf: RHF,
    frag: Frags,
    vqe_args: VQE_ArgsUser,
    be_energy: float | None = None,
) -> tuple[Matrix, Matrix]:
    """
    Solve fragment using VQE with UCCSD ansatz.

    This function:
    1. Loads pre-computed Hamiltonian from FCIDUMP file
    2. Builds UCCSD ansatz circuit
    3. Runs VQE with adaptive convergence
    4. Computes 1-RDM and 2-RDM from optimized statevector
    5. Supports warm-start for faster convergence

    Parameters
    ----------
    mf : RHF
        Mean-field object (for interface compatibility, not used)
    frag : Frags
        Fragment object containing fragment index
    vqe_args : VQE_ArgsUser
        VQE configuration parameters
    be_energy : float, optional
        Current BE total energy (for adaptive convergence)

    Returns
    -------
    rdm1 : ndarray (norb, norb)
        One-particle reduced density matrix
    rdm2 : ndarray (norb, norb, norb, norb)
        Two-particle reduced density matrix
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for VQE solver. "
            "Install with: pip install qiskit qiskit-nature qiskit-algorithms"
        )

    global _vqe_state

    global_start = perf_counter()
    phase_timings: dict[str, float] = {}

    # Update BE iteration tracking
    if be_energy is not None:
        _vqe_state.update_be_iteration(be_energy)

    # Determine convergence stage
    be_energy_change = _vqe_state.get_be_energy_change()
    stage = _vqe_state.get_stage(be_energy_change, vqe_args)
    max_iter, energy_tol = _vqe_state.get_vqe_params(stage, vqe_args)

    if vqe_args.verbose >= 1:
        print(
            f"VQE Fragment {frag.dname}: Stage {stage}, "
            f"BE ΔE={be_energy_change:.2e}, "
            f"max_iter={max_iter}, tol={energy_tol:.2e}"
        )

    # Regenerate FCIDUMP with current effective Hamiltonian
    # This ensures VQE sees the updated chemical potential from BE optimization
    ham_dir = Path(vqe_args.hamiltonian_dir)
    frag_name = str(frag.dname)  # Needed for warm-start logic
    ham_file = regenerate_fcidump_with_heff(frag, ham_dir, vqe_args)

    if vqe_args.verbose >= 2:
        print(f"  Regenerated FCIDUMP with current heff: {ham_file}")

    # Manual occupied-orbital downfolding consumes the exact in-memory
    # canonical Hamiltonian used to write the FCIDUMP.  This is the matched
    # path shared with FCI; no text round-trip or independent orbital sorting
    # can change its integrals or frozen indices.
    canonical = getattr(frag, "_canonical_fragment_hamiltonian", None)
    if not isinstance(canonical, CanonicalFragmentHamiltonian):
        raise RuntimeError("Regenerated canonical Hamiltonian was not retained")
    if vqe_args.frozen_core == "manual":
        active_hamiltonian = build_vqe_active_space_hamiltonian(
            canonical,
            vqe_args,
        )
        qubit_hamiltonian = _map_active_space_hamiltonian(active_hamiltonian)
        norb = active_hamiltonian.norb_active
        nelec = active_hamiltonian.nelec_active
        core_energy = active_hamiltonian.frozen_energy
        frozen_indices = list(active_hamiltonian.frozen_indices)
        discarded_virtual_indices = list(
            active_hamiltonian.discarded_virtual_indices
        )
        norb_full = active_hamiltonian.norb_full
        frag.active_space_provenance = active_hamiltonian.provenance()
        frag.active_space_provenance["solver"] = "VQE"
        frag.active_space_provenance["mode"] = "manual"
        record_fragment_ao_character_diagnostic(
            frag,
            canonical,
            active_hamiltonian,
        )
    elif vqe_args.frozen_core == "auto":
        (
            qubit_hamiltonian,
            norb,
            nelec,
            core_energy,
            frozen_indices,
            norb_full,
        ) = parse_fcidump_hamiltonian_with_frozen_core(
            ham_file,
            frozen_core=vqe_args.frozen_core,
            frozen_core_threshold=vqe_args.frozen_core_threshold,
            frozen_core_num_orbitals=vqe_args.frozen_core_num_orbitals,
            verbose=vqe_args.verbose,
        )
        frag.active_space_provenance = {
            "schema": "quemb.active-space-provenance/legacy-auto-v1",
            "solver": "VQE",
            "mode": "auto",
            "norb_full": norb_full,
            "nelec_full": canonical.nelec,
            "norb_active": norb,
            "nelec_active": nelec,
            "frozen_indices": list(frozen_indices),
            "active_indices": [
                index
                for index in range(norb_full)
                if index not in frozen_indices
            ],
            "full_hamiltonian_digest": canonical.digest,
            "active_hamiltonian_digest": None,
        }
        discarded_virtual_indices = []
    else:
        full_space = build_active_space_hamiltonian(canonical, 0)
        qubit_hamiltonian = _map_active_space_hamiltonian(full_space)
        norb = full_space.norb_active
        nelec = full_space.nelec_active
        core_energy = 0.0
        frozen_indices = []
        discarded_virtual_indices = []
        norb_full = full_space.norb_full
        frag.active_space_provenance = full_space.provenance()
        frag.active_space_provenance["solver"] = "VQE"
        frag.active_space_provenance["mode"] = "none"

    t_after_parse = perf_counter()
    phase_timings["load_hamiltonian"] = t_after_parse - global_start

    if vqe_args.verbose >= 2:
        reduction_info = ""
        if frozen_indices or discarded_virtual_indices:
            reduction_info = (
                f", frozen_occ={len(frozen_indices)}, "
                f"discarded_virt={len(discarded_virtual_indices)}"
            )
        print(
            f"  Loaded Hamiltonian: norb={norb}, nelec={nelec}, "
            f"nqubits={2 * norb}, core_energy={core_energy:.6f}"
            f"{reduction_info}"
        )

    if vqe_args.ansatz_type == "adapt_sector":
        manifest_path = _get_exact_sparse_ceo_manifest()
        if vqe_args.frozen_core == "auto":
            raise ValueError(
                "adapt_sector rejects automatic frozen-core selection; use an "
                "explicit ActiveSpaceSpec"
            )
        if not isinstance(active_hamiltonian, ActiveSpaceHamiltonian):
            raise RuntimeError("adapt_sector did not receive an active Hamiltonian")
        if vqe_args.verbose >= 1:
            print(
                "  Using fixed-sector CEO-ADAPT: "
                f"norb={norb}, sector=({(nelec + 1) // 2},{nelec // 2}), "
                "full_statevector=False",
                flush=True,
            )
        sector_started = perf_counter()
        sector_result, rdm1_active, rdm2_active, sector_evidence = (
            _solve_sector_adapt_active(
                active_hamiltonian,
                vqe_args,
                manifest_path,
            )
        )
        phase_timings["build_ansatz"] = 0.0
        phase_timings["initial_parameters"] = 0.0
        phase_timings["backend_setup"] = 0.0
        phase_timings["sector_total"] = perf_counter() - sector_started
        return _finalize_sector_adapt_fragment(
            frag=frag,
            vqe_args=vqe_args,
            result=sector_result,
            rdm1_active=rdm1_active,
            rdm2_active=rdm2_active,
            evidence=sector_evidence,
            frozen_indices=frozen_indices,
            discarded_virtual_indices=discarded_virtual_indices,
            norb_full=norb_full,
            phase_timings=phase_timings,
        )

    # Build ansatz based on ansatz_type selection
    # Phase 24: Added ADAPT-VQE support
    use_adapt = vqe_args.ansatz_type in ("adapt", "adapt_fast", "adapt_matrix_free")
    use_fast_adapt = vqe_args.ansatz_type == "adapt_fast"
    use_matrix_free = vqe_args.ansatz_type == "adapt_matrix_free"
    private_ceo_manifest = _get_exact_sparse_ceo_manifest()
    use_private_ceo_pool = private_ceo_manifest is not None
    selector_factory = _validate_adapt_selector_factory_usage(
        vqe_args,
        use_adapt=use_adapt,
    )
    if use_private_ceo_pool:
        if not use_adapt:
            raise ValueError(
                "Private CEO manifest context requires an ADAPT ansatz "
                "('adapt_fast' or 'adapt_matrix_free')."
            )
        if vqe_args.ansatz_type not in ("adapt_fast", "adapt_matrix_free"):
            raise ValueError(
                "Private CEO manifest context requires ansatz_type "
                "'adapt_fast' or 'adapt_matrix_free'"
            )
        if vqe_args.estimator_type != "direct_sv":
            raise ValueError(
                "Private CEO manifest context requires "
                "estimator_type='direct_sv'"
            )
        if vqe_args.frozen_core == "auto":
            raise ValueError(
                "Private CEO manifest context rejects automatic frozen-core "
                "selection; use frozen_core='manual' with an explicit "
                "ActiveSpaceSpec or frozen_core='none'"
            )
        private_active_spec = resolve_active_space_spec(vqe_args)
        if vqe_args.frozen_core == "manual" and not (
            private_active_spec.frozen_occupied_orbitals > 0
            or private_active_spec.discarded_virtual_orbitals > 0
        ):
            raise ValueError(
                "Private CEO manifest manual downfolding requires a nonempty "
                "ActiveSpaceSpec"
            )

    if use_adapt:
        # For ADAPT-VQE, we build the operator pool (UCC) instead of fixed UCCSD
        # ADAPT-VQE dynamically grows the ansatz during optimization
        if vqe_args.verbose >= 1:
            print(f"  Using ADAPT-VQE (iteratively grows ansatz)")

        from qiskit_nature.second_q.circuit.library import UCC, UCCSD, HartreeFock

        mapper = JordanWignerMapper()
        n_alpha = (nelec + 1) // 2
        n_beta = nelec - n_alpha
        num_particles = (n_alpha, n_beta)
        adapt_pool_provenance: dict[str, Any] = {
            "pool_type": (
                "ceo_ovp_manifest" if use_private_ceo_pool else "ucc"
            ),
            "n_qubits": 2 * norb,
            "n_electrons": nelec,
            "active_space": dict(frag.active_space_provenance),
        }

        # Initial state: Hartree-Fock
        hf_state = HartreeFock(norb, num_particles, mapper)

        if use_private_ceo_pool:
            from quemb.molbe.ceo_manifest import (
                load_ceo_manifest_for_system,
            )

            expected_n_qubits = 2 * norb
            ceo_manifest = load_ceo_manifest_for_system(
                private_ceo_manifest,
                n_qubits=expected_n_qubits,
                n_electrons=nelec,
            )
            _validate_private_ceo_manifest_downfolding(
                ceo_manifest,
                vqe_args,
            )

            expected_hf_bitstring = tuple(
                int(
                    qubit < n_alpha
                    or norb <= qubit < norb + n_beta
                )
                for qubit in range(expected_n_qubits)
            )
            manifest_hf_bitstring = tuple(
                int(bit) for bit in ceo_manifest.hf_bitstring
            )
            if manifest_hf_bitstring != expected_hf_bitstring:
                raise ValueError(
                    "CEO manifest HF reference does not match the fragment "
                    "Hartree-Fock occupation in blocked-spin order"
                )

            adapt_operators = list(ceo_manifest.generators)
            adapt_pool_provenance.update(
                {
                    "manifest_digest": ceo_manifest.digest,
                    "pool_digest": ceo_manifest.pool_digest,
                    "pool_size": len(adapt_operators),
                }
            )
            if ceo_manifest.hamiltonian_digest is not None:
                adapt_pool_provenance["manifest_hamiltonian_digest"] = (
                    ceo_manifest.hamiltonian_digest
                )
            else:
                adapt_pool_provenance["manifest_artifact_kind"] = (
                    ceo_manifest.artifact_kind
                )
            ansatz = hf_state
            if vqe_args.verbose >= 1:
                print(
                    "  Loaded CEO OVP manifest: "
                    f"operators={len(adapt_operators)}, "
                    f"pool_digest={ceo_manifest.pool_digest}, "
                    f"manifest_digest={ceo_manifest.digest}"
                )
        else:
            # UCC operator pool for ADAPT-VQE. Generalized=True allows
            # any-to-any excitations and reps creates k-UpCCGSD repetitions.
            use_generalized = vqe_args.ucc_generalized
            ucc_reps = vqe_args.ucc_reps
            preserve_spin = vqe_args.ucc_preserve_spin

            if use_generalized and vqe_args.verbose >= 1:
                print(
                    "  Using k-UpCCGSD ansatz "
                    f"(generalized={use_generalized}, reps={ucc_reps})"
                )

            adapt_ansatz = UCC(
                num_spatial_orbitals=norb,
                num_particles=num_particles,
                excitations="sd",
                qubit_mapper=mapper,
                initial_state=hf_state,
                generalized=use_generalized,
                reps=ucc_reps,
                preserve_spin=preserve_spin,
            )
            adapt_operators = list(adapt_ansatz.operators)
            adapt_pool_provenance["pool_size"] = len(adapt_operators)
            ansatz = adapt_ansatz

        if vqe_args.verbose >= 2:
            print(f"  ADAPT operator pool size: {len(adapt_operators)} operators")

        t_after_ansatz = perf_counter()
        phase_timings["build_ansatz"] = t_after_ansatz - t_after_parse
        if use_private_ceo_pool:
            ordered_parameters = []
            num_parameters = len(adapt_operators)
        else:
            ordered_parameters = list(ansatz.parameters)
            num_parameters = len(ordered_parameters)
    else:
        # Build and transpile UCCSD ansatz (with caching to avoid repeated transpilation)
        # This is the main performance optimization for large systems
        opt_level = vqe_args.transpile_optimization_level
        ansatz = get_cached_ansatz(norb, nelec, opt_level, vqe_args.verbose)
        t_after_ansatz = perf_counter()
        phase_timings["build_ansatz"] = t_after_ansatz - t_after_parse
        ordered_parameters = list(ansatz.parameters)
        num_parameters = len(ordered_parameters)

    # Initial parameters (warm-start or cold-start)
    if vqe_args.warm_start and frag_name in _vqe_state.fragment_params:
        initial_point = _vqe_state.fragment_params[frag_name]
        if len(initial_point) != num_parameters:
            if vqe_args.verbose >= 1:
                print(
                    f"  Warm-start parameter length {len(initial_point)} mismatch "
                    f"with ansatz size {num_parameters}; reinitializing."
                )
            initial_point = np.zeros(num_parameters)
        elif vqe_args.verbose >= 2:
            print(f"  Using warm-start parameters (size={len(initial_point)})")
    else:
        initial_point = np.zeros(num_parameters)
        if vqe_args.verbose >= 2:
            print(f"  Using cold-start (zeros, size={len(initial_point)})")

    t_after_initial = perf_counter()
    phase_timings["initial_parameters"] = t_after_initial - t_after_ansatz

    # Setup optimizer based on selection
    optimizer_name = vqe_args.optimizer_name
    print(f"  Using optimizer: {optimizer_name}")

    if optimizer_name == "COBYLA":
        # Note: rhoend must be passed via options dict for Qiskit's COBYLA wrapper
        optimizer = COBYLA(
            maxiter=max_iter,
            tol=energy_tol,
            rhobeg=vqe_args.cobyla_rhobeg,
        )
    elif optimizer_name == "L_BFGS_B":
        optimizer = L_BFGS_B(
            maxiter=max_iter,
            ftol=energy_tol,
        )
    elif optimizer_name == "SPSA":
        optimizer = SPSA(
            maxiter=max_iter,
            callback=None,  # We'll use VQE callback instead
        )
    elif optimizer_name == "SLSQP":
        optimizer = SLSQP(
            maxiter=max_iter,
            ftol=energy_tol,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Use COBYLA, L_BFGS_B, SPSA, or SLSQP"
        )

    # Setup VQE with Aer backend (GPU or multi-threaded CPU)
    # Check environment variable for device preference
    import os

    device = os.environ.get("QISKIT_DEVICE", "CPU").upper()

    backend = None

    if device == "GPU":
        # GPU backend for 8-12x speedup on CUDA-enabled instances
        try:
            backend = AerSimulator(
                method="statevector",
                device="GPU",
                precision=vqe_args.aer_precision,
            )
            if vqe_args.verbose >= 1:
                print(f"  Using Aer GPU backend ({vqe_args.aer_precision})")
        except Exception as exc:
            if vqe_args.verbose >= 1:
                print(f"  GPU backend unavailable ({exc}); falling back to CPU")
            device = "CPU"

    if backend is None:
        # Multi-threaded CPU backend for 3-4x speedup
        max_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
        backend = AerSimulator(
            method="statevector",
            device="CPU",
            max_parallel_threads=max_threads,
            precision=vqe_args.aer_precision,
        )
        if vqe_args.verbose >= 1:
            print(
                f"  Using Aer CPU backend with {max_threads} threads "
                f"({vqe_args.aer_precision})"
            )

    t_after_backend = perf_counter()
    phase_timings["backend_setup"] = t_after_backend - t_after_initial

    rng = np.random.default_rng(vqe_args.random_seed)
    progress_bar_enabled = vqe_args.show_progress_bar and max_iter > 0
    progress_bar_width = 30
    callback_needed = (
        vqe_args.track_iteration_history
        or vqe_args.track_density_matrices
        or vqe_args.verbose >= 2
        or progress_bar_enabled
        or vqe_args.use_best_on_limit  # Phase 12: Need callback to track best params
    )
    interval = max(vqe_args.density_sample_interval, 1)

    def _build_param_dict(values: Any) -> dict[Parameter, float] | None:
        try:
            vector = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if vector.size < num_parameters:
            return None
        if vector.size > num_parameters:
            vector = vector[:num_parameters]
        return {
            ordered_parameters[idx]: float(vector[idx]) for idx in range(num_parameters)
        }

    def execute_vqe(
        initial_point: np.ndarray, run_index: int, label: str
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        prev_energy: float | None = None
        prev_rdm1: ndarray | None = None
        callback_start_time = 0.0
        last_callback_time = 0.0
        progress_last_eval = -1
        progress_bar_drawn = False

        # Best-selection tracking (Phase 12)
        best_energy_seen: float = float("inf")
        best_params_seen: np.ndarray | None = None
        best_iter_seen: int = 0

        def render_progress(eval_count: int) -> None:
            nonlocal progress_last_eval, progress_bar_drawn
            if not progress_bar_enabled:
                return
            eval_int = int(eval_count)
            if eval_int == progress_last_eval:
                return
            progress_last_eval = eval_int
            clamped_eval = min(max(eval_int, 0), max_iter)
            fraction = clamped_eval / max_iter if max_iter else 1.0
            filled = min(progress_bar_width, int(fraction * progress_bar_width))
            bar = "#" * filled + "-" * (progress_bar_width - filled)
            print(
                f"  Progress [{bar}] {fraction * 100:6.2f}% ({clamped_eval}/{max_iter})",
                end="",
                flush=True,
            )
            progress_bar_drawn = True

        interval_local = interval

        def vqe_callback(
            eval_count: int, parameters: np.ndarray, mean: float, metadata: Any
        ):  # type: ignore[override]
            nonlocal prev_energy, prev_rdm1, last_callback_time, callback_start_time
            nonlocal best_energy_seen, best_params_seen, best_iter_seen

            energy_raw = float(np.real(mean))
            energy_total = energy_raw + core_energy

            # Track best parameters (Phase 12: best-selection on iteration limit)
            # Skip burn-in period to avoid unstable early iterations
            if (
                vqe_args.use_best_on_limit
                and eval_count >= vqe_args.best_select_min_iter
                and energy_total < best_energy_seen
            ):
                best_energy_seen = energy_total
                best_params_seen = parameters.copy()
                best_iter_seen = int(eval_count)
            record: dict[str, Any] = {
                "eval_count": int(eval_count),
                "energy": energy_total,
                "raw_energy": energy_raw,
            }

            current_time = perf_counter()
            if callback_start_time == 0.0:
                callback_start_time = current_time
            if last_callback_time == 0.0:
                last_callback_time = current_time
            record["elapsed_time"] = current_time - callback_start_time
            record["delta_time"] = current_time - last_callback_time
            last_callback_time = current_time

            std_dev_val: float | None = None
            if metadata is not None:
                if isinstance(metadata, dict):
                    variance = metadata.get("variance") or metadata.get("variances")
                    if variance is not None:
                        if isinstance(variance, (list, tuple, np.ndarray)):
                            if len(variance) > 0:
                                var_value = float(variance[0])
                                std_dev_val = float(np.sqrt(max(var_value, 0.0)))
                        else:
                            var_value = float(variance)
                            std_dev_val = float(np.sqrt(max(var_value, 0.0)))
                    std_dev_candidate = metadata.get("stddev") or metadata.get(
                        "standard_error"
                    )
                    if std_dev_candidate is not None and std_dev_val is None:
                        if isinstance(std_dev_candidate, (list, tuple, np.ndarray)):
                            if len(std_dev_candidate) > 0:
                                std_dev_val = float(std_dev_candidate[0])
                        else:
                            std_dev_val = float(std_dev_candidate)
                elif isinstance(metadata, (float, int, np.floating)):
                    std_dev_val = float(metadata)

            if std_dev_val is not None:
                record["stddev"] = std_dev_val

            delta_energy = (
                energy_total - prev_energy if prev_energy is not None else None
            )
            record["delta_energy"] = delta_energy
            prev_energy = energy_total

            if progress_bar_enabled:
                render_progress(eval_count)

            if vqe_args.track_density_matrices:
                compute_density = (
                    eval_count % interval_local == 0
                ) or prev_rdm1 is None
                if compute_density:
                    param_binding = _build_param_dict(parameters)
                    if param_binding is not None:
                        density_t0 = perf_counter()
                        bound_circuit = ansatz.assign_parameters(param_binding)
                        statevector_iter = Statevector(bound_circuit)
                        # compute_rdms_from_statevector returns (rdm1, rdm2) tuple
                        rdm1_iter, _rdm2_iter = compute_rdms_from_statevector(
                            statevector_iter, norb, nelec
                        )
                        record["rdm1_trace"] = float(np.trace(rdm1_iter))
                        if prev_rdm1 is not None:
                            record["rdm1_delta"] = float(
                                np.linalg.norm(rdm1_iter - prev_rdm1)
                            )
                        else:
                            record["rdm1_delta"] = None
                        record["rdm1_matrix"] = rdm1_iter
                        prev_rdm1 = rdm1_iter
                        record["density_time"] = perf_counter() - density_t0
                    else:
                        record["rdm1_trace"] = None
                        record["rdm1_delta"] = None
                        record["rdm1_matrix"] = None
                        record["density_time"] = None
                else:
                    record["rdm1_trace"] = None
                    record["rdm1_delta"] = None
                    record["rdm1_matrix"] = None
                    record["density_time"] = None
            else:
                record["density_time"] = None

            records.append(record)

            if vqe_args.verbose >= 2:
                delta_str = (
                    f" ΔE={delta_energy:+.3e}" if delta_energy is not None else ""
                )
                std_str = (
                    f" σ={record['stddev']:.2e}"
                    if record.get("stddev") is not None
                    else ""
                )
                density_str = ""
                if vqe_args.track_density_matrices:
                    rdm_delta = record.get("rdm1_delta")
                    if rdm_delta is not None:
                        density_str = f" Δ||ρ||={rdm_delta:+.3e}"
                print(
                    f"    iter {eval_count:>3}: E={energy_total:.10f} Ha{delta_str}{std_str}{density_str}"
                )

        if vqe_args.verbose >= 1:
            print(f"  VQE restart {run_index + 1} ({label})", flush=True)

        def build_vqe(
            selected_backend: AerSimulator, init_point: np.ndarray
        ) -> QiskitVQE | AdaptVQE:
            # Phase 13: Choose estimator based on user preference
            # Phase 23: Added AerEstimatorV2 for 2-3x faster exact computation
            # StatevectorEstimator and AerEstimatorV2(precision=0) give EXACT expectation values
            # BackendEstimatorV2 adds sampling noise (~15 mHa by default)
            if vqe_args.estimator_type == "statevector":
                estimator_local = StatevectorEstimator()
                if vqe_args.verbose >= 2:
                    print("    Using StatevectorEstimator (exact, Python/NumPy)")
            elif vqe_args.estimator_type == "aer_exact":
                # AerEstimatorV2 with precision=0 for exact expectation values
                # Uses C++/OpenMP backend for 2-3x speedup over StatevectorEstimator
                # Use GPU if available (set via QISKIT_DEVICE env var)
                estimator_local = AerEstimatorV2(
                    options={
                        "default_precision": 0.0,  # Exact (no shot sampling)
                        "backend_options": {
                            "method": "statevector",
                            "device": device,  # GPU or CPU based on env var
                            "precision": vqe_args.aer_precision,
                            "max_parallel_threads": vqe_args.aer_max_parallel_threads,
                        },
                    }
                )
                if vqe_args.verbose >= 2:
                    if device == "GPU":
                        print(
                            "    Using AerEstimatorV2 "
                            f"(exact, GPU accelerated, {vqe_args.aer_precision})"
                        )
                    else:
                        print(
                            "    Using AerEstimatorV2 "
                            f"(exact, C++/OpenMP, {vqe_args.aer_max_parallel_threads} threads, "
                            f"{vqe_args.aer_precision})"
                        )
            elif vqe_args.estimator_type == "backend":
                # BackendEstimatorV2 with optional precision setting (for hardware compatibility)
                if vqe_args.backend_estimator_precision is not None:
                    estimator_local = BackendEstimatorV2(
                        backend=selected_backend,
                        options={
                            "default_precision": vqe_args.backend_estimator_precision
                        },
                    )
                    if vqe_args.verbose >= 2:
                        print(
                            f"    Using BackendEstimatorV2 (precision={vqe_args.backend_estimator_precision})"
                        )
                else:
                    estimator_local = BackendEstimatorV2(backend=selected_backend)
                    if vqe_args.verbose >= 2:
                        print(
                            "    Using BackendEstimatorV2 (default precision ~15.6 mHa)"
                        )
            elif vqe_args.estimator_type == "direct_sv":
                # Direct statevector solver.  Existing UCC behavior remains
                # circuit-backed; CEO exact_sparse uses sequential
                # expm_multiply and intentionally has no circuit Jacobian.
                from quemb.molbe.fast_adapt_vqe import (
                    DirectSVSolver,
                    ExactSparseSVSolver,
                )

                use_exact_sparse = use_adapt and use_private_ceo_pool
                if use_exact_sparse:
                    direct_solver = ExactSparseSVSolver(
                        optimizer_maxiter=optimizer.settings.get(
                            "maxiter", max_iter
                        ),
                        optimizer_ftol=optimizer.settings.get(
                            "ftol", energy_tol
                        ),
                        initial_state=hf_state,
                        initial_point=init_point,
                        verbose=vqe_args.verbose,
                    )
                else:
                    direct_solver = DirectSVSolver(
                        optimizer_maxiter=optimizer.settings.get(
                            "maxiter", max_iter
                        ),
                        optimizer_ftol=optimizer.settings.get(
                            "ftol", energy_tol
                        ),
                        ansatz=ansatz,
                        initial_point=init_point,
                        verbose=vqe_args.verbose,
                        use_exact_jacobian=(
                            vqe_args.direct_sv_use_exact_jacobian
                        ),
                    )
                if vqe_args.verbose >= 2:
                    print(
                        "    Using direct statevector solver "
                        f"(evolution={'exact_sparse' if use_exact_sparse else 'qiskit'}, "
                        "optimizer=SLSQP, "
                        f"exact_jac={False if use_exact_sparse else direct_solver.use_exact_jacobian})"
                    )
                # DirectSVSolver is a complete solver, not a QiskitVQE.
                # For ADAPT, wrap it directly; for non-ADAPT, return it.
                if use_adapt:
                    from qiskit.synthesis import LieTrotter
                    from quemb.molbe.fast_adapt_vqe import FastAdaptVQE

                    if use_matrix_free:
                        from quemb.molbe.matrix_free_adapt_vqe import (
                            MatrixFreeAdaptVQE,
                        )

                        adapt_class = MatrixFreeAdaptVQE
                        adapt_label = "Matrix-free ADAPT-VQE"
                    else:
                        adapt_class = FastAdaptVQE
                        adapt_label = "Fast ADAPT-VQE"

                    adapt_vqe = _build_adapt_vqe(
                        adapt_class,
                        adapt_kwargs={
                            "solver": direct_solver,
                            "gradient_threshold": (
                                vqe_args.adapt_gradient_threshold
                            ),
                            "eigenvalue_threshold": (
                                vqe_args.adapt_eigenvalue_threshold
                            ),
                            "max_iterations": vqe_args.adapt_max_iterations,
                            "operators": adapt_operators,
                            "initial_state": hf_state,
                            "evolution": (
                                None if use_exact_sparse else LieTrotter()
                            ),
                            "flatten": True,
                            "check_cyclicity": vqe_args.adapt_check_cyclicity,
                            "cyclicity_action": vqe_args.adapt_cyclicity_action,
                            "gradient_log_top_k": (
                                vqe_args.adapt_gradient_log_top_k
                            ),
                            "tracked_operator_indices": (
                                vqe_args.adapt_tracked_operator_indices
                            ),
                            "verbose": vqe_args.verbose,
                        },
                        pool_provenance=adapt_pool_provenance,
                        selector_factory=selector_factory,
                    )
                    if vqe_args.verbose >= 2:
                        print(
                            f"    {adapt_label} (direct_sv): "
                            f"grad_thresh={vqe_args.adapt_gradient_threshold}, "
                            f"max_iter={vqe_args.adapt_max_iterations}"
                        )
                    return adapt_vqe
                return direct_solver
            else:
                # Fail fast for invalid estimator_type - don't silently fall back!
                # This catches typos like "aer" instead of "aer_exact"
                raise ValueError(
                    f"Unknown estimator_type: '{vqe_args.estimator_type}'. "
                    f"Valid options are: 'statevector', 'aer_exact', 'backend', 'direct_sv'"
                )
            vqe_kwargs: dict[str, Any] = {"initial_point": init_point}
            if callback_needed and not use_adapt:
                # ADAPT-VQE has its own callback mechanism
                vqe_kwargs["callback"] = vqe_callback

            base_vqe = QiskitVQE(estimator_local, ansatz, optimizer, **vqe_kwargs)

            if use_adapt:
                # Wrap base VQE with ADAPT-VQE for iterative ansatz growth
                # Use LieTrotter synthesis to decompose PauliEvolution gates into basic gates.
                # This avoids StatevectorEstimator converting to dense 2^n x 2^n matrix
                # (which would require 64GB for 16 qubits!)
                from qiskit.synthesis import LieTrotter

                if use_matrix_free:
                    # Matrix-free ADAPT-VQE: same algorithm as fast ADAPT but
                    # uses direct Pauli-string application instead of sparse matrices.
                    # Avoids OOM on 20+ qubit fragments where sparse matrices exceed RAM.
                    from quemb.molbe.matrix_free_adapt_vqe import MatrixFreeAdaptVQE

                    adapt_vqe = MatrixFreeAdaptVQE(
                        solver=base_vqe,
                        gradient_threshold=vqe_args.adapt_gradient_threshold,
                        eigenvalue_threshold=vqe_args.adapt_eigenvalue_threshold,
                        max_iterations=vqe_args.adapt_max_iterations,
                        operators=adapt_operators,
                        initial_state=hf_state,
                        evolution=LieTrotter(),
                        flatten=True,
                        check_cyclicity=vqe_args.adapt_check_cyclicity,
                        cyclicity_action=vqe_args.adapt_cyclicity_action,
                        gradient_log_top_k=vqe_args.adapt_gradient_log_top_k,
                        tracked_operator_indices=vqe_args.adapt_tracked_operator_indices,
                        verbose=vqe_args.verbose,
                    )
                    adapt_vqe.pool_provenance = dict(
                        adapt_pool_provenance
                    )
                    if vqe_args.verbose >= 2:
                        print(
                            f"    Matrix-free ADAPT-VQE: grad_thresh={vqe_args.adapt_gradient_threshold}, "
                            f"max_iter={vqe_args.adapt_max_iterations}"
                        )
                elif use_fast_adapt:
                    # Fast ADAPT-VQE: uses direct statevector gradients instead of
                    # symbolic commutator algebra. 10-100x faster gradient evaluation.
                    from quemb.molbe.fast_adapt_vqe import FastAdaptVQE

                    adapt_vqe = FastAdaptVQE(
                        solver=base_vqe,
                        gradient_threshold=vqe_args.adapt_gradient_threshold,
                        eigenvalue_threshold=vqe_args.adapt_eigenvalue_threshold,
                        max_iterations=vqe_args.adapt_max_iterations,
                        operators=adapt_operators,
                        initial_state=hf_state,
                        evolution=LieTrotter(),
                        flatten=True,
                        check_cyclicity=vqe_args.adapt_check_cyclicity,
                        cyclicity_action=vqe_args.adapt_cyclicity_action,
                        gradient_log_top_k=vqe_args.adapt_gradient_log_top_k,
                        tracked_operator_indices=vqe_args.adapt_tracked_operator_indices,
                        verbose=vqe_args.verbose,
                    )
                    adapt_vqe.pool_provenance = dict(
                        adapt_pool_provenance
                    )
                    if vqe_args.verbose >= 2:
                        print(
                            f"    Fast ADAPT-VQE: grad_thresh={vqe_args.adapt_gradient_threshold}, "
                            f"max_iter={vqe_args.adapt_max_iterations}"
                        )
                else:
                    # Standard Qiskit ADAPT-VQE (symbolic commutator gradients)
                    # Pass operators and initial_state explicitly to ensure flatten=True is used.
                    # When using an EvolvedOperatorAnsatz (like UCC) directly, ADAPT-VQE uses
                    # a code path that ignores flatten=True. By passing operators explicitly,
                    # we force it to use _build_ansatz() which respects flatten=True.
                    adapt_vqe = AdaptVQE(
                        solver=base_vqe,
                        gradient_threshold=vqe_args.adapt_gradient_threshold,
                        eigenvalue_threshold=vqe_args.adapt_eigenvalue_threshold,
                        max_iterations=vqe_args.adapt_max_iterations,
                        operators=adapt_operators,  # Explicit operator pool
                        initial_state=hf_state,  # Explicit HF initial state
                        evolution=LieTrotter(),  # Decompose to basic gates, avoid dense matrix
                        flatten=True,  # Force flattening to prevent StatevectorEstimator OOM on large systems
                    )
                    if vqe_args.verbose >= 2:
                        print(
                            f"    ADAPT-VQE: grad_thresh={vqe_args.adapt_gradient_threshold}, "
                            f"max_iter={vqe_args.adapt_max_iterations}"
                        )
                return adapt_vqe
            return base_vqe

        vqe = build_vqe(backend, initial_point)

        if vqe_args.verbose >= 2:
            print(f"  Running VQE optimization...")

        optimization_start = perf_counter()
        pre_opt = optimization_start - t_after_backend

        try:
            result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
        except AlgorithmError as exc:
            if device == "GPU":
                if vqe_args.verbose >= 1:
                    print(f"  GPU execution failed ({exc}); retrying on CPU backend")
                max_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
                cpu_backend = AerSimulator(
                    method="statevector", device="CPU", max_parallel_threads=max_threads
                )
                if vqe_args.verbose >= 1:
                    print(f"  Using Aer CPU backend with {max_threads} threads")
                vqe = build_vqe(cpu_backend, initial_point)
                result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
            else:
                raise
        opt_end = perf_counter()
        optimizer_time = opt_end - optimization_start
        if progress_bar_enabled:
            render_progress(result.cost_function_evals)
            if progress_bar_drawn:
                print()

        optimal_energy = result.eigenvalue.real + core_energy

        # Handle different result structures for ADAPT vs UCCSD
        if use_adapt:
            # ADAPT-VQE result has optimal_parameters dict, not optimal_point array
            if (
                hasattr(result, "optimal_parameters")
                and result.optimal_parameters is not None
            ):
                if isinstance(result.optimal_parameters, dict):
                    optimal_params = np.array(
                        list(result.optimal_parameters.values()), dtype=float
                    )
                else:
                    optimal_params = np.asarray(result.optimal_parameters, dtype=float)
            elif hasattr(result, "optimal_point") and result.optimal_point is not None:
                optimal_params = np.asarray(result.optimal_point, dtype=float)
            else:
                optimal_params = np.zeros(0)  # Empty if no params found

            if vqe_args.verbose >= 2:
                n_adapt_iters = getattr(result, "num_iterations", "?")
                print(
                    f"    ADAPT-VQE completed: {n_adapt_iters} iterations, "
                    f"{len(optimal_params)} final parameters"
                )
        else:
            optimal_params = np.asarray(result.optimal_point, dtype=float)

        # Check if we hit iteration limit vs actual convergence
        hit_limit = result.cost_function_evals >= max_iter
        used_best_selection = False

        # Phase 12: Best-selection on iteration limit
        # If we hit the limit and have a better result from earlier, use it
        if (
            vqe_args.use_best_on_limit
            and hit_limit
            and best_params_seen is not None
            and best_energy_seen < optimal_energy
        ):
            improvement = (optimal_energy - best_energy_seen) * 1000  # mHa
            if vqe_args.verbose >= 1:
                print(
                    f"  VQE hit limit ({result.cost_function_evals}/{max_iter}): "
                    f"final_E={optimal_energy:.6f} Ha, best_E={best_energy_seen:.6f} Ha (iter {best_iter_seen})"
                )
                print(
                    f"    -> Substituting BEST parameters (improvement: {improvement:.1f} mHa)"
                )
            optimal_energy = best_energy_seen
            optimal_params = best_params_seen
            used_best_selection = True
        elif vqe_args.verbose >= 1:
            status_msg = "hit iteration limit" if hit_limit else "converged"
            if hit_limit and best_params_seen is not None:
                # Hit limit but final is already the best
                print(
                    f"  VQE {status_msg}: E={optimal_energy:.8f} Ha, "
                    f"iterations={result.cost_function_evals} (final IS best)"
                )
            else:
                print(
                    f"  VQE {status_msg}: E={optimal_energy:.8f} Ha, "
                    f"iterations={result.cost_function_evals}",
                    flush=True,
                )

        # Get the final state for RDM computation.  Custom exact-sparse ADAPT
        # returns the state directly; circuit reconstruction would silently
        # replace its exact expm_multiply semantics with a synthesized circuit.
        optimal_state = getattr(result, "optimal_state", None)
        exact_state_solver = getattr(vqe, "solver", None)
        if use_adapt and optimal_state is not None:
            if (
                used_best_selection
                and exact_state_solver is not None
                and hasattr(exact_state_solver, "statevector_for")
            ):
                optimal_state = exact_state_solver.statevector_for(optimal_params)
            statevector = Statevector(
                np.asarray(optimal_state, dtype=np.complex128)
            )
            if vqe_args.verbose >= 2:
                print("    Using ADAPT-VQE exact returned state for RDM computation")
        # For legacy ADAPT-VQE, the optimal_circuit is already bound.
        elif (
            use_adapt
            and hasattr(result, "optimal_circuit")
            and result.optimal_circuit is not None
        ):
            bound_circuit = result.optimal_circuit
            # ADAPT-VQE optimal_circuit may still have unbound parameters
            if bound_circuit.num_parameters > 0:
                if (
                    hasattr(result, "optimal_parameters")
                    and result.optimal_parameters is not None
                ):
                    if isinstance(result.optimal_parameters, dict):
                        bound_circuit = bound_circuit.assign_parameters(
                            result.optimal_parameters
                        )
                    else:
                        param_dict = dict(
                            zip(bound_circuit.parameters, result.optimal_parameters)
                        )
                        bound_circuit = bound_circuit.assign_parameters(param_dict)
            if vqe_args.verbose >= 2:
                print(f"    Using ADAPT-VQE optimal circuit for RDM computation")
            statevector = Statevector(bound_circuit)
        else:
            # Standard UCCSD: bind optimal parameters to ansatz
            final_param_binding = _build_param_dict(optimal_params)
            if final_param_binding is None:
                raise ValueError(
                    "Failed to bind VQE optimal parameters to ansatz; "
                    f"expected {num_parameters} values, "
                    f"got {len(np.asarray(optimal_params).reshape(-1)) if optimal_params is not None else 0}."
                )
            bound_circuit = ansatz.assign_parameters(final_param_binding)
            statevector = Statevector(bound_circuit)

        if vqe_args.verbose >= 2:
            print(f"  Computing RDMs from statevector...")

        rdm_time_start = perf_counter()
        rdm1_local, rdm2_local = compute_rdms_from_statevector(statevector, norb, nelec)
        rdm_time_end = perf_counter()

        run_timings = {
            "pre_optimization": pre_opt,
            "optimizer": optimizer_time,
            "statevector_and_rdms": rdm_time_end - rdm_time_start,
        }

        return {
            "energy": float(optimal_energy),
            "raw_energy": float(result.eigenvalue.real),
            "params": optimal_params,
            "rdm1": rdm1_local,
            "rdm2": rdm2_local,
            "records": records,
            "iterations": int(result.cost_function_evals),
            "timings": run_timings,
            # Phase 12: Best-selection tracking
            "used_best_selection": used_best_selection,
            "best_iter": best_iter_seen if used_best_selection else None,
            "hit_limit": hit_limit,
        }

    candidate_points: list[tuple[str, np.ndarray]] = []
    stored_params = _vqe_state.fragment_params.get(frag_name)
    if vqe_args.warm_start and stored_params is not None:
        stored_arr = np.asarray(stored_params, dtype=float)
        # Only use warm-start if parameter sizes match (ADAPT-VQE can change ansatz size)
        if len(stored_arr) == num_parameters:
            candidate_points.append(("warm", stored_arr))
        elif vqe_args.verbose >= 1:
            print(
                f"  Warm-start parameter length {len(stored_arr)} mismatch "
                f"with ansatz size {num_parameters}; skipping warm-start.",
                flush=True,
            )

    zero_point = np.zeros(num_parameters)
    # Check sizes match before np.allclose to avoid broadcast errors
    if (
        not candidate_points
        or len(candidate_points[0][1]) != len(zero_point)
        or not np.allclose(candidate_points[0][1], zero_point)
    ):
        candidate_points.append(("hf", zero_point))

    while len(candidate_points) < max(1, vqe_args.max_restarts):
        candidate_points.append(
            (
                f"random_{len(candidate_points)}",
                rng.uniform(-np.pi, np.pi, size=num_parameters),
            )
        )

    best_run: dict[str, Any] | None = None
    best_energy = float("inf")

    # Determine number of restarts to run
    restart_candidates = candidate_points[: max(1, vqe_args.max_restarts)]
    num_parallel = min(vqe_args.parallel_restarts, len(restart_candidates))

    if num_parallel > 1:
        # Parallel restart execution using ThreadPoolExecutor
        if vqe_args.verbose >= 1:
            print(
                f"  Running {len(restart_candidates)} restarts in parallel "
                f"(max {num_parallel} concurrent)",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            # Submit all restart jobs
            future_to_restart = {
                executor.submit(execute_vqe, init_point, idx, label): (idx, label)
                for idx, (label, init_point) in enumerate(restart_candidates)
            }

            # Collect results as they complete
            for future in as_completed(future_to_restart):
                idx, label = future_to_restart[future]
                try:
                    run_data = future.result()
                    energy = run_data["energy"]
                    if vqe_args.verbose >= 1:
                        print(
                            f"  Restart {idx} ({label}) completed: E={energy:.8f} Ha",
                            flush=True,
                        )
                    if energy < best_energy:
                        best_energy = energy
                        best_run = run_data
                except Exception as e:
                    print(
                        f"  Restart {idx} ({label}) FAILED: {e}",
                        flush=True,
                    )
                    raise
    else:
        # Sequential restart execution (original behavior)
        for idx, (label, init_point) in enumerate(restart_candidates):
            run_data = execute_vqe(init_point, idx, label)
            energy = run_data["energy"]
            if energy < best_energy:
                best_energy = energy
                best_run = run_data

            if idx > 0 and abs(best_energy - energy) < vqe_args.restart_energy_tol:
                if vqe_args.verbose >= 1:
                    print(
                        f"  Restart improvement below {vqe_args.restart_energy_tol:.1e}; stopping restarts.",
                        flush=True,
                    )
                break

    if best_run is None:
        raise RuntimeError("VQE failed to produce a valid result")

    optimal_energy = float(best_run["energy"])
    frag.active_space_provenance["active_objective_energy_ha"] = (
        optimal_energy - core_energy
    )
    frag.active_space_provenance["reported_total_energy_ha"] = optimal_energy
    optimal_params = np.asarray(best_run["params"], dtype=float)
    rdm1_canonical = best_run["rdm1"]
    rdm2_canonical = best_run["rdm2"]
    callback_records = best_run["records"]

    # ========== DEBUG: Check canonical RDM consistency BEFORE transformation ==========
    # NOTE: RDM2 is already in CHEMIST convention from compute_rdms_from_statevector()
    print(
        f"\n--- DEBUG: Canonical RDM2 consistency check (CHEMIST convention, BEFORE basis transform) ---"
    )
    print(f"RDM1 canonical trace: {np.trace(rdm1_canonical):.6f}")
    print(f"RDM1 canonical diagonal: {np.diag(rdm1_canonical)}")

    # Check trace relation using CHEMIST convention: sum_r rdm2[p,q,r,r] = (N-1)*rdm1[p,q]
    norb_check = rdm1_canonical.shape[0]
    nelec_check = int(round(np.trace(rdm1_canonical)))
    print(f"Trace relation check (chemist): Σ_r rdm2[p,p,r,r] = (N-1)*rdm1[p,p]")
    for p in range(norb_check):
        sum_r = sum(rdm2_canonical[p, p, r, r] for r in range(norb_check))
        expected = (nelec_check - 1) * rdm1_canonical[p, p]
        print(
            f"  p={p}: sum={sum_r:.4f}, expected={expected:.4f}, diff={sum_r - expected:.2e}"
        )

    # Check RDM1 from RDM2 contraction (CHEMIST: pqrr->pq)
    rdm1_from_rdm2 = np.einsum("pqrr->pq", rdm2_canonical) / (nelec_check - 1)
    contraction_error = np.max(np.abs(rdm1_canonical - rdm1_from_rdm2))
    print(f"Max |rdm1 - contract(rdm2)/(N-1)| (chemist): {contraction_error:.2e}")
    if contraction_error > 1e-6:
        raise ValueError(
            "Canonical VQE RDM2 violates the chemist contraction relation: "
            f"max error={contraction_error:.3e}"
        )
    print(f"--- End canonical RDM2 check ---\n")

    # STEP 1: Expand RDMs from active space to full space if frozen core was used
    # This must happen BEFORE the canonical-to-original basis transformation
    if frozen_indices or discarded_virtual_indices:
        if vqe_args.verbose >= 2:
            print(
                f"\n  Expanding RDMs from active space ({norb} orbs) "
                f"to full space ({norb_full} orbs)"
            )
            print(f"  Frozen occupied orbitals: {frozen_indices}")
            print(f"  Discarded virtual orbitals: {discarded_virtual_indices}")

        # Dimension check before expansion
        expected_active_dim = (
            norb_full
            - len(frozen_indices)
            - len(discarded_virtual_indices)
        )
        if rdm1_canonical.shape[0] != expected_active_dim:
            raise ValueError(
                f"RDM dimension mismatch: got {rdm1_canonical.shape[0]}, "
                f"expected {expected_active_dim} (norb_full={norb_full}, "
                f"frozen={len(frozen_indices)}, "
                f"discarded={len(discarded_virtual_indices)})"
            )

        rdm1_canonical, rdm2_canonical = expand_rdm_from_active_space(
            rdm1_canonical,
            rdm2_canonical,
            frozen_indices,
            norb_full,
            discarded_virtual_indices,
        )

        # Update norb to full space for subsequent operations
        norb = norb_full
        nelec = nelec + 2 * len(frozen_indices)  # Restore total electron count

        if vqe_args.verbose >= 2:
            print(
                "  RDM1 trace after expansion: "
                f"{np.trace(rdm1_canonical):.6f} (expected: {nelec})"
            )
            print(f"  RDM1 shape after expansion: {rdm1_canonical.shape}")

    # STEP 2: Transform RDMs back from canonical to original MO basis
    # VQE computed RDMs in the occupied/virtual block-canonical basis.
    # But BE assembly expects RDMs in the original MO basis (matching frag.mo_coeffs).
    # The rotation matrix U transforms original → canonical: MO_canonical = U.T @ MO_original
    # So to transform RDMs back: rdm_original = U @ rdm_canonical @ U.T
    if hasattr(frag, "canonical_rotation"):
        # Transform RDMs from canonical basis back to fragment MO basis
        # BLOCKED spin-orbital indexing is correct for Qiskit - no permutation needed
        U = frag.canonical_rotation

        # Dimension check: U must match RDM dimensions (both should be norb_full after expansion)
        if U.shape[0] != rdm1_canonical.shape[0]:
            raise ValueError(
                f"Canonical rotation dimension mismatch: U is {U.shape[0]}x{U.shape[0]}, "
                f"but RDM1 is {rdm1_canonical.shape[0]}x{rdm1_canonical.shape[0]}. "
                f"This may indicate a frozen core expansion issue."
            )

        # Transform 1-RDM: rdm1_original[i,j] = U[k,i] rdm1_canonical[k,l] U[l,j]
        rdm1 = U @ rdm1_canonical @ U.T

        # RDM2 is already in chemist convention from compute_rdms_from_statevector().
        # (Operator +_p +_q -_s -_r stored at rdm2[p,r,q,s]; see proof in that function.)

        # Transform 2-RDM from canonical to fragment MO basis
        # rdm2_original[i,j,k,l] = U[i,p] U[j,q] U[k,r] U[l,s] rdm2_canonical[p,q,r,s]
        # NOTE: einsum indices must be "ip,jq,kr,ls" NOT "pi,qj,rk,sl" to match
        # the 1-RDM transform: rdm1_new = U @ rdm1 @ U.T
        rdm2 = np.einsum(
            "ip,jq,kr,ls,pqrs->ijkl", U, U, U, U, rdm2_canonical, optimize=True
        )

        # DEBUG: Check transformed RDM consistency (chemist convention preserved through transform)
        print(
            f"\n--- DEBUG: Transformed RDM2 consistency check (AFTER basis transform) ---"
        )
        print(
            f"U is orthogonal: max|U@U.T - I| = {np.max(np.abs(U @ U.T - np.eye(U.shape[0]))):.2e}"
        )
        # Chemists convention contraction: pqrr->pq (sum over last two indices when equal)
        rdm1_from_rdm2_transformed = np.einsum("pqrr->pq", rdm2) / (nelec - 1)
        transform_contraction_error = np.max(np.abs(rdm1 - rdm1_from_rdm2_transformed))
        print(
            f"Contraction check (chemists): max |rdm1 - sum_r rdm2[p,q,r,r]/(N-1)| = {transform_contraction_error:.2e}"
        )
        if transform_contraction_error > 1e-6:
            raise ValueError(
                "Transformed VQE RDM2 violates the chemist contraction "
                f"relation: max error={transform_contraction_error:.3e}"
            )
        else:
            print("  OK: Transformed RDM2 satisfies contraction relation")
        print(f"--- End transformed RDM2 check ---\n")

        # ========== DEBUG: VQE fragment results ==========
        # TODO: Print RDM occupations, natural occupations [DONE]
        print(f"\n{'=' * 80}")
        print(f"DEBUG vqe_solver.py - VQE FRAGMENT RESULTS")
        print(f"Fragment: {frag.dname}")
        print(f"{'=' * 80}")

        # Basic RDM info
        print(f"\n--- RDM Traces ---")
        print(f"RDM1 trace (canonical): {np.trace(rdm1_canonical):.6f}")
        print(f"RDM1 trace (original):  {np.trace(rdm1):.6f}")
        print(f"Expected (nelec):       {nelec}")
        print(f"Trace error:            {abs(np.trace(rdm1) - nelec):.2e}")

        # Diagonal occupations
        occ_canonical = np.diag(rdm1_canonical)
        occ_original = np.diag(rdm1)
        print(f"\n--- Diagonal Occupations ---")
        print(f"Canonical MO basis: {np.round(occ_canonical, 6)}")
        print(f"Fragment MO basis:  {np.round(occ_original, 6)}")

        # Integer-rounded occupations
        int_occ_canonical = [int(round(x)) for x in occ_canonical]
        int_occ_original = [int(round(x)) for x in occ_original]
        print(f"\n--- Integer Occupations ---")
        print(f"VQE Canonical: {int_occ_canonical}")
        print(f"VQE Original:  {int_occ_original}")

        # Natural occupations (basis-invariant)
        nat_occ_canonical = np.linalg.eigvalsh(rdm1_canonical)
        nat_occ_original = np.linalg.eigvalsh(rdm1)
        nat_occ_canonical_sorted = np.sort(nat_occ_canonical)[::-1]
        nat_occ_original_sorted = np.sort(nat_occ_original)[::-1]
        print(f"\n--- Natural Occupations (basis-invariant) ---")
        print(f"VQE (canonical): {np.round(nat_occ_canonical_sorted, 6)}")
        print(f"VQE (original):  {np.round(nat_occ_original_sorted, 6)}")

        # VQE energy
        print(f"\n--- VQE Energy ---")
        print(f"VQE fragment energy: {optimal_energy:.8f} Ha")

        print(f"{'=' * 80}\n")
    else:
        # No canonical rotation was applied (shouldn't happen, but handle gracefully)
        rdm1 = rdm1_canonical
        rdm2 = rdm2_canonical
        # NOTE: RDM2 is already in chemist convention from compute_rdms_from_statevector()
        # No transpose needed!
        if vqe_args.verbose >= 1:
            print("WARNING: No canonical_rotation found on fragment. Using RDMs as-is.")

    validate_rdm_pair(rdm1, rdm2, label="VQE fragment-basis")

    timings = best_run["timings"]
    phase_timings["pre_optimization"] = timings["pre_optimization"]
    phase_timings["optimizer"] = timings["optimizer"]
    phase_timings["statevector_and_rdms"] = timings["statevector_and_rdms"]
    phase_timings["total"] = (
        phase_timings.get("load_hamiltonian", 0.0)
        + phase_timings.get("build_ansatz", 0.0)
        + phase_timings.get("initial_parameters", 0.0)
        + phase_timings.get("backend_setup", 0.0)
        + timings["pre_optimization"]
        + timings["optimizer"]
        + timings["statevector_and_rdms"]
    )
    phase_timings["optimizer_iterations"] = best_run["iterations"]

    if vqe_args.warm_start:
        _vqe_state.fragment_params[frag_name] = optimal_params

    if vqe_args.verbose >= 2:
        print(f"  RDM1 trace: {np.trace(rdm1):.6f} (expected: {nelec})")
        print(f"  RDM2 computed successfully")

    # Ensure final record reflects the converged solution
    if callback_needed:
        needs_final_record = True
        if callback_records:
            last_energy = callback_records[-1].get("energy")
            if last_energy is not None and np.isclose(
                last_energy, optimal_energy, atol=1e-12
            ):
                needs_final_record = False
                callback_records[-1]["timings"] = phase_timings.copy()
        if needs_final_record:
            final_record: dict[str, Any] = {
                "eval_count": best_run["iterations"],
                "energy": optimal_energy,
                "raw_energy": float(best_run["raw_energy"]),
                "delta_energy": (
                    optimal_energy - callback_records[-1]["energy"]
                    if callback_records
                    else None
                ),
                "timings": phase_timings.copy(),
                "elapsed_time": phase_timings.get("total"),
                "delta_time": None,
            }
            if vqe_args.track_density_matrices:
                final_record["rdm1_trace"] = float(np.trace(rdm1))
                prev_matrix = None
                if callback_records:
                    prev_matrix = callback_records[-1].get("rdm1_matrix")
                if prev_matrix is not None:
                    final_record["rdm1_delta"] = float(
                        np.linalg.norm(rdm1 - prev_matrix)
                    )
                else:
                    final_record["rdm1_delta"] = None
                final_record["density_time"] = phase_timings.get("statevector_and_rdms")
            else:
                final_record["rdm1_delta"] = None
                final_record["density_time"] = phase_timings.get("statevector_and_rdms")
            final_record["rdm1_matrix"] = rdm1
            callback_records.append(final_record)

    _vqe_state.fragment_timings[frag_name] = phase_timings.copy()

    # Cache diagnostics for downstream inspection
    if vqe_args.track_iteration_history:
        _vqe_state.fragment_iteration_history[frag_name] = [
            {
                key: (
                    value.copy()
                    if isinstance(value, np.ndarray)
                    else value.copy()
                    if isinstance(value, dict)
                    else value
                )
                for key, value in record.items()
            }
            for record in callback_records
        ]
    elif frag_name in _vqe_state.fragment_iteration_history:
        # Drop stale history if tracking disabled for this run
        _vqe_state.fragment_iteration_history.pop(frag_name, None)

    _vqe_state.fragment_energies[frag_name] = optimal_energy
    _vqe_state.fragment_rdm1[frag_name] = rdm1
    _vqe_state.fragment_rdm2[frag_name] = rdm2

    # Invoke registered callbacks for result tracking
    for callback in _vqe_result_callbacks:
        try:
            callback(frag_name, optimal_energy, rdm1, rdm2)
        except Exception as e:
            if vqe_args.verbose >= 1:
                print(f"  Warning: VQE callback failed: {e}")

    return rdm1, rdm2


def get_vqe_iteration_history(
    frag_name: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve stored VQE iteration history.

    Parameters
    ----------
    frag_name : str | None
        Specific fragment name. If None, return history for all fragments.

    Returns
    -------
    dict
        Mapping of fragment name to list of iteration records.
    """
    if frag_name is not None:
        history = _vqe_state.fragment_iteration_history.get(frag_name, [])
        return {frag_name: deepcopy(history)}

    return {
        key: deepcopy(val) for key, val in _vqe_state.fragment_iteration_history.items()
    }


def get_vqe_fragment_observables(
    frag_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve converged VQE observables for fragments.

    Parameters
    ----------
    frag_name : str | None
        Specific fragment name. If None, return data for all fragments.

    Returns
    -------
    dict
        Mapping of fragment name to observables (energy, RDM1, RDM2).
    """

    def _build_payload(name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name in _vqe_state.fragment_energies:
            payload["energy"] = _vqe_state.fragment_energies[name]
        if name in _vqe_state.fragment_rdm1:
            payload["rdm1"] = _vqe_state.fragment_rdm1[name].copy()
        if name in _vqe_state.fragment_rdm2:
            payload["rdm2"] = _vqe_state.fragment_rdm2[name].copy()
        if name in _vqe_state.fragment_timings:
            payload["timings"] = _vqe_state.fragment_timings[name].copy()
        return payload

    if frag_name is not None:
        return {frag_name: _build_payload(frag_name)}

    return {name: _build_payload(name) for name in _vqe_state.fragment_energies}
