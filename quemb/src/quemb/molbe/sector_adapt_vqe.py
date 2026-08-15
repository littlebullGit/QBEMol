"""Exact sparse-support CEO primitives in a fixed spin determinant sector.

The accepted CEO manifests use qubit excitations with Jordan--Wigner Z strings
removed.  This module therefore implements their local qubit action directly;
it never substitutes fermionic excitation signs and never allocates an array of
length ``2 ** n_qubits``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar


ALGEBRA_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class SectorSelectorConfig:
    """Deterministic selector policy for fixed-sector CEO-ADAPT."""

    policy: str = "greedy_gradient"
    top_k: int = 5
    probe_max_iterations: int = 200
    probe_ftol: float = 1.0e-10
    energy_tie_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.policy not in {"greedy_gradient", "always_top5_energy"}:
            raise ValueError(
                "fixed-sector selector policy must be greedy_gradient or "
                "always_top5_energy"
            )
        if self.top_k < 1:
            raise ValueError("fixed-sector selector top_k must be positive")
        if self.probe_max_iterations < 1:
            raise ValueError(
                "fixed-sector probe iteration limit must be positive"
            )
        if self.probe_ftol <= 0.0 or not math.isfinite(self.probe_ftol):
            raise ValueError("fixed-sector probe ftol must be finite and positive")
        if (
            self.energy_tie_tolerance < 0.0
            or not math.isfinite(self.energy_tie_tolerance)
        ):
            raise ValueError(
                "fixed-sector selector energy tie tolerance must be finite "
                "and non-negative"
            )

    def evidence(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "top_k": self.top_k,
            "probe_max_iterations": self.probe_max_iterations,
            "probe_ftol": self.probe_ftol,
            "energy_tie_tolerance": self.energy_tie_tolerance,
            "serial_probes": True,
        }


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def interleaved_to_blocked(index: int, n_qubits: int) -> int:
    if n_qubits <= 0 or n_qubits % 2:
        raise ValueError("n_qubits must be positive and even")
    if index < 0 or index >= n_qubits:
        raise ValueError("spin-orbital index is out of range")
    n_spatial = n_qubits // 2
    return index // 2 if index % 2 == 0 else n_spatial + index // 2


@dataclass(frozen=True)
class CEOConstituent:
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not self.sources or len(self.sources) != len(self.targets):
            raise ValueError("CEO constituent source/target ranks must agree")
        if len(set(self.sources + self.targets)) != 2 * len(self.sources):
            raise ValueError("CEO constituent orbitals must be disjoint")
        if not math.isfinite(self.coefficient) or abs(self.coefficient) != 1.0:
            raise ValueError("CEO constituent coefficient must be +1 or -1")

    def payload(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "targets": list(self.targets),
            "coefficient": self.coefficient,
        }


@dataclass(frozen=True)
class CompactCEOOperator:
    index: int
    ceo_type: str
    constituents: tuple[CEOConstituent, ...]
    support: tuple[int, ...]
    n_qubits: int
    source_constituents_interleaved: tuple[tuple[int, ...], ...] = ()
    target_constituents_interleaved: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("CEO index must be nonnegative")
        if self.ceo_type not in {"single", "sum", "diff"}:
            raise ValueError("CEO type must be single, sum, or diff")
        expected = 1 if self.ceo_type == "single" else 2
        if len(self.constituents) != expected:
            raise ValueError("CEO constituent count does not match its type")
        if self.n_qubits <= 0 or self.n_qubits % 2:
            raise ValueError("n_qubits must be positive and even")
        if tuple(sorted(set(self.support))) != self.support:
            raise ValueError("CEO support must be unique and sorted")
        union = tuple(
            sorted(
                {
                    orbital
                    for constituent in self.constituents
                    for orbital in constituent.sources + constituent.targets
                }
            )
        )
        if union != self.support or any(
            orbital < 0 or orbital >= self.n_qubits for orbital in self.support
        ):
            raise ValueError("CEO support does not match its constituents")
        n_spatial = self.n_qubits // 2
        for constituent in self.constituents:
            source_alpha = sum(item < n_spatial for item in constituent.sources)
            target_alpha = sum(item < n_spatial for item in constituent.targets)
            if source_alpha != target_alpha:
                raise ValueError("CEO constituent does not preserve spin counts")

    def payload(self) -> dict[str, Any]:
        body = {
            "index": self.index,
            "ceo_type": self.ceo_type,
            "constituents": [item.payload() for item in self.constituents],
            "support": list(self.support),
            "n_qubits": self.n_qubits,
            "source_constituents_interleaved": [
                list(item) for item in self.source_constituents_interleaved
            ],
            "target_constituents_interleaved": [
                list(item) for item in self.target_constituents_interleaved
            ],
        }
        return {**body, "digest": _canonical_digest(body)}


@dataclass(frozen=True)
class SparseSectorState:
    n_spatial: int
    n_alpha: int
    n_beta: int
    amplitudes: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if self.n_spatial <= 0 or self.n_spatial > 31:
            raise ValueError("n_spatial must be in [1, 31]")
        if not 0 <= self.n_alpha <= self.n_spatial:
            raise ValueError("invalid alpha electron count")
        if not 0 <= self.n_beta <= self.n_spatial:
            raise ValueError("invalid beta electron count")
        if not self.amplitudes:
            raise ValueError("sector state cannot be empty")
        keys = [item[0] for item in self.amplitudes]
        if keys != sorted(set(keys)):
            raise ValueError("sector state keys must be sorted and unique")
        limit = 1 << (2 * self.n_spatial)
        alpha_mask = (1 << self.n_spatial) - 1
        for bitstring, amplitude in self.amplitudes:
            if bitstring < 0 or bitstring >= limit:
                raise ValueError("sector bitstring is out of range")
            if (bitstring & alpha_mask).bit_count() != self.n_alpha:
                raise ValueError("sector state has the wrong alpha count")
            if (bitstring >> self.n_spatial).bit_count() != self.n_beta:
                raise ValueError("sector state has the wrong beta count")
            if not math.isfinite(amplitude) or abs(amplitude) <= 0.0:
                raise ValueError("sector amplitudes must be finite and nonzero")

    @classmethod
    def from_mapping(
        cls,
        n_spatial: int,
        n_alpha: int,
        n_beta: int,
        amplitudes: Mapping[int, float],
        *,
        drop_tolerance: float = 1.0e-15,
    ) -> "SparseSectorState":
        cleaned = tuple(
            sorted(
                (int(bitstring), float(amplitude))
                for bitstring, amplitude in amplitudes.items()
                if abs(float(amplitude)) > drop_tolerance
            )
        )
        return cls(n_spatial, n_alpha, n_beta, cleaned)

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_spatial

    @property
    def support_size(self) -> int:
        return len(self.amplitudes)

    def as_dict(self) -> dict[int, float]:
        return dict(self.amplitudes)

    def norm_squared(self) -> float:
        return math.fsum(amplitude * amplitude for _, amplitude in self.amplitudes)

    def normalized(self) -> "SparseSectorState":
        norm = math.sqrt(self.norm_squared())
        if norm <= 0.0:
            raise ValueError("cannot normalize a zero state")
        return self.from_mapping(
            self.n_spatial,
            self.n_alpha,
            self.n_beta,
            {bitstring: amplitude / norm for bitstring, amplitude in self.amplitudes},
        )


def hartree_fock_sector_state(
    n_spatial: int, n_alpha: int, n_beta: int
) -> SparseSectorState:
    alpha = (1 << n_alpha) - 1
    beta = ((1 << n_beta) - 1) << n_spatial
    return SparseSectorState.from_mapping(
        n_spatial, n_alpha, n_beta, {alpha | beta: 1.0}
    )


def sector_inner_product(left: SparseSectorState, right: SparseSectorState) -> float:
    if (
        left.n_spatial,
        left.n_alpha,
        left.n_beta,
    ) != (right.n_spatial, right.n_alpha, right.n_beta):
        raise ValueError("sector states are incompatible")
    right_values = right.as_dict()
    return math.fsum(
        amplitude * right_values.get(bitstring, 0.0)
        for bitstring, amplitude in left.amplitudes
    )


def _descriptor_action_on_bit(
    bitstring: int, operator: CompactCEOOperator
) -> dict[int, float]:
    result: dict[int, float] = {}
    for constituent in operator.constituents:
        source_mask = sum(1 << item for item in constituent.sources)
        target_mask = sum(1 << item for item in constituent.targets)
        if bitstring & source_mask == source_mask and bitstring & target_mask == 0:
            target = bitstring ^ source_mask ^ target_mask
            result[target] = result.get(target, 0.0) + constituent.coefficient
        elif bitstring & source_mask == 0 and bitstring & target_mask == target_mask:
            target = bitstring ^ source_mask ^ target_mask
            result[target] = result.get(target, 0.0) - constituent.coefficient
    return {key: value for key, value in result.items() if abs(value) > 0.0}


def apply_ceo_generator(
    state: SparseSectorState, operator: CompactCEOOperator
) -> SparseSectorState:
    if state.n_qubits != operator.n_qubits:
        raise ValueError("CEO operator and state qubit counts differ")
    result: dict[int, float] = {}
    for bitstring, amplitude in state.amplitudes:
        for target, coefficient in _descriptor_action_on_bit(
            bitstring, operator
        ).items():
            result[target] = result.get(target, 0.0) + coefficient * amplitude
    if not result:
        # A mathematically zero generator action is represented outside the
        # public nonempty-state type.
        raise ValueError("CEO generator annihilates the state")
    return SparseSectorState.from_mapping(
        state.n_spatial, state.n_alpha, state.n_beta, result
    )


def compact_local_k(operator: CompactCEOOperator) -> np.ndarray:
    support = operator.support
    position = {orbital: local for local, orbital in enumerate(support)}
    size = 1 << len(support)
    matrix = np.zeros((size, size), dtype=np.float64)
    for column in range(size):
        global_bits = sum(
            ((column >> local) & 1) << orbital
            for local, orbital in enumerate(support)
        )
        for target, coefficient in _descriptor_action_on_bit(
            global_bits, operator
        ).items():
            row = sum(
                ((target >> orbital) & 1) << position[orbital]
                for orbital in support
            )
            matrix[row, column] += coefficient
    if np.max(np.abs(matrix + matrix.T), initial=0.0) > ALGEBRA_TOLERANCE:
        raise RuntimeError("compact CEO local matrix is not skew-symmetric")
    return matrix


def _pauli_action_on_basis(
    label: str, coefficient: complex, bitstring: int
) -> tuple[int, complex]:
    target = bitstring
    amplitude = complex(coefficient)
    for qubit, pauli in enumerate(reversed(label)):
        occupied = (bitstring >> qubit) & 1
        if pauli == "X":
            target ^= 1 << qubit
        elif pauli == "Y":
            target ^= 1 << qubit
            amplitude *= 1j if occupied == 0 else -1j
        elif pauli == "Z":
            if occupied:
                amplitude *= -1.0
        elif pauli != "I":
            raise ValueError(f"unsupported Pauli label {pauli!r}")
    return target, amplitude


def manifest_local_k(manifest_operator: Any) -> np.ndarray:
    support = tuple(manifest_operator.blocked_support_orbitals)
    position = {orbital: local for local, orbital in enumerate(support)}
    size = 1 << len(support)
    matrix = np.zeros((size, size), dtype=np.complex128)
    pauli = manifest_operator.anti_hermitian
    for column in range(size):
        global_bits = sum(
            ((column >> local) & 1) << orbital
            for local, orbital in enumerate(support)
        )
        for label, coefficient in zip(pauli.paulis.to_labels(), pauli.coeffs):
            target, amplitude = _pauli_action_on_basis(
                label, complex(coefficient), global_bits
            )
            row = sum(
                ((target >> orbital) & 1) << position[orbital]
                for orbital in support
            )
            matrix[row, column] += amplitude
    if np.max(np.abs(matrix.imag), initial=0.0) > ALGEBRA_TOLERANCE:
        raise RuntimeError("manifest CEO local matrix is not real")
    real = matrix.real
    if np.max(np.abs(real + real.T), initial=0.0) > ALGEBRA_TOLERANCE:
        raise RuntimeError("manifest CEO local matrix is not skew-symmetric")
    return real


def descriptor_from_manifest(manifest_operator: Any) -> CompactCEOOperator:
    ceo_type = str(manifest_operator.ceo_type)
    sources = tuple(tuple(group) for group in manifest_operator.blocked_source_orbitals)
    targets = tuple(tuple(group) for group in manifest_operator.blocked_target_orbitals)
    # The two source qubit excitations share Pauli support. Their sum/difference
    # cancels half the words before L1 normalization, so each surviving local
    # transition retains unit magnitude. Source normal ordering gives both
    # double-CEO families an overall minus sign relative to their orbital
    # metadata; the accepted manifests and source attestations fix that sign.
    coefficients = (1.0,) if ceo_type == "single" else (
        (-1.0, -1.0) if ceo_type == "sum" else (-1.0, 1.0)
    )
    constituents = tuple(
        CEOConstituent(source, target, coefficient)
        for source, target, coefficient in zip(sources, targets, coefficients)
    )
    return CompactCEOOperator(
        index=int(manifest_operator.index),
        ceo_type=ceo_type,
        constituents=constituents,
        support=tuple(manifest_operator.blocked_support_orbitals),
        n_qubits=int(manifest_operator.n_qubits),
        source_constituents_interleaved=tuple(
            tuple(group) for group in manifest_operator.source_orbitals
        ),
        target_constituents_interleaved=tuple(
            tuple(group) for group in manifest_operator.target_orbitals
        ),
    )


def _make_descriptor(
    index: int,
    ceo_type: str,
    sources_interleaved: Sequence[Sequence[int]],
    targets_interleaved: Sequence[Sequence[int]],
    n_qubits: int,
) -> CompactCEOOperator:
    sources_blocked = tuple(
        tuple(interleaved_to_blocked(item, n_qubits) for item in group)
        for group in sources_interleaved
    )
    targets_blocked = tuple(
        tuple(interleaved_to_blocked(item, n_qubits) for item in group)
        for group in targets_interleaved
    )
    coefficients = (1.0,) if ceo_type == "single" else (
        (-1.0, -1.0) if ceo_type == "sum" else (-1.0, 1.0)
    )
    constituents = tuple(
        CEOConstituent(source, target, coefficient)
        for source, target, coefficient in zip(
            sources_blocked, targets_blocked, coefficients
        )
    )
    support = tuple(
        sorted(
            {
                item
                for constituent in constituents
                for item in constituent.sources + constituent.targets
            }
        )
    )
    return CompactCEOOperator(
        index,
        ceo_type,
        constituents,
        support,
        n_qubits,
        tuple(tuple(item) for item in sources_interleaved),
        tuple(tuple(item) for item in targets_interleaved),
    )


def generate_compact_ovp_ceo_pool(n_qubits: int) -> tuple[CompactCEOOperator, ...]:
    """Reproduce source ``OVP_CEO(sum=True,diff=True,fermionic_swaps=False)`` order."""

    if n_qubits <= 0 or n_qubits % 2:
        raise ValueError("n_qubits must be positive and even")
    result: list[CompactCEOOperator] = []
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            if (p + q) % 2 == 0:
                result.append(
                    _make_descriptor(
                        len(result), "single", [[q]], [[p]], n_qubits
                    )
                )
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            for r in range(q + 1, n_qubits):
                for s in range(r + 1, n_qubits):
                    if (p + q + r + s) % 2 != 0:
                        continue
                    pairs: list[tuple[list[list[int]], list[list[int]]]] = []
                    if (p + r) % 2 == 0:
                        pairs.append(([[r, s], [p, s]], [[p, q], [q, r]]))
                    if (p + q) % 2 == 0:
                        pairs.append(([[q, s], [p, s]], [[p, r], [q, r]]))
                    if (p + s) % 2 == 0:
                        pairs.append(([[r, s], [q, s]], [[p, q], [p, r]]))
                    for sources, targets in pairs:
                        result.append(
                            _make_descriptor(
                                len(result), "sum", sources, targets, n_qubits
                            )
                        )
                        result.append(
                            _make_descriptor(
                                len(result), "diff", sources, targets, n_qubits
                            )
                        )
    return tuple(result)


def validate_compact_pool_against_manifest(
    compact_pool: Sequence[CompactCEOOperator],
    manifest: Any,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if len(compact_pool) != len(manifest.operators):
        raise ValueError("compact and manifest pool sizes differ")
    maximum_residual = 0.0
    descriptors = []
    total = len(compact_pool)
    for index, (compact, manifest_operator) in enumerate(
        zip(compact_pool, manifest.operators)
    ):
        observed = descriptor_from_manifest(manifest_operator)
        if compact.payload() != observed.payload():
            raise ValueError(f"CEO descriptor mismatch at index {compact.index}")
        residual = float(
            np.max(
                np.abs(compact_local_k(compact) - manifest_local_k(manifest_operator)),
                initial=0.0,
            )
        )
        maximum_residual = max(maximum_residual, residual)
        if residual > ALGEBRA_TOLERANCE:
            raise ValueError(
                f"CEO local algebra mismatch at index {compact.index}: {residual}"
            )
        descriptors.append(compact.payload())
        if progress_callback is not None:
            progress_callback(index + 1, total)
    return {
        "operator_count": len(compact_pool),
        "descriptor_stream_digest": _canonical_digest(descriptors),
        "maximum_local_k_residual": maximum_residual,
    }


def apply_sparse_pauli(
    operator: SparsePauliOp,
    state: SparseSectorState,
    *,
    cancellation_tolerance: float = 1.0e-12,
) -> SparseSectorState:
    if operator.num_qubits != state.n_qubits:
        raise ValueError("Hamiltonian and sector state qubit counts differ")
    result: dict[int, complex] = {}
    labels = operator.paulis.to_labels()
    for bitstring, state_amplitude in state.amplitudes:
        for label, coefficient in zip(labels, operator.coeffs):
            target, amplitude = _pauli_action_on_basis(
                label, complex(coefficient), bitstring
            )
            result[target] = result.get(target, 0.0j) + amplitude * state_amplitude
    real_result: dict[int, float] = {}
    alpha_mask = (1 << state.n_spatial) - 1
    for bitstring, amplitude in result.items():
        if abs(amplitude) <= cancellation_tolerance:
            continue
        if abs(amplitude.imag) > cancellation_tolerance:
            raise RuntimeError("real Hamiltonian produced an imaginary sector amplitude")
        if (bitstring & alpha_mask).bit_count() != state.n_alpha or (
            bitstring >> state.n_spatial
        ).bit_count() != state.n_beta:
            raise RuntimeError("Hamiltonian action leaked outside the fixed sector")
        real_result[bitstring] = float(amplitude.real)
    return SparseSectorState.from_mapping(
        state.n_spatial, state.n_alpha, state.n_beta, real_result
    )


def sparse_pauli_basis_diagonal(operator: SparsePauliOp, bitstring: int) -> float:
    value = 0.0j
    for label, coefficient in zip(operator.paulis.to_labels(), operator.coeffs):
        target, amplitude = _pauli_action_on_basis(
            label, complex(coefficient), bitstring
        )
        if target == bitstring:
            value += amplitude
    if abs(value.imag) > ALGEBRA_TOLERANCE:
        raise RuntimeError("basis diagonal is not real")
    return float(value.real)


def sector_energy(operator: SparsePauliOp, state: SparseSectorState) -> float:
    return sector_inner_product(state, apply_sparse_pauli(operator, state))


def rotate_ceo(
    state: SparseSectorState, operator: CompactCEOOperator, theta: float
) -> SparseSectorState:
    if not math.isfinite(theta):
        raise ValueError("rotation angle must be finite")
    if state.n_qubits != operator.n_qubits:
        raise ValueError("CEO operator and state qubit counts differ")
    support = operator.support
    local_position = {orbital: position for position, orbital in enumerate(support)}
    outside_mask = ((1 << state.n_qubits) - 1) ^ sum(1 << item for item in support)
    grouped: dict[int, np.ndarray] = {}
    for bitstring, amplitude in state.amplitudes:
        outside = bitstring & outside_mask
        local = sum(
            ((bitstring >> orbital) & 1) << local_position[orbital]
            for orbital in support
        )
        vector = grouped.setdefault(outside, np.zeros(1 << len(support)))
        vector[local] += amplitude
    unitary = expm(theta * compact_local_k(operator))
    result: dict[int, float] = {}
    for outside, vector in grouped.items():
        rotated = unitary @ vector
        for local, amplitude in enumerate(rotated):
            if abs(float(amplitude)) <= 1.0e-15:
                continue
            bitstring = outside | sum(
                ((local >> position) & 1) << orbital
                for position, orbital in enumerate(support)
            )
            result[bitstring] = result.get(bitstring, 0.0) + float(amplitude)
    return SparseSectorState.from_mapping(
        state.n_spatial, state.n_alpha, state.n_beta, result
    )


def all_pool_gradients(
    hamiltonian: SparsePauliOp,
    state: SparseSectorState,
    pool: Sequence[CompactCEOOperator],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    h_state = apply_sparse_pauli(hamiltonian, state)
    gradients = np.zeros(len(pool), dtype=np.float64)
    total = len(pool)
    for index, operator in enumerate(pool):
        action: dict[int, float] = {}
        for bitstring, amplitude in state.amplitudes:
            for target, coefficient in _descriptor_action_on_bit(
                bitstring, operator
            ).items():
                action[target] = action.get(target, 0.0) + coefficient * amplitude
        if action:
            k_state = SparseSectorState.from_mapping(
                state.n_spatial, state.n_alpha, state.n_beta, action
            )
            gradients[index] = 2.0 * sector_inner_product(h_state, k_state)
        if progress_callback is not None:
            progress_callback(index + 1, total)
    return gradients


def deterministic_gradient_ranking(
    gradients: Sequence[float], *, tie_tolerance: float = 1.0e-12
) -> tuple[int, ...]:
    values = [abs(float(value)) for value in gradients]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("gradients must be finite")
    # Quantized tie keys make the index rule explicit and reproducible.
    return tuple(
        sorted(
            range(len(values)),
            key=lambda index: (
                -round(values[index] / tie_tolerance) if tie_tolerance > 0 else -values[index],
                index,
            ),
        )
    )


@dataclass(frozen=True)
class OneParameterProbe:
    operator_index: int
    theta: float
    energy: float
    state: SparseSectorState


@dataclass(frozen=True)
class SectorAdaptResult:
    """Terminal result of a determinant-sector CEO-ADAPT optimization."""

    energy: float
    state: SparseSectorState
    selected_indices: tuple[int, ...]
    parameters: tuple[float, ...]
    gradient_history: tuple[dict[str, Any], ...]
    energy_history: tuple[float, ...]
    termination: str
    optimizer_success: bool
    optimizer_evaluations: int
    optimizer_iterations: int
    selector_config: SectorSelectorConfig
    selector_events: tuple[dict[str, Any], ...]

    def evidence(self) -> dict[str, Any]:
        """Return the JSON-safe scientific record, excluding amplitudes."""

        return {
            "schema": "quemb.fixed-sector-ceo-adapt/v1",
            "energy_ha": self.energy,
            "selected_indices": list(self.selected_indices),
            "parameters": list(self.parameters),
            "gradient_history": list(self.gradient_history),
            "energy_history_ha": list(self.energy_history),
            "termination": self.termination,
            "optimizer_success": self.optimizer_success,
            "optimizer_evaluations": self.optimizer_evaluations,
            "optimizer_iterations": self.optimizer_iterations,
            "selector_config": self.selector_config.evidence(),
            "selector_events": list(self.selector_events),
            "n_spatial": self.state.n_spatial,
            "n_alpha": self.state.n_alpha,
            "n_beta": self.state.n_beta,
            "determinant_dimension": fixed_sector_dimension(
                self.state.n_spatial,
                self.state.n_alpha,
                self.state.n_beta,
            ),
            "state_support_size": self.state.support_size,
            "state_norm_squared": self.state.norm_squared(),
            "full_statevector_allocated": False,
        }


def optimize_one_parameter(
    hamiltonian: SparsePauliOp,
    state: SparseSectorState,
    operator: CompactCEOOperator,
) -> OneParameterProbe:
    def objective(theta: float) -> float:
        return sector_energy(hamiltonian, rotate_ceo(state, operator, theta))

    result = minimize_scalar(
        objective,
        bounds=(-math.pi, math.pi),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 256},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("one-parameter CEO optimization failed")
    theta = float(result.x)
    optimized = rotate_ceo(state, operator, theta)
    energy = sector_energy(hamiltonian, optimized)
    return OneParameterProbe(operator.index, theta, energy, optimized)


def apply_ceo_ansatz(
    reference: SparseSectorState,
    pool: Sequence[CompactCEOOperator],
    selected_indices: Sequence[int],
    parameters: Sequence[float],
) -> SparseSectorState:
    """Apply an ordered product of exact compact CEO rotations."""

    if len(selected_indices) != len(parameters):
        raise ValueError("selected CEO indices and parameters must have equal length")
    state = reference
    for raw_index, raw_parameter in zip(selected_indices, parameters):
        index = int(raw_index)
        if index < 0 or index >= len(pool):
            raise ValueError(f"selected CEO index {index} is outside the pool")
        state = rotate_ceo(state, pool[index], float(raw_parameter))
    return state


def _sector_string_addresses(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> tuple[dict[int, int], dict[int, int]]:
    from pyscf.fci import cistring

    orbitals = range(n_spatial)
    alpha_strings = cistring.make_strings(orbitals, n_alpha)
    beta_strings = cistring.make_strings(orbitals, n_beta)
    return (
        {int(bitstring): address for address, bitstring in enumerate(alpha_strings)},
        {int(bitstring): address for address, bitstring in enumerate(beta_strings)},
    )


def all_pool_gradients_from_pyscf_action(
    state: SparseSectorState,
    h_ci: np.ndarray,
    pool: Sequence[CompactCEOOperator],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Evaluate ``dE/dtheta`` for every CEO using one fixed-sector ``H|psi>``.

    This is algebraically the same commutator screen as
    :func:`all_pool_gradients`, but it consumes PySCF's determinant-sector
    Hamiltonian action.  Its largest dense object therefore has
    ``comb(n, n_alpha) * comb(n, n_beta)`` entries, never ``2 ** (2*n)``.
    """

    expected_shape = (
        math.comb(state.n_spatial, state.n_alpha),
        math.comb(state.n_spatial, state.n_beta),
    )
    h_values = np.asarray(h_ci, dtype=np.float64)
    if h_values.shape != expected_shape:
        raise ValueError(
            f"H|psi> has shape {h_values.shape}; expected {expected_shape}"
        )
    alpha_addresses, beta_addresses = _sector_string_addresses(
        state.n_spatial,
        state.n_alpha,
        state.n_beta,
    )
    alpha_mask = (1 << state.n_spatial) - 1
    gradients = np.zeros(len(pool), dtype=np.float64)
    total = len(pool)
    for pool_position, operator in enumerate(pool):
        inner_product = 0.0
        for bitstring, amplitude in state.amplitudes:
            for target, coefficient in _descriptor_action_on_bit(
                bitstring, operator
            ).items():
                alpha_string = target & alpha_mask
                beta_string = target >> state.n_spatial
                inner_product += (
                    coefficient
                    * amplitude
                    * h_values[
                        alpha_addresses[alpha_string],
                        beta_addresses[beta_string],
                    ]
                )
        gradients[pool_position] = 2.0 * inner_product
        if progress_callback is not None:
            progress_callback(pool_position + 1, total)
    return gradients


def _ansatz_state_and_derivatives(
    reference: SparseSectorState,
    pool: Sequence[CompactCEOOperator],
    selected_indices: Sequence[int],
    parameters: Sequence[float],
) -> tuple[SparseSectorState, tuple[SparseSectorState | None, ...]]:
    """Return the ansatz state and exact derivatives for all parameters."""

    if len(selected_indices) != len(parameters):
        raise ValueError("selected CEO indices and parameters must have equal length")
    forward_states: list[SparseSectorState] = []
    state = reference
    for raw_index, raw_parameter in zip(selected_indices, parameters):
        index = int(raw_index)
        if index < 0 or index >= len(pool):
            raise ValueError(f"selected CEO index {index} is outside the pool")
        state = rotate_ceo(state, pool[index], float(raw_parameter))
        forward_states.append(state)

    derivatives: list[SparseSectorState | None] = []
    for parameter_index, selected_index in enumerate(selected_indices):
        try:
            derivative = apply_ceo_generator(
                forward_states[parameter_index],
                pool[int(selected_index)],
            )
        except ValueError as exc:
            if "annihilates" not in str(exc):
                raise
            derivatives.append(None)
            continue
        for later_index in range(parameter_index + 1, len(selected_indices)):
            derivative = rotate_ceo(
                derivative,
                pool[int(selected_indices[later_index])],
                float(parameters[later_index]),
            )
        derivatives.append(derivative)
    return state, tuple(derivatives)


class _SectorObjective:
    """Cached exact energy/Jacobian evaluator for SciPy SLSQP."""

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        reference: SparseSectorState,
        pool: Sequence[CompactCEOOperator],
        selected_indices: Sequence[int],
    ) -> None:
        self.h1 = np.asarray(h1, dtype=np.float64)
        self.h2 = np.asarray(h2, dtype=np.float64)
        self.reference = reference
        self.pool = tuple(pool)
        self.selected_indices = tuple(int(item) for item in selected_indices)
        self._parameters: np.ndarray | None = None
        self._energy: float | None = None
        self._jacobian: np.ndarray | None = None
        self._state: SparseSectorState | None = None

    def _evaluate(self, parameters: Sequence[float]) -> None:
        values = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if (
            self._parameters is not None
            and values.shape == self._parameters.shape
            and np.array_equal(values, self._parameters)
        ):
            return
        state, derivatives = _ansatz_state_and_derivatives(
            self.reference,
            self.pool,
            self.selected_indices,
            values,
        )
        norm_residual = abs(state.norm_squared() - 1.0)
        if norm_residual > 1.0e-10:
            raise RuntimeError(
                f"CEO ansatz norm residual {norm_residual:.3e} exceeds 1e-10"
            )
        ci, h_ci = pyscf_fixed_sector_hamiltonian_action(self.h1, self.h2, state)
        energy = float(np.vdot(ci, h_ci).real)
        jacobian = np.zeros(len(derivatives), dtype=np.float64)
        for index, derivative in enumerate(derivatives):
            if derivative is None:
                continue
            derivative_ci = sector_state_to_pyscf_ci(derivative)
            jacobian[index] = 2.0 * float(np.vdot(h_ci, derivative_ci).real)
        if not math.isfinite(energy) or not np.all(np.isfinite(jacobian)):
            raise RuntimeError("fixed-sector objective produced a non-finite value")
        self._parameters = values.copy()
        self._energy = energy
        self._jacobian = jacobian
        self._state = state

    def energy(self, parameters: Sequence[float]) -> float:
        self._evaluate(parameters)
        assert self._energy is not None
        return self._energy

    def jacobian(self, parameters: Sequence[float]) -> np.ndarray:
        self._evaluate(parameters)
        assert self._jacobian is not None
        return self._jacobian.copy()

    def state(self, parameters: Sequence[float]) -> SparseSectorState:
        self._evaluate(parameters)
        assert self._state is not None
        return self._state


def sector_ansatz_energy_and_gradient(
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    n_alpha: int,
    n_beta: int,
    pool: Sequence[CompactCEOOperator],
    selected_indices: Sequence[int],
    parameters: Sequence[float],
) -> tuple[float, np.ndarray, SparseSectorState]:
    """Evaluate one fixed-sector CEO ansatz with its exact parameter gradient."""

    h1_values = np.asarray(h1, dtype=np.float64)
    if h1_values.ndim != 2 or h1_values.shape[0] != h1_values.shape[1]:
        raise ValueError("h1 must be square")
    reference = hartree_fock_sector_state(
        int(h1_values.shape[0]), int(n_alpha), int(n_beta)
    )
    objective = _SectorObjective(
        h1_values,
        np.asarray(h2, dtype=np.float64),
        reference,
        pool,
        selected_indices,
    )
    return (
        objective.energy(parameters),
        objective.jacobian(parameters),
        objective.state(parameters),
    )


def solve_sector_adapt(
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    n_alpha: int,
    n_beta: int,
    pool: Sequence[CompactCEOOperator],
    gradient_threshold: float = 1.0e-3,
    eigenvalue_threshold: float = 0.0,
    max_adapt_iterations: int = 20,
    optimizer_max_iterations: int = 200,
    optimizer_ftol: float = 1.0e-10,
    gradient_log_top_k: int = 5,
    selector_config: SectorSelectorConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> SectorAdaptResult:
    """Run deterministic CEO-ADAPT entirely in a fixed spin sector."""

    h1_values = np.asarray(h1, dtype=np.float64)
    h2_values = np.asarray(h2, dtype=np.float64)
    if h1_values.ndim != 2 or h1_values.shape[0] != h1_values.shape[1]:
        raise ValueError("h1 must be square")
    n_spatial = int(h1_values.shape[0])
    if h2_values.shape != (n_spatial,) * 4:
        raise ValueError("h2 shape does not match h1")
    if not pool:
        raise ValueError("CEO pool cannot be empty")
    if any(
        operator.index != position or operator.n_qubits != 2 * n_spatial
        for position, operator in enumerate(pool)
    ):
        raise ValueError("CEO pool must be complete, ordered, and dimension matched")
    if gradient_threshold < 0.0 or not math.isfinite(gradient_threshold):
        raise ValueError("gradient_threshold must be finite and non-negative")
    if eigenvalue_threshold < 0.0 or not math.isfinite(eigenvalue_threshold):
        raise ValueError("eigenvalue_threshold must be finite and non-negative")
    if max_adapt_iterations < 1 or optimizer_max_iterations < 1:
        raise ValueError("iteration limits must be positive")
    if optimizer_ftol <= 0.0 or not math.isfinite(optimizer_ftol):
        raise ValueError("optimizer_ftol must be finite and positive")
    selector = (
        SectorSelectorConfig() if selector_config is None else selector_config
    )

    reference = hartree_fock_sector_state(n_spatial, n_alpha, n_beta)
    state = reference
    ci, h_ci = pyscf_fixed_sector_hamiltonian_action(h1_values, h2_values, state)
    energy = float(np.vdot(ci, h_ci).real)
    selected_indices: list[int] = []
    parameters = np.zeros(0, dtype=np.float64)
    gradient_history: list[dict[str, Any]] = []
    selector_events: list[dict[str, Any]] = []
    energy_history = [energy]
    optimizer_success = True
    optimizer_evaluations = 0
    optimizer_iterations = 0
    termination = "operator_cap"

    for adapt_iteration in range(max_adapt_iterations):
        gradients = all_pool_gradients_from_pyscf_action(
            state,
            h_ci,
            pool,
            progress_callback=(
                None
                if progress_callback is None
                else lambda completed, total, iteration=adapt_iteration: progress_callback(
                    f"gradient_screen_{iteration}", completed, total
                )
            ),
        )
        ranking = deterministic_gradient_ranking(gradients)
        maximum_gradient = abs(float(gradients[ranking[0]]))
        top_count = min(max(gradient_log_top_k, 1), len(ranking))
        record: dict[str, Any] = {
            "adapt_iteration": adapt_iteration,
            "energy_before_optimization_ha": energy,
            "maximum_absolute_gradient": maximum_gradient,
            "top_gradients": [
                {
                    "index": int(index),
                    "gradient": float(gradients[index]),
                    "absolute_gradient": abs(float(gradients[index])),
                }
                for index in ranking[:top_count]
            ],
        }
        if maximum_gradient < gradient_threshold:
            record["selected_index"] = None
            gradient_history.append(record)
            termination = "gradient_threshold"
            break

        selected_index = int(ranking[0])
        selected_initial_parameters = np.append(parameters, 0.0)
        if selector.policy == "always_top5_energy":
            shortlist = tuple(
                int(index) for index in ranking[: min(selector.top_k, len(ranking))]
            )
            probe_records: list[dict[str, Any]] = []
            probe_started = perf_counter()
            for probe_position, candidate_index in enumerate(shortlist, start=1):
                candidate_indices = [*selected_indices, candidate_index]
                candidate_objective = _SectorObjective(
                    h1_values,
                    h2_values,
                    reference,
                    pool,
                    candidate_indices,
                )
                candidate_initial = np.append(parameters, 0.0)
                candidate_started = perf_counter()
                optimization = minimize(
                    candidate_objective.energy,
                    candidate_initial,
                    method="SLSQP",
                    jac=candidate_objective.jacobian,
                    options={
                        "maxiter": int(selector.probe_max_iterations),
                        "ftol": float(selector.probe_ftol),
                        "disp": False,
                    },
                )
                finite = bool(
                    np.all(np.isfinite(optimization.x))
                    and math.isfinite(float(optimization.fun))
                )
                probe_records.append(
                    {
                        "candidate_idx": candidate_index,
                        "gradient_rank": probe_position,
                        "gradient": float(gradients[candidate_index]),
                        "absolute_gradient": abs(
                            float(gradients[candidate_index])
                        ),
                        "relaxed_energy": (
                            float(optimization.fun) if finite else None
                        ),
                        "optimal_point": (
                            np.asarray(optimization.x, dtype=float).tolist()
                            if finite
                            else None
                        ),
                        "optimizer_success": bool(optimization.success and finite),
                        "optimizer_status": int(optimization.status),
                        "optimizer_message": str(optimization.message),
                        "optimizer_nfev": int(
                            getattr(optimization, "nfev", 0)
                        ),
                        "optimizer_nit": int(
                            getattr(optimization, "nit", 0)
                        ),
                        "elapsed_s": perf_counter() - candidate_started,
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        f"lookahead_probe_{adapt_iteration}",
                        probe_position,
                        len(shortlist),
                    )
            successful = [
                probe
                for probe in probe_records
                if probe["optimizer_success"]
                and probe["relaxed_energy"] is not None
            ]
            winner: dict[str, Any] | None = None
            if successful:
                minimum_energy = min(
                    float(probe["relaxed_energy"]) for probe in successful
                )
                tied = [
                    probe
                    for probe in successful
                    if float(probe["relaxed_energy"])
                    <= minimum_energy + selector.energy_tie_tolerance
                ]
                winner = min(
                    tied,
                    key=lambda probe: (
                        int(probe["gradient_rank"]),
                        int(probe["candidate_idx"]),
                    ),
                )
                selected_index = int(winner["candidate_idx"])
                selected_initial_parameters = np.asarray(
                    winner["optimal_point"], dtype=np.float64
                )
            event = {
                "iteration": adapt_iteration,
                "policy": selector.policy,
                "selector_config": selector.evidence(),
                "prefix_before": list(selected_indices),
                "theta_before": parameters.tolist(),
                "shortlist_indices": list(shortlist),
                "shortlist_count": len(shortlist),
                "shortlist": probe_records,
                "base_selected_idx": int(ranking[0]),
                "base_selected_rank": 1,
                "greedy_selected_idx": int(ranking[0]),
                "greedy_selected_rank": 1,
                "selected_idx": selected_index,
                "selected_rank": int(
                    next(
                        rank
                        for rank, candidate in enumerate(shortlist, start=1)
                        if candidate == selected_index
                    )
                ),
                "selected_relaxed_energy": (
                    None if winner is None else float(winner["relaxed_energy"])
                ),
                "selected_initial_point": selected_initial_parameters.tolist(),
                "selected_source": (
                    "greedy_fallback" if winner is None else "energy_probe"
                ),
                "used_successful_probe": winner is not None,
                "fallback_reason": (
                    "no_successful_probe" if winner is None else None
                ),
                "total_probe_nfev": sum(
                    int(probe["optimizer_nfev"]) for probe in probe_records
                ),
                "total_probe_nit": sum(
                    int(probe["optimizer_nit"]) for probe in probe_records
                ),
                "total_probe_elapsed_s": sum(
                    float(probe["elapsed_s"]) for probe in probe_records
                ),
                "selection_elapsed_s": perf_counter() - probe_started,
            }
            selector_events.append(event)
            record["selector_event_index"] = len(selector_events) - 1

        selected_indices.append(selected_index)
        parameters = selected_initial_parameters
        objective = _SectorObjective(
            h1_values,
            h2_values,
            reference,
            pool,
            selected_indices,
        )
        previous_energy = energy
        optimization = minimize(
            objective.energy,
            parameters,
            method="SLSQP",
            jac=objective.jacobian,
            options={
                "maxiter": int(optimizer_max_iterations),
                "ftol": float(optimizer_ftol),
                "disp": False,
            },
        )
        if not np.all(np.isfinite(optimization.x)) or not math.isfinite(
            float(optimization.fun)
        ):
            raise RuntimeError("fixed-sector SLSQP returned non-finite output")
        parameters = np.asarray(optimization.x, dtype=np.float64)
        state = objective.state(parameters)
        ci, h_ci = pyscf_fixed_sector_hamiltonian_action(
            h1_values, h2_values, state
        )
        energy = float(np.vdot(ci, h_ci).real)
        if energy > previous_energy + 1.0e-8:
            raise RuntimeError(
                "fixed-sector ansatz reoptimization increased the energy by "
                f"{energy - previous_energy:.3e} Ha"
            )
        optimizer_success = optimizer_success and bool(optimization.success)
        optimizer_evaluations += int(getattr(optimization, "nfev", 0))
        optimizer_iterations += int(getattr(optimization, "nit", 0))
        record.update(
            {
                "selected_index": selected_index,
                "energy_after_optimization_ha": energy,
                "energy_change_ha": energy - previous_energy,
                "optimizer_success": bool(optimization.success),
                "optimizer_status": int(optimization.status),
                "optimizer_message": str(optimization.message),
                "optimizer_evaluations": int(getattr(optimization, "nfev", 0)),
                "optimizer_iterations": int(getattr(optimization, "nit", 0)),
                "state_support_size": state.support_size,
            }
        )
        gradient_history.append(record)
        energy_history.append(energy)
        if progress_callback is not None:
            progress_callback("adapt_iteration", adapt_iteration + 1, max_adapt_iterations)
        if (
            eigenvalue_threshold > 0.0
            and abs(energy - previous_energy) < eigenvalue_threshold
        ):
            termination = "eigenvalue_threshold"
            break
    else:
        termination = "operator_cap"

    return SectorAdaptResult(
        energy=energy,
        state=state,
        selected_indices=tuple(selected_indices),
        parameters=tuple(float(value) for value in parameters),
        gradient_history=tuple(gradient_history),
        energy_history=tuple(energy_history),
        termination=termination,
        optimizer_success=optimizer_success,
        optimizer_evaluations=optimizer_evaluations,
        optimizer_iterations=optimizer_iterations,
        selector_config=selector,
        selector_events=tuple(selector_events),
    )


def fixed_sector_dimension(n_spatial: int, n_alpha: int, n_beta: int) -> int:
    return math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)


def sector_state_to_pyscf_ci(state: SparseSectorState) -> np.ndarray:
    from pyscf.fci import cistring

    na = math.comb(state.n_spatial, state.n_alpha)
    nb = math.comb(state.n_spatial, state.n_beta)
    ci = np.zeros((na, nb), dtype=np.float64)
    alpha_mask = (1 << state.n_spatial) - 1
    for bitstring, amplitude in state.amplitudes:
        alpha_string = bitstring & alpha_mask
        beta_string = bitstring >> state.n_spatial
        alpha_address = cistring.str2addr(
            state.n_spatial, state.n_alpha, alpha_string
        )
        beta_address = cistring.str2addr(
            state.n_spatial, state.n_beta, beta_string
        )
        ci[alpha_address, beta_address] = amplitude
    return ci


def pyscf_ci_amplitude(
    ci: np.ndarray,
    bitstring: int,
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> float:
    from pyscf.fci import cistring

    alpha_mask = (1 << n_spatial) - 1
    alpha_address = cistring.str2addr(
        n_spatial, n_alpha, bitstring & alpha_mask
    )
    beta_address = cistring.str2addr(
        n_spatial, n_beta, bitstring >> n_spatial
    )
    return float(ci[alpha_address, beta_address])


def pyscf_fixed_sector_hamiltonian_action(
    h1: np.ndarray,
    h2: np.ndarray,
    state: SparseSectorState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(CI, H@CI)`` using PySCF's real fixed-spin FCI contraction."""

    from pyscf.fci import direct_spin1

    ci = sector_state_to_pyscf_ci(state)
    h2e = direct_spin1.absorb_h1e(
        np.asarray(h1, dtype=np.float64),
        np.asarray(h2, dtype=np.float64),
        state.n_spatial,
        (state.n_alpha, state.n_beta),
        0.5,
    )
    h_ci = direct_spin1.contract_2e(
        h2e,
        ci,
        state.n_spatial,
        (state.n_alpha, state.n_beta),
    )
    return ci, np.asarray(h_ci)


def hf_gradients_from_pyscf_action(
    hf_state: SparseSectorState,
    h_ci: np.ndarray,
    pool: Sequence[CompactCEOOperator],
) -> np.ndarray:
    if hf_state.support_size != 1:
        raise ValueError("dense-action gradient shortcut requires an HF basis state")
    bitstring, amplitude = hf_state.amplitudes[0]
    gradients = np.zeros(len(pool), dtype=np.float64)
    for index, operator in enumerate(pool):
        value = 0.0
        for target, coefficient in _descriptor_action_on_bit(
            bitstring, operator
        ).items():
            value += coefficient * pyscf_ci_amplitude(
                h_ci,
                target,
                hf_state.n_spatial,
                hf_state.n_alpha,
                hf_state.n_beta,
            )
        gradients[index] = 2.0 * amplitude * value
    return gradients


__all__ = [
    "ALGEBRA_TOLERANCE",
    "CEOConstituent",
    "CompactCEOOperator",
    "OneParameterProbe",
    "SectorAdaptResult",
    "SectorSelectorConfig",
    "SparseSectorState",
    "all_pool_gradients",
    "all_pool_gradients_from_pyscf_action",
    "apply_ceo_ansatz",
    "apply_ceo_generator",
    "apply_sparse_pauli",
    "compact_local_k",
    "descriptor_from_manifest",
    "deterministic_gradient_ranking",
    "fixed_sector_dimension",
    "generate_compact_ovp_ceo_pool",
    "hartree_fock_sector_state",
    "hf_gradients_from_pyscf_action",
    "interleaved_to_blocked",
    "manifest_local_k",
    "optimize_one_parameter",
    "pyscf_ci_amplitude",
    "pyscf_fixed_sector_hamiltonian_action",
    "rotate_ceo",
    "sector_energy",
    "sector_ansatz_energy_and_gradient",
    "sector_inner_product",
    "sector_state_to_pyscf_ci",
    "solve_sector_adapt",
    "sparse_pauli_basis_diagonal",
    "validate_compact_pool_against_manifest",
]
