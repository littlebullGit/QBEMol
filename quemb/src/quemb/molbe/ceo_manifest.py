"""Dependency-free loader for exported OVP-CEO operator pools.

The source ``ceo-adapt-vqe`` project represents pool generators as
anti-Hermitian OpenFermion ``QubitOperator`` objects in interleaved
spin-orbital order.  QuEmb uses Qiskit's blocked spin-orbital order and
Hermitian generators.  This module is the deliberately narrow interchange
boundary between those projects:

* JSON is validated without importing OpenFermion or the source project.
* every payload level has a canonical SHA-256 digest;
* Qiskit-big-endian Pauli labels are permuted from interleaved to blocked
  spin-orbital order; and
* an exported anti-Hermitian generator ``K`` becomes the Hermitian
  ``G = i K`` consumed by QuEmb.

Manifest schemas
----------------

Both schemas are intentionally strict. Unknown and missing fields are rejected.
Version 1 is the original molecule-backed artifact and includes the exported
source Hamiltonian. Version 2 has ``artifact_kind="pool_only"`` and contains
only the size/filling-defined pool; a Hamiltonian key is forbidden.
``source_orbitals`` and ``target_orbitals`` are lists of constituent
excitations: one constituent for a single and two for a sum/difference CEO.
All orbital indices and the HF bitstring use ascending qubit-index order.
Pauli labels use Qiskit's big-endian label convention.

Use :func:`attach_manifest_digests` in the exporter after constructing the
payload without any ``digest`` fields.  Use :func:`load_ceo_manifest` in
QuEmb.  Both functions share the same canonical serialization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from qiskit.quantum_info import SparsePauliOp


SCHEMA = "quemb.ceo-ovp-manifest/v1"
SCHEMA_VERSION = 1
POOL_ONLY_SCHEMA = "quemb.ceo-ovp-manifest/v2"
POOL_ONLY_SCHEMA_VERSION = 2
POOL_ONLY_ARTIFACT_KIND = "pool_only"
REGISTRY_SCHEMA = "quemb.ceo-ovp-manifest-registry/v1"
REGISTRY_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA = "quemb.ceo_manifest_provenance"
PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FILENAME = "manifest_provenance.json"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{7,64}\Z")
_PAULI_RE = re.compile(r"[IXYZ]+\Z")
_ALGEBRA_ATOL = 1.0e-12


class CEOManifestError(ValueError):
    """Raised when a CEO manifest is malformed, inconsistent, or corrupted."""


@dataclass(frozen=True)
class PauliTerm:
    """One Pauli term in source (interleaved) spin-orbital order."""

    label: str
    coefficient: complex


@dataclass(frozen=True)
class CEOSource:
    """Provenance and source-pool options recorded by the exporter."""

    repository: str
    commit: str
    pool_class: str
    sum_enabled: bool
    diff_enabled: bool
    fermionic_swaps: bool


@dataclass(frozen=True)
class CEOPoolOperator:
    """One validated OVP-CEO operator.

    ``source_orbitals``, ``target_orbitals``, and ``support_orbitals`` retain
    the source interleaved indices. Their ``blocked_*`` properties expose the
    corresponding QuEmb indices. ``source_anti_hermitian`` is K in the source
    order, ``anti_hermitian`` is K after permutation, and ``generator`` is the
    QuEmb Hermitian G=iK.
    """

    index: int
    ceo_type: str
    source_orbitals: tuple[tuple[int, ...], ...]
    target_orbitals: tuple[tuple[int, ...], ...]
    support_orbitals: tuple[int, ...]
    normalization_kind: str
    normalization_value: float
    pauli_terms: tuple[PauliTerm, ...]
    digest: str
    n_qubits: int

    @property
    def blocked_source_orbitals(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(interleaved_to_blocked_index(i, self.n_qubits) for i in group)
            for group in self.source_orbitals
        )

    @property
    def blocked_target_orbitals(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(interleaved_to_blocked_index(i, self.n_qubits) for i in group)
            for group in self.target_orbitals
        )

    @property
    def blocked_support_orbitals(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                interleaved_to_blocked_index(i, self.n_qubits)
                for i in self.support_orbitals
            )
        )

    @property
    def source_anti_hermitian(self) -> SparsePauliOp:
        """Return exported K in source interleaved order."""

        return _sparse_pauli_op(self.pauli_terms)

    @property
    def anti_hermitian(self) -> SparsePauliOp:
        """Return K permuted to QuEmb blocked order."""

        terms = tuple(
            PauliTerm(
                remap_pauli_label_interleaved_to_blocked(
                    term.label, self.n_qubits
                ),
                term.coefficient,
            )
            for term in self.pauli_terms
        )
        return _sparse_pauli_op(terms)

    @property
    def generator(self) -> SparsePauliOp:
        """Return the QuEmb Hermitian generator G=iK in blocked order."""

        terms = tuple(
            PauliTerm(
                remap_pauli_label_interleaved_to_blocked(
                    term.label, self.n_qubits
                ),
                _real_if_close(1j * term.coefficient),
            )
            for term in self.pauli_terms
        )
        return _sparse_pauli_op(terms)


@dataclass(frozen=True)
class CEOManifest:
    """Validated CEO manifest ready for QuEmb integration."""

    source: CEOSource
    schema_version: int
    artifact_kind: str
    n_qubits: int
    n_electrons: int
    source_hf_bitstring: str
    operators: tuple[CEOPoolOperator, ...]
    source_hamiltonian_terms: tuple[PauliTerm, ...] | None
    pool_digest: str
    hamiltonian_digest: str | None
    digest: str

    @property
    def n_spatial_orbitals(self) -> int:
        return self.n_qubits // 2

    @property
    def hf_bitstring(self) -> str:
        """Return the HF bitstring in blocked, ascending-qubit-index order."""

        return remap_bitstring_interleaved_to_blocked(
            self.source_hf_bitstring
        )

    @property
    def generators(self) -> tuple[SparsePauliOp, ...]:
        """Return all Hermitian G=iK pool generators in manifest order."""

        return tuple(operator.generator for operator in self.operators)

    @property
    def source_hamiltonian(self) -> SparsePauliOp | None:
        """Return source-order H, or ``None`` for a pool-only artifact."""

        if self.source_hamiltonian_terms is None:
            return None
        return _sparse_pauli_op(self.source_hamiltonian_terms)

    @property
    def hamiltonian(self) -> SparsePauliOp | None:
        """Return blocked-order H, or ``None`` for a pool-only artifact."""

        if self.source_hamiltonian_terms is None:
            return None
        terms = tuple(
            PauliTerm(
                remap_pauli_label_interleaved_to_blocked(
                    term.label, self.n_qubits
                ),
                _real_if_close(term.coefficient),
            )
            for term in self.source_hamiltonian_terms
        )
        return _sparse_pauli_op(terms)


@dataclass(frozen=True)
class CEOManifestProvenanceEntry:
    """Out-of-band generation/consumption provenance for one manifest."""

    manifest_digest: str
    source_tree_digest: str
    environment_digest: str
    consuming_cells: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class CEOManifestProvenance:
    """Validated canonical provenance sidecar."""

    entries: tuple[CEOManifestProvenanceEntry, ...]
    digest: str


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing.

    Object keys are sorted, insignificant whitespace is removed, non-finite
    numbers are rejected, and negative zero is normalized to positive zero.
    List ordering remains significant because pool/operator ordering is part of
    the scientific artifact.
    """

    normalized = _normalize_for_digest(value, path="$")
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """Return a tagged canonical SHA-256 digest."""

    return f"sha256:{sha256(canonical_json_bytes(value)).hexdigest()}"


def attach_manifest_digests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with all required digests computed.

    The input must contain no ``digest`` keys. Digests are added from the
    leaves upward: operator, pool, optional v1 Hamiltonian, then whole
    manifest. The result can be written directly as JSON and later checked by
    :func:`parse_ceo_manifest`.
    """

    result = deepcopy(dict(payload))
    _reject_existing_digest(result, "$")

    pool = _mutable_mapping(result.get("pool"), "$.pool")
    operators = _mutable_sequence(pool.get("operators"), "$.pool.operators")
    for index, operator in enumerate(operators):
        operator_map = _mutable_mapping(
            operator, f"$.pool.operators[{index}]"
        )
        operator_map["digest"] = canonical_digest(operator_map)
    pool["digest"] = canonical_digest(pool)

    schema = result.get("schema")
    if schema == SCHEMA:
        hamiltonian = _mutable_mapping(
            result.get("hamiltonian"), "$.hamiltonian"
        )
        hamiltonian["digest"] = canonical_digest(hamiltonian)
    elif schema != POOL_ONLY_SCHEMA:
        _fail(
            "$.schema",
            f"must equal {SCHEMA!r} or {POOL_ONLY_SCHEMA!r}",
        )
    result["digest"] = canonical_digest(result)
    return result


def attach_provenance_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the canonical top-level digest to an unsigned sidecar.

    Persist the result with :func:`canonical_json_bytes` to satisfy the UTF-8,
    compact, sorted-key, and no-trailing-newline sidecar contract.
    """

    result = deepcopy(dict(payload))
    if "digest" in result:
        _fail(
            "$.digest",
            "must be absent before attach_provenance_digest()",
        )
    result["digest"] = canonical_digest(result)
    return result


def load_ceo_manifest(path: str | Path) -> CEOManifest:
    """Read and validate a CEO manifest from ``path``."""

    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise CEOManifestError(
            f"{manifest_path}: unable to read CEO manifest: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CEOManifestError(
            f"{manifest_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        return parse_ceo_manifest(payload)
    except CEOManifestError as exc:
        raise CEOManifestError(f"{manifest_path}: {exc}") from exc


def load_ceo_manifest_for_system(
    path: str | Path,
    *,
    n_qubits: int,
    n_electrons: int,
    provenance_path: str | Path | None = None,
) -> CEOManifest:
    """Load one manifest or resolve a strict fragment-size registry.

    A registry keeps the public ``adapt_pool_manifest`` interface as one path
    while allowing a BE calculation to select a source-generated OVP-CEO pool
    for each distinct fragment size. Relative entry paths are resolved against
    the registry file. Every entry pins both the complete manifest digest and
    the pool digest.
    """
    artifact_path = Path(path)
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise CEOManifestError(
            f"{artifact_path}: unable to read CEO manifest artifact: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CEOManifestError(
            f"{artifact_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if isinstance(payload, Mapping) and payload.get("schema") in (
        SCHEMA,
        POOL_ONLY_SCHEMA,
    ):
        try:
            manifest = parse_ceo_manifest(payload)
        except CEOManifestError as exc:
            raise CEOManifestError(f"{artifact_path}: {exc}") from exc
        _require_manifest_system(
            manifest,
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            path=str(artifact_path),
        )
        return manifest

    root = _mapping(payload, "$")
    _expect_keys(
        root,
        {
            "schema",
            "schema_version",
            "entries",
            "digest",
        },
        "$",
    )
    _expect_literal(root["schema"], REGISTRY_SCHEMA, "$.schema")
    _expect_literal(
        root["schema_version"],
        REGISTRY_SCHEMA_VERSION,
        "$.schema_version",
    )
    _verify_section_digest(root, "$")
    entries = _sequence(root["entries"], "$.entries")
    selected: tuple[Path, str, str] | None = None
    seen: set[tuple[int, int]] = set()
    registry_manifest_digests: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry_path = f"$.entries[{index}]"
        entry = _mapping(raw_entry, entry_path)
        _expect_keys(
            entry,
            {
                "n_qubits",
                "n_electrons",
                "manifest",
                "manifest_digest",
                "pool_digest",
            },
            entry_path,
        )
        entry_n_qubits = _integer(
            entry["n_qubits"],
            f"{entry_path}.n_qubits",
        )
        entry_n_electrons = _integer(
            entry["n_electrons"],
            f"{entry_path}.n_electrons",
        )
        if entry_n_qubits < 2 or entry_n_qubits % 2:
            _fail(
                f"{entry_path}.n_qubits",
                "must be an even integer of at least 2",
            )
        if not 0 < entry_n_electrons <= entry_n_qubits:
            _fail(
                f"{entry_path}.n_electrons",
                f"must be between 1 and n_qubits ({entry_n_qubits})",
            )
        key = (entry_n_qubits, entry_n_electrons)
        if key in seen:
            _fail(entry_path, f"duplicate fragment key {key}")
        seen.add(key)
        manifest_name = _string(
            entry["manifest"],
            f"{entry_path}.manifest",
        )
        if not manifest_name:
            _fail(f"{entry_path}.manifest", "must not be empty")
        manifest_digest = _digest_string(
            entry["manifest_digest"],
            f"{entry_path}.manifest_digest",
        )
        pool_digest = _digest_string(
            entry["pool_digest"],
            f"{entry_path}.pool_digest",
        )
        registry_manifest_digests.add(manifest_digest)
        if key == (n_qubits, n_electrons):
            candidate = Path(manifest_name)
            if not candidate.is_absolute():
                candidate = artifact_path.parent / candidate
            selected = (candidate, manifest_digest, pool_digest)

    resolved_provenance_path: Path | None
    if provenance_path is None:
        colocated = artifact_path.parent / PROVENANCE_FILENAME
        resolved_provenance_path = colocated if colocated.is_file() else None
    else:
        resolved_provenance_path = Path(provenance_path)
    if resolved_provenance_path is not None:
        load_ceo_manifest_provenance(
            resolved_provenance_path,
            registry_manifest_digests=registry_manifest_digests,
        )

    if selected is None:
        available = ", ".join(
            f"{qubits}q/{electrons}e"
            for qubits, electrons in sorted(seen)
        )
        raise CEOManifestError(
            f"{artifact_path}: no CEO manifest entry for "
            f"{n_qubits} qubits/{n_electrons} electrons; "
            f"available: {available or 'none'}"
        )

    manifest_path, expected_manifest_digest, expected_pool_digest = selected
    manifest = load_ceo_manifest(manifest_path)
    _require_manifest_system(
        manifest,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        path=str(manifest_path),
    )
    if manifest.digest != expected_manifest_digest:
        raise CEOManifestError(
            f"{artifact_path}: selected manifest digest mismatch: "
            f"expected {expected_manifest_digest}, found {manifest.digest}"
        )
    if manifest.pool_digest != expected_pool_digest:
        raise CEOManifestError(
            f"{artifact_path}: selected pool digest mismatch: "
            f"expected {expected_pool_digest}, found {manifest.pool_digest}"
        )
    return manifest


def load_ceo_manifest_provenance(
    path: str | Path,
    *,
    registry_manifest_digests: set[str] | frozenset[str],
) -> CEOManifestProvenance:
    """Load a strict, digest-pinned provenance sidecar.

    Sidecar entries may cover a subset of a registry, but they may not name a
    manifest outside that registry. This keeps operational metadata outside the
    immutable scientific artifact and the deliberately stable registry entry
    shape.
    """

    sidecar_path = Path(path)
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise CEOManifestError(
            f"{sidecar_path}: unable to read CEO manifest provenance: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CEOManifestError(
            f"{sidecar_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        return parse_ceo_manifest_provenance(
            payload,
            registry_manifest_digests=registry_manifest_digests,
        )
    except CEOManifestError as exc:
        raise CEOManifestError(f"{sidecar_path}: {exc}") from exc


def parse_ceo_manifest_provenance(
    payload: Any,
    *,
    registry_manifest_digests: set[str] | frozenset[str],
) -> CEOManifestProvenance:
    """Validate an already-decoded canonical provenance sidecar."""

    root = _mapping(payload, "$")
    _expect_keys(
        root,
        {"schema", "schema_version", "manifests", "digest"},
        "$",
    )
    _expect_literal(root["schema"], PROVENANCE_SCHEMA, "$.schema")
    _expect_literal(
        root["schema_version"],
        PROVENANCE_SCHEMA_VERSION,
        "$.schema_version",
    )
    _verify_section_digest(root, "$")

    known_digests = set(registry_manifest_digests)
    manifests_payload = _mapping(root["manifests"], "$.manifests")
    if not manifests_payload:
        _fail("$.manifests", "must contain at least one provenance entry")
    entries: list[CEOManifestProvenanceEntry] = []
    for raw_manifest_digest in sorted(manifests_payload):
        manifest_digest = _digest_string(
            raw_manifest_digest,
            "$.manifests.<manifest_digest>",
        )
        entry_path = f"$.manifests[{manifest_digest!r}]"
        raw_entry = manifests_payload[raw_manifest_digest]
        entry = _mapping(raw_entry, entry_path)
        _expect_keys(
            entry,
            {
                "source_tree_digest",
                "environment_digest",
                "consuming_cells",
            },
            entry_path,
        )
        if manifest_digest not in known_digests:
            _fail(
                "$.manifests.<manifest_digest>",
                "is absent from the strict manifest registry",
            )

        source_tree_digest = _digest_string(
            entry["source_tree_digest"],
            f"{entry_path}.source_tree_digest",
        )
        environment_digest = _digest_string(
            entry["environment_digest"],
            f"{entry_path}.environment_digest",
        )
        consuming_cells_payload = _sequence(
            entry["consuming_cells"], f"{entry_path}.consuming_cells"
        )
        if not consuming_cells_payload:
            _fail(
                f"{entry_path}.consuming_cells",
                "must contain at least one distance/fragment cell",
            )
        consuming_cells: list[tuple[float, str]] = []
        seen_cells: set[tuple[float, str]] = set()
        for cell_index, raw_cell in enumerate(consuming_cells_payload):
            cell_path = f"{entry_path}.consuming_cells[{cell_index}]"
            cell = _mapping(raw_cell, cell_path)
            _expect_keys(
                cell,
                {"distance_angstrom", "fragment_id"},
                cell_path,
            )
            distance = _finite_number(
                cell["distance_angstrom"], f"{cell_path}.distance_angstrom"
            )
            if distance <= 0.0:
                _fail(f"{cell_path}.distance_angstrom", "must be positive")
            fragment_id = _string(
                cell["fragment_id"], f"{cell_path}.fragment_id"
            )
            if not fragment_id:
                _fail(f"{cell_path}.fragment_id", "must not be empty")
            cell_key = (distance, fragment_id)
            if cell_key in seen_cells:
                _fail(cell_path, f"duplicate consuming cell {cell_key!r}")
            seen_cells.add(cell_key)
            consuming_cells.append(cell_key)

        if consuming_cells != sorted(consuming_cells):
            _fail(
                f"{entry_path}.consuming_cells",
                "must be sorted by distance_angstrom then fragment_id",
            )

        entries.append(
            CEOManifestProvenanceEntry(
                manifest_digest=manifest_digest,
                source_tree_digest=source_tree_digest,
                environment_digest=environment_digest,
                consuming_cells=tuple(consuming_cells),
            )
        )

    return CEOManifestProvenance(
        entries=tuple(entries),
        digest=_digest_string(root["digest"], "$.digest"),
    )


def _require_manifest_system(
    manifest: CEOManifest,
    *,
    n_qubits: int,
    n_electrons: int,
    path: str,
) -> None:
    if manifest.n_qubits != n_qubits:
        raise CEOManifestError(
            f"{path}: CEO manifest qubit count mismatch: "
            f"manifest={manifest.n_qubits}, requested={n_qubits}"
        )
    if manifest.n_electrons != n_electrons:
        raise CEOManifestError(
            f"{path}: CEO manifest electron count mismatch: "
            f"manifest={manifest.n_electrons}, requested={n_electrons}"
        )


def parse_ceo_manifest(payload: Any) -> CEOManifest:
    """Validate an already-decoded manifest and construct Qiskit-ready data."""

    root = _mapping(payload, "$")
    schema = _string(root.get("schema"), "$.schema")
    if schema == SCHEMA:
        _expect_keys(
            root,
            {
                "schema",
                "schema_version",
                "source",
                "system",
                "conventions",
                "pool",
                "hamiltonian",
                "digest",
            },
            "$",
        )
        _expect_literal(
            root["schema_version"], SCHEMA_VERSION, "$.schema_version"
        )
        schema_version = SCHEMA_VERSION
        artifact_kind = "molecule_hamiltonian"
    elif schema == POOL_ONLY_SCHEMA:
        _expect_keys(
            root,
            {
                "schema",
                "schema_version",
                "artifact_kind",
                "source",
                "system",
                "conventions",
                "pool",
                "digest",
            },
            "$",
        )
        _expect_literal(
            root["schema_version"],
            POOL_ONLY_SCHEMA_VERSION,
            "$.schema_version",
        )
        _expect_literal(
            root["artifact_kind"],
            POOL_ONLY_ARTIFACT_KIND,
            "$.artifact_kind",
        )
        schema_version = POOL_ONLY_SCHEMA_VERSION
        artifact_kind = POOL_ONLY_ARTIFACT_KIND
    else:
        _fail(
            "$.schema",
            f"must equal {SCHEMA!r} or {POOL_ONLY_SCHEMA!r}",
        )

    source_map = _mapping(root["source"], "$.source")
    _expect_keys(
        source_map,
        {"repository", "commit", "pool_class", "pool_options"},
        "$.source",
    )
    repository = _string(source_map["repository"], "$.source.repository")
    if not repository:
        _fail("$.source.repository", "must not be empty")
    commit = _string(source_map["commit"], "$.source.commit")
    if not _COMMIT_RE.fullmatch(commit):
        _fail(
            "$.source.commit",
            "must be a 7-64 character hexadecimal Git commit",
        )
    _expect_literal(
        source_map["pool_class"], "OVP_CEO", "$.source.pool_class"
    )
    options = _mapping(source_map["pool_options"], "$.source.pool_options")
    _expect_keys(
        options,
        {"sum", "diff", "fermionic_swaps"},
        "$.source.pool_options",
    )
    _expect_literal(options["sum"], True, "$.source.pool_options.sum")
    _expect_literal(options["diff"], True, "$.source.pool_options.diff")
    _expect_literal(
        options["fermionic_swaps"],
        False,
        "$.source.pool_options.fermionic_swaps",
    )
    source = CEOSource(
        repository=repository,
        commit=commit.lower(),
        pool_class="OVP_CEO",
        sum_enabled=True,
        diff_enabled=True,
        fermionic_swaps=False,
    )

    system = _mapping(root["system"], "$.system")
    _expect_keys(
        system, {"n_qubits", "n_electrons", "hf_bitstring"}, "$.system"
    )
    n_qubits = _integer(system["n_qubits"], "$.system.n_qubits")
    if n_qubits < 2 or n_qubits % 2:
        _fail("$.system.n_qubits", "must be an even integer of at least 2")
    n_electrons = _integer(
        system["n_electrons"], "$.system.n_electrons"
    )
    if not 0 < n_electrons <= n_qubits:
        _fail(
            "$.system.n_electrons",
            f"must be between 1 and n_qubits ({n_qubits})",
        )
    source_hf_bitstring = _string(
        system["hf_bitstring"], "$.system.hf_bitstring"
    )
    if (
        len(source_hf_bitstring) != n_qubits
        or set(source_hf_bitstring) - {"0", "1"}
    ):
        _fail(
            "$.system.hf_bitstring",
            f"must contain exactly {n_qubits} binary digits",
        )
    if source_hf_bitstring.count("1") != n_electrons:
        _fail(
            "$.system.hf_bitstring",
            "occupied-bit count does not equal n_electrons",
        )
    if schema == POOL_ONLY_SCHEMA:
        expected_hf_bitstring = (
            "1" * n_electrons + "0" * (n_qubits - n_electrons)
        )
        if source_hf_bitstring != expected_hf_bitstring:
            _fail(
                "$.system.hf_bitstring",
                "must use canonical interleaved Hartree-Fock ordering "
                f"{expected_hf_bitstring!r}",
            )

    conventions = _mapping(root["conventions"], "$.conventions")
    required_conventions = {
        "source_spin_orbital_order": "interleaved",
        "target_spin_orbital_order": "blocked",
        "pauli_label_order": "qiskit_big_endian",
        "hf_bitstring_order": "qubit_index_ascending",
        "source_generator": "anti_hermitian",
        "generator_relation": "G=iK",
    }
    _expect_keys(conventions, set(required_conventions), "$.conventions")
    for key, expected in required_conventions.items():
        _expect_literal(
            conventions[key], expected, f"$.conventions.{key}"
        )

    pool = _mapping(root["pool"], "$.pool")
    _expect_keys(pool, {"name", "operators", "digest"}, "$.pool")
    _expect_literal(pool["name"], "OVP_CEO", "$.pool.name")
    operator_payloads = _sequence(pool["operators"], "$.pool.operators")
    if not operator_payloads:
        _fail("$.pool.operators", "must contain at least one operator")

    operators: list[CEOPoolOperator] = []
    for position, operator_payload in enumerate(operator_payloads):
        operator_path = f"$.pool.operators[{position}]"
        operator_map = _mapping(operator_payload, operator_path)
        _expect_keys(
            operator_map,
            {
                "index",
                "ceo_type",
                "source_orbitals",
                "target_orbitals",
                "support_orbitals",
                "normalization",
                "pauli_terms",
                "digest",
            },
            operator_path,
        )
        _verify_section_digest(operator_map, operator_path)
        operators.append(
            _parse_pool_operator(
                operator_map, operator_path, position, n_qubits
            )
        )
    _verify_section_digest(pool, "$.pool")

    source_hamiltonian_terms: tuple[PauliTerm, ...] | None = None
    hamiltonian_digest: str | None = None
    if schema == SCHEMA:
        hamiltonian = _mapping(root["hamiltonian"], "$.hamiltonian")
        _expect_keys(
            hamiltonian, {"pauli_terms", "digest"}, "$.hamiltonian"
        )
        _verify_section_digest(hamiltonian, "$.hamiltonian")
        source_hamiltonian_terms = _parse_pauli_terms(
            hamiltonian["pauli_terms"],
            "$.hamiltonian.pauli_terms",
            n_qubits,
            coefficient_kind="hermitian",
        )
        hamiltonian_digest = _digest_string(
            hamiltonian["digest"], "$.hamiltonian.digest"
        )

    _verify_section_digest(root, "$")
    return CEOManifest(
        source=source,
        schema_version=schema_version,
        artifact_kind=artifact_kind,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        source_hf_bitstring=source_hf_bitstring,
        operators=tuple(operators),
        source_hamiltonian_terms=source_hamiltonian_terms,
        pool_digest=_digest_string(pool["digest"], "$.pool.digest"),
        hamiltonian_digest=hamiltonian_digest,
        digest=_digest_string(root["digest"], "$.digest"),
    )


def interleaved_to_blocked_index(index: int, n_qubits: int) -> int:
    """Map source spin-orbital index ``2p/2p+1`` to QuEmb blocked order."""

    if type(n_qubits) is not int or n_qubits < 2 or n_qubits % 2:
        raise ValueError("n_qubits must be an even integer of at least 2")
    if type(index) is not int or not 0 <= index < n_qubits:
        raise ValueError(f"index must be in [0, {n_qubits})")
    spatial = index // 2
    return spatial if index % 2 == 0 else spatial + n_qubits // 2


def remap_pauli_label_interleaved_to_blocked(
    label: str, n_qubits: int | None = None
) -> str:
    """Permute a Qiskit-big-endian Pauli label to blocked spin order."""

    if not isinstance(label, str) or not _PAULI_RE.fullmatch(label):
        raise ValueError("label must be a non-empty uppercase I/X/Y/Z string")
    if n_qubits is None:
        n_qubits = len(label)
    if len(label) != n_qubits:
        raise ValueError(
            f"label has length {len(label)}, expected n_qubits={n_qubits}"
        )

    target = ["I"] * n_qubits
    for source_index in range(n_qubits):
        source_label_position = n_qubits - 1 - source_index
        target_index = interleaved_to_blocked_index(source_index, n_qubits)
        target_label_position = n_qubits - 1 - target_index
        target[target_label_position] = label[source_label_position]
    return "".join(target)


def remap_bitstring_interleaved_to_blocked(bitstring: str) -> str:
    """Permute an ascending-qubit-index bitstring to blocked spin order."""

    if (
        not isinstance(bitstring, str)
        or not bitstring
        or set(bitstring) - {"0", "1"}
    ):
        raise ValueError("bitstring must be a non-empty binary string")
    n_qubits = len(bitstring)
    target = ["0"] * n_qubits
    for source_index, occupation in enumerate(bitstring):
        target_index = interleaved_to_blocked_index(source_index, n_qubits)
        target[target_index] = occupation
    return "".join(target)


def _parse_pool_operator(
    payload: Mapping[str, Any],
    path: str,
    position: int,
    n_qubits: int,
) -> CEOPoolOperator:
    index = _integer(payload["index"], f"{path}.index")
    if index != position:
        _fail(
            f"{path}.index",
            f"must equal its zero-based pool position {position}",
        )

    ceo_type = _string(payload["ceo_type"], f"{path}.ceo_type")
    if ceo_type not in {"single", "sum", "diff"}:
        _fail(f"{path}.ceo_type", "must be 'single', 'sum', or 'diff'")

    source_orbitals = _parse_orbital_groups(
        payload["source_orbitals"], f"{path}.source_orbitals", n_qubits
    )
    target_orbitals = _parse_orbital_groups(
        payload["target_orbitals"], f"{path}.target_orbitals", n_qubits
    )
    if len(source_orbitals) != len(target_orbitals):
        _fail(
            path,
            "source_orbitals and target_orbitals must have the same "
            "number of constituents",
        )
    expected_constituents = 1 if ceo_type == "single" else 2
    expected_rank = 1 if ceo_type == "single" else 2
    if len(source_orbitals) != expected_constituents:
        _fail(
            path,
            f"{ceo_type!r} requires {expected_constituents} constituent "
            "excitation(s)",
        )
    for constituent, (sources, targets) in enumerate(
        zip(source_orbitals, target_orbitals)
    ):
        if len(sources) != expected_rank or len(targets) != expected_rank:
            _fail(
                path,
                f"{ceo_type!r} constituent {constituent} must have "
                f"{expected_rank} source and target orbital(s)",
            )
        if set(sources) & set(targets):
            _fail(
                path,
                f"constituent {constituent} source and target orbitals "
                "must be disjoint",
            )
        if sum(i % 2 == 0 for i in sources) != sum(
            i % 2 == 0 for i in targets
        ):
            _fail(
                path,
                f"constituent {constituent} does not preserve alpha/beta "
                "electron counts in interleaved order",
            )

    support_values = _sequence(
        payload["support_orbitals"], f"{path}.support_orbitals"
    )
    support_orbitals = tuple(
        _orbital_index(
            value, f"{path}.support_orbitals[{i}]", n_qubits
        )
        for i, value in enumerate(support_values)
    )
    if tuple(sorted(set(support_orbitals))) != support_orbitals:
        _fail(
            f"{path}.support_orbitals",
            "must be unique and sorted in ascending source-index order",
        )
    expected_support = tuple(
        sorted(
            {
                orbital
                for group in source_orbitals + target_orbitals
                for orbital in group
            }
        )
    )
    if support_orbitals != expected_support:
        _fail(
            f"{path}.support_orbitals",
            f"must equal the source/target union {list(expected_support)}",
        )

    normalization = _mapping(
        payload["normalization"], f"{path}.normalization"
    )
    _expect_keys(
        normalization, {"kind", "value"}, f"{path}.normalization"
    )
    _expect_literal(
        normalization["kind"],
        "l1_pauli_coefficients",
        f"{path}.normalization.kind",
    )
    normalization_value = _finite_number(
        normalization["value"], f"{path}.normalization.value"
    )
    if normalization_value <= 0.0:
        _fail(f"{path}.normalization.value", "must be positive")

    pauli_terms = _parse_pauli_terms(
        payload["pauli_terms"],
        f"{path}.pauli_terms",
        n_qubits,
        coefficient_kind="anti_hermitian",
    )
    l1_norm = sum(abs(term.coefficient) for term in pauli_terms)
    if not math.isclose(
        l1_norm, normalization_value, rel_tol=0.0, abs_tol=_ALGEBRA_ATOL
    ):
        _fail(
            f"{path}.normalization.value",
            f"declares {normalization_value}, but Pauli coefficient L1 "
            f"norm is {l1_norm}",
        )

    term_support: set[int] = set()
    for term in pauli_terms:
        for label_position, pauli in enumerate(term.label):
            if pauli != "I":
                term_support.add(n_qubits - 1 - label_position)
    if term_support != set(support_orbitals):
        _fail(
            f"{path}.pauli_terms",
            "non-identity Pauli support does not equal support_orbitals",
        )

    return CEOPoolOperator(
        index=index,
        ceo_type=ceo_type,
        source_orbitals=source_orbitals,
        target_orbitals=target_orbitals,
        support_orbitals=support_orbitals,
        normalization_kind="l1_pauli_coefficients",
        normalization_value=normalization_value,
        pauli_terms=pauli_terms,
        digest=_digest_string(payload["digest"], f"{path}.digest"),
        n_qubits=n_qubits,
    )


def _parse_orbital_groups(
    payload: Any, path: str, n_qubits: int
) -> tuple[tuple[int, ...], ...]:
    groups = _sequence(payload, path)
    if not groups:
        _fail(path, "must contain at least one constituent")
    result: list[tuple[int, ...]] = []
    for group_index, group_payload in enumerate(groups):
        group_path = f"{path}[{group_index}]"
        values = _sequence(group_payload, group_path)
        if not values:
            _fail(group_path, "must not be empty")
        group = tuple(
            _orbital_index(value, f"{group_path}[{i}]", n_qubits)
            for i, value in enumerate(values)
        )
        if len(set(group)) != len(group):
            _fail(group_path, "must not contain duplicate orbitals")
        result.append(group)
    return tuple(result)


def _parse_pauli_terms(
    payload: Any,
    path: str,
    n_qubits: int,
    *,
    coefficient_kind: str,
) -> tuple[PauliTerm, ...]:
    values = _sequence(payload, path)
    if not values:
        _fail(path, "must contain at least one Pauli term")
    result: list[PauliTerm] = []
    labels: list[str] = []
    for index, value in enumerate(values):
        term_path = f"{path}[{index}]"
        term = _mapping(value, term_path)
        _expect_keys(term, {"label", "coefficient"}, term_path)
        label = _string(term["label"], f"{term_path}.label")
        if len(label) != n_qubits or not _PAULI_RE.fullmatch(label):
            _fail(
                f"{term_path}.label",
                f"must contain exactly {n_qubits} uppercase I/X/Y/Z symbols",
            )
        coefficient_payload = _mapping(
            term["coefficient"], f"{term_path}.coefficient"
        )
        _expect_keys(
            coefficient_payload,
            {"real", "imag"},
            f"{term_path}.coefficient",
        )
        coefficient = complex(
            _finite_number(
                coefficient_payload["real"],
                f"{term_path}.coefficient.real",
            ),
            _finite_number(
                coefficient_payload["imag"],
                f"{term_path}.coefficient.imag",
            ),
        )
        if abs(coefficient) <= _ALGEBRA_ATOL:
            _fail(f"{term_path}.coefficient", "must be nonzero")
        if (
            coefficient_kind == "anti_hermitian"
            and abs(coefficient.real) > _ALGEBRA_ATOL
        ):
            _fail(
                f"{term_path}.coefficient.real",
                "must be zero for an anti-Hermitian Pauli expansion",
            )
        if (
            coefficient_kind == "hermitian"
            and abs(coefficient.imag) > _ALGEBRA_ATOL
        ):
            _fail(
                f"{term_path}.coefficient.imag",
                "must be zero for a Hermitian Hamiltonian",
            )
        labels.append(label)
        result.append(PauliTerm(label, coefficient))

    if labels != sorted(labels):
        _fail(path, "Pauli terms must be sorted lexicographically by label")
    if len(set(labels)) != len(labels):
        _fail(path, "duplicate Pauli labels are not allowed")
    return tuple(result)


def _sparse_pauli_op(terms: Sequence[PauliTerm]) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [(term.label, term.coefficient) for term in terms]
    )


def _real_if_close(value: complex) -> complex | float:
    if abs(value.imag) <= _ALGEBRA_ATOL:
        return float(value.real)
    return value


def _normalize_for_digest(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CEOManifestError(
                    f"{path}: canonical JSON object keys must be strings"
                )
            result[key] = _normalize_for_digest(
                item, path=f"{path}.{key}"
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_for_digest(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is float:
        if not math.isfinite(value):
            raise CEOManifestError(
                f"{path}: canonical JSON numbers must be finite"
            )
        return 0.0 if value == 0.0 else value
    if value is None or type(value) in {str, int, bool}:
        return value
    raise CEOManifestError(
        f"{path}: {type(value).__name__} is not canonical JSON data"
    )


def _reject_existing_digest(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        if "digest" in value:
            _fail(
                f"{path}.digest",
                "must be absent before attach_manifest_digests()",
            )
        for key, item in value.items():
            _reject_existing_digest(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_existing_digest(item, f"{path}[{index}]")


def _verify_section_digest(section: Mapping[str, Any], path: str) -> None:
    actual = _digest_string(section["digest"], f"{path}.digest")
    content = {key: value for key, value in section.items() if key != "digest"}
    expected = canonical_digest(content)
    if actual != expected:
        _fail(
            f"{path}.digest",
            f"digest mismatch: expected {expected}, found {actual}",
        )


def _expect_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        _fail(path, "; ".join(details))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, f"must be a JSON object, found {type(value).__name__}")
    return value


def _mutable_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"must be a JSON object, found {type(value).__name__}")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(path, f"must be a JSON array, found {type(value).__name__}")
    return value


def _mutable_sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, f"must be a JSON array, found {type(value).__name__}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, f"must be a string, found {type(value).__name__}")
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        _fail(path, f"must be an integer, found {type(value).__name__}")
    return value


def _finite_number(value: Any, path: str) -> float:
    if type(value) not in {int, float}:
        _fail(path, f"must be a number, found {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    return result


def _orbital_index(value: Any, path: str, n_qubits: int) -> int:
    result = _integer(value, path)
    if not 0 <= result < n_qubits:
        _fail(path, f"must be in [0, {n_qubits})")
    return result


def _digest_string(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _DIGEST_RE.fullmatch(result):
        _fail(path, "must have form 'sha256:' followed by 64 lowercase hex digits")
    return result


def _expect_literal(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(path, f"must equal {expected!r}")


def _fail(path: str, message: str) -> None:
    raise CEOManifestError(f"{path}: {message}")


__all__ = [
    "CEOManifest",
    "CEOManifestError",
    "CEOManifestProvenance",
    "CEOManifestProvenanceEntry",
    "CEOPoolOperator",
    "CEOSource",
    "PauliTerm",
    "POOL_ONLY_ARTIFACT_KIND",
    "POOL_ONLY_SCHEMA",
    "POOL_ONLY_SCHEMA_VERSION",
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA",
    "PROVENANCE_SCHEMA_VERSION",
    "REGISTRY_SCHEMA",
    "REGISTRY_SCHEMA_VERSION",
    "SCHEMA",
    "SCHEMA_VERSION",
    "attach_manifest_digests",
    "attach_provenance_digest",
    "canonical_digest",
    "canonical_json_bytes",
    "interleaved_to_blocked_index",
    "load_ceo_manifest",
    "load_ceo_manifest_for_system",
    "load_ceo_manifest_provenance",
    "parse_ceo_manifest",
    "parse_ceo_manifest_provenance",
    "remap_bitstring_interleaved_to_blocked",
    "remap_pauli_label_interleaved_to_blocked",
]
