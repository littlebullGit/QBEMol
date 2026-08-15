#!/usr/bin/env python
"""Run the n-butane QBE-VQE scan with C2-C3 distances at 1.30, 1.54, and 2.10 Angstroms
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

QBEMOL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = QBEMOL_ROOT / "data"
DEFAULT_OUTPUT_DIR = QBEMOL_ROOT / "results" / "n_butane_qbe_vqe"
ASSET_PATH = DATA_DIR / "n_butane_cid7843.sdf"
RESULT_SCHEMA = "qbemol.n-butane-qbe-vqe.run.v1"
DRY_RUN_SCHEMA = "qbemol.n-butane-qbe-vqe.dry-run.v1"
DISTANCES_ANGSTROM = (1.30, 1.54, 2.10)
ANCHOR_DISTANCE_ANGSTROM = 1.54
SELECTOR_LABELS = {
    "greedy": "greedy_gradient",
    "lookahead": "always_top5_energy",
}
GENERATED_SECTOR_POOL = "__generated_ovp_ceo_pool__"
ACTIVE_WINDOW_SPEC = {
    "policy_id": "W12",
    "frozen_occupied_orbitals": 4,
    "discarded_virtual_orbitals": 4,
    "minimum_boundary_gap_ha": 1.0e-8,
}
R5_POLICY = {
    "policy_id": "R5",
    "frozen_core": True,
    "thr_bath": 1.0e-2,
}


@dataclass(frozen=True, slots=True)
class SDFAtom:
    """One raw atom record from the source SDF."""

    raw_index: int
    element: str
    xyz: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CanonicalAtom:
    """One canonical atom in deterministic C1..C4 / H(parent) order."""

    raw_index: int
    label: str
    element: str
    xyz: tuple[float, float, float]
    parent_carbon: str | None


@dataclass(frozen=True, slots=True)
class CanonicalButaneGeometry:
    """Canonical source or scanned butane geometry."""

    atoms: tuple[CanonicalAtom, ...]
    bonds: tuple[tuple[int, int, int], ...]
    raw_to_canonical: tuple[tuple[int, str], ...]
    source_sha256: str


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fresh outputs.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        default=[f"{value:.2f}" for value in DISTANCES_ANGSTROM],
        help="Subset of scan distances to run, e.g. 1.30 2.10.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=tuple(SELECTOR_LABELS),
        default=tuple(SELECTOR_LABELS),
        help="Selector treatments to run for each chosen distance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assets and emit a production plan without running chemistry.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=range(4),
        default=1,
        help="Verbosity passed to the bundled QuEmb/VQE stack in compute mode.",
    )
    return parser


def _json_default(value: Any) -> Any:
    """Convert common scientific values to JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON payload with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest of one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    """Return a stable SHA-256 for one JSON-compatible payload."""

    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _workspace_path(path: Path) -> str:
    """Return a stable path string rooted at QBEMol."""

    try:
        return path.resolve().relative_to(QBEMOL_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _parse_fixed_int(line: str, start: int, end: int, fallback: int) -> int:
    """Parse one fixed-width integer field from an SDF line."""

    field = line[start:end].strip()
    return int(field) if field else int(line.split()[fallback])


def parse_sdf_text(
    text: str,
) -> tuple[tuple[SDFAtom, ...], tuple[tuple[int, int, int], ...]]:
    """Parse the pinned V2000 SDF source record."""

    lines = text.splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise ValueError("Expected an MDL V2000 SDF record")
    atom_count = _parse_fixed_int(lines[3], 0, 3, 0)
    bond_count = _parse_fixed_int(lines[3], 3, 6, 1)
    if atom_count <= 0 or len(lines) < 4 + atom_count + bond_count:
        raise ValueError("SDF counts line is inconsistent with the record")
    atoms: list[SDFAtom] = []
    for raw_index, line in enumerate(lines[4 : 4 + atom_count], start=1):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed SDF atom line {raw_index}")
        atoms.append(
            SDFAtom(
                raw_index=raw_index,
                element=fields[3].capitalize(),
                xyz=(float(fields[0]), float(fields[1]), float(fields[2])),
            )
        )
    bonds: list[tuple[int, int, int]] = []
    start = 4 + atom_count
    for bond_index, line in enumerate(lines[start : start + bond_count], start=1):
        first = _parse_fixed_int(line, 0, 3, 0)
        second = _parse_fixed_int(line, 3, 6, 1)
        order = _parse_fixed_int(line, 6, 9, 2)
        if first == second:
            raise ValueError(f"Self bond in SDF line {bond_index}")
        bonds.append((min(first, second), max(first, second), order))
    return tuple(atoms), tuple(sorted(bonds))


def canonicalize_butane_sdf(text: str) -> CanonicalButaneGeometry:
    """Return deterministic C1..C4 and grouped-hydrogen ordering."""

    atoms, bonds = parse_sdf_text(text)
    elements = {atom.raw_index: atom.element for atom in atoms}
    carbons = sorted(index for index, element in elements.items() if element == "C")
    hydrogens = sorted(index for index, element in elements.items() if element == "H")
    if len(atoms) != 14 or len(carbons) != 4 or len(hydrogens) != 10:
        raise ValueError("Pinned asset must contain exactly C4H10")
    adjacency = {atom.raw_index: set() for atom in atoms}
    for first, second, order in bonds:
        if order != 1:
            raise ValueError("Canonical butane asset must contain only single bonds")
        adjacency[first].add(second)
        adjacency[second].add(first)
    carbon_graph = {
        carbon: sorted(
            neighbor for neighbor in adjacency[carbon] if neighbor in carbons
        )
        for carbon in carbons
    }
    terminals = sorted(
        carbon for carbon, neighbors in carbon_graph.items() if len(neighbors) == 1
    )
    if len(terminals) != 2:
        raise ValueError("Heavy-atom graph must be a four-carbon path")
    path = [terminals[0]]
    previous: int | None = None
    while len(path) < 4:
        choices = [item for item in carbon_graph[path[-1]] if item != previous]
        if len(choices) != 1:
            raise ValueError("Carbon path is disconnected or ambiguous")
        previous, next_carbon = path[-1], choices[0]
        path.append(next_carbon)
    carbon_labels = {raw: f"C{position}" for position, raw in enumerate(path, start=1)}
    hydrogens_by_carbon: dict[int, list[int]] = {carbon: [] for carbon in path}
    for hydrogen in hydrogens:
        parents = [neighbor for neighbor in adjacency[hydrogen] if neighbor in carbons]
        if len(parents) != 1:
            raise ValueError(f"Hydrogen {hydrogen} has ambiguous parentage")
        hydrogens_by_carbon[parents[0]].append(hydrogen)
    if [len(hydrogens_by_carbon[carbon]) for carbon in path] != [3, 2, 2, 3]:
        raise ValueError("Hydrogen parentage is inconsistent with n-butane")
    by_index = {atom.raw_index: atom for atom in atoms}
    canonical_atoms: list[CanonicalAtom] = []
    raw_to_label: dict[int, str] = {}
    for carbon in path:
        label = carbon_labels[carbon]
        raw_to_label[carbon] = label
        canonical_atoms.append(
            CanonicalAtom(carbon, label, "C", by_index[carbon].xyz, None)
        )
    for position, carbon in enumerate(path, start=1):
        for ordinal, hydrogen in enumerate(
            sorted(hydrogens_by_carbon[carbon]), start=1
        ):
            label = f"H(C{position})-{ordinal}"
            raw_to_label[hydrogen] = label
            canonical_atoms.append(
                CanonicalAtom(
                    hydrogen,
                    label,
                    "H",
                    by_index[hydrogen].xyz,
                    f"C{position}",
                )
            )
    return CanonicalButaneGeometry(
        atoms=tuple(canonical_atoms),
        bonds=bonds,
        raw_to_canonical=tuple(sorted(raw_to_label.items())),
        source_sha256=_hash_file(ASSET_PATH),
    )


def load_canonical_butane() -> CanonicalButaneGeometry:
    """Load the standalone source conformer from the bundled SDF."""

    return canonicalize_butane_sdf(ASSET_PATH.read_text(encoding="utf-8"))


def _subtract(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float]:
    """Subtract two 3-vectors."""

    return tuple(float(first[index]) - float(second[index]) for index in range(3))


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute a 3-vector dot product."""

    return sum(float(first[index]) * float(second[index]) for index in range(3))


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of one 3-vector."""

    return math.sqrt(_dot(vector, vector))


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Euclidean distance between two points."""

    return _norm(_subtract(first, second))


def backbone_dihedral_deg(geometry: CanonicalButaneGeometry) -> float:
    """Return |C1-C2-C3-C4| in degrees."""

    points = [atom.xyz for atom in geometry.atoms[:4]]
    b0 = _subtract(points[0], points[1])
    b1 = _subtract(points[2], points[1])
    b2 = _subtract(points[3], points[2])
    b1_norm = _norm(b1)
    if b1_norm == 0.0:
        raise ValueError("C2 and C3 coincide")
    unit = tuple(value / b1_norm for value in b1)
    v = tuple(b0[index] - _dot(b0, unit) * unit[index] for index in range(3))
    w = tuple(b2[index] - _dot(b2, unit) * unit[index] for index in range(3))
    angle = math.degrees(math.atan2(_dot(_cross(unit, v), w), _dot(v, w)))
    return abs(angle)


def _cross(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float]:
    """Compute a 3-vector cross product."""

    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def validate_anti_geometry(
    geometry: CanonicalButaneGeometry, minimum_deg: float = 170.0
) -> float:
    """Validate that the pinned source conformer is anti."""

    dihedral = backbone_dihedral_deg(geometry)
    if dihedral + 1.0e-12 < minimum_deg:
        raise ValueError(f"Pinned conformer is not anti: {dihedral:.8f} degrees")
    return dihedral


def _half_indices(
    geometry: CanonicalButaneGeometry, parents: set[str]
) -> tuple[int, ...]:
    """Return the atom indices on one rigid backbone half."""

    return tuple(
        index
        for index, atom in enumerate(geometry.atoms)
        if atom.label in parents or atom.parent_carbon in parents
    )


def assert_rigid_half_invariants(
    source: CanonicalButaneGeometry,
    target: CanonicalButaneGeometry,
    tolerance: float = 1.0e-10,
) -> None:
    """Ensure intra-half distances are unchanged by the rigid scan."""

    for parents in ({"C1", "C2"}, {"C3", "C4"}):
        indices = _half_indices(source, parents)
        for first in indices:
            for second in indices:
                if first >= second:
                    continue
                before = distance(source.atoms[first].xyz, source.atoms[second].xyz)
                after = distance(target.atoms[first].xyz, target.atoms[second].xyz)
                if abs(before - after) > tolerance:
                    raise RuntimeError(
                        f"Rigid-half invariant changed for atom pair {first}/{second}"
                    )


def scan_geometry(
    source: CanonicalButaneGeometry,
    target_distance_angstrom: float,
) -> CanonicalButaneGeometry:
    """Rigidly translate the C3/C4 half along the original C2-to-C3 axis."""

    target = float(target_distance_angstrom)
    if target not in DISTANCES_ANGSTROM:
        raise ValueError(f"Distance must be one of {DISTANCES_ANGSTROM}")
    c2, c3 = source.atoms[1], source.atoms[2]
    vector = _subtract(c3.xyz, c2.xyz)
    current = _norm(vector)
    shift = tuple((target - current) * value / current for value in vector)
    right_parents = {"C3", "C4"}
    shifted = []
    for atom in source.atoms:
        on_right = atom.label in right_parents or atom.parent_carbon in right_parents
        xyz = tuple(
            atom.xyz[index] + (shift[index] if on_right else 0.0) for index in range(3)
        )
        shifted.append(
            CanonicalAtom(
                atom.raw_index, atom.label, atom.element, xyz, atom.parent_carbon
            )
        )
    result = CanonicalButaneGeometry(
        atoms=tuple(shifted),
        bonds=source.bonds,
        raw_to_canonical=source.raw_to_canonical,
        source_sha256=source.source_sha256,
    )
    observed = distance(result.atoms[1].xyz, result.atoms[2].xyz)
    if abs(observed - target) > 1.0e-10:
        raise RuntimeError(f"Rigid scan produced {observed}, expected {target}")
    assert_rigid_half_invariants(source, result)
    return result


def geometry_payload(geometry: CanonicalButaneGeometry) -> dict[str, Any]:
    """Return a JSON-safe summary of one canonical geometry."""

    body = {
        "source_sha256": geometry.source_sha256,
        "central_distance_angstrom": distance(
            geometry.atoms[1].xyz, geometry.atoms[2].xyz
        ),
        "backbone_dihedral_abs_deg": backbone_dihedral_deg(geometry),
        "atoms": [
            {
                "label": atom.label,
                "element": atom.element,
                "xyz": list(atom.xyz),
                "parent_carbon": atom.parent_carbon,
            }
            for atom in geometry.atoms
        ],
    }
    return {**body, "digest": _json_digest(body)}


def _xyz_for_geometry(geometry: CanonicalButaneGeometry, comment: str) -> str:
    """Convert one canonical geometry into XYZ text."""

    lines = [str(len(geometry.atoms)), comment]
    for atom in geometry.atoms:
        x, y, z = atom.xyz
        lines.append(f"{atom.element} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def _load_compute_modules() -> dict[str, Any]:
    """Import the bundled standalone compute stack only when needed."""

    from pyscf import gto, scf
    from quemb.molbe import BE, fragmentate
    from quemb.molbe.solver import FCI_ArgsUser
    from quemb.molbe.vqe_solver import (
        ActiveSpaceSpec,
        VQE_ArgsUser,
        _GENERATED_SECTOR_POOL,
        _exact_sparse_ceo_manifest_context,
        clear_ansatz_cache,
        reset_vqe_state,
    )
    from quemb.molbe.sector_adapt_vqe import generate_compact_ovp_ceo_pool
    from quemb.shared.config import settings
    from quemb.shared.manage_scratch import WorkDir

    return {
        "BE": BE,
        "fragmentate": fragmentate,
        "FCI_ArgsUser": FCI_ArgsUser,
        "ActiveSpaceSpec": ActiveSpaceSpec,
        "VQE_ArgsUser": VQE_ArgsUser,
        "generated_sector_pool": _GENERATED_SECTOR_POOL,
        "sector_pool_context": _exact_sparse_ceo_manifest_context,
        "clear_ansatz_cache": clear_ansatz_cache,
        "reset_vqe_state": reset_vqe_state,
        "generate_compact_ovp_ceo_pool": generate_compact_ovp_ceo_pool,
        "settings": settings,
        "WorkDir": WorkDir,
        "gto": gto,
        "scf": scf,
    }


def build_pyscf_molecule(modules: Mapping[str, Any], geometry: CanonicalButaneGeometry):
    """Build one standalone PySCF molecule."""

    gto = modules["gto"]
    return gto.M(
        atom=[(atom.element, atom.xyz) for atom in geometry.atoms],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="angstrom",
        verbose=0,
    )


def assert_molecule_contract(molecule: Any) -> None:
    """Validate the neutral, closed-shell STO-3G n-butane input contract."""

    basis = molecule.basis
    if isinstance(basis, str):
        normalized_basis = basis.lower()
    elif isinstance(basis, Mapping):
        basis_values = {str(value).lower() for value in basis.values()}
        normalized_basis = basis_values.pop() if len(basis_values) == 1 else "mixed"
    else:
        normalized_basis = str(basis).lower()
    symbols = [molecule.atom_pure_symbol(index) for index in range(molecule.natm)]
    if normalized_basis != "sto-3g":
        raise ValueError(f"Production basis must be STO-3G, got {basis!r}")
    if int(molecule.charge) != 0 or int(molecule.spin) != 0:
        raise ValueError("Production n-butane must be neutral and closed shell")
    if symbols != ["C", "C", "C", "C", *(["H"] * 10)]:
        raise ValueError("Production atom order must be C1-C4 followed by ten H atoms")


def assert_locked_butane_topology(fragments: Any) -> None:
    """Validate the two-fragment conventional BE2 topology."""

    motifs = tuple({int(item) for item in motif} for motif in fragments.motifs_per_frag)
    origins = tuple(int(item) for item in fragments.origin_per_frag)
    hydrogen_groups = tuple(
        {int(item) for item in fragments.H_per_motif[index]} for index in range(4)
    )
    if int(fragments.n_frag) != 2 or motifs != ({0, 1, 2}, {1, 2, 3}):
        raise RuntimeError(
            "Locked BE2 must contain the C1-C2-C3 and C2-C3-C4 fragments"
        )
    if origins != (1, 2):
        raise RuntimeError("Locked BE2 fragment origins must be C2 and C3")
    if hydrogen_groups != ({4, 5, 6}, {7, 8}, {9, 10}, {11, 12, 13}):
        raise RuntimeError("Locked BE2 hydrogen parentage must remain 3/2/2/3")


def run_rhf(modules: Mapping[str, Any], molecule: Any, verbose: int):
    """Run the locked RHF setup for one geometry."""

    scf_module = modules["scf"]
    mean_field = scf_module.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.verbose = verbose
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("RHF did not converge")
    return mean_field


def _anchor_fragment_template(
    modules: Mapping[str, Any], anchor_molecule: Any, frozen_core: bool
):
    """Build the anchor autogen BE2 fragment template."""

    fragmentate = modules["fragmentate"]
    return fragmentate(
        mol=anchor_molecule,
        frag_type="autogen",
        n_BE=2,
        frozen_core=frozen_core,
        iao_valence_basis=None,
        print_frags=False,
    )


def build_locked_butane_be2_fragpart(
    modules: Mapping[str, Any],
    target_mol: Any,
    frozen_core: bool,
):
    """Clone the anchor autogen BE2 topology onto one target molecule."""

    assert_molecule_contract(target_mol)
    source = load_canonical_butane()
    validate_anti_geometry(source)
    anchor_geometry = scan_geometry(source, ANCHOR_DISTANCE_ANGSTROM)
    anchor_mol = build_pyscf_molecule(modules, anchor_geometry)
    if int(anchor_mol.nao) != int(target_mol.nao):
        raise RuntimeError("Anchor and target STO-3G AO counts differ")
    if list(anchor_mol.ao_labels(fmt=False)) != list(target_mol.ao_labels(fmt=False)):
        raise RuntimeError("Anchor and target STO-3G AO ordering differs")
    template = _anchor_fragment_template(modules, anchor_mol, frozen_core)
    assert_locked_butane_topology(template)
    clone = copy.deepcopy(template)
    clone.mol = target_mol
    assert_locked_butane_topology(clone)
    return clone


def build_active_space_spec(modules: Mapping[str, Any]) -> Any:
    """Return the locked W12 active-space specification."""

    ActiveSpaceSpec = modules["ActiveSpaceSpec"]
    return ActiveSpaceSpec(
        frozen_occupied_orbitals=ACTIVE_WINDOW_SPEC["frozen_occupied_orbitals"],
        discarded_virtual_orbitals=ACTIVE_WINDOW_SPEC["discarded_virtual_orbitals"],
        minimum_boundary_gap_ha=ACTIVE_WINDOW_SPEC["minimum_boundary_gap_ha"],
    )


def build_fci_args(modules: Mapping[str, Any]) -> Any:
    """Return the matched W12 FCI solver arguments."""

    FCI_ArgsUser = modules["FCI_ArgsUser"]
    return FCI_ArgsUser(active_space=build_active_space_spec(modules))


def build_vqe_args(
    modules: Mapping[str, Any],
    output_dir: Path,
    selector_policy: str,
    verbose: int,
) -> Any:
    """Return the matched W12 fixed-sector VQE arguments."""

    VQE_ArgsUser = modules["VQE_ArgsUser"]
    return VQE_ArgsUser(
        hamiltonian_dir=str(output_dir / "hamiltonians"),
        ansatz_type="adapt_sector",
        estimator_type="direct_sv",
        optimizer_name="SLSQP",
        frozen_core="manual",
        frozen_core_num_orbitals=0,
        active_space=build_active_space_spec(modules),
        stage1_max_iter=200,
        stage2_max_iter=200,
        stage3_max_iter=200,
        stage1_energy_tol=1.0e-10,
        stage2_energy_tol=1.0e-10,
        stage3_energy_tol=1.0e-10,
        adapt_gradient_threshold=1.0e-3,
        adapt_eigenvalue_threshold=0.0,
        adapt_max_iterations=20,
        adapt_check_cyclicity=False,
        adapt_sector_selector_policy=selector_policy,
        adapt_sector_selector_top_k=5,
        adapt_sector_probe_max_iterations=200,
        adapt_sector_probe_ftol=1.0e-10,
        adapt_sector_energy_tie_tolerance=1.0e-10,
        max_restarts=1,
        parallel_restarts=1,
        random_seed=281,
        warm_start=False,
        direct_sv_use_exact_jacobian=False,
        track_iteration_history=True,
        verbose=verbose,
    )


@contextmanager
def capture_beopt_state():
    """Capture the final local BE optimization state without Brown scaffolding."""

    import quemb.molbe.opt as opt_module

    captured: dict[str, Any] = {}
    original = opt_module.BEOPT.optimize

    def wrapped(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            captured["final"] = {
                "iterations_completed": int(self.iter),
                "chemical_potential_residual": float(self.err),
                "final_potentials": [float(value) for value in self.pot],
                "convergence_tolerance": float(self.conv_tol),
                "maximum_iterations": int(self.max_space),
                "only_chemical_potential": bool(self.only_chem),
            }

    opt_module.BEOPT.optimize = wrapped
    try:
        yield captured
    finally:
        opt_module.BEOPT.optimize = original


def prepare_be(
    modules: Mapping[str, Any],
    distance_angstrom: float,
    scratch_dir: Path,
    verbose: int,
) -> tuple[Any, Any, Any, Any]:
    """Build one locked R5 BE object for the requested distance."""

    geometry = scan_geometry(load_canonical_butane(), distance_angstrom)
    molecule = build_pyscf_molecule(modules, geometry)
    mean_field = run_rhf(modules, molecule, verbose)
    fragments = build_locked_butane_be2_fragpart(
        modules,
        molecule,
        frozen_core=bool(R5_POLICY["frozen_core"]),
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    modules["settings"].SCRATCH_ROOT = scratch_dir.resolve()
    work_dir = modules["WorkDir"](scratch_dir, cleanup_at_end=False, ensure_empty=True)
    be = modules["BE"](
        mean_field,
        fragments,
        thr_bath=R5_POLICY["thr_bath"],
        eri_file="eri_file.h5",
        scratch_dir=work_dir,
    )
    return geometry, mean_field, fragments, be


def run_fci_be(
    modules: Mapping[str, Any],
    distance_angstrom: float,
    output_dir: Path,
    verbose: int,
) -> dict[str, Any]:
    """Run one matched W12 FCI-BE calculation."""

    scratch = output_dir / "scratch"
    _, mean_field, _, be = prepare_be(modules, distance_angstrom, scratch, verbose)
    started = perf_counter()
    with capture_beopt_state() as captured:
        be.optimize(
            solver="FCI",
            solver_args=build_fci_args(modules),
            only_chem=True,
            conv_tol=1.0e-8,
            max_iter=50,
            nproc=1,
        )
    wall_s = perf_counter() - started
    record = {
        "distance_angstrom": distance_angstrom,
        "solver": "FCI",
        "energy_ha": float(be.ebe_tot),
        "rhf_energy_ha": float(mean_field.e_tot),
        "wall_s": wall_s,
        "be_optimization": captured.get("final", {}),
        "active_space": copy.deepcopy(
            getattr(be.Fobjs[0], "active_space_provenance", {})
        ),
    }
    _write_json(output_dir / "result.json", record)
    return record


def run_qbe_selector(
    modules: Mapping[str, Any],
    distance_angstrom: float,
    selector_label: str,
    output_dir: Path,
    verbose: int,
) -> dict[str, Any]:
    """Run one matched W12 fixed-sector QBE-VQE cell."""

    selector_policy = SELECTOR_LABELS[selector_label]
    scratch = output_dir / "scratch"
    _, mean_field, _, be = prepare_be(modules, distance_angstrom, scratch, verbose)
    modules["clear_ansatz_cache"]()
    modules["reset_vqe_state"]()
    started = perf_counter()
    with modules["sector_pool_context"](modules["generated_sector_pool"]):
        with capture_beopt_state() as captured:
            be.optimize(
                solver="VQE",
                solver_args=build_vqe_args(
                    modules, output_dir, selector_policy, verbose
                ),
                only_chem=True,
                conv_tol=1.0e-6,
                max_iter=50,
                nproc=1,
            )
    wall_s = perf_counter() - started
    fixed_sector = []
    for fragment in be.Fobjs:
        for evidence in getattr(fragment, "sector_adapt_evidence", []):
            fixed_sector.append(copy.deepcopy(evidence))
    pool = modules["generate_compact_ovp_ceo_pool"](24)
    record = {
        "distance_angstrom": distance_angstrom,
        "solver": "VQE",
        "selector": selector_label,
        "selector_policy": selector_policy,
        "energy_ha": float(be.ebe_tot),
        "rhf_energy_ha": float(mean_field.e_tot),
        "wall_s": wall_s,
        "be_optimization": captured.get("final", {}),
        "active_space": copy.deepcopy(
            getattr(be.Fobjs[0], "active_space_provenance", {})
        ),
        "fixed_sector_pool": {
            "n_qubits": 24,
            "n_electrons": 12,
            "pool_size": len(pool),
        },
        "fragment_evidence": fixed_sector,
    }
    _write_json(output_dir / "result.json", record)
    return record


def dry_run_payload(
    distances: Sequence[float],
    selectors: Sequence[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Return the pure-Python dry-run plan payload."""

    source = load_canonical_butane()
    anti = validate_anti_geometry(source)
    geometry_dir = output_dir.resolve() / "geometries"
    geometries = []
    for distance_angstrom in distances:
        geometry = scan_geometry(source, distance_angstrom)
        geometries.append(
            {
                "distance_angstrom": distance_angstrom,
                "payload": geometry_payload(geometry),
                "planned_xyz": _workspace_path(
                    geometry_dir / f"n_butane_c2c3_{distance_angstrom:.2f}A.xyz"
                ),
            }
        )
    return {
        "schema": DRY_RUN_SCHEMA,
        "source_asset": _workspace_path(ASSET_PATH),
        "source_sha256": _hash_file(ASSET_PATH),
        "validated_backbone_dihedral_abs_deg": anti,
        "selected_distances_angstrom": list(distances),
        "selected_selectors": list(selectors),
        "reduction_policy": copy.deepcopy(R5_POLICY),
        "active_window": copy.deepcopy(ACTIVE_WINDOW_SPEC),
        "generated_sector_pool_sentinel": GENERATED_SECTOR_POOL,
        "geometries": geometries,
    }


def write_geometries(output_dir: Path, distances: Sequence[float]) -> list[str]:
    """Write the selected scanned geometries as XYZ files."""

    source = load_canonical_butane()
    geometry_dir = output_dir / "geometries"
    written = []
    for distance_angstrom in distances:
        geometry = scan_geometry(source, distance_angstrom)
        path = geometry_dir / f"n_butane_c2c3_{distance_angstrom:.2f}A.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _xyz_for_geometry(
                geometry,
                f"n-butane C2-C3 = {distance_angstrom:.2f} Angstrom",
            ),
            encoding="utf-8",
        )
        written.append(_workspace_path(path))
    return written


def parse_distances(values: Sequence[str]) -> tuple[float, ...]:
    """Normalize the requested distance list."""

    parsed = tuple(round(float(value), 2) for value in values)
    if any(value not in DISTANCES_ANGSTROM for value in parsed):
        raise ValueError(f"Distances must be chosen from {DISTANCES_ANGSTROM}")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Distances must not repeat")
    return parsed


def run_production(args: argparse.Namespace) -> Path:
    """Run the fresh standalone production lane and return result.json."""

    modules = _load_compute_modules()
    distances = parse_distances(args.distances)
    output_dir = args.output_dir.resolve()
    written_xyz = write_geometries(output_dir, distances)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "source_asset": _workspace_path(ASSET_PATH),
        "selected_distances_angstrom": list(distances),
        "selected_selectors": list(args.selectors),
        "written_xyz": written_xyz,
        "cells": {},
    }
    for distance_angstrom in distances:
        distance_dir = output_dir / f"distance_{distance_angstrom:.2f}A"
        fci_dir = distance_dir / "fci_be"
        fci_record = run_fci_be(modules, distance_angstrom, fci_dir, args.verbose)
        selector_records = {}
        for selector_label in args.selectors:
            selector_dir = distance_dir / selector_label
            selector_records[selector_label] = run_qbe_selector(
                modules,
                distance_angstrom,
                selector_label,
                selector_dir,
                args.verbose,
            )
        result["cells"][f"{distance_angstrom:.2f}"] = {
            "fci_be": fci_record,
            "selectors": selector_records,
        }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    distances = parse_distances(args.distances)
    if args.dry_run:
        output_dir = args.output_dir.resolve()
        payload = dry_run_payload(distances, args.selectors, output_dir)
        path = output_dir / "dry_run.json"
        _write_json(path, payload)
        print(f"n-butane dry run: {path}")
        return 0
    result_path = run_production(args)
    print(f"n-butane production result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
