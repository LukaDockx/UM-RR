"""Single relaxed-torsion path model for large-amplitude sampling."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .constants import ANGSTROM_TO_BOHR, BOLTZMANN_EV_PER_K, HARTREE_TO_EV
from .momentum import (
    PeriodicMomentumDistribution,
    periodic_path_momentum_distributions,
    periodic_path_momentum_wavefunctions,
)
from .qchem_io import (
    generate_qchem_input_text,
    parse_standard_orientation_angstrom,
    qchem_template_with_jobtype,
    qchem_template_with_scan_block,
)
from .rotors import build_bond_graph, periodic_second_derivative_matrix
from .serialization import write_json

HARTREE_TO_CM1 = 219474.63
HARTREE_TO_KCAL_MOL = 627.5094740631
MAX_RMSD_PERMUTATION_CANDIDATES = 20000


@dataclass(slots=True)
class RelaxedScanPoint:
    angle_deg: float
    energy_hartree: float
    geometry_angstrom: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "angle_deg": self.angle_deg,
            "energy_hartree": self.energy_hartree,
            "geometry_angstrom": self.geometry_angstrom,
        }


@dataclass(slots=True)
class RelaxedTorsionBasin:
    index: int
    minimum_path_coordinate: float
    minimum_angle_deg: float
    minimum_energy_hartree: float
    left_boundary_path_coordinate: float
    right_boundary_path_coordinate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "minimum_path_coordinate": self.minimum_path_coordinate,
            "minimum_angle_deg": self.minimum_angle_deg,
            "minimum_energy_hartree": self.minimum_energy_hartree,
            "left_boundary_path_coordinate": self.left_boundary_path_coordinate,
            "right_boundary_path_coordinate": self.right_boundary_path_coordinate,
        }


@dataclass(slots=True)
class RelaxedTorsionSolution:
    scan_points: list[RelaxedScanPoint]
    atom_symbols: list[str]
    atom_indices: list[int]
    path_coordinates: np.ndarray
    segment_lengths: np.ndarray
    total_path_length: float
    masses_electron: np.ndarray
    angles_rad: np.ndarray
    geometries_angstrom: np.ndarray
    solver_path_grid: np.ndarray
    potential_hartree: np.ndarray
    potential_reference_hartree: float
    eigenvalues_hartree: np.ndarray
    wavefunctions: np.ndarray
    basins: list[RelaxedTorsionBasin]
    projected_mode_index: int | None
    projection_score: float
    state_convergence: dict | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scan_points": self.scan_points,
            "atom_symbols": self.atom_symbols,
            "atom_indices": self.atom_indices,
            "path_coordinates": self.path_coordinates,
            "segment_lengths": self.segment_lengths,
            "total_path_length": self.total_path_length,
            "masses_electron": self.masses_electron,
            "angles_rad": self.angles_rad,
            "geometries_angstrom": self.geometries_angstrom,
            "solver_path_grid": self.solver_path_grid,
            "potential_hartree": self.potential_hartree,
            "potential_reference_hartree": self.potential_reference_hartree,
            "eigenvalues_hartree": self.eigenvalues_hartree,
            "wavefunctions": self.wavefunctions,
            "basins": self.basins,
            "projected_mode_index": self.projected_mode_index,
            "projection_score": self.projection_score,
            "state_convergence": self.state_convergence,
        }


@dataclass(slots=True)
class RelaxedTorsionSample:
    state: int
    path_coordinate: float
    angle_rad: float
    basin_index: int
    potential_hartree: float
    potential_relative_hartree: float
    state_energy_hartree: float
    expected_kinetic_energy_hartree: float
    path_momentum_direction_au: float
    geometry_angstrom: np.ndarray
    velocity_au: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "path_coordinate": self.path_coordinate,
            "angle_rad": self.angle_rad,
            "angle_deg": float(np.degrees(self.angle_rad)),
            "basin_index": self.basin_index,
            "potential_hartree": self.potential_hartree,
            "potential_relative_hartree": self.potential_relative_hartree,
            "state_energy_hartree": self.state_energy_hartree,
            "expected_kinetic_energy_hartree": self.expected_kinetic_energy_hartree,
            "path_momentum_direction_au": self.path_momentum_direction_au,
            "geometry_angstrom": self.geometry_angstrom,
            "velocity_au": self.velocity_au,
        }


@dataclass(slots=True)
class EquivalentMinimum:
    index: int
    representative_basin_index: int
    basin_indices: list[int]
    rmsd_to_representative_angstrom: list[float]
    geometry_angstrom: np.ndarray
    basin_permutations: dict[int, list[int]]
    basin_rotations: dict[int, np.ndarray]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "representative_basin_index": self.representative_basin_index,
            "basin_indices": self.basin_indices,
            "rmsd_to_representative_angstrom": self.rmsd_to_representative_angstrom,
            "geometry_angstrom": self.geometry_angstrom,
            "basin_permutations": self.basin_permutations,
            "basin_rotations": self.basin_rotations,
        }


@dataclass(slots=True)
class EquivalentMinimumAlignment:
    rmsd_angstrom: float
    permutation: np.ndarray
    rotation: np.ndarray


def parse_relaxed_torsion_scan_output(
    path: str | Path,
) -> tuple[list[RelaxedScanPoint], list[str]]:
    """Parse optimized points from a Q-Chem ``jobtype pes_scan`` output."""
    scan_points: list[RelaxedScanPoint] = []
    last_geometry = None
    last_atoms = None
    current_geom: list[list[float]] = []
    current_atoms: list[str] = []
    in_orientation = False
    skip_lines = 0
    scan_pattern = re.compile(
        r"PES scan,\s*value:\s*([-+0-9.]+)\s+energy:\s*([-+0-9.]+)",
        re.IGNORECASE,
    )

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if "Standard Nuclear Orientation (Angstroms)" in line:
                in_orientation = True
                skip_lines = 2
                current_geom = []
                current_atoms = []
                continue
            if in_orientation:
                if skip_lines:
                    skip_lines -= 1
                    continue
                if "---" in line:
                    if current_geom:
                        last_geometry = np.asarray(current_geom, dtype=float)
                        last_atoms = list(current_atoms)
                    in_orientation = False
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    current_atoms.append(parts[1])
                    current_geom.append([float(parts[2]), float(parts[3]), float(parts[4])])
                continue

            match = scan_pattern.search(line)
            if match:
                if last_geometry is None or last_atoms is None:
                    raise ValueError(f"PES scan point found before geometry in {path}")
                scan_points.append(
                    RelaxedScanPoint(
                        angle_deg=float(match.group(1)),
                        energy_hartree=float(match.group(2)),
                        geometry_angstrom=last_geometry.copy(),
                    )
                )

    if not scan_points or last_atoms is None:
        raise ValueError(f"No relaxed torsion scan points found in {path}")
    return scan_points, last_atoms


def relaxed_torsion_scan_paths(
    scan_dir: str | Path,
    scan_input: str,
    scan_output: str,
) -> tuple[Path, Path]:
    scan_path = Path(scan_dir)
    return scan_path / scan_input, scan_path / scan_output


def write_relaxed_torsion_scan_input(
    input_path: str | Path,
    geometry_angstrom: np.ndarray,
    atom_symbols: Sequence[str],
    *,
    atom_indices: Sequence[int],
    charge: int,
    multiplicity: int,
    qchem_template: str,
    start_deg: float,
    end_deg: float,
    step_deg: float,
) -> Path:
    if len(atom_indices) != 4:
        raise ValueError("relaxed_torsion.atom_indices must contain four 1-based atom indices")
    path = Path(input_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scan_line = (
        "tors "
        + " ".join(str(int(index)) for index in atom_indices)
        + f" {float(start_deg):.8g} {float(end_deg):.8g} {float(step_deg):.8g}"
    )
    template = qchem_template_with_jobtype(qchem_template, "pes_scan")
    template = qchem_template_with_scan_block(template, scan_line)
    path.write_text(
        generate_qchem_input_text(
            geometry_angstrom,
            atom_symbols,
            charge=charge,
            multiplicity=multiplicity,
            template=template,
        ),
        encoding="utf-8",
    )
    (path.parent / "filelist.txt").write_text(path.name + "\n", encoding="utf-8")
    return path


def build_relaxed_torsion_solution(
    scan_output: str | Path,
    *,
    mass_matrix: np.ndarray,
    normal_mode_eigenvectors: np.ndarray,
    normal_mode_frequencies_cm1: Sequence[float],
    frequency_cutoff_cm1: float,
    grid_points: int,
    num_states: int,
    atom_indices: Sequence[int] = (),
    temperature_kelvin: float | None = None,
    population_tail_tolerance: float = 0.01,
) -> RelaxedTorsionSolution:
    scan_points, atom_symbols = parse_relaxed_torsion_scan_output(scan_output)
    merged_points = merge_duplicate_scan_angles(scan_points)
    masses = np.asarray(mass_matrix, dtype=float)[:, 0]
    angles_rad = np.asarray([np.deg2rad(point.angle_deg % 360.0) for point in merged_points])
    energies = np.asarray([point.energy_hartree for point in merged_points], dtype=float)
    geometries = np.asarray([point.geometry_angstrom for point in merged_points], dtype=float)

    order = np.argsort(angles_rad)
    angles_rad = angles_rad[order]
    energies = energies[order]
    geometries = geometries[order]
    merged_points = [merged_points[index] for index in order]

    aligned_geometries = align_geometry_path(geometries, masses)
    segment_lengths = path_segment_lengths(aligned_geometries, masses)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        raise ValueError("Relaxed torsion path has zero mass-weighted length")
    path_coordinates = np.r_[0.0, np.cumsum(segment_lengths[:-1])]
    potential_reference = float(np.min(energies))
    potential = energies - potential_reference
    solver_grid, solver_potential = interpolate_periodic_path_potential(
        path_coordinates,
        potential,
        total_length,
        grid_points=grid_points,
    )
    state_convergence = None
    if temperature_kelvin is None:
        eigenvalues, wavefunctions = solve_periodic_path_schrodinger(
            solver_potential,
            total_length,
            num_states=num_states,
        )
    else:
        from .rotor_states import solve_thermal_rotor_states

        eigenvalues, wavefunctions, state_convergence = solve_thermal_rotor_states(
            solver_potential,
            total_length,
            num_states=num_states,
            temperature_kelvin=temperature_kelvin,
            population_tail_tolerance=population_tail_tolerance,
        )
    basins = identify_relaxed_torsion_basins(
        solver_grid,
        solver_potential,
        total_length,
        path_coordinates,
        angles_rad,
    )
    projected_mode_index, projection_score = project_path_tangent_on_normal_modes(
        aligned_geometries,
        segment_lengths,
        potential,
        masses,
        normal_mode_eigenvectors,
        normal_mode_frequencies_cm1,
        frequency_cutoff_cm1=frequency_cutoff_cm1,
    )
    return RelaxedTorsionSolution(
        scan_points=merged_points,
        atom_symbols=atom_symbols,
        atom_indices=[int(index) for index in atom_indices],
        path_coordinates=path_coordinates,
        segment_lengths=segment_lengths,
        total_path_length=total_length,
        masses_electron=masses,
        angles_rad=angles_rad,
        geometries_angstrom=aligned_geometries,
        solver_path_grid=solver_grid,
        potential_hartree=solver_potential,
        potential_reference_hartree=potential_reference,
        eigenvalues_hartree=eigenvalues,
        wavefunctions=wavefunctions,
        basins=basins,
        projected_mode_index=projected_mode_index,
        projection_score=projection_score,
        state_convergence=state_convergence,
    )


def group_equivalent_minima(
    solution: RelaxedTorsionSolution,
    *,
    rmsd_threshold_angstrom: float,
) -> tuple[list[EquivalentMinimum], dict[int, int]]:
    unique_minima: list[EquivalentMinimum] = []
    basin_to_unique: dict[int, int] = {}
    for basin in solution.basins:
        geometry = interpolate_geometry_from_path(solution, basin.minimum_path_coordinate)
        matched_unique = None
        matched_alignment = None
        for unique in unique_minima:
            alignment = equivalent_minimum_alignment(
                solution.atom_symbols,
                geometry,
                unique.geometry_angstrom,
            )
            if alignment.rmsd_angstrom <= rmsd_threshold_angstrom:
                matched_unique = unique
                matched_alignment = alignment
                break
        if matched_unique is None:
            unique_index = len(unique_minima)
            unique = EquivalentMinimum(
                index=unique_index,
                representative_basin_index=basin.index,
                basin_indices=[basin.index],
                rmsd_to_representative_angstrom=[0.0],
                geometry_angstrom=geometry,
                basin_permutations={
                    basin.index: list(range(len(solution.atom_symbols))),
                },
                basin_rotations={
                    basin.index: np.eye(3),
                },
            )
            unique_minima.append(unique)
            basin_to_unique[basin.index] = unique_index
        else:
            if matched_alignment is None:
                raise RuntimeError("Matched equivalent minimum without an alignment")
            matched_unique.basin_indices.append(basin.index)
            matched_unique.rmsd_to_representative_angstrom.append(
                float(matched_alignment.rmsd_angstrom)
            )
            matched_unique.basin_permutations[basin.index] = [
                int(index) for index in matched_alignment.permutation
            ]
            matched_unique.basin_rotations[basin.index] = matched_alignment.rotation
            basin_to_unique[basin.index] = matched_unique.index
    return unique_minima, basin_to_unique


def equivalent_minimum_alignment(
    atom_symbols: Sequence[str],
    target_geometry_angstrom: np.ndarray,
    reference_geometry_angstrom: np.ndarray,
) -> EquivalentMinimumAlignment:
    """Return the atom mapping and rotation from a representative minimum to a target basin.

    The returned permutation has target atom order: ``reference[permutation]``
    is the reference geometry reordered to match ``target`` before rotation.
    Displacement or velocity arrays from the reference frame can therefore be
    transported to the target frame with ``array[permutation] @ rotation``.
    """
    import rmsd

    atoms = np.asarray(atom_symbols)
    target = np.asarray(target_geometry_angstrom, dtype=float)
    reference = np.asarray(reference_geometry_angstrom, dtype=float)
    best: EquivalentMinimumAlignment | None = None
    for order in graph_restricted_permutation_orders(atoms, target, reference):
        alignment = alignment_for_permutation(target, reference, np.asarray(order, dtype=int))
        if best is None or alignment.rmsd_angstrom < best.rmsd_angstrom:
            best = alignment
    if best is not None:
        return best

    order = rmsd.reorder_hungarian(atoms, atoms, target, reference)
    return alignment_for_permutation(target, reference, np.asarray(order, dtype=int))


def alignment_for_permutation(
    target_geometry_angstrom: np.ndarray,
    reference_geometry_angstrom: np.ndarray,
    permutation: np.ndarray,
) -> EquivalentMinimumAlignment:
    target = np.asarray(target_geometry_angstrom, dtype=float)
    reference = np.asarray(reference_geometry_angstrom, dtype=float)[permutation]
    rotation = kabsch_rotation_matrix(reference, target)
    reference_centered = reference - np.mean(reference, axis=0)
    target_centered = target - np.mean(target, axis=0)
    aligned = reference_centered @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1))))
    return EquivalentMinimumAlignment(
        rmsd_angstrom=rmsd,
        permutation=np.asarray(permutation, dtype=int),
        rotation=rotation,
    )


def permutation_invariant_rmsd_angstrom(
    atom_symbols: Sequence[str],
    geometry_a_angstrom: np.ndarray,
    geometry_b_angstrom: np.ndarray,
) -> float:
    """Return translation/rotation/permutation-invariant RMSD for same-composition structures.

    Atom permutations are restricted by a graph-derived local environment. This is important for
    molecules such as methanol: the three methyl hydrogens are interchangeable, but the hydroxyl
    hydrogen is not.
    """
    import rmsd

    atoms = np.asarray(atom_symbols)
    geometry_a = np.asarray(geometry_a_angstrom, dtype=float)
    geometry_b = np.asarray(geometry_b_angstrom, dtype=float)
    best = math.inf
    for order in graph_restricted_permutation_orders(atoms, geometry_a, geometry_b):
        reordered_b = geometry_b[order]
        best = min(best, float(rmsd.kabsch_rmsd(geometry_a, reordered_b, translate=True)))
    if math.isfinite(best):
        return best

    order = rmsd.reorder_hungarian(atoms, atoms, geometry_a, geometry_b)
    return float(rmsd.kabsch_rmsd(geometry_a, geometry_b[order], translate=True))


def graph_restricted_permutation_orders(
    atom_symbols: Sequence[str],
    geometry_a_angstrom: np.ndarray,
    geometry_b_angstrom: np.ndarray,
    *,
    max_candidates: int = MAX_RMSD_PERMUTATION_CANDIDATES,
) -> list[np.ndarray]:
    signatures_a = atom_environment_signatures(atom_symbols, geometry_a_angstrom)
    signatures_b = atom_environment_signatures(atom_symbols, geometry_b_angstrom)
    if sorted(signatures_a) != sorted(signatures_b):
        return []

    import rmsd

    signatures = sorted(set(signatures_a))
    fixed_order = np.arange(len(atom_symbols), dtype=int)
    brute_groups: list[tuple[list[int], list[int]]] = []
    candidate_count = 1
    atoms = np.asarray(atom_symbols)
    geometry_a = np.asarray(geometry_a_angstrom, dtype=float)
    geometry_b = np.asarray(geometry_b_angstrom, dtype=float)
    for signature in signatures:
        p_group = [index for index, value in enumerate(signatures_a) if value == signature]
        q_group = [index for index, value in enumerate(signatures_b) if value == signature]
        if len(p_group) != len(q_group):
            return []
        if len(p_group) == 1:
            fixed_order[p_group[0]] = q_group[0]
            continue
        group_factorial = math.factorial(len(p_group))
        if len(p_group) <= 7 and candidate_count * group_factorial <= max_candidates:
            brute_groups.append((p_group, q_group))
            candidate_count *= group_factorial
            continue
        sub_order = rmsd.reorder_hungarian(
            atoms[p_group],
            atoms[q_group],
            geometry_a[p_group],
            geometry_b[q_group],
        )
        fixed_order[p_group] = np.asarray(q_group, dtype=int)[sub_order]

    if not brute_groups:
        return [fixed_order]

    orders = []
    permutation_sets = [
        list(permutations(q_group, len(p_group))) for p_group, q_group in brute_groups
    ]
    for choice in product(*permutation_sets):
        order = fixed_order.copy()
        for (p_group, _q_group), q_permutation in zip(brute_groups, choice):
            order[p_group] = np.asarray(q_permutation, dtype=int)
        orders.append(order)
    return orders


def atom_environment_signatures(
    atom_symbols: Sequence[str],
    geometry_angstrom: np.ndarray,
) -> list[str]:
    graph = build_bond_graph(geometry_angstrom, atom_symbols)
    signatures: list[object] = list(atom_symbols)
    for _iteration in range(len(atom_symbols)):
        signatures = [
            (
                atom_symbols[index],
                tuple(sorted(signatures[neighbor] for neighbor in graph[index])),
            )
            for index in range(len(atom_symbols))
        ]
    return [repr(signature) for signature in signatures]


def write_minimum_optimization_inputs(
    minimum_dir: str | Path,
    unique_minima: Sequence[EquivalentMinimum],
    atom_symbols: Sequence[str],
    *,
    charge: int,
    multiplicity: int,
    qchem_template: str,
) -> Path:
    base_dir = Path(minimum_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    template = qchem_template_with_jobtype(qchem_template, "opt")
    for minimum in unique_minima:
        minimum_path = base_dir / f"minimum_{minimum.index}"
        minimum_path.mkdir(parents=True, exist_ok=True)
        input_path = minimum_path / "opt.in"
        input_path.write_text(
            generate_qchem_input_text(
                minimum.geometry_angstrom,
                atom_symbols,
                charge=charge,
                multiplicity=multiplicity,
                template=template,
            ),
            encoding="utf-8",
        )
        lines.append(str(input_path.relative_to(base_dir)))
    filelist_path = base_dir / "opt_filelist.txt"
    filelist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filelist_path


def write_minimum_frequency_inputs(
    minimum_dir: str | Path,
    unique_minima: Sequence[EquivalentMinimum],
    atom_symbols: Sequence[str],
    *,
    charge: int,
    multiplicity: int,
    qchem_template: str,
) -> Path | None:
    base_dir = Path(minimum_dir)
    lines = []
    template = qchem_template_with_jobtype(qchem_template, "freq", extra_rem={"VIBMAN_PRINT": 4})
    for minimum in unique_minima:
        minimum_path = base_dir / f"minimum_{minimum.index}"
        opt_output = minimum_path / "opt.out"
        if not opt_output.exists():
            continue
        geometry, parsed_atoms = parse_standard_orientation_angstrom(opt_output)
        if parsed_atoms != list(atom_symbols):
            raise ValueError(f"Atom order changed in {opt_output}")
        input_path = minimum_path / "freq.in"
        input_path.write_text(
            generate_qchem_input_text(
                geometry,
                atom_symbols,
                charge=charge,
                multiplicity=multiplicity,
                template=template,
            ),
            encoding="utf-8",
        )
        lines.append(str(input_path.relative_to(base_dir)))
    if not lines:
        return None
    filelist_path = base_dir / "freq_filelist.txt"
    filelist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filelist_path


def minimum_frequency_output_path(minimum_dir: str | Path, unique_minimum_index: int) -> Path:
    return Path(minimum_dir) / f"minimum_{unique_minimum_index}" / "freq.out"


def merge_duplicate_scan_angles(
    scan_points: Sequence[RelaxedScanPoint],
) -> list[RelaxedScanPoint]:
    grouped: dict[float, list[RelaxedScanPoint]] = {}
    for point in scan_points:
        key = round(point.angle_deg % 360.0, 8)
        grouped.setdefault(key, []).append(point)

    merged = []
    for key, points in grouped.items():
        energy = float(np.mean([point.energy_hartree for point in points]))
        geometry_point = min(points, key=lambda point: point.energy_hartree)
        angle = key if key < 180.0 else key - 360.0
        merged.append(
            RelaxedScanPoint(
                angle_deg=angle,
                energy_hartree=energy,
                geometry_angstrom=geometry_point.geometry_angstrom.copy(),
            )
        )
    return merged


def align_geometry_path(geometries_angstrom: np.ndarray, masses: Sequence[float]) -> np.ndarray:
    geometries = np.asarray(geometries_angstrom, dtype=float)
    weights = np.asarray(masses, dtype=float)
    aligned = np.empty_like(geometries)
    aligned[0] = geometries[0]
    for index in range(1, len(geometries)):
        aligned[index] = kabsch_align(geometries[index], aligned[index - 1], weights)
    return aligned


def kabsch_align(mobile: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mobile_center = np.average(mobile, axis=0, weights=weights)
    reference_center = np.average(reference, axis=0, weights=weights)
    mobile_centered = mobile - mobile_center
    rotation = kabsch_rotation_matrix(mobile, reference, weights)
    return mobile_centered @ rotation + reference_center


def kabsch_rotation_matrix(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    mobile_array = np.asarray(mobile, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if weights is None:
        mobile_centered = mobile_array - np.mean(mobile_array, axis=0)
        reference_centered = reference_array - np.mean(reference_array, axis=0)
        covariance = mobile_centered.T @ reference_centered
    else:
        weights_array = np.asarray(weights, dtype=float)
        mobile_centered = mobile_array - np.average(mobile_array, axis=0, weights=weights_array)
        reference_centered = reference_array - np.average(
            reference_array,
            axis=0,
            weights=weights_array,
        )
        covariance = (mobile_centered * weights_array[:, None]).T @ reference_centered
    left, _singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    return left @ correction @ right_t


def path_segment_lengths(
    aligned_geometries_angstrom: np.ndarray, masses: Sequence[float]
) -> np.ndarray:
    geometries = np.asarray(aligned_geometries_angstrom, dtype=float)
    masses_array = np.asarray(masses, dtype=float)
    lengths = []
    for index in range(len(geometries)):
        next_index = (index + 1) % len(geometries)
        if next_index == 0:
            next_geometry = kabsch_align(geometries[0], geometries[index], masses_array)
        else:
            next_geometry = geometries[next_index]
        displacement_bohr = (next_geometry - geometries[index]) * ANGSTROM_TO_BOHR
        lengths.append(float(np.sqrt(np.sum(masses_array[:, None] * displacement_bohr**2))))
    return np.asarray(lengths, dtype=float)


def interpolate_periodic_path_potential(
    path_coordinates: np.ndarray,
    potential_hartree: np.ndarray,
    total_path_length: float,
    *,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    x_periodic = np.r_[path_coordinates, total_path_length]
    y_periodic = np.r_[potential_hartree, potential_hartree[0]]
    spline = CubicSpline(x_periodic, y_periodic, bc_type="periodic")
    solver_grid = np.linspace(0.0, total_path_length, grid_points, endpoint=False)
    solver_potential = np.asarray(spline(solver_grid), dtype=float)
    solver_potential -= float(np.min(solver_potential))
    return solver_grid, solver_potential


def solve_periodic_path_schrodinger(
    potential_hartree: np.ndarray,
    total_path_length: float,
    *,
    num_states: int,
    eigensolver_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    potential = np.asarray(potential_hartree, dtype=float)
    npoints = len(potential)
    if npoints < 3:
        raise ValueError("potential_hartree must contain at least three points")
    if not 1 <= int(num_states) <= npoints:
        raise ValueError("num_states must be positive and no larger than the grid")
    k = int(num_states)
    dx = float(total_path_length) / npoints
    laplacian = periodic_second_derivative_matrix(npoints)
    hamiltonian = diags(potential, 0, format="csr") + (-0.5) * (laplacian / dx**2)
    if npoints <= 400 or k >= npoints - 1:
        eigenvalues, eigenvectors_columns = np.linalg.eigh(hamiltonian.toarray())
        eigenvalues = eigenvalues[:k]
        eigenvectors_columns = eigenvectors_columns[:, :k]
    else:
        initial = (
            None
            if eigensolver_seed is None
            else np.random.default_rng(eigensolver_seed).normal(size=npoints)
        )
        eigenvalues, eigenvectors_columns = eigsh(hamiltonian, k=k, which="SA", v0=initial)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors_columns = eigenvectors_columns[:, order]
    wavefunctions = eigenvectors_columns.T.copy()
    for index in range(len(wavefunctions)):
        norm = np.sqrt(np.sum(wavefunctions[index] ** 2) * dx)
        if norm == 0.0:
            raise ValueError("Encountered a zero-norm relaxed torsion wavefunction")
        wavefunctions[index] /= norm
    return eigenvalues, wavefunctions


def identify_relaxed_torsion_basins(
    path_grid: np.ndarray,
    potential_hartree: np.ndarray,
    total_path_length: float,
    raw_path_coordinates: np.ndarray,
    raw_angles_rad: np.ndarray,
) -> list[RelaxedTorsionBasin]:
    potential = np.asarray(potential_hartree, dtype=float)
    minima = [
        index
        for index, value in enumerate(potential)
        if value <= potential[(index - 1) % len(potential)]
        and value <= potential[(index + 1) % len(potential)]
    ]
    if not minima:
        minima = [int(np.argmin(potential))]
    maxima = [
        index
        for index, value in enumerate(potential)
        if value >= potential[(index - 1) % len(potential)]
        and value >= potential[(index + 1) % len(potential)]
    ]
    if len(minima) == 1 or not maxima:
        minimum = minima[0]
        return [
            RelaxedTorsionBasin(
                index=0,
                minimum_path_coordinate=float(path_grid[minimum]),
                minimum_angle_deg=float(
                    np.degrees(
                        interpolate_angle_from_path(
                            path_grid[minimum],
                            raw_path_coordinates,
                            raw_angles_rad,
                            total_path_length,
                        )
                    )
                ),
                minimum_energy_hartree=float(potential[minimum]),
                left_boundary_path_coordinate=0.0,
                right_boundary_path_coordinate=float(total_path_length),
            )
        ]

    minima = sorted(minima)
    maxima = sorted(maxima)
    basins = []
    for basin_index, minimum in enumerate(minima):
        left_candidates = [maximum for maximum in maxima if maximum < minimum]
        right_candidates = [maximum for maximum in maxima if maximum > minimum]
        left = left_candidates[-1] if left_candidates else maxima[-1]
        right = right_candidates[0] if right_candidates else maxima[0]
        angle = interpolate_angle_from_path(
            float(path_grid[minimum]),
            raw_path_coordinates,
            raw_angles_rad,
            total_path_length,
        )
        basins.append(
            RelaxedTorsionBasin(
                index=basin_index,
                minimum_path_coordinate=float(path_grid[minimum]),
                minimum_angle_deg=float(np.degrees(angle)),
                minimum_energy_hartree=float(potential[minimum]),
                left_boundary_path_coordinate=float(path_grid[left]),
                right_boundary_path_coordinate=float(path_grid[right]),
            )
        )
    return basins


def project_path_tangent_on_normal_modes(
    aligned_geometries_angstrom: np.ndarray,
    segment_lengths: np.ndarray,
    potential_hartree: np.ndarray,
    masses: Sequence[float],
    normal_mode_eigenvectors: np.ndarray,
    normal_mode_frequencies_cm1: Sequence[float],
    *,
    frequency_cutoff_cm1: float,
) -> tuple[int | None, float]:
    minimum = int(np.argmin(potential_hartree))
    prev_index = (minimum - 1) % len(aligned_geometries_angstrom)
    next_index = (minimum + 1) % len(aligned_geometries_angstrom)
    displacement_bohr = (
        aligned_geometries_angstrom[next_index] - aligned_geometries_angstrom[prev_index]
    ) * ANGSTROM_TO_BOHR
    tangent = displacement_bohr * np.sqrt(np.asarray(masses, dtype=float)[:, None])
    tangent = tangent.reshape(-1)
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0 or normal_mode_eigenvectors.size == 0:
        return None, 0.0
    tangent /= norm
    scores = []
    for mode, frequency in zip(normal_mode_eigenvectors, normal_mode_frequencies_cm1):
        if float(frequency) > frequency_cutoff_cm1:
            scores.append(0.0)
            continue
        mode_array = np.asarray(mode, dtype=float)
        mode_norm = float(np.linalg.norm(mode_array))
        if mode_norm == 0.0:
            scores.append(0.0)
        else:
            scores.append(abs(float(np.dot(tangent, mode_array / mode_norm))))
    if not scores:
        return None, 0.0
    best = int(np.argmax(scores))
    return best, float(scores[best])


def project_tangent_at_path_on_normal_modes(
    solution: RelaxedTorsionSolution,
    path_coordinate: float,
    normal_mode_eigenvectors: np.ndarray,
    normal_mode_frequencies_cm1: Sequence[float],
    *,
    frequency_cutoff_cm1: float,
) -> tuple[int | None, float]:
    derivative = geometry_derivative_from_path(solution, path_coordinate)
    tangent = derivative * np.sqrt(solution.masses_electron[:, None])
    tangent = tangent.reshape(-1)
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return None, 0.0
    tangent /= norm
    scores = []
    for mode, frequency in zip(normal_mode_eigenvectors, normal_mode_frequencies_cm1):
        if float(frequency) > frequency_cutoff_cm1:
            scores.append(0.0)
            continue
        mode_array = np.asarray(mode, dtype=float)
        mode_norm = float(np.linalg.norm(mode_array))
        scores.append(
            0.0 if mode_norm == 0.0 else abs(float(np.dot(tangent, mode_array / mode_norm)))
        )
    if not scores:
        return None, 0.0
    best = int(np.argmax(scores))
    return best, float(scores[best])


def sample_relaxed_torsion(
    solution: RelaxedTorsionSolution,
    temperature_kelvin: float,
    rng: np.random.Generator,
    *,
    state_energy_offsets_hartree: Sequence[float] | None = None,
    state: int | None = None,
    momentum_distributions: Sequence[PeriodicMomentumDistribution] | None = None,
) -> RelaxedTorsionSample:
    state_energies = np.asarray(solution.eigenvalues_hartree, dtype=float)
    if state_energy_offsets_hartree is not None:
        offsets = np.asarray(state_energy_offsets_hartree, dtype=float)
        if offsets.shape != state_energies.shape:
            raise ValueError(
                "state_energy_offsets_hartree must have one value per relaxed torsion state"
            )
        state_energies = state_energies + offsets
    if state is None:
        probabilities = boltzmann_state_probabilities(state_energies, temperature_kelvin)
        state = int(rng.choice(np.arange(len(probabilities)), p=probabilities))
    else:
        state = int(state)
        if state < 0 or state >= len(state_energies):
            raise ValueError(f"Relaxed torsion state {state} is out of range")
    density = solution.wavefunctions[state] ** 2
    grid_probabilities = density / np.sum(density)
    grid_index = int(rng.choice(np.arange(len(grid_probabilities)), p=grid_probabilities))
    path_coordinate = float(solution.solver_path_grid[grid_index])
    angle = interpolate_angle_from_path(
        path_coordinate,
        solution.path_coordinates,
        solution.angles_rad,
        solution.total_path_length,
    )
    geometry = interpolate_geometry_from_path(solution, path_coordinate)
    potential = float(solution.potential_hartree[grid_index])
    potential_absolute = potential + float(solution.potential_reference_hartree)
    if momentum_distributions is None:
        distributions = periodic_path_momentum_distributions(
            solution.wavefunctions,
            solution.total_path_length,
        )
    else:
        distributions = list(momentum_distributions)
    if len(distributions) != len(solution.eigenvalues_hartree):
        raise ValueError("momentum_distributions must have one value per relaxed torsion state")
    expected_kinetic = np.asarray(
        [distribution.finite_difference_kinetic_energy_hartree for distribution in distributions],
        dtype=float,
    )
    qdot = distributions[state].sample(rng)
    derivative = geometry_derivative_from_path(solution, path_coordinate)
    velocity = derivative * qdot
    return RelaxedTorsionSample(
        state=state,
        path_coordinate=path_coordinate,
        angle_rad=float(angle),
        basin_index=basin_index_for_path_coordinate(
            solution.basins, path_coordinate, solution.total_path_length
        ),
        potential_hartree=potential_absolute,
        potential_relative_hartree=potential,
        state_energy_hartree=float(solution.eigenvalues_hartree[state]),
        expected_kinetic_energy_hartree=float(expected_kinetic[state]),
        path_momentum_direction_au=float(qdot),
        geometry_angstrom=geometry,
        velocity_au=velocity,
    )


def relaxed_torsion_expected_kinetic_energies_hartree(
    solution: RelaxedTorsionSolution,
) -> np.ndarray:
    """Return ``<T> = epsilon - <V>`` for every periodic rotor state."""
    density = np.asarray(solution.wavefunctions, dtype=float) ** 2
    normalization = np.sum(density, axis=1)
    if np.any(normalization <= 0.0):
        raise ValueError("Relaxed torsion wavefunction has zero probability density")
    expected_potential = (
        np.sum(
            density * np.asarray(solution.potential_hartree, dtype=float)[None, :],
            axis=1,
        )
        / normalization
    )
    kinetic = np.asarray(solution.eigenvalues_hartree, dtype=float) - expected_potential
    kinetic[np.logical_and(kinetic < 0.0, np.abs(kinetic) < 1.0e-10)] = 0.0
    if np.any(kinetic < 0.0):
        raise ValueError("Relaxed torsion state has a negative expected kinetic energy")
    return kinetic


def boltzmann_state_probabilities(
    eigenvalues_hartree: Sequence[float], temperature_kelvin: float
) -> np.ndarray:
    if temperature_kelvin < 0.0:
        raise ValueError("Sampling temperature cannot be negative")
    eigenvalues = np.asarray(eigenvalues_hartree, dtype=float)
    shifted = eigenvalues - np.min(eigenvalues)
    if temperature_kelvin == 0.0:
        ground = np.isclose(shifted, 0.0, rtol=0.0, atol=1.0e-12)
        return ground.astype(float) / np.count_nonzero(ground)
    energies_ev = shifted * HARTREE_TO_EV
    weights = np.exp(-energies_ev / (BOLTZMANN_EV_PER_K * temperature_kelvin))
    return weights / np.sum(weights)


def state_basin_probability_matrix(solution: RelaxedTorsionSolution) -> np.ndarray:
    """Return P(basin | torsion state) from the normalized rotor densities."""
    basin_indices = [basin.index for basin in solution.basins]
    if basin_indices != list(range(len(basin_indices))):
        raise ValueError("Relaxed torsion basin indices must be contiguous from zero")
    dx = float(solution.total_path_length) / len(solution.solver_path_grid)
    probabilities = np.zeros((len(solution.eigenvalues_hartree), len(solution.basins)), dtype=float)
    grid_basins = [
        basin_index_for_path_coordinate(solution.basins, coordinate, solution.total_path_length)
        for coordinate in solution.solver_path_grid
    ]
    for state_index, wavefunction in enumerate(solution.wavefunctions):
        density = np.asarray(wavefunction, dtype=float) ** 2
        for grid_index, basin_index in enumerate(grid_basins):
            probabilities[state_index, basin_index] += density[grid_index] * dx
        total = float(np.sum(probabilities[state_index]))
        if total > 0.0:
            probabilities[state_index] /= total
    return probabilities


def state_energy_offsets_from_basin_offsets(
    solution: RelaxedTorsionSolution,
    basin_offsets_hartree: Sequence[float],
) -> np.ndarray:
    """Average basin-local energy offsets over each torsion state's density."""
    offsets = np.asarray(basin_offsets_hartree, dtype=float)
    if offsets.shape != (len(solution.basins),):
        raise ValueError("basin_offsets_hartree must have one value per relaxed torsion basin")
    return state_basin_probability_matrix(solution) @ offsets


def interpolate_angle_from_path(
    path_coordinate: float,
    path_coordinates: np.ndarray,
    angles_rad: np.ndarray,
    total_path_length: float,
) -> float:
    wrapped = path_coordinate % total_path_length
    x = np.r_[path_coordinates, total_path_length]
    y = np.r_[angles_rad, 2.0 * np.pi]
    return float(np.interp(wrapped, x, y) % (2.0 * np.pi))


def interpolate_geometry_from_path(
    solution: RelaxedTorsionSolution, path_coordinate: float
) -> np.ndarray:
    index, next_index, fraction = path_segment_fraction(solution, path_coordinate)
    next_geometry = solution.geometries_angstrom[next_index]
    if next_index == 0:
        next_geometry = kabsch_align(
            next_geometry,
            solution.geometries_angstrom[index],
            solution.masses_electron,
        )
    return (1.0 - fraction) * solution.geometries_angstrom[index] + fraction * next_geometry


def geometry_derivative_from_path(
    solution: RelaxedTorsionSolution, path_coordinate: float
) -> np.ndarray:
    index, next_index, _fraction = path_segment_fraction(solution, path_coordinate)
    length = float(solution.segment_lengths[index])
    if length <= 0.0:
        return np.zeros_like(solution.geometries_angstrom[index])
    next_geometry = solution.geometries_angstrom[next_index]
    if next_index == 0:
        next_geometry = kabsch_align(
            next_geometry,
            solution.geometries_angstrom[index],
            solution.masses_electron,
        )
    displacement_bohr = (next_geometry - solution.geometries_angstrom[index]) * ANGSTROM_TO_BOHR
    return displacement_bohr / length


def path_segment_fraction(
    solution: RelaxedTorsionSolution,
    path_coordinate: float,
) -> tuple[int, int, float]:
    wrapped = path_coordinate % solution.total_path_length
    index = int(np.searchsorted(solution.path_coordinates, wrapped, side="right") - 1)
    index = max(0, min(index, len(solution.path_coordinates) - 1))
    next_index = (index + 1) % len(solution.path_coordinates)
    segment_start = float(solution.path_coordinates[index])
    segment_length = float(solution.segment_lengths[index])
    if segment_length <= 0.0:
        return index, next_index, 0.0
    if next_index == 0 and wrapped < segment_start:
        wrapped += solution.total_path_length
    return index, next_index, float((wrapped - segment_start) / segment_length)


def basin_index_for_path_coordinate(
    basins: Sequence[RelaxedTorsionBasin],
    path_coordinate: float,
    total_path_length: float,
) -> int:
    wrapped = path_coordinate % total_path_length
    for basin in basins:
        left = basin.left_boundary_path_coordinate % total_path_length
        right = basin.right_boundary_path_coordinate % total_path_length
        if left <= right:
            inside = left <= wrapped < right
        else:
            inside = wrapped >= left or wrapped < right
        if inside:
            return int(basin.index)
    distances = [
        min(
            abs(wrapped - basin.minimum_path_coordinate),
            total_path_length - abs(wrapped - basin.minimum_path_coordinate),
        )
        for basin in basins
    ]
    return int(basins[int(np.argmin(distances))].index)


def closest_basin_index_for_path_coordinate(
    basins: Sequence[RelaxedTorsionBasin],
    path_coordinate: float,
    total_path_length: float,
) -> int:
    wrapped = path_coordinate % total_path_length
    distances = [
        min(
            abs(wrapped - basin.minimum_path_coordinate),
            total_path_length - abs(wrapped - basin.minimum_path_coordinate),
        )
        for basin in basins
    ]
    return int(basins[int(np.argmin(distances))].index)


def write_relaxed_torsion_outputs(
    output_dir: str | Path,
    solution: RelaxedTorsionSolution,
    *,
    write_plots: bool,
    max_wavefunction_states: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "relaxed_torsion.json", solution, indent=2)
    if write_plots:
        write_relaxed_torsion_plots(
            output_path / "relaxed_torsion_plots", solution, max_wavefunction_states
        )


def write_relaxed_torsion_plots(
    plot_dir: str | Path,
    solution: RelaxedTorsionSolution,
    max_wavefunction_states: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    plot_files = []
    angle_grid = np.degrees(
        [
            interpolate_angle_from_path(
                coordinate,
                solution.path_coordinates,
                solution.angles_rad,
                solution.total_path_length,
            )
            for coordinate in solution.solver_path_grid
        ]
    )
    raw_angles = np.degrees(solution.angles_rad)
    raw_energies = np.asarray([point.energy_hartree for point in solution.scan_points], dtype=float)
    raw_energies = (raw_energies - np.min(raw_energies)) * HARTREE_TO_KCAL_MOL

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(angle_grid, solution.potential_hartree * HARTREE_TO_KCAL_MOL, linewidth=2.0)
    ax.plot(raw_angles, raw_energies, "o", markersize=4.0)
    for basin in solution.basins:
        ax.axvline(basin.minimum_angle_deg, color="0.3", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel("torsion angle / degrees")
    ax.set_ylabel("relative energy / kcal mol^-1")
    ax.set_title("Relaxed torsion potential")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    potential_path = path / "relaxed_torsion_potential.png"
    fig.savefig(potential_path, dpi=180)
    plt.close(fig)
    plot_files.append(potential_path)

    state_count = min(max_wavefunction_states, solution.wavefunctions.shape[0])
    vibrational_energies_cm1 = (
        solution.eigenvalues_hartree - float(np.min(solution.potential_hartree))
    ) * HARTREE_TO_CM1
    psi = solution.wavefunctions[:state_count]
    max_abs = max(float(np.max(np.abs(psi))), 1.0)
    offset = 1.35 * max_abs
    fig, (ax_wave, ax_density) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for state in range(state_count):
        label = state_energy_label(state, vibrational_energies_cm1[state])
        ax_wave.plot(angle_grid, psi[state] + state * offset, linewidth=1.4, label=label)
        ax_density.plot(angle_grid, psi[state] ** 2, linewidth=1.4, label=f"n={state}")
    ax_wave.set_ylabel("psi + offset")
    ax_wave.set_title("Relaxed torsion wavefunctions")
    ax_wave.grid(alpha=0.25)
    ax_wave.legend(frameon=False, fontsize=8, ncol=2)
    ax_density.set_xlabel("torsion angle / degrees")
    ax_density.set_ylabel("|psi|^2")
    ax_density.grid(alpha=0.25)
    fig.tight_layout()
    wavefunction_path = path / "relaxed_torsion_wavefunctions.png"
    fig.savefig(wavefunction_path, dpi=180)
    plt.close(fig)
    plot_files.append(wavefunction_path)

    momentum_state_count = min(state_count, 5)
    momentum_grid, momentum_wavefunctions = periodic_path_momentum_wavefunctions(
        solution.wavefunctions[:momentum_state_count],
        solution.total_path_length,
    )
    momentum_probabilities = np.abs(momentum_wavefunctions) ** 2
    significant = np.any(momentum_probabilities > 1.0e-9, axis=0)
    if np.any(significant):
        momentum_limit = max(
            float(np.max(np.abs(momentum_grid[significant]))) * 1.1,
            5.0 * 2.0 * np.pi / solution.total_path_length,
        )
    else:
        momentum_limit = 5.0 * 2.0 * np.pi / solution.total_path_length
    visible = np.abs(momentum_grid) <= momentum_limit
    fig, axes = plt.subplots(
        momentum_state_count,
        2,
        figsize=(10.0, max(4.0, 1.9 * momentum_state_count)),
        sharex=True,
        squeeze=False,
    )
    for state in range(momentum_state_count):
        amplitude_axis, probability_axis = axes[state]
        amplitude_axis.stem(
            momentum_grid[visible],
            np.real(momentum_wavefunctions[state, visible]),
            linefmt="C0-",
            markerfmt="C0o",
            basefmt=" ",
            label="Re",
        )
        amplitude_axis.stem(
            momentum_grid[visible],
            np.imag(momentum_wavefunctions[state, visible]),
            linefmt="C1-",
            markerfmt="C1s",
            basefmt=" ",
            label="Im",
        )
        probability_axis.stem(
            momentum_grid[visible],
            momentum_probabilities[state, visible],
            basefmt=" ",
        )
        amplitude_axis.set_ylabel(f"$c_{{{state},k}}$")
        probability_axis.set_ylabel(f"$|c_{{{state},k}}|^2$")
        amplitude_axis.grid(alpha=0.25)
        probability_axis.grid(alpha=0.25)
        if state == 0:
            amplitude_axis.set_title("complex Fourier coefficients")
            probability_axis.set_title("discrete momentum probabilities")
            amplitude_axis.legend(frameon=False)
    axes[-1, 0].set_xlabel("mass-weighted path momentum / atomic units")
    axes[-1, 1].set_xlabel("mass-weighted path momentum / atomic units")
    fig.suptitle("Exact periodic relaxed-torsion momentum wavefunctions")
    fig.tight_layout()
    momentum_path = path / "relaxed_torsion_momentum_wavefunctions.png"
    fig.savefig(momentum_path, dpi=180)
    plt.close(fig)
    plot_files.append(momentum_path)

    return plot_files


def state_energy_label(state: int, vibrational_energy_cm1: float) -> str:
    return f"n={state}, Ev={vibrational_energy_cm1:.1f} cm^-1"
