"""Hindered-rotor detection utilities used by the relaxed torsion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import diags

from .constants import ANGSTROM_TO_BOHR, ATOMIC_MASSES_ELECTRON, COVALENT_RADII_ANGSTROM
from .qchem_io import mass_matrix_electron_masses


@dataclass(slots=True)
class RotorCandidate:
    index: int
    fixed_atom: int
    rotating_axis_atom: int
    rotating_atoms: list[int]
    label: str
    symmetry_number: int
    moment_of_inertia: float
    projected_mode_index: int | None
    projected_mode_frequency_cm1: float | None
    projection_score: float
    is_rotor: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "fixed_atom": self.fixed_atom,
            "rotating_axis_atom": self.rotating_axis_atom,
            "rotating_atoms": self.rotating_atoms,
            "label": self.label,
            "symmetry_number": self.symmetry_number,
            "moment_of_inertia_electron_bohr2": self.moment_of_inertia,
            "projected_mode_index": self.projected_mode_index,
            "projected_mode_frequency_cm1": self.projected_mode_frequency_cm1,
            "projection_score": self.projection_score,
            "is_rotor": self.is_rotor,
        }


@dataclass(slots=True)
class RotorAnalysis:
    bond_graph: dict[int, list[int]]
    candidates: list[RotorCandidate]
    rotors: list[RotorCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "bond_graph": {str(key): value for key, value in self.bond_graph.items()},
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rotors": [rotor.to_dict() for rotor in self.rotors],
        }


def analyze_hindered_rotors(
    geometry_angstrom: np.ndarray,
    atom_symbols: Sequence[str],
    normal_mode_eigenvectors: np.ndarray,
    frequencies_cm1: Sequence[float],
    *,
    frequency_cutoff_cm1: float = 500.0,
    projection_threshold: float = 0.35,
    bond_scale: float = 1.25,
    bond_tolerance_angstrom: float = 0.10,
) -> RotorAnalysis:
    """Detect rotatable bonds whose motion projects onto a low-frequency normal mode."""
    geometry = np.asarray(geometry_angstrom, dtype=float)
    graph = build_bond_graph(
        geometry,
        atom_symbols,
        bond_scale=bond_scale,
        bond_tolerance_angstrom=bond_tolerance_angstrom,
    )
    masses = mass_matrix_electron_masses(atom_symbols)
    candidates: list[RotorCandidate] = []
    for fixed_atom, rotating_axis_atom, rotating_atoms in rotatable_bond_sides(atom_symbols, graph):
        direction = rotor_mass_weighted_direction(
            geometry,
            atom_symbols,
            fixed_atom,
            rotating_axis_atom,
            rotating_atoms,
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        direction /= norm

        best_mode_index = None
        best_frequency = None
        best_score = 0.0
        for mode_index, frequency in enumerate(frequencies_cm1):
            if frequency > frequency_cutoff_cm1:
                continue
            mode = np.asarray(normal_mode_eigenvectors[mode_index], dtype=float)
            mode_norm = float(np.linalg.norm(mode))
            if mode_norm == 0.0:
                continue
            score = abs(float(np.dot(mode / mode_norm, direction)))
            if score > best_score:
                best_score = score
                best_mode_index = mode_index
                best_frequency = float(frequency)

        candidate_index = len(candidates)
        candidate = RotorCandidate(
            index=candidate_index,
            fixed_atom=fixed_atom,
            rotating_axis_atom=rotating_axis_atom,
            rotating_atoms=list(rotating_atoms),
            label=rotor_label(atom_symbols, fixed_atom, rotating_axis_atom, rotating_atoms),
            symmetry_number=rotor_symmetry_number(
                atom_symbols,
                graph,
                rotating_axis_atom,
                rotating_atoms,
            ),
            moment_of_inertia=rotor_moment_of_inertia(
                geometry,
                masses,
                fixed_atom,
                rotating_axis_atom,
                rotating_atoms,
            ),
            projected_mode_index=best_mode_index,
            projected_mode_frequency_cm1=best_frequency,
            projection_score=best_score,
            is_rotor=best_mode_index is not None and best_score >= projection_threshold,
        )
        candidates.append(candidate)

    return RotorAnalysis(
        bond_graph=graph,
        candidates=candidates,
        rotors=[candidate for candidate in candidates if candidate.is_rotor],
    )


def build_bond_graph(
    geometry_angstrom: np.ndarray,
    atom_symbols: Sequence[str],
    *,
    bond_scale: float = 1.25,
    bond_tolerance_angstrom: float = 0.10,
) -> dict[int, list[int]]:
    graph = {index: [] for index in range(len(atom_symbols))}
    for i in range(len(atom_symbols)):
        for j in range(i + 1, len(atom_symbols)):
            radius_i = COVALENT_RADII_ANGSTROM.get(atom_symbols[i], 0.77)
            radius_j = COVALENT_RADII_ANGSTROM.get(atom_symbols[j], 0.77)
            cutoff = bond_scale * (radius_i + radius_j) + bond_tolerance_angstrom
            distance = float(np.linalg.norm(geometry_angstrom[i] - geometry_angstrom[j]))
            if distance <= cutoff:
                graph[i].append(j)
                graph[j].append(i)
    return {key: sorted(value) for key, value in graph.items()}


def rotatable_bond_sides(
    atom_symbols: Sequence[str],
    graph: dict[int, list[int]],
) -> list[tuple[int, int, list[int]]]:
    sides: list[tuple[int, int, list[int]]] = []
    seen_bonds: set[tuple[int, int]] = set()
    for atom_i, neighbors in graph.items():
        for atom_j in neighbors:
            bond = tuple(sorted((atom_i, atom_j)))
            if bond in seen_bonds:
                continue
            seen_bonds.add(bond)
            if atom_symbols[atom_i] == "H" or atom_symbols[atom_j] == "H":
                continue
            side_i = component_after_removing_bond(graph, atom_i, atom_j)
            side_j = component_after_removing_bond(graph, atom_j, atom_i)
            if not side_i or not side_j:
                continue
            fixed_atom, rotating_axis_atom, rotating_atoms = choose_rotating_side(
                atom_symbols,
                graph,
                atom_i,
                atom_j,
                side_i,
                side_j,
            )
            if len(rotating_atoms) < 2:
                continue
            if not any(atom != rotating_axis_atom for atom in rotating_atoms):
                continue
            sides.append((fixed_atom, rotating_axis_atom, sorted(rotating_atoms)))
    return sides


def component_after_removing_bond(
    graph: dict[int, list[int]],
    start: int,
    blocked_neighbor: int,
) -> list[int]:
    stack = [start]
    visited: set[int] = set()
    while stack:
        atom = stack.pop()
        if atom in visited:
            continue
        visited.add(atom)
        for neighbor in graph[atom]:
            if atom == start and neighbor == blocked_neighbor:
                continue
            if neighbor not in visited:
                stack.append(neighbor)
    if blocked_neighbor in visited:
        return []
    return sorted(visited)


def choose_rotating_side(
    atom_symbols: Sequence[str],
    graph: dict[int, list[int]],
    atom_i: int,
    atom_j: int,
    side_i: list[int],
    side_j: list[int],
) -> tuple[int, int, list[int]]:
    side_i_score = rotor_side_score(atom_symbols, graph, atom_i, side_i)
    side_j_score = rotor_side_score(atom_symbols, graph, atom_j, side_j)
    if side_i_score > side_j_score:
        return atom_j, atom_i, side_i
    if side_j_score > side_i_score:
        return atom_i, atom_j, side_j
    if len(side_i) <= len(side_j):
        return atom_j, atom_i, side_i
    return atom_i, atom_j, side_j


def rotor_side_score(
    atom_symbols: Sequence[str],
    graph: dict[int, list[int]],
    axis_atom: int,
    side_atoms: Sequence[int],
) -> int:
    side_set = set(side_atoms)
    hydrogens = [
        neighbor
        for neighbor in graph[axis_atom]
        if neighbor in side_set and atom_symbols[neighbor] == "H"
    ]
    if atom_symbols[axis_atom] == "C" and len(hydrogens) == 3:
        return 4
    if atom_symbols[axis_atom] == "O" and len(hydrogens) == 1:
        return 3
    if len(side_atoms) <= 4:
        return 2
    return 1


def rotor_label(
    atom_symbols: Sequence[str],
    fixed_atom: int,
    rotating_axis_atom: int,
    rotating_atoms: Sequence[int],
) -> str:
    moving = ",".join(f"{atom_symbols[index]}{index + 1}" for index in rotating_atoms)
    return (
        f"{atom_symbols[fixed_atom]}{fixed_atom + 1}-"
        f"{atom_symbols[rotating_axis_atom]}{rotating_axis_atom + 1} rotor ({moving})"
    )


def rotor_symmetry_number(
    atom_symbols: Sequence[str],
    graph: dict[int, list[int]],
    rotating_axis_atom: int,
    rotating_atoms: Sequence[int],
) -> int:
    side = set(rotating_atoms)
    side_hydrogens = [
        neighbor
        for neighbor in graph[rotating_axis_atom]
        if neighbor in side and atom_symbols[neighbor] == "H"
    ]
    if atom_symbols[rotating_axis_atom] == "C" and len(side_hydrogens) == 3:
        return 3
    return 1


def torsion_atom_indices_for_rotor(
    rotor: RotorCandidate,
    atom_symbols: Sequence[str],
    geometry_angstrom: np.ndarray,
    graph: dict[int, list[int]],
) -> list[int]:
    """Choose a concrete 1-based Q-Chem ``tors a b c d`` definition for a rotor."""
    rotating_side = component_after_removing_bond(
        graph,
        rotor.rotating_axis_atom,
        rotor.fixed_atom,
    )
    fixed_side = component_after_removing_bond(
        graph,
        rotor.fixed_atom,
        rotor.rotating_axis_atom,
    )
    rotating_terminal = rotor_terminal_atom(
        atom_symbols,
        graph,
        axis_atom=rotor.rotating_axis_atom,
        blocked_atom=rotor.fixed_atom,
        side_atoms=rotating_side,
    )
    fixed_terminal = rotor_terminal_atom(
        atom_symbols,
        graph,
        axis_atom=rotor.fixed_atom,
        blocked_atom=rotor.rotating_axis_atom,
        side_atoms=fixed_side,
    )
    return [
        rotating_terminal + 1,
        rotor.rotating_axis_atom + 1,
        rotor.fixed_atom + 1,
        fixed_terminal + 1,
    ]


def rotor_terminal_atom(
    atom_symbols: Sequence[str],
    graph: dict[int, list[int]],
    *,
    axis_atom: int,
    blocked_atom: int,
    side_atoms: Sequence[int],
) -> int:
    side = set(side_atoms)
    candidates = [
        neighbor for neighbor in graph[axis_atom] if neighbor != blocked_atom and neighbor in side
    ]
    if not candidates:
        raise ValueError(
            f"Could not choose a terminal atom for the torsion around atom {axis_atom + 1}"
        )
    return max(
        candidates,
        key=lambda atom: (
            ATOMIC_MASSES_ELECTRON.get(atom_symbols[atom], 0.0),
            atom,
        ),
    )


def rotor_moment_of_inertia(
    geometry_angstrom: np.ndarray,
    mass_matrix: np.ndarray,
    fixed_atom: int,
    rotating_axis_atom: int,
    rotating_atoms: Sequence[int],
) -> float:
    geometry_bohr = np.asarray(geometry_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    axis_origin = geometry_bohr[fixed_atom]
    axis = geometry_bohr[rotating_axis_atom] - axis_origin
    axis /= np.linalg.norm(axis)
    inertia = 0.0
    for atom in rotating_atoms:
        relative = geometry_bohr[atom] - axis_origin
        perpendicular = relative - np.dot(relative, axis) * axis
        inertia += float(mass_matrix[atom, 0] * np.dot(perpendicular, perpendicular))
    return inertia


def rotor_mass_weighted_direction(
    geometry_angstrom: np.ndarray,
    atom_symbols: Sequence[str],
    fixed_atom: int,
    rotating_axis_atom: int,
    rotating_atoms: Sequence[int],
) -> np.ndarray:
    geometry_bohr = np.asarray(geometry_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    masses = mass_matrix_electron_masses(atom_symbols)
    axis_origin = geometry_bohr[fixed_atom]
    axis = geometry_bohr[rotating_axis_atom] - axis_origin
    axis /= np.linalg.norm(axis)
    direction = np.zeros_like(geometry_bohr)
    for atom in rotating_atoms:
        relative = geometry_bohr[atom] - axis_origin
        parallel = np.dot(relative, axis) * axis
        derivative = np.cross(axis, relative - parallel)
        direction[atom] = np.sqrt(masses[atom]) * derivative
    return direction.reshape(-1)


def periodic_second_derivative_matrix(npoints: int):
    matrix = diags(
        [np.ones(npoints - 1), -2.0 * np.ones(npoints), np.ones(npoints - 1)],
        offsets=[-1, 0, 1],
        shape=(npoints, npoints),
        format="lil",
    )
    matrix[0, npoints - 1] = 1.0
    matrix[npoints - 1, 0] = 1.0
    return matrix.tocsr()
