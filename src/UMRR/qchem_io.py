"""Q-Chem parsing, input generation, and coordinate formatting."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import (
    ANGSTROM_TO_BOHR,
    ATOMIC_CHARGES,
    ATOMIC_MASSES_ELECTRON,
    ELECTRON_MASS_TO_AMU,
    HESSIAN_AU_TO_WAVENUMBER,
)


def parse_standard_orientation_angstrom(
    qchem_output: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Return the last standard nuclear orientation in Angstrom."""
    path = Path(qchem_output)
    geometries: list[list[list[float]]] = []
    atom_blocks: list[list[str]] = []
    current_geom: list[list[float]] = []
    current_atoms: list[str] = []
    in_section = False
    skip_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if "Standard Nuclear Orientation (Angstroms)" in line:
                in_section = True
                skip_lines = 2
                current_geom = []
                current_atoms = []
                continue
            if not in_section:
                continue
            if skip_lines:
                skip_lines -= 1
                continue
            if "---" in line:
                if current_geom:
                    geometries.append(current_geom)
                    atom_blocks.append(current_atoms)
                in_section = False
                continue
            parts = line.split()
            if len(parts) >= 5:
                current_atoms.append(parts[1])
                current_geom.append([float(parts[2]), float(parts[3]), float(parts[4])])

    if not geometries:
        raise ValueError(f"Could not extract standard orientation from {path}")
    return np.asarray(geometries[-1], dtype=float), atom_blocks[-1]


def parse_projected_mass_weighted_hessian(qchem_output: str | Path, num_atoms: int) -> np.ndarray:
    """Parse the projected mass-weighted Hessian matrix from a frequency output."""
    dimension = 3 * num_atoms
    rows: list[list[float]] = [[] for _ in range(dimension)]
    row_index = 0
    in_section = False

    with Path(qchem_output).open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Projected Mass-Weighted Hessian Matrix"):
                in_section = True
                continue
            if line.startswith("Vibrational Frequencies in atomic units"):
                break
            if not in_section:
                continue
            values = [float(part) for part in line.split()]
            rows[row_index].extend(values)
            row_index = 0 if row_index == dimension - 1 else row_index + 1

    hessian = np.asarray(rows, dtype=float)
    if hessian.shape != (dimension, dimension):
        raise ValueError(
            f"Expected a {dimension}x{dimension} Hessian in {qchem_output}, "
            f"got shape {hessian.shape}"
        )
    return hessian


def parse_total_energies_hartree(qchem_output: str | Path) -> list[float]:
    """Parse ground-state and optional EOM-IP total energies from a Q-Chem output."""
    lines = Path(qchem_output).read_text(encoding="utf-8", errors="replace").splitlines()
    energies: list[float] = []

    # MP2 prints an SCF total first, followed by the correlated total with
    # different spacing and a trailing unit. Frequencies may then print totals
    # for numerical displacements; the first MP2 total is the reference energy.
    mp2 = re.compile(r"(?:RI-)?MP2\s+total energy\s*=\s*([-+\d.EeDd]+)", re.I)
    match = next((m for line in lines if (m := mp2.search(line))), None)
    if match:
        energies.append(float(match.group(1).replace("D", "E").replace("d", "e")))
    else:
        if any(
            re.search(r"^\s*(?:method|correlation)\s+(?:ri-?)?mp2\s*(?:!.*)?$", line, re.I)
            for line in lines
        ):
            raise ValueError(
                f"MP2 total energy not found in {qchem_output}; refusing to use an SCF energy"
            )
        for line in lines:
            if "Total energy = " in line:
                energies.append(float(line.split()[-1]))
                break
    if not energies:
        raise ValueError(f"Ground-state total energy not found in {qchem_output}")

    in_eomip_section = False
    for line in lines:
        if "EOMIP transition" in line:
            in_eomip_section = True
            continue
        if in_eomip_section and "Running a double precision version" in line:
            in_eomip_section = False
        if in_eomip_section and "Total energy = " in line:
            parts = line.split()
            try:
                energies.append(float(parts[3]))
            except (IndexError, ValueError):
                energies.append(float(parts[-1]))

    return energies


def mass_matrix_electron_masses(atom_symbols: Sequence[str]) -> np.ndarray:
    """Return an ``(natoms, 3)`` Cartesian mass matrix in electron masses."""
    try:
        return np.asarray(
            [[ATOMIC_MASSES_ELECTRON[atom]] * 3 for atom in atom_symbols], dtype=float
        )
    except KeyError as exc:
        raise KeyError(f"No mass is available for element {exc.args[0]!r}") from exc


def atomic_charges(atom_symbols: Sequence[str]) -> list[float]:
    try:
        return [ATOMIC_CHARGES[atom] for atom in atom_symbols]
    except KeyError as exc:
        raise KeyError(f"No atomic charge is available for element {exc.args[0]!r}") from exc


def internal_coordinate_basis(
    geometry_angstrom: np.ndarray,
    masses_electron: Sequence[float],
    *,
    excluded_directions_mass_weighted: np.ndarray | None = None,
) -> np.ndarray:
    """Orthonormal Cartesian mass-weighted basis excluding rigid motion.

    Translations are sqrt(M) times a uniform displacement; rotations are
    sqrt(M) times axis x (R - COM). Their geometric rank, not their Hessian
    frequencies, determines how many directions are removed. Optional rows
    specify additional mass-weighted directions (e.g. a rotor tangent).
    """
    geometry = np.asarray(geometry_angstrom, dtype=float)
    masses = np.asarray(masses_electron, dtype=float)
    if geometry.ndim != 2 or geometry.shape[1] != 3 or masses.shape != (len(geometry),):
        raise ValueError("Expected geometry (natoms, 3) and one mass per atom")
    if (
        not len(masses)
        or not np.all(np.isfinite(geometry))
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0)
    ):
        raise ValueError("Geometry and positive atomic masses must be finite")
    root_mass = np.sqrt(masses)[:, None]
    centered = geometry - np.average(geometry, axis=0, weights=masses)
    translations = np.column_stack(
        [(root_mass * axis).reshape(-1) for axis in np.eye(3)]
    ) / np.sqrt(np.sum(masses))
    rotations = np.column_stack(
        [(root_mass * np.cross(axis, centered)).reshape(-1) for axis in np.eye(3)]
    )
    rotations -= translations @ (translations.T @ rotations)
    left, singular, _right = np.linalg.svd(rotations, full_matrices=False)
    rotation_rank = (
        int(np.count_nonzero(singular > 1.0e-10 * singular[0])) if singular[0] > 0 else 0
    )
    external = np.column_stack((translations, left[:, :rotation_rank]))
    left, _singular, _right = np.linalg.svd(external, full_matrices=True)
    basis = left[:, external.shape[1] :]

    if excluded_directions_mass_weighted is not None:
        directions = np.atleast_2d(np.asarray(excluded_directions_mass_weighted, dtype=float))
        if directions.shape[1] != geometry.size or not np.all(np.isfinite(directions)):
            raise ValueError("Excluded directions must have shape (n_directions, 3*natoms)")
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Cannot project a zero internal direction")
        internal = basis.T @ (directions / lengths[:, None]).T
        left, singular, _right = np.linalg.svd(internal, full_matrices=True)
        rank = int(np.count_nonzero(singular > 1.0e-10))
        if rank != len(directions):
            raise ValueError("Excluded directions must be independent and contain internal motion")
        basis = basis @ left[:, rank:]
    return basis


def vibrational_modes_from_hessian(
    hessian_mass_weighted: np.ndarray,
    geometry_angstrom: np.ndarray,
    masses_electron: Sequence[float],
    *,
    excluded_directions_mass_weighted: np.ndarray | None = None,
    translational_rotational_cutoff: float = 1.0e-4,
    allow_imaginary_modes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize the Hessian only in the allowed internal subspace.

    Geometry and masses must be in the Hessian's atom order and Cartesian
    frame, even if the input Hessian was already mass weighted/projected.
    Every real internal mode is retained, including modes below 20 cm^-1.
    ``translational_rotational_cutoff`` retains its previous role as the
    negative-eigenvalue tolerance; it no longer selects external modes.
    """
    basis = internal_coordinate_basis(
        geometry_angstrom,
        masses_electron,
        excluded_directions_mass_weighted=excluded_directions_mass_weighted,
    )
    hessian = np.asarray(hessian_mass_weighted, dtype=float)
    if hessian.shape != (basis.shape[0], basis.shape[0]) or not np.all(np.isfinite(hessian)):
        raise ValueError("Mass-weighted Hessian has the wrong shape or nonfinite entries")
    projected = basis.T @ ((hessian + hessian.T) * 0.5) @ basis
    eigenvalues, eigenvectors_columns = np.linalg.eigh(projected)
    eigenvectors = (basis @ eigenvectors_columns).T
    # Fix the arbitrary eigenvector sign for repeatable input preparation.
    if len(eigenvectors):
        largest = np.argmax(np.abs(eigenvectors), axis=1)
        eigenvectors *= np.where(
            eigenvectors[np.arange(len(eigenvectors)), largest] < 0, -1.0, 1.0
        )[:, None]

    negative = eigenvalues < -translational_rotational_cutoff
    if np.any(negative):
        count = int(np.count_nonzero(negative))
        if not allow_imaginary_modes:
            raise ValueError(
                f"Frequency Hessian contains {count} significant negative eigenvalue(s). "
                "Initial-condition sampling requires a local minimum; optimize the "
                "geometry and recompute frequencies. Set allow_imaginary_modes=True "
                "only for an explicitly unsupported non-minimum analysis."
            )
        warnings.warn(
            f"Removing {count} negative Hessian eigenvalue(s) below "
            f"{-translational_rotational_cutoff:g}.",
            RuntimeWarning,
            stacklevel=2,
        )
        keep = ~negative
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[keep]

    frequencies_cm1 = np.sqrt(np.clip(eigenvalues, 0.0, None)) * HESSIAN_AU_TO_WAVENUMBER
    return eigenvalues, eigenvectors, frequencies_cm1


def reduced_masses_electron(
    mass_matrix: np.ndarray,
    normal_mode_eigenvectors: np.ndarray,
    num_atoms: int,
) -> np.ndarray:
    """Return reduced masses in electron masses for each normal mode."""
    masses = np.asarray(mass_matrix, dtype=float)
    reduced_masses: list[float] = []
    for eigenvector in normal_mode_eigenvectors:
        split = np.asarray(np.split(eigenvector, num_atoms), dtype=float)
        reduced_masses.append(float(1.0 / np.sum(split * split / masses)))
    return np.asarray(reduced_masses, dtype=float)


def force_constants_au(
    reduced_masses_electron: Sequence[float], frequencies_cm1: Sequence[float]
) -> np.ndarray:
    """Return harmonic force constants in Hartree/Bohr^2."""
    speed_of_light_m_s = 299792458.0
    reduced_mass_si = (np.asarray(reduced_masses_electron, dtype=float) / 1822.8857) / 6.02214076e26
    frequency_si = np.asarray(frequencies_cm1, dtype=float) * 100.0 * speed_of_light_m_s
    force_si = (frequency_si * 2.0 * np.pi) ** 2 * reduced_mass_si
    return force_si * 1.2137802661e7 / 1.8897261246e10


def geometry_to_qchem_string_angstrom(
    geometry_angstrom: np.ndarray, atom_symbols: Sequence[str]
) -> str:
    lines = []
    for atom, xyz in zip(atom_symbols, geometry_angstrom):
        lines.append(f"{atom} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    return "\n".join(lines)


def geometry_to_sharc_string_bohr(geometry_bohr: np.ndarray, atom_symbols: Sequence[str]) -> str:
    masses_amu = mass_matrix_electron_masses(atom_symbols) * ELECTRON_MASS_TO_AMU
    charges = atomic_charges(atom_symbols)
    lines = []
    for atom, charge, xyz, mass in zip(atom_symbols, charges, geometry_bohr, masses_amu[:, 0]):
        lines.append(f"{atom} {charge:.1f} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} {mass}")
    return "\n".join(lines)


def velocities_to_sharc_string_atomic_units(velocities_au: np.ndarray) -> str:
    return "\n".join(
        f"{xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}" for xyz in np.asarray(velocities_au)
    )


def generate_qchem_input_text(
    geometry_angstrom: np.ndarray,
    atom_symbols: Sequence[str],
    *,
    charge: int,
    multiplicity: int,
    template: str,
) -> str:
    return template.format(
        charge=charge,
        multiplicity=multiplicity,
        geometry_angstrom=geometry_to_qchem_string_angstrom(geometry_angstrom, atom_symbols),
    )


def qchem_template_with_rem_settings(template: str, settings: dict[str, object]) -> str:
    """Return a Q-Chem template with selected ``$rem`` settings inserted or replaced."""
    lines = template.splitlines()
    output_lines: list[str] = []
    in_rem = False
    seen = {key.lower(): False for key in settings}

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "$rem":
            in_rem = True
            output_lines.append(line)
            continue
        if in_rem and lower == "$end":
            for key, value in settings.items():
                if not seen[key.lower()]:
                    output_lines.append(f"{key} {value}")
            output_lines.append(line)
            in_rem = False
            continue
        if in_rem and stripped:
            parts = stripped.split(maxsplit=1)
            key = parts[0].lower()
            if key in seen:
                canonical_key = next(original for original in settings if original.lower() == key)
                output_lines.append(f"{canonical_key} {settings[canonical_key]}")
                seen[key] = True
                continue
        output_lines.append(line)

    if not any(line.strip().lower() == "$rem" for line in lines):
        output_lines.extend(["", "$rem"])
        for key, value in settings.items():
            output_lines.append(f"{key} {value}")
        output_lines.append("$end")
    return "\n".join(output_lines).rstrip() + "\n"


def qchem_template_with_jobtype(
    template: str,
    jobtype: str,
    *,
    extra_rem: dict[str, object] | None = None,
) -> str:
    settings: dict[str, object] = {"jobtype": jobtype}
    if extra_rem:
        settings.update(extra_rem)
    return qchem_template_with_rem_settings(template, settings)


def qchem_template_with_scan_block(template: str, scan_line: str) -> str:
    return template.rstrip() + "\n\n$scan\n" + scan_line.strip() + "\n$end\n"


def cartesian_geometry_angstrom_from_normal_displacement(
    equilibrium_geometry_angstrom: np.ndarray,
    mass_matrix: np.ndarray,
    normal_mode_eigenvectors: np.ndarray,
    mode_index: int,
    mass_weighted_displacement: float,
) -> np.ndarray:
    """Displace a geometry along one mass-weighted normal-mode coordinate."""
    eigenvector = normal_mode_eigenvectors[mode_index]
    split = np.asarray(np.split(eigenvector, len(equilibrium_geometry_angstrom)), dtype=float)
    displacement_bohr = split * mass_weighted_displacement / np.sqrt(mass_matrix)
    return (
        np.asarray(equilibrium_geometry_angstrom, dtype=float)
        + displacement_bohr / ANGSTROM_TO_BOHR
    )


def cartesian_displacement_bohr_from_normal_coordinates(
    atom_symbols: Sequence[str],
    normal_coordinates: Sequence[float],
    normal_mode_eigenvectors: np.ndarray,
    *,
    random_signs: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert mass-weighted normal-mode positions to Cartesian displacements in Bohr."""
    num_atoms = len(atom_symbols)
    masses = mass_matrix_electron_masses(atom_symbols)
    coordinates = np.asarray(normal_coordinates, dtype=float).copy()
    if random_signs is not None:
        coordinates *= np.asarray(random_signs, dtype=float)

    displacement = np.zeros((num_atoms, 3), dtype=float)
    for mode_value, eigenvector in zip(coordinates, normal_mode_eigenvectors):
        split = np.asarray(np.split(eigenvector, num_atoms), dtype=float)
        displacement += split * mode_value / np.sqrt(masses)
    return displacement


def cartesian_velocities_au_from_normal_velocities(
    atom_symbols: Sequence[str],
    normal_velocities: Sequence[float],
    normal_mode_eigenvectors: np.ndarray,
    *,
    random_signs: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert mass-weighted normal velocities to Cartesian velocities in atomic units."""
    num_atoms = len(atom_symbols)
    masses = mass_matrix_electron_masses(atom_symbols)
    velocities = np.asarray(normal_velocities, dtype=float).copy()
    if random_signs is not None:
        velocities *= np.asarray(random_signs, dtype=float)

    cartesian = np.zeros((num_atoms, 3), dtype=float)
    for mode_value, eigenvector in zip(velocities, normal_mode_eigenvectors):
        split = np.asarray(np.split(eigenvector, num_atoms), dtype=float)
        cartesian += split * mode_value / np.sqrt(masses)
    return cartesian, masses


def displacement_label(value: float) -> str:
    """Match the historical Q-Chem scan filename rounding."""
    return str(round(float(value), 3))
