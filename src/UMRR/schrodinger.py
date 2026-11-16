"""Efficient finite-difference solver for 1D vibrational states."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal


def solve_schrodinger_1d(
    potential: Callable[[np.ndarray], np.ndarray],
    coordinate_grid_extent: np.ndarray,
    reduced_mass_electron: float,
    *,
    grid_points: int = 1000,
    num_states: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve -hbar^2/(2*mu) d^2/dq^2 + U(q) in atomic units.

    The three-point second derivative gives a tridiagonal Hamiltonian.
    Zero ghost values immediately outside the stored grid define its finite
    interval. Returned states have shape (n_states, grid_points) and satisfy
    sum(abs(psi)**2)*dq=1. For mass-weighted q, mu is dimensionless.
    """
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    if num_states < 1:
        raise ValueError("num_states must be at least 1")
    if num_states >= grid_points:
        raise ValueError("num_states must be smaller than grid_points")

    extent = np.asarray(coordinate_grid_extent, dtype=float)
    x_grid = np.linspace(float(np.min(extent)), float(np.max(extent)), grid_points)
    dx = float(x_grid[1] - x_grid[0])
    potential_values = np.asarray(potential(x_grid), dtype=float)
    if potential_values.shape != x_grid.shape:
        raise ValueError("potential callable must return one value per grid point")

    kinetic_diagonal = 1.0 / (reduced_mass_electron * dx**2)
    kinetic_off_diagonal = -0.5 / (reduced_mass_electron * dx**2)
    diagonal = potential_values + kinetic_diagonal
    off_diagonal = np.full(grid_points - 1, kinetic_off_diagonal, dtype=float)

    eigenvalues, eigenvectors_columns = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, num_states - 1),
    )
    wavefunctions = eigenvectors_columns.T.copy()
    normalize_wavefunctions_in_place(wavefunctions, dx)
    return eigenvalues, wavefunctions, x_grid


def normalize_wavefunctions_in_place(wavefunctions: np.ndarray, dx: float) -> None:
    for index, wavefunction in enumerate(wavefunctions):
        norm = np.sqrt(np.sum(wavefunction**2) * dx)
        if norm == 0.0:
            raise ValueError("Encountered a zero-norm wavefunction")
        wavefunctions[index] = wavefunction / norm
