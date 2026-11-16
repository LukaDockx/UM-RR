from __future__ import annotations

import unittest

import numpy as np

from UMRR.schrodinger import solve_schrodinger_1d


class SchrodingerSolverTests(unittest.TestCase):
    def test_tridiagonal_solver_matches_dense_hamiltonian(self) -> None:
        reduced_mass = 2.5
        grid_points = 80
        num_states = 4

        def potential(x: np.ndarray) -> np.ndarray:
            return 0.25 * x**2 + 0.01 * x**4

        eigenvalues, wavefunctions, x_grid = solve_schrodinger_1d(
            potential,
            np.asarray([-4.0, 4.0]),
            reduced_mass,
            grid_points=grid_points,
            num_states=num_states,
        )

        dx = x_grid[1] - x_grid[0]
        diagonal = potential(x_grid) + 1.0 / (reduced_mass * dx**2)
        off_diagonal = np.full(grid_points - 1, -0.5 / (reduced_mass * dx**2))
        dense = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        dense_eigenvalues = np.linalg.eigh(dense)[0][:num_states]

        np.testing.assert_allclose(eigenvalues, dense_eigenvalues, rtol=0, atol=1.0e-10)
        norms = np.sum(wavefunctions**2, axis=1) * dx
        np.testing.assert_allclose(norms, np.ones(num_states), rtol=0, atol=1.0e-12)

    def test_harmonic_oscillator_uses_reduced_mass_in_kinetic_operator(self) -> None:
        reduced_mass = 2.5
        force_constant = 0.4
        num_states = 4

        def potential(q: np.ndarray) -> np.ndarray:
            return 0.5 * force_constant * q**2

        eigenvalues, _wavefunctions, _grid = solve_schrodinger_1d(
            potential,
            np.asarray([-8.0, 8.0]),
            reduced_mass,
            grid_points=800,
            num_states=num_states,
        )

        omega = np.sqrt(force_constant / reduced_mass)
        exact = omega * (np.arange(num_states) + 0.5)
        np.testing.assert_allclose(eigenvalues, exact, rtol=0, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
