"""Retain complete rotor multiplets until omitted thermal population is small."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .constants import BOLTZMANN_EV_PER_K, HARTREE_TO_EV
from .relaxed_torsion import solve_periodic_path_schrodinger


def solve_thermal_rotor_states(
    potential_hartree,
    total_path_length,
    *,
    num_states,
    temperature_kelvin,
    population_tail_tolerance=0.01,
    degeneracy_tolerance_hartree=1.0e-8,
    population_offsets_hartree=None,
):
    """Bound the omitted population of the *discretized* rotor Hamiltonian.

    A state-count expansion alone can miss an uncomputed tail. The ordered
    kinetic eigenvalues plus min(V) bound every ordered Hamiltonian eigenvalue
    from below (the min-max principle). Use that bound for uncomputed levels,
    and computed energies for the rest. This controls state truncation, not
    spatial-grid convergence or changes caused by proposal rejection.

    Optional grid offsets affect Boltzmann weights through their expectation
    values, as in the legacy transverse-ZPE correction. Their minimum also
    bounds offsets for uncomputed levels; the eigenfunctions stay unchanged.
    """
    v = np.asarray(potential_hartree, float)
    n = len(v)
    temperature = float(temperature_kelvin)
    tol = float(population_tail_tolerance)
    if n < 3 or not np.all(np.isfinite(v)) or total_path_length <= 0:
        raise ValueError("Rotor potential needs finite values on a positive-length grid")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Rotor temperature must be finite and nonnegative")
    if not 0 < tol < 1:
        raise ValueError("population_tail_tolerance must be between zero and one")
    if int(num_states) != num_states or not 1 <= num_states < n:
        raise ValueError("Rotor num_states must be positive and smaller than the grid")
    if degeneracy_tolerance_hartree < 0:
        raise ValueError("Degeneracy tolerance must be nonnegative")
    offset = (
        np.zeros(n)
        if population_offsets_hartree is None
        else np.asarray(population_offsets_hartree, float)
    )
    if offset.shape != v.shape or not np.all(np.isfinite(offset)):
        raise ValueError("Population offsets must match the rotor potential grid")
    dx = float(total_path_length) / n
    kinetic = np.sort(2 * np.sin(np.pi * np.arange(n) / n) ** 2 / dx**2)
    lower = kinetic + float(v.min())
    requested = int(num_states)
    count = min(n, max(requested + 4, 16))
    expansions = []
    kT = BOLTZMANN_EV_PER_K / HARTREE_TO_EV * temperature
    while True:
        energies, psi = solve_periodic_path_schrodinger(
            v,
            total_path_length,
            num_states=count,
            eigensolver_seed=0,
        )
        actual = len(energies)
        shifted = energies + dx * (psi**2 @ offset)
        # Ordered uncomputed energies also cannot lie below the last computed
        # eigenvalue. Subtract a small safety margin before using that value.
        unknown = np.maximum(
            lower[actual:],
            energies[-1] - degeneracy_tolerance_hartree,
        ) + float(offset.min())
        if kT > 0:
            zero = float(shifted.min())
            logweights = -(shifted - zero) / kT
            log_unknown = logsumexp(-(unknown - zero) / kT) if len(unknown) else -np.inf
        else:
            # Ground-state degeneracy is kept as a whole below. For nonconstant
            # population offsets, compute enough levels to exclude an unseen
            # lower corrected state before claiming zero omitted population.
            zero = float(shifted.min())
        for keep in range(requested, actual + 1):
            if (
                keep < actual
                and energies[keep] - energies[keep - 1] <= degeneracy_tolerance_hartree
            ):
                continue
            if keep == actual and actual < n:
                continue  # Need a following level to establish a complete group.
            if kT > 0:
                log_retained = logsumexp(logweights[:keep])
                log_tail = np.logaddexp(logsumexp(logweights[keep:]), log_unknown)
                bound = float(np.exp(log_tail - np.logaddexp(log_retained, log_tail)))
            else:
                retained_ground = np.any(shifted[:keep] <= zero + degeneracy_tolerance_hartree)
                omitted_ground = np.any(shifted[keep:] <= zero + degeneracy_tolerance_hartree)
                unknown_ground = np.any(unknown <= zero + degeneracy_tolerance_hartree)
                bound = (
                    0.0 if retained_ground and not omitted_ground and not unknown_ground else 1.0
                )
            if bound <= tol:
                return (
                    energies[:keep],
                    psi[:keep],
                    dict(
                        requested_minimum_states=requested,
                        retained_states=keep,
                        computed_states=actual,
                        omitted_population_upper_bound=bound,
                        population_tail_tolerance=tol,
                        temperature_kelvin=temperature,
                        degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
                        state_count_expansions=expansions + [actual],
                        scope="Finite-difference rotor Hamiltonian before proposal rejection; spatial-grid convergence is separate",
                        population_offsets_included=population_offsets_hartree is not None,
                    ),
                )
        if actual >= n:
            raise RuntimeError(
                "Failed to converge rotor state population with the full grid spectrum"
            )
        expansions.append(actual)
        count = min(n, max(actual + 8, int(np.ceil(1.6 * actual))))
