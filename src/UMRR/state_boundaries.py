"""Report whether independent-mode states occupy a finite scan's edges."""

import numpy as np

from .momentum import (
    DEFAULT_MAX_ENDPOINT_DENSITY_RATIO,
    normal_mode_momentum_distributions,
)


def prepare_transverse_state_distributions(
    energies, wavefunctions, grid, mass, endpoint_energy, kT, tolerance, log, label
):
    """Continue with the supplied finite-grid states when confinement fails.

    Bound levels keep the existing selection below both endpoint potentials.
    If none qualify, retain all computed levels rather than changing the
    potential or substituting a harmonic ground state. The warning and saved
    diagnostics distinguish these box-dependent levels from confined states.
    """
    confined = int(np.count_nonzero(np.asarray(energies) < endpoint_energy))
    retained = confined if confined else len(energies)
    if not confined:
        log(
            f"WARNING: {label}: no computed state lies below both endpoint potentials. "
            f"Continuing with {retained} finite-grid states from the unchanged spline; "
            "their energies and densities may be controlled by the scan boundary. Inspect the plots."
        )
    # Finite-grid states have a defined Fourier transform even if their tails
    # make that transform a poor approximation to an isolated molecular state.
    # Report that limitation without preventing the requested sampling.
    momentum = normal_mode_momentum_distributions(
        wavefunctions[:retained], grid, mass, max_endpoint_density_ratio=np.inf
    )
    ratios = np.array([p.endpoint_density_ratio for p in momentum])
    bad = np.flatnonzero(ratios > DEFAULT_MAX_ENDPOINT_DENSITY_RATIO)
    if len(bad):
        log(
            f"WARNING: {label}: {len(bad)} retained state(s) have appreciable density "
            "at an endpoint. Continuing with their finite-grid Fourier momentum distributions."
        )
    boundary = boundary_diagnostics(energies, wavefunctions, grid, retained, kT, tolerance)
    boundary.update(
        confined_state_count=confined,
        confinement_check_passed=confined > 0,
        finite_grid_fallback=confined == 0,
        state_selection=(
            "levels_below_both_endpoints" if confined else "all_computed_finite_grid_levels"
        ),
        lower_endpoint_energy_hartree=float(endpoint_energy),
        endpoint_density_ratios=ratios.tolist(),
        momentum_boundary_check_passed=len(bad) == 0,
        momentum_boundary_density_ratio_limit=DEFAULT_MAX_ENDPOINT_DENSITY_RATIO,
    )
    if (
        not boundary["boundary_check_passed"]
        or boundary["excluded_probability_within_computed_levels"] > 0.01
    ):
        log(
            f"WARNING: {label}: boundary probability {boundary['thermal_boundary_probability']:.3%}; "
            f"computed-level excluded population {boundary['excluded_probability_within_computed_levels']:.3%}. "
            "Continuing; these finite-domain states require inspection."
        )
    return retained, momentum, boundary


def boundary_diagnostics(energies, wavefunctions, grid, retained, kT, tolerance):
    if not 0 < tolerance < 1:
        raise ValueError("Boundary probability tolerance must be between zero and one")
    e = np.asarray(energies)
    x = np.asarray(grid)
    psi = np.asarray(wavefunctions)
    dx = float(x[1] - x[0])
    strip = 0.05 * (x[-1] - x[0])
    negative = np.sum(psi[:, x <= x[0] + strip] ** 2, axis=1) * dx
    positive = np.sum(psi[:, x >= x[-1] - strip] ** 2, axis=1) * dx
    if kT > 0:
        weights = np.exp(-(e - e[0]) / kT)
        weights /= weights.sum()
    else:
        weights = np.isclose(e, e[0], rtol=0, atol=1.0e-12).astype(float)
        weights /= weights.sum()
    norm = float(weights[:retained].sum())
    boundary = float(np.dot(weights[:retained], (negative + positive)[:retained]) / norm)
    return dict(
        outer_fraction_each_side=0.05,
        negative_boundary_probabilities=negative[:retained].tolist(),
        positive_boundary_probabilities=positive[:retained].tolist(),
        thermal_boundary_probability=boundary,
        tolerance=tolerance,
        boundary_check_passed=boundary <= tolerance,
        excluded_probability_within_computed_levels=float(weights[retained:].sum()),
        highest_computed_level_probability=float(weights[-1]),
        retained_states=retained,
        computed_states=len(e),
        interpretation="Finite-domain independent-mode proposal. Excluded probability only covers computed levels; it is not a bound on missing rotor states or the final energy-conditioned ensemble.",
    )
