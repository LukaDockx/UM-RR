"""State truncation must control total omitted weight, including multiplets."""

import numpy as np
import pytest

from UMRR.constants import BOLTZMANN_EV_PER_K, HARTREE_TO_EV
from UMRR.relaxed_torsion import solve_periodic_path_schrodinger
from UMRR.rotor_states import solve_thermal_rotor_states


@pytest.mark.parametrize("temperature", [0.0, 100.0, 300.0, 1500.0])
@pytest.mark.parametrize("offsets", [False, True])
def test_tail_bound_against_complete_spectrum(temperature, offsets):
    n = 180
    length = 600.0
    angle = np.arange(n) * 2 * np.pi / n
    potential = 0.004 * (1 - np.cos(3 * angle)) + 0.001 * np.sin(angle)
    grid_offsets = 0.006 * (angle > np.pi) if offsets else None
    energies, psi, report = solve_thermal_rotor_states(
        potential,
        length,
        num_states=3,
        temperature_kelvin=temperature,
        population_offsets_hartree=grid_offsets,
    )
    full, waves = solve_periodic_path_schrodinger(potential, length, num_states=n)
    populations = full.copy()
    if offsets:
        populations += (length / n) * (waves**2 @ grid_offsets)
    shifted = populations - populations.min()
    if temperature:
        weights = np.exp(-shifted / (BOLTZMANN_EV_PER_K / HARTREE_TO_EV * temperature))
    else:
        weights = np.isclose(shifted, 0.0, atol=1.0e-12, rtol=0.0).astype(float)
    tail = weights[len(energies) :].sum() / weights.sum()
    assert tail <= report["omitted_population_upper_bound"] + 1.0e-12
    assert report["omitted_population_upper_bound"] <= 0.01
    np.testing.assert_allclose(energies, full[: len(energies)], atol=1.0e-12)
    np.testing.assert_allclose(np.sum(psi**2, axis=1) * length / n, 1.0, atol=1.0e-12)


def test_degenerate_pair_cannot_be_split_at_requested_cutoff():
    # The analytically known periodic spectrum checks grouping, not the
    # separate free-rotor momentum-sampling edge case.
    energies, _, report = solve_thermal_rotor_states(
        np.zeros(100), 100.0, num_states=2, temperature_kelvin=0.0
    )
    assert len(energies) == 3
    assert abs(energies[1] - energies[2]) < 1.0e-12
    assert report["omitted_population_upper_bound"] == 0.0


def test_tail_convergence_can_require_more_than_one_expansion():
    _, _, report = solve_thermal_rotor_states(
        np.zeros(200), 2000.0, num_states=2, temperature_kelvin=2000.0
    )
    assert len(report["state_count_expansions"]) > 1
    assert report["retained_states"] > 2


@pytest.mark.parametrize(
    "temperature,tolerance", [(-1.0, 0.01), (300.0, 0.0), (300.0, 1.0), (float("nan"), 0.01)]
)
def test_invalid_thermal_controls(temperature, tolerance):
    with pytest.raises(ValueError):
        solve_thermal_rotor_states(
            np.zeros(40),
            100.0,
            num_states=2,
            temperature_kelvin=temperature,
            population_tail_tolerance=tolerance,
        )
