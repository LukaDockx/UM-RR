"""Momentum distributions obtained from numerical one-dimensional wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq, fftshift, next_fast_len

DEFAULT_ZERO_PADDING_FACTOR = 8
DEFAULT_MAX_ENDPOINT_DENSITY_RATIO = 1.0e-3


@dataclass(slots=True)
class MomentumDistribution:
    """Inverse-CDF representation of one mass-weighted momentum marginal."""

    momenta_au: np.ndarray
    cdf: np.ndarray
    normalization_before_rescaling: float
    second_moment_au: float
    endpoint_density_ratio: float | None = None

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one momentum from the continuous inverse-CDF representation."""
        probability = float(rng.random())
        upper = int(np.searchsorted(self.cdf, probability, side="right"))
        upper = min(max(upper, 1), len(self.cdf) - 1)
        lower = upper - 1
        cdf_width = float(self.cdf[upper] - self.cdf[lower])
        if cdf_width <= 0.0:
            return float(self.momenta_au[upper])
        fraction = (probability - float(self.cdf[lower])) / cdf_width
        return float(
            self.momenta_au[lower] + fraction * (self.momenta_au[upper] - self.momenta_au[lower])
        )


@dataclass(slots=True)
class PeriodicMomentumDistribution:
    """Categorical momentum marginal for a wavefunction on a periodic path."""

    momenta_au: np.ndarray
    probabilities: np.ndarray
    normalization_before_rescaling: float
    continuum_kinetic_energy_hartree: float
    finite_difference_kinetic_energy_hartree: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one of the exact Fourier-series momentum eigenvalues."""
        index = int(rng.choice(np.arange(len(self.momenta_au)), p=self.probabilities))
        return float(self.momenta_au[index])

    @property
    def zero_momentum_probability(self) -> float:
        index = int(np.argmin(np.abs(self.momenta_au)))
        if not np.isclose(self.momenta_au[index], 0.0, rtol=0.0, atol=1.0e-14):
            return 0.0
        return float(self.probabilities[index])


def periodic_path_momentum_distributions(
    wavefunctions: np.ndarray,
    total_path_length: float,
) -> list[PeriodicMomentumDistribution]:
    """Return exact discrete Fourier momentum marginals on a periodic path.

    The coordinate is assumed to be mass weighted, so the allowed canonical
    path momenta ``2*pi*k/L`` are numerically equal to path velocities in atomic
    units.  Both the continuum kinetic moment ``p**2/2`` and the exact
    second-order finite-difference kinetic moment are reported.
    """
    momenta, amplitudes = periodic_path_momentum_wavefunctions(
        wavefunctions,
        total_path_length,
    )
    raw_probabilities = np.abs(amplitudes) ** 2
    point_count = raw_probabilities.shape[1]
    spacing = float(total_path_length) / point_count
    finite_difference_kinetic = (1.0 - np.cos(momenta * spacing)) / spacing**2
    distributions = []
    for state_probabilities in raw_probabilities:
        normalization = float(np.sum(state_probabilities))
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("Periodic momentum distribution has zero normalization")
        probabilities = np.asarray(state_probabilities / normalization, dtype=float)
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= np.sum(probabilities)
        distributions.append(
            PeriodicMomentumDistribution(
                momenta_au=momenta,
                probabilities=probabilities,
                normalization_before_rescaling=normalization,
                continuum_kinetic_energy_hartree=float(np.sum(0.5 * momenta**2 * probabilities)),
                finite_difference_kinetic_energy_hartree=float(
                    np.sum(finite_difference_kinetic * probabilities)
                ),
            )
        )
    return distributions


def periodic_path_momentum_wavefunctions(
    wavefunctions: np.ndarray,
    total_path_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return allowed path momenta and normalized Fourier-series coefficients."""
    psi = _as_wavefunction_matrix(wavefunctions)
    length = float(total_path_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("total_path_length must be positive and finite")
    point_count = psi.shape[1]
    spacing = length / point_count
    norms = np.sqrt(np.sum(np.abs(psi) ** 2, axis=1) * spacing)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Encountered a zero-norm periodic wavefunction")
    psi = psi / norms[:, None]
    amplitudes = fftshift(fft(psi, axis=1), axes=1) * spacing / np.sqrt(length)
    momenta = 2.0 * np.pi * fftshift(fftfreq(point_count, d=spacing))
    return momenta, amplitudes


def normal_mode_momentum_distributions(
    wavefunctions: np.ndarray,
    coordinate_grid: np.ndarray,
    reduced_mass_electron: float,
    *,
    zero_padding_factor: int = DEFAULT_ZERO_PADDING_FACTOR,
    max_endpoint_density_ratio: float = DEFAULT_MAX_ENDPOINT_DENSITY_RATIO,
) -> list[MomentumDistribution]:
    """Fourier transform bound-state wavefunctions on a uniform finite grid.

    The finite-difference solver represents Dirichlet ghost points immediately
    outside its stored coordinate grid.  Explicit zero-valued ghost points are
    therefore added before zero padding.  The Fourier-conjugate momentum of the
    reduced coordinate is divided by ``sqrt(reduced_mass_electron)`` so that the
    returned variable is the mass-weighted normal momentum, numerically equal to
    the mass-weighted normal velocity in atomic units.
    """
    mass_weighted_momenta, amplitudes, endpoint_ratios = _normal_mode_momentum_transform(
        wavefunctions,
        coordinate_grid,
        reduced_mass_electron,
        zero_padding_factor=zero_padding_factor,
        max_endpoint_density_ratio=max_endpoint_density_ratio,
    )
    momentum_density = np.abs(amplitudes) ** 2
    distributions = []
    for state_index, state_density in enumerate(momentum_density):
        cdf, normalization = _continuous_cdf(state_density, mass_weighted_momenta)
        second_moment = float(
            np.sum(
                0.5
                * (
                    state_density[:-1] * mass_weighted_momenta[:-1] ** 2
                    + state_density[1:] * mass_weighted_momenta[1:] ** 2
                )
                * np.diff(mass_weighted_momenta)
            )
            / normalization
        )
        distributions.append(
            MomentumDistribution(
                momenta_au=mass_weighted_momenta,
                cdf=cdf,
                normalization_before_rescaling=float(normalization),
                second_moment_au=second_moment,
                endpoint_density_ratio=float(endpoint_ratios[state_index]),
            )
        )
    return distributions


def normal_mode_momentum_wavefunctions(
    wavefunctions: np.ndarray,
    coordinate_grid: np.ndarray,
    reduced_mass_electron: float,
    *,
    zero_padding_factor: int = DEFAULT_ZERO_PADDING_FACTOR,
    max_endpoint_density_ratio: float = DEFAULT_MAX_ENDPOINT_DENSITY_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass-weighted momentum grid and normalized complex amplitudes."""
    momenta, amplitudes, _endpoint_ratios = _normal_mode_momentum_transform(
        wavefunctions,
        coordinate_grid,
        reduced_mass_electron,
        zero_padding_factor=zero_padding_factor,
        max_endpoint_density_ratio=max_endpoint_density_ratio,
    )
    return momenta, amplitudes


def _normal_mode_momentum_transform(
    wavefunctions: np.ndarray,
    coordinate_grid: np.ndarray,
    reduced_mass_electron: float,
    *,
    zero_padding_factor: int,
    max_endpoint_density_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi = _as_wavefunction_matrix(wavefunctions)
    grid, spacing = _uniform_grid(coordinate_grid, len(psi[0]))
    reduced_mass = float(reduced_mass_electron)
    if not np.isfinite(reduced_mass) or reduced_mass <= 0.0:
        raise ValueError("reduced_mass_electron must be positive and finite")
    if int(zero_padding_factor) != zero_padding_factor or zero_padding_factor < 1:
        raise ValueError("zero_padding_factor must be a positive integer")
    if max_endpoint_density_ratio < 0.0:
        raise ValueError("max_endpoint_density_ratio cannot be negative")

    density = np.abs(psi) ** 2
    maxima = np.max(density, axis=1)
    if np.any(maxima <= 0.0):
        raise ValueError("Encountered a zero-norm wavefunction")
    endpoint_ratios = np.max(density[:, [0, -1]], axis=1) / maxima
    worst_state = int(np.argmax(endpoint_ratios))
    if endpoint_ratios[worst_state] > max_endpoint_density_ratio:
        raise ValueError(
            "Wavefunction amplitude is too large at the finite-grid boundary for a "
            "reliable Fourier transform: state "
            f"{worst_state} has endpoint density/max density="
            f"{endpoint_ratios[worst_state]:.3e}, above "
            f"{max_endpoint_density_ratio:.3e}. Increase the coordinate scan range "
            "or solve fewer states."
        )

    norms = np.sqrt(np.sum(density, axis=1) * spacing)
    psi = psi / norms[:, None]
    stored_points = psi.shape[1]
    signal_points = stored_points + 2
    padded_points = next_fast_len(int(zero_padding_factor) * signal_points)
    signal = np.zeros((len(psi), padded_points), dtype=np.result_type(psi, complex))
    signal[:, 1 : stored_points + 1] = psi

    amplitudes = fftshift(fft(signal, axis=1), axes=1) * spacing / np.sqrt(2.0 * np.pi)
    conjugate_momenta = 2.0 * np.pi * fftshift(fftfreq(padded_points, d=spacing))
    if padded_points % 2 == 0:
        conjugate_momenta = conjugate_momenta[1:]
        amplitudes = amplitudes[:, 1:]

    # Restore the phase associated with the physical coordinate origin. The
    # first FFT sample is the left zero-valued ghost point at q_min - dq.
    ghost_origin = float(grid[0] - spacing)
    amplitudes *= np.exp(-1.0j * conjugate_momenta * ghost_origin)[None, :]

    mass_weighted_momenta = conjugate_momenta / np.sqrt(reduced_mass)
    mass_weighted_amplitudes = amplitudes * reduced_mass**0.25
    return mass_weighted_momenta, mass_weighted_amplitudes, endpoint_ratios


def _as_wavefunction_matrix(wavefunctions: np.ndarray) -> np.ndarray:
    psi = np.asarray(wavefunctions)
    if psi.ndim == 1:
        psi = psi[None, :]
    if psi.ndim != 2 or psi.shape[1] < 3:
        raise ValueError("wavefunctions must contain at least three grid values per state")
    if not np.all(np.isfinite(psi)):
        raise ValueError("wavefunctions contain non-finite values")
    return psi


def _uniform_grid(coordinate_grid: np.ndarray, expected_points: int) -> tuple[np.ndarray, float]:
    grid = np.asarray(coordinate_grid, dtype=float)
    if grid.ndim != 1 or len(grid) != expected_points:
        raise ValueError("coordinate_grid must have one value per wavefunction grid point")
    if not np.all(np.isfinite(grid)):
        raise ValueError("coordinate_grid contains non-finite values")
    differences = np.diff(grid)
    if np.any(differences <= 0.0):
        raise ValueError("coordinate_grid must be strictly increasing")
    spacing = float(differences[0])
    if not np.allclose(differences, spacing, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("A uniform coordinate grid is required for the Fourier transform")
    return grid, spacing


def _trapezoid_interval_masses(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return 0.5 * (density[:-1] + density[1:]) * np.diff(grid)


def _continuous_cdf(density: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    interval_masses = _trapezoid_interval_masses(density, grid)
    normalization = float(np.sum(interval_masses))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Momentum distribution has zero normalization")
    cdf = np.r_[0.0, np.cumsum(interval_masses) / normalization]
    cdf[-1] = 1.0
    return cdf, normalization
