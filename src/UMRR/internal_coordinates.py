"""Coordinate domains and the Cartesian kinetic energy of internal motion.

The direct map supplies positions and their derivatives. This base class
provides the physical mass matrix and velocity conversion, using Cartesian
lengths in bohr, atomic masses in electron masses, and angles in radians.
"""

from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


class CoordinateDomainError(ValueError):
    """A requested shape cannot be represented in the current coordinate chart."""

    def __init__(self, message, reason="coordinate_domain"):
        super().__init__(message)
        self.reason = reason


def periodic_delta(value):
    return (np.asarray(value) + np.pi) % TWOPI - np.pi


class InternalCoordinateMap:
    """Kinematics common to the direct path and minimum coordinate maps."""

    def _phase(self, angle):
        return (float(angle) - self.path_angles[0]) % TWOPI + self.path_angles[0]

    def base(self, angle, derivative=0):
        if not self.has_rotor:
            return self.base_values.copy() if derivative == 0 else np.zeros_like(self.base_values)
        phi = self._phase(angle)
        return self.base_spline(phi, derivative) + (
            self.windings * phi if derivative == 0 else self.windings if derivative == 1 else 0.0
        )

    def mass_metric(self, geometry_bohr):
        """Return g=(J M^-1 J^T)^-1 for z=(phi,q), or z=q without a rotor."""
        b = self.jacobian(geometry_bohr) / self.sqrt_masses
        singular = np.linalg.svd(b, compute_uv=False)
        if singular[-1] < 1.0e-10 * singular[0]:
            raise CoordinateDomainError("Singular internal mass metric")
        return np.linalg.inv(b @ b.T)

    def velocity(self, geometry_bohr, canonical_momenta):
        """Convert internal-coordinate momenta to rigid-body-free velocities.

        Internal-coordinate gradients annihilate external translations and
        rotations. Therefore M^-1 J^T p has P=L=0 and contains all kinetic
        cross terms. One uniform velocity scaling then enforces the fixed energy constraint.
        """
        return (
            (self.jacobian(geometry_bohr).T @ np.asarray(canonical_momenta)) / self.sqrt_masses**2
        ).reshape((-1, 3))

    @classmethod
    def from_dict(cls, record):
        """Restore an ordinary direct map and its saved mode coefficients."""
        from .direct_coordinates import DirectBasinChart, DirectInternalCoordinateMap

        definition = record.get("definition")
        if definition == "direct_independent_internal_v2":
            return DirectInternalCoordinateMap.from_dict(record)
        if definition == "direct_minimum_chart_v2":
            return DirectBasinChart.from_dict(record)
        raise ValueError(f"Unsupported paper coordinate definition: {definition}")
