"""Direct internal displacements with local normal-mode tangents.

The electronic potential is scanned along the same geometry map used for
sampling. The transverse quantum problems still use masses evaluated at the
reference geometry; direct placement does not make that approximation exact.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator, PPoly
from scipy.linalg import null_space

from .direct_internal import DirectDihedral, DirectInternals
from .internal_coordinates import (
    TWOPI,
    CoordinateDomainError,
    InternalCoordinateMap,
    periodic_delta,
)
from .qchem_io import internal_coordinate_basis


def _path_spline(angles, values, lower, upper):
    """Retain periodic cubic interpolation unless a length or bend overshoots.

    For an offending coordinate only, periodic shape-preserving Hermite slopes
    keep interpolated values between neighboring scan values. No values are
    clipped, and the path and its first derivative remain continuous.
    """
    x = np.r_[angles, angles[0] + TWOPI]
    y = np.vstack([values, values[0]])
    spline = CubicSpline(x, y, bc_type="periodic")
    changed = []
    for k in range(values.shape[1]):
        if not np.isfinite(lower[k]) and not np.isfinite(upper[k]):
            continue
        cut = PPoly(spline.c[:, :, k], spline.x)
        locations = np.r_[x, cut.derivative().roots(extrapolate=False)]
        extrema = cut(locations[np.isfinite(locations)])
        if np.min(extrema) > lower[k] and np.max(extrema) < upper[k]:
            continue
        if np.any(y[:, k] <= lower[k]) or np.any(y[:, k] >= upper[k]):
            raise CoordinateDomainError(
                "Relaxed scan includes an invalid distance or bend",
                "path_coordinate_domain",
            )
        extended = PchipInterpolator(
            np.r_[angles - TWOPI, angles, angles + TWOPI], np.tile(values[:, k], 3)
        )
        slopes = extended.derivative()(angles)
        bounded = CubicHermiteSpline(x, y[:, k], np.r_[slopes, slopes[0]])
        spline.c[:, :, k] = bounded.c
        changed.append(k)
    return spline, changed


def _align(geometry, reference, masses):
    """Choose a continuous external orientation without changing shape."""
    masses = np.asarray(masses)
    x = np.asarray(geometry) - np.average(geometry, axis=0, weights=masses)
    y = np.asarray(reference) - np.average(reference, axis=0, weights=masses)
    u, _, vh = np.linalg.svd((x * masses[:, None]).T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    return x @ (u @ correction @ vh)


def _define_transverse(coordinates, modes):
    """Convert projected Cartesian modes to fixed internal-coordinate changes.

    modes has one orthonormal mass-weighted Cartesian mode per column.
    Multiplying by the internal-coordinate Jacobian and M^(-1/2) gives D,
    the changes in bond lengths and angles per unit mode displacement.
    The inverse of its transverse block converts a geometry back to q.
    """
    jacobian = coordinates.primitives.derivatives(coordinates.reference).reshape(
        len(coordinates.periodic), -1
    )
    bounded_dihedrals = np.flatnonzero(
        coordinates.periodic
        & ~np.isin(np.arange(len(jacobian)), coordinates.primitives.rotor_positions)
    )
    displacement = (jacobian / coordinates.sqrt_masses) @ modes
    if coordinates.has_rotor:
        displacement[coordinates.primitives.rotor_positions] = 0.0
    transverse_positions = np.setdiff1d(
        np.arange(len(jacobian)), coordinates.primitives.rotor_positions
    )
    transverse_block = displacement[transverse_positions]
    if (
        transverse_block.shape[0] != transverse_block.shape[1]
        or np.linalg.cond(transverse_block) > 1.0e10
    ):
        raise CoordinateDomainError(
            "Direct internal coordinates do not span every transverse vibration"
        )
    coordinates.displacement = displacement
    coordinates.coefficients = np.zeros((modes.shape[1], len(jacobian)))
    coordinates.coefficients[:, transverse_positions] = np.linalg.inv(transverse_block)
    coordinates.transverse_positions = transverse_positions
    coordinates.bounded_dihedrals = bounded_dihedrals


class DirectInternalCoordinateMap(InternalCoordinateMap):
    """A noniterative geometry map with a single optional periodic rotor.

    Other dihedrals describe a local vibration on one angular branch. Requested
    displacements outside (-pi, pi) are rejected, not wrapped into a new geometry.
    """

    definition = "direct_independent_internal_v2"

    def __init__(
        self,
        symbols,
        masses,
        reference_bohr,
        hessian_mass_weighted,
        *,
        rotor_indices=None,
        rotor_permutations=None,
        rotor_phases=None,
        path_geometries_bohr=None,
        tolerance=1.0e-8,
        max_iterations=80,
    ):

        self.definition = "direct_independent_internal_v2"
        self.symbols = list(symbols)
        self.masses = np.asarray(masses, float)
        self.sqrt_masses = np.repeat(np.sqrt(self.masses), 3)
        self.reference = np.asarray(reference_bohr, float)
        self.hessian_mass_weighted = np.asarray(hessian_mass_weighted, float)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.rotor_indices = None if rotor_indices is None else list(map(int, rotor_indices))
        self.rotor = None if rotor_indices is None else DirectDihedral(*self.rotor_indices)
        self.has_rotor = self.rotor is not None
        # Symmetry belongs to complete potential copies. A circular average
        # of several dihedrals is not one independent atom-placement variable.
        self.rotor_permutations = None
        self.rotor_phases = None
        self.input_path_geometries = (
            None if not self.has_rotor else np.asarray(path_geometries_bohr, float)
        )
        paths = self.reference[None] if not self.has_rotor else self.input_path_geometries
        self.primitives = DirectInternals(
            self.reference,
            paths,
            [] if not self.has_rotor else [self.rotor_indices],
            self.symbols,
        )
        self.periodic = self.primitives.periodic.copy()
        self.bounded_path_interpolation_columns = []
        if self.has_rotor:
            phi = np.array([self.rotor.value(x) % TWOPI for x in paths])
            order = np.argsort(phi)
            keep = np.r_[True, np.diff(phi[order]) > 1.0e-7]
            self.path_order = order[keep]
            self.path_angles = phi[self.path_order]
            if len(self.path_angles) < 4:
                raise ValueError("At least four distinct relaxed torsion geometries are required")
            values = np.array([self.primitives.calculate(x) for x in paths[self.path_order]])
            values[:, self.periodic] = np.unwrap(values[:, self.periodic], axis=0)
            self.windings = np.zeros(values.shape[1])
            self.windings[self.periodic] = np.rint(
                (
                    values[-1, self.periodic]
                    - values[0, self.periodic]
                    + periodic_delta(values[0, self.periodic] - values[-1, self.periodic])
                )
                / TWOPI
            )
            detrended = values - self.path_angles[:, None] * self.windings
            self.base_spline, self.bounded_path_interpolation_columns = _path_spline(
                self.path_angles, detrended, *self.primitives.limits()
            )
            self.base_values = None
        else:
            self.path_angles = np.array([0.0])
            self.path_order = np.array([0])
            self.windings = np.zeros(len(self.periodic))
            self.base_values = self.primitives.calculate(self.reference)
        # First remove rigid translations and rotations in mass-weighted
        # Cartesian space, then impose d(phi)=0 for the transverse Hessian.
        physical = internal_coordinate_basis(self.reference, self.masses)
        if self.has_rotor:
            row = self.rotor.derivative(self.reference).ravel() / self.sqrt_masses
            physical = physical @ null_space((row @ physical)[None])
        h = physical.T @ self.hessian_mass_weighted @ physical
        # Diagonalizing this restricted Hessian gives 3N-7 modes with a
        # selected rotor, or 3N-6 modes without one.
        eigen, rotation = np.linalg.eigh((h + h.T) / 2)
        if eigen.min() < -1.0e-12:
            raise ValueError(
                "Reference has an imaginary internal vibration; use a minimum frequency calculation"
            )
        self.frequencies_hartree = np.sqrt(np.maximum(eigen, 0.0))
        modes = physical @ rotation
        self.normal_modes_mass_weighted = modes.T
        self.n_modes = modes.shape[1]
        self.row_scale = np.ones(self.n_modes + int(self.has_rotor))
        _define_transverse(self, modes)

    def base(self, angle, derivative=0):
        result = super().base(angle, derivative)
        if self.has_rotor:
            result[self.primitives.rotor_positions] = (
                angle if derivative == 0 else 1.0 if derivative == 1 else 0.0
            )
        return result

    def _base_at(self, angle):
        return self.base(angle), self.base(angle, 1)

    def coordinates(self, geometry_bohr):
        x = np.asarray(geometry_bohr)
        phi = float(self.rotor.value(x) % TWOPI) if self.has_rotor else 0.0
        values, _ = self._base_at(phi)
        diff = self.primitives.calculate(x) - values
        diff[self.periodic] = periodic_delta(diff[self.periodic])
        if not np.isfinite(diff).all():
            raise CoordinateDomainError("Geometry lies on a transverse dihedral chart boundary")
        return phi, self.coefficients @ diff

    def jacobian(self, geometry_bohr):
        x = np.asarray(geometry_bohr)
        phi = float(self.rotor.value(x) % TWOPI) if self.has_rotor else 0.0
        base, derivative = self._base_at(phi)
        b = self.primitives.derivatives(x).reshape(len(self.periodic), -1)
        if self.has_rotor:
            row = self.rotor.derivative(x).ravel()
            b -= derivative[:, None] * row
        transverse = self.coefficients @ b
        return np.vstack([row, transverse]) if self.has_rotor else transverse

    def geometry(self, angle, amplitudes, *, seed=None):
        q = np.asarray(amplitudes, float)
        if q.shape != (self.n_modes,) or not np.isfinite(q).all():
            raise ValueError("One finite amplitude per transverse vibration is required")
        delta = self.displacement @ q
        if np.any(np.abs(delta[self.bounded_dihedrals]) >= np.pi):
            raise CoordinateDomainError(
                "Nonrotor dihedral displacement leaves (-pi, pi)",
                "nonperiodic_dihedral_limit",
            )
        values, _ = self._base_at(float(angle))
        return _align(self.primitives.geometry(values + delta), self.reference, self.masses)

    def axis_scan_limits(self, max_dihedral_change_deg=150.0):
        """Stop an ordinary vibration before its nonperiodic angular boundary."""
        if not 0 < max_dihedral_change_deg < 180:
            raise ValueError("Transverse dihedral scan limits must be between 0 and 180 degrees")
        rates = (
            np.max(np.abs(self.displacement[self.bounded_dihedrals]), axis=0)
            if len(self.bounded_dihedrals)
            else np.zeros(self.n_modes)
        )
        result = np.full(self.n_modes, np.inf)
        moving = rates > 1.0e-14
        angle = np.deg2rad(max_dihedral_change_deg)
        result[moving] = angle / rates[moving]
        return result

    def axis_scan_bounds(self, angles, max_dihedral_change_deg=150.0):
        """Intersect the physical domains at all requested reference angles.

        Only one amplitude changes on an axis scan, so lengths and bends give
        exact linear inequalities. Mixed proposals require a separate geometry
        check because valid individual amplitudes need not have a valid sum.
        """
        angular = self.axis_scan_limits(max_dihedral_change_deg)
        lower, upper = -angular.copy(), angular.copy()
        reasons = [["nonperiodic_dihedral_limit"] * self.n_modes for _ in range(2)]
        primitives = [[None] * self.n_modes for _ in range(2)]
        lo, hi = self.primitives.limits()
        for angle in angles:
            base, _ = self._base_at(float(angle))
            self.primitives.validate(base)
            for k, (kind, _) in enumerate(self.primitives.definitions):
                if not np.isfinite(lo[k]) and not np.isfinite(hi[k]):
                    continue
                for m, rate in enumerate(self.displacement[k]):
                    if abs(rate) < 1.0e-14:
                        continue
                    a, b = sorted(((lo[k] - base[k]) / rate, (hi[k] - base[k]) / rate))
                    reason = "distance_limit" if kind == "length" else "bend_limit"
                    if a > lower[m]:
                        lower[m] = a
                        reasons[0][m], primitives[0][m] = (
                            reason,
                            self.primitives.names[k],
                        )
                    if b < upper[m]:
                        upper[m] = b
                        reasons[1][m], primitives[1][m] = (
                            reason,
                            self.primitives.names[k],
                        )
        # Stay strictly inside the open numerical domains, without clipping a
        # requested geometry after it has been constructed.
        return dict(
            lower=lower * (1 - 1.0e-6),
            upper=upper * (1 - 1.0e-6),
            reasons=reasons,
            primitives=primitives,
        )

    def to_dict(self):
        return dict(
            definition=self.definition,
            backend="direct_atom_placement",
            units="Cartesian bohr, electron masses, rotor radians, mass-weighted transverse amplitudes",
            symbols=self.symbols,
            masses_electron=self.masses,
            reference_bohr=self.reference,
            hessian_mass_weighted=self.hessian_mass_weighted,
            path_geometries_bohr=self.input_path_geometries,
            rotor_indices_zero_based=self.rotor_indices,
            rotor_permutations=None,
            rotor_phases=None,
            primitives=self.primitives.names,
            coefficients=self.coefficients,
            displacement=self.displacement,
            windings=self.windings,
            frequencies_hartree=self.frequencies_hartree,
            coordinate_variables="ordinary_lengths_and_angles",
            bounded_path_interpolation_columns=self.bounded_path_interpolation_columns,
            normal_modes_mass_weighted=self.normal_modes_mass_weighted,
            n_modes=self.n_modes,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            approximation="One periodic rotor; other dihedrals are bounded nonperiodic transverse coordinates; transverse kinetic masses evaluated locally",
        )

    @classmethod
    def from_dict(cls, record):
        if record.get("definition") not in {"direct_independent_internal_v2"}:
            raise ValueError("Unknown direct coordinate definition")
        c = cls(
            record["symbols"],
            record["masses_electron"],
            record["reference_bohr"],
            record["hessian_mass_weighted"],
            rotor_indices=record["rotor_indices_zero_based"],
            path_geometries_bohr=record["path_geometries_bohr"],
            tolerance=record.get("tolerance", 1.0e-8),
            max_iterations=record.get("max_iterations", 80),
        )
        if c.primitives.names != record["primitives"]:
            raise ValueError("Direct atom placement differs from the saved coordinate definition")
        for name in (
            "coefficients",
            "displacement",
            "frequencies_hartree",
            "normal_modes_mass_weighted",
        ):
            setattr(c, name, np.asarray(record[name], float))
        return c


class DirectBasinChart(DirectInternalCoordinateMap):
    """Each minimum's modes about the same physical relaxed geometry.

    Equivalent minima are evaluated through complete scalar-potential relabeling.
    This chart constructs proposals at a specified global rotor angle. Different
    minima retain their own coefficients and their own electronic spline scans.
    """

    definition = "direct_minimum_chart_v2"

    def __init__(self, local, shared_path, permutation):
        self.__dict__.update(local.__dict__)
        self.definition = "direct_minimum_chart_v2"
        self.local = local
        self.shared_path = shared_path
        self.permutation = np.asarray(permutation, int)
        self.inverse_permutation = np.argsort(self.permutation)
        n = len(local.symbols)
        if (
            self.permutation.shape != (n,)
            or not np.array_equal(self.permutation, permutation)
            or sorted(self.permutation.tolist()) != list(range(n))
        ):
            raise ValueError("A basin mapping must be a permutation of every atom")
        if not np.array_equal(
            np.asarray(local.symbols)[self.permutation], local.symbols
        ) or not np.array_equal(local.masses[self.permutation], local.masses):
            raise ValueError("Equivalent-basin permutations must preserve elements and masses")
        global_c = shared_path.coordinates
        self.rotor = global_c.rotor
        self.rotor_indices = global_c.rotor_indices
        self.has_rotor = global_c.has_rotor
        self.path_angles = global_c.path_angles
        self.reference = local.reference[self.permutation]
        paths = self.reference[None] if not self.has_rotor else global_c.input_path_geometries
        self.primitives = DirectInternals(
            self.reference,
            paths,
            [] if not self.has_rotor else [self.rotor_indices],
            self.symbols,
        )
        self.periodic = self.primitives.periodic.copy()
        modes = (
            local.normal_modes_mass_weighted.reshape(local.n_modes, n, 3)[:, self.permutation]
            .reshape(local.n_modes, -1)
            .T
        )
        if self.has_rotor and not np.array_equal(self.permutation, np.arange(n)):
            # A relabeled methyl H defines a different constant-angle slice.
            # Adjust only the proposal tangent; complete potential evaluation
            # still uses the representative chart on the relabeled geometry.
            row = self.rotor.derivative(self.reference).ravel() / self.sqrt_masses
            modes = modes - np.outer(row, row @ modes) / (row @ row)
        _define_transverse(self, modes)

    def _base_at(self, angle):
        x, dx = self.shared_path.at(float(angle % TWOPI))
        values = self.primitives.calculate(x)
        b = self.primitives.derivatives(x).reshape(len(self.periodic), -1)
        return values, b @ dx.ravel()

    def to_dict(self):
        return dict(
            definition=self.definition,
            local_coordinate_map=self.local.to_dict(),
            path_coordinate_map=self.shared_path.coordinates.to_dict(),
            permutation=self.permutation,
            primitives=self.primitives.names,
            coefficients=self.coefficients,
            displacement=self.displacement,
        )

    @classmethod
    def from_dict(cls, record):
        from .basin_sampler import SharedPath

        chart = cls(
            DirectInternalCoordinateMap.from_dict(record["local_coordinate_map"]),
            SharedPath(DirectInternalCoordinateMap.from_dict(record["path_coordinate_map"])),
            record["permutation"],
        )
        if chart.definition != record["definition"]:
            raise ValueError("Minimum and saved coordinate definitions disagree")
        if chart.primitives.names != record["primitives"]:
            raise ValueError(
                "Direct basin atom placement differs from the saved coordinate definition"
            )
        coefficients = np.asarray(record["coefficients"], float)
        displacement = np.asarray(record["displacement"], float)
        if (
            coefficients.shape != chart.coefficients.shape
            or displacement.shape != chart.displacement.shape
            or not np.isfinite(coefficients).all()
            or not np.isfinite(displacement).all()
            or not np.allclose(
                coefficients @ displacement, np.eye(chart.n_modes), atol=1.0e-9, rtol=0
            )
            or (
                chart.has_rotor
                and np.max(np.abs(displacement[chart.primitives.rotor_positions])) > 1.0e-12
            )
        ):
            raise ValueError(
                "Saved direct basin displacement and inverse matrices are inconsistent"
            )
        chart.coefficients = coefficients
        chart.displacement = displacement
        return chart
