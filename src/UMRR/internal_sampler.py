"""Electronic one-mode curves and common steps of the UM-RR calculation.

BasinSampler supplies the minimum-specific geometry, quantum states and
ensemble generation. This module supplies scan jobs, independent potentials,
normalized coordinate densities and Boltzmann probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

from .constants import (
    ANGSTROM_TO_BOHR,
)
from .internal_coordinates import TWOPI, CoordinateDomainError
from .qchem_io import generate_qchem_input_text, parse_total_energies_hartree
from .qchem_runner import qchem_output_matches_input, run_qchem_filelist
from .serialization import write_json


def periodic_spline(angles, values):
    angles, values = np.asarray(angles), np.asarray(values)
    if len(angles) == 1:
        return lambda phi, nu=0: np.broadcast_to(
            values[0] if nu == 0 else np.zeros_like(values[0]),
            np.shape(phi) + values.shape[1:],
        )
    spline = CubicSpline(
        np.r_[angles, angles[0] + TWOPI],
        np.concatenate([values, values[:1]]),
        bc_type="periodic",
    )
    return lambda phi, nu=0: spline((np.asarray(phi) - angles[0]) % TWOPI + angles[0], nu)


def cdf_from_density(grid, density):
    cdf = np.r_[0.0, cumulative_trapezoid(np.maximum(density, 0.0), grid)]
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("Non-normalizable coordinate density")
    return cdf / cdf[-1]


class InternalPotential:
    """Sum electronic one-mode curves relative to one reference geometry.

    A minimum has one angle entry. Arrays retain an angle dimension so saved
    MeOOH curves can be loaded without altering their electronic energy zeros.
    """

    def __init__(
        self,
        coordinates,
        angles,
        grids,
        cuts,
        base_energies,
        symmetry_permutations=None,
        pair_coefficients=None,
        pair_cubic_coefficients=None,
    ):
        for values in (pair_coefficients, pair_cubic_coefficients):
            if values is not None and np.any(np.asarray(values) != 0):
                raise ValueError("The paper potential requires independent mode curves")
        self.coordinates = coordinates
        self.angles = np.asarray(angles)
        if len(self.angles) != 1:
            raise ValueError("Each minimum must have one set of transverse curves")
        self.grids = [np.asarray(grid) for grid in grids]
        self.base_energies = np.asarray(base_energies)
        self.reference_energy = float(np.min(self.base_energies))
        self._mode_cuts = [np.asarray([node[m] for node in cuts], float) for m in range(len(grids))]
        self._splines = [
            CubicSpline(grid, values[0], extrapolate=False)
            for grid, values in zip(self.grids, self._mode_cuts)
        ]

    def one_mode(self, angle, mode):
        """Return U_i(q_i), using the reference zero stored with that curve."""
        return self._splines[mode]

    def raw(self, angle, amplitudes):
        """Evaluate all curves at the same geometry's mode coordinates."""
        q = np.asarray(amplitudes, dtype=float)
        if q.shape != (len(self.grids),) or not np.all(np.isfinite(q)):
            raise CoordinateDomainError("One finite displacement per mode is required")
        if any(v < grid[0] or v > grid[-1] for v, grid in zip(q, self.grids)):
            raise CoordinateDomainError("Geometry lies outside a computed internal-coordinate cut")
        return float(sum(curve(value) for curve, value in zip(self._splines, q)))

    def evaluate_geometry(self, geometry_bohr):
        angle, q = self.coordinates.coordinates(geometry_bohr)
        return self.raw(angle, q)

    def save(self, path):
        arrays = dict(angles=self.angles, base_energies=self.base_energies)
        for mode, (grid, values) in enumerate(zip(self.grids, self._mode_cuts)):
            arrays[f"grid_{mode:03d}"] = grid
            arrays[f"cut_{mode:03d}"] = values
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, coordinates):
        """Load numerical curves, rejecting archives with mixed-mode terms."""
        with np.load(path, allow_pickle=False) as arrays:
            grids = [arrays[f"grid_{m:03d}"] for m in range(coordinates.n_modes)]
            cuts = [[arrays[f"cut_{m:03d}"][0] for m in range(coordinates.n_modes)]]
            return cls(
                coordinates,
                arrays["angles"],
                grids,
                cuts,
                arrays["base_energies"],
                pair_coefficients=arrays.get("pair_coefficients"),
                pair_cubic_coefficients=arrays.get("pair_cubic_coefficients"),
            )


def coordinate_map_type(settings):
    if settings.coordinate_construction == "direct":
        from .direct_coordinates import DirectInternalCoordinateMap

        return DirectInternalCoordinateMap
    raise ValueError("The paper workflow uses coordinate_construction: direct")


class InternalSampler:
    def __init__(self, generator):
        self.generator = generator
        self.config = generator.config
        self.settings = self.config.internal_coordinates
        self.rng = generator.numpy_rng
        self.scan_dir = generator.scan_dir / "internal_coordinates"
        self.output_dir = generator.output_dir
        self.log = generator._log
        self.rejected = {}

    def _write_job(self, name, geometry):
        path = self.scan_dir / (name + ".in")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = generate_qchem_input_text(
            geometry / ANGSTROM_TO_BOHR,
            self.generator.atom_symbols,
            charge=self.config.qchem.charge,
            multiplicity=self.config.qchem.multiplicity,
            template=self.config.qchem.input_template,
        )
        path.write_text(text)
        return path

    def _run_jobs(self, paths, name):
        filelist = self.scan_dir / (name + "_filelist.txt")
        filelist.write_text("".join(str(p.resolve()) + "\n" for p in paths))
        if self.config.qchem.run_jobs:
            run_qchem_filelist(
                filelist,
                processes=self.config.qchem.num_processors,
                command_template=self.config.qchem.command_template,
                skip_existing_outputs=self.config.qchem.skip_existing_outputs,
                skip_failed_jobs=False,
                verbose=self.generator.verbose,
            )
        energies = []
        for p in paths:
            out = p.with_suffix(".out")
            if not qchem_output_matches_input(p, out):
                raise FileNotFoundError(
                    f"Missing, stale or unverified internal scan output: {out}. Run the generated {filelist} or enable qchem.run_jobs."
                )
            energies.append(float(parse_total_energies_hartree(out)[0]))
        return np.array(energies)

    def build_potential(self):
        """Scan each mode, extending its two sides by calculated electronic energy."""
        from .scan_domains import build_energy_guided_scans

        grids, cuts, bases, self.scan_domain_report = build_energy_guided_scans(self)
        self.potential = InternalPotential(self.coordinates, self.angles, grids, cuts, bases)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.potential.save(self.output_dir / "internal_potential.npz")
        write_json(self.output_dir / "internal_coordinates.json", self.coordinates.to_dict())

    def validate_potential(self):
        count = self.settings.validation_points
        if count <= 0:
            return
        rng = np.random.default_rng(90817)
        c = self.coordinates
        sigma = 1 / np.sqrt(2 * np.maximum(c.frequencies_hartree, 1.0e-6))
        points = []
        paths = []
        predicted = []
        for i in range(count):
            for _ in range(100):
                phi = float(rng.uniform(0, TWOPI)) if c.has_rotor else 0.0
                q = rng.normal(size=c.n_modes) * sigma
                try:
                    x = c.geometry(phi, q)
                    e = self.potential.evaluate_geometry(x) + self.potential.reference_energy
                    break
                except CoordinateDomainError:
                    continue
            else:
                raise CoordinateDomainError(
                    "Could not construct a validation point inside the computed domain"
                )
            paths.append(self._write_job(f"validation/point_{i:04d}", x))
            points.append(x)
            predicted.append(e)
        actual = self._run_jobs(paths, "validation")
        error = np.asarray(predicted) - actual
        write_json(
            self.output_dir / "electronic_validation.json",
            dict(
                description="Independent deterministic multi-mode geometries; not training points",
                count=count,
                rmse_hartree=float(np.sqrt(np.mean(error**2))),
                bias_hartree=float(np.mean(error)),
                maximum_absolute_error_hartree=float(np.max(abs(error))),
                actual_hartree=actual,
                predicted_hartree=predicted,
                geometry_bohr=points,
            ),
        )
        self.log(
            f"Independent electronic validation RMSE: {np.sqrt(np.mean(error**2)) * 627.509474:.3f} kcal/mol"
        )

    def run(self):
        self.generator.process_relaxed_torsion()
        self.initialize_coordinates()
        self.build_potential()
        self.validate_potential()
        self.prepare_quantum_priors()
        result = self.sample()
        self.write_outputs(result)
        self.generator.result = result
        self.generator.internal_sampler = self
        return result

    def _boltzmann(self, energies):
        energies = np.asarray(energies)
        if self.kT <= 0:
            weights = np.isclose(energies, energies.min(), rtol=0, atol=1.0e-12).astype(float)
            return weights / weights.sum()
        values = -(energies - energies.min()) / self.kT
        return np.exp(values - logsumexp(values))
