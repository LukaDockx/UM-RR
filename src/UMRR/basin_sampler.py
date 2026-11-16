"""Minimum-specific transverse potentials and fixed-energy UM-RR ensembles.

Each basin uses its own fixed internal-coordinate mode combinations and
one-dimensional electronic curves. Torsional barriers assign the basin.
Equivalent minima reuse the entire local potential through atom relabeling.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from .constants import (
    ANGSTROM_TO_BOHR,
    BOLTZMANN_EV_PER_K,
    ELECTRON_MASS_TO_AMU,
    HARTREE_TO_EV,
)
from .direct_coordinates import DirectBasinChart as BasinChart
from .internal_coordinates import (
    TWOPI,
    CoordinateDomainError,
    InternalCoordinateMap,
)
from .internal_sampler import (
    InternalPotential,
    InternalSampler,
    cdf_from_density,
    coordinate_map_type,
    periodic_spline,
)
from .momentum import (
    periodic_path_momentum_distributions,
)
from .qchem_io import (
    parse_projected_mass_weighted_hessian,
    parse_standard_orientation_angstrom,
)
from .qchem_runner import file_sha256
from .relaxed_torsion import (
    interpolate_geometry_from_path,
)
from .rotor_states import solve_thermal_rotor_states
from .schrodinger import solve_schrodinger_1d
from .serialization import write_json
from .state_boundaries import prepare_transverse_state_distributions


class SharedPath:
    """One actual Cartesian reference geometry for every local chart at an angle."""

    def __init__(self, coordinates):
        self.coordinates = coordinates

    @lru_cache(maxsize=4096)
    def at(self, angle):
        """Return the relaxed geometry and its rigid-body-free path tangent.

        J dx/dphi=e_phi fixes the internal-coordinate change. The tangent
        M^-1 J^T g e_phi has the smallest mass-weighted Cartesian norm and
        therefore excludes translation and whole-molecule rotation.
        """
        c = self.coordinates
        x = c.geometry(angle, np.zeros(c.n_modes))
        if c.has_rotor:
            j = c.jacobian(x)
            dx = ((j.T @ c.mass_metric(x)[:, 0]) / c.sqrt_masses**2).reshape(x.shape)
        else:
            dx = np.zeros_like(x)
        return x, dx


# A basin retains its minimum's normal-mode combinations while taking its
# undisplaced geometry from the common relaxed path.


class BasinPotential:
    """V_rot(phi) plus the independent transverse curves of the assigned basin.

    Energies returned by evaluate_geometry are relative to reference_energy,
    the lowest electronic scan energy. Transverse curves use U_i(0)=0, so
    adding them does not count a conformer's electronic energy twice.
    """

    def __init__(
        self,
        path_coordinates,
        local_coordinates,
        models,
        basins,
        path_angles,
        path_energies,
        *,
        interpolation="hard",
        complete_symmetry=False,
    ):
        if interpolation != "hard":
            raise ValueError("The paper workflow uses hard basin assignment")
        self.coordinates = path_coordinates
        self.local_coordinates = local_coordinates
        self.models = models
        self.basins = sorted(basins, key=lambda b: b["left_angle"])
        self.interpolation = interpolation
        self.shared_path = SharedPath(path_coordinates)
        self.charts = [
            BasinChart(local_coordinates[b["unique_index"]], self.shared_path, b["permutation"])
            for b in self.basins
        ]
        self.reference_energy = float(np.min(path_energies))
        self.path_angles = np.asarray(path_angles)
        self.path_energies = np.asarray(path_energies)
        self.path_potential = periodic_spline(
            path_angles, self.path_energies - self.reference_energy
        )
        self.boundaries = np.array([b["left_angle"] for b in self.basins])
        self.complete_symmetry = bool(complete_symmetry)
        self.symmetry_copies = self._make_symmetry_copies() if complete_symmetry else []

    def _make_symmetry_copies(self):
        """Reuse a minimum only through element- and mass-preserving relabeling."""
        identity = list(range(len(self.coordinates.symbols)))
        copies = [None] * len(self.basins)
        for unique in range(len(self.models)):
            indices = [i for i, b in enumerate(self.basins) if b["unique_index"] == unique]
            representatives = [
                i for i in indices if list(self.basins[i]["permutation"]) == identity
            ]
            if len(representatives) != 1:
                raise ValueError("Each distinct minimum needs exactly one identity permutation")
            rep = representatives[0]
            group = [np.asarray(self.basins[i]["permutation"], int) for i in indices]
            permutations = {tuple(g) for g in group}
            if len(permutations) != len(group) or any(
                tuple(g[h]) not in permutations for g in group for h in group
            ):
                raise ValueError("Equivalent-minimum permutations must form a closed group")
            for index, g in zip(indices, group):
                if sorted(g.tolist()) != identity or not (
                    np.array_equal(
                        np.asarray(self.coordinates.symbols)[g],
                        self.coordinates.symbols,
                    )
                    and np.array_equal(self.coordinates.masses[g], self.coordinates.masses)
                ):
                    raise ValueError("Equivalent-minimum permutation changes elements or masses")
                copies[index] = (rep, np.argsort(g))
        return copies

    def weights(self, angle, *, interpolation=None):
        # These weights select proposal minima. Actual energy evaluation below
        # relabels the geometry before evaluating each complete local function.
        if not self.complete_symmetry:
            return self._partition_weights(angle, interpolation=interpolation)
        x = self.shared_path.at(float(angle % TWOPI))[0]
        return self._copy_terms(x, interpolation=interpolation)[0]

    def _partition_weights(self, angle, *, interpolation=None):
        """Assign the interval between neighboring torsional barriers."""
        if interpolation not in (None, "hard"):
            raise ValueError("The paper workflow uses hard basin assignment")
        weights = np.zeros(len(self.basins))
        owner = (np.searchsorted(self.boundaries, float(angle) % TWOPI, side="right") - 1) % len(
            self.basins
        )
        weights[owner] = 1.0
        return weights

    def _copy_terms(self, geometry, *, interpolation=None):
        angles = []
        support = []
        for rep, inverse in self.symmetry_copies:
            y = np.asarray(geometry)[inverse]
            phi = (
                float(self.coordinates.rotor.value(y) % TWOPI)
                if self.coordinates.has_rotor
                else 0.0
            )
            angles.append(phi)
            support.append(self._partition_weights(phi, interpolation=interpolation)[rep])
        support = np.asarray(support)
        if support.sum() <= 0:
            raise CoordinateDomainError(
                "No symmetry-related minimum supports this geometry in its assigned basin"
            )
        return support / support.sum(), angles

    def reference_path_energy(self, angle):
        """Use the accepted-geometry energy on the path in the rotor Hamiltonian."""
        return self.evaluate_geometry(self.shared_path.at(float(angle % TWOPI))[0])

    def local_energy(self, index, geometry):
        b = self.basins[index]
        _, q = self.charts[index].coordinates(geometry)
        model = self.models[b["unique_index"]]
        # The one-angle model's electronic zero is subtracted: Vpath carries
        # every conformer's absolute energy and the relaxed rotor barrier.
        return model.raw(model.angles[0], q)

    def evaluate_geometry(self, geometry_bohr):
        """Return the rotor potential plus transverse curves in the assigned basin."""
        geometry = np.asarray(geometry_bohr)
        if self.complete_symmetry:
            weights, angles = self._copy_terms(geometry)
            energy = 0.0
            for weight, angle, (representative, inverse) in zip(
                weights, angles, self.symmetry_copies
            ):
                if weight > 0:
                    # Relabel the complete local energy, including the path
                    # reference, to reuse an equivalent minimum consistently.
                    local = self.path_potential(angle) + self.local_energy(
                        representative, geometry[inverse]
                    )
                    energy += float(weight) * float(local)
            return energy
        angle = (
            float(self.coordinates.rotor.value(geometry) % TWOPI)
            if self.coordinates.has_rotor
            else 0.0
        )
        owner = int(np.argmax(self.weights(angle)))
        return float(self.path_potential(angle)) + self.local_energy(owner, geometry)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(directory / f"minimum_{i:03d}.npz")
        write_json(
            directory / "basin_model.json",
            dict(
                schema_version=4,
                complete_symmetry=self.complete_symmetry,
                coordinate_representation=(
                    "direct independent internals; complete minimum potentials preserved"
                    if getattr(self.coordinates, "definition", "")
                    in {"direct_independent_internal_v2"}
                    else "fixed_global_primitives; full minimum potentials preserved"
                ),
                path_coordinates=self.coordinates.to_dict(),
                local_coordinates=[c.to_dict() for c in self.local_coordinates],
                coordinate_charts=(
                    [c.to_dict() for c in self.charts]
                    if getattr(self.coordinates, "definition", "")
                    in {"direct_independent_internal_v2"}
                    else None
                ),
                basins=self.basins,
                path_angles=self.path_angles,
                path_energies=self.path_energies,
                interpolation=self.interpolation,
            ),
        )

    @classmethod
    def load(cls, directory):
        """Restore the path, minimum curves and exact coordinate definitions."""
        directory = Path(directory)
        record = json.loads((directory / "basin_model.json").read_text())
        path = InternalCoordinateMap.from_dict(record.pop("path_coordinates"))
        locals_ = [InternalCoordinateMap.from_dict(c) for c in record.pop("local_coordinates")]
        models = [
            InternalPotential.load(directory / f"minimum_{i:03d}.npz", c)
            for i, c in enumerate(locals_)
        ]
        if record.pop("schema_version") != 4:
            raise ValueError("The paper branch loads basin model schema 4")
        record.pop("coordinate_representation", None)
        record.pop("common_primitives", None)
        if record.pop("barrier_corrections", []):
            raise ValueError("The paper potential uses transverse curves at minima")
        record.pop("smoothing_fraction", None)
        charts = record.pop("coordinate_charts", None)
        potential = cls(path, locals_, models, **record)
        if charts is not None:
            if len(charts) != len(potential.charts):
                raise ValueError("Saved basin charts do not match basin count")
            potential.charts = [BasinChart.from_dict(chart) for chart in charts]
            for chart in potential.charts:
                chart.shared_path = potential.shared_path
        return potential


class BasinSampler(InternalSampler):
    """Build local quantum states and draw fixed-energy initial conditions."""

    def initialize_coordinates(self):
        from .direct_coordinates import DirectInternalCoordinateMap

        g, cfg = self.generator, self.config
        solution = g.relaxed_torsion_solution
        has_rotor = solution is not None and cfg.relaxed_torsion.sample
        self.coordinates = DirectInternalCoordinateMap(
            g.atom_symbols,
            g.mass_matrix[:, 0],
            g.equilibrium_geometry_angstrom * ANGSTROM_TO_BOHR,
            g.hessian * ELECTRON_MASS_TO_AMU,
            rotor_indices=np.asarray(solution.atom_indices) - 1 if has_rotor else None,
            path_geometries_bohr=solution.geometries_angstrom * ANGSTROM_TO_BOHR
            if has_rotor
            else None,
            tolerance=self.settings.map_tolerance,
            max_iterations=self.settings.map_max_iterations,
        )
        self.path_coordinates = self.coordinates
        g, cfg = self.generator, self.config
        self.scan_dir = g.scan_dir / "basin_internal"
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_coordinates = []
        self.basin_records = []
        solution = g.relaxed_torsion_solution
        if not self.coordinates.has_rotor:
            self.local_coordinates = [self.coordinates]
            self.basin_records = [
                dict(
                    unique_index=0,
                    minimum_angle=0.0,
                    left_angle=0.0,
                    permutation=list(range(len(g.atom_symbols))),
                )
            ]
            self.path_angles = np.array([0.0])
            self.path_energies = np.array([g.electronic_reference_hartree])
            return
        # Locate each optimized minimum's Hessian before defining its own
        # fixed-dihedral transverse normal modes.
        g.prepare_relaxed_torsion_minimum_jobs()
        from .relaxed_torsion import minimum_frequency_output_path

        for minimum in g.relaxed_torsion_unique_minima:
            freq = minimum_frequency_output_path(
                g._resolve(cfg.relaxed_torsion.minimum_dir), minimum.index
            )
            if not freq.exists():
                raise FileNotFoundError(
                    f"Missing minimum frequency output {freq}; enable relaxed_torsion.run_minimum_jobs"
                )
            xyz, symbols = parse_standard_orientation_angstrom(freq)
            if symbols != list(g.atom_symbols):
                raise ValueError(f"Atom order differs in {freq}")
            h = parse_projected_mass_weighted_hessian(freq, len(symbols)) * ELECTRON_MASS_TO_AMU
            self.local_coordinates.append(
                coordinate_map_type(self.settings)(
                    symbols,
                    self.coordinates.masses,
                    xyz * ANGSTROM_TO_BOHR,
                    h,
                    rotor_indices=self.coordinates.rotor_indices,
                    rotor_permutations=self.coordinates.rotor_permutations,
                    rotor_phases=self.coordinates.rotor_phases,
                    path_geometries_bohr=solution.geometries_angstrom * ANGSTROM_TO_BOHR,
                    tolerance=self.settings.map_tolerance,
                    max_iterations=self.settings.map_max_iterations,
                )
            )

        # Angular order can be the reverse of the mass-weighted scan order.
        # Assign each minimum to its enclosing pair of actual barrier angles.
        def angle_at(s):
            x = interpolate_geometry_from_path(solution, s) * ANGSTROM_TO_BOHR
            return float(self.coordinates.rotor.value(x) % TWOPI)

        boundaries = sorted(angle_at(b.left_boundary_path_coordinate) for b in solution.basins)
        for b in solution.basins:
            phi = angle_at(b.minimum_path_coordinate)
            left = boundaries[
                (np.searchsorted(boundaries, phi, side="right") - 1) % len(boundaries)
            ]
            unique = g.relaxed_torsion_basin_to_unique_minimum[b.index]
            m = g.relaxed_torsion_unique_minima[unique]
            self.basin_records.append(
                dict(
                    unique_index=unique,
                    original_basin_index=b.index,
                    minimum_angle=phi,
                    left_angle=left,
                    permutation=m.basin_permutations[b.index],
                )
            )
        order = self.coordinates.path_order
        geometries = solution.geometries_angstrom[order] * ANGSTROM_TO_BOHR
        self.path_angles = np.array([self.coordinates.rotor.value(x) % TWOPI for x in geometries])
        # solution.scan_points and geometries share ordering after duplicate merge.
        self.path_energies = np.array([p.energy_hartree for p in solution.scan_points])[order]
        self.log(
            f"Minima-only model: {len(self.local_coordinates)} distinct minima, {len(self.basin_records)} basins; "
            "independent transverse curves"
        )

    def build_potential(self):
        models = []
        self.scan_count = 0
        self.scan_counts = dict(one_mode=0, pair_corners=0, pair_axes=0)
        shared_path = SharedPath(self.coordinates)
        for i, c in enumerate(self.local_coordinates):
            cfg = deepcopy(self.config)
            points = cfg.pes.num_scan_points
            cfg.internal_coordinates.scan_points = points + int(points % 2 == 0)
            cfg.internal_coordinates.scan_energy_hartree = cfg.pes.scan_energy_range_hartree
            child = InternalSampler(self.generator)
            child.config = cfg
            child.settings = cfg.internal_coordinates
            representative = next(
                b
                for b in self.basin_records
                if b["unique_index"] == i and b["permutation"] == list(range(len(c.symbols)))
            )
            chart = BasinChart(c, shared_path, representative["permutation"])
            child.coordinates = chart
            phi = representative["minimum_angle"]
            child.angles = np.array([phi])
            child.base_geometries = np.array([chart.geometry(phi, np.zeros(c.n_modes))])
            child.scan_dir = self.scan_dir / f"minimum_{i:03d}"
            child.output_dir = self.output_dir / f"minimum_{i:03d}"
            child.symmetry_permutations = [np.arange(len(c.symbols))]
            child.build_potential()
            models.append(child.potential)
            self.scan_counts["one_mode"] += child.scan_domain_report["electronic_points"]
        self.scan_count = sum(self.scan_counts.values())
        self.potential = BasinPotential(
            self.coordinates,
            self.local_coordinates,
            models,
            self.basin_records,
            self.path_angles,
            self.path_energies,
            interpolation=self.settings.basin_interpolation,
            complete_symmetry=True,
        )
        self.potential.save(self.output_dir)
        self.log(f"Minima-only electronic scan count: {self.scan_count} (reference points shared)")

    def prepare_quantum_priors(self):
        """Solve transverse equations with reference masses and the path rotor.

        Each minimum retains its own transverse levels and probability
        densities. The periodic rotor uses the electronic relaxed-path
        potential and the mass-weighted path length.
        """
        self.kT = BOLTZMANN_EV_PER_K / HARTREE_TO_EV * self.config.sampling.temperature_kelvin
        self.priors = []
        for i, (c, model) in enumerate(zip(self.local_coordinates, self.potential.models)):
            phi = model.angles[0]
            b = next(
                k
                for k, r in enumerate(self.potential.basins)
                if r["unique_index"] == i and r["permutation"] == list(range(len(c.symbols)))
            )
            chart = self.potential.charts[b]
            x = chart.geometry(phi, np.zeros(c.n_modes))
            masses = np.diag(chart.mass_metric(x))[int(c.has_rotor) :]
            local = []
            for m, mass in enumerate(masses):
                v = model.one_mode(phi, m)
                eps, psi, grid = solve_schrodinger_1d(
                    v,
                    model.grids[m],
                    mass,
                    grid_points=self.config.schrodinger.grid_points,
                    num_states=self.config.schrodinger.num_states,
                )
                limit = min(v(grid[0]), v(grid[-1]))
                n, states, boundary = prepare_transverse_state_distributions(
                    eps,
                    psi,
                    grid,
                    float(mass),
                    limit,
                    self.kT,
                    self.settings.boundary_probability_tolerance,
                    self.log,
                    f"Minimum {i + 1}, vibration {m + 1}",
                )
                local.append(
                    dict(
                        energies=eps[:n],
                        wavefunctions=psi[:n],
                        boundary_diagnostics=boundary,
                        grid=grid,
                        mass=mass,
                        momentum=states,
                        cdfs=np.array([cdf_from_density(grid, p * p) for p in psi[:n]]),
                    )
                )
            self.priors.append(local)
        c = self.coordinates
        if c.has_rotor:
            angular_grid = np.linspace(0, TWOPI, 721)
            mass = [
                c.mass_metric(self.potential.shared_path.at(float(phi))[0])[0, 0]
                for phi in angular_grid
            ]
            self.ds_dphi = PchipInterpolator(angular_grid, np.sqrt(mass))
            s = np.r_[0.0, cumulative_trapezoid(np.sqrt(mass), angular_grid)]
            self.total_length = float(s[-1])
            self.s_to_phi = PchipInterpolator(s, angular_grid)
            self.rotor_s = np.linspace(
                0,
                self.total_length,
                self.config.schrodinger.grid_points,
                endpoint=False,
            )
            angles = self.s_to_phi(self.rotor_s)
            # Use the same electronic energy reference as geometry acceptance.
            # Transverse levels enter the sampled budget once, below; they do
            # not alter this rotor Hamiltonian or create basin-boundary shifts.
            self.effective_rotor_potential = np.array(
                [self.potential.reference_path_energy(a) for a in angles]
            )
            (
                self.rotor_energies,
                self.rotor_wavefunctions,
                self.rotor_state_convergence,
            ) = solve_thermal_rotor_states(
                self.effective_rotor_potential,
                self.total_length,
                num_states=self.config.relaxed_torsion.num_states,
                temperature_kelvin=self.config.sampling.temperature_kelvin,
                population_tail_tolerance=self.config.relaxed_torsion.population_tail_tolerance,
            )
            self.log(
                f"Rotor states: retained {len(self.rotor_energies)}; omitted population "
                f"bound {self.rotor_state_convergence['omitted_population_upper_bound']:.3%}"
            )
            self.rotor_momenta = periodic_path_momentum_distributions(
                self.rotor_wavefunctions, self.total_length
            )
            self.rotor_cdf_grid = np.r_[self.rotor_s, self.total_length]
            self.rotor_cdfs = np.array(
                [
                    cdf_from_density(self.rotor_cdf_grid, np.r_[p * p, p[0] ** 2])
                    for p in self.rotor_wavefunctions
                ]
            )
        else:
            self.total_length = 0.0
            self.rotor_energies = np.array([0.0])
        self.rotor_probabilities = self._boltzmann(self.rotor_energies)
        from .neutral_plots import save_neutral_states

        save_neutral_states(self)

    def sample(self):
        """Draw states, positions and momenta, then apply the fixed energy constraint.

        Every rejection repeats the state selection as well as the coordinate
        and momentum draws. An accepted geometry is retained while one scalar
        multiplies all Cartesian velocities to supply E_target-V_model.
        """
        from .qchem_io import (
            geometry_to_sharc_string_bohr,
            velocities_to_sharc_string_atomic_units,
        )
        from .sampler import InitialConditionResult

        pot = self.potential
        geometries, velocities, quantum, pots, kins, targets, attempts, records = (
            [] for _ in range(8)
        )
        for number in range(self.config.sampling.num_samples):
            for attempt in range(1, self.config.sampling.max_coordinate_sampling_attempts + 1):
                state = int(self.rng.choice(len(self.rotor_energies), p=self.rotor_probabilities))
                if self.coordinates.has_rotor:
                    s = np.interp(self.rng.random(), self.rotor_cdfs[state], self.rotor_cdf_grid)
                    phi = float(self.s_to_phi(s))
                    rotor_p = self.rotor_momenta[state].sample(self.rng) * float(self.ds_dphi(phi))
                else:
                    phi, rotor_p = 0.0, 0.0
                basin = int(self.rng.choice(len(pot.basins), p=pot.weights(phi)))
                unique = pot.basins[basin]["unique_index"]
                chart = pot.charts[basin]
                local = self.priors[unique]
                ns = [
                    int(self.rng.choice(len(a["energies"]), p=self._boltzmann(a["energies"])))
                    for a in local
                ]
                q = [
                    float(np.interp(self.rng.random(), a["cdfs"][n], a["grid"]))
                    for a, n in zip(local, ns)
                ]
                p = [
                    a["momentum"][n].sample(self.rng) * np.sqrt(a["mass"])
                    for a, n in zip(local, ns)
                ]
                # The rotor and transverse eigenenergies use the same energy
                # zero as the model potential. Include each selected level once.
                target = float(
                    self.rotor_energies[state] + sum(a["energies"][n] for a, n in zip(local, ns))
                )
                try:
                    x = chart.geometry(phi, q)
                    energy = pot.evaluate_geometry(x)
                except CoordinateDomainError as exc:
                    self.rejected["coordinate_domain"] = (
                        self.rejected.get("coordinate_domain", 0) + 1
                    )
                    cause = "coordinate_domain:" + exc.reason
                    self.rejected[cause] = self.rejected.get(cause, 0) + 1
                    continue
                kinetic = target - energy
                if kinetic < 0:
                    self.rejected["energy_shell"] = self.rejected.get("energy_shell", 0) + 1
                    continue
                # J^T pulls the internal momenta into Cartesian momenta.
                # Internal gradients give zero net translation and angular
                # momentum; uniform velocity scaling preserves both.
                momentum = np.r_[rotor_p, p] if chart.has_rotor else np.array(p)
                v = chart.velocity(x, momentum)
                raw = 0.5 * float(np.sum(chart.masses[:, None] * v * v))
                if raw <= 1.0e-30 and kinetic > 1.0e-14:
                    continue
                scale = np.sqrt(kinetic / raw) if raw > 0 else 0.0
                v *= scale
                actual_phi, actual_q = chart.coordinates(x)
                records.append(
                    dict(
                        type="basin_internal",
                        basin_index=basin,
                        unique_minimum_index=unique,
                        angle_rad=actual_phi,
                        angle_deg=float(np.degrees(actual_phi)),
                        amplitudes=actual_q,
                        proposal_amplitudes=q,
                        proposal_quantum_numbers=ns,
                        rotor_state=state,
                        velocity_scale=float(scale),
                        conditional_energy_budget_hartree=target,
                        coordinate_sampling_attempts=attempt,
                    )
                )
                geometries.append(x)
                velocities.append(v)
                quantum.append(ns)
                pots.append(energy + pot.reference_energy)
                kins.append(0.5 * float(np.sum(chart.masses[:, None] * v * v)))
                targets.append(target)
                attempts.append(attempt)
                break
            else:
                raise RuntimeError("Basin sampler exhausted the allowed proposal attempts")
            if number == 0 or (number + 1) % 1000 == 0:
                self.log(
                    f"Sampled {number + 1}/{self.config.sampling.num_samples} basin initial conditions"
                )
        n = len(geometries)
        symbols = self.coordinates.symbols
        gs = (
            [geometry_to_sharc_string_bohr(x, symbols) for x in geometries]
            if self.config.output.write_sharc_strings
            else []
        )
        vs = (
            [velocities_to_sharc_string_atomic_units(v) for v in velocities]
            if self.config.output.write_sharc_strings
            else []
        )
        return InitialConditionResult(
            symbols,
            np.array(geometries),
            np.array(velocities),
            np.array(quantum),
            np.array(pots),
            np.array(kins),
            pot.reference_energy + np.array(targets),
            np.array(targets),
            np.full(n, pot.reference_energy),
            np.array(attempts),
            gs,
            vs,
            records,
        )

    def write_outputs(self, result):
        data = result.to_dict()
        data.update(
            schema_version=4,
            coordinate_system="basin_internal",
            quantum_numbers_definition="Local proposal states of the recorded unique minimum; labels in different minima are not identified",
        )
        data["energy_schema"].update(
            Vvib_relative="Rotor potential plus independent transverse curves in the selected basin",
            Evib_selected="Electronic-path rotor energy plus selected transverse levels; not a coupled molecular eigenenergy",
            rotor_transverse_free_energy_included=False,
        )
        write_json(self.output_dir / self.config.output.initial_conditions_json, data)
        write_json(
            self.output_dir / self.config.output.metadata_json,
            dict(
                config=self.config.to_dict(),
                electronic_scan_count=self.scan_count,
                electronic_scan_counts=self.scan_counts,
                distinct_minima=len(self.local_coordinates),
                basin_count=len(self.potential.basins),
                rejected_proposals=self.rejected,
                sampling_policy=dict(
                    geometry=(
                        "Direct independent atom placement about one physical relaxed path; each minimum retains its own modes"
                    ),
                    boundaries=self.settings.basin_interpolation,
                    quantum=("Independent 1D minimum states; hard angular basin limits"),
                    rotor="1D path Hamiltonian uses only the electronic model potential evaluated on the reference path",
                    rotor_transverse_free_energy_included=False,
                    energy_budget="Electronic-path rotor energy + selected transverse levels of the chosen minimum",
                    momenta="Local canonical Fourier proposals, full chart Jacobian, then one scalar energy scaling",
                    rejection="Whole proposal redrawn; accepted state populations and momentum distributions can differ from proposals",
                    symmetry="Equivalent minima reuse complete local energy functions, including path references and angular supports, by atom permutation; distinct minima retain separate modes and splines",
                ),
                state_counts=[[len(a["energies"]) for a in local] for local in self.priors],
                last_state_probabilities=[
                    [float(self._boltzmann(a["energies"])[-1]) for a in local]
                    for local in self.priors
                ],
                rotor_last_state_probability=float(self.rotor_probabilities[-1]),
                rotor_state_convergence=getattr(self, "rotor_state_convergence", None),
                implementation_sha256={
                    name: file_sha256(Path(__file__).with_name(name))
                    for name in [
                        "basin_sampler.py",
                        "internal_coordinates.py",
                        "internal_sampler.py",
                        "direct_coordinates.py",
                        "direct_internal.py",
                        "scan_domains.py",
                        "state_boundaries.py",
                        "neutral_plots.py",
                    ]
                },
                model_sha256=file_sha256(self.output_dir / "basin_model.json"),
            ),
        )
        write_json(
            self.output_dir / self.config.output.vibrational_states_json,
            dict(
                minimum_energies_hartree=[[a["energies"] for a in local] for local in self.priors],
                rotor_energies_hartree=self.rotor_energies,
                rotor_transverse_free_energy_included=False,
                rotor_path_length=self.total_length,
                effective_rotor_potential_hartree=getattr(self, "effective_rotor_potential", None),
                rotor_path_grid=getattr(self, "rotor_s", None),
            ),
        )
