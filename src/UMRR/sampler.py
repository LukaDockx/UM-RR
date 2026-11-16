"""Orchestrate the relaxed scan, minimum Hessians and UM-RR ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import InitialConditionConfig
from .momentum import (
    periodic_path_momentum_distributions,
)
from .qchem_io import (
    mass_matrix_electron_masses,
    parse_projected_mass_weighted_hessian,
    parse_standard_orientation_angstrom,
    parse_total_energies_hartree,
    vibrational_modes_from_hessian,
)
from .qchem_runner import (
    qchem_output_has_normal_termination,
    run_qchem_filelist,
)
from .relaxed_torsion import (
    build_relaxed_torsion_solution,
    group_equivalent_minima,
    relaxed_torsion_expected_kinetic_energies_hartree,
    relaxed_torsion_scan_paths,
    write_minimum_frequency_inputs,
    write_minimum_optimization_inputs,
    write_relaxed_torsion_scan_input,
)
from .rotors import (
    analyze_hindered_rotors,
    torsion_atom_indices_for_rotor,
)

HARTREE_TO_CM1 = 219474.63
MAX_FOURIER_KINETIC_RELATIVE_ERROR = 5.0e-2
MAX_PERIODIC_FD_KINETIC_RELATIVE_ERROR = 1.0e-7


@dataclass(slots=True)
class InitialConditionResult:
    atom_symbols: list[str]
    geometries_bohr: np.ndarray
    velocities_au: np.ndarray
    quantum_numbers: np.ndarray
    potential_energies_hartree: np.ndarray
    kinetic_energies_hartree: np.ndarray
    target_energies_hartree: np.ndarray
    selected_vibrational_energies_hartree: np.ndarray
    potential_references_hartree: np.ndarray
    coordinate_sampling_attempts: np.ndarray
    geometry_strings_bohr: list[str]
    velocity_strings_au: list[str]
    large_amplitude_samples: list[dict[str, object]] | None = None

    @property
    def num_samples(self) -> int:
        return int(self.geometries_bohr.shape[0])

    def to_dict(self) -> dict[str, object]:
        samples = []
        for index in range(self.num_samples):
            reference = self.potential_references_hartree[index]
            relative_potential = self.potential_energies_hartree[index] - reference
            relative_total = relative_potential + self.kinetic_energies_hartree[index]
            sample: dict[str, object] = {
                "index": index,
                "quantum_numbers": self.quantum_numbers[index],
                "Epot": self.potential_energies_hartree[index],
                "Ekin": self.kinetic_energies_hartree[index],
                "Etot": (
                    self.potential_energies_hartree[index] + self.kinetic_energies_hartree[index]
                ),
                "Etarget": self.target_energies_hartree[index],
                "Evib": self.selected_vibrational_energies_hartree[index],
                "Eref": self.potential_references_hartree[index],
                "Vvib_relative": relative_potential,
                "Kinternal": self.kinetic_energies_hartree[index],
                "Evib_selected": self.selected_vibrational_energies_hartree[index],
                "Eelectronic_reference": reference,
                "Etotal_relative": relative_total,
                "Epot_absolute": self.potential_energies_hartree[index],
                "Etotal_absolute": (
                    self.potential_energies_hartree[index] + self.kinetic_energies_hartree[index]
                ),
                "coordinate_sampling_attempts": self.coordinate_sampling_attempts[index],
                "geometry_bohr": self.geometries_bohr[index],
                "velocity_au": self.velocities_au[index],
            }
            if self.geometry_strings_bohr and self.velocity_strings_au:
                sample["geometry_sharc_bohr"] = self.geometry_strings_bohr[index]
                sample["velocity_sharc_au"] = self.velocity_strings_au[index]
            samples.append(sample)
            if self.large_amplitude_samples is not None:
                samples[-1]["large_amplitude_sample"] = self.large_amplitude_samples[index]
        return {
            "schema_version": 2,
            "energy_units": "hartree",
            "energy_schema": {
                "Vvib_relative": "separable vibrational potential above Eelectronic_reference",
                "Kinternal": "final rigid-body-free Cartesian kinetic energy",
                "Evib_selected": "sum of selected rotor and transverse vibrational eigenenergies",
                "Eelectronic_reference": "electronic reference energy for the separable potential",
                "Etotal_relative": "Vvib_relative + Kinternal; equals Evib_selected",
                "Epot_absolute": "Eelectronic_reference + Vvib_relative",
                "Etotal_absolute": "Eelectronic_reference + Etotal_relative",
                "legacy_aliases": {
                    "Epot": "Epot_absolute",
                    "Ekin": "Kinternal",
                    "Etot": "Etotal_absolute",
                    "Etarget": "Eelectronic_reference + Evib_selected",
                    "Evib": "Evib_selected",
                    "Eref": "Eelectronic_reference",
                },
            },
            "atoms": self.atom_symbols,
            "num_samples": self.num_samples,
            "samples": samples,
        }


class InitialConditionGenerator:
    """Generate neutral-PES initial conditions from Q-Chem normal-mode scans."""

    def __init__(self, config, *, base_dir=None, verbose=False):
        """Read the equilibrium Hessian; all subsequent energies use this method."""
        config.validate()
        self.config = config
        self.base_dir = (Path.cwd() if base_dir is None else Path(base_dir)).resolve()
        self.verbose = verbose
        self.numpy_rng = np.random.default_rng(config.sampling.random_seed)
        self.freq_output_path = self._resolve(config.paths.freq_output)
        self.scan_dir = self._resolve(config.paths.scan_dir)
        self.output_dir = self._resolve(config.paths.output_dir)
        if not qchem_output_has_normal_termination(self.freq_output_path):
            raise ValueError(
                f"Q-Chem frequency output did not terminate normally: {self.freq_output_path}"
            )
        self.equilibrium_geometry_angstrom, self.atom_symbols = parse_standard_orientation_angstrom(
            self.freq_output_path
        )
        self.electronic_reference_hartree = float(
            parse_total_energies_hartree(self.freq_output_path)[0]
        )
        self.num_atoms = len(self.atom_symbols)
        self.mass_matrix = mass_matrix_electron_masses(self.atom_symbols)
        self.hessian = parse_projected_mass_weighted_hessian(self.freq_output_path, self.num_atoms)
        _, self.normal_mode_eigenvectors, self.raw_frequencies_cm1 = vibrational_modes_from_hessian(
            self.hessian, self.equilibrium_geometry_angstrom, self.mass_matrix[:, 0]
        )
        self.relaxed_torsion_solution = None
        self.relaxed_torsion_unique_minima = []
        self.relaxed_torsion_basin_to_unique_minimum = {}
        self.result = None

    def run(self):
        """Build the relaxed path and minimum curves, then generate the ensemble."""
        from .basin_sampler import BasinSampler

        self._log("Starting UM-RR initial-condition generation")
        return BasinSampler(self).run()

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        base_dir: str | Path | None = None,
    ) -> "InitialConditionGenerator":
        path = Path(config_path).resolve()
        config = InitialConditionConfig.from_yaml(path)
        return cls(config, base_dir=path.parent if base_dir is None else base_dir)

    def process_relaxed_torsion(self) -> None:
        if not self.config.relaxed_torsion.enabled:
            return
        self.relaxed_torsion_atom_indices = self._relaxed_torsion_atom_indices()
        scan_input_path, scan_output_path = relaxed_torsion_scan_paths(
            self._resolve(self.config.relaxed_torsion.scan_dir),
            self.config.relaxed_torsion.scan_input,
            self.config.relaxed_torsion.scan_output,
        )
        if self.config.relaxed_torsion.write_scan_input:
            self._log(f"Writing relaxed torsion scan input: {scan_input_path}")
            write_relaxed_torsion_scan_input(
                scan_input_path,
                self.equilibrium_geometry_angstrom,
                self.atom_symbols,
                atom_indices=self.relaxed_torsion_atom_indices,
                charge=self.config.qchem.charge,
                multiplicity=self.config.qchem.multiplicity,
                qchem_template=self.config.qchem.input_template,
                start_deg=self.config.relaxed_torsion.scan_start_deg,
                end_deg=self.config.relaxed_torsion.scan_end_deg,
                step_deg=self.config.relaxed_torsion.scan_step_deg,
            )
        if self.config.relaxed_torsion.run_qchem:
            self._log("Running relaxed torsion Q-Chem scan")
            run_qchem_filelist(
                scan_input_path.parent / "filelist.txt",
                processes=1,
                command_template=(
                    self.config.relaxed_torsion.scan_command_template
                    or self.config.qchem.command_template
                ),
                skip_existing_outputs=self.config.relaxed_torsion.skip_existing_outputs,
                verbose=self.verbose,
            )
        if not scan_output_path.exists():
            raise FileNotFoundError(f"Relaxed torsion scan output is missing: {scan_output_path}")
        if not qchem_output_has_normal_termination(scan_output_path):
            raise ValueError(
                f"Relaxed torsion Q-Chem output did not terminate normally: {scan_output_path}"
            )
        self.relaxed_torsion_solution = build_relaxed_torsion_solution(
            scan_output_path,
            mass_matrix=self.mass_matrix,
            normal_mode_eigenvectors=self.normal_mode_eigenvectors,
            normal_mode_frequencies_cm1=self.raw_frequencies_cm1,
            frequency_cutoff_cm1=self.config.relaxed_torsion.frequency_cutoff_cm1,
            grid_points=self.config.schrodinger.grid_points,
            num_states=self.config.relaxed_torsion.num_states,
            atom_indices=self.relaxed_torsion_atom_indices,
            temperature_kelvin=self.config.sampling.temperature_kelvin,
            population_tail_tolerance=self.config.relaxed_torsion.population_tail_tolerance,
        )
        self.relaxed_torsion_expected_kinetic_hartree = (
            relaxed_torsion_expected_kinetic_energies_hartree(self.relaxed_torsion_solution)
        )
        self.relaxed_torsion_momentum_distributions = periodic_path_momentum_distributions(
            self.relaxed_torsion_solution.wavefunctions,
            self.relaxed_torsion_solution.total_path_length,
        )
        periodic_fd_kinetic = np.asarray(
            [
                distribution.finite_difference_kinetic_energy_hartree
                for distribution in self.relaxed_torsion_momentum_distributions
            ],
            dtype=float,
        )
        periodic_fd_relative_error = np.abs(
            periodic_fd_kinetic - self.relaxed_torsion_expected_kinetic_hartree
        ) / np.maximum(self.relaxed_torsion_expected_kinetic_hartree, 1.0e-14)
        if np.any(periodic_fd_relative_error > MAX_PERIODIC_FD_KINETIC_RELATIVE_ERROR):
            state = int(np.argmax(periodic_fd_relative_error))
            raise ValueError(
                "Periodic relaxed-torsion Fourier kinetic moment is inconsistent "
                "with epsilon-<V>: state "
                f"{state} has relative error "
                f"{periodic_fd_relative_error[state]:.3e}."
            )
        self._log(
            "Relaxed torsion path: "
            f"{len(self.relaxed_torsion_solution.scan_points)} optimized scan point(s), "
            f"{len(self.relaxed_torsion_solution.basins)} basin(s), "
            f"path length={self.relaxed_torsion_solution.total_path_length:.3f}"
        )
        if self.relaxed_torsion_solution.projected_mode_index is not None:
            self._log(
                "Relaxed torsion projects onto normal mode "
                f"{self.relaxed_torsion_solution.projected_mode_index} "
                f"(score={self.relaxed_torsion_solution.projection_score:.3f})"
            )
        (
            self.relaxed_torsion_unique_minima,
            self.relaxed_torsion_basin_to_unique_minimum,
        ) = group_equivalent_minima(
            self.relaxed_torsion_solution,
            rmsd_threshold_angstrom=self.config.relaxed_torsion.rmsd_threshold_angstrom,
        )
        self._log(
            f"Relaxed torsion unique minima: {len(self.relaxed_torsion_unique_minima)} "
            f"from {len(self.relaxed_torsion_solution.basins)} basin(s)"
        )

    def _relaxed_torsion_atom_indices(self) -> list[int]:
        atom_indices = [int(index) for index in self.config.relaxed_torsion.atom_indices]
        if atom_indices:
            if len(atom_indices) != 4:
                raise ValueError(
                    "relaxed_torsion.atom_indices must contain four 1-based atom indices"
                )
            self._log(
                "Using configured relaxed torsion atom indices: "
                + "-".join(str(index) for index in atom_indices)
            )
            return atom_indices

        analysis = analyze_hindered_rotors(
            self.equilibrium_geometry_angstrom,
            self.atom_symbols,
            self.normal_mode_eigenvectors,
            self.raw_frequencies_cm1,
            frequency_cutoff_cm1=self.config.relaxed_torsion.frequency_cutoff_cm1,
            projection_threshold=self.config.relaxed_torsion.projection_threshold,
            bond_scale=self.config.relaxed_torsion.bond_scale,
            bond_tolerance_angstrom=self.config.relaxed_torsion.bond_tolerance_angstrom,
        )
        if len(analysis.rotors) != 1:
            raise ValueError(
                "Automatic relaxed torsion selection requires exactly one accepted "
                f"rotor, but detected {len(analysis.rotors)}. Set "
                "relaxed_torsion.atom_indices explicitly."
            )
        rotor = analysis.rotors[0]
        detected = torsion_atom_indices_for_rotor(
            rotor,
            self.atom_symbols,
            self.equilibrium_geometry_angstrom,
            analysis.bond_graph,
        )
        self._log(
            "Auto-selected relaxed torsion "
            + "-".join(str(index) for index in detected)
            + f" from {rotor.label}; mode={rotor.projected_mode_index}, "
            + f"score={rotor.projection_score:.3f}"
        )
        return detected

    def prepare_relaxed_torsion_minimum_jobs(self) -> None:
        if self.relaxed_torsion_solution is None:
            return
        if (
            self.config.relaxed_torsion.run_minimum_jobs
            and not self.config.relaxed_torsion.write_minimum_inputs
        ):
            raise ValueError(
                "relaxed_torsion.run_minimum_jobs requires "
                "relaxed_torsion.write_minimum_inputs: true"
            )
        minimum_dir = self._resolve(self.config.relaxed_torsion.minimum_dir)
        if self.config.relaxed_torsion.write_minimum_inputs:
            opt_filelist = write_minimum_optimization_inputs(
                minimum_dir,
                self.relaxed_torsion_unique_minima,
                self.atom_symbols,
                charge=self.config.qchem.charge,
                multiplicity=self.config.qchem.multiplicity,
                qchem_template=self.config.qchem.input_template,
            )
            self._log(f"Wrote relaxed torsion minimum optimization inputs: {opt_filelist}")
            if self.config.relaxed_torsion.run_minimum_jobs:
                run_qchem_filelist(
                    opt_filelist,
                    processes=self.config.qchem.num_processors,
                    command_template=self.config.qchem.command_template,
                    skip_existing_outputs=self.config.relaxed_torsion.skip_existing_outputs,
                    verbose=self.verbose,
                )
            freq_filelist = write_minimum_frequency_inputs(
                minimum_dir,
                self.relaxed_torsion_unique_minima,
                self.atom_symbols,
                charge=self.config.qchem.charge,
                multiplicity=self.config.qchem.multiplicity,
                qchem_template=self.config.qchem.input_template,
            )
            if freq_filelist is not None:
                self._log(f"Wrote relaxed torsion minimum frequency inputs: {freq_filelist}")
                if self.config.relaxed_torsion.run_minimum_jobs:
                    run_qchem_filelist(
                        freq_filelist,
                        processes=self.config.qchem.num_processors,
                        command_template=self.config.qchem.command_template,
                        skip_existing_outputs=self.config.relaxed_torsion.skip_existing_outputs,
                        verbose=self.verbose,
                    )

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[UMRR] {message}", flush=True)
