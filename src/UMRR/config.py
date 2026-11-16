"""Read the numerical and electronic-structure settings for UM-RR.

Paths are relative to the YAML file. Atomic units are used internally;
configuration angles and temperatures use degrees and kelvin respectively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

DEFAULT_QCHEM_TEMPLATE = """$molecule
{charge} {multiplicity}
{geometry_angstrom}
$end

$rem
METHOD RIMP2
BASIS aug-cc-pVDZ
AUX_BASIS RIMP2-aug-cc-pVDZ
N_FROZEN_CORE fc
SCF_CONVERGENCE 9
THRESH 14
MAX_SCF_CYCLES 300
SCF_ALGORITHM diis_gdm
VIBMAN_PRINT 4
GEOM_OPT_MAX_CYCLES 200
MEM_TOTAL 3000
MEM_STATIC 256
SYM_IGNORE true
SYMMETRY false
$end
"""


@dataclass(slots=True)
class PathConfig:
    freq_output: str = "freq.out"
    scan_dir: str = "qchem_scan"
    output_dir: str = "generated"
    filelist_name: str = "filelist.txt"


@dataclass(slots=True)
class SamplingConfig:
    num_samples: int = 10000
    temperature_kelvin: float = 298.0
    random_seed: int | None = None
    max_coordinate_sampling_attempts: int = 100000
    phase_space_method: str = "collective_anharmonic"
    coordinate_system: str = "internal"


@dataclass(slots=True)
class InternalCoordinatesConfig:
    """Reference-coordinate definitions, scan domains and basin assignment."""

    model: str = "minima"
    coordinate_construction: str = "direct"
    basin_interpolation: str = "hard"
    scan_extension_factor: float = 1.35
    scan_max_extensions: int = 8
    scan_max_harmonic_energy_factor: float = 4.0
    scan_nonrotor_dihedral_limit_deg: float = 150.0
    boundary_probability_tolerance: float = 0.001
    map_tolerance: float = 1.0e-8
    map_max_iterations: int = 80
    validation_points: int = 0
    # The minimum workflow derives these two values from the public pes section
    # before constructing each set of curves.
    scan_points: int = 25
    scan_energy_hartree: float = 0.3


@dataclass(slots=True)
class PESConfig:
    scan_energy_range_hartree: float = 0.3
    num_scan_points: int = 25


@dataclass(slots=True)
class SchrodingerConfig:
    grid_points: int = 5000
    num_states: int = 20


@dataclass(slots=True)
class RelaxedTorsionConfig:
    enabled: bool = False
    atom_indices: list[int] = field(default_factory=list)
    frequency_cutoff_cm1: float = 500.0
    projection_threshold: float = 0.35
    bond_scale: float = 1.25
    bond_tolerance_angstrom: float = 0.10
    scan_dir: str = "relaxed_rotor_scan"
    scan_input: str = "rotor_opt.in"
    scan_output: str = "rotor_opt.out"
    scan_start_deg: float = 180.0
    scan_end_deg: float = -180.0
    scan_step_deg: float = 10.0
    write_scan_input: bool = True
    run_qchem: bool = False
    # The initial scan is sequential and can use the full core allocation.
    scan_command_template: str | None = None
    skip_existing_outputs: bool = True
    num_states: int = 20
    population_tail_tolerance: float = 0.01
    sample: bool = False
    rmsd_threshold_angstrom: float = 0.15
    minimum_dir: str = "relaxed_torsion_minima"
    write_minimum_inputs: bool = True
    run_minimum_jobs: bool = False


@dataclass(slots=True)
class QChemConfig:
    charge: int = 0
    multiplicity: int = 1
    # Number of concurrent single-point jobs. With -nt 1, this is also the
    # total core allocation for a transverse scan.
    num_processors: int = 8
    run_jobs: bool = False
    skip_existing_outputs: bool = True
    skip_failed_jobs: bool = False
    command_template: str = "qchem -nt 1 {input} {output}"
    input_template: str = DEFAULT_QCHEM_TEMPLATE


@dataclass(slots=True)
class OutputConfig:
    write_spline_plots: bool = False
    write_relaxed_torsion_plots: bool = False
    write_normal_mode_wavefunction_plots: bool = False
    write_sharc_strings: bool = False
    wavefunction_plot_states: int = 8
    vibrational_states_json: str = "vibrational_states.json"
    initial_conditions_json: str = "initial_conditions.json"
    metadata_json: str = "metadata.json"


@dataclass(slots=True)
class InitialConditionConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    pes: PESConfig = field(default_factory=PESConfig)
    schrodinger: SchrodingerConfig = field(default_factory=SchrodingerConfig)
    relaxed_torsion: RelaxedTorsionConfig = field(default_factory=RelaxedTorsionConfig)
    qchem: QChemConfig = field(default_factory=QChemConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    internal_coordinates: InternalCoordinatesConfig = field(
        default_factory=InternalCoordinatesConfig
    )

    @classmethod
    def default(cls) -> InitialConditionConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> InitialConditionConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    @classmethod
    def from_default_yaml(cls) -> InitialConditionConfig:
        resource = files("UMRR").joinpath("default_config.yaml")
        with resource.open(encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> InitialConditionConfig:
        """Reject misspelled settings before starting electronic calculations."""
        if not isinstance(data, dict):
            raise ValueError("The YAML input must contain named configuration sections")
        section_types = dict(
            paths=PathConfig,
            sampling=SamplingConfig,
            pes=PESConfig,
            schrodinger=SchrodingerConfig,
            relaxed_torsion=RelaxedTorsionConfig,
            qchem=QChemConfig,
            output=OutputConfig,
            internal_coordinates=InternalCoordinatesConfig,
        )
        unknown = data.keys() - section_types.keys()
        if unknown:
            raise ValueError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")
        config = cls(
            **{
                name: _merge_dataclass(kind, data.get(name, {}))
                for name, kind in section_types.items()
            }
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Check the paper model and numerical domains before dispatching jobs."""
        if self.sampling.coordinate_system != "internal":
            raise ValueError("UM-RR uses sampling.coordinate_system: internal")
        if self.sampling.phase_space_method != "collective_anharmonic":
            raise ValueError("UM-RR uses phase_space_method: collective_anharmonic")
        if self.internal_coordinates.model != "minima":
            raise ValueError("UM-RR uses internal_coordinates.model: minima")
        if self.internal_coordinates.coordinate_construction != "direct":
            raise ValueError("UM-RR uses coordinate_construction: direct")
        if self.internal_coordinates.basin_interpolation != "hard":
            raise ValueError("UM-RR uses basin_interpolation: hard")
        positive_counts = {
            "sampling.num_samples": self.sampling.num_samples,
            "sampling.max_coordinate_sampling_attempts": self.sampling.max_coordinate_sampling_attempts,
            "qchem.num_processors": self.qchem.num_processors,
            "schrodinger.num_states": self.schrodinger.num_states,
            "schrodinger.grid_points": self.schrodinger.grid_points,
            "relaxed_torsion.num_states": self.relaxed_torsion.num_states,
            "internal_coordinates.map_max_iterations": self.internal_coordinates.map_max_iterations,
            "output.wavefunction_plot_states": self.output.wavefunction_plot_states,
        }
        for name, value in positive_counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.sampling.temperature_kelvin) or self.sampling.temperature_kelvin < 0:
            raise ValueError("sampling.temperature_kelvin must be finite and nonnegative")
        if (
            self.schrodinger.grid_points < 3
            or self.schrodinger.num_states >= self.schrodinger.grid_points
        ):
            raise ValueError("The Schrödinger grid must have more points than retained states")
        if self.relaxed_torsion.num_states >= self.schrodinger.grid_points:
            raise ValueError("The rotor starting state count must be smaller than the grid")
        if not isinstance(self.pes.num_scan_points, int):
            raise ValueError("pes.num_scan_points must be an integer")
        settings = self.internal_coordinates
        if not isfinite(settings.scan_extension_factor) or settings.scan_extension_factor <= 1:
            raise ValueError("scan_extension_factor must exceed one")
        if not isinstance(settings.scan_max_extensions, int) or settings.scan_max_extensions < 0:
            raise ValueError("scan_max_extensions must be a nonnegative integer")
        if (
            not isfinite(settings.scan_max_harmonic_energy_factor)
            or settings.scan_max_harmonic_energy_factor < 1
        ):
            raise ValueError("scan_max_harmonic_energy_factor must be at least one")
        if not 0 < settings.scan_nonrotor_dihedral_limit_deg < 180:
            raise ValueError("The local dihedral scan limit must be between 0 and 180 degrees")
        if not 0 < settings.boundary_probability_tolerance < 1:
            raise ValueError("boundary_probability_tolerance must be between zero and one")
        if not isinstance(settings.validation_points, int) or settings.validation_points < 0:
            raise ValueError("validation_points must be a nonnegative integer")
        if not isfinite(settings.map_tolerance) or settings.map_tolerance <= 0:
            raise ValueError("map_tolerance must be positive and finite")
        if not 0 < self.relaxed_torsion.population_tail_tolerance < 1:
            raise ValueError("population_tail_tolerance must be between zero and one")
        if self.pes.num_scan_points < 9:
            raise ValueError("pes.num_scan_points must be at least nine")
        if (
            not isfinite(self.pes.scan_energy_range_hartree)
            or self.pes.scan_energy_range_hartree <= 0
        ):
            raise ValueError("The transverse scan energy must be finite and positive")
        if self.relaxed_torsion.enabled != self.relaxed_torsion.sample:
            raise ValueError("Set relaxed_torsion.enabled and relaxed_torsion.sample together")
        indices = self.relaxed_torsion.atom_indices
        if indices and (
            len(indices) != 4
            or any(not isinstance(i, int) or i < 1 for i in indices)
            or len(set(indices)) != 4
        ):
            raise ValueError("Select four distinct, positive, one-based rotor atom indices")
        if self.qchem.skip_failed_jobs:
            raise ValueError("Each electronic scan point must complete successfully")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: type, values: dict[str, Any]) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} settings must be a mapping")
    valid_keys = set(cls.__dataclass_fields__)
    unknown = values.keys() - valid_keys
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} setting(s): {', '.join(sorted(unknown))}")
    return cls(**values)
