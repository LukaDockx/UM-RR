"""Prepare the equilibrium optimization and Hessian from an XYZ input."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import InitialConditionConfig
from .qchem_io import (
    generate_qchem_input_text,
    parse_projected_mass_weighted_hessian,
    parse_standard_orientation_angstrom,
    qchem_template_with_jobtype,
)
from .qchem_runner import run_qchem_filelist


def read_xyz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a single XYZ geometry in angstrom, preserving the atom order."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + count]]
    if count < 3 or len(rows) != count or any(len(row) != 4 for row in rows):
        raise ValueError(f"Expected a complete molecular XYZ input: {path}")
    geometry = np.asarray([[float(v) for v in row[1:]] for row in rows])
    if not np.all(np.isfinite(geometry)):
        raise ValueError(f"XYZ coordinates must be finite: {path}")
    return geometry, [row[0] for row in rows]


def prepare_reference(
    config: InitialConditionConfig,
    base_dir: str | Path,
    seed_xyz: str | Path,
    *,
    prepare_only: bool = False,
) -> Path:
    """Optimize the seed, then calculate a Hessian at the optimized geometry.

    The two jobs run sequentially. Their command uses the relaxed-scan thread
    allocation, while later transverse single points use the worker command.
    Completed jobs are reused only when input and output hashes still match.
    With prepare_only, write the optimization input and return without jobs.
    """
    config.validate()
    directory = Path(base_dir).resolve()
    seed = Path(seed_xyz)
    if not seed.is_absolute():
        seed = directory / seed
    geometry, symbols = read_xyz(seed)
    optimization = directory / "opt.in"
    frequency = directory / config.paths.freq_output
    command = config.relaxed_torsion.scan_command_template or config.qchem.command_template

    def write_input(path, xyz, jobtype):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            generate_qchem_input_text(
                xyz,
                symbols,
                charge=config.qchem.charge,
                multiplicity=config.qchem.multiplicity,
                template=qchem_template_with_jobtype(config.qchem.input_template, jobtype),
            ),
            encoding="utf-8",
        )

    def run_one(path, name):
        listing = directory / f"{name}_filelist.txt"
        listing.write_text(str(path.resolve()) + "\n", encoding="utf-8")
        run_qchem_filelist(
            listing,
            processes=1,
            command_template=command,
            skip_existing_outputs=config.qchem.skip_existing_outputs,
            skip_failed_jobs=False,
            verbose=True,
        )

    write_input(optimization, geometry, "opt")
    if prepare_only:
        return optimization
    run_one(optimization, "optimization")
    optimized_output = optimization.with_suffix(".out")
    if "OPTIMIZATION CONVERGED" not in optimized_output.read_text(errors="replace"):
        raise RuntimeError(f"Optimization did not converge: {optimized_output}")
    geometry, optimized_symbols = parse_standard_orientation_angstrom(optimized_output)
    if optimized_symbols != symbols:
        raise ValueError("Optimization changed the XYZ atom order")
    frequency_input = frequency.with_suffix(".in")
    write_input(frequency_input, geometry, "freq")
    run_one(frequency_input, "frequency")
    # Check the printed mass-weighted Hessian before starting a torsional scan.
    parse_projected_mass_weighted_hessian(frequency, len(symbols))
    return frequency
