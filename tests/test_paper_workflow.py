"""Test the complete scan, quantum-state and ensemble workflow analytically."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from UMRR.basin_sampler import BasinPotential, BasinSampler
from UMRR.cli import main
from UMRR.config import InitialConditionConfig
from UMRR.constants import ANGSTROM_TO_BOHR, ELECTRON_MASS_TO_AMU
from UMRR.internal_sampler import InternalSampler
from UMRR.reference import prepare_reference

ROOT = Path(__file__).resolve().parents[1]


def test_meooh_configuration_check_needs_no_outputs(capsys):
    assert main([str(ROOT / "examples/MeOOH/config.yaml"), "--check"]) == 0
    assert "configuration is valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    "section, values",
    [
        ("sampling", {"num_samples": 0}),
        ("sampling", {"temperature_kelvin": -1}),
        ("internal_coordinates", {"basin_interpolation": "smooth"}),
        ("internal_coordinates", {"coordinate_construction": "iterative"}),
        ("sampling", {"coordinate_system": "cartesian"}),
        ("qchem", {"num_processor": 8}),
        ("relaxed_torsion", {"enabled": True, "sample": True, "atom_indices": [1, 2, 2, 7]}),
    ],
)
def test_invalid_input_is_rejected_before_calculations(section, values):
    with pytest.raises(ValueError):
        InitialConditionConfig.from_mapping({section: values})


def test_reference_preparation_writes_only_an_optimization_input(tmp_path, monkeypatch):
    import UMRR.reference as reference

    cfg = InitialConditionConfig.from_yaml(ROOT / "examples/MeOOH/config.yaml")
    monkeypatch.setattr(
        reference,
        "run_qchem_filelist",
        lambda *args, **kwargs: pytest.fail("Unexpected electronic job"),
    )
    path = prepare_reference(cfg, tmp_path, ROOT / "examples/MeOOH/seed.xyz", prepare_only=True)
    text = path.read_text()
    assert "jobtype opt" in text.lower()
    assert "METHOD RIMP2" in text
    assert "AUX_BASIS RIMP2-aug-cc-pVDZ" in text
    assert [p.name for p in tmp_path.iterdir()] == ["opt.in"]


def test_full_harmonic_water_workflow_retains_energy_and_rigid_body_constraints(
    tmp_path, monkeypatch
):
    cfg = InitialConditionConfig()
    cfg.sampling.num_samples = 40
    cfg.sampling.temperature_kelvin = 0
    cfg.sampling.random_seed = 71
    cfg.pes.num_scan_points = 9
    cfg.pes.scan_energy_range_hartree = 0.03
    cfg.schrodinger.grid_points = 400
    cfg.schrodinger.num_states = 4
    cfg.validate()
    xyz = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]])
    masses = np.array([16.0, 1.0, 1.0]) / ELECTRON_MASS_TO_AMU
    generator = SimpleNamespace(
        config=cfg,
        numpy_rng=np.random.default_rng(71),
        scan_dir=tmp_path / "scan",
        output_dir=tmp_path / "out",
        _log=lambda message: None,
        atom_symbols=["O", "H", "H"],
        mass_matrix=np.tile(masses[:, None], (1, 3)),
        equilibrium_geometry_angstrom=xyz / ANGSTROM_TO_BOHR,
        hessian=np.eye(9) * 1.0e-4 / ELECTRON_MASS_TO_AMU,
        relaxed_torsion_solution=None,
        electronic_reference_hartree=-75.0,
        verbose=False,
        process_relaxed_torsion=lambda: None,
    )

    def analytic_electronic_energies(sampler, paths, name):
        energies = []
        for path in paths:
            block = path.read_text().split("$molecule")[1].split("$end")[0]
            rows = block.strip().splitlines()[1:]
            geometry = np.array([[float(value) for value in row.split()[1:]] for row in rows])
            _, q = sampler.coordinates.coordinates(geometry * ANGSTROM_TO_BOHR)
            energies.append(-75 + 0.5 * np.sum((q * sampler.coordinates.frequencies_hartree) ** 2))
        return np.asarray(energies)

    monkeypatch.setattr(InternalSampler, "_run_jobs", analytic_electronic_energies)
    sampler = BasinSampler(generator)
    result = sampler.run()
    assert result.num_samples == 40
    np.testing.assert_allclose(
        result.potential_energies_hartree + result.kinetic_energies_hartree,
        result.target_energies_hartree,
        atol=2.0e-14,
        rtol=0,
    )
    momenta = result.velocities_au * masses[None, :, None]
    np.testing.assert_allclose(momenta.sum(axis=1), 0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.cross(result.geometries_bohr, momenta).sum(axis=1), 0, atol=1.0e-12
    )
    restored = BasinPotential.load(generator.output_dir)
    for geometry, energy in zip(result.geometries_bohr, result.potential_energies_hartree):
        np.testing.assert_allclose(
            restored.evaluate_geometry(geometry) + restored.reference_energy,
            energy,
            atol=2.0e-14,
            rtol=0,
        )
    metadata = json.loads((generator.output_dir / cfg.output.metadata_json).read_text())
    assert metadata["electronic_scan_counts"]["one_mode"] == 1 + sum(
        len(grid) - 1 for grid in sampler.potential.models[0].grids
    )
    assert (
        generator.output_dir / "neutral_potentials_and_wavefunctions/minimum_001_states.npz"
    ).exists()
