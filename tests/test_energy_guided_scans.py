"""Scan stopping checks use measured energies and preserve all previous knots."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from UMRR.config import InternalCoordinatesConfig
from UMRR.internal_coordinates import CoordinateDomainError
from UMRR.internal_sampler import InternalPotential
from UMRR.scan_domains import build_energy_guided_scans
from UMRR.schrodinger import solve_schrodinger_1d
from UMRR.state_boundaries import (
    boundary_diagnostics,
    prepare_transverse_state_distributions,
)


class Coordinate:
    symbols = ["H", "H"]
    n_modes = 2
    frequencies_hartree = np.array([0.01, 0.02])

    def geometry(self, angle, q, seed=None):
        return np.array(q)

    def to_dict(self):
        return {"test": "analytic coordinates"}


def sampler(tmp_path, potential, coordinates=None, **kw):
    c = coordinates or Coordinate()
    cfg = InternalCoordinatesConfig(scan_points=9, scan_energy_hartree=0.01, **kw)
    calls = {}

    def write(name, x):
        calls[name] = np.array(x)
        return Path(name)

    def run(paths, name):
        return np.array([-75 + potential(calls[str(p)]) for p in paths])

    return SimpleNamespace(
        settings=cfg,
        coordinates=c,
        angles=[0.0],
        scan_dir=tmp_path / "scan",
        output_dir=tmp_path / "out",
        base_geometries=[np.zeros(2)],
        _write_job=write,
        _run_jobs=run,
        log=lambda *args: None,
    )


def test_one_sided_growth_keeps_every_original_knot_and_ragged_serialization(tmp_path):
    def potential(q):
        return 0.00005 * q[0] ** 2 * (1 + 0.01 * q[0]) + 0.0002 * q[1] ** 2

    s = sampler(tmp_path, potential)
    grids, cuts, base, r = build_energy_guided_scans(s)
    initial = np.sqrt(0.02) / s.coordinates.frequencies_hartree
    assert grids[0][0] < -initial[0] and grids[0][-1] == initial[0]
    for m, g in enumerate(grids):
        assert np.all(np.isin(np.linspace(-initial[m], initial[m], 9), g))
        assert np.count_nonzero(np.isclose(g, 0, atol=1.0e-12)) == 1
    assert r["all_energy_targets_reached"]
    assert len(grids[0]) != len(grids[1])
    p = InternalPotential(s.coordinates, [0], grids, cuts, base)
    p.save(tmp_path / "potential.npz")
    q = InternalPotential.load(tmp_path / "potential.npz", s.coordinates)
    for i in [0, 1]:
        np.testing.assert_allclose(p.one_mode(0, i)(grids[i]), q.one_mode(0, i)(grids[i]))
    np.testing.assert_allclose(p.raw(0, [1, 2]), q.raw(0, [1, 2]))


def test_angular_limit_is_reported_as_unmet_energy_not_confinement(tmp_path):
    c = Coordinate()
    c.axis_scan_limits = lambda angle: np.array([20.0, 20.0])
    s = sampler(
        tmp_path,
        lambda q: 0.001 * np.sum(1 - np.cos(q / 10)),
        c,
        scan_max_harmonic_energy_factor=16,
    )
    _, _, _, r = build_energy_guided_scans(s)
    assert not r["all_energy_targets_reached"]
    assert all(
        m[side]["stop_reason"] == "nonperiodic_dihedral_limit"
        for m in r["modes"]
        for side in ["negative", "positive"]
    )
    assert all(
        m[side]["amplitude"] == sign * 20
        for m in r["modes"]
        for side, sign in [("negative", -1), ("positive", 1)]
    )


def test_geometry_failure_is_not_silently_shrunk(tmp_path):
    c = Coordinate()

    def geometry(angle, q, seed=None):
        if abs(q[0]) > 10:
            raise CoordinateDomainError("deliberate invalid shape")
        return q

    c.geometry = geometry
    s = sampler(tmp_path, lambda q: 0, c)
    with pytest.raises(CoordinateDomainError, match="not silently shortened"):
        build_energy_guided_scans(s)
    r = json.loads((s.output_dir / "scan_domains.json").read_text())
    assert "failure" in r and not r["complete"]


def test_geometric_limits_are_asymmetric_and_do_not_abort_scan(tmp_path):
    c = Coordinate()
    c.axis_scan_bounds = lambda angles, limit: dict(
        lower=[-8.0, -np.inf],
        upper=[25.0, np.inf],
        reasons=[
            ["bend_limit", "distance_limit"],
            ["nonperiodic_dihedral_limit", "distance_limit"],
        ],
        primitives=[["bend:0,1,2", None], ["dihedral:0,1,2,3", None]],
    )

    def geometry(angle, q, seed=None):
        assert -8.0 <= q[0] <= 25.0
        return q

    c.geometry = geometry
    s = sampler(tmp_path, lambda q: 0.0, c)
    grids, _, _, report = build_energy_guided_scans(s)
    assert grids[0][0] == -8.0 and grids[0][-1] == 25.0
    assert np.all(np.diff(grids[0]) > 0) and np.count_nonzero(grids[0] == 0) == 1
    assert report["complete"] and not report["all_energy_targets_reached"]
    assert report["modes"][0]["negative"]["stop_reason"] == "bend_limit"
    assert report["modes"][0]["positive"]["stop_reason"] == "nonperiodic_dihedral_limit"


def test_outer_strip_check_detects_box_state_with_zero_endpoint_values():
    x = np.linspace(0, 1, 1001)
    psi = np.sqrt(2) * np.sin(np.pi * x)[None, :]
    r = boundary_diagnostics([0.1], psi, x, 1, 0, 0.001)
    assert not r["boundary_check_passed"]
    assert r["thermal_boundary_probability"] > 0.001
    assert psi[0, 0] == 0 and abs(psi[0, -1]) < 1.0e-14


@pytest.mark.parametrize("factor", [1.0, 4.0, 9.0])
def test_dissociating_cut_stops_at_harmonic_energy_not_amplitude_ratio(tmp_path, factor):
    # A flat electronic curve never reaches the electronic target.
    s = sampler(tmp_path, lambda q: 0.0, scan_max_harmonic_energy_factor=factor)
    grids, _, _, report = build_energy_guided_scans(s)
    target = s.settings.scan_energy_hartree
    for m, g in enumerate(grids):
        np.testing.assert_allclose(
            0.5 * (s.coordinates.frequencies_hartree[m] * g[[0, -1]]) ** 2,
            factor * target,
            rtol=1.0e-12,
        )
        for side in ["negative", "positive"]:
            assert report["modes"][m][side]["stop_reason"] == "maximum_harmonic_energy"
            assert not report["modes"][m][side]["target_reached"]


def test_no_confined_state_warns_and_retains_unchanged_finite_grid_solutions():
    e, p, x = solve_schrodinger_1d(
        lambda q: -q, np.array([-5.0, 5.0]), 1.0, grid_points=200, num_states=6
    )
    before = p.copy()
    messages = []
    n, momentum, d = prepare_transverse_state_distributions(
        e, p, x, 1.0, -5.0, 0.1, 0.001, messages.append, "Mode 1"
    )
    assert n == 6 and len(momentum) == 6
    assert d["finite_grid_fallback"] and not d["confinement_check_passed"]
    assert any("no computed state" in s for s in messages)
    np.testing.assert_array_equal(p, before)
    for distribution in momentum:
        assert np.isfinite(distribution.second_moment_au)
        assert distribution.cdf[0] == 0 and distribution.cdf[-1] == 1


def test_endpoint_density_warning_does_not_reject_a_computed_state():
    x = np.linspace(-1, 1, 101)
    p = np.ones((1, len(x))) / np.sqrt((x[1] - x[0]) * len(x))
    messages = []
    n, momentum, d = prepare_transverse_state_distributions(
        [0.0], p, x, 1.0, 1.0, 0.0, 0.001, messages.append, "Mode 2"
    )
    assert n == 1 and len(momentum) == 1
    assert not d["finite_grid_fallback"] and not d["momentum_boundary_check_passed"]
    assert d["endpoint_density_ratios"] == [1.0]
    assert any("endpoint" in s for s in messages)
