"""Geometry, kinetic-energy and minimum-specific contracts for direct maps."""

import json
from pathlib import Path

import numpy as np
import pytest

from UMRR.basin_sampler import SharedPath
from UMRR.direct_coordinates import DirectBasinChart, DirectInternalCoordinateMap
from UMRR.internal_coordinates import periodic_delta

DATA = Path(__file__).parent / "data"


def case(name):
    return DirectInternalCoordinateMap(
        **json.loads((DATA / "internal_coordinate_cases.json").read_text())[name]
    )


def convert(record):
    return DirectInternalCoordinateMap(
        record["symbols"],
        record["masses_electron"],
        record["reference_bohr"],
        record["hessian_mass_weighted"],
        rotor_indices=record["rotor_indices_zero_based"],
        path_geometries_bohr=record["path_geometries_bohr"],
    )


@pytest.mark.parametrize("name", ["methanol_rotor_projected", "hydrazine", "CH2ClCHFCl/Rotor"])
def test_direct_roundtrip_across_rotor(name):
    c = case(name)
    rng = np.random.default_rng(43)
    for angle in np.linspace(0, 2 * np.pi, 19):
        q = rng.normal(size=c.n_modes) / np.sqrt(2 * c.frequencies_hartree)
        x = c.geometry(angle, q)
        phi, actual = c.coordinates(x)
        assert abs(periodic_delta(phi - angle)) < 2.0e-12
        np.testing.assert_allclose(actual, q, atol=3.0e-10, rtol=0)
        assert np.linalg.eigvalsh(c.mass_metric(x)).min() > 0


def test_direct_jacobian_and_cartesian_momentum():
    c = case("methanol_rotor_projected")
    charts = [c, DirectBasinChart(c, SharedPath(c), [0, 1, 4, 2, 3, 5])]
    rng = np.random.default_rng(51)
    for chart in charts:
        q = rng.normal(size=c.n_modes) * 10
        x = chart.geometry(1.3, q)
        direction = rng.normal(size=x.shape)
        step = 1.0e-6
        before, after = (
            chart.coordinates(x - step * direction),
            chart.coordinates(x + step * direction),
        )
        finite = np.r_[periodic_delta(after[0] - before[0]), after[1] - before[1]] / (2 * step)
        j = chart.jacobian(x)
        np.testing.assert_allclose(j @ direction.ravel(), finite, atol=2.0e-6, rtol=2.0e-7)
        p = rng.normal(size=j.shape[0])
        velocity = chart.velocity(x, p)
        np.testing.assert_allclose((chart.masses[:, None] * velocity).sum(axis=0), 0, atol=3.0e-12)
        np.testing.assert_allclose(
            np.cross(x, chart.masses[:, None] * velocity).sum(axis=0), 0, atol=3.0e-12
        )
        physical = 0.5 * np.sum(chart.masses[:, None] * velocity**2)
        generalized = 0.5 * p @ np.linalg.solve(chart.mass_metric(x), p)
        np.testing.assert_allclose(physical, generalized, rtol=2.0e-13)


def test_harmonic_tangents_and_mass_are_preserved():
    c = case("methanol_rotor_projected")
    tangent = c.normal_modes_mass_weighted.T / c.sqrt_masses[:, None]
    np.testing.assert_allclose(
        c.jacobian(c.reference)[1:] @ tangent, np.eye(c.n_modes), atol=2.0e-7
    )
    np.testing.assert_allclose(np.diag(c.mass_metric(c.reference))[1:], 1.0, atol=3.0e-7)
    reduced = (
        c.normal_modes_mass_weighted @ c.hessian_mass_weighted @ c.normal_modes_mass_weighted.T
    )
    np.testing.assert_allclose(reduced, np.diag(c.frequencies_hartree**2), atol=1.0e-13)


def test_three_distinct_minima_remain_distinct():
    d = json.loads((DATA / "inequivalent_basin_cases.json").read_text())
    path = SharedPath(convert(d["path"]))
    locals_ = [convert(c) for c in d["local_coordinates"]]
    assert len(locals_) == 3
    assert not np.allclose(locals_[0].frequencies_hartree, locals_[1].frequencies_hartree)
    assert not np.allclose(locals_[0].coefficients, locals_[1].coefficients)
    for b in d["basins"]:
        c = locals_[b["unique_index"]]
        chart = DirectBasinChart(c, path, b["permutation"])
        for delta in [-0.5, 0.0, 0.5]:
            q = np.linspace(-2, 3, c.n_modes)
            x = chart.geometry(b["minimum_angle"] + delta, q)
            np.testing.assert_allclose(chart.coordinates(x)[1], q, atol=3.0e-10)
        tangent = c.normal_modes_mass_weighted.T / c.sqrt_masses[:, None]
        np.testing.assert_allclose(
            chart.jacobian(c.reference)[1:] @ tangent, np.eye(c.n_modes), atol=3.0e-7
        )


def test_direct_serialization_and_explicit_local_dihedral_limit():
    c = case("methanol_rotor_projected")
    restored = DirectInternalCoordinateMap.from_dict(c.to_dict())
    q = np.linspace(-10, 10, c.n_modes)
    np.testing.assert_allclose(restored.geometry(2.1, q), c.geometry(2.1, q), atol=1.0e-13)
    caps = c.axis_scan_limits(150.0)
    for m, cap in enumerate(caps):
        if not np.isfinite(cap):
            continue
        displacement = c.displacement[c.bounded_dihedrals, m] * cap
        np.testing.assert_allclose(np.max(np.abs(displacement)), np.deg2rad(150.0), atol=2.0e-14)
    for invalid in [0.0, 180.0, -1.0]:
        with pytest.raises(ValueError):
            c.axis_scan_limits(invalid)


def test_nonrotor_water_uses_direct_geometry():
    c = DirectInternalCoordinateMap(
        ["O", "H", "D"],
        [29157.0, 1837.0, 3672.0],
        [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0], [-0.45, 1.65, 0.0]],
        np.eye(9) * 1.0e-4,
    )
    q = np.array([4.0, -9.0, 11.0])
    x = c.geometry(0.0, q)
    np.testing.assert_allclose(c.coordinates(x)[1], q, atol=1.0e-11)
    np.testing.assert_allclose(c.mass_metric(c.reference), np.eye(3), atol=1.0e-12)


def test_invalid_equivalent_atom_permutations_are_rejected():
    c = case("methanol_rotor_projected")
    with pytest.raises(ValueError, match="permutation"):
        DirectBasinChart(c, SharedPath(c), [0] * 6)
    with pytest.raises(ValueError, match="permutation"):
        DirectBasinChart(c, SharedPath(c), [0, 1, 2.5, 3, 4, 5])
    with pytest.raises(ValueError, match="elements"):
        DirectBasinChart(c, SharedPath(c), [1, 0, 2, 3, 4, 5])


def test_complete_methanol_energy_relabeling_and_saved_potential(tmp_path):
    from UMRR.basin_sampler import BasinPotential
    from UMRR.internal_sampler import InternalPotential

    c = case("methanol_rotor_projected")
    group = [[0, 1, 2, 3, 4, 5], [0, 1, 4, 2, 3, 5], [0, 1, 3, 4, 2, 5]]
    grid = np.tile(np.linspace(-3000, 3000, 101), (c.n_modes, 1))
    cuts = (
        0.5 * c.frequencies_hartree[:, None] ** 2 * grid**2
        + np.arange(c.n_modes)[:, None] * 1.0e-14 * grid**3
    )[None]
    model = InternalPotential(c, [0], grid, cuts, [-115.0])
    basins = []
    for p in group:
        phi = c.rotor.value(c.reference[p]) % (2 * np.pi)
        basins.append(
            dict(
                unique_index=0,
                minimum_angle=phi,
                left_angle=(phi - np.pi / 3) % (2 * np.pi),
                permutation=p,
            )
        )
    angle = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    pot = BasinPotential(
        c,
        [c],
        [model],
        basins,
        angle,
        -115.0 + 0.002 * (1 - np.cos(3 * angle)),
        interpolation="hard",
        complete_symmetry=True,
    )
    rng = np.random.default_rng(61)
    for phi in [b["minimum_angle"] for b in basins]:
        x = pot.charts[0].geometry(phi, rng.normal(size=c.n_modes) * 2)
        expected = pot.evaluate_geometry(x)
        for p in group:
            np.testing.assert_allclose(pot.evaluate_geometry(x[p]), expected, atol=1.0e-12)
    pot.save(tmp_path)
    restored = BasinPotential.load(tmp_path)
    np.testing.assert_allclose(
        restored.evaluate_geometry(x), pot.evaluate_geometry(x), atol=1.0e-13
    )


def test_direct_basin_restores_exact_saved_definition():
    c = case("methanol_rotor_projected")
    chart = DirectBasinChart(c, SharedPath(c), [0, 1, 4, 2, 3, 5])
    record = chart.to_dict()
    restored = DirectBasinChart.from_dict(record)
    q = np.linspace(-5, 6, c.n_modes)
    np.testing.assert_allclose(restored.geometry(0.7, q), chart.geometry(0.7, q), atol=1.0e-13)
    changed = dict(record, primitives=list(reversed(record["primitives"])))
    with pytest.raises(ValueError, match="atom placement"):
        DirectBasinChart.from_dict(changed)
    changed = dict(record, displacement=2 * record["displacement"])
    with pytest.raises(ValueError, match="inconsistent"):
        DirectBasinChart.from_dict(changed)


def test_rebuilding_after_passive_hydrogen_relabeling_preserves_scan_paths():
    from scipy.optimize import linear_sum_assignment

    record = json.loads((DATA / "internal_coordinate_cases.json").read_text())[
        "methanol_rotor_projected"
    ]
    p = np.array([0, 1, 4, 2, 3, 5])
    inverse = np.argsort(p)
    cartesian = (3 * p[:, None] + np.arange(3)).ravel()
    changed = dict(record)
    for key in ("symbols", "masses", "reference_bohr"):
        changed[key] = np.asarray(record[key])[p]
    changed["hessian_mass_weighted"] = np.asarray(record["hessian_mass_weighted"])[
        cartesian[:, None], cartesian[None, :]
    ]
    changed["rotor_indices"] = inverse[np.asarray(record["rotor_indices"])]
    changed["path_geometries_bohr"] = np.asarray(record["path_geometries_bohr"])[:, p]
    a, b = DirectInternalCoordinateMap(**record), DirectInternalCoordinateMap(**changed)
    reordered = b.normal_modes_mass_weighted.reshape(b.n_modes, len(p), 3)[:, inverse].reshape(
        b.n_modes, -1
    )
    overlap = a.normal_modes_mass_weighted @ reordered.T
    rows, columns = linear_sum_assignment(-abs(overlap))
    np.testing.assert_allclose(abs(overlap[rows, columns]), 1.0, atol=1.0e-12)
    signs = np.sign(overlap[rows, columns])
    for angle in [0.0, 1.5, 3.0]:
        for mode in range(a.n_modes):
            qa = np.zeros(a.n_modes)
            qa[mode] = min(100.0, 0.8 * a.axis_scan_bounds([angle])["upper"][mode])
            qb = np.zeros(b.n_modes)
            qb[columns] = qa * signs
            xa, xb = a.geometry(angle, qa), b.geometry(angle, qb)[inverse]
            distance_a = np.linalg.norm(xa[:, None] - xa[None, :], axis=-1)
            distance_b = np.linalg.norm(xb[:, None] - xb[None, :], axis=-1)
            np.testing.assert_allclose(distance_a, distance_b, atol=2.0e-11, rtol=0)


def test_finite_terminal_fragment_rotation_keeps_bonds_and_hydrogen_separations():
    from scipy.spatial.transform import Rotation

    from UMRR.constants import ANGSTROM_TO_BOHR

    reference = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [1.8, 1.3, 0.0],
                [-0.36, 1.0, 0.0],
                [-0.36, -0.5, 0.8660254],
                [-0.36, -0.5, -0.8660254],
                [1.2, 1.9, 0.4],
            ]
        )
        * ANGSTROM_TO_BOHR
    )
    axis = reference[2] - reference[1]
    axis /= np.linalg.norm(axis)
    paths = []
    for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        x = reference.copy()
        x[6] = reference[2] + Rotation.from_rotvec(theta * axis).apply(reference[6] - reference[2])
        paths.append(x)
    c = DirectInternalCoordinateMap(
        ["C", "O", "O", "H", "H", "H", "H"],
        [21875.0, 29157.0, 29157.0, 1837.0, 1837.0, 1837.0, 1837.0],
        reference,
        np.eye(21) * 1.0e-4,
        rotor_indices=[0, 1, 2, 6],
        path_geometries_bohr=paths,
    )
    phi = c.rotor.value(reference)
    base = c.geometry(phi, np.zeros(c.n_modes))
    axis = base[1] - base[0]
    axis /= np.linalg.norm(axis)
    velocity = np.zeros_like(base)
    velocity[3:6] = np.cross(axis, base[3:6] - base[0])
    tangent = c.jacobian(base)[1:] @ velocity.ravel()
    for amplitude in [-3.0, -1.0, 1.0, 3.0]:
        x = c.geometry(phi, amplitude * tangent)
        expected = base.copy()
        theta = amplitude
        expected[3:6] = base[0] + Rotation.from_rotvec(theta * axis).apply(base[3:6] - base[0])
        distance_x = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        distance_expected = np.linalg.norm(expected[:, None] - expected[None, :], axis=-1)
        np.testing.assert_allclose(distance_x, distance_expected, atol=2.0e-11, rtol=0)
        # No second periodic coordinate was introduced by the placement rule.
        assert len(c.primitives.rotor_positions) == 1
    from UMRR.internal_coordinates import CoordinateDomainError

    with pytest.raises(CoordinateDomainError, match="Nonrotor dihedral"):
        c.geometry(phi, 7 * tangent)


def test_invalid_primitive_values_are_rejected_without_folding():
    from UMRR.internal_coordinates import CoordinateDomainError

    c = case("methanol_rotor_projected")
    base = c.primitives.calculate(c.reference)
    for k, (kind, _) in enumerate(c.primitives.definitions):
        if kind not in {"length", "bend"}:
            continue
        trial = base.copy()
        trial[k] = -1.0 if kind == "length" else np.pi + 0.1
        with pytest.raises(CoordinateDomainError):
            c.primitives.geometry(trial)


def test_periodic_path_interpolation_preserves_positive_lengths_and_derivatives():
    from UMRR.direct_coordinates import _path_spline

    x = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    values = np.array([0.1, 0.1, 4.0, 4.0, 0.1, 0.1])[:, None]
    spline, changed = _path_spline(x, values, np.array([0.0]), np.array([np.inf]))
    assert changed == [0]
    assert spline(np.linspace(0, 2 * np.pi, 1001)).min() >= 0.1 - 1.0e-14
    np.testing.assert_allclose(spline(x), values, atol=1.0e-14)
    np.testing.assert_allclose(spline(0, 1), spline(2 * np.pi, 1), atol=1.0e-13)


def test_nearly_linear_group_uses_two_bending_components():
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 4.0, 1.0e-4]])
    c = DirectInternalCoordinateMap(
        ["C", "C", "C", "H"], [21875.0, 21875.0, 21875.0, 1837.0], reference, np.eye(12) * 1.0e-4
    )
    # Reference selection can avoid a linear plane altogether; the resulting
    # coordinate system must reproduce both perpendicular physical motions.
    for direction in ([0.001, 0.0, 0.0], [0.0, 0.0, 0.001]):
        x = reference.copy()
        x[3] += direction
        phi, q = c.coordinates(x)
        rebuilt = c.geometry(phi, q)
        np.testing.assert_allclose(
            np.linalg.norm(x[:, None] - x[None, :], axis=-1),
            np.linalg.norm(rebuilt[:, None] - rebuilt[None, :], axis=-1),
            atol=1.0e-10,
        )


def test_collinear_terminal_group_selects_cartesian_components():
    from scipy.spatial.transform import Rotation

    x = np.array(
        [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [2.6, 2.6, 0.0], [3.6, 2.6, 1.0], [2.6, 4.8, 0.0]]
    )
    axis = x[2] - x[1]
    axis /= np.linalg.norm(axis)
    paths = []
    for phi in np.linspace(0.0, 2 * np.pi, 36, endpoint=False):
        r = x.copy()
        r[3] = x[2] + Rotation.from_rotvec(phi * axis).apply(x[3] - x[2])
        paths.append(r)
    c = DirectInternalCoordinateMap(
        ["C", "C", "C", "H", "N"],
        [21875.0, 21875.0, 21875.0, 1837.0, 25519.0],
        x,
        np.eye(15) * 1.0e-4,
        rotor_indices=[0, 1, 2, 3],
        path_geometries_bohr=paths,
    )
    assert any(kind == "cartesian" for kind, _, _ in c.primitives.steps)
    for direction in ([0.01, 0, 0], [0, 0, 0.01], [0.01, 0, 0.01]):
        r = x.copy()
        r[4] += direction
        phi, q = c.coordinates(r)
        rebuilt = c.geometry(phi, q)
        np.testing.assert_allclose(
            np.linalg.norm(r[:, None] - r[None, :], axis=-1),
            np.linalg.norm(rebuilt[:, None] - rebuilt[None, :], axis=-1),
            atol=1.0e-11,
        )


@pytest.mark.parametrize("bend", [2.0e-6, 2.0e-4, np.pi - 2.0e-6, np.pi - 2.0e-4])
def test_dihedral_derivative_remains_correct_near_straight_bends(bend):
    from UMRR.direct_internal import DirectDihedral

    x = np.array(
        [[np.cos(bend), np.sin(bend), 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.4, 0.8, 0.3]]
    )
    dihedral = DirectDihedral(0, 1, 2, 3)
    derivative = dihedral.derivative(x)
    direction = np.random.default_rng(86).normal(size=x.shape)
    h = 1.0e-5 * abs(np.sin(bend))
    difference = (dihedral.value(x + h * direction) - dihedral.value(x - h * direction) + np.pi) % (
        2 * np.pi
    ) - np.pi
    np.testing.assert_allclose(difference / (2 * h), np.sum(derivative * direction), rtol=2.0e-7)
    np.testing.assert_allclose(derivative.sum(axis=0), 0.0, atol=2.0e-10)
    np.testing.assert_allclose(np.cross(x, derivative).sum(axis=0), 0.0, atol=2.0e-10)
