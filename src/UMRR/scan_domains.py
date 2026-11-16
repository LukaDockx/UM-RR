"""Electronic-energy checks for independent, possibly asymmetric scan domains."""

from __future__ import annotations

import numpy as np

from .internal_coordinates import CoordinateDomainError
from .serialization import write_json


def build_energy_guided_scans(sampler):
    c, cfg = sampler.coordinates, sampler.settings
    target = float(cfg.scan_energy_hartree)
    if not (np.isfinite(target) and target > 0):
        raise ValueError("Scan energy target must be finite and positive")
    if not (np.isfinite(cfg.scan_extension_factor) and cfg.scan_extension_factor > 1):
        raise ValueError("scan_extension_factor must exceed one")
    energy_factor = float(cfg.scan_max_harmonic_energy_factor)
    if cfg.scan_max_extensions < 0 or not np.isfinite(energy_factor) or energy_factor < 1:
        raise ValueError(
            "Scan extension count must be nonnegative and harmonic energy factor finite and at least one"
        )
    if not 0 < cfg.scan_nonrotor_dihedral_limit_deg < 180:
        raise ValueError("Nonrotor dihedral scan limit must lie between zero and 180 degrees")
    frequencies = np.asarray(c.frequencies_hartree, float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("Energy-guided scans require positive transverse frequencies")
    # This only initializes distances. Actual electronic energies decide extension.
    initial = np.sqrt(2 * target) / frequencies
    angular = (
        np.asarray(c.axis_scan_limits(cfg.scan_nonrotor_dihedral_limit_deg))
        if hasattr(c, "axis_scan_limits")
        else np.full(c.n_modes, np.inf)
    )
    # In this mass-weighted coordinate E_harmonic = omega**2 * q**2 / 2.
    # Four times the harmonic energy permits twice the initial amplitude.
    harmonic_caps = initial * np.sqrt(energy_factor)
    if hasattr(c, "axis_scan_bounds"):
        domain = c.axis_scan_bounds(sampler.angles, cfg.scan_nonrotor_dihedral_limit_deg)
    else:
        domain = dict(
            lower=-angular,
            upper=angular,
            reasons=[["nonperiodic_dihedral_limit"] * c.n_modes for _ in range(2)],
            primitives=[[None] * c.n_modes for _ in range(2)],
        )
    domain_caps = np.array([-np.asarray(domain["lower"]), domain["upper"]]).T
    caps = np.minimum(domain_caps, harmonic_caps[:, None])
    if np.any(caps <= 0):
        raise CoordinateDomainError("The reference is outside the allowed scan interval")
    extents = np.minimum(initial[:, None], caps)
    grids = []
    for left, right in extents:
        if left == right:
            grid = np.linspace(-left, right, cfg.scan_points)
            grid[len(grid) // 2] = 0.0
        else:
            middle = cfg.scan_points // 2
            grid = np.r_[
                np.linspace(-left, 0.0, middle + 1),
                np.linspace(0.0, right, cfg.scan_points - middle)[1:],
            ]
        grids.append(grid)
    spacing = np.array([[g[1] - g[0], g[-1] - g[-2]] for g in grids])
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("No finite scan interval is available")
    sampler.scan_dir.mkdir(parents=True, exist_ok=True)
    sampler.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(sampler.scan_dir / "coordinate_definition.json", c.to_dict())
    report = dict(
        policy="Calculated electronic energy per side; existing knots retained",
        target_hartree=target,
        maximum_harmonic_energy_hartree=target * energy_factor,
        maximum_harmonic_energy_factor=energy_factor,
        iteration=0,
        complete=False,
        electronic_points=0,
        modes=[],
    )
    energies = [[{} for _ in range(c.n_modes)] for _ in sampler.angles]

    def save():
        write_json(sampler.scan_dir / "scan_domains.json", report)
        write_json(sampler.output_dir / "scan_domains.json", report)

    def evaluate(requests):
        paths, labels = [], []
        for a, m, value in requests:
            if value in energies[a][m]:
                continue
            q = np.zeros(c.n_modes)
            q[m] = value
            try:
                x = c.geometry(float(sampler.angles[a]), q, seed=sampler.base_geometries[a])
            except (CoordinateDomainError, FloatingPointError) as exc:
                report["failure"] = dict(
                    mode=m + 1,
                    angle=float(sampler.angles[a]),
                    amplitude=value,
                    reason=str(exc),
                )
                save()
                raise CoordinateDomainError(
                    f"Vibration {m + 1}, amplitude {value:.8g}: {exc}. "
                    "The requested scan was not silently shortened; inspect scan_domains.json."
                ) from exc
            path = sampler._write_job(f"angle_{a:03d}/axis_{m:03d}_{value:+.15e}", x)
            paths.append(path)
            labels.append((a, m, value))
        if paths:
            sampler.log(f"Running {len(paths)} Q-Chem singlepoints for transverse scan energies")
            values = sampler._run_jobs(paths, "one_mode")
            if len(values) != len(paths) or not np.all(np.isfinite(values)):
                raise ValueError("Each scan geometry needs a finite electronic energy")
            for (a, m, q), e in zip(labels, values):
                energies[a][m][q] = float(e)
            report["electronic_points"] += len(paths)

    references = [
        sampler._write_job(f"angle_{a:03d}/reference", x)
        for a, x in enumerate(sampler.base_geometries)
    ]
    bases = np.asarray(sampler._run_jobs(references, "one_mode"), float)
    if bases.shape != (len(sampler.angles),) or not np.all(np.isfinite(bases)):
        raise ValueError("Each reference geometry needs a finite electronic energy")
    report["electronic_points"] += len(references)
    for a in range(len(sampler.angles)):
        for m in range(c.n_modes):
            energies[a][m][0.0] = float(bases[a])
    for iteration in range(cfg.scan_max_extensions + 1):
        evaluate(
            [
                (a, m, float(q))
                for a in range(len(sampler.angles))
                for m, g in enumerate(grids)
                for q in g
            ]
        )
        report.update(iteration=iteration, modes=[])
        extended = False
        next_grids = []
        for m, g in enumerate(grids):
            record = dict(
                mode=m + 1,
                frequency_cm1=float(frequencies[m] * 219474.63137),
                harmonic_initial_extent=float(initial[m]),
                harmonic_cap_amplitude=float(harmonic_caps[m]),
                angular_cap_amplitude=float(angular[m]) if np.isfinite(angular[m]) else None,
                negative={},
                positive={},
            )
            additions = []
            for side, (sign, key) in enumerate([(-1, "negative"), (1, "positive")]):
                end = float(g[0] if sign < 0 else g[-1])
                rises = np.array([energies[a][m][end] - bases[a] for a in range(len(bases))])
                reached = bool(np.min(rises) >= target * (1 - 1.0e-10))
                at_cap = bool(abs(end) >= caps[m, side] * (1 - 1.0e-12))
                reason = (
                    "energy_target"
                    if reached
                    else domain["reasons"][side][m]
                    if at_cap and domain_caps[m, side] <= harmonic_caps[m]
                    else "maximum_harmonic_energy"
                    if at_cap
                    else "maximum_extensions"
                    if iteration == cfg.scan_max_extensions
                    else "extending"
                )
                record[key] = dict(
                    amplitude=end,
                    electronic_energy_rises_hartree=rises.tolist(),
                    harmonic_energy_hartree=float(0.5 * (frequencies[m] * end) ** 2),
                    target_reached=reached,
                    stop_reason=reason,
                    geometric_limit_amplitude=float(domain_caps[m, side])
                    if np.isfinite(domain_caps[m, side])
                    else None,
                    limiting_primitive=domain["primitives"][side][m],
                )
                if reason == "extending":
                    bound = min(abs(end) * cfg.scan_extension_factor, caps[m, side])
                    n = max(1, int(np.ceil((bound - abs(end)) / spacing[m, side])))
                    additions.extend(sign * np.linspace(abs(end), bound, n + 1)[1:])
                    extended = True
            record["number_electronic_knots"] = len(g)
            report["modes"].append(record)
            next_grids.append(np.sort(np.r_[g, additions]))
        save()
        if not extended:
            break
        grids = next_grids
    report.update(
        complete=True,
        all_energy_targets_reached=all(
            r[side]["target_reached"] for r in report["modes"] for side in ("negative", "positive")
        ),
    )
    for r in report["modes"]:
        for side in ("negative", "positive"):
            data = r[side]
            if not data["target_reached"]:
                sampler.log(
                    f"WARNING: vibration {r['mode']} {side} scan stops at "
                    f"{data['stop_reason']}; minimum endpoint rise "
                    f"{min(data['electronic_energy_rises_hartree']) * 627.509474:.3f} kcal/mol, "
                    f"requested {target * 627.509474:.3f}. Boundary-state diagnostics will be reported."
                )
    cuts = [
        [np.array([energies[a][m][float(q)] - bases[a] for q in g]) for m, g in enumerate(grids)]
        for a in range(len(bases))
    ]
    save()
    return grids, cuts, bases, report
