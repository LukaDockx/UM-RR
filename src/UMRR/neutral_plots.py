"""Save the actual independent-mode solutions and their electronic scan knots."""

import html
import json
from pathlib import Path

import numpy as np

from .serialization import write_json


def save_neutral_states(sampler):
    folder = Path(sampler.output_dir) / "neutral_potentials_and_wavefunctions"
    folder.mkdir(parents=True, exist_ok=True)
    basin = hasattr(sampler.potential, "models")
    models = sampler.potential.models if basin else [sampler.potential]
    coordinates = sampler.local_coordinates if basin else [sampler.coordinates]
    groups = sampler.priors if basin else sampler.priors[:1]
    checks = []
    images = []
    requested = (
        sampler.config.output.write_spline_plots
        or sampler.config.output.write_normal_mode_wavefunction_plots
        or sampler.config.output.write_relaxed_torsion_plots
    )
    if requested:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    factor = 627.509474
    for i, (model, c, local) in enumerate(zip(models, coordinates, groups)):
        report_path = (
            Path(sampler.output_dir) / f"minimum_{i:03d}" / "scan_domains.json"
            if basin
            else Path(sampler.output_dir) / "scan_domains.json"
        )
        domain_report = json.loads(report_path.read_text()) if report_path.exists() else None
        arrays = {}
        for m, prior in enumerate(local):
            x = prior["grid"]
            e = prior["energies"]
            psi = prior["wavefunctions"]
            v = model.one_mode(model.angles[0], m)
            arrays.update(
                {
                    f"mode_{m + 1}_grid": x,
                    f"mode_{m + 1}_energies": e,
                    f"mode_{m + 1}_wavefunctions": psi,
                    f"mode_{m + 1}_mass": np.asarray(prior["mass"]),
                }
            )
            checks.append(dict(minimum=i + 1, mode=m + 1, **prior["boundary_diagnostics"]))
            motion = ""
            if hasattr(c, "displacement"):
                j = int(np.argmax(abs(c.displacement[:, m])))
                kind, atoms = c.primitives.definitions[j]
                motion = f"; strongest internal component: {kind} " + "-".join(
                    f"{c.symbols[a]}{a + 1}" for a in atoms
                )
            title = f"Minimum {i + 1}, mode {m + 1}: harmonic {c.frequencies_hartree[m] * 219474.63137:.0f} cm⁻¹"
            if len(e) > 1:
                title += f"; 1D fundamental {(e[1] - e[0]) * 219474.63137:.0f} cm⁻¹"
            zero = float(np.min(v(x)))
            if (
                sampler.config.output.write_spline_plots
                or sampler.config.output.write_normal_mode_wavefunction_plots
            ):
                fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
                knots = model.grids[m]
                ax[0].plot(x, (v(x) - zero) * factor, "k-", lw=1.4)
                ax[0].scatter(knots, (v(knots) - zero) * factor, c="k", s=10)
                ax[0].set(
                    title="Full electronic scan and spline",
                    ylabel="Potential / kcal mol⁻¹",
                )
                ax[1].plot(x, (v(x) - zero) * factor, "k-", lw=1.2)
                count = min(len(e), sampler.config.output.wavefunction_plot_states)
                scale = max(0.25, float((e[count - 1] - e[0]) * factor) * 0.08)
                for n in range(count):
                    level = (e[n] - zero) * factor
                    ax[1].plot(
                        x,
                        level + scale * psi[n] / max(abs(psi[n])),
                        label=f"n={n}",
                        lw=1,
                    )
                    ax[2].plot(x, psi[n] ** 2, label=f"n={n}", lw=1)
                ax[1].set(
                    title="Eigenstates, offset by energy",
                    ylabel="Energy / kcal mol⁻¹; amplitudes scaled",
                    ylim=(-scale, (e[count - 1] - zero) * factor + 2 * scale),
                )
                ax[2].set(
                    title="Probability densities",
                    ylabel="Probability per coordinate unit",
                )
                for a in ax:
                    a.set_xlabel("Internal amplitude / mass-weighted a.u.")
                ax[2].legend(fontsize=7, ncol=2)
                fig.suptitle(title + "\n" + motion.lstrip("; "), fontsize=11)
                if domain_report is not None:
                    limits = domain_report["modes"][m]
                    reasons = {
                        "energy_target": "energy target reached",
                        "nonperiodic_dihedral_limit": "local dihedral limit; energy target not reached",
                        "maximum_extent": "amplitude limit; energy target not reached",
                        "maximum_harmonic_energy": "harmonic-energy limit; electronic target not reached",
                        "maximum_extensions": "extension limit; energy target not reached",
                    }
                    note = "Scan stops: " + "; ".join(
                        f"{label}: {reasons.get(limits[side]['stop_reason'], limits[side]['stop_reason'])}"
                        for side, label in [("negative", "left"), ("positive", "right")]
                    )
                    fig.text(0.5, 0.015, note, ha="center", fontsize=8)
                    fig.tight_layout(rect=(0, 0.05, 1, 1))
                else:
                    fig.tight_layout()
                if prior["boundary_diagnostics"].get("finite_grid_fallback", False):
                    ax[1].set_title("Finite-grid states: no confined state", color="firebrick")
                    ax[2].set_title("Boundary-dependent densities; inspect", color="firebrick")
                elif not prior["boundary_diagnostics"].get("momentum_boundary_check_passed", True):
                    ax[2].set_title("Endpoint density warning; inspect", color="firebrick")
                name = f"minimum_{i + 1:03d}_mode_{m + 1:03d}.png"
                fig.savefig(folder / name, dpi=155)
                plt.close(fig)
                images.append(name)
        np.savez_compressed(folder / f"minimum_{i + 1:03d}_states.npz", **arrays)
    if sampler.coordinates.has_rotor:
        x = sampler.rotor_s
        v = sampler.effective_rotor_potential
        e = sampler.rotor_energies
        psi = sampler.rotor_wavefunctions
        np.savez_compressed(
            folder / "rotor_states.npz",
            grid=x,
            potential=v,
            energies=e,
            wavefunctions=psi,
            total_path_length=sampler.total_length,
            rotor_transverse_free_energy_included=False,
        )
        if sampler.config.output.write_relaxed_torsion_plots:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            zero = float(np.min(v))
            ax[0].plot(x, (v - zero) * factor, "k-", label="Electronic path potential")
            for n in range(min(len(e), sampler.config.output.wavefunction_plot_states)):
                ax[0].plot(
                    x,
                    (e[n] - zero) * factor + 0.12 * psi[n] / max(abs(psi[n])),
                    label=f"n={n}",
                )
                ax[1].plot(x, psi[n] ** 2, label=f"n={n}")
            ax[0].set_ylabel("Energy / kcal mol⁻¹; amplitudes scaled")
            ax[1].set_ylabel("Probability per path-length unit")
            for a in ax:
                a.set_xlabel("Mass-weighted path length / a.u.")
                a.legend(fontsize=7, ncol=2)
            fig.suptitle("Single periodic rotor — electronic path potential")
            fig.tight_layout()
            name = "rotor_potential_and_wavefunctions.png"
            fig.savefig(folder / name, dpi=160)
            plt.close(fig)
            images.insert(0, name)
    write_json(folder / "boundary_diagnostics.json", checks)
    (folder / "README.txt").write_text(
        "Independent 1D proposal eigenstates before fixed-energy acceptance.\n"
        "Ordinary transverse dihedrals have nonperiodic boundaries; only the configured relaxed rotor is periodic.\n"
        "The full scan is shown separately from the low-energy eigenstates. Black points are calculated electronic energies.\n"
        "Boundary probabilities integrate the outer 5% at each end; consult scan_domains.json for reasons scans stopped.\n"
    )
    (folder / "index.html").write_text(
        '<html><body style="max-width:1500px;margin:auto;font-family:Arial"><h1>Neutral potentials and vibrational wavefunctions</h1><p>'
        + html.escape((folder / "README.txt").read_text())
        + "</p>"
        + "".join(f'<h2>{name}</h2><img src="{name}" style="max-width:100%">' for name in images)
        + "</body></html>"
    )
    sampler.log(f"Neutral spline plots and saved wavefunctions: {folder}")
