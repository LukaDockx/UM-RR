MeOOH input example
===================

``seed.xyz`` supplies the seven-atom geometry in angstrom. Its atom order is
C, O, O, H, H, H, H; ``atom_indices: [1, 2, 3, 7]`` selects the C--O--O--H
rotor using one-based indices. Keep that numbering consistent if editing the
geometry. ``config.yaml`` specifies the numerical and electronic settings.

From the repository root, after installing the package::

    UMRR --check examples/MeOOH/config.yaml
    python examples/MeOOH/run.py

To write only the initial optimization input::

    python examples/MeOOH/run.py --prepare-only

The run proceeds through equilibrium optimization and frequency calculation,
the relaxed torsional scan, optimization and Hessians of distinct minima,
transverse energy scans, independent quantum states, and ensemble generation.
The electronic stages require Q-Chem on PATH. The example reserves at most
eight CPU cores: one eight-thread job for each initial stage, and up to eight
one-thread jobs for minimum and transverse calculations. Each concurrent job
has ``MEM_TOTAL 3000`` MB in its Q-Chem input.

Change ``sampling.num_samples`` or ``sampling.random_seed`` for another
ensemble. Electronic outputs are reused when their hashes match the inputs.
The two mirror-related gauche minima receive separate Hessians and curves:
the equivalence test uses proper rotation and atom permutation.

A completed run writes:

* ``opt.in`` / ``opt.out`` and ``freq.in`` / ``freq.out`` for the reference;
* ``relaxed_rotor_scan/`` for the constrained torsional scan;
* ``relaxed_torsion_minima/`` for distinct-minimum optimizations and Hessians;
* ``qchem_scan/basin_internal/`` for transverse inputs, energies and scan limits;
* ``generated/`` for the model, quantum states, diagnostics and initial conditions.

For manual electronic execution, set ``qchem.run_jobs``,
``relaxed_torsion.run_qchem`` and ``relaxed_torsion.run_minimum_jobs`` to false
and use ``UMRR config.yaml`` after preparing ``freq.out``.
The program writes the next missing stage's inputs and names the required
outputs. The command ``UMRR-qchem-run --help`` describes execution of
those file lists with input/output hash records.
