UM-RR: uncoupled modes with a relaxed rotor
=========================================

UM-RR constructs molecular geometries and momenta from a relaxed internal
rotation and independent anharmonic transverse vibrations. The workflow
locates minima along the selected torsion, constructs electronic potential
curves, solves one-dimensional Schrödinger equations, and generates
quasiclassical initial conditions with a fixed energy constraint.

This branch contains the implementation used in the paper: ordinary internal
coordinates, basin assignment at neighboring torsional barriers, and
minimum-specific transverse modes. The source tree includes an input-only
MeOOH example, numerical tests and an explanation of the algorithms.

Installation
------------

Use Python 3.10 or later::

    python -m pip install -e '.[plot]'

NumPy, SciPy, PyYAML, geomeTRIC and rmsd are installed with the package.
Matplotlib supplies the optional potential, wavefunction and density plots.
Electronic calculations require a separate Q-Chem installation available as
``qchem``. The MeOOH input uses frozen-core RI-MP2/aug-cc-pVDZ with the
RIMP2-aug-cc-pVDZ auxiliary basis, as in the paper's Q-Chem 6.2.1 calculations.

Run MeOOH
---------

Check the settings without starting a calculation::

    UMRR --check examples/MeOOH/config.yaml

Run optimization, frequency calculation, relaxed scan, minimum calculations,
transverse scans, quantum solutions and sampling::

    python examples/MeOOH/run.py

The example starts from ``examples/MeOOH/seed.xyz``. Its configuration creates
10,000 conditions at 298 K. The initial optimization, frequency calculation
and relaxed scan run sequentially with eight threads. Minimum calculations
and transverse single points use up to eight concurrent one-thread jobs.
``qchem.num_processors`` controls that concurrent job count; the ``-nt`` flag
in each command controls threads within a job.

All paths are relative to the YAML file. Completed electronic calculations
are reused when the corresponding input and output hashes match their saved
records. Changing only the sample count or random seed therefore reuses the
calculated electronic curves. See ``examples/MeOOH/README.rst`` for execution
controls and the generated directory layout.

For an existing normally terminated frequency calculation::

    UMRR path/to/config.yaml

Read the results
----------------

``generated/initial_conditions.json`` contains the atomic symbols, geometries
in bohr, Cartesian velocities in atomic units, quantum-state labels, energy
budgets, basin assignments and proposal counts. Absolute and relative model
energies are labeled explicitly in its ``energy_schema``.

``generated/basin_model.json`` records the relaxed path, basin boundaries,
coordinate definitions and atom permutations. ``minimum_000.npz``, etc., store
the electronic curves. A stored model can evaluate additional geometries::

    from UMRR.basin_sampler import BasinPotential

    potential = BasinPotential.load('examples/MeOOH/generated')
    relative_energy_hartree = potential.evaluate_geometry(geometry_bohr)
    absolute_energy_hartree = relative_energy_hartree + potential.reference_energy

``generated/neutral_potentials_and_wavefunctions/`` contains compressed numerical
wavefunctions, grids, level energies, masses and boundary diagnostics. When
plotting is enabled, its ``index.html`` displays the electronic curves,
wavefunctions and densities. ``generated/metadata.json`` records the settings,
scan counts, rejection counts and implementation hashes.

Numerical method and code organization
-------------------------------------

``docs/algorithms.rst`` connects the paper's equations to the source. The main
steps are:

1. Optimize the selected dihedral scan and identify its basins and minima.
2. Remove rigid motion and the selected dihedral direction from each minimum's
   Hessian, then express the remaining modes in ordinary internal coordinates.
3. Scan each transverse mode, extending each side according to its calculated
   electronic energy, and interpolate its potential with a cubic spline.
4. Solve independent finite-interval transverse equations and the periodic
   rotor equation in mass-weighted path length.
5. Draw quantum levels, coordinates and Fourier momenta. Redraw proposals
   whose model potential exceeds the selected energy, then uniformly scale
   accepted velocities to supply the remaining kinetic energy.

Coordinate construction rejects nonpositive distances, singular angles and
local dihedral changes outside their domain. Scan reports identify geometric
and energy limits; state diagnostics report probability near finite boundaries.
The transverse equations use masses fixed at their reference geometries.

Development and verification
----------------------------

Install development tools and run the checks::

    python -m pip install -e '.[dev,plot]'
    python -m pytest
    ruff check src tests examples
    ruff format --check src tests examples

Tests use compact geometries and Hessians or analytic potentials. They cover
coordinate reconstruction and derivatives, rigid-body-free momenta, basin
assignment and symmetry, finite-difference levels, thermal rotor truncation,
scan extension, electronic-job reuse, and a complete analytic ensemble run.
``docs/verification.rst`` records the additional comparison with the archived
MeOOH calculations. No Q-Chem calculation is required to run the test suite.

License
-------

MIT; see ``LICENSE``.
