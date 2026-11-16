Verification of the paper branch
================================

Checked on 13 November 2026. Electronic energies were reused from the completed
MeOOH calculations. The checks below started no electronic jobs.

Numerical equivalence
---------------------

The refactored source and the research checkout independently loaded the same
archived ordinary-coordinate MeOOH model. Each evaluated the same 4,000
geometries from four ensembles, for 16,000 evaluations in total. Maximum
absolute differences in the relative potential were:

* UM-RR, 298 K: 1.39 x 10^-17 hartree;
* Wigner, 0 K: 2.78 x 10^-17 hartree;
* Wigner*, 0 K: 2.78 x 10^-17 hartree;
* Wigner, 298 K: 2.78 x 10^-17 hartree.

Both implementations then independently solved the 28 transverse equations
and the periodic rotor using 5,000-point grids, 20 starting rotor states, a
one-percent omitted-population tolerance and 298 K. They generated 128
conditions with random seed 61015. Rotor energies, rotor probability densities,
all retained transverse energies, quantum labels, proposal counts and Cartesian
geometries agreed exactly. Cartesian velocities differed by at most
1.74 x 10^-18 atomic units and kinetic energies by 2.09 x 10^-17 hartree.

These comparisons check the preservation of the published numerical method
through the refactoring. They reuse completed electronic curves; they do not
repeat electronic-structure calculations with another Q-Chem installation.

Automated checks
----------------

The test suite passes 62 tests using compact numerical inputs and analytic
potentials. It includes a complete scan-to-ensemble calculation for an analytic
water potential, model saving and loading, and conservation of the assigned
model energy with zero net translation and angular momentum. The MeOOH YAML
check and optimization-input preparation run without any electronic outputs.
Q-Chem execution tests use a small local stand-in command to check job failures
and input/output hash matching.

The source, tests and example pass Ruff's formatting and configured code checks.
A Python wheel builds successfully and includes the default YAML settings.
The wheel contains no calculation outputs.
