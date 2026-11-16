Algorithms and units
====================

The package implements a sum of independent transverse potentials about a
relaxed torsional path. A basin is the angular interval between neighboring
maxima of the torsional potential. Distinct minima have independent normal
modes and energy curves. Equivalent minima reuse a complete local potential
through an element- and mass-preserving atom permutation.

Workflow
--------

``reference.prepare_reference`` optimizes the supplied XYZ structure and
calculates its Hessian. ``sampler.InitialConditionGenerator`` reads that
frequency output and prepares the relaxed torsion and minimum calculations.
``basin_sampler.BasinSampler.run`` constructs the transverse potentials,
solves the independent equations, samples conditions and writes the results.
The command-line functions call this same Python API.

Geometry and transverse modes
-----------------------------

``direct_internal.DirectInternals`` selects independent atom-placement
coordinates: distances, angles and dihedrals. Three Cartesian components in
a frame defined by earlier atoms handle a nearly linear local arrangement.
It constructs Cartesian positions directly from those coordinates.

``direct_coordinates.DirectInternalCoordinateMap`` removes overall translation
and rotation from the mass-weighted Cartesian Hessian. With a selected rotor,
it also restricts displacements to leave the selected dihedral unchanged to
first order. The projected Hessian supplies 3N-7 transverse modes. Without a
selected rotor it supplies 3N-6 modes for a nonlinear molecule.

Let B be the derivative of the atom-placement coordinates with respect to
Cartesian positions, M the diagonal Cartesian mass matrix, and L the matrix
of projected mass-weighted modes. Then D = B M^(-1/2) L converts the scalar
mode coordinates q into internal-coordinate changes. The inverse of its
transverse block converts a geometry back into q. These matrices are fixed
within each minimum. ``DirectBasinChart`` adds those internal-coordinate
changes to the relaxed geometry at the requested rotor angle.

The reference path is interpolated in internal coordinates. Dihedral winding
is separated from its periodic remainder before interpolation. If a cubic
interpolant would make a distance or angle invalid between scan nodes, that
coordinate uses periodic shape-preserving Hermite slopes. Reference geometries
and derivatives therefore remain continuous and inside their allowed domains.

Kinetic quantities
------------------

For z = (phi, q), define J = dz/dR. The inverse mass matrix is G = J M^(-1) J^T,
and the mass matrix is g = G^(-1). Internal-coordinate gradients remove external
translation and rotation. The reference transverse equations use the diagonal
coefficients mu_i = g_ii evaluated at their undisplaced reference geometry.

The path inertia is g_phi_phi at q=0. ``BasinSampler.prepare_quantum_priors``
evaluates it every 0.5 degrees and integrates sqrt(g_phi_phi) by the trapezoidal
rule. A monotone cubic interpolation gives phi(s). The periodic rotor grid is
uniform in mass-weighted path length s and excludes the repeated endpoint.

All internal calculations use bohr, electron masses, hartree and radians.
The mass-weighted transverse displacements have units sqrt(electron mass)*bohr;
their kinetic coefficients are dimensionless. Atomic-unit hbar is one.
Q-Chem input geometries use angstrom. ``qchem_io`` converts the units of its
printed mass-weighted Hessian explicitly when forming the internal mode model.

Electronic curves
-----------------

``scan_domains.build_energy_guided_scans`` starts with the configured odd
number of points, including q=0. An even requested count gains a central point.
Each initial extent corresponds to the configured harmonic energy. Both sides
then grow independently until their electronic energy rise reaches that target
or a geometric, extension-count or harmonic-energy limit stops the scan.
Previously computed knots are retained. All modes at one minimum use one
common electronic reference point.

``internal_sampler.InternalPotential`` stores cubic splines U_i(q_i), with
U_i(0)=0. ``BasinPotential`` evaluates V_rot(phi)+sum_i U_bi(q_i) using the basin
bounded by the torsional barriers. Absolute electronic reference energies are
stored separately and added only when requested.

Quantum states and proposals
----------------------------

``schrodinger.solve_schrodinger_1d`` diagonalizes the three-point, tridiagonal
finite-interval Hamiltonian using a constant reference mass. Zero ghost values
immediately beyond the stored grid define its boundary. The wavefunctions are
normalized with the uniform grid spacing. ``state_boundaries`` checks their
endpoint density and probability in the outer five percent of each interval.

``relaxed_torsion.solve_periodic_path_schrodinger`` supplies the periodic
three-point operator. ``rotor_states.solve_thermal_rotor_states`` increases its
state count until a bound on omitted population meets the requested tolerance,
retaining complete numerically degenerate groups. This bound concerns the
finite spatial grid's state count.

Rotor levels are selected by their normalized Boltzmann weights. The rotor
position is drawn from abs(chi(s))^2, and phi(s) selects the minimum. Each of
that minimum's transverse levels is then chosen by its own Boltzmann weight,
and q_i is drawn from abs(psi_i(q_i))^2. At zero temperature the lowest levels
are selected, retaining numerical degeneracies.

``momentum`` constructs squared Fourier amplitudes. Transverse momenta use
zero ghost points and zero padding for their continuous inverse cumulative
distributions. Rotor momenta remain discrete, p_s=2*pi*k/L; p_phi=sqrt(g)*p_s.
Coordinate and momentum draws are independent before the fixed energy step.

Fixed energy constraint
-----------------------

``BasinSampler.sample`` assigns E_target=E_rot+sum_i epsilon_i. A proposal is
redrawn if its combined coordinate displacement is invalid or if
K_target=E_target-V_model is negative. Every rejection repeats the quantum-state,
coordinate and momentum choices.

Cartesian velocities are v0=M^(-1) J^T p, using the derivatives at the proposed
geometry. One scalar sqrt(K_target/K0) multiplies all velocities. The accepted
geometry is retained and its kinetic energy becomes K_target. Internal
coordinate gradients give zero total linear and angular momentum, which the
uniform scaling preserves. Each accepted point therefore has model energy
E_target.

Source map
----------

* ``reference.py``: equilibrium geometry and frequency preparation;
* ``sampler.py``: relaxed scan/minimum orchestration and result structure;
* ``relaxed_torsion.py``: scan parsing, path geometry, minima and equivalence;
* ``direct_internal.py``: ordinary internal coordinates and direct atom placement;
* ``direct_coordinates.py``: projected modes and minimum-specific displacement maps;
* ``internal_coordinates.py``: mass matrices, velocity conversion and domains;
* ``scan_domains.py``: asymmetric electronic scan extension;
* ``internal_sampler.py``: independent curves, job execution and common workflow steps;
* ``basin_sampler.py``: basin potentials, quantum distributions and ensemble generation;
* ``schrodinger.py`` / ``rotor_states.py``: one-dimensional quantum solutions;
* ``momentum.py``: Fourier momentum distributions;
* ``state_boundaries.py`` / ``neutral_plots.py``: numerical state files and diagnostics;
* ``qchem_io.py`` / ``qchem_runner.py``: Q-Chem parsing, inputs and completed-job checks.
