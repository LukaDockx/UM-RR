"""Independent internal coordinates with direct Cartesian construction.

Distances and ordinary bends are expressed in bohr and radians. Invalid
distances or singular angles are rejected before Cartesian construction.
An atom close to a straight line instead has three Cartesian components in a
frame fixed by previously placed atoms. Two components describe its bending;
no dihedral is assigned to that straight line.
"""

from __future__ import annotations

import itertools
from collections import deque

import numpy as np
from geometric.internal import Angle, Dihedral, Distance

from .constants import ANGSTROM_TO_BOHR
from .internal_coordinates import CoordinateDomainError
from .rotors import build_bond_graph

# Numerical margins exclude coincident atoms and undefined reference planes;
# they are not physical repulsive walls or fitted potential parameters.
MIN_DISTANCE = 1.0e-8
MIN_ANGLE = 1.0e-6
MIN_FRAME_SINE = 1.0e-10


class DirectDihedral(Dihedral):
    """Differentiate throughout the nonsingular direct-coordinate domain.

    The optimizer's dihedral derivative zeros terms near a straight angle.
    That approximation would give incorrect kinetic masses at allowed scan
    geometries. Cross-product norms avoid cancellation in 1-cos(angle)**2;
    an actually singular reference plane is rejected instead.
    """

    def derivative(self, xyz):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        a, b, c, d = self.a, self.b, self.c, self.d
        vectors = np.array([xyz[a] - xyz[b], xyz[c] - xyz[b], xyz[d] - xyz[c]])
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= MIN_DISTANCE):
            raise CoordinateDomainError("Coincident dihedral atoms", "nonpositive_distance")
        u, w, v = vectors / lengths[:, None]
        uw, vw = np.cross(u, w), np.cross(v, w)
        sin2u, sin2v = np.dot(uw, uw), np.dot(vw, vw)
        if min(sin2u, sin2v) <= MIN_FRAME_SINE**2:
            raise CoordinateDomainError(
                "Collinear dihedral reference plane", "singular_reference_plane"
            )
        t1 = uw / (lengths[0] * sin2u)
        t2 = vw / (lengths[2] * sin2v)
        t3 = uw * np.dot(u, w) / (lengths[1] * sin2u)
        t4 = vw * np.dot(v, w) / (lengths[1] * sin2v)
        rows = np.zeros_like(xyz)
        rows[a], rows[d] = t1, -t2
        rows[b], rows[c] = -t1 + t3 - t4, t2 - t3 + t4
        return rows


def _length(x):
    # This form also permits complex-step derivatives of a moving frame.
    return np.sqrt(np.sum(x * x, axis=-1))


def _frame(r, a, b, c):
    e1 = r[b] - r[c]
    length = _length(e1)
    if abs(length) <= MIN_DISTANCE:
        raise CoordinateDomainError("Coincident reference atoms", "nonpositive_distance")
    e1 = e1 / length
    e2 = r[a] - r[c]
    e2 = e2 - np.sum(e2 * e1) * e1
    length = _length(e2)
    if abs(length) <= MIN_FRAME_SINE * abs(_length(r[a] - r[c])):
        raise CoordinateDomainError(
            "Collinear atom-placement reference plane", "singular_reference_plane"
        )
    e2 = e2 / length
    return np.array([e1, e2, -np.cross(e1, e2)])


def _sine(points, a, b, c):
    u, v = points[..., a, :] - points[..., b, :], points[..., c, :] - points[..., b, :]
    return _length(np.cross(u, v)) / (_length(u) * _length(v))


class DirectInternals:
    """A Z-matrix with explicit two-direction bends at linear substructures.

    Each requested rotor is one of the independent dihedral coordinates.
    Reference atoms need not be chemically bonded: they define placement, not
    extra bonds or force constants. Fully linear molecules use the separate
    Cartesian normal-coordinate implementation, with 3N-5 vibrations.
    """

    def __init__(self, reference, paths, rotors, symbols=None):

        self.reference = np.asarray(reference, float)
        paths = np.asarray(paths, float).reshape(-1, *self.reference.shape)
        points = np.concatenate(([self.reference], paths))
        # Atom-list order must not choose a different nonlinear displacement
        # path for otherwise unchanged input. Sorted distance signatures are
        # unchanged by rigid motion or passive relabeling. Exactly symmetric
        # atoms can remain tied; the saved placement definition resolves that
        # unavoidable choice and complete potential copies retain symmetry.
        distances = _length(self.reference[:, None] - self.reference[None, :])
        atom_keys = [tuple(np.round(np.sort(row), 12)) for row in distances]
        rotors = [tuple(map(int, row)) for row in rotors]
        labels = None if symbols is None else list(symbols)
        if labels is not None and len(labels) != len(reference):
            raise ValueError("One element label is required per atom")
        graph = (
            build_bond_graph(self.reference / ANGSTROM_TO_BOHR, labels)
            if labels is not None
            else {i: [] for i in range(len(reference))}
        )
        heavy = (
            np.array([s not in {"H", "D", "T"} for s in labels])
            if labels is not None
            else np.zeros(len(reference), bool)
        )
        skeleton = {a for rotor in rotors for a in rotor[1:3]}
        if not skeleton:
            skeleton = set(np.flatnonzero(heavy))
        depth = np.full(len(reference), len(reference), int)
        queue = deque(sorted(skeleton))
        for a in skeleton:
            depth[a] = 0
        while queue:
            a = queue.popleft()
            for b in graph[a]:
                if depth[b] > depth[a] + 1:
                    depth[b] = depth[a] + 1
                    queue.append(b)
        if len({frozenset(t[1:3]) for t in rotors}) != len(rotors):
            raise ValueError("Select only one dihedral per rotor bond")
        # Trying both orientations leaves the molecular atom numbering intact.
        first_choices = [t for rotor in rotors for t in (rotor, rotor[::-1])]
        if not rotors:
            first_choices = sorted(
                itertools.permutations(range(len(reference)), 3),
                key=lambda t: (
                    -float(np.min(_sine(points, *t))),
                    tuple(atom_keys[a] for a in t),
                ),
            )
        for first in first_choices:
            if np.min(_sine(points, *first[:3])) < 0.05:
                continue
            built = list(first[:3])
            pending = set(range(len(rotors)))
            steps = []
            planes = {frozenset(built)}
            while len(built) < len(reference):
                ready = [
                    (j, t)
                    for j in sorted(pending)
                    for t in (rotors[j], rotors[j][::-1])
                    if all(a in built for a in t[:3]) and t[3] not in built
                ]
                if ready:
                    j, t = ready[0]
                    if frozenset(t[:3]) not in planes:
                        break
                    if any(
                        k != j and set(rotors[k]).issubset(set(built) | {t[3]}) for k in pending
                    ):
                        break
                    steps.append(("spherical", t, j))
                    built.append(t[3])
                    pending.remove(j)
                    planes.add(frozenset(t[1:]))
                    continue
                choices = [
                    a
                    for a in range(len(reference))
                    if a not in built
                    and not any(set(rotors[j]).issubset(set(built) | {a}) for j in pending)
                ]
                if not choices:
                    break
                # Build the bonded skeleton before its remaining hydrogens.
                # A hydrogen already placed on a methyl group must not become
                # the rotation axis for a different hydrogen on that group.
                d = min(
                    choices,
                    key=lambda a: (
                        -sum(a in rotors[j] for j in pending),
                        -any(c in graph[a] for c in built),
                        -int(heavy[a]),
                        atom_keys[a],
                    ),
                )
                parents = sorted(
                    built,
                    key=lambda c: (
                        -int(c in graph[d]),
                        -int(heavy[c]),
                        _length(reference[d] - reference[c]),
                        atom_keys[c],
                    ),
                )
                placements = []
                needed = [frozenset(t[:3]) for j in pending for t in (rotors[j], rotors[j][::-1])]
                for priority, c in enumerate(parents):
                    candidates = [
                        (float(np.min(_sine(points, a, c, b))), a, b)
                        for a, b in itertools.permutations(built, 2)
                        if c not in (a, b) and frozenset((a, b, c)) in planes
                    ]
                    for score, a, b in candidates:
                        if score > 0.2:
                            useful = needed.count(frozenset((b, c, d)))
                            # A well-conditioned but chemically remote axis
                            # can turn a rigid methyl rotation into H-H
                            # compression. Prefer the bonded axis toward the
                            # rotor skeleton and a reference on its far side;
                            # conditioning remains a mandatory admissibility
                            # check, then resolves otherwise equal choices.
                            frame_priority = (
                                int(b in graph[c]),
                                int(depth[b] <= depth[c]),
                                int(heavy[b]),
                                int(len(graph[b]) > 1),
                                -int(depth[b]),
                                int(a in graph[b]),
                                int(heavy[a]),
                                -int(depth[a]),
                            )
                            placements.append((useful, -priority, frame_priority, score, a, b, c))
                if not placements:
                    break
                _, _, _, _, a, b, c = max(
                    placements,
                    key=lambda row: (*row[:4], *(atom_keys[a] for a in row[4:])),
                )
                chosen = (a, b, c, d)
                a, b, c, d = chosen
                kind = "cartesian" if np.min(_sine(points, b, c, d)) < 0.15 else "spherical"
                steps.append((kind, chosen, None))
                built.append(d)
                if kind == "spherical":
                    planes.add(frozenset((b, c, d)))
            if len(built) == len(reference) and not pending:
                self.order, self.steps = built, steps
                break
        else:
            raise CoordinateDomainError(
                "Cannot construct independent atom placements with nonsingular reference planes"
            )
        a, b, c = self.order[:3]
        self.definitions = [("length", (a, b)), ("length", (b, c)), ("bend", (a, b, c))]
        self.rotor_positions = np.empty(len(rotors), int)
        for kind, (a, b, c, d), rotor in self.steps:
            if kind == "spherical":
                self.definitions += [
                    ("length", (c, d)),
                    ("bend", (b, c, d)),
                    ("dihedral", (a, b, c, d)),
                ]
            else:
                self.definitions += [(f"component_{k}", (a, b, c, d)) for k in range(3)]
            if rotor is not None:
                self.rotor_positions[rotor] = len(self.definitions) - 1
        self.periodic = np.array([kind == "dihedral" for kind, _ in self.definitions])
        self.names = [f"{kind}:{','.join(map(str, atoms))}" for kind, atoms in self.definitions]
        self.Internals = self.names

    def calculate(self, r):
        r = np.asarray(r, float)
        values = []
        for kind, atoms in self.definitions:
            if kind == "length":
                value = Distance(*atoms).value(r)
                if value <= MIN_DISTANCE:
                    raise CoordinateDomainError(
                        "Nonpositive atom-placement distance", "nonpositive_distance"
                    )
            elif kind == "bend":
                value = Angle(*atoms).value(r)
                if not MIN_ANGLE < value < np.pi - MIN_ANGLE:
                    raise CoordinateDomainError(
                        "Atom-placement bend reaches a linear configuration",
                        "bend_limit",
                    )
            elif kind == "dihedral":
                value = Dihedral(*atoms).value(r)
            else:
                a, b, c, d = atoms
                value = (_frame(r, a, b, c) @ (r[d] - r[c]))[int(kind[-1])]
            values.append(value)
        if not np.isfinite(values).all():
            raise CoordinateDomainError(
                "Internal coordinates require nonzero distances and defined reference planes"
            )
        return np.asarray(values)

    def derivatives(self, r):
        r = np.asarray(r, float)
        rows = []
        self.calculate(r)
        for kind, atoms in self.definitions:
            if kind == "length":
                item = Distance(*atoms)
                row = item.derivative(r)
            elif kind == "bend":
                item = Angle(*atoms)
                row = item.derivative(r)
            elif kind == "dihedral":
                item = DirectDihedral(*atoms)
                row = item.derivative(r)
            else:
                a, b, c, d = atoms
                row = np.zeros_like(r)
                for atom in atoms:
                    for axis in range(3):
                        x = r.astype(complex)
                        x[atom, axis] += 1.0e-30j
                        row[atom, axis] = (_frame(x, a, b, c) @ (x[d] - x[c]))[
                            int(kind[-1])
                        ].imag / 1.0e-30
            rows.append(row)
        return np.asarray(rows)

    def limits(self):
        """Open numerical domains for ordinary lengths and bends."""
        lower = np.full(len(self.definitions), -np.inf)
        upper = np.full(len(self.definitions), np.inf)
        for k, (kind, _) in enumerate(self.definitions):
            if kind == "length":
                lower[k] = MIN_DISTANCE
            elif kind == "bend":
                lower[k], upper[k] = MIN_ANGLE, np.pi - MIN_ANGLE
        return lower, upper

    def validate(self, values):
        lower, upper = self.limits()
        bad = np.flatnonzero((values <= lower) | (values >= upper))
        if len(bad):
            k = bad[0]
            kind, atoms = self.definitions[k]
            reason = "nonpositive_distance" if kind == "length" else "bend_limit"
            raise CoordinateDomainError(
                f"{kind} {atoms} outside its coordinate domain: {values[k]:.9g}", reason
            )

    def geometry(self, values):
        values = np.asarray(values, float)
        if values.shape != (len(self.definitions),) or not np.isfinite(values).all():
            raise ValueError("One finite value is required per independent internal coordinate")
        self.validate(values)

        def polar(distance, bend):
            radius = distance
            cosine, sine = np.cos(bend), np.sin(bend)
            return radius, cosine, sine

        r = np.zeros_like(self.reference)
        a, b, c = self.order[:3]
        r[b, 0] = values[0]
        radius, cosine, sine = polar(values[1], values[2])
        r[c] = r[b] + radius * np.array([-cosine, sine, 0.0])
        for k, (kind, (a, b, c, d), _) in enumerate(self.steps):
            v = values[3 + 3 * k : 6 + 3 * k]
            frame = _frame(r, a, b, c)
            if kind == "spherical":
                radius, cosine, sine = polar(v[0], v[1])
                v = radius * np.array([cosine, sine * np.cos(v[2]), sine * np.sin(v[2])])
            r[d] = r[c] + v @ frame
        if not np.isfinite(r).all():
            raise CoordinateDomainError(
                "Requested displacement exceeds the representable internal-coordinate range"
            )
        return r
