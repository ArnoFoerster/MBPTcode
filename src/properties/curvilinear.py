"""Displacements between two geometries in curvilinear (valence internal)
coordinates, for Huang-Rhys factors on the ground-state normal modes.

The straight Cartesian line between two minima is the wrong path whenever a
group rotates. A methyl or amino group turned by 45 degrees moves its hydrogens
along a chord of the circle they sweep, so the Cartesian difference shortens
every C-H bond by ~(1 - cos 22.5) of its length and projects onto the C-H
stretches at 3000 cm^-1; the harmonic sum rule sum_k S_k omega_k then counts
that artefact as reorganization. Bond lengths, valence angles and dihedrals
change by exactly what the two geometries differ by, so the displacement
measured in them and mapped linearly to Cartesians at the reference puts a
rotation on the torsions and nothing on the stretches.

The map follows Reimers, J. Chem. Phys. 115, 9103 (2001): the normal
coordinates are linear in the internal ones at the geometry where the Hessian
was taken,

    dq = (B M^-1/2)^+ ds,

with B = ds/dx the Wilson B matrix there (Wilson, Decius and Cross, Molecular
Vibrations, 1955) and ds = s(R_b) - s(R_a) the change of the primitive
internals, dihedrals taken modulo 2 pi. Linearizing anywhere else expands the
displacement in another frame than the one the modes live in. Directions the
primitives do not span (the rigid-body ones, a bend through a linear angle) keep
their Cartesian displacement, so a small displacement reproduces the Cartesian
projection exactly.
"""
import itertools

import numpy as np

from src.Base.constants import (CURVILINEAR_LINEAR_ANGLE_DEG,
                                CURVILINEAR_PINV_RCOND)
from src.properties.conformers import molecular_graph
from src.properties.rates import require_vibrational_modes


def primitives(mol):
    """The redundant valence internals of `mol`: (bonds, angles, dihedrals),
    tuples of atom indices; angles above CURVILINEAR_LINEAR_ANGLE_DEG and the
    dihedrals through them are left out."""
    adj = molecular_graph(mol)
    x = np.asarray(mol.atom_coords())
    bonds = sorted((i, j) for i in range(len(adj)) for j in adj[i] if i < j)
    linear = np.radians(CURVILINEAR_LINEAR_ANGLE_DEG)

    def bent(i, j, k):
        return _angle(x[i], x[j], x[k]) < linear

    angles = [(i, j, k) for j in range(len(adj))
              for i, k in itertools.combinations(sorted(adj[j]), 2)
              if bent(i, j, k)]
    dihedrals = [(i, j, k, l) for j, k in bonds
                 for i in sorted(adj[j] - {k}) for l in sorted(adj[k] - {j})
                 if i != l and bent(i, j, k) and bent(j, k, l)]
    return bonds, angles, dihedrals


def _angle(a, b, c):
    """The valence angle a-b-c in radians."""
    u, v = a - b, c - b
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral_frame(x, i, j, k, l):
    """F, G, H and the plane normals A = F x G, B = H x G of i-j-k-l
    (Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996))."""
    f, g, h = x[i] - x[j], x[j] - x[k], x[l] - x[k]
    return f, g, h, np.cross(f, g), np.cross(h, g)


def values(coords, prims):
    """Every primitive at `coords` (natm, 3) Bohr: lengths in Bohr, angles and
    dihedrals in radians."""
    x = np.asarray(coords, float)
    bonds, angles, dihedrals = prims
    out = [np.linalg.norm(x[i] - x[j]) for i, j in bonds]
    out += [_angle(x[i], x[j], x[k]) for i, j, k in angles]
    for q in dihedrals:
        _, g, _, a, b = _dihedral_frame(x, *q)
        out.append(np.arctan2(np.cross(b, a) @ g / np.linalg.norm(g), a @ b))
    return np.array(out)


def difference(s_b, s_a, prims):
    """s_b - s_a with every dihedral change wrapped into [-pi, pi)."""
    d = np.asarray(s_b, float) - np.asarray(s_a, float)
    first = len(prims[0]) + len(prims[1])
    d[first:] = (d[first:] + np.pi) % (2.0 * np.pi) - np.pi
    return d


def wilson_b(coords, prims):
    """B = ds/dx at `coords`, (n_primitives, 3 natm), analytic."""
    x = np.asarray(coords, float)
    bonds, angles, dihedrals = prims
    rows = []

    def row(parts):
        r = np.zeros_like(x)
        for atom, vec in parts:
            r[atom] += vec
        rows.append(r.ravel())

    for i, j in bonds:
        u = (x[i] - x[j]) / np.linalg.norm(x[i] - x[j])
        row([(i, u), (j, -u)])
    for i, j, k in angles:
        u, v = x[i] - x[j], x[k] - x[j]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        u, v = u / lu, v / lv
        cos = u @ v
        sin = np.sqrt(max(1.0 - cos * cos, 0.0))
        di, dk = (cos * u - v) / (lu * sin), (cos * v - u) / (lv * sin)
        row([(i, di), (k, dk), (j, -di - dk)])
    for i, j, k, l in dihedrals:
        f, g, h, a, b = _dihedral_frame(x, i, j, k, l)
        lg, a2, b2 = np.linalg.norm(g), a @ a, b @ b
        di, dl = -lg / a2 * a, lg / b2 * b
        cross = (f @ g) / (a2 * lg) * a - (h @ g) / (b2 * lg) * b
        row([(i, di), (l, dl), (j, -di + cross), (k, -dl - cross)])
    return np.array(rows)


def curvilinear_displacement(mol_ref, mol_a, mol_b, masses):
    """The mass-weighted displacement a -> b, (3 natm,), measured in the
    primitive internals of `mol_ref` and mapped to Cartesians linearly at
    `mol_ref`; the three geometries share one orientation (`align_to`).

    Returns (dq, record): record gives the primitive counts and the largest
    dihedral change in degrees.
    """
    prims = primitives(mol_ref)
    w = np.repeat(np.asarray(masses, float), 3) ** -0.5
    bm = wilson_b(mol_ref.atom_coords(), prims) * w[None, :]
    bm_pinv = np.linalg.pinv(bm, rcond=CURVILINEAR_PINV_RCOND)
    ds = difference(values(mol_b.atom_coords(), prims),
                    values(mol_a.atom_coords(), prims), prims)
    dq_cart = ((np.asarray(mol_b.atom_coords())
                - np.asarray(mol_a.atom_coords())).ravel() / w)
    # what the primitives span comes from them, the rest stays Cartesian
    dq = bm_pinv @ ds + dq_cart - bm_pinv @ (bm @ dq_cart)
    first = len(prims[0]) + len(prims[1])
    record = {'n_bonds': len(prims[0]), 'n_angles': len(prims[1]),
              'n_dihedrals': len(prims[2]),
              'max_dihedral_change_deg': (float(np.degrees(
                  np.abs(ds[first:]).max())) if prims[2] else 0.0)}
    return dq, record


def huang_rhys_curvilinear(mol_ref, mol_a, mol_b, omega, modes, masses):
    """S_k = omega_k dq_k^2 / 2 of the a -> b displacement on the modes of
    `mol_ref` (the geometry of the Hessian), measured in curvilinear
    coordinates; the same return as `vibronic.huang_rhys_from_displacement`.
    """
    dq, _ = curvilinear_displacement(mol_ref, mol_a, mol_b, masses)
    dqk = np.asarray(modes).T @ dq
    require_vibrational_modes(omega, 'huang_rhys_curvilinear')
    return omega * dqk ** 2 / 2.0, dqk
