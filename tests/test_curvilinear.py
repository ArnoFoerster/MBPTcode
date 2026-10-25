"""Huang-Rhys factors of a group rotation measured in curvilinear coordinates.

The Cartesian difference between two geometries that differ by a methyl
rotation is a chord: it shortens the C-H bonds and puts the rotation on the
stretches, so the harmonic sum rule sum_k S_k omega_k counts stretch energy the
molecule never pays. Ethane on a valence force field (stiff stretches and
bends, a soft H-C-C-H torsion), whose Cartesian Hessian B^T K B is exactly
harmonic in the internals, gives the answer in closed form: a rigid 60 degree
rotation of one methyl changes the nine dihedrals by the same angle and
nothing else, so its harmonic energy is 9 k_phi dphi^2 / 2.

No SCF: the force field and its modes are built here.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import AMU_TO_ME
from src.properties.curvilinear import (difference, huang_rhys_curvilinear,
                                        primitives, values, wilson_b)
from src.properties.vibronic import huang_rhys_from_displacement

ETHANE = '''C 0.0000 0.0000 0.7650
C 0.0000 0.0000 -0.7650
H 1.0185 0.0000 1.1580
H -0.5093 0.8821 1.1580
H -0.5093 -0.8821 1.1580
H -1.0185 0.0000 -1.1580
H 0.5093 -0.8821 -1.1580
H 0.5093 0.8821 -1.1580'''
#: Hartree per Bohr^2 (stretch), per rad^2 (bend, torsion)
K_STRETCH, K_BEND, K_TORSION = 0.35, 0.08, 0.002


def ethane():
    return gto.M(atom=ETHANE, basis='sto-3g', verbose=0)


def force_field(mol):
    """Modes of the valence force field at `mol`: (omega, modes, masses, K)."""
    prims = primitives(mol)
    k = np.concatenate([np.full(len(prims[0]), K_STRETCH),
                        np.full(len(prims[1]), K_BEND),
                        np.full(len(prims[2]), K_TORSION)])
    b = wilson_b(mol.atom_coords(), prims)
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True)) * AMU_TO_ME
    w = np.repeat(masses, 3) ** -0.5
    hm = (b * w).T @ np.diag(k) @ (b * w)
    ev, vec = np.linalg.eigh(hm)
    keep = ev > 1e-10
    return np.sqrt(ev[keep]), vec[:, keep], masses, k


def rotated(mol, angle_deg):
    """`mol` with its second methyl turned rigidly about the C-C axis."""
    x = np.asarray(mol.atom_coords()).copy()
    t = np.radians(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                    [0, 0, 1]])
    x[5:] = x[5:] @ rot.T
    out = mol.copy()
    out.set_geom_(x, unit='Bohr')
    out.build(False, False)
    return out


def test_wilson_b_is_the_derivative_of_the_primitives():
    mol = rotated(ethane(), 17.0)
    prims = primitives(mol)
    assert [len(p) for p in prims] == [7, 12, 9]
    x = np.asarray(mol.atom_coords())
    b = wilson_b(x, prims)
    h = 1e-5
    fd = np.zeros_like(b)
    for c in range(x.size):
        xp, xm = x.ravel().copy(), x.ravel().copy()
        xp[c] += h
        xm[c] -= h
        fd[:, c] = difference(values(xp.reshape(-1, 3), prims),
                              values(xm.reshape(-1, 3), prims),
                              prims) / (2 * h)
    assert np.abs(b - fd).max() < 1e-8


def test_a_methyl_rotation_costs_only_its_torsion():
    mol0 = ethane()
    omega, modes, masses, _ = force_field(mol0)
    dphi = np.radians(60.0)
    exact = 9 * K_TORSION * dphi ** 2 / 2
    moved = rotated(mol0, 60.0)
    s_curv, _ = huang_rhys_curvilinear(mol0, mol0, moved, omega, modes,
                                       masses)
    s_cart, _ = huang_rhys_from_displacement(mol0, moved, omega, modes,
                                             masses)
    assert (s_curv * omega).sum() == pytest.approx(exact, rel=1e-8)
    # the chord shortens every C-H and lands on the stiff coordinates
    assert (s_cart * omega).sum() > 5 * exact


def test_a_small_displacement_is_the_cartesian_projection():
    mol0 = ethane()
    omega, modes, masses, _ = force_field(mol0)
    rng = np.random.default_rng(7)
    x = np.asarray(mol0.atom_coords()) + 1e-4 * rng.standard_normal((8, 3))
    moved = mol0.copy()
    moved.set_geom_(x, unit='Bohr')
    moved.build(False, False)
    _, dq_curv = huang_rhys_curvilinear(mol0, mol0, moved, omega, modes,
                                        masses)
    _, dq_cart = huang_rhys_from_displacement(mol0, moved, omega, modes,
                                              masses)
    # second order in the displacement: 1e-4 Bohr against bond lengths ~2
    assert np.abs(dq_curv - dq_cart).max() < 1e-3 * np.abs(dq_cart).max()
