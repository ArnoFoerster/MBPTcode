"""Internal coordinates of a molecule, and one way to displace along them.

Structural comparisons are made in these, not in Cartesians: a relaxed geometry
is reported as bond lengths and angles, two methods are compared by the same,
and an excited-state minimum is recognised by an out-of-plane angle that the
ground state does not have. All angles are in degrees and all distances in
Angstrom, because that is how the literature tabulates them.
"""
import numpy as np

from src.Base.constants import BOHR_TO_ANGSTROM


def coordinates(mol):
    """(natm, 3) in Angstrom; pyscf holds them in Bohr."""
    return np.asarray(mol.atom_coords()) * BOHR_TO_ANGSTROM


def bond(mol, i, j):
    """Distance i-j in Angstrom."""
    c = coordinates(mol)
    return float(np.linalg.norm(c[i] - c[j]))


def angle(mol, i, j, k):
    """Angle i-j-k in degrees, j the vertex."""
    c = coordinates(mol)
    u, v = c[i] - c[j], c[k] - c[j]
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def dihedral(mol, i, j, k, l):
    """Signed i-j-k-l dihedral in degrees."""
    x = coordinates(mol)
    b0, b1, b2 = x[i] - x[j], x[k] - x[j], x[l] - x[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def out_of_plane(mol, c, m, s1, s2):
    """Angle in degrees between the c-m bond and the plane through (c, s1, s2).

    Zero for a planar centre. This is the coordinate an n -> pi* excitation
    pyramidalizes along, so it is what separates the excited-state minimum of a
    carbonyl from its ground state.
    """
    x = coordinates(mol)
    n = np.cross(x[s1] - x[c], x[s2] - x[c])
    n /= np.linalg.norm(n)
    d = x[m] - x[c]
    d /= np.linalg.norm(d)
    return float(90.0 - np.degrees(np.arccos(np.clip(abs(n @ d), -1.0, 1.0))))


def pyramidalize(mol, c, m, s1, s2, degrees):
    """Tilt the c-m bond `degrees` out of the (c, s1, s2) plane; a new Mole.

    A PLANAR CENTRE IS A STATIONARY POINT of the out-of-plane coordinate by
    symmetry, so the gradient along it vanishes there and an optimization
    started planar can never pyramidalize however much the state wants to. It
    converges to a saddle and reports it as a minimum. Displacing first is what
    makes the search able to find the real one.
    """
    x = mol.atom_coords().copy()
    axis = x[s2] - x[s1]
    axis /= np.linalg.norm(axis)
    v = x[m] - x[c]
    th = np.radians(degrees)
    # Rodrigues. The axis lies in the plane, so the rotation angle IS the angle
    # the bond ends up making with it.
    v_rot = (v * np.cos(th) + np.cross(axis, v) * np.sin(th)
             + axis * (axis @ v) * (1.0 - np.cos(th)))
    x[m] = x[c] + v_rot
    out = mol.copy()
    out.set_geom_(x, unit='Bohr')
    out.build(False, False)
    return out
