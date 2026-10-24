"""Internal coordinates of a molecule, one way to displace along them, and
the molecule moved to given Cartesian coordinates (`molecule_at`).

Structural comparisons are made in these, not in Cartesians: a relaxed geometry
is reported as bond lengths and angles, two methods are compared by the same,
and an excited-state minimum is recognised by an out-of-plane angle that the
ground state does not have. All angles are in degrees and all distances in
Angstrom, because that is how the literature tabulates them.
"""
import numpy as np

from src.Base.numerics.constants import BOHR_TO_ANGSTROM


def molecule_at(template, coords):
    """`template` moved to `coords` (natm, 3) in Bohr, rebuilt."""
    mol = template.copy()
    mol.set_geom_(np.asarray(coords, float).reshape(-1, 3), unit='Bohr')
    mol.build(False, False)
    return mol


def displaced(mol, delta, coords=None):
    """`mol` rebuilt at coords + delta (Bohr), coords its own where None: the
    one displaced molecule of every finite difference."""
    base = mol.atom_coords() if coords is None else coords
    return molecule_at(mol, np.asarray(base, float) + delta)


def coordinates(mol):
    """(natm, 3) in Angstrom; pyscf holds them in Bohr."""
    return np.asarray(mol.atom_coords()) * BOHR_TO_ANGSTROM


def elements_of(mol):
    """The element symbols of `mol`, sorted, each once."""
    return sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})


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
    return molecule_at(mol, x)
