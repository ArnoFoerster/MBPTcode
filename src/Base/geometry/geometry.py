"""Coordinates and bond lengths of a molecule, and the molecule moved to
given Cartesian coordinates (`molecule_at`, `displaced`).

Distances are in Angstrom, because that is how the literature tabulates them;
pyscf holds the coordinates in Bohr.
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
