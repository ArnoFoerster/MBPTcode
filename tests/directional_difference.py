"""A finite difference of an energy along one seeded direction over every
nuclear coordinate, for the fast twins of the force-follows-energy gates.

A per-component difference costs four energies per component; this one costs
four in all and sees every atom. The price is sensitivity per component: with
a unit direction d, an error e in one component shows as d_i e, and |d_i| is
about 1/sqrt(3 natm). The twins keep the original bars, and the nightly files
they shadow keep the per-component gates.
"""
import numpy as np


def direction(mol, seed=11):
    """A seeded unit vector over the (natm, 3) coordinates."""
    d = np.random.default_rng(seed).standard_normal((mol.natm, 3))
    return d / np.linalg.norm(d)


def displaced(mol, d, s):
    """`mol` moved by s * d, Bohr."""
    return mol.set_geom_(mol.atom_coords() + s * d, unit='Bohr', inplace=False)


def along(force, d):
    """The analytic derivative along `d`: sum_i d_i dE/dR_i."""
    return float(np.sum(np.asarray(force) * d))


def difference(energy, mol, d, h):
    """The 4-point central difference of energy(mol) along `d` at step h."""
    e = {k: energy(displaced(mol, d, k * h)) for k in (2, 1, -1, -2)}
    return (8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)


def richardson(energy, mol, d, steps):
    """(limit, truncation): the Richardson limit of 2-point central
    differences along `d` at two steps, and the finer step's h^2 error."""
    diff = {}
    for h in steps:
        diff[h] = (energy(displaced(mol, d, h))
                   - energy(displaced(mol, d, -h))) / (2.0 * h)
    h1, h2 = steps
    ratio = (h1 / h2) ** 2
    return ((ratio * diff[h2] - diff[h1]) / (ratio - 1.0),
            abs(diff[h1] - diff[h2]) / (ratio - 1.0))
