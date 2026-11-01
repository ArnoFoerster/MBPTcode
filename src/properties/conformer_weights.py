"""Conformer populations from free energies: the conformational entropy.

`conformers.search_conformers` weights each minimum by g exp(-E/kT), its
electronic energy and its mirror degeneracy g. Minima a few k_B T apart also
differ in how much room they have: a floppy twist-boat has softer modes than
a stiff chair, a symmetric conformer fewer distinct orientations than a
crooked one. In the rigid-rotor, harmonic-oscillator model each minimum i has

    G_i = E_i + sum_k [ w_k / 2 + kT ln(1 - e^{-w_k / kT}) ]
              - kT ln sqrt(I_A I_B I_C) + kT ln sigma_i - kT ln g_i

(the constants every conformer of one molecule shares dropped): the zero-point
energy and the vibrational free energy of its own modes, the rotational
partition function's moments of inertia and rotational symmetry number, and
the mirror degeneracy. `free_energy_weights` turns them into populations; the
Hessians come from wherever the caller has them (`duschinsky.CartesianHessian`
-- an analytic ground-state Hessian, a file another program wrote), one per
conformer, and nothing here computes one.

The harmonic treatment of the softest modes is the weakest term: a
torsion of 20 cm^-1 contributes kT ln(kT / w) ~ 60 meV at 300 K with an
error of the same order when the mode is really a hindered rotor. The terms
are returned separately so that the spread they cause can be read off.
"""
from dataclasses import replace

import numpy as np
from pyscf import symm

from src.Base.numerics.constants import (AMU_TO_ME,
                                         BOLTZMANN_HARTREE_PER_KELVIN,
                                         CONFORMER_RMSD_TOL,
                                         CONFORMER_TEMPERATURE)
from src.properties.conformers import boltzmann_weights, mirror_degeneracy
from src.properties.duschinsky import harmonic_state

#: Geometric tolerance (Bohr) of the point-group detection behind the
#: rotational symmetry number: a relaxed minimum is symmetric to the
#: optimizer's residual, a few 1e-3 Bohr, not to pyscf's default 1e-5.
SYMMETRY_TOLERANCE = 0.05
#: Rotational symmetry number of each axial point group family; n is the
#: order of the principal axis (`symmetry_number`).
_SIGMA_FIXED = {'C1': 1, 'Ci': 1, 'Cs': 1, 'Coov': 1, 'Dooh': 2, 'T': 12,
                'Td': 12, 'Th': 12, 'O': 24, 'Oh': 24, 'I': 60, 'Ih': 60}


def symmetry_number(mol):
    """(sigma, point group): the number of proper rotations that map `mol`
    onto itself, from its point group (pyscf's detection, all atoms,
    SYMMETRY_TOLERANCE Bohr): n for C_n, C_nv, C_nh, n/2 for S_n, and 2n for
    D_n, D_nd, D_nh."""
    atoms = [(mol.atom_symbol(i), mol.atom_coord(i)) for i in range(mol.natm)]
    old = symm.geom.TOLERANCE
    try:
        symm.geom.TOLERANCE = SYMMETRY_TOLERANCE
        group = symm.detect_symm(atoms)[0]
    finally:
        symm.geom.TOLERANCE = old
    if group in _SIGMA_FIXED:
        return _SIGMA_FIXED[group], group
    n = int(''.join(ch for ch in group if ch.isdigit()) or 1)
    if group.startswith('D'):
        return 2 * n, group
    if group.startswith('S'):
        return n // 2, group
    return n, group


def harmonic_free_energy(omega, temperature):
    """sum_k w_k / 2 + kT ln(1 - e^{-w_k / kT}), Hartree: zero-point energy
    plus vibrational free energy of harmonic modes `omega` (Hartree)."""
    w = np.asarray(omega, float)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    with np.errstate(under='ignore'):
        return float((0.5 * w + kt * np.log1p(-np.exp(-w / kt))).sum())


def log_moments(mol):
    """ln sqrt(I_A I_B I_C) (a.u.) of `mol` with isotope-averaged masses; for
    a linear molecule ln I."""
    m = np.asarray(mol.atom_mass_list(isotope_avg=True)) * AMU_TO_ME
    x = np.asarray(mol.atom_coords(), float)
    x = x - (m[:, None] * x).sum(0) / m.sum()
    inertia = (m * (x ** 2).sum(1)).sum() * np.eye(3) - (m[:, None] * x).T @ x
    ev = np.linalg.eigvalsh(inertia)
    if ev[0] < 1e-6 * ev[-1]:
        return float(np.log(ev[-1]))
    return float(0.5 * np.log(ev).sum())


def free_energy_terms(mol, energy, temperature, hessian=None,
                      degeneracy=None, rmsd_tol=CONFORMER_RMSD_TOL):
    """The terms of G for one minimum (module docstring), Hartree.

    hessian: a `duschinsky.CartesianHessian` (or anything with .mol and
        .hessian) at this minimum; None leaves the vibrational term out --
        then the populations differ from Boltzmann factors of E only by the
        rotational and symmetry terms, which is a statement, not an
        approximation of the vibrations.
    """
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    g = mirror_degeneracy(mol, rmsd_tol=rmsd_tol) if degeneracy is None \
        else int(degeneracy)
    sigma, group = symmetry_number(mol)
    vib = None
    if hessian is not None:
        vib = harmonic_free_energy(harmonic_state(hessian).omega, temperature)
    rot = -kt * log_moments(mol)
    total = (float(energy) + (vib or 0.0) + rot + kt * np.log(sigma)
             - kt * np.log(g))
    return {'energy': float(energy), 'vibrational': vib, 'rotational': rot,
            'symmetry_number': sigma, 'point_group': group,
            'degeneracy': g, 'free_energy': float(total)}


def free_energy_weights(conformers, temperature=CONFORMER_TEMPERATURE,
                        hessians=None):
    """The conformers re-weighted by their free energies.

    conformers: [Conformer] (`search_conformers`). hessians: None, a list
        with one Hessian per conformer, or a callable(conformer) -> Hessian;
        with None, only the rotational and symmetry terms are added to the
        energies (the mirror degeneracy is in both).

    Returns (conformers with `weight` replaced and the terms in
    info['free_energy'], [terms]). The free energies, not the electronic
    energies, then set every average `conformer_average` takes.
    """
    if hessians is None:
        hess = [None] * len(conformers)
    elif callable(hessians):
        hess = [hessians(c) for c in conformers]
    else:
        hess = list(hessians)
        if len(hess) != len(conformers):
            raise ValueError(f'{len(hess)} Hessians for {len(conformers)} '
                             f'conformers')
    if any(h is not None for h in hess) and any(h is None for h in hess):
        raise ValueError('a Hessian for some conformers and not others '
                         'compares free energies with electronic ones')
    terms = [free_energy_terms(c.mol, c.energy, temperature, h, c.degeneracy)
             for c, h in zip(conformers, hess)]
    w = boltzmann_weights([t['free_energy'] for t in terms], temperature)
    out = [replace(c, weight=float(wi),
                   info=dict(c.info, free_energy=t))
           for c, wi, t in zip(conformers, w, terms)]
    return out, terms
