"""Electric transition dipoles and oscillator strengths from one-particle
transition densities, whatever method supplied them.

    mu^n = sum_pq <p|r|q> gamma^0n_pq,    f_n = (2/3) omega_n |mu^n|^2

gamma^0n_pq = <Psi_0|a^dag_p a_q|n> is the CHARGE transition density (alpha
plus beta) in the molecular orbitals C; the electron's charge is left out, so
mu is the moment of the density, as `BSE.solve.oscillator_strengths` returns
it for Casida vectors (whose density is sqrt 2 (X + Y) on the
occupied-virtual block). Length gauge; the origin is fixed at zero only for
definiteness, a transition moment being origin-independent.
"""
import numpy as np

from src.properties import spin_orbit


def dipole_mo(mol, C):
    """(3, nmo, nmo) <p|r|q> of the orbitals C, origin at zero."""
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao = mol.intor_symmetric('int1e_r', comp=3)
    C = np.asarray(C, float)
    return np.einsum('xmn,mp,nq->xpq', ao, C, C, optimize=True)


def transition_dipoles(mol, C, gammas):
    """(n, 3) mu^n of the charge transition densities `gammas` (each
    (nmo, nmo) in the orbitals C)."""
    r = dipole_mo(mol, C)
    return np.array([np.einsum('xpq,pq->x', r, np.asarray(g, float))
                     for g in gammas]).reshape(-1, 3)


def oscillator_strengths(omega, dipoles):
    """(n,) f_n = (2/3) omega_n |mu^n|^2, omega in Hartree."""
    dipoles = np.asarray(dipoles, float).reshape(-1, 3)
    return (2.0 / 3.0) * np.asarray(omega, float) * np.einsum('nx,nx->n', dipoles,
                                                              dipoles)


def spectrum_properties(spectrum, n_states, two_electron='somf',
                        one_electron='x2c'):
    """The rate layer's ingredients of one geometry from a spectrum that
    supplies transition densities -- any object with `mol`, `mf`, `omega`
    ({'singlet', 'triplet'}: (n,) Hartree), `flags` (its named numerical
    flags, a list) and `transition_densities(pairs)` -> {(bra, ket):
    (charge, spin)} in the MO basis of mf (bra ('ground', 0) or
    ('singlet', i), ket (spin, k)), as `gradients.adc.ee_spectrum`'s, all
    pairs asked in one call:

        socme   (n_states + 1, n_states) |<S_I|H_SO|T_J>|, row 0 the ground
                state (`spin_orbit.socme_table_from_densities`)
        dipole  (n_states, 3) <0|r|S_I>; f_osc (n_states,) their oscillator
                strengths
        h_mo    the spin-orbit operator in the MO basis
        flags   the spectrum's flags, for the record (empty when clean)
    """
    h_mo = spin_orbit.soc_operator_mo(spectrum.mf, spectrum.mol, two_electron,
                                      one_electron)
    ground, singlets = ('ground', 0), [('singlet', i) for i in range(n_states)]
    triplets = [('triplet', j) for j in range(n_states)]
    pairs = ([(ground, t) for t in triplets] + [(s, t) for s in singlets for t in triplets]
             + [(ground, s) for s in singlets])
    gam = spectrum.transition_densities(pairs)
    table = spin_orbit.socme_table_from_densities(
        h_mo, [gam[ground, t][1] for t in triplets],
        [[gam[s, t][1] for t in triplets] for s in singlets])
    dipole = transition_dipoles(spectrum.mol, spectrum.mf.mo_coeff,
                                [gam[ground, s][0] for s in singlets])
    return {'socme': table,
            'dipole': dipole,
            'f_osc': oscillator_strengths(spectrum.omega['singlet'][:n_states], dipole),
            'h_mo': h_mo,
            'flags': list(spectrum.flags)}
