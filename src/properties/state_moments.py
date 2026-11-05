"""Transition dipoles between two excited states of one singles solve.

An excited molecule absorbs a second photon through <m|r|n>, the dipole between
two excited states. Förster annihilation needs it (`src.photophysics.forster`):
the acceptor already carries an exciton, and its absorption spectrum is that of
T1 -> Tn or S1 -> Sn. For states written in single excitations out of one
closed-shell determinant,

    |n> = sum_ia X^n_ia |Phi_ia>,

with |Phi_ia> the spin-adapted singlet or the M_s = 0 triplet configuration and
<X^n|X^n> = 1 in the spatial-orbital normalization of the solvers, the
Slater-Condon rules give

    <m|r|n> = sum_iab X^m_ia X^n_ib r_ab - sum_ija X^m_ia X^n_ja r_ji
              + delta_mn <0|r|0>,

the same contraction for two singlets and for two triplets of one M_s (the
dipole is spin free, so it never connects the two manifolds). It is the
virtual-virtual and occupied-occupied blocks of the state-to-state density
`gradients.gwbse.derivative_coupling.state_to_state_density` with Y = 0. The
ground-state term is left out: it is zero between orthogonal states, and on the
diagonal the result is the unrelaxed difference dipole.

The formula is exact for Tamm-Dancoff amplitudes (CIS, BSE-TDA) and for any
other method whose states are given by their singles part on the same
orbitals, as the leading term. A full (X, Y) solve has no wavefunction of this
form: its excited-to-excited moment is the residue of the quadratic response
function, which is not built here, so a nonzero Y is refused rather than
dropped.

Ground-to-excited moments of any method that supplies a transition density
are `properties.transition_moments`; this module adds the excited-to-excited
ones from singles amplitudes, on the same orbital dipole matrix.
"""
import numpy as np

from src.properties.transition_moments import dipole_mo


def excited_transition_dipoles(mol, mo_coeff, nocc, x_from, x_to,
                               y_from=None, y_to=None):
    """<m|r|n> for every m of `x_from` and n of `x_to`, atomic units.

    mo_coeff: the (nao, nmo) orbitals the amplitudes refer to; the first nocc
        are occupied. The dipole origin is fixed at zero: between orthogonal
        states the moment does not depend on it.
    x_from, x_to: amplitudes of shape (nocc * nvir, n_states) or
        (nocc * nvir,), occupied-major as the solvers return them, each
        column normalized to one; both of one spin manifold.
    y_from, y_to: the de-excitation amplitudes, if the solve returned any.
        They must be zero (see the module docstring).

    Returns an array of shape (n_from, n_to, 3).
    """
    mo = np.asarray(mo_coeff, float)
    nmo = mo.shape[1]
    nvir = nmo - int(nocc)
    xm, xn = (_amplitudes(x, nocc, nvir, name)
              for x, name in ((x_from, 'x_from'), (x_to, 'x_to')))
    for y, name in ((y_from, 'y_from'), (y_to, 'y_to')):
        if y is not None and np.any(np.asarray(y, float) != 0.0):
            raise ValueError(
                f'{name} is not zero: a full (X, Y) solve has no '
                'excited-to-excited moment of this single-excitation form '
                '(it is a quadratic-response residue); solve in the '
                'Tamm-Dancoff approximation')
    d = dipole_mo(mol, mo)
    d_oo, d_vv = d[:, :nocc, :nocc], d[:, nocc:, nocc:]
    # sum_ab X^m_ia r_ab X^n_ib - sum_ij X^m_ia r_ji X^n_ja
    vv = np.einsum('mia,xab,nib->mnx', xm, d_vv, xn, optimize=True)
    oo = np.einsum('mia,xji,nja->mnx', xm, d_oo, xn, optimize=True)
    return vv - oo


def _amplitudes(x, nocc, nvir, name):
    """(n_states, nocc, nvir) from the solvers' (nocc * nvir, n_states)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != nocc * nvir:
        raise ValueError(f'{name} has {x.shape[0]} rows; the orbitals give '
                         f'nocc * nvir = {nocc * nvir}')
    norms = np.einsum('pn,pn->n', x, x)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f'{name} columns are not normalized to one '
                         f'(norms {norms.min():.6g} .. {norms.max():.6g}); '
                         'the spatial-orbital convention has <X|X> = 1')
    return x.T.reshape(-1, nocc, nvir)
