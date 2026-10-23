"""Which folded ADC(2) root a state model or a force takes, and the flags
when that is not well defined.

A root is followed by the overlap of its singles amplitudes y_ia with the
previous call's, written as an AO transition amplitude T = C_o y C_v^T
(FollowedRoot). Its overlap with a root at another geometry, through the
mixed AO overlap S12 = <mu(R_prev)|nu(R)>,

    <T_prev|T> = sum T_prev (S12 T S12^T),

relative to the two norms, does not depend on the orbitals' phases or on
rotations within the occupied or the virtual space, so a fresh SCF's
arbitrary gauge cannot pick another root (a dot product of MO vectors can).
At one geometry it is the MO dot product.

The best overlap should reach ROOT_FOLLOW_WEIGHT_MIN (else the followed root
is not among the roots solved) and beat the runner-up by
ROOT_FOLLOW_MARGIN_MIN (else two roots have mixed and the pick is decided by
noise). Either is a numerical event (refuse inputs,
warn numerics): the best overlap is taken, warned about (FollowingWarning) and
flagged 'root_following_lost' or 'root_following_mixed'; a root that is not
converged is flagged 'adc_roots_not_converged' (flag_unconverged). A root
that is not there at all -- a solve shorter than the root index asked for --
has no number to take in its place: it is warned about and flagged
'adc_root_missing' (root_missing), and the caller returns NaN for it, which a
walk rejects and retries shorter (properties.optimize.NonFiniteEvaluation).
"""
import warnings

import numpy as np
from pyscf import gto

from src.Base.numerics.constants import ROOT_FOLLOW_MARGIN_MIN, ROOT_FOLLOW_WEIGHT_MIN


class FollowingWarning(RuntimeWarning):
    """A folded ADC(2) root taken with a caveat: not converged, followed
    with an overlap below ROOT_FOLLOW_WEIGHT_MIN or a margin below
    ROOT_FOLLOW_MARGIN_MIN, missing, or a force on another root than the
    call's (module docstring)."""


def flag(flags, name, text):
    """Warn (FollowingWarning) and append {'flag': name, 'message': text} to
    `flags` when it is a list."""
    warnings.warn(text, FollowingWarning, stacklevel=3)
    if flags is not None:
        flags.append({'flag': name, 'message': text})


def flag_unconverged(converged, label, flags=None):
    """Warn about, and append to `flags` as 'adc_roots_not_converged', a
    solve with any root not converged (every root competes for the pick: by
    energy order or by overlap); the roots are kept as solved."""
    conv = np.atleast_1d(np.asarray(converged, bool))
    if not conv.all():
        flag(flags, 'adc_roots_not_converged',
              f'{label}: root(s) {np.flatnonzero(~conv).tolist()} of {conv.size} '
              'not converged; kept as solved')


def root_missing(k, nroots, label, flags=None):
    """Whether root index `k` is missing from a solve that returned `nroots`
    roots; a missing one is warned about and flagged 'adc_root_missing'
    (module docstring)."""
    if 0 <= int(k) < int(nroots):
        return False
    flag(flags, 'adc_root_missing',
         f'{label}: root {k} asked for, the solve returned {nroots}: no value '
         '(NaN) is returned for it; raise nroots or shorten the step')
    return True


def pick_followed(weights, label, flags=None):
    """Index of the best of `weights` (relative overlaps in [0, 1]); below
    ROOT_FOLLOW_WEIGHT_MIN, or within ROOT_FOLLOW_MARGIN_MIN of the runner-up,
    it is still taken, warned about and flagged in `flags`."""
    w = np.asarray(weights, float)
    order = np.argsort(w)[::-1]
    k, best = int(order[0]), float(w[order[0]])
    second = float(w[order[1]]) if w.size > 1 else 0.0
    if best < ROOT_FOLLOW_WEIGHT_MIN:
        flag(flags, 'root_following_lost',
              f'{label}: the followed root overlaps the roots solved here by at most '
              f'{best:.2f} (< {ROOT_FOLLOW_WEIGHT_MIN}): it may not be among them; '
              f'root {k} taken (raise nroots or shorten the step)')
    elif best - second < ROOT_FOLLOW_MARGIN_MIN:
        flag(flags, 'root_following_mixed',
              f'{label}: two roots share the followed one (overlaps {best:.2f} and '
              f'{second:.2f}, margin < {ROOT_FOLLOW_MARGIN_MIN}): they have mixed; '
              f'root {k} taken')
    return k


def transition_amplitudes(mf, frozen, Y):
    """(nroots, nao, nao) AO transition amplitudes C_o y C_v^T of the singles
    columns Y (no * nv, nroots), active occupied o = frozen..nocc."""
    C = np.asarray(mf.mo_coeff, float)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    Co, Cv = C[:, frozen:nocc], C[:, nocc:]
    Y = np.asarray(Y, float).reshape(Co.shape[1], Cv.shape[1], -1)
    return np.einsum('mi,iak,na->kmn', Co, Y, Cv, optimize=True)


class FollowedRoot:
    """A root's AO transition amplitude T on the geometry `mol`, normalized in
    that geometry's AO metric (module docstring)."""

    def __init__(self, mol, T):
        S = mol.intor('int1e_ovlp')
        self.mol = mol
        self.T = np.asarray(T, float) / np.sqrt(float(np.sum(T * (S @ T @ S))))

    def copy(self):
        new = FollowedRoot.__new__(FollowedRoot)
        new.mol, new.T = self.mol, self.T.copy()
        return new

    def weights(self, mol, T_all):
        """|<T_prev|T_k>| / |T_k| for the roots T_all (k, nao, nao) on mol."""
        S12 = gto.intor_cross('int1e_ovlp', self.mol, mol)
        S = mol.intor('int1e_ovlp')
        out = []
        for T in T_all:
            norm = np.sqrt(float(np.sum(T * (S @ T @ S))))
            out.append(abs(float(np.sum(self.T * (S12 @ T @ S12.T)))) / norm)
        return np.array(out)

    def in_orbitals(self, mf, frozen):
        """The amplitude as an MO singles vector (no * nv) of mf at self.mol's
        geometry: C_o^T S T S C_v."""
        C = np.asarray(mf.mo_coeff, float)
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        S = self.mol.intor('int1e_ovlp')
        return (C[:, frozen:nocc].T @ S @ self.T @ S @ C[:, nocc:]).ravel()
