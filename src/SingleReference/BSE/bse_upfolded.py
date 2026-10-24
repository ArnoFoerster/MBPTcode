"""Upfolded (frequency-free) dynamical Bethe-Salpeter equation.

Bintrim & Berkelbach, J. Chem. Phys. 156, 044114 (2022):
"Full-frequency dynamical Bethe-Salpeter equation without frequency and a
study of double excitations". Equation numbers below are that paper's.

The BSE from the GW self-energy is a frequency-DEPENDENT eigenvalue problem,

    A(Omega) X = Omega X,    A(Omega) = A - K^(p)(Omega)                  (1)

because the screened interaction W(omega) is dynamic. Solving it exactly by
sum over states (6) needs every RPA excitation, i.e. a full O(N^6)
diagonalization. The paper's observation is that the frequency dependence is
a sum of simple poles, so it can be UPFOLDED: a frequency-independent matrix
in a larger space of single AND double excitations whose eigenvalues are
exactly those of (1),

            /   A      -V^e   -V^h \\
    H  =    | (V^h)^T    D      0  |                                      (7)
            \\ (V^e)^T    0      D  /

    A_ia,jb   = (E_a - E_i) d_ij d_ab + kappa (ia|jb) - (ij|ab)          (1b)
    D         = [-E_occ] (+) E_vir (+) S                                  (8a)
    V^h_ia,ldkc = sqrt(2) (il|kc) d_ad                                    (8b)
    V^e_ia,ldkc = sqrt(2) (kc|ad) d_il                                    (8c)
    S_kc,ia   = (eps_c - eps_k) d_ki d_ca + 2 (kc|ia)      [direct RPA, TDA]

kappa = 2 for a singlet, 0 for a triplet. `E` are GW QUASIPARTICLE energies;
`eps` are the mean-field ones that build the screening. Only A carries kappa:
the doubles blocks and the couplings are direct-only (no exchange), which is
what makes them the RPA screening rather than a correlation ladder.

**This is not ADC.** The structure is ADC-like -- an upfolded static matrix
over singles + doubles, so that downfolding (9) returns the frequency
dependence -- and the paper notes the resemblance itself. But H is
ASYMMETRIC (the (1,2) blocks are -V^e/-V^h while the (2,1) blocks are
+(V^h)^T/+(V^e)^T), it carries TWO distinct doubles sectors rather than one,
its doubles-doubles block is the direct RPA matrix instead of an
antisymmetrized 2p2h block, and it needs correlated GW eigenvalues as input.
The paper's own Table I benchmarks it AGAINST ADC(2)-s/ADC(2)-x/ADC(3) as a
different method. `build_hamiltonian_familiar` below is the symmetric,
mean-field variant the paper mentions (12); the paper reports it lands 2-3 eV
low, and it is here for study, not production.

Vector layout, flat:

    r1 (no*nv) | r2 (no*nv*no*nv) | r2bar (no*nv*no*nv)

with the doubles stored as [l, d, k, c]: the OUTER particle-hole pair (l, d)
carries the quasiparticle energies, the INNER pair (k, c) is the screening
excitation that S acts on.
"""
import warnings

import numpy as np
from pyscf import scf
from scipy import linalg

from src.Base.numerics.constants import (HARTREE_TO_EV, KAPPA, UPFOLDED_BSE_DAVIDSON_MAX_CYCLE,
                                         UPFOLDED_BSE_DAVIDSON_TOL, UPFOLDED_BSE_DENSE_LIMIT,
                                         UPFOLDED_BSE_IMAG_TOL, UPFOLDED_BSE_LINDEP_TOL,
                                         UPFOLDED_BSE_PRECOND_FLOOR)
from src.Base.pyscf.pyscf_interface import (DFIntegrals, get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.step_failed import StepFailed
from src.Base.utils.linear_algebra.orthogonalization import extend_orthonormal
from src.SingleReference.common.eri_blocks import as_blocks, g_slice_df
from src.SingleReference.GW.qp_energy import calc_qp_energy

SQRT2 = np.sqrt(2.0)


class UpfoldedBSERefusal(StepFailed):
    """The upfolded-BSE Davidson ended without converged roots
    (refuse_unconverged=True)."""


def _complex_roots(theta, imag, imag_tol, scale, X, flags, what):
    """Set the roots whose imaginary part exceeds imag_tol * scale to NaN
    (value and vector), warn and flag 'upfolded_bse_complex_roots': the
    dynamical BSE has no real root there, and the
    real part is not one, so no number is returned in its place."""
    theta = np.array(theta, float)
    bad = np.abs(np.asarray(imag)) > imag_tol * scale
    if bad.any():
        message = (f'{int(bad.sum())} of the {len(theta)} lowest upfolded-BSE '
                   f'{what} are complex (max |Im| = {np.abs(imag).max():.2e} Ha, '
                   f'above {imag_tol:.0e} relative): the dynamical BSE has no real '
                   f'solution there; root(s) {np.flatnonzero(bad).tolist()} are '
                   'returned as NaN')
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        if flags is not None:
            flags.append({'flag': 'upfolded_bse_complex_roots', 'message': message})
        X = np.array(X, float)
        theta[bad] = np.nan
        X[:, bad] = np.nan
    return theta, X


def dimensions(no, nv):
    n_s = no * nv
    n_d = no * nv * no * nv
    return {'no': no, 'nv': nv, 'n_s': n_s, 'n_d': n_d, 'nH': n_s + 2 * n_d}


def _slices(d):
    n_s, n_d = d['n_s'], d['n_d']
    return (slice(0, n_s), slice(n_s, n_s + n_d),
            slice(n_s + n_d, n_s + 2 * n_d))


def eri_blocks(eri_chemist, no, norb):
    """The four chemist blocks the equations touch, (pq|rs) ordering:
    (ia|jb) and (kc|ia), (ij|ab), (il|kc) and (kc|ad), read off the route's
    one block set (`common.eri_blocks`): ooov and ovvv are slices of its
    (pq|ia)."""
    b = as_blocks(eri_chemist, no)
    return {'ovov': b.ovov, 'oovv': b.oovv, 'ooov': b.pqov[:no, :no],
            'ovvv': b.pqov[no:norb, no:norb].transpose(2, 3, 0, 1)}


def df_eri_blocks(B, no, norb):
    """The same four blocks from a three-index factor, (pq|rs) = sum_P
    B[P,p,q] B[P,r,s]: `g_slice_df`'s <pr|qs>, transposed."""
    o, v = slice(0, no), slice(no, norb)

    def blk(a, b, c, d):
        return g_slice_df(B, a, c, b, d).transpose(0, 2, 1, 3)

    return {'ovov': blk(o, v, o, v), 'oovv': blk(o, o, v, v),
            'ooov': blk(o, o, o, v), 'ovvv': blk(o, v, v, v)}


def to_blocks(vec, no, nv):
    d = dimensions(no, nv)
    s1, s2, s3 = _slices(d)
    vec = np.asarray(vec).ravel()
    return (vec[s1].reshape(no, nv),
            vec[s2].reshape(no, nv, no, nv),
            vec[s3].reshape(no, nv, no, nv))


def from_blocks(r1, r2, r2bar):
    return np.concatenate([r1.ravel(), r2.ravel(), r2bar.ravel()])


# ----------------------------------------------------------------------
# the blocks
# ----------------------------------------------------------------------

def block_A(e_qp, gb, no, nv, kappa):
    """A_ia,jb, Eq. (1b) -- the only block that knows about spin."""
    eo, ev = e_qp[:no], e_qp[no:]
    A = np.zeros((no, nv, no, nv))
    idx_o, idx_v = np.arange(no), np.arange(nv)
    A[idx_o[:, None], idx_v[None, :], idx_o[:, None], idx_v[None, :]] = \
        ev[None, :] - eo[:, None]
    A = A + kappa * gb['ovov']
    A = A - gb['oovv'].transpose(0, 2, 1, 3)          # (ij|ab) -> [i,a,j,b]
    return A


def block_S(eps, gb, no, nv):
    """S_kc,ia = (eps_c - eps_k) d_ki d_ca + 2 (kc|ia): the direct RPA matrix
    in the TDA. MEAN-FIELD energies -- this is the screening, not the
    quasiparticle problem."""
    eo, ev = eps[:no], eps[no:]
    S = np.zeros((no, nv, no, nv))
    idx_o, idx_v = np.arange(no), np.arange(nv)
    S[idx_o[:, None], idx_v[None, :], idx_o[:, None], idx_v[None, :]] = \
        ev[None, :] - eo[:, None]
    return S + 2.0 * gb['ovov']


def doubles_diagonal(eps, e_qp, no, nv):
    """The FULL diagonal of D, (E_d - E_l) + (eps_c - eps_k).

    This is what Eq. (10b) writes explicitly, because the matvec adds only
    the OFF-diagonal 2(kc|ia) part of S on top of it. The dense builder must
    not use both this and the full S -- (eps_c - eps_k) lives inside S too,
    and adding both counts it twice (a ~1.5e-2 break in the downfolding
    identity, which is how it was caught). Use `outer_diagonal` there."""
    return (outer_diagonal(e_qp, no, nv)[:, :, None, None]
            + inner_diagonal(eps, no, nv)[None, None, :, :])


def outer_diagonal(e_qp, no, nv):
    """E_d - E_l over the quasiparticle pair, [l, d]."""
    Eo, Ev = e_qp[:no], e_qp[no:]
    return Ev[None, :] - Eo[:, None]


def inner_diagonal(eps, no, nv):
    """eps_c - eps_k over the screening pair, [k, c]. Already inside S."""
    eo, ev = eps[:no], eps[no:]
    return ev[None, :] - eo[:, None]


def _build_D(eps, e_qp, S, no, nv):
    """D = [-E_occ] (+) E_vir (+) S, Eq. (8a), as a dense n_d x n_d matrix.

    A Kronecker SUM: the outer (l, d) pair contributes only E_d - E_l, and
    everything about the screening pair -- including its own eps_c - eps_k
    diagonal -- comes from S."""
    n_s = no * nv
    outer = np.repeat(outer_diagonal(e_qp, no, nv).ravel(), n_s)
    return np.diag(outer) + np.kron(np.eye(n_s), S)


def block_V(gb, no, nv):
    """(V^h, V^e) as [i,a,l,d,k,c] arrays, Eqs. (8b) and (8c).

    Dense only -- the matrix-free path never forms these."""
    ooov, ovvv = gb['ooov'], gb['ovvv']
    eye_v, eye_o = np.eye(nv), np.eye(no)
    # V^h_ia,ldkc = sqrt(2) (il|kc) delta_ad
    Vh = SQRT2 * np.einsum('ilkc,ad->ialdkc', ooov, eye_v, optimize=True)
    # V^e_ia,ldkc = sqrt(2) (kc|ad) delta_il
    Ve = SQRT2 * np.einsum('kcad,il->ialdkc', ovvv, eye_o, optimize=True)
    return Vh, Ve


def build_hamiltonian(eps, e_qp, gb, no, nv, spin='singlet'):
    """The dense H of Eq. (7). ASYMMETRIC by construction."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    d = dimensions(no, nv)
    n_s, n_d, n = d['n_s'], d['n_d'], d['nH']
    A = block_A(e_qp, gb, no, nv, kappa).reshape(n_s, n_s)
    S = block_S(eps, gb, no, nv).reshape(n_s, n_s)
    Vh, Ve = block_V(gb, no, nv)
    Vh = Vh.reshape(n_s, n_d)
    Ve = Ve.reshape(n_s, n_d)
    # D: the (E_d - E_l + eps_c - eps_k) diagonal, plus S acting on the
    # INNER (k, c) pair only -- with the [l,d,k,c] flattening that is
    # I_(ld) (x) S_(kc).
    D = _build_D(eps, e_qp, S, no, nv)

    H = np.zeros((n, n))
    H[:n_s, :n_s] = A
    H[:n_s, n_s:n_s + n_d] = -Ve
    H[:n_s, n_s + n_d:] = -Vh
    H[n_s:n_s + n_d, :n_s] = Vh.T
    H[n_s + n_d:, :n_s] = Ve.T
    H[n_s:n_s + n_d, n_s:n_s + n_d] = D
    H[n_s + n_d:, n_s + n_d:] = D
    return H


# ----------------------------------------------------------------------
# the two oracles
# ----------------------------------------------------------------------

def downfold(eps, e_qp, gb, no, nv, omega, spin='singlet'):
    """A(omega) = A - V^e (omega I - D)^-1 (V^h)^T
                    - V^h (omega I - D)^-1 (V^e)^T          Eq. (9)

    Downfolding the doubles back out of H must return the frequency-dependent
    BSE matrix (1). Checking that against `dynamical_bse_matrix` -- which is
    built from a completely different route, the sum-over-states (6) -- is
    what proves the upfolded construction, before any solver is involved."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    d = dimensions(no, nv)
    n_s, n_d = d['n_s'], d['n_d']
    A = block_A(e_qp, gb, no, nv, kappa).reshape(n_s, n_s)
    S = block_S(eps, gb, no, nv).reshape(n_s, n_s)
    Vh, Ve = block_V(gb, no, nv)
    Vh, Ve = Vh.reshape(n_s, n_d), Ve.reshape(n_s, n_d)
    D = _build_D(eps, e_qp, S, no, nv)
    G = np.linalg.inv(omega * np.eye(n_d) - D)
    return A - Ve @ G @ Vh.T - Vh @ G @ Ve.T


def dynamical_bse_matrix(eps, e_qp, gb, no, nv, omega, spin='singlet'):
    """A(Omega) = A - K^(p)(Omega) with K^(p) from the SUM OVER STATES (6).

    The expensive, textbook route: diagonalize the TDA-RPA matrix S for every
    excitation Omega_m, form (pq|rho_m) = sum_kc X^m_kc (pq|kc), and sum the
    poles

        K^(p)_abij = 2 sum_m (ij|rho_m)(ab|rho_m)
                     [ 1/(Om - (E_b - E_i) - Om_m)
                     + 1/(Om - (E_a - E_j) - Om_m) ].

    Note which indices pair with which: the first denominator takes the
    COLUMN virtual with the ROW occupied, the second the ROW virtual with the
    COLUMN occupied, and the result enters (1a) as A_ia,jb - K_abij.

    Independent of the upfolding in every respect, which is the point of
    keeping it: `downfold` must reproduce this at every omega."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    n_s = no * nv
    S = block_S(eps, gb, no, nv).reshape(n_s, n_s)
    Om, X = np.linalg.eigh(S)                       # S is symmetric here
    X = X.reshape(no, nv, n_s)
    oo_rho = np.einsum('ijkc,kcm->ijm', gb['ooov'], X, optimize=True)
    vv_ov = gb['ovvv'].transpose(2, 3, 0, 1)        # (ab|kc) = (kc|ab)
    vv_rho = np.einsum('abkc,kcm->abm', vv_ov, X, optimize=True)
    Eo, Ev = e_qp[:no], e_qp[no:]
    g1 = 1.0 / (omega - (Ev[:, None, None] - Eo[None, :, None])
                - Om[None, None, :])                # [b, i, m]
    g2 = 1.0 / (omega - (Ev[:, None, None] - Eo[None, :, None])
                - Om[None, None, :])                # [a, j, m], same array
    K = 2.0 * (np.einsum('ijm,abm,bim->iajb', oo_rho, vv_rho, g1,
                         optimize=True)
               + np.einsum('ijm,abm,ajm->iajb', oo_rho, vv_rho, g2,
                           optimize=True))
    A = block_A(e_qp, gb, no, nv, kappa)
    return (A - K).reshape(n_s, n_s)


# ----------------------------------------------------------------------
# matrix-free sigma, Eq. (10)
# ----------------------------------------------------------------------

def sigma(vec, eps, e_qp, gb, no, nv, spin='singlet'):
    """H R = sigma, Eqs. (10a)-(10c), never forming H.

    The dominant term is the S coupling `2 sum_ia (kc|ia) r_li^da`, which is
    O(o^3 v^3); `sigma_df` factorizes exactly that one."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    r1, r2, r2b = to_blocks(vec, no, nv)
    ooov, ovvv, ovov, oovv = gb['ooov'], gb['ovvv'], gb['ovov'], gb['oovv']
    Eo, Ev = e_qp[:no], e_qp[no:]
    dfull = doubles_diagonal(eps, e_qp, no, nv)

    # --- 10a ----------------------------------------------------------
    s1 = (Ev[None, :] - Eo[:, None]) * r1
    s1 = s1 + kappa * np.einsum('iajb,jb->ia', ovov, r1, optimize=True)
    s1 = s1 - np.einsum('ijab,jb->ia', oovv, r1, optimize=True)
    # -sqrt(2) sum_dkc (kc|ad) r_ik^dc   ==  -(V^e r)
    s1 = s1 - SQRT2 * np.einsum('kcad,idkc->ia', ovvv, r2, optimize=True)
    # -sqrt(2) sum_lkc (il|kc) rbar_lk^ac  ==  -(V^h rbar)
    s1 = s1 - SQRT2 * np.einsum('ilkc,lakc->ia', ooov, r2b, optimize=True)

    # --- 10b ----------------------------------------------------------
    s2 = dfull * r2
    s2 = s2 + SQRT2 * np.einsum('ilkc,id->ldkc', ooov, r1, optimize=True)
    s2 = s2 + 2.0 * np.einsum('kcia,ldia->ldkc', ovov, r2, optimize=True)

    # --- 10c ----------------------------------------------------------
    s3 = dfull * r2b
    s3 = s3 + SQRT2 * np.einsum('kcad,la->ldkc', ovvv, r1, optimize=True)
    s3 = s3 + 2.0 * np.einsum('kcia,ldia->ldkc', ovov, r2b, optimize=True)
    return from_blocks(s1, s2, s3)


def sigma_df(vec, eps, e_qp, gbo, B, no, nv, spin='singlet'):
    """Eq. (10) with the O(N^6) screening term factorized through the DF
    factor, Eq. (11):

        2 sum_ia (kc|ia) r_li^da  =  2 sum_P L^P_kc sum_ia L^P_ia r_li^da

    -- two O(naux o^2 v^2) steps instead of one O(o^3 v^3) one. The absence
    of exchange integrals in the direct RPA is what allows it; nothing else
    in the BSE matvec has this structure.

    `gbo` carries the blocks that stay dense (ooov, ovvv, ovov, oovv is only
    needed for A); B is (naux, norb, norb) with (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s].
    """
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    r1, r2, r2b = to_blocks(vec, no, nv)
    ooov, ovvv, ovov, oovv = gbo['ooov'], gbo['ovvv'], gbo['ovov'], gbo['oovv']
    Bov = B[:, :no, no:]
    Eo, Ev = e_qp[:no], e_qp[no:]
    dfull = doubles_diagonal(eps, e_qp, no, nv)

    s1 = (Ev[None, :] - Eo[:, None]) * r1
    s1 = s1 + kappa * np.einsum('iajb,jb->ia', ovov, r1, optimize=True)
    s1 = s1 - np.einsum('ijab,jb->ia', oovv, r1, optimize=True)
    s1 = s1 - SQRT2 * np.einsum('kcad,idkc->ia', ovvv, r2, optimize=True)
    s1 = s1 - SQRT2 * np.einsum('ilkc,lakc->ia', ooov, r2b, optimize=True)

    def screen(x):
        """2 sum_P L^P_kc (sum_ia L^P_ia x_ldia)."""
        t = np.einsum('Pia,ldia->Pld', Bov, x, optimize=True)
        return 2.0 * np.einsum('Pkc,Pld->ldkc', Bov, t, optimize=True)

    s2 = dfull * r2 + SQRT2 * np.einsum('ilkc,id->ldkc', ooov, r1,
                                        optimize=True) + screen(r2)
    s3 = dfull * r2b + SQRT2 * np.einsum('kcad,la->ldkc', ovvv, r1,
                                         optimize=True) + screen(r2b)
    return from_blocks(s1, s2, s3)


def diagonal(eps, e_qp, gb, no, nv, spin='singlet'):
    """Operator diagonal, for a Davidson preconditioner."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    Eo, Ev = e_qp[:no], e_qp[no:]
    d1 = (Ev[None, :] - Eo[:, None]) \
        + kappa * np.einsum('iaia->ia', gb['ovov'], optimize=True) \
        - np.einsum('iiaa->ia', gb['oovv'], optimize=True)
    dd = doubles_diagonal(eps, e_qp, no, nv) \
        + 2.0 * np.einsum('kckc->kc', gb['ovov'], optimize=True)[None, None, :, :]
    return np.concatenate([d1.ravel(), dd.ravel(), dd.ravel()])


# ----------------------------------------------------------------------
# solving
# ----------------------------------------------------------------------

def doubles_weight(vec, no, nv):
    """%R2: the fraction of an eigenvector living in the two doubles
    sectors -- the paper's measure of double-excitation character."""
    r1, r2, r2b = to_blocks(vec, no, nv)
    n1 = float(np.vdot(r1, r1).real)
    n2 = float(np.vdot(r2, r2).real + np.vdot(r2b, r2b).real)
    return 100.0 * n2 / (n1 + n2)


def solve_dense(eps, e_qp, gb, no, nv, spin='singlet', nroots=5,
                imag_tol=UPFOLDED_BSE_IMAG_TOL, flags=None):
    """Lowest `nroots` positive eigenvalues of H and their right eigenvectors.

    H is NOT symmetric, so this is scipy.linalg.eig, not eigh, and the
    eigenvalues are complex in general. The physical ones are real; a
    significant imaginary part means the upfolded problem has gone unstable
    (it can, just as the underlying dynamical BSE can): such a root is
    returned as NaN, warned about and flagged 'upfolded_bse_complex_roots' in
    `flags` (a list, when given), never reported by its real part."""
    H = build_hamiltonian(eps, e_qp, gb, no, nv, spin)
    w, vr = linalg.eig(H)
    keep = np.argsort(w.real)
    keep = [k for k in keep if w.real[k] > 0][:nroots]
    # Complex eigenvalues DO occur, in conjugate pairs, high in the doubles
    # manifold -- H is real but not symmetric, and the dynamical BSE it
    # encodes has no theorem forcing every root real. They are not an error,
    # and warning about roots the caller never sees is noise; warn only if a
    # RETURNED root has a significant imaginary part, because that one is
    # being reported as a physical excitation, and return it as NaN, flagged.
    scale = max(1.0, np.abs(w.real).max())
    return _complex_roots(w.real[keep], w.imag[keep], imag_tol, scale,
                          vr[:, keep].real, flags, 'roots')


def davidson_nonsymmetric(aop, diag, nroots=5, tol=UPFOLDED_BSE_DAVIDSON_TOL,
                          max_space=None, max_cycle=UPFOLDED_BSE_DAVIDSON_MAX_CYCLE,
                          imag_tol=UPFOLDED_BSE_IMAG_TOL, refuse_unconverged=False,
                          flags=None):
    """Davidson for a NON-symmetric operator with (expected) real spectrum.

    The only structural difference from the symmetric Davidson the ADC side
    uses is that the projected subspace matrix V^T H V is not symmetric, so
    it goes through scipy.linalg.eig and the Ritz values are sorted by real
    part. Everything else -- diagonal preconditioning, re-orthogonalization,
    collapse on overflow -- is the usual algorithm.

    Returns (energies, right eigenvectors). Left eigenvectors are not
    computed; nothing here needs them, and %R2 is defined on the right one.
    A returned Ritz value with relative imaginary part above imag_tol is
    returned as NaN (value and vector), warned about and flagged
    'upfolded_bse_complex_roots': the dynamical BSE has no real root there, so
    no number stands in its place. A loop that ends on max_cycle or on
    a linearly dependent correction above tol is a numerical event (refuse
    inputs, warn numerics): its Ritz pairs are
    returned with a RuntimeWarning and the flag 'upfolded_bse_not_converged'
    appended to `flags` (a list, when given); refuse_unconverged=True refuses
    instead."""
    n = len(diag)
    nroots = min(nroots, n)
    max_space = max_space or max(4 * nroots + 10, 20)
    order = np.argsort(diag)
    V = np.zeros((n, 0))
    guess = np.eye(n)[:, order[:2 * nroots]]
    V = extend_orthonormal(V, guess, atol=UPFOLDED_BSE_LINDEP_TOL)
    W = np.column_stack([aop(V[:, k]) for k in range(V.shape[1])])
    theta = None
    converged, stop = False, f'{max_cycle} cycles'
    for _ in range(max_cycle):
        Hs = V.T @ W
        w, y = linalg.eig(Hs)
        idx = np.argsort(w.real)
        idx = [k for k in idx if w.real[k] > 0][:nroots] or list(idx[:nroots])
        theta_c = w[idx]
        theta, Y = theta_c.real, y[:, idx].real
        X = V @ Y
        R = W @ Y - X * theta[None, :]
        if np.linalg.norm(R) < tol:
            converged = True
            break
        if V.shape[1] + len(idx) > max_space:            # collapse
            V = extend_orthonormal(np.zeros((n, 0)), X, atol=UPFOLDED_BSE_LINDEP_TOL)
            W = np.column_stack([aop(V[:, k]) for k in range(V.shape[1])])
            continue
        prec = []
        for k in range(len(idx)):
            denom = theta[k] - diag
            denom = np.where(np.abs(denom) < UPFOLDED_BSE_PRECOND_FLOOR,
                             UPFOLDED_BSE_PRECOND_FLOOR, denom)
            prec.append(R[:, k] / denom)
        new = extend_orthonormal(V, np.column_stack(prec), atol=UPFOLDED_BSE_LINDEP_TOL)
        if new.shape[1] == 0:
            stop = 'a correction already spanned by the subspace'
            break
        V = np.hstack([V, new])
        W = np.hstack([W, np.column_stack([aop(new[:, k])
                                           for k in range(new.shape[1])])])
    scale = max(1.0, float(np.abs(theta).max()))
    theta, X = _complex_roots(theta, theta_c.imag, imag_tol, scale, X, flags,
                              'Ritz values')
    if not converged:
        message = (f'the upfolded-BSE Davidson stopped on {stop} with residual '
                   f'norm {np.linalg.norm(R):.2e} above {tol:.0e}; its Ritz '
                   f'values are not converged roots.')
        if refuse_unconverged:
            raise UpfoldedBSERefusal(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        if flags is not None:
            flags.append({'flag': 'upfolded_bse_not_converged', 'message': message})
    return theta, X


# ----------------------------------------------------------------------
# the symmetric mean-field variant, Eq. (12)
# ----------------------------------------------------------------------

def build_hamiltonian_familiar(eps, gb, no, nv, spin='singlet'):
    """H~ of Eq. (12): mean-field energies throughout, ONE doubles sector,
    and the coupling V^e - V^h on both sides, which makes it symmetric.

    The paper offers this as the form that looks like ADC and needs no prior
    GW, then reports (their words, results not shown) that its eigenvalues
    come out 2-3 eV BELOW the exact dynamical BSE, because the self-energy
    correction it implies keeps only forward time-ordered diagrams. Provided
    for study; `build_hamiltonian` is the method."""
    kappa = KAPPA[spin] if isinstance(spin, str) else float(spin)
    d = dimensions(no, nv)
    n_s, n_d = d['n_s'], d['n_d']
    A = block_A(eps, gb, no, nv, kappa).reshape(n_s, n_s)
    S = block_S(eps, gb, no, nv).reshape(n_s, n_s)
    Vh, Ve = block_V(gb, no, nv)
    C = (Ve - Vh).reshape(n_s, n_d)
    D = _build_D(eps, eps, S, no, nv)
    H = np.zeros((n_s + n_d, n_s + n_d))
    H[:n_s, :n_s] = A
    H[:n_s, n_s:] = C
    H[n_s:, :n_s] = C.T
    H[n_s:, n_s:] = D
    return H


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def qp_energies(mf, states=None, tda=True, df=True, selfenergy='GW'):
    """All GW quasiparticle energies, in HARTREE, for the upfolded BSE.

    Eq. (7) takes correlated E_p as input -- that is what makes it GW/BSE
    rather than a screened CIS. The paper finds GW/BSE@HF with TDA screening
    the most accurate combination of those it tested (0.2-0.3 eV on the Thiel
    set), so tda=True is the default here.

    This calls the existing GW module once per orbital, which is the honest
    cost: O(N) quasiparticle solves. Pass `states` to restrict it."""
    norb = mf.mo_coeff.shape[1]
    states = list(range(norb)) if states is None else list(states)
    res = calc_qp_energy(mf, selfenergy=selfenergy, state=states, tda=tda,
                         df=df)
    out = np.array([res[p][selfenergy] for p in states]) / HARTREE_TO_EV
    return out


def solve_bse_upfolded(mf, mol=None, spin='singlet', nroots=5, e_qp=None,
                       df=True, auxbasis=None, matrix_free=True,
                       conv_tol=UPFOLDED_BSE_DAVIDSON_TOL, frozen=0,
                       refuse_unconverged=False, flags=None):
    """(energies in Hartree, right eigenvectors) of the upfolded BSE.

    df: every integral block from the density-fitting factor (True) or from
    the exact four-index ERI (False), on the dense and the matrix-free path
    alike, so the system size picks the solver and never the integrals.

    e_qp: GW quasiparticle energies (Hartree), length norb. `None` falls back
    to the mean-field eigenvalues, which is NOT GW/BSE -- it is the same
    matrix with an uncorrelated one-particle spectrum, useful for testing the
    algebra and nothing else. Use `qp_energies(mf)` for the real thing.

    flags: a list that receives the solve's named numerical events
    ('upfolded_bse_complex_roots', NaN roots; davidson_nonsymmetric:
    'upfolded_bse_not_converged'); refuse_unconverged
    refuses an unconverged Davidson instead.
    """
    mol = mol if mol is not None else mf.mol
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(
            'the upfolded BSE here is closed-shell (kappa = 2/0 for '
            'singlet/triplet assumes a spin-restricted reference)')
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'))
    act = slice(frozen, len(eps))
    if e_qp is None:
        warnings.warn(
            'no GW quasiparticle energies given; falling back to mean-field '
            'eigenvalues. That is not GW/BSE -- pass e_qp=qp_energies(mf).',
            RuntimeWarning, stacklevel=2)
        e_qp = eps
    e_qp = np.asarray(e_qp)
    eps, e_qp = eps[act], e_qp[act]
    norb = len(eps)
    no = mol.nelectron // 2 - frozen
    nv = norb - no

    if df:
        src = mf if (auxbasis is None and getattr(mf, 'with_df', None)) \
            else mf.density_fit(auxbasis=auxbasis)
        B = DFIntegrals.from_scf(mol, src).B_aa[:, act, act]
        gb = df_eri_blocks(B, no, norb)
    else:
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
        gb = eri_blocks(eri[act, act, act, act], no, norb)
    d = dimensions(no, nv)

    if not matrix_free or d['nH'] <= UPFOLDED_BSE_DENSE_LIMIT:
        return solve_dense(eps, e_qp, gb, no, nv, spin, nroots, flags=flags)
    if df:
        aop = lambda v: sigma_df(v, eps, e_qp, gb, B, no, nv, spin)
    else:
        aop = lambda v: sigma(v, eps, e_qp, gb, no, nv, spin)
    diag = diagonal(eps, e_qp, gb, no, nv, spin)
    return davidson_nonsymmetric(aop, diag, nroots=nroots, tol=conv_tol,
                                 refuse_unconverged=refuse_unconverged, flags=flags)
