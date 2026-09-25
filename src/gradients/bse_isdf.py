"""BSE@GW excitation energies from the ISDF factors, and their adjoints.

The cubic quasiparticle gradient chains eps^QP back to (eps, X, D), and
`isdf_derivatives` chains those to the nuclei. This module does the same for a
BSE excitation energy, so that the nuclear assembly is reached unchanged and
nothing new is needed downstream.

Everything is written in the variables production already uses
(`davidson.isdf_bse_factors`): the separable factors X (M, norb) and D
(M, naux), and the STATIC screening as an (naux, naux) matrix

    W_aux = [1 - chi0(0)]^-1 ,

which is what keeps the screening off the pair space. The pilot's
W = v - 4 V (A+B)^-1 V^T inverts an (n_ov, n_ov) matrix -- 840 GB at 150 atoms
-- and this does not; the same object is an auxiliary-basis inverse of side
naux instead. The three-index tensor is likewise never formed: only the
blocks a kernel actually needs, B[:, occ, virt], B[:, occ, occ], B[:, virt,
virt] and B[:, virt, occ], each built directly from (X, D).

    A[ia,jb] = delta_ij delta_ab (eps^QP_a - eps^QP_i) + kappa (ia|jb) - W_ij,ab
    B[ia,jb] = kappa (ia|jb) - W_aj,bi
    (ia|jb)  = sum_P B_ov[P,ia] B_ov[P,jb]
    W_ij,ab  = sum_PQ B_oo[P,ij] W_aux[P,Q] B_vv[Q,ab]

kappa = 2 for a singlet and 0 for a triplet, the one factor that separates
them (`davidson` and `bse_upfolded` use the same convention). The forward pass
reproduces production's ISDF Davidson root for root.

WHAT IS STILL PAIR-SPACE HERE: the blocks themselves are (n_ov, n_ov) and the
Casida step is a dense eigensolve, so this closes the ADJOINT CHAIN at small
size, not yet the cost. Production already has the matrix-free Davidson that
replaces it; wiring the adjoint to iterative eigenvectors does not change any
of the algebra below.

X and D may be `SlicedFactors`, each rank's grid rows of X_mo and D: every
block is a sum over the grid, so each entry point gathers X_mo and D whole
once and drops them on return, and its output is the whole factors' bit for
bit.
"""
import numpy as np

from src.Base.constants import ISDF_TILE_GB, KAPPA
from src.SingleReference.LinearResponse.davidson import bse_pair_diagonal
from src.SingleReference.LinearResponse.space_time import b_block  # noqa: F401
from src.SingleReference.base import get_occ_virt_indices
from src.Base.sliced_factors import whole_factor


def b_block_backward(X, D, p_idx, q_idx, B_bar, X_bar, D_bar):
    """Accumulate a block's adjoint onto (X, D). Both AO slots of B carry X, so
    the bra and ket index sets each collect a term.

    GEMMs for the same reason as `b_block`, and the first two terms share their
    intermediate: t[k,P] = sum_b B_bar[P,a,b] X_q[k,b] serves both the D adjoint
    and the bra-side X adjoint, so it is formed once per bra function.
    """
    Xp, Xq = X[:, p_idx], X[:, q_idx]
    npi, nqi = Xp.shape[1], Xq.shape[1]
    U = np.empty((X.shape[0], npi))
    for a in range(npi):
        t = Xq @ B_bar[:, a, :].T                     # (M, naux)
        D_bar += Xp[:, a, None] * t
        U[:, a] = np.einsum('kP,kP->k', t, D)
    np.add.at(X_bar.T, p_idx, U.T)
    V = np.empty((X.shape[0], nqi))
    for b in range(nqi):
        sm = Xp @ B_bar[:, :, b].T                    # (M, naux)
        V[:, b] = np.einsum('kP,kP->k', sm, D)
    np.add.at(X_bar.T, q_idx, V.T)


def _factorized_block_backward(X, D, p_idx, q_idx, cP, alpha, beta,
                               X_bar, D_bar, tile_gb=ISDF_TILE_GB):
    """`b_block_backward` for an adjoint that is itself factorized on the grid.

    Every screened adjoint in the BSE arrives in the form

        Bbar[P,p,q] = sum_k cP[k,P] alpha[k,p] beta[k,q],

    because B itself is sum_k X[k,p] X[k,q] D[k,P]. Contracting that against
    another B closes over GRID indices rather than orbital ones:

        Dbar[k',P] += sum_k cP[k,P] S1[k,k'] S2[k,k']
        Xbar[k',p] += sum_k E[k',k] S2[k,k'] alpha[k,p]
        Xbar[k',q] += sum_k E[k',k] S1[k,k'] beta[k,q]
        S1 = alpha Xp^T,  S2 = beta Xq^T,  E = D cP^T

    which is O(M^2 (n_p + n_q + n_aux)) instead of the O(M n_aux n_p n_q) of
    forming the adjoint explicitly -- cubic rather than quartic, and it removes
    the (n_aux, n_vir, n_vir) block entirely.

    S1, S2 and E are (M, M), the objects the polarizability sweep also refuses
    to hold, so they are tiled over k' by the same budget: only a block is live.
    """
    Xp, Xq = X[:, p_idx], X[:, q_idx]
    M = X.shape[0]
    rows = max(1, min(M, int(tile_gb * 1e9 / max(3 * M * 8, 1))))
    same = p_idx is q_idx or (np.array_equal(np.asarray(p_idx),
                                             np.asarray(q_idx)))
    for k0 in range(0, M, rows):
        k1 = min(k0 + rows, M)
        S1 = alpha @ Xp[k0:k1].T                  # (M, blk)
        S2 = S1 if (same and alpha is beta) else beta @ Xq[k0:k1].T
        E = D[k0:k1] @ cP.T                       # (blk, M)
        D_bar[k0:k1] += (S1 * S2).T @ cP
        X_bar[k0:k1, p_idx] += (E * S2.T) @ alpha
        X_bar[k0:k1, q_idx] += (E * S1.T) @ beta


def bse_cache(X, D, eps_qp, W_aux, nocc, spin='singlet', bse_tda=False):
    """The three-index blocks `bse_backward` needs, WITHOUT any (n_ov, n_ov).

    `bse_blocks` builds the dense Casida matrices and is the small-system
    reference; production solves the same problem matrix-free
    (`davidson.solve_bse_isdf`), which returns the eigenvectors and never forms
    a block. The adjoint needs only those eigenvectors and these B blocks, each
    (naux, n_occ, n_vir) or smaller, so pairing this with the Davidson solver
    removes the last pair-space array from the route.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    occ, virt = get_occ_virt_indices(np.asarray(eps_qp, float), nocc)
    n_ov = len(occ) * len(virt)
    return {'occ': occ, 'virt': virt, 'kappa': KAPPA[spin],
            'B_ov': b_block(X, D, occ, virt).reshape(-1, n_ov),
            'B_oo': b_block(X, D, occ, occ),
            'B_vo': None if bse_tda else b_block(X, D, virt, occ)}


def bse_blocks(X, D, eps_qp, W_aux, nocc, spin='singlet', bse_tda=False):
    """(A, B, cache) of the BSE in ISDF variables; B is None for a TDA problem."""
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    kappa = KAPPA[spin]
    occ, virt = get_occ_virt_indices(np.asarray(eps_qp, float), nocc)
    n_ov = len(occ) * len(virt)
    B_ov = b_block(X, D, occ, virt).reshape(-1, n_ov)
    B_oo, B_vv = b_block(X, D, occ, occ), b_block(X, D, virt, virt)
    v = B_ov.T @ B_ov
    W_ijab = np.einsum('Pij,PQ,Qab->ijab', B_oo, W_aux, B_vv, optimize=True)
    d = bse_pair_diagonal(eps_qp, nocc).ravel()
    A = np.diag(d) + kappa * v - W_ijab.transpose(0, 2, 1, 3).reshape(n_ov, n_ov)
    cache = {'occ': occ, 'virt': virt, 'B_ov': B_ov, 'B_oo': B_oo, 'B_vv': B_vv,
             'kappa': kappa, 'B_vo': None}
    if bse_tda:
        return A, None, cache
    B_vo = b_block(X, D, virt, occ)
    W_ajbi = np.einsum('Paj,PQ,Qbi->ajbi', B_vo, W_aux, B_vo, optimize=True)
    cache['B_vo'] = B_vo
    Bb = kappa * v - np.einsum('ajbi->iajb', W_ajbi).reshape(n_ov, n_ov)
    return A, Bb, cache


def bse_solve(A, Bb, nroots=None):
    """(Omega, Xn, Yn) from the Casida problem; Xn, Yn are (n_ov, nroots)."""
    if Bb is None:
        w, Z = np.linalg.eigh(0.5 * (A + A.T))
        return w, Z, np.zeros_like(Z)
    AmB = 0.5 * ((A - Bb) + (A - Bb).T)
    L = np.linalg.cholesky(AmB)
    Mh = L.T @ (A + Bb) @ L
    w2, Z = np.linalg.eigh(0.5 * (Mh + Mh.T))
    if w2.min() <= 0:
        raise ValueError(f"BSE instability: min Omega^2 = {w2.min():.3e}; the "
                         f"reference is singlet/triplet unstable at this "
                         f"geometry (triplets are the usual culprit -- use the "
                         f"Tamm-Dancoff form).")
    Om = np.sqrt(w2)
    XpY = (L @ Z) / np.sqrt(Om)[None, :]
    XmY = np.linalg.solve(L.T, Z) * np.sqrt(Om)[None, :]
    return Om, 0.5 * (XpY + XmY), 0.5 * (XpY - XmY)


def bse_backward(n, X, D, eps_qp, W_aux, nocc, cache, Xn, Yn, omega_bar=1.0,
                 tile_gb=ISDF_TILE_GB, bra=None):
    """(eps_qp_bar, X_bar, D_bar, W_aux_bar) of omega_bar * Omega_n.

    `bra` makes the element INTERSTATE: with bra = m the routine returns the
    adjoint of <m| dH |n> instead of dOmega_n, which is the numerator of the
    derivative coupling. C^A and C^B become x_m x_n^T + y_m y_n^T and
    x_m y_n^T + y_m x_n^T -- still rank two, so nothing below changes shape and
    the (n_ov, n_ov) matrix is still never formed. The one-sided contraction is
    not symmetric in m <-> n; `interstate_backward` averages both orderings,
    which is exact because dH is.

    Hellmann-Feynman on the Casida vectors, with

        C^A = x x^T + y y^T,   C^B = x y^T + y x^T,
        dOmega = sum C^A dA + sum C^B dB,

    distributed over the quasiparticle diagonal, the bare kernel, the direct
    screened term and, for a full BSE, the swap term.

    NOTHING (n_ov, n_ov) IS EVER FORMED. C^A and C^B are RANK TWO, so every
    contraction is carried in that form: at 150 atoms n_ov ~ 3e5 and a dense
    C^A would be 840 GB, while each intermediate below is (naux, n_occ, n_vir)
    or smaller. Two identities do most of the work -- the bare kernel needs
    only C^A + C^B = (x+y)(x+y)^T, and every screened contraction reduces to
    triple products of the (n_occ, n_vir)-shaped eigenvector with a B block.
    Flops stay at the level of a density-fitted exchange build; storage is
    three-index.
    """
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    occ, virt = cache['occ'], cache['virt']
    B_ov, B_oo = cache['B_ov'], cache['B_oo']
    kappa, B_vo = cache['kappa'], cache['B_vo']
    Xo, Xv = X[:, occ], X[:, virt]
    no, nv = len(occ), len(virt)
    m = n if bra is None else bra
    x, y = Xn[:, m], Yn[:, m]                      # bra
    xk, yk = Xn[:, n], Yn[:, n]                    # ket
    Xm, Ym = x.reshape(no, nv), y.reshape(no, nv)
    Xk, Yk = xk.reshape(no, nv), yk.reshape(no, nv)

    eps_qp_bar = np.zeros_like(np.asarray(eps_qp, float))
    X_bar = np.zeros_like(X)
    D_bar = np.zeros_like(D)
    W_aux_bar = np.zeros_like(W_aux)

    # (i) quasiparticle diagonal: diag(C^A) needs no matrix at all
    diag = omega_bar * (Xm * Xk + Ym * Yk)
    eps_qp_bar[virt] += diag.sum(axis=0)
    eps_qp_bar[occ] -= diag.sum(axis=1)

    # (ii) bare kernel. For a full BSE C^A + C^B = (x+y)(x+y)^T exactly; for a
    # TDA problem y = 0 and it degenerates to x x^T, so one line covers both.
    u, uk = x + y, xk + yk
    B_ov_bar = (kappa * omega_bar) * (np.outer(B_ov @ u, uk)
                                      + np.outer(B_ov @ uk, u))
    b_block_backward(X, D, occ, virt, B_ov_bar.reshape(-1, no, nv),
                     X_bar, D_bar)

    # (iii) direct screened term, -W_ij,ab, as two rank-one pieces. Both the
    # G intermediate and the virtual-virtual adjoint are routed through the
    # GRID index rather than the virtual one, so no (naux, n_vir, n_vir) array
    # is built anywhere -- see `_factorized_block_backward`.
    for T, Tk in ((Xm, Xk), (Ym, Yk)):
        u, uq = Xv @ T.T, Xv @ Tk.T                        # (M, n_occ)
        w, wq = Xo @ T, Xo @ Tk                            # (M, n_vir)
        G = b_block(u, D, slice(None), slice(None), Y=uq)  # (naux, no, no)
        W_aux_bar -= omega_bar * np.einsum('Pij,Qij->PQ', B_oo, G, optimize=True)
        b_block_backward(X, D, occ, occ,
                         -omega_bar * np.einsum('PQ,Qij->Pij', W_aux, G,
                                                optimize=True), X_bar, D_bar)
        _factorized_block_backward(X, D, virt, virt,
                                   -omega_bar * (D @ W_aux), w, wq,
                                   X_bar, D_bar, tile_gb=tile_gb)

    # (iv) swap term, -W_aj,bi, only for a full BSE, through the
    # (n_aux, n_vir, n_occ) block. Factorized on the grid term by term it
    # needs four (M, M) passes of M^2 n_aux each; sharing ONE Hadamard weight
    # and one Zt tile across every screened term instead is
    # `LinearResponse.isdf_bse_adjoint`, which forms no three-index block and
    # costs M^2 (4 n_aux + 32 n_occ) against M n_aux n_occ n_vir here.
    if B_vo is not None:
        g = np.zeros_like(B_vo)
        M1 = np.einsum('PQ,Qbi->Pbi', W_aux, B_vo, optimize=True)
        M2 = np.einsum('PQ,Paj->Qaj', W_aux, B_vo, optimize=True)
        for T1, T2 in ((Xm, Yk), (Ym, Xk)):
            K = np.einsum('ia,Paj,jb->Pib', T1, B_vo, T2, optimize=True)
            W_aux_bar -= omega_bar * np.einsum('Pib,Qbi->PQ', K, B_vo,
                                               optimize=True)
            g -= omega_bar * np.einsum('ia,jb,Pbi->Paj', T1, T2, M1,
                                       optimize=True)
            g -= omega_bar * np.einsum('ia,jb,Qaj->Qbi', T1, T2, M2,
                                       optimize=True)
        b_block_backward(X, D, virt, occ, g, X_bar, D_bar)
    return eps_qp_bar, X_bar, D_bar, W_aux_bar


def interstate_backward(m, n, X, D, eps_qp, W_aux, nocc, cache, Xn, Yn,
                        omega_bar=1.0, tile_gb=ISDF_TILE_GB):
    """Adjoints of the SYMMETRIZED interstate element <m| dH |n>, m != n.

    The numerator of the derivative coupling: d_mn = <m|dH|n> / (Omega_n -
    Omega_m). The one-sided contraction is averaged over both orderings, which
    is exact because dH is symmetric and which removes any dependence on where
    the bra and the ket sit inside each of the four terms.

    Returns the same four adjoints `bse_backward` does, so the rest of the
    reverse chain -- quasiparticle, screening, factorization, integrals -- is
    reached unchanged.
    """
    if m == n:
        raise ValueError('an interstate element needs two different roots; '
                         'use bse_backward for dOmega_n')
    # once for both orderings
    X, D = whole_factor(X, 'X_mo'), whole_factor(D, 'D')
    out = None
    for bra, ket in ((m, n), (n, m)):
        part = bse_backward(ket, X, D, eps_qp, W_aux, nocc, cache, Xn, Yn,
                            omega_bar=0.5 * omega_bar, tile_gb=tile_gb, bra=bra)
        out = part if out is None else tuple(a + b for a, b in zip(out, part))
    return out


def screening_backward(W_aux, W_aux_bar):
    """chi0(0) adjoint for an adjoint on W_aux = [1 - chi0(0)]^-1.

    dW = W dchi0 W, so the reverse is chi0_bar = W^T W_bar W^T -- the same
    identity the self-energy route uses, at the single static frequency.
    """
    return W_aux.T @ W_aux_bar @ W_aux.T
