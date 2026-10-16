"""Doubles fold of the EE-ADC operator: the singles problem at its own frequency.

At block order o_dd = 0 (levels adc2 and gf2) the doubles block of the EE-ADC
supermatrix is the bare diagonal D_ijab = ε_a + ε_b - ε_i - ε_j, so the doubles
can be eliminated without approximation: the supermatrix eigenproblem

    sum_jb M_ia,jb y_jb + sum_K V_K,ia Y_K = ω y_ia ,
    sum_jb V_K,jb y_jb + D_K Y_K = ω Y_K

gives Y_K = sum_jb V_K,jb y_jb / (ω - D_K) and sum_jb A_eff(ω)_ia,jb y_jb = ω y_ia,

    A_eff(ω)_ia,jb = M_ia,jb - sum_K V_K,ia V_K,jb / (D_K - ω) ,

with ia, jb the singles and K the doubles of the flat adc2 layout, in whose
metric the supermatrix is symmetric (K is a spin-resolved (k, l, c, d)). Level
gw (ee_gw_pieces, eq 66: BSE@GW with one doubles set) has the same shape with
K = (k, c, m) per spin block, a particle-hole pair times a screening mode,
D_kcm = Ω_m + ε_c - ε_k, and the plain sum over its blocks as the doubles norm
(pieces['dnorm2']).
For the eigenpair (λ(ω), y) of A_eff(ω) with sum_ia y_ia² = 1,

    dλ/dω = -sum_K (sum_jb V_K,jb y_jb)² / (D_K - ω)²,
    T1 = 1 / (1 - dλ/dω)      (the singles weight of the full eigenvector),

and self-consistency λ(ω) = ω is reached per root by Newton,

    ω_{k+1} = ω_k + (λ(ω_k) - ω_k) T1(ω_k),

with a fixed-point fallback ω_{k+1} = λ(ω_k), DIIS-accelerated, when T1 falls
under t_min (a doubles element near ω makes the derivative unreliable). This
is the shape of Turbomole's RI-CC2 solver on the ADC operator; the doubles
vector exists only inside one matvec.

Flat singles vectors here are the singles segments of ee_r_sigma's layout,
length 2 no nv (aa block then bb block); with a spin channel, the channel basis
of ee_driver._channel_basis at the singles layout (level 'adc1').

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eq 53 (gf2), eq 66
(gw).
C. Hättig and F. Weigend, J. Chem. Phys. (2000), doi:10.1063/1.1290013 (the
folded-doubles solver of RI-CC2).
"""
import warnings

import numpy as np

from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.Base.constants import (EE_FOLD_BLOCK_BRACKET, EE_FOLD_BLOCK_BUFFER,
                                 EE_FOLD_BLOCK_FOLD_BOUND, EE_FOLD_BLOCK_LINDEP,
                                 EE_FOLD_BLOCK_MARGIN, EE_FOLD_BLOCK_MAX_ITER,
                                 EE_FOLD_BLOCK_MAX_MOVES, EE_FOLD_BLOCK_MAX_NODES,
                                 EE_FOLD_BLOCK_MODEL_RESIDUAL,
                                 EE_FOLD_BLOCK_NODE_TARGET, EE_FOLD_MODEL_ROOT_MAX_ITER,
                                 EE_FOLD_MODEL_ROOT_STEP)
from src.Solvers.davidson import (diagonal_seeds, jacobi_davidson_precond,
                                  overlap_pick, solve_symmetric)

FOLD_LEVELS = ('adc2', 'gf2', 'gw')
_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}
_DEGENERATE = 1e-8      # Hartree: seeds spread by at most this are one level
_SEED_RESIDUAL = 1e-9   # tight seeds: Ritz error |r|²/gap far below _DEGENERATE
_M_SEED_RESIDUAL = 1e-3     # seeds of M: they only place the seed frequency
_FOLD_SEED_RESIDUAL = 1e-4  # seeds of A_eff at the seed frequency
_SEED_SPLIT = 1e-5      # Hartree: seeds closer than this are re-solved tight
_INNER_CAP = 1e-3       # loosest residual of a Newton step's eigensolve
_INNER_RATIO = 0.1      # its residual: this times the previous |λ - ω|
_CHECK_SHIFT = 0.1      # Hartree: the count check's correction r / (d - e + shift)
_CHECK_RESIDUAL = 1e-4  # the count check decides μ against a window, no tighter
_CHECK_SEEDS = 8        # unit vectors on the lowest lifted diagonal beside the random one
_SPAN = 1e-8            # relative norm left off a span below which a start vector is in it
_DUPLICATE = 0.5        # full-vector overlap above which two roots are one
_COUNT_ROUNDS = 3       # rounds of the count check before it warns


def singles_flat_to_sb(u, no, nv):
    """Flat singles vector, shape (2 no nv,), aa then bb -> SB {'aa', 'bb'} of
    shape (no, nv), index order (i, a)."""
    return _r.to_blocks(np.asarray(u, float), no, nv, 'adc1')[0]


def singles_sb_to_flat(w1, no, nv):
    """SB singles {'aa', 'bb'} shape (no, nv) -> flat vector, shape (2 no nv,)."""
    return _r.from_blocks(w1, SB(), no, nv, 'adc1')


def doubles_sb_to_flat(W, no, nv):
    """SB doubles -> the doubles segment of the adc2 layout, shape
    (2 n_same + n_mixed,), in the metric the full eigenvector is normalised in."""
    n_s = no * nv
    return _r.from_blocks(SB(), W, no, nv, 'adc2')[2 * n_s:]


def _check_level(pieces):
    if pieces.get('operator') is not None:
        # an operator (ee_ri_fold) carries its own spin channel and levels
        return
    if pieces['level'] not in FOLD_LEVELS:
        raise ValueError(f"the fold needs a bare-diagonal doubles block; level="
                         f"{pieces['level']!r} is not one of {FOLD_LEVELS}")
    if pieces['level'] == 'gw' and pieces.get('dnorm2') is None:
        # the adc2 flat layout would read none of the (k, c, m) blocks
        raise ValueError("level='gw' needs pieces['dnorm2'], the norm of its "
                         "(k, c, m) doubles")


def _check_vector(u, n):
    u = np.asarray(u)
    if np.iscomplexobj(u):
        raise ValueError('the fold is real symmetric; a complex vector was passed')
    u = u.ravel()
    if u.size != n:
        raise ValueError(f'singles vector of length {u.size}, expected {n}')
    return u.astype(float, copy=False)


def folded_operator(pieces, omega, spin=None):
    """A_eff(ω) on flat singles vectors, one spin channel or both.

    Parameters
    ----------
    pieces : dict
        From ``build_operator(..., pieces=True)``: 'M' (SB, blocks of shape
        (no, nv, no, nv), index order (i, a, j, b)), 'V' (SB singles, blocks
        (no, nv), (i, a) -> SB doubles, blocks (no, no, nv, nv), (i, j, a, b), or,
        at level gw, blocks 'aa' and 'bb' of shape (no, nv, nm), (k, c, m)),
        'Vt' (the reverse map), 'D' (ndarray, shape (no, no, nv, nv), index
        order (i, j, a, b), or, at level gw, shape (no, nv, nm), index order
        (k, c, m)), 'level', 'no', 'nv', 'be'; 'dnorm2' (optional: SB doubles
        -> float sum_K Y_K², for a doubles layout other than adc2's);
        'dense_limit' (optional int, default 0: solve_folded's dense_limit when
        the caller passes None).
    omega : float or None
        Frequency in Hartree; None gives the bare singles block M.
    spin : {'singlet', 'triplet', None}
        Restrict to one eigenspace of the alpha<->beta flip.

    Returns
    -------
    matvec : callable
        u (n,) -> A_eff(ω) u (n,), n = 2 no nv or the channel's size.
    dmatvec : callable
        u (n,) -> sum_K (sum_jb V_K,jb u_jb)² / (D_K - ω)² (a float), so that
        dλ/dω = -dmatvec(y) for a unit eigenvector y; 0.0 when omega is None.
    diag_s : ndarray, shape (n,)
        Diagonal of M on the channel's representative entries (preconditioner).
    embed, restrict : callables
        Channel basis -> full singles vector and back; identities for spin=None.
    """
    _check_level(pieces)
    if pieces.get('operator') is not None:
        return _operator_folded(pieces, omega, spin)
    no, nv, be = pieces['no'], pieces['nv'], pieces['be']
    M, V, Vt, D = pieces['M'], pieces['V'], pieces['Vt'], pieces['D']
    dnorm2 = pieces.get('dnorm2')
    n_full = 2 * no * nv
    if spin is None:
        def embed(u):
            return np.asarray(u, float)

        def restrict(v):
            return np.asarray(v, float)
        reps = np.arange(n_full)
    else:
        if spin not in _CHANNEL:
            raise ValueError(f"spin={spin!r}; expected 'singlet', 'triplet' or None")
        from src.SingleReference.ADC.eeADC.ee_driver import _channel_basis
        embed, restrict, reps = _channel_basis(n_full, no, nv, 'adc1',
                                               _CHANNEL[spin])
    n = reps.size
    # diag of M on the singles layout: M_ia,ia of the aa and the bb block
    m_diag = np.concatenate([np.einsum('iaia->ia', M.get('aaaa')).ravel(),
                             np.einsum('iaia->ia', M.get('bbbb')).ravel()])
    diag_s = m_diag[reps]
    denom = None if omega is None else D - omega        # (D - ω), sign kept

    def matvec(u):
        u = _check_vector(u, n)
        y1 = singles_flat_to_sb(embed(u), no, nv)
        w1 = be.ein('iajb,jb->ia', M, y1)                 # sum_jb M_ia,jb y_jb
        if denom is not None:
            # - sum_K V_K,ia (sum_jb V_K,jb y_jb) / (D_K - ω) : the folded doubles
            Y = be.divide(V(y1), denom)
            w1 = w1 - Vt(Y)
        return restrict(singles_sb_to_flat(w1, no, nv))

    def dmatvec(u):
        if denom is None:
            return 0.0
        u = _check_vector(u, n)
        y1 = singles_flat_to_sb(embed(u), no, nv)
        Yt = be.divide(V(y1), denom)
        if dnorm2 is not None:
            # sum_K Y_K² in the pieces' own doubles layout
            return float(dnorm2(Yt))
        # Y_K = sum_jb V_K,jb y_jb / (D_K - ω), flat; then sum_K Y_K²
        Yf = doubles_sb_to_flat(Yt, no, nv)
        return float(Yf @ Yf)

    return matvec, dmatvec, diag_s, embed, restrict


def _operator_folded(pieces, omega, spin):
    """folded_operator for pieces carrying an operator (ee_ri_fold.RIFoldADC2):
    it acts on its own spin channel, whose basis is the alpha block of the
    flip pairs (ee_driver._channel_basis at level 'adc1'), so embed and
    restrict are those of that basis and the matvec has a block form."""
    from src.SingleReference.ADC.eeADC.ee_driver import _channel_basis
    from src.SingleReference.ADC.eeADC.ee_ri_fold import FoldedMatvec
    op = pieces['operator']
    if spin != op.spin:
        raise ValueError(f"the operator solves the {op.spin} channel; spin="
                         f"{spin!r} asked")
    no, nv = pieces['no'], pieces['nv']
    embed, restrict, _ = _channel_basis(2 * no * nv, no, nv, 'adc1',
                                        _CHANNEL[spin])
    mv = FoldedMatvec(op, omega)

    def dmatvec(u):
        if omega is None:
            return 0.0
        u = _check_vector(u, op.n)
        return float(op.doubles_dot(u, omega, u, omega))

    return mv, dmatvec, op.diag.copy(), embed, restrict


def _d_min(pieces):
    """The lowest doubles energy min_K D_K."""
    if pieces.get('operator') is not None:
        return float(pieces['d_min'])
    return float(np.min(pieces['D']))


def dense_effective(matvec, n):
    """A_eff as an (n, n) array from n unit-vector matvecs, symmetrised."""
    A = np.column_stack([matvec(np.eye(n)[:, k]) for k in range(n)])
    return 0.5 * (A + A.T)


class FoldResult:
    """Per-root results of solve_folded.

    omega : ndarray, shape (nout,), Hartree, ascending
    y : ndarray, shape (n, nout), unit singles vectors in the channel basis
    x1 : ndarray, shape (n, nout), sqrt(T1) y, the singles part of the unit full
        eigenvector; embed(x1[:, r]) is the full flat spin-orbital singles vector
    t1 : ndarray, shape (nout,), singles weights in (0, 1]
    level : ndarray, shape (nout,), int, the level of each root; the partners of a
        degenerate level share its index and its omega, and a seed level that
        A_eff splits gives each of its roots a level of its own
    steps : ndarray, shape (nout,), outer iterations used
    loop : list of str, 'newton', 'fixed' or 'index' (solved by branch index) per
        root
    converged : ndarray, shape (nout,), bool
    embed : callable, channel vector -> full flat singles vector

    nout >= nroots unless a RuntimeWarning reports roots not found: a degenerate
    level cut by nroots comes back whole. Level indices need not be consecutive:
    roots solved past the cut and copies of one root are dropped.
    """

    def __init__(self, omega, y, t1, steps, loop, converged, embed, level):
        self.omega, self.y, self.t1 = omega, y, t1
        self.steps, self.loop, self.converged = steps, loop, converged
        self.embed = embed
        self.level = np.asarray(level, int)
        self.x1 = y * np.sqrt(t1)[None, :]


class _Shifted:
    """u -> M u + d u for a diagonal d (n,), with M's block form where it has one."""

    def __init__(self, mv, d):
        self.mv, self.d = mv, np.asarray(d, float)
        if hasattr(mv, 'matmat'):
            self.matmat = lambda U: mv.matmat(U) + self.d[:, None] * U

    def __call__(self, u):
        return self.mv(u) + self.d * np.ravel(u)


def _matmat(mv, U):
    """sum_q A_pq U_qr for every column of U (n, m), in one block where mv has one."""
    if hasattr(mv, 'matmat'):
        return mv.matmat(U)
    return np.column_stack([mv(U[:, r]) for r in range(U.shape[1])])


def _diis_step(hist_omega, hist_err):
    """Pulay step on the scalar history: ω_next = sum_k c_k λ_k, λ_k = ω_k + e_k,
    with the c_k of least norm under sum_k c_k = 1 and sum_k c_k e_k = 0.

    One residual per step makes the Pulay matrix e_k e_l rank one, so from three
    entries on its bordered system is singular. The least-norm weights are
    c_k = a + b e_k, which make ω_next the least-squares line λ = α + β e over
    the history, taken at e = 0: the secant step for two entries, λ for one."""
    e = np.asarray(hist_err, float)
    lam = np.asarray(hist_omega, float) + e
    de = e - e.mean()
    var = float(de @ de)                   # sum_k (e_k - ē)²
    if var == 0.0:                         # one entry, or residuals that stay put
        return float(lam[-1])
    beta = float(de @ (lam - lam.mean())) / var
    return float(lam.mean() - beta * e.mean())


def _levels(e, tol=_DEGENERATE):
    """(start, stop) runs of ascending values e, shape (m,), whose spread, max minus
    min, is at most tol: the degenerate levels, a lone value being a level of one."""
    runs, start = [], 0
    for k in range(1, len(e) + 1):
        if k == len(e) or e[k] - e[start] > tol:
            runs.append((start, k))
            start = k
    return runs


def _doubles_dot(pieces, Ya, Yb):
    """sum_K Ya_K Yb_K over SB doubles, in the metric of the full eigenvector."""
    if pieces.get('operator') is not None:
        # images of an operator's pieces are kept as (u, omega), never formed
        return float(pieces['operator'].doubles_dot(Ya[0], Ya[1], Yb[0], Yb[1]))
    dnorm2 = pieces.get('dnorm2')
    if dnorm2 is not None:
        # polarisation: <a, b> = (|a + b|² - |a - b|²) / 4
        return 0.25 * (float(dnorm2(Ya + Yb)) - float(dnorm2(Ya - Yb)))
    no, nv = pieces['no'], pieces['nv']
    # sum_K Ya_K Yb_K on the flat adc2 layout
    return float(doubles_sb_to_flat(Ya, no, nv) @ doubles_sb_to_flat(Yb, no, nv))


def _doubles_image(pieces, u, omega, embed):
    """Ỹ_K = sum_jb V_K,jb u_jb / (ω - D_K) of a channel vector u, SB doubles."""
    if pieces.get('operator') is not None:
        # the image itself is doubles-sized: keep (u, omega), from which
        # the operator rebuilds the overlap of two images tile by tile
        return (np.asarray(u, float).ravel(), float(omega))
    no, nv, be = pieces['no'], pieces['nv'], pieces['be']
    y1 = singles_flat_to_sb(embed(u), no, nv)
    return be.divide(pieces['V'](y1), omega - pieces['D'])


def _duplicates(omega, y, t1, Yt, dot, tol_omega):
    """Pairs (r, s), r < s, of roots that landed on one root: |ω_r - ω_s| below
    2 tol_omega, tol_omega the caller's half-window (two copies of a root
    converged to tol_omega differ by up to 2 tol_omega), and a
    full-vector overlap

        |x_r · x_s| = sqrt(T1_r T1_s) |y_r · y_s + Ỹ_r · Ỹ_s|

    above _DUPLICATE; it vanishes for distinct roots at any T1, where the singles
    overlap alone need not. omega, t1: shape (nout,); y: shape (n, nout), unit
    singles in the channel basis; Yt: the roots' Ỹ (_doubles_image), indexed by
    root, read only for roots within that window of another;
    dot: the doubles inner product (_doubles_dot)."""
    out = []
    for s in range(len(omega)):
        for r in range(s):
            if abs(omega[s] - omega[r]) >= 2.0 * tol_omega:
                continue
            ov = np.sqrt(t1[r] * t1[s]) * abs(float(y[:, r] @ y[:, s])
                                              + dot(Yt[r], Yt[s]))
            if ov > _DUPLICATE:
                out.append((r, s))
    return out


def _duplicate_drop(r, s, converged):
    """Of two copies r < s of one root, the one to mark unconverged: r when s alone
    converged, so a converged copy is kept, else s (converged: shape (nout,))."""
    return r if converged[s] and not converged[r] else s


def _eig_at(pieces, omega, spin, ref, dense, tol_residual, label):
    """(λ, y, T1) of A_eff(ω): the eigenpair of maximal overlap with ref (n,), by
    eigh below dense_limit, else by Davidson."""
    matvec, dmatvec, diag_s, embed, restrict = folded_operator(pieces, omega, spin)
    n = diag_s.size
    if dense:
        # sum_q A_pq v_qm = w_m v_pm, A built from n matvecs
        A = dense_effective(matvec, n)
        w, v = np.linalg.eigh(A)
        k = int(np.argmax(np.abs(ref @ v)))
        # a degenerate eigenspace has no preferred basis: follow ref's
        # projection onto it, y_p = sum_m v_pm sum_q v_qm ref_q over the m
        # with w_m = w_k, so partners seeded orthogonal stay orthogonal
        cl = np.abs(w - w[k]) < _DEGENERATE
        lam, y = float(w[k]), v[:, cl] @ (v[:, cl].T @ ref)
    else:
        # sum_q A_pq x_q = λ x_p by Davidson on the matvec, A never built;
        # overlap_pick ranks Ritz vectors by |<ref|x>|; past the followed
        # root that ranking is noise and never converges, so follow one
        e, X, conv = solve_symmetric(matvec, diag_s, nroots=1, x0=ref,
                                     pick=overlap_pick(ref),
                                     tol_residual=tol_residual, label=label)
        lam, y = float(e[0]), np.asarray(X[:, 0], float)
    y = y / np.linalg.norm(y)
    ov = abs(float(ref @ y))
    if ov < 0.5:
        warnings.warn(f'{label}: root crossing, overlap with the previous vector '
                      f'{ov:.3f} at omega = {omega:.6f} Ha; the picked vector is '
                      'kept', RuntimeWarning, stacklevel=3)
    t1 = 1.0 / (1.0 + dmatvec(y))
    return lam, y, t1, embed


def _subspace_pick(Y):
    """A davidson1 `pick` keeping the nroots Ritz vectors of largest projection
    |Y^T x|² onto span(Y), Y (n, g) orthonormal, sorted ascending; the score is the
    same for any basis of span(Y), unlike overlap_pick's with one vector."""
    Y = np.asarray(Y, float)

    def pick(w, v, nroots, envs):
        xs = envs['xs']
        # R_ck = sum_p Y_pc (xs_k)_p, the trial vectors on span(Y)
        R = np.array([[float(Y[:, c] @ x) for x in xs] for c in range(Y.shape[1])])
        # score_m = sum_c (sum_k R_ck v_km)², Ritz vector m projected on span(Y)
        score = np.sum((R @ v) ** 2, axis=0)
        idx = np.argsort(-score)[:nroots]
        order = idx[np.argsort(w[idx])]
        return w[order], v[:, order], order

    return pick


def _eig_level(pieces, omega, spin, Y, dense, tol_residual, label):
    """The g eigenpairs of A_eff(ω) of largest projection onto span(Y), Y (n, g)
    orthonormal. Returns λ (g,), Z (n, g) the eigenvectors, ZR (n, g) = Z rotated
    onto Y by the orthogonal Procrustes R, and T1 (g,) of the columns of ZR."""
    matvec, dmatvec, diag_s, _, _ = folded_operator(pieces, omega, spin)
    n, g = Y.shape
    if dense:
        # sum_q A_pq v_qm = w_m v_pm, A built from n matvecs
        w, v = np.linalg.eigh(dense_effective(matvec, n))
        score = np.sum((Y.T @ v) ** 2, axis=0)    # sum_c (sum_p Y_pc v_pm)²
        k = np.sort(np.argsort(-score)[:g])
        lam, Z = w[k], v[:, k]
    else:
        # sum_q A_pq x_qm = λ_m x_pm for the g Ritz vectors of largest projection
        lam, Z, _ = solve_symmetric(matvec, diag_s, nroots=g, x0=Y,
                                    pick=_subspace_pick(Y),
                                    tol_residual=tol_residual, label=label)
        lam, Z = np.asarray(lam, float), np.asarray(Z, float)
    # sum_p Z_pk Y_pc = sum_j U_kj s_j Wt_jc; the orthogonal R_kc = sum_j U_kj Wt_jc
    # minimises sum_pc (sum_k Z_pk R_kc - Y_pc)², and ZR_pc = sum_k Z_pk R_kc
    U, s, Wt = np.linalg.svd(Z.T @ Y)
    if s.min() < 0.5:
        # the per-root path's crossing test, for a subspace: the worst-kept
        # direction of span(Y) overlaps the new span by s_min
        warnings.warn(f'{label}: level crossing, smallest overlap of the level with '
                      f'its previous span {s.min():.3f} at omega = {omega:.6f} Ha; '
                      'the picked vectors are kept', RuntimeWarning, stacklevel=3)
    ZR = Z @ (U @ Wt)
    t1 = np.array([1.0 / (1.0 + dmatvec(ZR[:, j])) for j in range(g)])
    return lam, Z, ZR, t1


def _iterate(step, om, tol_omega, t_min, max_newton, max_fixed, label, verbose):
    """λ(ω) = ω for one root or one level: step(ω, tight) -> (λ, T1, state,
    exact), exact when the eigensolve ran to tol_residual, which tight forces;
    Newton ω + (λ - ω) T1 while T1 >= t_min, else the DIIS fixed point. A
    loose step inside tol_omega is followed by one tight step from its update,
    which is the one that may converge. Returns (λ, T1, state, steps, loop,
    converged, |λ - ω|) of the last step."""
    mode, hist_o, hist_e, k, tight = 'newton', [], [], 0, False
    while True:
        lam, t1, state, exact = step(om, tight)
        err = lam - om
        k += 1
        if verbose:
            print(f'{label} step {k} ({mode}): omega = {om:.8f} '
                  f'lambda = {lam:.8f} T1 = {t1:.4f}', flush=True)
        if abs(err) < tol_omega and exact:
            return lam, t1, state, k, mode, True, abs(err)
        # within tol_omega on a loose eigensolve: one tight step, off the budget
        tight = abs(err) < tol_omega
        if mode == 'newton' and t1 < t_min:
            mode, hist_o, hist_e = 'fixed', [], []
        if mode == 'newton':
            if k >= max_newton and not tight:
                return lam, t1, state, k, mode, False, abs(err)
            om = om + err * t1                         # Newton on λ(ω) - ω
        else:
            hist_o.append(om)
            hist_e.append(err)
            hist_o, hist_e = hist_o[-6:], hist_e[-6:]
            if k >= max_newton + max_fixed and not tight:
                return lam, t1, state, k, mode, False, abs(err)
            om = _diis_step(hist_o, hist_e)


def _inner_tol(err, tol_residual):
    """The residual of a Newton step's eigensolve: _INNER_RATIO times the
    previous step's |λ - ω| (err; None before the first), at most _INNER_CAP,
    never below tol_residual. The Ritz value's error goes as |r|²/gap, so
    this keeps it well under the Newton step's own error while ω is far from
    the root; within a factor 1/_INNER_RATIO of tol_residual the loose solve
    saves too little to pay for the tight step that must follow it, and gives
    way to tol_residual."""
    tol = _INNER_CAP if err is None else min(_INNER_CAP, _INNER_RATIO * err)
    return tol if tol * _INNER_RATIO > tol_residual else tol_residual


def _root_step(pieces, spin, y, dense, tol_residual, label):
    """step(ω, tight) for one root, following the previous vector, its
    eigensolve to _inner_tol, or to tol_residual when tight (eigh: exact)."""
    st = {'ref': y, 'err': None}

    def step(om, tight):
        tol = tol_residual if tight else _inner_tol(st['err'], tol_residual)
        lam, yk, t1, _ = _eig_at(pieces, om, spin, st['ref'], dense, tol, label)
        st['ref'], st['err'] = yk, abs(lam - om)
        return lam, t1, yk, dense or tol <= tol_residual

    return step


def _level_step(pieces, spin, Y, dense, tol_residual, label):
    """step(ω, tight) for a level: the mean eigenvalue and the mean T1 of its
    partners, its eigensolve as _root_step's."""
    st = {'ref': Y, 'err': None}

    def step(om, tight):
        tol = tol_residual if tight else _inner_tol(st['err'], tol_residual)
        lam, Z, ZR, t1 = _eig_level(pieces, om, spin, st['ref'], dense, tol, label)
        st['ref'], st['err'] = ZR, abs(float(lam.mean()) - om)
        return (float(lam.mean()), float(t1.mean()), (lam, Z, ZR, t1),
                dense or tol <= tol_residual)

    return step


def _seed_levels(pieces, spin, m0, diag_s, n, nroots, dense, tol_residual, label):
    """Seeds, lowest first, e (ns,) and X (n, ns), and the levels among them that
    start below nroots, a level cut by nroots whole.

    Dense branch: the eigenpairs of M. Davidson branch: the eigenpairs of
    A_eff(ω*), whose order below min_K D_K is the order of the roots (the
    count's argument in solve_folded), where M's need not be: M's second-order
    singles block and the folded doubles shift its states by different
    amounts. ω* is the median of M's nroots lowest, solved to _M_SEED_RESIDUAL,
    lowered by the doubles shift of M's lowest, (e_0 - λ_0(e_0)) T1; A_eff(ω*)
    is solved to _FOLD_SEED_RESIDUAL, re-solved to _SEED_RESIDUAL when two
    seeds lie within _SEED_SPLIT, so that the levels (spread _DEGENERATE) are
    told apart, and widened until the last level that starts below nroots ends
    before the last seed. e is λ_j(ω*), the first ω of each root's Newton."""
    if dense:
        # sum_q M_pq v_qr = w_r v_pr, M built from n matvecs
        e, X = np.linalg.eigh(dense_effective(m0, n))
        return e, X, [(a, b) for a, b in _levels(e) if a < nroots]
    ns = min(nroots + 2, n)
    # sum_q M_pq x_qr = e_r x_pr, the ns lowest by Davidson, loosely
    e_m, X_m, _ = solve_symmetric(m0, diag_s, nroots=ns,
                                  tol_residual=max(tol_residual, _M_SEED_RESIDUAL),
                                  label=label + ' seeds of M')
    e_m, X_m = np.asarray(e_m, float), np.asarray(X_m, float)
    # the doubles shift of the lowest: one Newton step from e_0
    lam0, _, t10, _ = _eig_at(pieces, float(e_m[0]), spin, X_m[:, 0], dense,
                              max(tol_residual, _M_SEED_RESIDUAL), label)
    w_star = float(np.median(e_m[:nroots])) - (float(e_m[0]) - lam0) * t10
    if w_star >= _d_min(pieces):
        w_star = float(e_m[0]) + (lam0 - float(e_m[0])) * t10
    matvec, _, diag_w, _, _ = folded_operator(pieces, w_star, spin)
    tol = max(tol_residual, _FOLD_SEED_RESIDUAL)
    cols = X_m
    while True:
        # sum_q A_pq(ω*) x_qr = λ_r x_pr, the ns lowest
        e, X, _ = solve_symmetric(matvec, diag_w, nroots=ns,
                                  x0=_guess(cols, diag_w, ns + 4),
                                  tol_residual=tol, label=label + ' seeds')
        e, X = np.asarray(e, float), np.asarray(X, float)
        if tol > _SEED_RESIDUAL and np.any(np.diff(e) < _SEED_SPLIT):
            tol, cols = min(tol_residual, _SEED_RESIDUAL), X
            continue
        runs = [(a, b) for a, b in _levels(e) if a < nroots]
        if runs[-1][1] < ns or ns == n:
            return e, X, runs
        cols, ns = X, min(2 * ns, n)


def _guess(cols, diag_s, width):
    """An orthonormal Davidson start, shape (n, m ≤ width): the columns cols (n, c)
    first, then one fixed random vector, which has a component in every block the
    others miss, then the unit vectors on the lowest entries of diag_s; a vector
    in the span of those before it is skipped and the next seed takes its place."""
    n = diag_s.size
    width = min(width, n)
    rng = np.random.default_rng(0)
    cand = (list(np.reshape(cols, (n, -1)).T[:width - 1]) + [rng.standard_normal(n)]
            + diagonal_seeds(diag_s, min(n, 2 * width)))
    return _extend_orthonormal(np.empty((n, 0)), cand, rtol=_SPAN, limit=width)


def _extend_orthonormal(B, cand, rtol=None, atol=None, limit=None):
    """Q (n, m), orthonormal and orthogonal to B (n, b, orthonormal): each
    candidate in turn with its components along B and the columns kept before it
    removed twice, normalised, and kept while its norm stays above rtol times
    its own or above atol; at most limit columns."""
    Q = np.empty((B.shape[0], 0))
    for g in cand:
        r = g
        for _ in range(2):
            # r_p - sum_j B_pj (B r)_j - sum_j Q_pj (Q r)_j
            r = r - B @ (B.T @ r) - Q @ (Q.T @ r)
        nr = np.linalg.norm(r)
        if nr > (atol if atol is not None else rtol * np.linalg.norm(g)):
            Q = np.column_stack([Q, r / nr])
        if limit is not None and Q.shape[1] == limit:
            break
    return Q


def _lowest_beside(matvec, diag_s, V, w0, top, tol_residual, label):
    """The lowest eigenpair (μ, z (n,)) of A_eff off the span of V (n, m), its
    columns orthonormal near-eigenvectors with eigenvalues ≥ w0: Davidson on
    A' = A_eff + σ V V^T, σ = top - w0 + 1 Ha, which lifts them 1 Ha above top,
    with the shifted diagonal correction and to residual
    max(tol_residual, _CHECK_RESIDUAL): the Ritz value bounds μ from above, so a
    value at or below top is a branch there at any residual. μ > top says A_eff
    has no eigenvalue at or below top off span V. Two starts, off span V: a
    random vector with the unit vectors on the _CHECK_SEEDS lowest entries of
    the lifted diagonal, then, if that finds μ above top, the random vector
    alone."""
    n = diag_s.size
    sigma = top - w0 + 1.0

    def lifted(u):
        # sum_q A'_pq u_q = sum_q A_pq u_q + σ sum_j V_pj (sum_q V_qj u_q)
        return matvec(u) + sigma * (V @ (V.T @ u))

    diag = diag_s + sigma * np.sum(V ** 2, axis=1)

    def precond(res, e, u):
        # res_p / (d_p - e + shift): the Jacobi-Davidson projection amplifies the
        # configurations whose diagonal of M sits at e and walks down M's diagonal
        # past a state that A_eff pulls far below it; without the shift the
        # correction equals u where A_eff is diagonal, and the solve stalls
        d = diag - e + _CHECK_SHIFT
        return np.asarray(res) / np.where(np.abs(d) < 1e-8, 1e-8, d)

    # a new vector per span: in a degenerate level the previous vector's whole
    # component went to the partner it found, which V now holds
    r = np.random.default_rng(V.shape[1]).standard_normal(n)
    # the low configurations carry a missed low state directly, which the
    # random vector holds at weight ~ n^(-1/2) only and can converge past; the
    # random vector alone walks down M's diagonal to a state A_eff pulls far
    # below its configurations, which the low seeds never reach
    for cand in ([r] + diagonal_seeds(diag, _CHECK_SEEDS), [r]):
        # sum_q A'_pq z_q = μ z_p, the lowest, by Davidson from the start
        mu, z, _ = solve_symmetric(lifted, diag, nroots=1,
                                   x0=_extend_orthonormal(V, cand, rtol=_SPAN),
                                   precond=precond,
                                   tol_residual=max(tol_residual, _CHECK_RESIDUAL),
                                   label=label + ' check')
        if float(mu[0]) <= top:
            break
    return float(mu[0]), np.asarray(z, float)[:, 0]


def _lowest_at(pieces, omega, spin, k, dense, tol_residual, label, cols, tol=None,
               check=False):
    """The k lowest eigenpairs of A_eff(ω), λ (m,) ascending and X (n, m), and
    dmatvec at ω; with tol also every one with λ_j ≤ ω + tol (m ≥ k). Davidson
    starts from cols (n, c), vectors expected near that span, padded by _guess;
    with tol it widens from its own Ritz vectors until the highest returned lies
    above ω + tol, and with check a search off their span (_lowest_beside) then
    looks for an eigenvalue at or below ω + tol the start hid, which joins the
    next start."""
    matvec, dmatvec, diag_s, _, _ = folded_operator(pieces, omega, spin)
    n = diag_s.size
    k = min(k, n)
    if dense:
        # sum_q A_pq v_qm = w_m v_pm, A built from n matvecs
        w, v = np.linalg.eigh(dense_effective(matvec, n))
    else:
        while True:
            # sum_q A_pq x_qm = w_m x_pm, the k lowest by Davidson
            w, v, _ = solve_symmetric(matvec, diag_s, nroots=k,
                                      x0=_guess(cols, diag_s, 2 * k + 4),
                                      tol_residual=tol_residual,
                                      label=label + ' count')
            w, v = np.asarray(w, float), np.asarray(v, float)
            if tol is None or k == n:
                break
            if w[-1] <= omega + tol:
                cols, k = v, min(2 * k, n)
                continue
            if not check:
                break
            # a start whose vectors are exact eigenvectors converges at once
            # and can hide a branch below ω + tol: look for one off span v
            mu, z = _lowest_beside(matvec, diag_s, v, float(w[0]), omega + tol,
                                   tol_residual, label)
            if mu > omega + tol:
                break
            cols, k = np.column_stack([v, z]), k + 1
    m = k if tol is None else max(k, int(np.count_nonzero(w <= omega + tol)))
    return w[:m], v[:, :m], dmatvec


def _index_solve(pieces, spin, a, b, om, lo, hi, X, dense, tol_residual, tol_omega,
                 max_steps, label, verbose):
    """The root of the branches a..b-1 of A_eff, counted from the lowest (one
    branch, or a degenerate group solved at one ω): λ̄(ω) = ω, λ̄ the mean of
    λ_j(ω) over the group. Below min_K D_K λ̄(ω) - ω falls strictly, so the root
    is unique and stays inside [lo, hi]; Newton ω + (λ̄ - ω) T̄1 from om,
    bisection when a step leaves the bracket, a stop (unconverged) once the bracket
    is narrower than tol_omega, where the eigensolver's precision limits |λ̄ - ω|.
    X (n, c): the cut's eigenvectors, the first Davidson start and part of every
    later one, beside the previous step's. Returns λ_j (g,), Z (n, g), T1 (g,),
    steps, converged and |λ̄ - ω| of the last step."""
    k, X_c = 0, X
    while True:
        # a branch the Newton point moved out of the b lowest is found again from
        # the cut's vectors, where it was low
        lam, X, dmv = _lowest_at(pieces, om, spin, b, dense, tol_residual, label,
                                 np.column_stack([X, X_c]) if k else X)
        lam, Z = lam[a:b], X[:, a:b]
        # T1_j = 1 / (1 + sum_K Y_Kj²), Y the doubles image of the unit y_j
        t1 = np.array([1.0 / (1.0 + dmv(Z[:, j])) for j in range(b - a)])
        err = float(lam.mean()) - om
        k += 1
        if verbose:
            print(f'{label} branch {a} step {k} (index): omega = {om:.8f} '
                  f'lambda = {lam.mean():.8f} T1 = {t1.mean():.4f}', flush=True)
        if abs(err) < tol_omega or k >= max_steps:
            return lam, Z, t1, k, abs(err) < tol_omega, abs(err)
        lo, hi = (om, hi) if err > 0 else (lo, om)
        if hi - lo < tol_omega:
            return lam, Z, t1, k, False, abs(err)
        nxt = om + err * float(t1.mean())          # Newton on λ̄(ω) - ω
        om = nxt if lo < nxt < hi else 0.5 * (lo + hi)


def _node_images(pieces, spin):
    """images(U, nodes) -> (M U, F), U (n, m), F (p, n, m) with
    F[q] = A_eff(nodes[q]) U - M U: the operator's own, where it has one,
    else from folded_operator at each frequency."""
    op = pieces.get('operator')
    if op is not None:
        return op.fold_images
    m0 = folded_operator(pieces, None, spin)[0]

    def images(U, nodes):
        MU = _matmat(m0, U)
        F = np.empty((len(nodes),) + MU.shape)
        for q, w in enumerate(nodes):
            F[q] = _matmat(folded_operator(pieces, float(w), spin)[0], U) - MU
        return MU, F

    return images


def _chebyshev_nodes(lo, hi, p):
    """The p Chebyshev points of [lo, hi]."""
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(
        (2 * np.arange(p) + 1) * np.pi / (2 * p))


def _node_count(lo, hi, d_min, tol_omega, fold):
    """Nodes on [lo, hi] for which the interpolated fold, of size up to fold
    (Hartree), is within EE_FOLD_BLOCK_NODE_TARGET tol_omega: relative error
    2 (h/2)^p / (d - h)^p for every pole at or above d_min (constants.py)."""
    h, d = 0.5 * (hi - lo), d_min - 0.5 * (lo + hi)
    p = 2
    while (p < EE_FOLD_BLOCK_MAX_NODES and 2.0 * (h / (2.0 * (d - h))) ** p
           * fold > EE_FOLD_BLOCK_NODE_TARGET * tol_omega):
        p += 1
    return p


def _lagrange(nodes, w):
    """(l, dl): the Lagrange basis polynomials on nodes at w, and dl/dw."""
    p = nodes.size
    l, dl = np.empty(p), np.empty(p)
    for j in range(p):
        x = np.delete(nodes, j)
        den = np.prod(nodes[j] - x)
        f = w - x
        l[j] = np.prod(f) / den
        # d/dw prod_k (w - x_k) = sum_i prod_{k != i} (w - x_k)
        dl[j] = sum(np.prod(np.delete(f, i)) for i in range(p - 1)) / den
    return l, dl


def _model_root(Mk, Fk, nodes, r, w0, lo, hi):
    """The root λ_r(ω) = ω of the subspace model, λ_r the r-th eigenvalue of
    Mk + sum_q l_q(ω) Fk_q: Newton from w0, bisection once a bracket inside
    [lo, hi] is known and a step leaves it. Returns ω, the Ritz coefficients
    c (k,), the weights l (p,) at ω."""
    a, b, w = lo, hi, w0
    for _ in range(EE_FOLD_MODEL_ROOT_MAX_ITER):
        l, dl = _lagrange(nodes, w)
        e, v = np.linalg.eigh(Mk + np.tensordot(l, Fk, 1))
        c = v[:, r]
        g = e[r] - w
        # dg/dω = c (sum_q dl_q Fk_q) c - 1 < 0 below min D
        dg = float(c @ np.tensordot(dl, Fk, 1) @ c) - 1.0
        if g > 0:
            a = max(a, w)
        else:
            b = min(b, w)
        nxt = w - g / dg
        if not a < nxt < b:
            nxt = 0.5 * (a + b)
        if abs(nxt - w) <= EE_FOLD_MODEL_ROOT_STEP * max(1.0, abs(w)):
            break
        w = nxt
    return w, c, l


def _block_roots(pieces, spin, X0, nroots, k, lo, hi, fold, tol_residual, tol_omega,
                 label, verbose):
    """The nroots lowest folded roots by one block Davidson from the columns of
    X0, ω_r and y_r (n, k) for the k lowest branches (nroots of them asked, the
    rest carried as Ritz pairs that add no corrections); fold bounds
    |y A_fold y| for the node count.

    Every trial vector v enters with M v and F(ω_q) v at the Chebyshev nodes
    ω_q of [lo, hi] (_node_count of them), so the subspace model
    A(ω) = V^T M V + sum_q l_q(ω) V^T F_q V holds every root at its own ω;
    root r is λ_r(ω) = ω of the model, its residual
    (M + sum_q l_q(ω_r) F_q) y_r - ω_r y_r, its correction the Jacobi-Davidson
    one on diag(M) + y_r F(ω_r) y_r. Near-degenerate roots share the subspace,
    so no root follows a vector of its partner's. An asked root that leaves
    [lo, hi] widens the window to it (EE_FOLD_BLOCK_MARGIN beyond), and a
    root whose fold y_r F y_r needs more nodes than the window has raises
    their count; either restarts from the k Ritz vectors, at most
    EE_FOLD_BLOCK_MAX_MOVES times; a window reaching min_K D_K stops there. Stops when the nroots
    lowest have residual below tol_residual and ω moved less than tol_omega.
    Returns ω (k,), Y (n, k), the iterations and ritz(ω, m), the m lowest
    eigenvectors of the subspace model at ω."""
    _, _, diag_s, _, _ = folded_operator(pieces, None, spin)
    images = _node_images(pieces, spin)
    d_min = _d_min(pieces)
    V, om, moves = X0, None, 0
    for it in range(1, EE_FOLD_BLOCK_MAX_ITER + 1):
        if it == 1 or moved:
            nodes = _chebyshev_nodes(lo, hi, _node_count(lo, hi, d_min, tol_omega,
                                                         fold))
            h = 0.5 * (hi - lo)
            V = np.linalg.qr(V)[0]
            MV, FV = images(V, nodes)
        moved = False
        # the projected pieces, symmetrised against rounding
        Mk = V.T @ MV
        Mk = 0.5 * (Mk + Mk.T)
        Fk = np.einsum('pk,qpl->qkl', V, FV)
        Fk = 0.5 * (Fk + Fk.transpose(0, 2, 1))
        if om is None:
            l0, _ = _lagrange(nodes, 0.5 * (lo + hi))
            om = np.linalg.eigvalsh(Mk + np.tensordot(l0, Fk, 1))[:k]
        new, C, res, folds = [], [], np.empty(k), np.zeros(nroots)
        conv = np.zeros(k, bool)
        om_prev, om = om, np.empty(k)
        for r in range(k):
            om[r], c, l = _model_root(Mk, Fk, nodes, r, om_prev[r],
                                      lo - EE_FOLD_BLOCK_BRACKET * h,
                                      hi + EE_FOLD_BLOCK_BRACKET * h)
            y = V @ c
            # (M + sum_q l_q F_q) y - ω y from the stored images
            rv = MV @ c + np.tensordot(l, FV, 1) @ c - om[r] * y
            res[r] = np.linalg.norm(rv)
            C.append(c)
            conv[r] = res[r] < tol_residual and abs(om[r] - om_prev[r]) < tol_omega
            if r < nroots:
                # the root's fold y F(ω_r) y, which the node count must cover
                folds[r] = float(c @ np.tensordot(l, Fk, 1) @ c)
                if not conv[r]:
                    new.append(jacobi_davidson_precond(diag_s + folds[r])(rv, om[r], y))
        # the subspace C belongs to: a move or new corrections replace V below
        Vc, MVc, FVc, nodes_c = V, MV, FV, nodes
        if verbose:
            print(f'{label} block {it}: {V.shape[1]} vectors, {nodes.size} nodes, '
                  f'max residual {res[:nroots].max():.1e}', flush=True)
        if conv[:nroots].all():
            break
        outside = om[:nroots].min() < lo or om[:nroots].max() > hi
        bound = EE_FOLD_BLOCK_FOLD_BOUND * float(np.abs(folds).max())
        if (outside or _node_count(lo, hi, d_min, tol_omega, bound) > nodes.size) \
                and moves < EE_FOLD_BLOCK_MAX_MOVES:
            # outside the window the interpolant extrapolates, and a root's fold
            # beyond the one the nodes were counted for is not resolved: move
            # the nodes to the roots and restart from the k Ritz vectors
            if outside:
                lo = min(lo, float(om[:nroots].min()) - EE_FOLD_BLOCK_MARGIN)
                hi = max(hi, float(om[:nroots].max()) + EE_FOLD_BLOCK_MARGIN)
            if hi >= d_min:
                break
            fold = max(fold, bound)
            V, moved, moves = V @ np.column_stack(C), True, moves + 1
            continue
        # the corrections, orthonormal to V and to each other (twice)
        Z = _extend_orthonormal(V, new, atol=EE_FOLD_BLOCK_LINDEP)
        if Z.shape[1] == 0:
            break
        MZ, FZ = images(Z, nodes)
        V, MV = np.column_stack([V, Z]), np.column_stack([MV, MZ])
        FV = np.concatenate([FV, FZ], axis=2)
    Y = Vc @ np.column_stack(C)

    def ritz(w, m):
        # the m lowest eigenvectors of the model at w, (n, m)
        l, _ = _lagrange(nodes_c, w)
        A = Vc.T @ (MVc + np.tensordot(l, FVc, 1))
        return Vc @ np.linalg.eigh(0.5 * (A + A.T))[1][:, :m]

    return om, Y, it, ritz


def _exact_roots(pieces, spin, om, Y):
    """λ_r = y_r A_eff(ω_r) y_r, the residual |A_eff(ω_r) y_r - ω_r y_r| and
    T1 at each column's own ω, from exact products, y_r normalised."""
    Y = Y / np.linalg.norm(Y, axis=0)
    op = pieces.get('operator')
    if op is not None:
        AY, nrm = op.apply(Y, om)
    else:
        AY, nrm = np.empty_like(Y), np.empty(om.size)
        for r, w in enumerate(om):
            mv, dmv, _, _, _ = folded_operator(pieces, float(w), spin)
            AY[:, r], nrm[r] = mv(Y[:, r]), dmv(Y[:, r])
    lam = np.einsum('pk,pk->k', Y, AY)
    res = np.linalg.norm(AY - Y * om, axis=0)
    return lam, res, 1.0 / (1.0 + nrm), Y


def solve_folded(pieces, nroots, spin=None, tol_omega=1e-6, tol_residual=1e-6,
                 t_min=0.3, max_newton=12, max_fixed=30, dense_limit=None,
                 method='newton', verbose=0):
    """The nroots lowest folded roots, each level at its own frequency.

    method 'block' (Davidson branch only) solves them together instead of
    one by one (_block_roots): the nroots + 2 lowest of M, loosely, or, where
    the operator models the fold's diagonal (fold_diagonal), of M plus that
    model, which ranks the configurations the doubles pull furthest down where
    the fold puts them, start one block Davidson whose trial vectors carry the
    folded doubles at Chebyshev frequencies spanning the roots (the window: one exact Newton step of each
    seed, widened by 0.01 Ha); each root is λ_r(ω) = ω of the interpolated
    subspace model. A root is converged only by an exact product at its own
    ω (|λ - ω| < tol_omega, residual < tol_residual); one that fails, or a
    window that reaches min_K D_K, goes to the per-root Newton below from
    the block's vector. Completeness: a Davidson off the span of the found
    roots (_lowest_beside at the highest root ω_c), from a random vector with
    the lowest diagonal unit vectors and then from the random vector alone, looks for
    an eigenvalue of A_eff(ω_c) at or below ω_c; one it finds restarts the
    block with it. The count below is not run.

    Seeds: for n ≤ dense_limit the eigenpairs of the bare singles block M on the
    channel (eigh); on the Davidson branch the lowest eigenpairs of A_eff(ω*), ω*
    near the roots (M's seeds solved loosely, lowered by the doubles shift of the
    lowest), to residual 1e-4 and to 1e-9 when two lie within 1e-5 Ha
    (_seed_levels); grouped into levels whose eigenvalues spread by at most 1e-8
    Ha. Below min_K D_K the order of A_eff's eigenvalues at one ω is the order of
    the roots, so these seeds start each root on its own branch where M's,
    reordered by the fold, need not. A level of one is solved per root: λ, y
    at ω_k by eigh or Davidson with overlap following, the Davidson to residual
    min(1e-3, 0.1 |λ - ω| of the previous step), never below tol_residual, then
    the Newton step ω_{k+1} = ω_k + (λ - ω_k) T1 while T1 ≥ t_min, else the
    DIIS-accelerated fixed point ω_{k+1} = λ; a loose step inside tol_omega is
    followed by one more at tol_residual, which alone may converge. A level of
    g > 1 is solved jointly: one ω, the g eigenpairs
    of A_eff(ω) of largest projection onto the span of its partners, rotated onto
    them (orthogonal Procrustes), Newton on the mean eigenvalue with the mean T1;
    the partners come back at one ω, orthonormal, with one level index, their T1
    equal to the vectors' accuracy (on the Davidson branch set by tol_residual:
    3.6e-9 at 1e-6 on CH4's T2 triple). A level whose partners leave their previous
    span (overlap below 0.5) warns, as a root crossing does. A level whose g
    eigenvalues spread by more than 1e-8 Ha at convergence is re-solved per root,
    with a RuntimeWarning. nroots = 0 returns an empty result. Stops at
    |λ - ω| < tol_omega; a root or a level that exhausts max_newton (then
    max_fixed) steps comes back converged False with a RuntimeWarning. A level cut
    by nroots comes back whole: up to g - 1 roots more than asked. Two roots that
    land on one root (|Δω| below the larger of 2 tol_omega and tol_residual,
    full-vector overlap above 0.5) warn, and one copy is dropped from the result,
    the stalled one if the other converged, else the later; while ω_c (below)
    lies under min_K D_K, the count check then solves the root it missed.
    Warnings name a root by its ω and a level by
    its FoldResult.level index, since the result is sorted by ω; a split level's
    roots count its joint steps in their own.

    Completeness (method 'newton'): below min_K D_K each eigenvalue λ_j(ω) of
    A_eff(ω), counted from the lowest, falls with ω (dλ/dω ≤ 0), so λ_j(ω) = ω
    has one root ω_j, and
    ω_1 ≤ ω_2 ≤ ...: the folded roots at or below ω number the λ_j(ω) ≤ ω, and
    the nroots lowest roots are those of branches 1 to nroots. The fold can
    reorder the seeds of M, so at ω_c, the nroots-th lowest root found, the
    branches below the window max(2 tol_omega, 1e-8 Ha) T1 around ω_c are
    compared with the roots found there; a branch inside the window is
    degenerate with ω_c to that resolution. When roots are missing, or fewer
    than nroots were found, each root found gets its branch index from the
    branches below its own window, or, where that window holds more branches
    than roots, from the window branches its vector projects onto most; every
    branch up to nroots that no root covers is solved by its index: Newton on
    λ_j(ω) - ω inside the
    bracket between λ_j(ω_c) and ω_c, bisection when a step leaves it, a group of
    branches degenerate at ω_c at one ω (loop 'index'). At most 3 rounds, then a
    RuntimeWarning names the count; one also warns when more roots are found at or
    below ω_c than the count admits. With ω_c at or above min_K D_K the count runs
    at the highest root found below min_K D_K and checks the branches up to it; a
    result reaching min_K D_K warns that the roots from there up are not checked.
    Roots solved past the cut are dropped. The Davidson solves of the count start from
    the roots found, one random vector and M's lowest diagonal entries; where those are
    exact eigenvectors of A_eff the solve converges at once and can hide a branch, so at
    the first cut, the highest, a second Davidson, from a random vector alone and off
    the span of the count's vectors, looks for an eigenvalue in the window or below it;
    one it finds joins the count's start, and the count's vectors join the start of
    every per-root count and of every step of the index solve.

    Limits: the vectors of two distinct roots split by δ are determined to about
    tol_residual/δ on the Davidson branch, and to the ω error times |dA/dω|/δ on
    either, so a pair closer than the ω accuracy has ill-determined vectors; a
    near-degeneracy the fold creates, absent from M, can collapse two seeds onto one
    root, which the duplicate check reports.

    Parameters
    ----------
    pieces : dict, from ``build_operator(..., pieces=True)`` at adc2 or gf2, or
        ``ee_gw_pieces.build_pieces_gw`` (level gw).
    nroots : int
    spin : {'singlet', 'triplet', None}
    tol_omega, tol_residual : float, Hartree and residual norm.
    t_min : float, singles weight below which the fixed-point loop takes over.
    max_newton, max_fixed : int, step budgets per root or level.
    dense_limit : int or None, channel size at or below which A_eff is built and
        eigh'd; None takes pieces['dense_limit'], 0 (Davidson at every size) when
        the pieces set none.
    verbose : int, 1 prints one line per outer step.

    Returns
    -------
    FoldResult
    """
    _check_level(pieces)
    m0, _, diag_s, embed, _ = folded_operator(pieces, None, spin)
    n = diag_s.size
    if dense_limit is None:
        dense_limit = pieces.get('dense_limit', 0)
    dense = n <= dense_limit
    nroots = int(nroots)
    if nroots < 0:
        raise ValueError(f'nroots={nroots}; expected a count >= 0')
    if nroots == 0:
        return FoldResult(np.empty(0), np.empty((n, 0)), np.empty(0),
                          np.zeros(0, int), [], np.zeros(0, bool), embed,
                          np.empty(0, int))
    if n < nroots:
        warnings.warn(f'the {spin or "combined"} channel holds {n} states, fewer '
                      f'than nroots={nroots}; returning {n}', RuntimeWarning,
                      stacklevel=2)
        nroots = n
    if method not in ('newton', 'block'):
        raise ValueError(f"method={method!r}; expected 'newton' or 'block'")
    label = f'ee-ADC fold ({spin or "both"})'
    # per root, appended as levels are solved; copy marks a duplicate's copy
    omega, y_out, t1_out, level, steps, loop, converged, copy = ([] for _ in range(8))
    kw = dict(tol_omega=tol_omega, t_min=t_min, max_newton=max_newton,
              max_fixed=max_fixed, verbose=verbose)
    nlev = 0

    def solve_root(om, y, lv, k0=0):
        step = _root_step(pieces, spin, y, dense, tol_residual, label)
        lam, t1, yk, k, mode, conv, err = _iterate(
            step, om, label=f'{label} root {len(omega)}', **kw)
        if not conv:
            # named by ω: the result is re-sorted, so the seed index is not the
            # caller's index
            warnings.warn(f'{label}: the root at omega = {lam:.8f} Ha not converged '
                          f'after {k} steps ({mode}), |lambda - omega| = {err:.2e} Ha',
                          RuntimeWarning, stacklevel=4)
        for out, v in zip((omega, y_out, t1_out, level, steps, loop, converged, copy),
                          (lam, yk, t1, lv, k + k0, mode, conv, False)):
            out.append(v)

    def solve_seeds(e, X, runs):
        """Solve the levels runs, (start, stop) into the seeds e (m,), X (n, m)."""
        nonlocal nlev
        for a, b in runs:
            lv, nlev = nlev, nlev + 1
            if b - a == 1:
                # λ(ω) = ω on the root's own followed vector
                solve_root(float(e[a]), X[:, a], lv)
                continue
            # λ̄(ω) = ω, λ̄ the mean of the b - a eigenvalues of largest projection
            # onto the span of the level's seeds X[:, a:b]
            step = _level_step(pieces, spin, X[:, a:b], dense, tol_residual,
                               label)
            lbar, _, (lam, Z, ZR, t1), k, mode, conv, err = _iterate(
                step, float(e[a:b].mean()), label=f'{label} level {lv}', **kw)
            if conv and lam.max() - lam.min() > _DEGENERATE:
                warnings.warn(f'{label}: level {lv} at omega = {lbar:.8f} Ha split '
                              f'by {lam.max() - lam.min():.1e} Ha in A_eff, not '
                              'degenerate; re-solved per root, each a level of its '
                              'own', RuntimeWarning, stacklevel=3)
                # λ_j(ω) = ω per root, each from the level's final eigenvector
                # Z_pj; no partners, one level each, the joint steps counted too
                for j in range(b - a):
                    if j:
                        lv, nlev = nlev, nlev + 1
                    solve_root(float(lam[j]), Z[:, j], lv, k0=k)
                continue
            if not conv:
                warnings.warn(f'{label}: level {lv} at omega = {lbar:.8f} Ha not '
                              f'converged after {k} steps ({mode}), |lambda - omega| '
                              f'= {err:.2e} Ha', RuntimeWarning, stacklevel=3)
            for j in range(b - a):
                for out, v in zip((omega, y_out, t1_out, level, steps, loop,
                                   converged, copy),
                                  (lbar, ZR[:, j], t1[j], lv, k, mode, conv, False)):
                    out.append(v)

    def mark_copies():
        """Mark the later copy of two roots that landed on one root, and warn."""
        om, Y = np.asarray(omega), np.column_stack(y_out)
        # two copies differ by up to 2 tol_omega, and on the Davidson branch also
        # by the eigenvalue error |r|²/gap, which in a collapsing pair passes
        # tol_omega: the window is the larger of 2 tol_omega and tol_residual.
        # An image is a whole doubles vector: build it only for a root within
        # the window of another, the only pairs _duplicates compares
        half = max(tol_omega, 0.5 * tol_residual)
        near = {i for s in range(om.size) for r in range(s)
                if abs(om[s] - om[r]) < 2.0 * half for i in (r, s)}
        Yt = {r: _doubles_image(pieces, Y[:, r], om[r], embed) for r in near}
        for r, s in _duplicates(om, Y, np.asarray(t1_out), Yt,
                                lambda Ya, Yb: _doubles_dot(pieces, Ya, Yb), half):
            if copy[r] or copy[s]:
                continue
            drop = _duplicate_drop(r, s, converged)
            keep = r + s - drop
            warnings.warn(f'{label}: two roots landed on one root, at omega = '
                          f'{om[keep]:.8f} and {om[drop]:.8f} Ha; the copy at '
                          f'{om[drop]:.8f} Ha dropped', RuntimeWarning, stacklevel=3)
            converged[drop], copy[drop] = False, True

    def live():
        """The roots that are not copies, ascending in ω."""
        return sorted((r for r in range(len(omega)) if not copy[r]),
                      key=lambda r: omega[r])

    def count_below(om, k, cols, check=False):
        """(N_lo, N, λ, X, dmatvec): N_lo the branches of A_eff(om) with
        λ_j(om) < om - tol, their roots below om's window; N those with
        λ_j(om) ≤ om + tol, their roots at or below om; λ, X the lowest
        eigenpairs, at least k; check searches off the count's span."""
        lam, X, dmv = _lowest_at(pieces, om, spin, k, dense, tol_residual, label,
                                 cols, tol, check)
        return (int(np.count_nonzero(lam < om - tol)),
                int(np.count_nonzero(lam <= om + tol)), lam, X, dmv)

    def solve_index(a, b, lam_c, X_c, dmv_c, cut):
        """Solve branches a..b-1 (0-based) by index from their eigenpairs at the
        cut, λ_j(ω_c) = lam_c[j], x_j = X_c[:, j]; append them as one level."""
        nonlocal nlev
        lam_g = float(lam_c[a:b].mean())
        t1_g = float(np.mean([1.0 / (1.0 + dmv_c(X_c[:, j])) for j in range(a, b)]))
        # the root lies between λ̄(ω_c) and ω_c; Newton from the cut to start
        lam, Z, t1, k, conv, err = _index_solve(
            pieces, spin, a, b, cut + (lam_g - cut) * t1_g, min(lam_g, cut),
            max(lam_g, cut), X_c, dense, tol_residual, tol_omega,
            max_newton + max_fixed, label, verbose)
        if conv and lam.max() - lam.min() > _DEGENERATE:
            warnings.warn(f'{label}: branches {a} to {b - 1} split by '
                          f'{lam.max() - lam.min():.1e} Ha in A_eff, not degenerate; '
                          're-solved per branch', RuntimeWarning, stacklevel=3)
            for j in range(a, b):
                solve_index(j, j + 1, lam_c, X_c, dmv_c, cut)
            return
        if not conv:
            why = ('' if k >= max_newton + max_fixed else '; its bracket closed '
                   'first, so tol_residual limits it')
            warnings.warn(f'{label}: the root of branch {a} at omega = '
                          f'{lam.mean():.8f} Ha not converged after {k} steps '
                          f'(index), |lambda - omega| = {err:.2e} Ha{why}',
                          RuntimeWarning, stacklevel=3)
        lv, nlev = nlev, nlev + 1
        for j in range(b - a):
            for out, v in zip((omega, y_out, t1_out, level, steps, loop, converged,
                               copy),
                              (float(lam.mean()), Z[:, j], t1[j], lv, k, 'index',
                               conv, False)):
                out.append(v)

    def solve_block():
        """All nroots at once (_block_roots); False when the window reaches
        min_K D_K, where the interpolation has no bound."""
        nonlocal nlev
        k = min(nroots + EE_FOLD_BLOCK_BUFFER, n)
        # sum_q M_pq x_qr = e_r x_pr, the k lowest, loosely: the start
        e_m, X_m, _ = solve_symmetric(m0, diag_s, nroots=k,
                                      tol_residual=max(tol_residual, _M_SEED_RESIDUAL),
                                      label=label + ' seeds of M')
        e_m, X_m = np.asarray(e_m, float), np.asarray(X_m, float)
        fold_diag = getattr(pieces.get('operator'), 'fold_diagonal', None)
        if fold_diag is not None:
            # the doubles pull some configurations (charge transfer) far below
            # where M puts them: the seeds are the k lowest of M plus the
            # model of the fold's diagonal, e their M Rayleigh quotients
            d = fold_diag(float(np.median(e_m[:nroots])))
            _, X_m, _ = solve_symmetric(_Shifted(m0, d), diag_s + d, nroots=k,
                                        tol_residual=max(tol_residual,
                                                         _M_SEED_RESIDUAL),
                                        label=label + ' seeds of M + fold model')
            X_m = np.asarray(X_m, float)
            e_m = np.einsum('pr,pr->r', X_m, _matmat(m0, X_m))
        # one exact Newton step per asked seed, ω_1 = e + (λ(e) - e) T1(e);
        # λ(e) - e = y A_fold(e) y is the seed's fold
        lam, _, t1, _ = _exact_roots(pieces, spin, e_m[:nroots], X_m[:, :nroots])
        fold = lam - e_m[:nroots]
        w1 = e_m[:nroots] + fold * t1
        lo, hi = w1.min() - EE_FOLD_BLOCK_MARGIN, w1.max() + EE_FOLD_BLOCK_MARGIN
        if hi >= _d_min(pieces):
            warnings.warn(f'{label}: the roots reach the lowest doubles energy '
                          f'{_d_min(pieces):.6f} Ha; solved per root',
                          RuntimeWarning, stacklevel=3)
            return False
        fold = EE_FOLD_BLOCK_FOLD_BOUND * float(np.abs(fold).max())
        win = max(2.0 * tol_omega, _DEGENERATE)
        X0 = X_m
        for rnd in range(_COUNT_ROUNDS + 1):
            # the model converges below tol_residual, leaving the rest to the
            # interpolation error the exact check sees
            om, Y, it, ritz = _block_roots(pieces, spin, X0, nroots, k, lo, hi, fold,
                                           EE_FOLD_BLOCK_MODEL_RESIDUAL * tol_residual,
                                           tol_omega, label, verbose)
            # a degenerate level cut by nroots comes back whole
            nr = nroots
            while nr < om.size and om[nr] - om[nr - 1] <= _DEGENERATE:
                nr += 1
            # completeness: below min_K D_K the roots at or below ω_c number the
            # eigenvalues of A_eff(ω_c) at or below it; one off the span of the
            # found roots (the model's eigenvectors at ω_c) is a branch the
            # block's start missed, which a Davidson from a random vector finds
            cut = float(om[nr - 1])
            mu, z = _lowest_beside(folded_operator(pieces, cut, spin)[0], diag_s,
                                   ritz(cut, nr), float(om[0]), cut + win,
                                   tol_residual, label)
            if verbose:
                print(f'{label} check: mu - cut {mu - cut:.2e} Ha, window {win:.1e}',
                      flush=True)
            if mu > cut + win:
                break
            if rnd == _COUNT_ROUNDS:
                warnings.warn(f'{label}: a branch at or below omega = {cut:.8f} Ha '
                              f'is still outside the block after {rnd} restarts',
                              RuntimeWarning, stacklevel=3)
                break
            # its root lies in [μ, ω_c]; one exact Newton step from ω_c places
            # the window, and the block restarts with it
            lz, _, tz, _ = _exact_roots(pieces, spin, np.array([cut]), z[:, None])
            wz = cut + (float(lz[0]) - cut) * float(tz[0])
            X0, lo = np.column_stack([Y, z]), min(lo, wz - EE_FOLD_BLOCK_MARGIN)
        lam, res, t1, Y = _exact_roots(pieces, spin, om[:nr], Y[:, :nr])
        for a, b in _levels(om[:nr]):
            lv = None
            for r in range(a, b):
                if abs(lam[r] - om[r]) < tol_omega and res[r] < tol_residual:
                    if lv is None:
                        lv, nlev = nlev, nlev + 1
                    for out, v in zip((omega, y_out, t1_out, level, steps, loop,
                                       converged, copy),
                                      (float(om[r]), Y[:, r], t1[r], lv, it,
                                       'block', True, False)):
                        out.append(v)
                    continue
                # λ(ω) = ω by Newton on this root's own followed vector
                solve_root(float(om[r]), Y[:, r], nlev, k0=it)
                nlev += 1
        return True

    block = not dense and method == 'block' and solve_block()
    if not block:
        e0, X0, runs = _seed_levels(pieces, spin, m0, diag_s, n, nroots, dense,
                                    tol_residual, label)
        solve_seeds(e0, X0, runs)
    mark_copies()
    # the count check: below min_K D_K the folded roots at or below ω number the
    # eigenvalues λ_j(ω) ≤ ω of A_eff(ω), and the j-th lowest root is the root
    # of the j-th branch; a branch no root found covers is solved by its index.
    # A level spreads by up to _DEGENERATE and is solved at its mean, so the
    # window holds that spread too
    tol = max(2.0 * tol_omega, _DEGENERATE)
    d_min = _d_min(pieces)
    for rnd in range(0 if block else _COUNT_ROUNDS + 1):
        L = live()
        found = np.array([omega[r] for r in L])
        t1f = np.array([t1_out[r] for r in L])
        cut = float(found[min(nroots, found.size) - 1])
        deficit = max(nroots - found.size, 0)
        upto = nroots
        if cut >= d_min:
            # the branch order holds below min_K D_K only: count at the highest
            # root found below it, which checks the branches up to that root
            below = found[found < d_min]
            if below.size == 0:
                break
            cut, deficit = float(below[-1]), 0
            upto = None
        # λ_j(ω_c) - ω_c ≈ (ω_j - ω_c) / T1_j: a branch whose root lies within
        # tol T1_j of ω_c falls in the window; count the found roots the same way
        nf_lo = int(np.count_nonzero(found < cut - tol * t1f))
        nf = int(np.count_nonzero(found <= cut + tol * t1f))
        Yl = np.column_stack([y_out[r] for r in L])
        # the first cut is the highest: a branch below a later cut lies below it,
        # and one the search finds there is solved and joins every later start
        nb_lo, nb, lam, X, dmv = count_below(cut, max(nroots, nf + 1), Yl,
                                             check=rnd == 0)
        if nf > nb:
            warnings.warn(f'{label}: {nf} roots found at or below omega = {cut:.8f} '
                          f'Ha where A_eff has {nb}: a copy inside the duplicate '
                          'window, or an unconverged root', RuntimeWarning,
                          stacklevel=2)
        # a branch inside the cut's window is degenerate with the cut root to the
        # window's resolution: missing only when roots are short of nroots
        if nb_lo <= nf_lo and not deficit:
            break
        # the branch index of each found root: the branches below its window, a
        # group of roots within tol taking the next ones
        covered, a = set(), 0
        while a < found.size and found[a] <= cut + tol * t1f[a]:
            b = a + 1
            while b < found.size and found[b] - found[a] <= tol:
                b += 1
            # the cut's eigenvectors in the start carry what its search found
            c, c_hi, _, Xf, _ = count_below(float(found[a]), b + 1,
                                            np.column_stack([Yl[:, :b], X]))
            if c_hi - c > b - a:
                # the window holds a branch no root of the group found: the group
                # takes the window branches its vectors project onto most,
                # p_j = sum_r (sum_p Xf_pj Yl_pr)², r over the group
                p = np.sum((Xf[:, c:c_hi].T @ Yl[:, a:b]) ** 2, axis=1)
                covered.update(int(c + j) for j in np.argsort(p)[::-1][:b - a])
            else:
                covered.update(range(c, c + b - a))
            a = b
        missing = [j for j in range(nb if upto is None else upto)
                   if j not in covered]
        if rnd == _COUNT_ROUNDS or not missing:
            warnings.warn(f'{label}: {len(missing) or nb_lo - nf_lo + deficit} of the '
                          f'{nroots} lowest roots not found after {rnd} round(s)',
                          RuntimeWarning, stacklevel=2)
            break
        # consecutive missing branches degenerate at the cut form one group, and
        # a group takes its degenerate partners past nroots too: a level whole
        groups = []
        for j in missing:
            if not groups or j >= groups[-1][1]:
                groups.append([j, j + 1])
            while groups[-1][1] < lam.size and groups[-1][1] not in covered and \
                    lam[groups[-1][1]] - lam[groups[-1][0]] <= _DEGENERATE:
                groups[-1][1] += 1
        before = len(L)
        for a, b in groups:
            solve_index(a, b, lam, X, dmv, cut)
        mark_copies()
        if len(live()) == before:
            warnings.warn(f'{label}: the branches {missing} solved by index landed on '
                          'roots already found; stopped, their roots missing from the '
                          'result', RuntimeWarning, stacklevel=2)
            break
    # the nroots lowest roots, copies left out, a level cut by nroots whole;
    # roots solved past the cut are dropped
    L = live()
    top = omega[L[min(nroots, len(L)) - 1]]
    if top >= d_min:
        short = (f'; {nroots - len(L)} root(s) short of nroots' if len(L) < nroots
                 else '')
        warnings.warn(f'{label}: omega = {top:.8f} Ha lies at or above the lowest '
                      f'doubles energy {d_min:.8f} Ha; the roots from there up are '
                      f'not checked by the count{short}', RuntimeWarning, stacklevel=2)
    sel = np.asarray([r for r in L if omega[r] <= top], int)
    return FoldResult(np.asarray(omega)[sel], np.column_stack(y_out)[:, sel],
                      np.asarray(t1_out)[sel], np.asarray(steps, int)[sel],
                      [loop[i] for i in sel], np.asarray(converged, bool)[sel],
                      embed, np.asarray(level, int)[sel])
