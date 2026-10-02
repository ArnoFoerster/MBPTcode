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
from src.Solvers.davidson import diagonal_seeds, overlap_pick, solve_symmetric

FOLD_LEVELS = ('adc2', 'gf2', 'gw')
_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}
_DEGENERATE = 1e-8      # Hartree: seeds spread by at most this are one level
_SEED_RESIDUAL = 1e-9   # seeds of M: Ritz error |r|²/gap far below _DEGENERATE
_CHECK_SHIFT = 0.1      # Hartree: the count check's correction r / (d - e + shift)
_CHECK_RESIDUAL = 1e-4  # the count check decides μ against a window, no tighter
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
    dnorm2 = pieces.get('dnorm2')
    if dnorm2 is not None:
        # polarisation: <a, b> = (|a + b|² - |a - b|²) / 4
        return 0.25 * (float(dnorm2(Ya + Yb)) - float(dnorm2(Ya - Yb)))
    no, nv = pieces['no'], pieces['nv']
    # sum_K Ya_K Yb_K on the flat adc2 layout
    return float(doubles_sb_to_flat(Ya, no, nv) @ doubles_sb_to_flat(Yb, no, nv))


def _doubles_image(pieces, u, omega, embed):
    """Ỹ_K = sum_jb V_K,jb u_jb / (ω - D_K) of a channel vector u, SB doubles."""
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
    """λ(ω) = ω for one root or one level: step(ω) -> (λ, T1, state); Newton
    ω + (λ - ω) T1 while T1 >= t_min, else the DIIS fixed point. Returns
    (λ, T1, state, steps, loop, converged, |λ - ω|) of the last step."""
    mode, hist_o, hist_e, k = 'newton', [], [], 0
    while True:
        lam, t1, state = step(om)
        err = lam - om
        k += 1
        if verbose:
            print(f'{label} step {k} ({mode}): omega = {om:.8f} '
                  f'lambda = {lam:.8f} T1 = {t1:.4f}', flush=True)
        if abs(err) < tol_omega:
            return lam, t1, state, k, mode, True, abs(err)
        if mode == 'newton' and t1 < t_min:
            mode, hist_o, hist_e = 'fixed', [], []
        if mode == 'newton':
            if k >= max_newton:
                return lam, t1, state, k, mode, False, abs(err)
            om = om + err * t1                         # Newton on λ(ω) - ω
        else:
            hist_o.append(om)
            hist_e.append(err)
            hist_o, hist_e = hist_o[-6:], hist_e[-6:]
            if k >= max_newton + max_fixed:
                return lam, t1, state, k, mode, False, abs(err)
            om = _diis_step(hist_o, hist_e)


def _root_step(pieces, spin, y, dense, tol_residual, label):
    """step(ω) for one root, following the previous vector."""
    ref = [y]

    def step(om):
        lam, yk, t1, _ = _eig_at(pieces, om, spin, ref[0], dense, tol_residual,
                                 label)
        ref[0] = yk
        return lam, t1, yk

    return step


def _level_step(pieces, spin, Y, dense, tol_residual, label):
    """step(ω) for a level: the mean eigenvalue and the mean T1 of its partners."""
    ref = [Y]

    def step(om):
        lam, Z, ZR, t1 = _eig_level(pieces, om, spin, ref[0], dense, tol_residual,
                                    label)
        ref[0] = ZR
        return float(lam.mean()), float(t1.mean()), (lam, Z, ZR, t1)

    return step


def _seed_levels(m0, diag_s, n, nroots, dense, tol_residual, label):
    """Seeds of M, lowest first, e (ns,) and X (n, ns), and the levels among them
    that start below nroots; on the Davidson branch the seeds are solved to
    _SEED_RESIDUAL and extended until the last such level ends before the last seed,
    so a level cut by nroots comes back whole."""
    if dense:
        # sum_q M_pq v_qr = w_r v_pr, M built from n matvecs
        e, X = np.linalg.eigh(dense_effective(m0, n))
        return e, X, [(a, b) for a, b in _levels(e) if a < nroots]
    extra = 2
    while True:
        ns = min(nroots + extra, n)
        # sum_q M_pq x_qr = e_r x_pr, the ns lowest by Davidson
        e, X, _ = solve_symmetric(m0, diag_s, nroots=ns,
                                  tol_residual=min(tol_residual, _SEED_RESIDUAL),
                                  label=label + ' seeds')
        e, X = np.asarray(e, float), np.asarray(X, float)
        runs = [(a, b) for a, b in _levels(e) if a < nroots]
        if runs[-1][1] < ns or ns == n:
            return e, X, runs
        extra *= 2


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
    Q = np.empty((n, 0))
    for g in cand:
        # g_p - sum_j Q_pj (sum_q Q_qj g_q), twice; kept unless it lies in span Q
        r = g - Q @ (Q.T @ g)
        r -= Q @ (Q.T @ r)
        if np.linalg.norm(r) > 1e-8 * np.linalg.norm(g):
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
        if Q.shape[1] == width:
            break
    return Q


def _lowest_beside(matvec, diag_s, V, w0, top, tol_residual, label):
    """The lowest eigenpair (μ, z (n,)) of A_eff off the span of V (n, m), its
    columns orthonormal near-eigenvectors with eigenvalues ≥ w0: Davidson on
    A' = A_eff + σ V V^T, σ = top - w0 + 1 Ha, which lifts them 1 Ha above top,
    started from one random vector alone, so that no seed can converge it
    elsewhere, with the shifted diagonal correction and to residual
    max(tol_residual, _CHECK_RESIDUAL): the Ritz value bounds μ from above, so a
    value at or below top is a branch there at any residual. μ > top says A_eff
    has no eigenvalue at or below top off span V."""
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
    r -= V @ (V.T @ r)
    # sum_q A'_pq z_q = μ z_p, the lowest, by Davidson from r alone
    mu, z, _ = solve_symmetric(lifted, diag, nroots=1, x0=r, precond=precond,
                               tol_residual=max(tol_residual, _CHECK_RESIDUAL),
                               label=label + ' check')
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


def solve_folded(pieces, nroots, spin=None, tol_omega=1e-6, tol_residual=1e-6,
                 t_min=0.3, max_newton=12, max_fixed=30, dense_limit=None,
                 verbose=0):
    """The nroots lowest folded roots, each level at its own frequency.

    Seeds: the lowest eigenpairs of the bare singles block M on the channel (eigh
    for n ≤ dense_limit, else Davidson to residual 1e-9), grouped into levels whose
    eigenvalues spread by at most 1e-8 Ha. A level of one is solved per root: λ, y
    at ω_k by eigh or Davidson with overlap following, then the Newton step
    ω_{k+1} = ω_k + (λ - ω_k) T1 while T1 ≥ t_min, else the DIIS-accelerated fixed
    point ω_{k+1} = λ. A level of g > 1 is solved jointly: one ω, the g eigenpairs
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

    Completeness: below min_K D_K each eigenvalue λ_j(ω) of A_eff(ω), counted from
    the lowest, falls with ω (dλ/dω ≤ 0), so λ_j(ω) = ω has one root ω_j, and
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
    label = f'ee-ADC fold ({spin or "both"})'
    e0, X0, runs = _seed_levels(m0, diag_s, n, nroots, dense, tol_residual, label)
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
            step = _level_step(pieces, spin, X[:, a:b], dense, tol_residual, label)
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

    solve_seeds(e0, X0, runs)
    mark_copies()
    # the count check: below min_K D_K the folded roots at or below ω number the
    # eigenvalues λ_j(ω) ≤ ω of A_eff(ω), and the j-th lowest root is the root
    # of the j-th branch; a branch no root found covers is solved by its index.
    # A level spreads by up to _DEGENERATE and is solved at its mean, so the
    # window holds that spread too
    tol = max(2.0 * tol_omega, _DEGENERATE)
    d_min = float(np.min(pieces['D']))
    for rnd in range(_COUNT_ROUNDS + 1):
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
