"""Eq 66 of Monino and Loos 2023 as the `pieces` of the EE fold (level 'gw').

Eq 66 is the symmetric one-doubles-set BSE@GW supermatrix at G0W0@HF: singles block
A^HF + Ā^GW, doubles (k, c, m) a particle-hole pair times a screening mode m,

    V_kcm,ia = -δ_ac W^m_ik + δ_ik W^m_ac                        (eqs 64a, 64b)
    D_kcm    = Ω_m + ε_c - ε_k                                     (eq 67)
    Ā_ia,jb  = δ_ij sum_km ρ^m_ak ρ^m_bk [1/(ε_a - ε_k + Ω_m) + 1/(ε_b - ε_k + Ω_m)]
             - δ_ab sum_cm ρ^m_ic ρ^m_jc [1/(ε_i - ε_c - Ω_m) + 1/(ε_j - ε_c - Ω_m)]
                                                                   (eq 70)

with ρ^m_pq = sum_jb (pq|jb) (X+Y)_jb,m, W^m_pq = sqrt(2) ρ^m_pq (the paper's M^ph)
and HF orbital energies throughout. Each spin block carries the same W and no
cross-spin coupling, so ee_fold's singlet/triplet projection of the singles gives
the two spin-adapted problems. The doubles are never stored: V and Vt act on one
vector at a time inside the fold's matvec.

With TDA screening and Ā left out this is Bintrim and Berkelbach's symmetric H~
(their eq 12, BSE.bse_upfolded.build_hamiltonian_familiar) with its inner pair
rotated onto the screening modes.

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eqs 64, 66, 67, 70.
S. J. Bintrim and T. C. Berkelbach, J. Chem. Phys. 156, 044114 (2022), eq 12.
"""
import math

import numpy as np
import scipy.linalg

from src.Base.eri_blocks import MOEriBlocks
from src.Base.utils.grids import minimax_time_grid
from src.Base.utils.time_frequency import minimax_points_for_accuracy
from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_r_sigma_df
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB, _ew, _ksum, _stream
from src.SingleReference.GW.quasi_boson import QPqb

# (nk, nv, nm) slab blocks one slab of the eq 66 kernel holds at once (traced peak,
# check 8 of tests/test_ee_gw_fold.py), what _sb._slab_plan budgets with
_K_LIVE = 3
# kernel_laplace's grid target: max |1 - x sum_t w_t exp(-τ_t x)| on its range scaled
# to [1, R] (time_frequency.laplace_test_error, relative=True), the relative error of
# every term; the absolute residual lets terms far above min D drift as R grows
_LAPLACE_TARGET = 1e-9


def _eri_blocks(B, nocc):
    """MOEriBlocks (ovov, oovv, pqov) from DF factors B, shape (naux, norb, norb),
    index order (Q, p, q), with (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s]."""
    o, v = slice(0, nocc), slice(nocc, B.shape[1])
    Bov = B[:, o, v]
    # (ia|jb) = sum_Q B_Q,ia B_Q,jb
    ovov = np.einsum('Qia,Qjb->iajb', Bov, Bov, optimize=True)
    # (ij|ab) = sum_Q B_Q,ij B_Q,ab
    oovv = np.einsum('Qij,Qab->ijab', B[:, o, o], B[:, v, v], optimize=True)
    # (pq|ia) = sum_Q B_Q,pq B_Q,ia
    pqov = np.einsum('Qpq,Qia->pqia', B, Bov, optimize=True)
    return MOEriBlocks(ovov, oovv, pqov, nocc)


def gw_modes(eps, B, nocc, screening='tda'):
    """Screening modes of G0W0@HF from the singlet direct RPA on eps.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (Q, p, q)
    nocc : int
    screening : {'tda', 'rpa'}; QPqb raises ValueError on anything else

    Returns
    -------
    omega : ndarray, shape (nm,), Ω_m ascending, Hartree
    W : ndarray, shape (norb, norb, nm), index order (p, q, m), C-contiguous, W^m_pq =
        sqrt(2) sum_jb (pq|jb) (X+Y)_jb,m. QPqb's Bogoliubov form has
        (X+Y)_Jm = sum_K (e^t)_JK U_Km and sum_J (X-Y)_Jm (X+Y)_Jn = δ_mn,
        QuAcK's normalisation of XpY; t = 0 for 'tda'.
    """
    qp = QPqb(np.asarray(eps, float), _eri_blocks(B, nocc), nocc,
              screening=screening)
    # QPqb's einsum leaves m the slowest axis; every matvec reshapes slices of W
    return np.asarray(qp.omega, float), np.ascontiguousarray(qp.Wnu, dtype=float)


def static_term(eps, W, omega, nocc):
    """Ā^GW of eq 70 on one spin block.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies
    W : ndarray, shape (norb, norb, nm), index order (p, q, m), sqrt(2) ρ^m_pq
    omega : ndarray, shape (nm,)
    nocc : int

    Returns
    -------
    ndarray, shape (no, nv, no, nv), index order (i, a, j, b):

        Ā_ia,jb = δ_ij (P_ab + P_ba) - δ_ab (Q_ij + Q_ji),
        P_ab = sum_km ρ^m_ak ρ^m_bk / (ε_a - ε_k + Ω_m)   (positive denominators)
        Q_ij = sum_cm ρ^m_ic ρ^m_jc / (ε_i - ε_c - Ω_m)   (negative denominators)
    """
    no = nocc
    nv = len(eps) - no
    eo, ev = eps[:no], eps[no:]
    rho_vo = W[no:, :no, :] / np.sqrt(2.0)                 # ρ^m_ak
    rho_ov = W[:no, no:, :] / np.sqrt(2.0)                 # ρ^m_ic
    # 1 / (ε_a - ε_k + Ω_m), (a, k, m)
    dp = 1.0 / (ev[:, None, None] - eo[None, :, None] + omega[None, None, :])
    # 1 / (ε_i - ε_c - Ω_m), (i, c, m)
    dh = 1.0 / (eo[:, None, None] - ev[None, :, None] - omega[None, None, :])
    # P_ab = sum_km ρ^m_ak ρ^m_bk / (ε_a - ε_k + Ω_m)
    P = np.einsum('akm,bkm->ab', rho_vo * dp, rho_vo, optimize=True)
    # Q_ij = sum_cm ρ^m_ic ρ^m_jc / (ε_i - ε_c - Ω_m)
    Q = np.einsum('icm,jcm->ij', rho_ov * dh, rho_ov, optimize=True)
    # δ_ij (P_ab + P_ba) - δ_ab (Q_ij + Q_ji)
    return (np.einsum('ij,ab->iajb', np.eye(no), P + P.T, optimize=True)
            - np.einsum('ab,ij->iajb', np.eye(nv), Q + Q.T, optimize=True))


def gw_modes_df(eps, B, nocc, screening='tda'):
    """gw_modes' screening modes in the DF factors, without W or (pq|ia).

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (P, p, q)
    nocc : int
    screening : {'tda', 'rpa'}

    Returns
    -------
    omega : ndarray, shape (nm,), Ω_m ascending, Hartree
    rho : ndarray, shape (naux, nm), index order (P, m): ρ_P,m = sum_jb B_P,jb
        (X+Y)_jb,m with QPqb's normalisation sum_J (X-Y)_Jm (X+Y)_Jn = δ_mn, so
        W^m_pq = sqrt(2) sum_P B_P,pq ρ_P,m is gw_modes' W up to the sign and the
        rotation of degenerate modes, which no eq 66 quantity sees.

    Notes
    -----
    build_rpa_AB's singlet blocks are A_ia,jb = δ_ij δ_ab Δ_ia + 2 (ia|jb) and
    B_ia,jb = 2 (ia|jb), or 0 for TDA screening, with Δ_ia = ε_a - ε_i. A - B = Δ
    is diagonal, so one symmetric eigh gives the modes at a few (nm, nm) arrays:

        TDA:  A U = U Ω,                                (X+Y) = U,
        RPA:  Δ^(1/2) (Δ + 4 (ia|jb)) Δ^(1/2) Z = Z Ω², (X+Y) = Δ^(1/2) Z Ω^(-1/2).
    """
    if screening not in ('tda', 'rpa'):
        raise ValueError(f"screening={screening!r}; expected 'tda' or 'rpa'")
    eps = np.asarray(eps, float)
    no = nocc
    Bov = B[:, :no, no:].reshape(B.shape[0], -1)          # B_P,jb, (P, jb)
    d = (eps[no:][None, :] - eps[:no][:, None]).ravel()   # Δ_jb = ε_b - ε_j
    # (ia|jb) = sum_P B_P,ia B_P,jb, times 2 for A (TDA), 4 for A + B (RPA)
    C = (2.0 if screening == 'tda' else 4.0) * (Bov.T @ Bov)
    C[np.diag_indices_from(C)] += d
    if screening == 'tda':
        # A U = U Ω; ρ_P,m = sum_I B_P,I U_I,m
        # overwrite_a: LAPACK works in C itself rather than in a copy of it
        omega, U = scipy.linalg.eigh(C, overwrite_a=True, check_finite=False)
        del C
        return omega, Bov @ U
    sd = np.sqrt(d)
    # Δ^(1/2)_I (A + B)_IJ Δ^(1/2)_J Z_J,m = Z_I,m Ω_m²
    C *= sd[:, None]
    C *= sd[None, :]
    w2, Z = scipy.linalg.eigh(C, overwrite_a=True, check_finite=False)
    del C
    if w2[0] <= 0:
        raise ValueError(f'RPA instability: lowest Omega^2 = {w2[0]:.3e}')
    omega = np.sqrt(w2)
    # ρ_P,m = sum_I B_P,I Δ^(1/2)_I Z_I,m Ω_m^(-1/2)
    return omega, ((Bov * sd) @ Z) / np.sqrt(omega)


def static_term_df(eps, B, rho, omega, nocc):
    """static_term's Ā^GW of eq 70 from gw_modes_df's ρ_P,m.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies
    B : ndarray, shape (naux, norb, norb), index order (P, p, q), B_P,pq = B_P,qp
    rho : ndarray, shape (naux, nm), index order (P, m)
    omega : ndarray, shape (nm,)
    nocc : int

    Returns
    -------
    ndarray, shape (no, nv, no, nv), index order (i, a, j, b): static_term's Ā,

        P_ab = sum_km ρ^m_ka ρ^m_kb / D_kam,   Q_ij = - sum_cm ρ^m_ic ρ^m_jc / D_icm,

    with ρ^m_kc = sum_P B_P,kc ρ_P,m (= ρ^m_ck) and D_kcm = Ω_m + ε_c - ε_k, formed
    in slabs of m that _sb._slab_plan sizes, two (m, k, c) slab blocks each.
    """
    no = nocc
    nv = len(eps) - no
    eo, ev = eps[:no], eps[no:]
    Bov = B[:, :no, no:].reshape(B.shape[0], no * nv)     # B_P,kc, (P, kc)

    def slab(m0, m1):
        # ρ^m_kc = sum_P ρ_P,m B_P,kc, (m, k, c)
        R = (rho[:, m0:m1].T @ Bov).reshape(m1 - m0, no, nv)
        # X_mkc = ρ^m_kc / D_kcm
        X = omega[m0:m1, None, None] + ev[None, None, :] - eo[None, :, None]
        np.divide(R, X, out=X)
        # P_ab = sum_mk X_mka ρ^m_kb
        P = X.reshape(-1, nv).T @ R.reshape(-1, nv)
        # Q_ij = - sum_mc X_mic ρ^m_jc
        Q = -np.matmul(X, R.transpose(0, 2, 1)).sum(axis=0)
        return [P, Q], 0.0

    width, workers = _sb._slab_plan(omega.size, no * nv, 2)
    (P, Q), _ = _stream(slab, omega.size, width, workers)
    # δ_ij (P_ab + P_ba) - δ_ab (Q_ij + Q_ji)
    return (np.einsum('ij,ab->iajb', np.eye(no), P + P.T, optimize=True)
            - np.einsum('ab,ij->iajb', np.eye(nv), Q + Q.T, optimize=True))


def _df_couple(Boo, Bvv, y):
    """Z_Q,kc = sum_b B_Q,bc y_kb - sum_j B_Q,jk y_jc, shape (naux, no, nv), from
    Boo (naux, no, no) and Bvv (naux, nv, nv), index order (Q, p, q), and y (no, nv):
    V y in the DF factors, (V y)_kcm = sqrt(2) sum_Q ρ_Q,m Z_Q,kc."""
    return np.matmul(y, Bvv) - np.matmul(Boo.transpose(0, 2, 1), y)


def _df_couple_t(Boo, Bvv, T):
    """w_ia = sum_Pc B_P,ac T_P,ic - sum_Pk B_P,ik T_P,ka, shape (no, nv), from T
    (naux, no, nv), index order (P, k, c): Vt in the DF factors, with T_P,kc =
    sqrt(2) sum_m ρ_P,m Y_kcm."""
    return (np.matmul(T, Bvv.transpose(0, 2, 1)).sum(axis=0)
            - np.matmul(Boo, T).sum(axis=0))


def kernel_df(eps, B, rho, omega, nocc):
    """build_pieces_gw's fused kernel 'K' through the DF factors and the modes ρ_P,m,
    exact at every ω and streamed over slabs of k: no (norb, norb, nm) array forms,
    and a (k, c, m) array for the slab's k only.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (P, p, q)
    rho : ndarray, shape (naux, nm), index order (P, m), gw_modes_df's ρ_P,m
    omega : ndarray, shape (nm,), Ω_m
    nocc : int

    Returns
    -------
    callable K(y1, om, powers=(1,)) -> (ws, dn2), kernel_laplace's contract at every
    om. Per spin and slab k0 ≤ k < k1, with Z_Q,kc = sum_b B_Q,bc y_kb - sum_j
    B_Q,jk y_jc,

        Y_kcm  = sqrt(2) sum_Q Z_Q,kc ρ_Q,m / (D_kcm - ω),
        T_P,kc = sqrt(2) sum_m ρ_P,m Y_kcm / (D_kcm - ω)^(p-1),
        w_ia  += sum_Pc B_P,ac T_P,ic (i in the slab) - sum_Pk B_P,ik T_P,ka,

    and dn2 = sum_kcm Y_kcm², at 2 naux no nv nm per spin and call, naux / norb
    times the W pieces' kernel.
    """
    no = nocc
    nv = len(eps) - no
    naux, nm = B.shape[0], omega.size
    eo, ev = eps[:no], eps[no:]
    Boo, Bvv = B[:, :no, :no], B[:, no:, no:]

    def K(y1, om, powers=(1,)):
        if y1.parity is not None:
            raise NotImplementedError('kernel_df takes y1 without a flip parity')
        Z = {s: _df_couple(Boo, Bvv, y1.get(s)) for s in ('aa', 'bb')}
        width, workers = _sb._slab_plan(no, nv * nm, _K_LIVE)

        def slab(k0, k1):
            nk = k1 - k0
            # D_kcm - ω on the slab's k, D summed in build_pieces_gw's order
            den = omega[None, None, :] + ev[None, :, None] - eo[k0:k1, None, None]
            _ew(np.subtract, den, om, out=den)
            ws, dn2 = {p: {} for p in powers}, 0.0
            for s in ('aa', 'bb'):
                # Y_kcm / sqrt(2) = sum_Q Z_Q,kc ρ_Q,m / (D_kcm - ω)
                Zs = Z[s][:, k0:k1].reshape(naux, nk * nv)
                Y = (Zs.T @ rho).reshape(nk, nv, nm)
                _ew(np.true_divide, Y, den, out=Y)
                dn2 += 2.0 * float(np.vdot(Y, Y))
                # each power in ascending order, Y divided in place on the way up
                p0 = 1
                for p in sorted(ws):
                    for _ in range(p - p0):
                        _ew(np.true_divide, Y, den, out=Y)
                    p0 = max(p0, p)
                    # T_P,kc = sqrt(2) sum_m ρ_P,m Y_kcm = 2 sum_m ρ_P,m (Y_kcm /
                    # sqrt(2)), for the slab's k
                    T = (rho @ Y.reshape(nk * nv, nm).T).reshape(naux, nk, nv)
                    T *= 2.0
                    # w_ia = sum_Pc B_P,ac T_P,ic on the rows i = k of the slab,
                    # minus sum_Pk B_P,ik T_P,ka over its k
                    w = -np.matmul(Boo[:, :, k0:k1], T).sum(axis=0)
                    w[k0:k1] += np.matmul(T, Bvv.transpose(0, 2, 1)).sum(axis=0)
                    ws[p][s] = w
                del Y                       # before the next spin's Y is formed
            return [SB(ws[p]) for p in powers], dn2

        return _stream(slab, no, width, workers)

    return K


def kernel_laplace(eps, B, rho, omega, nocc):
    """build_pieces_gw's fused kernel 'K' by a Laplace quadrature of 1/(D - ω), from
    the DF factors and the modes ρ_P,m: no (k, c, m) or (norb, norb, nm) array forms.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (P, p, q)
    rho : ndarray, shape (naux, nm), index order (P, m), gw_modes_df's ρ_P,m
    omega : ndarray, shape (nm,), Ω_m
    nocc : int

    Returns
    -------
    callable K(y1, om, powers=(1,)) -> (ws, dn2). y1 is an SB singles {'aa', 'bb'}
    of (no, nv), index order (i, a), without a flip parity; om the fold's ω; ws a
    list of SB singles of (no, nv), one per p in powers,

        ws[p]_ia = sum_kcm V_kcm,ia (V y1)_kcm / (D_kcm - ω)^p,

    and dn2 = sum_kcm (V y1)_kcm² / (D_kcm - ω)², both spin blocks summed. At
    ω ≥ min D, where 1/(D - ω) has no Laplace transform, and where no tabulated
    grid reaches _LAPLACE_TARGET (ω close below min D), K is kernel_df's, exact.

    Notes
    -----
    On x = D_kcm - ω in [min D - ω, max D - ω], the minimax grid (τ_t, w_t) of
    grids.minimax_time_grid, sized by minimax_points_for_accuracy for relative
    error _LAPLACE_TARGET, gives

        1 / x^p = sum_t c_p,t exp(-τ_t x),    c_p,t = w_t τ_t^(p-1) / (p-1)!,

    p = 2 being the ω-derivative of p = 1. D_kcm = Ω_m + ε_c - ε_k factorises the
    exponential, and with W^m_pq = sqrt(2) sum_P B_P,pq ρ_P,m the mode sum is the
    screened interaction at imaginary time τ, (naux, naux) per τ,

        Π^τ_PQ = sum_m exp(-τ Ω_m) ρ_P,m ρ_Q,m.

    V_kcm,ia = δ_ik W^m_ac - δ_ac W^m_ik (eqs 64a, 64b) splits ws into four
    terms, each a chain of DF contractions over one orbital index at a time; each
    carries 2 sum_t c_p,t and the pair factor g^τ_kc = exp(τ (ω + ε_k - ε_c)):

        vv-vv  w_ia: + g^τ_ic sum_Pc B_P,ac sum_Q Π^τ_PQ sum_b B_Q,bc y_ib
        vv-oo  w_ia: - g^τ_ic sum_Pc B_P,ac sum_Q Π^τ_PQ sum_j B_Q,ji y_jc
        oo-vv  w_ia: - g^τ_ka sum_Pk B_P,ik sum_Q Π^τ_PQ sum_b B_Q,ba y_kb
        oo-oo  w_ia: + g^τ_ka sum_Pk B_P,ik sum_Q Π^τ_PQ sum_j B_Q,jk y_ja

    The right-hand sums of each pair are one Z_Q,kc = sum_b B_Q,bc y_kb - sum_j
    B_Q,jk y_jc, so per spin

        T_P,kc = 2 sum_t c_p,t g^τ_kc sum_Q Π^τ_PQ Z_Q,kc,
        w_ia   = sum_Pc B_P,ac T_P,ic - sum_Pk B_P,ik T_P,ka,
        dn2    = 2 sum_t c_2,t sum_kc g^τ_kc sum_PQ Z_P,kc Π^τ_PQ Z_Q,kc,

    at naux² nm for Π^τ and naux² no nv per spin for its product, per τ. Ω_min
    moves from Π^τ into g^τ_kc, so both factors of exp(-τ (D_kcm - ω)) stay at or
    below one.
    """
    no = nocc
    naux = B.shape[0]
    eo, ev = eps[:no], eps[no:]
    Boo, Bvv = B[:, :no, :no], B[:, no:, no:]
    om0 = float(omega.min())
    # min D and max D of D_kcm = Ω_m + ε_c - ε_k, summed in D's order
    d_min = float(om0 + ev.min() - eo.max())
    d_max = float(omega.max() + ev.max() - eo.min())
    # ε_k - ε_c - Ω_min, (k, c): g^τ_kc's exponent over τ, less ω
    e_kc = eo[:, None] - ev[None, :] - om0
    exact = kernel_df(eps, B, rho, omega, nocc)

    def K(y1, om, powers=(1,)):
        if y1.parity is not None:
            raise NotImplementedError('kernel_laplace takes y1 without a flip parity')
        if om >= d_min:
            return exact(y1, om, powers)
        lo, hi = d_min - om, d_max - om
        ntau, err = minimax_points_for_accuracy(lo, hi, _LAPLACE_TARGET,
                                                relative=True)
        if err > _LAPLACE_TARGET:
            return exact(y1, om, powers)
        tau, wt = minimax_time_grid(ntau, lo, hi)
        # Z_Q,kc of both spins, (Q, s, k, c)
        Z = np.stack([_df_couple(Boo, Bvv, y1.get(s)) for s in ('aa', 'bb')], axis=1)
        T = {p: np.zeros_like(Z) for p in powers}
        dn2 = 0.0
        for t, w in zip(tau, wt):
            # Π^τ_PQ = sum_m exp(-τ (Ω_m - Ω_min)) ρ_P,m ρ_Q,m
            r = rho * np.exp(-0.5 * t * (omega - om0))
            Pi = r @ r.T
            del r
            # H_P,skc = g^τ_kc sum_Q Π^τ_PQ Z_Q,skc
            H = (Pi @ Z.reshape(naux, -1)).reshape(Z.shape)
            H *= np.exp(t * (om + e_kc))
            for p in powers:
                # T_P,skc += 2 c_p,t H_P,skc
                T[p] += (2.0 * w * t ** (p - 1) / math.factorial(p - 1)) * H
            # dn2 += 2 c_2,t sum_Pskc Z_P,skc H_P,skc
            dn2 += 2.0 * w * t * float(np.vdot(Z, H))
        # w_ia = sum_Pc B_P,ac T_P,ic - sum_Pk B_P,ik T_P,ka, per spin
        return ([SB({s: _df_couple_t(Boo, Bvv, T[p][:, n])
                     for n, s in enumerate(('aa', 'bb'))}) for p in powers], dn2)

    return K


def build_pieces_gw(eps, B, nocc, screening='tda', laplace=False, df=False):
    """The pieces of ee_fold.folded_operator for eq 66.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (Q, p, q), DF factors with
        (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s]
    nocc : int, doubly occupied orbitals
    screening : {'tda', 'rpa'}
    laplace : bool, True kernel_laplace as 'K', the Laplace quadrature of the
        doubles' denominator on df's pieces (implies df)
    df : bool, False the modes as W (gw_modes) and the kernel streamed over slabs
        of k through W; True the modes as ρ_P,m (gw_modes_df), V and Vt through B
        and ρ, and kernel_df as 'K', exact, so neither W nor (pq|ia) is formed

    Returns
    -------
    dict
        'M' SB, blocks (no, nv, no, nv), index order (i, a, j, b): the gf2 builder's
        A^HF plus eq 70 on 'aaaa' and 'bbbb'; 'V' SB singles {'aa', 'bb'} of
        (no, nv), index order (i, a) -> SB doubles {'aa', 'bb'} of (no, nv, nm),
        index order (k, c, m); 'Vt' the transpose; 'D' ndarray, shape (no, nv, nm),
        index order (k, c, m); 'dnorm2' SB doubles -> sum of the squares over both
        blocks; 'K' the fused kernel of ee_fold.folded_operator, K(y1, omega,
        powers) -> (ws, dn2), Vt (D - ω)^-p V y1 for each p and dnorm2 of
        (D - ω)^-1 V y1, streamed over slabs of k so no (no, nv, nm) array is
        formed, through W, through B and ρ (kernel_df) for df=True, or by
        kernel_laplace for laplace=True; 'level' 'gw'; 'no', 'nv', 'be';
        'dense_limit' 40, solve_folded's default; 'omega' (nm,) and 'W' (norb,
        norb, nm) for inspection, None for df=True.
    """
    if screening not in ('tda', 'rpa'):
        raise ValueError(f"screening={screening!r}; expected 'tda' or 'rpa'")
    eps = np.asarray(eps, float)
    no = nocc
    nv = len(eps) - no
    # A^HF alone: the gf2 builder's other pieces hold its doubles blocks
    M_hf = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)[3]['M']
    df = df or laplace
    if df:
        # Ω_m and ρ_P,m = sum_jb B_P,jb (X+Y)_jb,m; W^m_pq = sqrt(2) sum_P B_P,pq ρ_P,m
        omega, rho = gw_modes_df(eps, B, no, screening)
        W, Abar = None, static_term_df(eps, B, rho, omega, no)
    else:
        # Ω_m and W^m_pq = sqrt(2) sum_jb (pq|jb) (X+Y)_jb,m
        omega, W = gw_modes(eps, B, no, screening)
        Abar = static_term(eps, W, omega, no)
    M = M_hf + SB({'aaaa': Abar, 'bbbb': Abar})
    eo, ev = eps[:no], eps[no:]
    # D_kcm = Ω_m + ε_c - ε_k
    D = omega[None, None, :] + ev[None, :, None] - eo[:, None, None]

    nm = omega.size
    if df:
        naux = B.shape[0]
        Boo, Bvv = B[:, :no, :no], B[:, no:, no:]

        def couple(y):
            # Y_kcm = sqrt(2) sum_Q Z_Q,kc ρ_Q,m
            Z = _df_couple(Boo, Bvv, y).reshape(naux, no * nv)
            return np.sqrt(2.0) * (Z.T @ rho).reshape(no, nv, nm)

        def couple_t(Y):
            # T_P,kc = sqrt(2) sum_m ρ_P,m Y_kcm
            T = np.sqrt(2.0) * (rho @ Y.reshape(no * nv, nm).T)
            return _df_couple_t(Boo, Bvv, T.reshape(naux, no, nv))

        if laplace:
            # ws_ia = sum_kcm V_kcm,ia (V y1)_kcm sum_t c_p,t exp(-τ_t (D_kcm - ω))
            kernel = kernel_laplace(eps, B, rho, omega, no)
        else:
            # ws_ia = sum_kcm V_kcm,ia (V y1)_kcm / (D_kcm - ω)^p, slabs of k
            kernel = kernel_df(eps, B, rho, omega, no)
    else:
        Woo, Wvv = W[:no, :no, :], W[no:, no:, :]          # W^m_ik, W^m_ac
        # matmuls on views of W: the slices keep (c, m) and (k, m) contiguous, so
        # no call copies W, the largest array

        def couple(y):
            # Y_kcm = sum_a W^m_ac y_ka - sum_i W^m_ik y_ic
            Y = (y @ Wvv.reshape(nv, nv * nm)).reshape(no, nv, nm)
            # sum_i W^m_ik y_ic as the batch over k of sum_i y^T_ci W^m_ik
            return _ew(np.subtract, Y, np.matmul(y.T, Woo.transpose(1, 0, 2)), out=Y)

        def couple_t(Y):
            # w_ia = sum_cm W^m_ac Y_icm - sum_km W^m_ik Y_kam, each summed over slabs
            # of its orbital index by _ksum, since the (i, a) output is small

            def vv(c0, c1):
                # sum_cm W^m_ac Y_icm over c0 <= c < c1, on views
                return Y[:, c0:c1].reshape(no, -1) @ Wvv[:, c0:c1].reshape(nv, -1).T

            def oo(k0, k1):
                # sum over k0 <= k < k1 of the batch sum_m W^m_ik Y^T_kma
                return np.matmul(Woo[:, k0:k1].transpose(1, 0, 2),
                                 Y[k0:k1].transpose(0, 2, 1)).sum(axis=0)

            return _ksum(vv, nv, Y.size) - _ksum(oo, no, Y.size)

        def kernel(y1, omega, powers=(1,)):
            # sum_K V_K,ia (V y1)_K / (D_K - ω)^p for each p in powers, and
            # sum_K (V y1)_K² / (D_K - ω)², streamed over slabs of k, the pair's
            # occupied index: Y_kcm exists for the slab's k alone, (k1 - k0, nv, nm)
            if y1.parity is not None:
                raise NotImplementedError('kernel takes y1 without a flip parity')
            width, workers = _sb._slab_plan(no, nv * nm, _K_LIVE)

            def slab(k0, k1):
                nk = k1 - k0
                Wook = Woo[:, k0:k1]                              # W^m_ik, (i, k, m)
                den = _ew(np.subtract, D[k0:k1], omega)           # D_kcm - ω

                def vt(Y):
                    # the slab's share of couple_t: w_ia = sum_cm W^m_ac Y_icm on the
                    # slab's rows i = k, minus sum_km W^m_ik Y_kam over the slab's k

                    def vv(c0, c1):
                        return (Y[:, c0:c1].reshape(nk, -1)
                                @ Wvv[:, c0:c1].reshape(nv, -1).T)

                    def oo(a0, a1):
                        return np.matmul(Wook[:, a0:a1].transpose(1, 0, 2),
                                         Y[a0:a1].transpose(0, 2, 1)).sum(axis=0)

                    w = np.zeros((no, nv))
                    w[k0:k1] = _ksum(vv, nv, Y.size)
                    return w - _ksum(oo, nk, Y.size)

                ws, dn2 = {p: {} for p in powers}, 0.0
                for s in ('aa', 'bb'):
                    y = y1.get(s)
                    # Y_kcm = sum_a W^m_ac y_ka - sum_i W^m_ik y_ic, k in the slab, the
                    # second term one k at a time so no second slab-sized array forms
                    Y = (y[k0:k1] @ Wvv.reshape(nv, nv * nm)).reshape(nk, nv, nm)
                    for kk in range(nk):
                        _ew(np.subtract, Y[kk], y.T @ Wook[:, kk], out=Y[kk])
                    _ew(np.true_divide, Y, den, out=Y)
                    dn2 += float(np.vdot(Y, Y))
                    # each power in ascending order, Y divided in place on the way up
                    p0 = 1
                    for p in sorted(ws):
                        for _ in range(p - p0):
                            _ew(np.true_divide, Y, den, out=Y)
                        p0 = max(p0, p)
                        ws[p][s] = vt(Y)
                    del Y                       # before the next spin's Y is formed
                return [SB(ws[p]) for p in powers], dn2

            return _stream(slab, no, width, workers)

    def V(y1):
        return SB({s: couple(y1.get(s)) for s in ('aa', 'bb')})

    def Vt(Y):
        return SB({s: couple_t(Y.get(s)) for s in ('aa', 'bb')})

    def dnorm2(Y):
        return float(sum(np.vdot(Y.get(s), Y.get(s)) for s in ('aa', 'bb')))

    # a channel of up to 40 singles builds A_eff densely
    return {'M': M, 'V': V, 'Vt': Vt, 'D': D, 'dnorm2': dnorm2, 'K': kernel,
            'level': 'gw', 'no': no, 'nv': nv, 'be': _eq.SPIN_BLOCKED,
            'dense_limit': 40, 'omega': omega, 'W': W}
