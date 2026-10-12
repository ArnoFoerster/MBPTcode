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
import numpy as np

from src.Base.eri_blocks import MOEriBlocks
from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_r_sigma_df
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB, _ew, _ksum, _stream
from src.SingleReference.GW.quasi_boson import QPqb

# (nk, nv, nm) slab blocks one slab of the eq 66 kernel holds at once (traced peak,
# check 8 of tests/test_ee_gw_fold.py), what _sb._slab_plan budgets with
_K_LIVE = 3


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


def build_pieces_gw(eps, B, nocc, screening='tda'):
    """The pieces of ee_fold.folded_operator for eq 66.

    Parameters
    ----------
    eps : ndarray, shape (norb,), HF orbital energies, Hartree
    B : ndarray, shape (naux, norb, norb), index order (Q, p, q), DF factors with
        (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s]
    nocc : int, doubly occupied orbitals
    screening : {'tda', 'rpa'}

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
        formed; 'level' 'gw'; 'no', 'nv', 'be'; 'dense_limit' 40, solve_folded's
        default; 'omega' (nm,) and 'W' (norb, norb, nm) for inspection.
    """
    if screening not in ('tda', 'rpa'):
        raise ValueError(f"screening={screening!r}; expected 'tda' or 'rpa'")
    eps = np.asarray(eps, float)
    no = nocc
    nv = len(eps) - no
    # A^HF alone: the gf2 builder's other pieces hold its doubles blocks
    M_hf = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)[3]['M']
    omega, W = gw_modes(eps, B, no, screening)
    Abar = static_term(eps, W, omega, no)
    M = M_hf + SB({'aaaa': Abar, 'bbbb': Abar})
    Woo, Wvv = W[:no, :no, :], W[no:, no:, :]              # W^m_ik, W^m_ac
    eo, ev = eps[:no], eps[no:]
    # D_kcm = Ω_m + ε_c - ε_k
    D = omega[None, None, :] + ev[None, :, None] - eo[:, None, None]

    nm = omega.size
    # matmuls on views of W: the slices keep (c, m) and (k, m) contiguous, so no
    # call copies W, the largest array

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
            Wook = Woo[:, k0:k1]                                  # W^m_ik, (i, k, m)
            den = _ew(np.subtract, D[k0:k1], omega)               # D_kcm - ω

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
