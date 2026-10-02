"""Restricted (closed-shell, spin-free) matrix-free Faddeev-ADC(3).

The physics is that of adc_u_faddeev, here in the CSF basis of the restricted ADC(3) route: 2h1p classes I
(i=i; a), II (hole pair triplet-coupled) and III (singlet-coupled), 2p1h
classes I' (i; a=a), II', III'. The Faddeev-ADC(3) operator is

    [[F,             U^T N^{-1/2}          ],
     [N^{-1/2} U,    N^{-1/2} H N^{-1/2}   ]],     H = K + C + dH,

a plain symmetric eigenproblem: N^{-1/2} is applied to vectors by
Lanczos (solve.apply_inverse_sqrt), never formed. K + C is the restricted
ADC(3) sigma operator on a satellite vector; dH, the metric N = 1 + dN and
the coupling U come from the pair channels.

Pair channels (spatial, closed shell): the particle-hole pair in its TDHF
singlet and triplet channels, the particle-particle and hole-hole pairs in
the ppRPA singlet and triplet channels. Each channel gives (T, dH, dN) on
its pair space, from the phonons or from the Riccati amplitudes
(LinearResponse/riccati.py; the ladder amplitudes at scale from
solve_ladder_amplitudes_restricted below).

Spin-free representation. A doublet 2p1h state of the kept M_S = +1/2
sector is carried by one spatial array Q[i,a,b], the amplitude of
a+_{a alpha} a+_{b beta} a_{i beta}|HF>; its all-alpha amplitude is
Q[i,a,b] - Q[i,b,a]. The CSFs are

    I'(i,a)     = Q[i,a,a],
    II'(i,a<b)  = sqrt(3/2) (Q[i,a,b] - Q[i,b,a]),
    III'(i,a<b) = (Q[i,a,b] + Q[i,b,a]) / sqrt(2),

and the 2h1p mirror with R[i,j,a], the amplitude of
a+_{a beta} a_{j beta} a_{i alpha}|HF>. The particle-particle (hole-hole)
corrections are block diagonal in this basis (singlet channel on I'+III',
triplet on II'); the particle-hole ones recouple the spins and act on Q (R).
Both follow from the spin-orbital formulas of adc_u_faddeev evaluated on
these components, and are checked against it through the CSF isometry.
"""
import numpy as np
import scipy.linalg as la

from src.Base.constants import RICCATI_CONV_TOL, RICCATI_MAX_ITER_JACOBI
from src.Base.eri_blocks import g_slice
from src.SingleReference.ADC import adc_r_dense_df, adc_r_dense_full
from src.SingleReference.ADC import adc_r_sigma_df, adc_r_sigma_full
from src.SingleReference.ADC.adc_r_utils import _u_2h1p_unfold, _u_2p1h_unfold
from src.SingleReference.ADC.eeADC.ee_r_sigma_df import DFVvvvKernels
from src.SingleReference.ADC.solve import apply_inverse_sqrt, inverse_sqrt_psd
from src.SingleReference.CC.diis import DIIS
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.pp_rpa import (
    PPRPASolver, build_pprpa_matrices_restricted, pair_indices,
    singlet_pair_indices)
from src.SingleReference.LinearResponse.riccati import (
    channel_corrections_from_phonons, require_stable_channel, ring_channel)

EH_TRIPLET = ('rpa', 'first_order', 'off')
PAIR_ROUTES = ('phonon', 'riccati')
SPINS = ('singlet', 'triplet')
_S16, _S12, _S32 = np.sqrt(1.0 / 6.0), np.sqrt(0.5), np.sqrt(1.5)


def _df_or_dense(s):
    """(B_aa, eri) of a restricted solver; one of them is None."""
    return s.B_aa, (s.eri if s.B_aa is None else None)


def _eh_channel(eps, nocc, spin, B_aa, eri_chemist, route, treatment='rpa'):
    """{'T', 'dH', 'dN'} of one TDHF spin channel of the particle-hole pair;
    (A, B) are the Casida matrices with the bare kernel.

    treatment='rpa': the RPA phonons (route 'phonon') or the closed-form ring
        amplitudes (route 'riccati', riccati.ring_channel); refuses an
        unstable channel.
    treatment='first_order': the ADC(3) level -- the bare pair vertex, no
        phonon dressing (dH = dN = 0), first-order amplitudes in U.
    treatment='off': a non-interacting pair -- no vertex, no amplitudes
        (T = 0, dN = 0, dH = D - A cancels the TDA vertex K + C carries), the
        singlet-only construction of the PSD self-energies.

    route='phonon' also returns the pair's phonons 'omega', 'X', 'Y', the
    solutions of (A + dH) X = (1 + dN) X omega with Y = T X: the TDHF phonons
    for 'rpa', the eigenvectors of A for 'first_order', the bare pairs
    (omega = D, X = 1) for 'off'.
    """
    eps = np.asarray(eps)
    A, B = LinearResponseSolver(
        eps, coeff_df=B_aa, eri_chemist=eri_chemist).build_casida_matrices(
            nocc, lBSE=True, triplet=(spin == 'triplet'))
    n = A.shape[0]
    D = (eps[None, nocc:] - eps[:nocc, None]).ravel()          # eps_a - eps_i
    if treatment == 'rpa':
        require_stable_channel(A, B, f'TDHF {spin} channel')
        if route == 'phonon':
            omega, X, Y = CasidaSolver(A, B).solve()
            T, dH, dN = channel_corrections_from_phonons(A, omega, X, Y)
            return {'T': T, 'dH': dH, 'dN': dN, 'omega': omega, 'X': X, 'Y': Y}
        T, dH, dN = ring_channel(A, B)
        return {'T': T, 'dH': dH, 'dN': dN}
    if treatment == 'first_order':
        T = -B / (D[:, None] + D[None, :])       # B + A T + T A = 0 to first order
        ch = {'T': T, 'dH': np.zeros((n, n)), 'dN': np.zeros((n, n))}
        if route == 'phonon':
            ch['omega'], ch['X'] = la.eigh(A)
            ch['Y'] = T @ ch['X']
        return ch
    if treatment == 'off':
        ch = {'T': np.zeros((n, n)), 'dH': np.diag(D) - A,
              'dN': np.zeros((n, n))}
        if route == 'phonon':
            ch.update(omega=D.copy(), X=np.eye(n), Y=np.zeros((n, n)))
        return ch
    raise ValueError(f"treatment={treatment!r}; expected one of {EH_TRIPLET}")


def pair_channels(eps, nocc, B_aa=None, eri_chemist=None, route='phonon',
                  verbose=0, eh_triplet='rpa'):
    """The pair channels of the Faddeev-ADC(3).

    eh: {'singlet', 'triplet'} -> {'T', 'dH', 'dN'}, TDHF on the (i, a)
        pairs, i outer; dH is block II's (block I carries -dH). eh_triplet
        selects the triplet channel's treatment (see _eh_channel): 'rpa' (full
        TDHF), 'first_order' (ADC(3) level) or 'off' (singlet
        eh pairs only). The singlet channel is always RPA.

    route='phonon': pp and hh likewise per spin, from the ppRPA N+2 and N-2
        phonons (pair-space matrices; singlet pairs in singlet_pair_indices
        order, triplet pairs in pair_indices order).

    route='riccati': pp and hh in the ordered alpha-beta pairs, from the
        spin-free ladder amplitudes alone (DF-streamed, no pair-space
        matrix of the virtuals):
        pp = {'form': 'ab', 't', 'V', 'D'}  t, V = (kc|ld) as [k,l,c,d]
             and D the alpha-beta C_pp (o^2 x o^2),
        hh = {'form': 'ab', 'dH', 'dN'}     (o^2 x o^2) over (i alpha, j beta).
    """
    if route not in PAIR_ROUTES:
        raise ValueError(f"route={route!r}; expected one of {PAIR_ROUTES}")
    if eh_triplet not in EH_TRIPLET:
        raise ValueError(f"eh_triplet={eh_triplet!r}; expected one of {EH_TRIPLET}")
    out = {'eh': {'singlet': _eh_channel(eps, nocc, 'singlet', B_aa, eri_chemist,
                                         route),
                  'triplet': _eh_channel(eps, nocc, 'triplet', B_aa, eri_chemist,
                                         route, eh_triplet)},
           'pp': {}, 'hh': {}}
    if route == 'riccati':
        out['pp'], out['hh'] = ladder_channels(eps, nocc, B_aa, eri_chemist,
                                               verbose=verbose)
        return out
    for spin in SPINS:
        App, Bpp, Cpp = build_pprpa_matrices_restricted(eps, nocc, spin, B_aa,
                                                        eri_chemist)
        res = PPRPASolver(App, Bpp, Cpp).solve()
        T, dH, dN = channel_corrections_from_phonons(
            App, res.omega_add, res.X_add, res.Y_add)
        out['pp'][spin] = {'T': T, 'dH': dH, 'dN': dN}
        T, dH, dN = channel_corrections_from_phonons(
            -Cpp, res.omega_rem, res.Y_rem, res.X_rem)
        out['hh'][spin] = {'T': T, 'dH': dH, 'dN': dN}
    return out


class LadderAmplitudes:
    """Converged spin-free ladder (ppRPA Riccati) amplitudes.

    t[k,l,c,d] is the amplitude of the ordered alpha-beta pairs
    (k alpha, l beta) -> (c alpha, d beta); its symmetric and antisymmetric
    parts are the singlet and triplet ladder amplitudes. TA[k,l,c,d] =
    sum_ef t[k,l,e,f] A_{ef,cd}, the ladder term at convergence (A the
    alpha-beta ppRPA A with the orbital energies), which the hole-hole
    channel needs as T A T^T. V = <kl|cd> and g_oooo = <kl|mn>, the
    integrals of the equations."""
    def __init__(self, t, TA, V, g_oooo, info):
        self.t, self.TA, self.info = t, TA, info
        self.V, self.g_oooo = V, g_oooo


def solve_ladder_amplitudes_restricted(eps, nocc, B_aa=None, eri_chemist=None,
                                       conv_tol=RICCATI_CONV_TOL,
                                       max_iter=RICCATI_MAX_ITER_JACOBI,
                                       diis_space=8, verbose=0):
    """Spin-free ladder CCD, the Riccati form of the closed-shell ppRPA,
    B^T + C T + T A + T B T = 0 in the ordered alpha-beta pairs:

        R[k,l,c,d] = <kl|cd> + (eps_c + eps_d - eps_k - eps_l) t[k,l,c,d]
                   + sum_mn <kl|mn> t[m,n,c,d] + sum_ef t[k,l,e,f] <ef|cd>
                   + sum_efmn t[k,l,e,f] <mn|ef> t[m,n,c,d] = 0.

    Jacobi steps with the orbital-energy denominators and DIIS; the vvvv
    ladder goes through the DF factor (eeADC DFVvvvKernels), or the dense
    tensor. Converged when max|R| < conv_tol. Returns a LadderAmplitudes."""
    eps = np.asarray(eps)
    norb = len(eps)
    o, v = slice(0, nocc), slice(nocc, norb)
    eo, ev = eps[o], eps[v]
    V = g_slice(B_aa, eri_chemist, o, o, v, v)                 # <kl|cd>
    G_oooo = g_slice(B_aa, eri_chemist, o, o, o, o)           # <kl|mn>
    D = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])
    if B_aa is not None:
        vvvv = DFVvvvKernels(B_aa, nocc, norb)._ldir
    else:
        g_vvvv = g_slice(None, eri_chemist, v, v, v, v)        # <ef|cd>
        vvvv = lambda t: np.einsum('klef,efcd->klcd', t, g_vvvv, optimize=True)

    def terms(t):
        lad = vvvv(t)
        X = np.einsum('klef,mnef->klmn', t, V, optimize=True)
        R = (V + D * t + np.einsum('klmn,mncd->klcd', G_oooo, t, optimize=True)
             + lad + np.einsum('klmn,mncd->klcd', X, t, optimize=True))
        return R, lad

    t = -V / D
    diis = DIIS(diis_space, start_iter=2)
    for it in range(max_iter + 1):
        R, lad = terms(t)
        rmax = float(np.max(np.abs(R)))
        if verbose:
            print(f"  ladder CCD iter {it:3d}  max|R| = {rmax:.3e}", flush=True)
        if rmax < conv_tol:
            TA = lad + (ev[None, None, :, None] + ev[None, None, None, :]) * t
            return LadderAmplitudes(t, TA, V, G_oooo,
                                    {'niter': it, 'residual': rmax})
        if it == max_iter:
            break
        t_new = t - R / D
        t = diis.compute_new_vec(t_new.ravel(), (t_new - t).ravel()).reshape(t.shape)
    raise RuntimeError(f"ladder CCD not converged in {max_iter} iterations "
                       f"(max|R| = {rmax:.3e})")


def ladder_channels(eps, nocc, B_aa=None, eri_chemist=None, verbose=0):
    """(pp, hh) of the Riccati route from the spin-free ladder amplitudes.

    In the ordered alpha-beta pairs, with T[kl,cd] = t, B[cd,kl] = (ck|dl),
    D = C_pp and A the ppRPA A,

        pp: dH = B T + T^T B^T + T^T D T,     dN = -T^T T     (applied),
        hh: dH = -(B^T T^T + T B + T A T^T),  dN = -T T^T     (o^2 x o^2),

    the channel corrections of riccati.channel_corrections_from_amplitudes
    for (A, B, C) and (-C, -B^T, -A); T A is the converged ladder term."""
    lad = solve_ladder_amplitudes_restricted(eps, nocc, B_aa, eri_chemist,
                                             verbose=verbose)
    O = nocc
    nv = len(eps) - O
    t, V = lad.t, lad.V
    eo = np.asarray(eps)[:O]
    Dm = lad.g_oooo.reshape(O * O, O * O) - np.diag((eo[:, None] + eo[None, :]).ravel())
    T2 = t.reshape(O * O, nv * nv)
    V2 = V.reshape(O * O, nv * nv)
    TA2 = lad.TA.reshape(O * O, nv * nv)
    dH = -(V2 @ T2.T + T2 @ V2.T + TA2 @ T2.T)
    dH = 0.5 * (dH + dH.T)
    dN = -(T2 @ T2.T)
    pp = {'form': 'ab', 't': t, 'V': V, 'D': 0.5 * (Dm + Dm.T), 'info': lad.info}
    hh = {'form': 'ab', 'dH': dH, 'dN': dN}
    return pp, hh


def first_order_channels(eps, nocc, B_aa=None, eri_chemist=None):
    """First-order amplitudes and dH = dN = 0: the ADC(3) limit."""
    eps = np.asarray(eps)
    nv = len(eps) - nocc
    eo, ev = eps[:nocc], eps[nocc:]
    out = {'eh': {}, 'pp': {}, 'hh': {}}
    for spin in SPINS:
        out['eh'][spin] = _eh_channel(eps, nocc, spin, B_aa, eri_chemist,
                                      'riccati', 'first_order')
        App, Bpp, Cpp = build_pprpa_matrices_restricted(eps, nocc, spin, B_aa,
                                                        eri_chemist)
        pairs = singlet_pair_indices if spin == 'singlet' else pair_indices
        pv, po = pairs(nv), pairs(nocc)
        dpp = ((ev[pv[0]] + ev[pv[1]])[None, :]
               - (eo[po[0]] + eo[po[1]])[:, None])
        Tpp = -Bpp.T / dpp                # B^T + C T + T A = 0 to first order
        out['pp'][spin] = {'T': Tpp, 'dH': np.zeros_like(App),
                           'dN': np.zeros_like(App)}
        out['hh'][spin] = {'T': Tpp.T, 'dH': np.zeros_like(Cpp),
                           'dN': np.zeros_like(Cpp)}
    return out


# ---------------------------------------------------------------------------
# CSF <-> spin-free component maps
# ---------------------------------------------------------------------------

def csf_to_Q(z_Ip, z_IIp, z_IIIp, nv):
    """Q[i,a,b] of a 2p1h CSF vector (z_Ip (O,V), z_IIp/z_IIIp (O,nPv))."""
    au, bu = pair_indices(nv)
    O = z_Ip.shape[0]
    Q = (_S16 * _u_2p1h_unfold(z_IIp, au, bu, O, nv, True)
         + _S12 * _u_2p1h_unfold(z_IIIp, au, bu, O, nv, False))
    d = np.arange(nv)
    Q[:, d, d] = z_Ip
    return Q


def Q_to_csf(Q):
    """CSF components of a doublet Q; the left inverse of csf_to_Q."""
    nv = Q.shape[1]
    au, bu = pair_indices(nv)
    d = np.arange(nv)
    return (Q[:, d, d].copy(), _S32 * (Q[:, au, bu] - Q[:, bu, au]),
            _S12 * (Q[:, au, bu] + Q[:, bu, au]))


def Q_to_csf_adjoint(y_Ip, y_IIp, y_IIIp, nv):
    """Transpose of Q_to_csf: the Q-space vector whose contraction with a
    doublet Q equals the CSF inner product with Q_to_csf(Q)."""
    au, bu = pair_indices(nv)
    O = y_Ip.shape[0]
    Q = (_S32 * _u_2p1h_unfold(y_IIp, au, bu, O, nv, True)
         + _S12 * _u_2p1h_unfold(y_IIIp, au, bu, O, nv, False))
    d = np.arange(nv)
    Q[:, d, d] = y_Ip
    return Q


def csf_to_R(z_I, z_II, z_III, nocc):
    """R[i,j,a] of a 2h1p CSF vector (z_I (O,V), z_II/z_III (nPo,V))."""
    iu, ju = pair_indices(nocc)
    V = z_I.shape[1]
    R = (_S16 * _u_2h1p_unfold(z_II, iu, ju, nocc, V, True)
         + _S12 * _u_2h1p_unfold(z_III, iu, ju, nocc, V, False))
    d = np.arange(nocc)
    R[d, d, :] = z_I
    return R


def R_to_csf(R):
    """CSF components of a doublet R; the left inverse of csf_to_R."""
    nocc = R.shape[0]
    iu, ju = pair_indices(nocc)
    d = np.arange(nocc)
    return (R[d, d, :].copy(), _S32 * (R[iu, ju, :] - R[ju, iu, :]),
            _S12 * (R[iu, ju, :] + R[ju, iu, :]))


def R_to_csf_adjoint(y_I, y_II, y_III, nocc):
    """Transpose of R_to_csf, the 2h1p mirror of Q_to_csf_adjoint."""
    iu, ju = pair_indices(nocc)
    V = y_I.shape[1]
    R = (_S32 * _u_2h1p_unfold(y_II, iu, ju, nocc, V, True)
         + _S12 * _u_2h1p_unfold(y_III, iu, ju, nocc, V, False))
    d = np.arange(nocc)
    R[d, d, :] = y_I
    return R


# ---------------------------------------------------------------------------
# pair-channel corrections on satellite vectors
# ---------------------------------------------------------------------------

def _spectator_apply(M, N, X, eps_col):
    """M X + (N X) diag(eps_col): a pair operator M^(u) = M + eps_u N on the
    columns X[:, u] of one spectator u each; N=None drops the spectator term."""
    Y = M @ X
    if N is not None:
        Y += (N @ X) * eps_col[None, :]
    return Y


def _eh_Q(Q, MS, MT, NS, NT, eps_v):
    """Particle-hole corrections on a 2p1h doublet Q (block II):

        Qy[i,a,b] = sum_jc M_T^(b)[ia,jc] Q[j,c,b]
                  + sum_jc M_S^(a)[ib,jc] Q[j,a,c]
                  - 1/2 sum_jc (M_S - M_T)^(a)[ib,jc] Q[j,c,a],

    M^(u) = M + eps_u N the pair operator with the spectator u. MS/MT/NS/NT
    are (O*V, O*V) or None (N=None: no spectator term)."""
    O, V, _ = Q.shape
    n = O * V
    Qc_b = Q.transpose(0, 1, 2).reshape(n, V)          # [(j,c), b] = Q[j,c,b]
    Qa_c = Q.transpose(0, 2, 1).reshape(n, V)          # [(j,c), a] = Q[j,a,c]
    t1 = _spectator_apply(MT, NT, Qc_b, eps_v).reshape(O, V, V)    # [i,a,b]
    t2 = _spectator_apply(MS, NS, Qa_c, eps_v).reshape(O, V, V)    # [i,b,a]
    MD = 0.5 * (MS - MT)
    ND = None if NS is None else 0.5 * (NS - NT)
    t3 = _spectator_apply(MD, ND, Qc_b, eps_v).reshape(O, V, V)    # [i,b,a]
    return t1 + (t2 - t3).transpose(0, 2, 1)


def _eh_R(R, MS, MT, NS, NT, eps_o):
    """Particle-hole corrections on a 2h1p doublet R (block I):

        Ry[i,j,a] = sum_kc M_T^(j)[ia,kc] R[k,j,c]
                  + sum_kc M_S^(i)[ja,kc] R[i,k,c]
                  - 1/2 sum_kc (M_S - M_T)^(i)[ja,kc] R[k,i,c],

    with M^(u) = M + eps_u N and M = -dH for block I."""
    O, _, V = R.shape
    n = O * V
    R_kcj = R.transpose(0, 2, 1).reshape(n, O)          # [(k,c), j] = R[k,j,c]
    R_kci = R.transpose(1, 2, 0).reshape(n, O)          # [(k,c), i] = R[i,k,c]
    t1 = _spectator_apply(MT, NT, R_kcj, eps_o).reshape(O, V, O)   # [i,a,j]
    t2 = _spectator_apply(MS, NS, R_kci, eps_o).reshape(O, V, O)   # [j,a,i]
    MD = 0.5 * (MS - MT)
    ND = None if NS is None else 0.5 * (NS - NT)
    t3 = _spectator_apply(MD, ND, R_kcj, eps_o).reshape(O, V, O)   # [j,a,i]
    return t1.transpose(0, 2, 1) + (t2 - t3).transpose(2, 0, 1)


def _pp_ab_Q(Q, pp, eps_o, which):
    """Ladder-channel corrections on a 2p1h doublet Q in the alpha-beta pair
    form: (dH - eps_i dN) Q[i] with dH = B T + T^T B^T + T^T D T and
    dN = -T^T T, through the amplitudes (o^3 v^2 work)."""
    O, V, _ = Q.shape
    T2 = pp['t'].reshape(-1, V * V)
    Q2 = Q.reshape(O, V * V)
    X = Q2 @ T2.T                                   # [i, kl] = (T Q)
    if which == 'N':
        return -(X @ T2).reshape(O, V, V)
    V2 = pp['V'].reshape(-1, V * V)
    Y = Q2 @ V2.T + X @ pp['D'] + eps_o[:, None] * X
    return (X @ V2 + Y @ T2).reshape(O, V, V)


def _hh_ab_R(R, hh, eps_v, which):
    """Hole-pair corrections on a 2h1p doublet R[i,j,a] in the alpha-beta
    pair form: (dH - eps_a dN) acting on the (i, j) pair of each a."""
    O, _, V = R.shape
    R2 = R.reshape(O * O, V)
    if which == 'N':
        return (hh['dN'] @ R2).reshape(O, O, V)
    return (hh['dH'] @ R2 - (hh['dN'] @ R2) * eps_v[None, :]).reshape(O, O, V)


def _pp_block_II(z_Ip, z_IIp, z_IIIp, pp, eps_o, which):
    """pp corrections in the 2p1h CSF basis: singlet channel on (I', III'),
    triplet on II', spectator -eps_i. which='H' (dH - eps_i dN) or 'N'."""
    V = z_Ip.shape[1]
    Zs = np.concatenate([z_Ip, z_IIIp], axis=1)
    out = []
    for Z, ch in ((Zs, pp['singlet']), (z_IIp, pp['triplet'])):
        if which == 'H':
            out.append(Z @ ch['dH'] - eps_o[:, None] * (Z @ ch['dN']))
        else:
            out.append(Z @ ch['dN'])
    Ys, Yt = out
    return Ys[:, :V], Yt, Ys[:, V:]


def _hh_block_I(z_I, z_II, z_III, hh, eps_v, which):
    """hh corrections in the 2h1p CSF basis: singlet channel on (I, III),
    triplet on II, spectator -eps_a."""
    O = z_I.shape[0]
    Zs = np.concatenate([z_I, z_III], axis=0)
    out = []
    for Z, ch in ((Zs, hh['singlet']), (z_II, hh['triplet'])):
        if which == 'H':
            out.append(ch['dH'] @ Z - (ch['dN'] @ Z) * eps_v[None, :])
        else:
            out.append(ch['dN'] @ Z)
    Ys, Yt = out
    return Ys[:O], Yt, Ys[O:]


# ---------------------------------------------------------------------------
# coupling U (and its block-I mirror), matrix-free through B_aa
# ---------------------------------------------------------------------------

def _alpha_beta_ladder(pp, nocc, nv):
    """T^{ab}[k,l,a,b], the ladder amplitude of the ordered alpha-beta pairs
    (k alpha, l beta) -> (a alpha, b beta): half the sum of the symmetric
    (singlet) and antisymmetric (triplet) extensions."""
    ks, ls = singlet_pair_indices(nocc)
    cs, ds = singlet_pair_indices(nv)
    TS = pp['singlet']['T'] * (np.sqrt(1.0 + (ks == ls))[:, None]
                               * np.sqrt(1.0 + (cs == ds))[None, :])
    kt, lt = pair_indices(nocc)
    ct, dt = pair_indices(nv)
    TT = pp['triplet']['T']
    T = np.zeros((nocc, nocc, nv, nv))
    for k, l in ((ks, ls), (ls, ks)):
        for c, d in ((cs, ds), (ds, cs)):
            T[k[:, None], l[:, None], c[None, :], d[None, :]] = 0.5 * TS
    for (k, l), sk in (((kt, lt), 1.0), ((lt, kt), -1.0)):
        for (c, d), sc in (((ct, dt), 1.0), ((dt, ct), -1.0)):
            T[k[:, None], l[:, None], c[None, :], d[None, :]] += 0.5 * sk * sc * TT
    return T


class _Coupling:
    """U of the Faddeev-ADC(3) in the spin-free components: forward
    (orbital -> Q, R) and adjoint, contracted through the DF factor (or the
    dense chemist tensor) without forming the (satellite x orbital) block.

        Qu[i,a,b;p] = (pa|ib) + sum_jc (pj|cb) T_T[jc,ia] + (pa|cj) T_S[jc,ib]
                      - (pj|ca) T_x[jc,ib] + sum_kl (pk|il) T^{ab}[k,l,a,b],
        Ru[i,j,a;p] = (pi|aj) + sum_kc (cp|jk) T_T[kc,ia] + (ck|ip) T_S[kc,ja]
                      - (cp|ik) T_x[kc,ja] + sum_cd (pc|ad) T^{ab}[i,j,c,d],

    T_S/T_T the TDHF singlet/triplet amplitudes as [j,c,i,a], T_x their
    opposite-spin block (T_S - T_T)/2."""

    def __init__(self, B_aa, eri, nocc, channels):
        if B_aa is None:
            # the dense route factorizes the chemist tensor exactly
            norb = eri.shape[0]
            M = eri.reshape(norb * norb, norb * norb)
            w, Vv = la.eigh(M)
            keep = w > 1e-12 * max(1.0, w.max())
            B_aa = (Vv[:, keep] * np.sqrt(w[keep])).T.reshape(-1, norb, norb)
        self.B = B_aa
        norb = B_aa.shape[1]
        self.O, self.V = nocc, norb - nocc
        O, V = self.O, self.V
        eh = channels['eh']
        self.TS = eh['singlet']['T'].reshape(O, V, O, V)
        self.TT = eh['triplet']['T'].reshape(O, V, O, V)
        self.Tx = 0.5 * (self.TS - self.TT)
        pp = channels['pp']
        self.Tab = (pp['t'] if pp.get('form') == 'ab'
                    else _alpha_beta_ladder(pp, O, V))

    def forward(self, z):
        """(Q, R) = U z for an orbital vector z (norb,)."""
        B, O = self.B, self.O
        o, v = slice(0, O), slice(O, None)
        zB = np.einsum('p,Qpq->Qq', z, B, optimize=True)            # (Q, norb)
        ein = lambda *a: np.einsum(*a, optimize=True)
        Q = ein('Qa,Qib->iab', zB[:, v], B[:, o, v])
        Q += ein('jcb,jcia->iab', ein('Qj,Qcb->jcb', zB[:, o], B[:, v, v]), self.TT)
        Q += ein('acj,jcib->iab', ein('Qa,Qcj->acj', zB[:, v], B[:, v, o]), self.TS)
        Q -= ein('jca,jcib->iab', ein('Qj,Qca->jca', zB[:, o], B[:, v, v]), self.Tx)
        Q += ein('kil,klab->iab', ein('Qk,Qil->kil', zB[:, o], B[:, o, o]), self.Tab)

        R = ein('Qi,Qaj->ija', zB[:, o], B[:, v, o])
        R += ein('cjk,kcia->ija', ein('Qc,Qjk->cjk', zB[:, v], B[:, o, o]), self.TT)
        R += ein('cki,kcja->ija', ein('Qck,Qi->cki', B[:, v, o], zB[:, o]), self.TS)
        R -= ein('cik,kcja->ija', ein('Qc,Qik->cik', zB[:, v], B[:, o, o]), self.Tx)
        R += ein('cad,ijcd->ija', ein('Qc,Qad->cad', zB[:, v], B[:, v, v]), self.Tab)
        return Q, R

    def adjoint(self, Qt, Rt):
        """U^T (Qt, Rt) -> orbital vector (norb,): the transpose of forward."""
        B, O = self.B, self.O
        o, v = slice(0, O), slice(O, None)
        ein = lambda *a: np.einsum(*a, optimize=True)
        Hv = ein('Qib,iab->Qa', B[:, o, v], Qt)
        Ho = ein('Qcb,jcb->Qj', B[:, v, v], ein('jcia,iab->jcb', self.TT, Qt))
        Hv += ein('Qcj,acj->Qa', B[:, v, o], ein('jcib,iab->acj', self.TS, Qt))
        Ho -= ein('Qca,jca->Qj', B[:, v, v], ein('jcib,iab->jca', self.Tx, Qt))
        Ho += ein('Qil,kil->Qk', B[:, o, o], ein('klab,iab->kil', self.Tab, Qt))

        Ho += ein('Qaj,ija->Qi', B[:, v, o], Rt)
        Hv += ein('Qjk,cjk->Qc', B[:, o, o], ein('kcia,ija->cjk', self.TT, Rt))
        Ho += ein('Qck,cki->Qi', B[:, v, o], ein('kcja,ija->cki', self.TS, Rt))
        Hv -= ein('Qik,cik->Qc', B[:, o, o], ein('kcja,ija->cik', self.Tx, Rt))
        Hv += ein('Qad,cad->Qc', B[:, v, v], ein('ijcd,ija->cad', self.Tab, Rt))
        return (ein('Qpa,Qa->p', B[:, :, v], Hv)
                + ein('Qpj,Qj->p', B[:, :, o], Ho))


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

class _Pieces:
    """The Faddeev-ADC(3) ingredients on the restricted CSF segments:
    dH/dN on satellite vectors, U forward/adjoint, and the ADC(3) operator
    whose satellite block is K + C."""

    def __init__(self, s, nocc, static_correction, route, channels):
        if (getattr(s, 'u2_denom_dress', None)
                or getattr(s, 'W_chemist', None) is not None
                or getattr(s, 'W_aux', None) is not None
                or getattr(s, '_is_adc2x', False)):
            raise ValueError("Faddeev-ADC(3) takes no EN dressing, screening "
                             "or ADC(2)-X: its couplings and pair channels "
                             "replace them")
        B_aa, eri = _df_or_dense(s)
        self.s, self.nocc = s, nocc
        self.norb = norb = s.norb
        self.O, self.V = O, V = nocc, norb - nocc
        self.eps_o, self.eps_v = s.eps[:O], s.eps[O:]
        if channels is None:
            channels = pair_channels(s.eps, O, B_aa, eri, route=route)
        self.channels = channels
        self.d = d = s.dimensions(nocc)
        self.F = (np.diag(s.eps) if static_correction is None
                  else np.diag(s.eps) + static_correction)
        self.coupling = _Coupling(B_aa, eri, O, channels)
        self.nPo, self.nPv = d['nII'] // V, d['nIIp'] // O
        self.nsat = d['n2h1p'] + d['n2p1h']

    def split(self, x):
        d, O, V, nPo, nPv = self.d, self.O, self.V, self.nPo, self.nPv
        nI, nII, nIII, nIp, nIIp = d['nI'], d['nII'], d['nIII'], d['nIp'], d['nIIp']
        xI = x[:nI].reshape(O, V)
        xII = x[nI:nI + nII].reshape(nPo, V)
        xIII = x[nI + nII:nI + nII + nIII].reshape(nPo, V)
        y = x[nI + nII + nIII:]
        xIp = y[:nIp].reshape(O, V)
        xIIp = y[nIp:nIp + nIIp].reshape(O, nPv)
        xIIIp = y[nIp + nIIp:].reshape(O, nPv)
        return (xI, xII, xIII), (xIp, xIIp, xIIIp)

    @staticmethod
    def join(b1, b2):
        return np.concatenate([b.ravel() for b in tuple(b1) + tuple(b2)])

    def correction(self, x, which):
        """dH x (which='H') or dN x (which='N') on a satellite vector."""
        eh, pp, hh = (self.channels[k] for k in ('eh', 'pp', 'hh'))
        ehS, ehT = eh['singlet'], eh['triplet']
        b1, b2 = self.split(x)
        if which == 'H':
            args2 = (ehS['dH'], ehT['dH'], ehS['dN'], ehT['dN'])
            args1 = (-ehS['dH'], -ehT['dH'], ehS['dN'], ehT['dN'])
        else:
            args2 = args1 = (ehS['dN'], ehT['dN'], None, None)
        Q = csf_to_Q(*b2, self.V)
        R = csf_to_R(*b1, self.O)
        Qy = _eh_Q(Q, *args2, self.eps_v)
        Ry = _eh_R(R, *args1, self.eps_o)
        if pp.get('form') == 'ab':
            Qy += _pp_ab_Q(Q, pp, self.eps_o, which)
            y2 = Q_to_csf(Qy)
        else:
            y2 = [a + b for a, b in zip(
                Q_to_csf(Qy), _pp_block_II(*b2, pp, self.eps_o, which))]
        if hh.get('form') == 'ab':
            Ry += _hh_ab_R(R, hh, self.eps_v, which)
            y1 = R_to_csf(Ry)
        else:
            y1 = [a + b for a, b in zip(
                R_to_csf(Ry), _hh_block_I(*b1, hh, self.eps_v, which))]
        return self.join(y1, y2)

    def U_forward(self, zp):
        Q, R = self.coupling.forward(zp)
        return self.join(R_to_csf(R), Q_to_csf(Q))

    def U_adjoint(self, x):
        b1, b2 = self.split(x)
        return self.coupling.adjoint(Q_to_csf_adjoint(*b2, self.V),
                                     R_to_csf_adjoint(*b1, self.O))


def build_operator(s, nocc, static_correction=None, route='phonon',
                   channels=None):
    """(aop, diag, dims) of the restricted Faddeev-ADC(3), same segments and
    CSF layout as the restricted ADC(3) operator, whose satellite block K + C
    it reuses (couplings=False: no ADC(3) U built or applied). `channels`
    overrides the pair channels (first_order_channels gives back ADC(3))."""
    P = _Pieces(s, nocc, static_correction, route, channels)
    mod = adc_r_sigma_df if s.B_aa is not None else adc_r_sigma_full
    aop3, diag3, d = mod.build_operator(s, nocc, static_correction,
                                        couplings=False)
    norb = P.norb
    zero_p = np.zeros(norb)

    def inv_sqrt_N(x):
        return apply_inverse_sqrt(lambda y: y + P.correction(y, 'N'), x)

    def aop(z):
        z = np.asarray(z, dtype=float)
        zp, zs = z[:norb], z[norb:]
        w = inv_sqrt_N(zs)
        Hw = aop3(np.concatenate([zero_p, w]))[norb:] + P.correction(w, 'H')
        ys = inv_sqrt_N(Hw + P.U_forward(zp))
        yp = P.F @ zp + P.U_adjoint(w)
        return np.concatenate([yp, ys])

    return aop, diag3.copy(), d


def build_supermatrix(s, nocc, static_correction=None, route='phonon',
                      channels=None):
    """(nH, nH) dense restricted Faddeev-ADC(3) supermatrix, for
    benchmarking: K + C from the dense ADC(3) supermatrix, dH/dN/U by
    columns, N^{-1/2} by eigh."""
    P = _Pieces(s, nocc, static_correction, route, channels)
    norb, nsat = P.norb, P.nsat
    mod = adc_r_dense_df if s.B_aa is not None else adc_r_dense_full
    H3 = mod.build_supermatrix(s, nocc, static_correction)
    eye = np.eye(nsat)
    dH = np.column_stack([P.correction(c, 'H') for c in eye])
    N = eye + np.column_stack([P.correction(c, 'N') for c in eye])
    U = np.column_stack([P.U_forward(c) for c in np.eye(norb)])
    Nm = inverse_sqrt_psd(N, "Faddeev-ADC(3): the metric N")
    Hs = H3[norb:, norb:] + dH
    H = np.empty_like(H3)
    H[:norb, :norb] = P.F
    H[norb:, :norb] = Nm @ U
    H[:norb, norb:] = H[norb:, :norb].T
    H[norb:, norb:] = Nm @ (0.5 * (Hs + Hs.T)) @ Nm
    return H
