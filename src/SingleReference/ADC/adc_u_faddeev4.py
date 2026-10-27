"""Spin-orbital dense Faddeev-ADC(4): the Faddeev pencil with the
second-order parquet kernel.

Block II of the self-energy (block I by the particle-hole mirror) is

    Sigma(omega) = U (omega N - H)^{-1} U^+     on 2p1h + 3p2h,
    N = diag(N_F, 1),   H = [[H_F + dC, E3^+], [E3, K3]],   U = (U_F + dU, U3),

with H_F, N_F, U_F the Faddeev-ADC(3) pencil (adc_u_faddeev), E3, K3, U3 the
k1-k2 coupling, the k2 diagonal and the orbital-k2 coupling of diagrammatic
Dyson ADC(4) (sigma_driver.build_sigma_operator, construction 'diagrammatic'),
and dC the static line dressing: the non-resonant second-order (GF2)
self-energy on every line at the line energies, Hermitized,

    block II  dC_{abi,cdj} = d_ij P-_ab P-_cd d_bd s<_ac - d_ac d_bd s>_ij,
    block I   dC_{ija,klb} = d_ab P-_ij P-_kl d_jl s>_ik - d_ik d_jl s<_ab,
    s_pq = [Sigma_pq(eps_p) + Sigma_pq(eps_q)] / 2,

Sigma< (2h1p intermediate states) on particle lines and Sigma> (2p1h) on hole
lines. The coupling is taken in the symmetric form

    N_F^{-1/2} (U_F + dU) = U_ADC4^(<=3) + [N_F^{-1/2} U_F - U_an^(<=3)],

U_an^(<=3) = U_F(T^(1) + T^(2)) + 1/2 S U^(1) the analytic order <= 3 part of
N_F^{-1/2} U_F (S = -dN^(2) = sum_a T^(1)+ T^(1), embedded). The bracket is the
Faddeev resummation remainder, O(v^4). T^(2) is the second-order part of the
route's amplitudes: on the RPA routes ('phonon', 'riccati') the own-channel
second-order Riccati amplitude D0 T2 + T2 A0 = -(D1 T1 + T1 A1) per channel,
on 'ccd' the complete second-order doubles.

The pencil is exact through fourth order in Sigma (Sigma(inf) through fourth
order passed as static_correction, as for ADC(4)): the difference to
diagrammatic ADC(4) goes as v^5 (tests/test_faddeev_adc4.py).

The supermatrix is returned in the symmetric form over the ADC(4) segments
(orbitals, 2h1p, 2p1h, 3h2p, 3p2h):

    [[F,      U^+,                 U3^+       ],
     [U,      N^{-1/2}(H_F+dC)N^{-1/2},  N^{-1/2} E3^+],
     [U3,     E3 N^{-1/2},          K3         ]].

This module is the dense reference; it needs the generated ADC(4) runtime.
"""
import numpy as np

from src.Base.orbital_spaces import ov_energies
from src.SingleReference.ADC.adc_u_faddeev import (build_pencil, build_tdhf_matrices,
                                                   correction_blocks, coupling_blocks,
                                                   first_order_channels, pair_channels)
from src.SingleReference.ADC.sigma_driver import build_sigma_operator
from src.SingleReference.ADC.solve import inverse_sqrt_psd
from src.SingleReference.CC.amplitudes import doubles_residual
from src.SingleReference.LinearResponse.pp_rpa import build_pprpa_matrices, pair_indices


def _zero_channels(T_eh, T_pp, dN=False):
    """Channels with amplitudes T and no phonon corrections; dN=True puts the
    leading metric dN = -T^+ T into each channel (hh: amplitude T_pp^T)."""
    z = lambda n: np.zeros((n, n))
    ch = {'eh': {'T': T_eh, 'dH': z(T_eh.shape[1]), 'dN': z(T_eh.shape[1])},
          'pp': {'T': T_pp, 'dH': z(T_pp.shape[1]), 'dN': z(T_pp.shape[1])},
          'hh': {'T': T_pp.T, 'dH': z(T_pp.shape[0]), 'dN': z(T_pp.shape[0])}}
    if dN:
        ch['eh']['dN'] = -T_eh.T @ T_eh
        ch['pp']['dN'] = -T_pp.T @ T_pp
        ch['hh']['dN'] = -T_pp @ T_pp.T
    return ch


def _doubles_to_channels(t, nocc, nv):
    """t[i,j,a,b] -> (T_eh, T_pp) in the channel layouts of adc_u_faddeev."""
    iu, ju = pair_indices(nocc)
    au, bu = pair_indices(nv)
    nov = nocc * nv
    return t.transpose(1, 3, 0, 2).reshape(nov, nov), t[iu, ju][:, au, bu]


def second_order_doubles(eps, g, nocc):
    """The complete second-order doubles t^(2)[i,j,a,b] (HF reference,
    canonical orbitals): the part of the CCD residual linear in t^(1),
    divided by the denominator. The CCD residual is quadratic in t, so its
    quadratic part at t^(1) is [R(2t) - 2R(t) + R(0)]/2 exactly."""
    nso = len(eps)
    o, v = slice(0, nocc), slice(nocc, nso)
    nv = nso - nocc
    f = np.diag(eps)
    eo, ev = eps[o], eps[v]
    e_abij = 1.0 / (eo[None, None, :, None] + eo[None, None, None, :]
                    - ev[:, None, None, None] - ev[None, :, None, None])
    z1 = np.zeros((nv, nocc))
    z3 = np.zeros((nv, nv, nv, nocc, nocc, nocc))
    R = lambda t: doubles_residual(z1, t, z3, f, g, o, v)
    t1 = g[v, v, o, o] * e_abij
    r0, r1, r2 = R(0 * t1), R(t1), R(2 * t1)
    quad = 0.5 * (r2 - 2 * r1 + r0)
    return ((r1 - quad) * e_abij).transpose(2, 3, 0, 1)


def channel_amplitudes(eps, g, nocc, route='phonon'):
    """{'eh': (T1, T2), 'pp': (T1, T2)}: the first- and second-order parts of
    the channel amplitudes of `route` (hh is pp transposed).

    RPA routes ('phonon', 'riccati'): own-channel second order from the
    Riccati equation B^+ + D T + T A + T B T = 0 split into zeroth-order
    diagonals D0, A0 and first-order parts D1, A1:
    D0 T2 + T2 A0 = -(D1 T1 + T1 A1). 'ccd': the complete second-order
    doubles (second_order_doubles) in every channel."""
    nso = len(eps)
    nv = nso - nocc
    ch1 = first_order_channels(eps, g, nocc)
    T1e, T1p = ch1['eh']['T'], ch1['pp']['T']
    if route == 'ccd':
        T2e, T2p = _doubles_to_channels(second_order_doubles(eps, g, nocc), nocc, nv)
        return {'eh': (T1e, T2e), 'pp': (T1p, T2p)}
    if route not in ('phonon', 'riccati'):
        raise ValueError(f"route={route!r}; expected 'phonon', 'riccati' or 'ccd'")
    ev, eo = eps[nocc:], eps[:nocc]
    A, _ = build_tdhf_matrices(eps, g, nocc)
    d0 = ov_energies(eps, nocc)
    A1 = A - np.diag(d0)
    T2e = -(A1 @ T1e + T1e @ A1) / (d0[:, None] + d0[None, :])
    App, _, Cpp = build_pprpa_matrices(eps, g, nocc)
    cu, du = pair_indices(nv)
    ku, lu = pair_indices(nocc)
    a0, c0 = ev[cu] + ev[du], -(eo[ku] + eo[lu])
    T2p = -((Cpp - np.diag(c0)) @ T1p + T1p @ (App - np.diag(a0))) / (c0[:, None] + a0[None, :])
    return {'eh': (T1e, T2e), 'pp': (T1p, T2p)}


def coupling_taylor(s, nocc, amps):
    """(U_I, U_II): the analytic order <= 3 part of N_F^{-1/2} U_F,
    U_F(T1 + T2) + 1/2 S U^(1) with S = -dN^(2) = sum_a T1^+ T1 embedded."""
    T1e, T2e = amps['eh']
    T1p, T2p = amps['pp']
    UI, UII = coupling_blocks(s, nocc, _zero_channels(T1e + T2e, T1p + T2p))
    U0I, U0II = coupling_blocks(s, nocc, _zero_channels(0 * T1e, 0 * T1p))
    _, dNI, _, dNII = correction_blocks(s.eps, nocc, _zero_channels(T1e, T1p, dN=True))
    return UI - 0.5 * dNI @ U0I, UII - 0.5 * dNII @ U0II


def _pair_one_body(h, n):
    """The one-body operator h on the antisymmetric pairs p<q of n orbitals:
    <pq| h1 + h2 |rs> = d_qs h_pr - d_qr h_ps - d_ps h_qr + d_pr h_qs."""
    pu, qu = pair_indices(n)
    e = np.eye(n)
    M = (np.einsum('pr,qs->pqrs', h, e) + np.einsum('pr,qs->pqrs', e, h))
    M = M - M.transpose(0, 1, 3, 2)
    return M[pu, qu][:, pu, qu]


def static_line_dressing(eps, g, nocc):
    """(dC_I, dC_II): the static line dressing on the 2h1p (i<j, a) and 2p1h
    (i, a<b) spaces. Sigma<_pq(z) = 1/2 sum_klc <pc||kl><kl||qc>/(z + e_c - e_k - e_l)
    on particle lines, Sigma>_pq(z) = 1/2 sum_kcd <pk||cd><cd||qk>/(z + e_k - e_c - e_d)
    on hole lines, each at the line energy of its row and Hermitized."""
    n = len(eps)
    nv = n - nocc
    o, v = slice(0, nocc), slice(nocc, n)
    eo, ev = eps[o], eps[v]
    gv = g[v, v, o, o]                                       # <pc||kl>
    den = (ev[:, None, None, None] + ev[None, :, None, None]
           - eo[None, None, :, None] - eo[None, None, None, :])
    s_lt = 0.5 * np.einsum('pckl,qckl->pq', gv / den, gv)    # rows at eps_p
    go = g[o, o, v, v]                                       # <pk||cd>
    den = (eo[:, None, None, None] + eo[None, :, None, None]
           - ev[None, None, :, None] - ev[None, None, None, :])
    s_gt = 0.5 * np.einsum('pkcd,qkcd->pq', go / den, go)
    hv = 0.5 * (s_lt + s_lt.T)
    ho = 0.5 * (s_gt + s_gt.T)
    npo, npv = nocc * (nocc - 1) // 2, nv * (nv - 1) // 2
    dC_II = np.kron(np.eye(nocc), _pair_one_body(hv, nv)) - np.kron(ho, np.eye(npv))
    dC_I = np.kron(_pair_one_body(ho, nocc), np.eye(nv)) - np.kron(np.eye(npo), hv)
    return dC_I, dC_II


def _dense(op):
    n = op.shape[0]
    return np.column_stack([op.matvec(e) for e in np.eye(n)])


def adc4_supermatrix(s, nocc, static_correction=None):
    """Dense diagrammatic Dyson ADC(4) supermatrix (orbitals, 2h1p, 2p1h,
    3h2p, 3p2h) through its matvecs, symmetrized (it is symmetric to the
    arithmetic floor)."""
    op, _, _ = build_sigma_operator(s, nocc, 'adc4', static_correction=static_correction,
                                    construction='diagrammatic')
    M = _dense(op)
    return 0.5 * (M + M.T)


def build_supermatrix(s, nocc, static_correction=None, route='phonon', t=None,
                      line_dressing=True, adc4=None):
    """(n, n) symmetric Faddeev-ADC(4) supermatrix over the ADC(4) segments.

    route, t: the pair route of adc_u_faddeev.pair_channels ('ccd' needs the
    spin-orbital CCD amplitude t[i,j,a,b]). line_dressing=False drops dC
    (a control: exact through third order only). adc4: the dense ADC(4)
    supermatrix at the same static_correction, if already built."""
    s.require_dense_g('Faddeev-ADC(4) (dense spin-orbital reference)')
    eps, g, norb = s.eps, s.g, s.norb
    d = s.dimensions(nocc)
    k1 = slice(norb, d['nH'])
    off2 = norb + d['n2h1p']
    if adc4 is None:
        adc4 = adc4_supermatrix(s, nocc, static_correction)
    channels = pair_channels(eps, g, nocc, route=route, t=t)
    H, S = build_pencil(s, nocc, static_correction, route, channels)
    Hk1 = H[k1, k1].copy()
    if line_dressing:
        dC_I, dC_II = static_line_dressing(eps, g, nocc)
        Hk1[:d['n2h1p'], :d['n2h1p']] += dC_I
        Hk1[d['n2h1p']:, d['n2h1p']:] += dC_II
    Nm = np.zeros_like(Hk1)
    Nm[:d['n2h1p'], :d['n2h1p']] = inverse_sqrt_psd(
        S[norb:off2, norb:off2], 'Faddeev-ADC(4): the 2h1p metric N')
    Nm[d['n2h1p']:, d['n2h1p']:] = inverse_sqrt_psd(
        S[off2:d['nH'], off2:d['nH']], 'Faddeev-ADC(4): the 2p1h metric N')
    UI, UII = coupling_taylor(s, nocc, channel_amplitudes(eps, g, nocc, route))
    rem = Nm @ H[k1, :norb] - np.vstack([UI, UII])
    M = adc4.copy()
    M[k1, k1] = Nm @ Hk1 @ Nm
    M[k1, :norb] = adc4[k1, :norb] + rem
    M[:norb, k1] = M[k1, :norb].T
    k2 = slice(d['nH'], M.shape[0])
    M[k1, k2] = Nm @ adc4[k1, k2]
    M[k2, k1] = M[k1, k2].T
    return M
