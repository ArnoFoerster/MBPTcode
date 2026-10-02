"""Spin-orbital dense Faddeev-ADC(3): the own-channel, no-overlap
approximation of the Faddeev kernel with RPA phonons.

ADC(3) is this approximation with Tamm--Dancoff phonons: its 2p1h resolvent
(omega - K - C)^{-1}, C = C_12 + C_31 + C_32, is the Faddeev combination of
the three pairs. Replacing each pair's TDA propagator by its RPA phonons
turns the resolvent into (omega N - H)^{-1} with

    N = 1 + sum_a [(X_a X_a^+)^{-1} - 1],
    H = K + C + sum_a [X_a^{-+} (Omega_a + s_a) X_a^{-1} - K - C_a],
    U = U^(1) + sum_a U^(1)_{Y,a} T_a,         T_a = Y_a X_a^{-1},

and the self-energy U^+ (omega N - H)^{-1} U. The pairs of block II
(2p1h, (i; a<b)) are the particle-particle pair (ab) with the hole spectator
i (s = -eps_i, the N+2 ppRPA phonons) and the two particle-hole pairs (ia),
(ib) with the particle spectator (s = eps_b, eps_a; TDHF phonons). Block I
(2h1p, (i<j; a)) has the hole-hole pair (ij) with s = -eps_a (the N-2 ppRPA
phonons) and the particle-hole pairs (ia), (ja) with s = eps_j, eps_i.

Every pair enters through three arrays of its own pair space, computed on
the phonon route from the eigenvectors or on the Riccati route from the
amplitudes alone (LinearResponse/riccati.py):

    T  : the channel amplitudes,
    dH : X^{-+} Omega X^{-1} - P   (P the TDA pair Hamiltonian),
    dN : (X X^+)^{-1} - 1,

and a pair operator M (dH or dN) is embedded into the three-particle space
as M + s*dN with its spectator.

This module is the dense reference the restricted route is checked against;
the supermatrix is returned in the symmetric form
[[F, U^+ N^{-1/2}], [N^{-1/2} U, N^{-1/2} H N^{-1/2}]].
"""
import numpy as np

from src.SingleReference.ADC import adc_u_dense_full
from src.SingleReference.ADC.adc_u_sigma_full import apply_U_2h1p, apply_U_2p1h
from src.SingleReference.ADC.adc_u_utils import u2_denominators
from src.SingleReference.ADC.eeADC.ee_utils import unfold_doubles
from src.SingleReference.ADC.solve import inverse_sqrt_psd
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.pp_rpa import (PPRPASolver,
                                                       build_pprpa_matrices,
                                                       pair_indices)
from src.SingleReference.LinearResponse.riccati import (
    channel_corrections_from_amplitudes, channel_corrections_from_phonons,
    require_stable_channel, ring_channel, solve_riccati)

PAIR_ROUTES = ('phonon', 'riccati')


def build_tdhf_matrices(eps, g, nocc):
    """(A, B) of spin-orbital TDHF on the (i, a) pairs, i outer:
    A_{ia,jb} = (eps_a - eps_i) delta + <aj||ib>, B_{ia,jb} = <ab||ij>."""
    nso = len(eps)
    o, v = slice(0, nocc), slice(nocc, nso)
    nv = nso - nocc
    n = nocc * nv
    A = g[v, o, o, v].transpose(2, 0, 1, 3).reshape(n, n).copy()
    A[np.diag_indices(n)] += (eps[None, v] - eps[o, None]).ravel()
    B = g[v, v, o, o].transpose(2, 0, 3, 1).reshape(n, n).copy()
    return A, B


def pair_channels(eps, g, nocc, route='phonon'):
    """{'eh', 'pp', 'hh'} -> {'T', 'dH', 'dN'} for the spin-orbital pairs.

    eh: TDHF (A, B, A), T of shape (nov, nov); dH is block II's, block I
        carries -dH (its pair Hamiltonian is -A at -Omega).
    pp: ppRPA N+2 (A_pp, B_pp, C_pp), T (n_oo, n_vv), dH and dN (n_vv, n_vv).
    hh: ppRPA N-2, amplitudes T_pp^T (n_vv, n_oo), dH and dN (n_oo, n_oo);
        pair Hamiltonian -C_pp.
    """
    if route not in PAIR_ROUTES:
        raise ValueError(f"route={route!r}; expected one of {PAIR_ROUTES}")
    A, B = build_tdhf_matrices(eps, g, nocc)
    App, Bpp, Cpp = build_pprpa_matrices(eps, g, nocc)
    require_stable_channel(A, B, 'spin-orbital TDHF channel')
    if route == 'phonon':
        omega, X, Y = CasidaSolver(A, B).solve()
        T_eh, dH_eh, dN_eh = channel_corrections_from_phonons(A, omega, X, Y)
        res = PPRPASolver(App, Bpp, Cpp).solve()
        T_pp, dH_pp, dN_pp = channel_corrections_from_phonons(
            App, res.omega_add, res.X_add, res.Y_add)
        T_hh, dH_hh, dN_hh = channel_corrections_from_phonons(
            -Cpp, res.omega_rem, res.Y_rem, res.X_rem)
    else:
        T_eh, dH_eh, dN_eh = ring_channel(A, B)
        T_pp, _ = solve_riccati(App, Bpp, Cpp)
        dH_pp, dN_pp = channel_corrections_from_amplitudes(Bpp, Cpp, T_pp)
        T_hh = T_pp.T
        dH_hh, dN_hh = channel_corrections_from_amplitudes(-Bpp.T, -App, T_hh)
    return {'eh': {'T': T_eh, 'dH': dH_eh, 'dN': dN_eh},
            'pp': {'T': T_pp, 'dH': dH_pp, 'dN': dN_pp},
            'hh': {'T': T_hh, 'dH': dH_hh, 'dN': dN_hh}}


def _add_eh_block_II(Hblk, M, dN, eps_v, nocc, nv, spectator=True):
    """Block II particle-hole pairs: the (i,a) pair with spectator b, both
    orderings of the particle pair, into Hblk over (i, a<b)."""
    au, bu = pair_indices(nv)
    npair = len(au)
    pidx = -np.ones((nv, nv), dtype=int)
    pidx[au, bu] = np.arange(npair)
    pidx[bu, au] = np.arange(npair)
    occ = np.arange(nocc)
    for v in range(nv):
        part = np.array([a for a in range(nv) if a != v], dtype=int)
        I = np.repeat(occ, len(part))
        Apart = np.tile(part, nocc)
        rows = I * npair + pidx[Apart, v]
        sign = np.where(Apart < v, 1.0, -1.0)
        sub = I * nv + Apart
        Mv = M[np.ix_(sub, sub)]
        if spectator:
            Mv = Mv + eps_v[v] * dN[np.ix_(sub, sub)]
        Hblk[np.ix_(rows, rows)] += sign[:, None] * Mv * sign[None, :]


def _add_eh_block_I(Hblk, M, dN, eps_o, nocc, nv, spectator=True):
    """Block I particle-hole pairs: the (i,a) pair with spectator j, both
    orderings of the hole pair, into Hblk over (i<j, a)."""
    iu, ju = pair_indices(nocc)
    npair = len(iu)
    pidx = -np.ones((nocc, nocc), dtype=int)
    pidx[iu, ju] = np.arange(npair)
    pidx[ju, iu] = np.arange(npair)
    virt = np.arange(nv)
    for u in range(nocc):
        part = np.array([i for i in range(nocc) if i != u], dtype=int)
        Ipart = np.repeat(part, nv)
        Av = np.tile(virt, len(part))
        rows = pidx[Ipart, u] * nv + Av
        sign = np.where(Ipart < u, 1.0, -1.0)
        sub = Ipart * nv + Av
        Mu = M[np.ix_(sub, sub)]
        if spectator:
            Mu = Mu + eps_o[u] * dN[np.ix_(sub, sub)]
        Hblk[np.ix_(rows, rows)] += sign[:, None] * Mu * sign[None, :]


def correction_blocks(eps, nocc, channels):
    """(dH_I, dN_I, dH_II, dN_II): the pair corrections to H and N in the
    2h1p (i<j, a) and 2p1h (i, a<b) spaces, spectator energies included."""
    nso = len(eps)
    nv = nso - nocc
    eps_o, eps_v = eps[:nocc], eps[nocc:]
    eh, pp, hh = channels['eh'], channels['pp'], channels['hh']

    npo, npv = nocc * (nocc - 1) // 2, nv * (nv - 1) // 2
    dH_II = (np.kron(np.eye(nocc), pp['dH'])
             + np.kron(np.diag(-eps_o), pp['dN']))
    dN_II = np.kron(np.eye(nocc), pp['dN'])
    _add_eh_block_II(dH_II, eh['dH'], eh['dN'], eps_v, nocc, nv)
    _add_eh_block_II(dN_II, eh['dN'], None, eps_v, nocc, nv, spectator=False)

    dH_I = (np.kron(hh['dH'], np.eye(nv))
            + np.kron(hh['dN'], np.diag(-eps_v)))
    dN_I = np.kron(hh['dN'], np.eye(nv))
    _add_eh_block_I(dH_I, -eh['dH'], eh['dN'], eps_o, nocc, nv)
    _add_eh_block_I(dN_I, eh['dN'], None, eps_o, nocc, nv, spectator=False)
    assert dH_I.shape == (npo * nv,) * 2 and dH_II.shape == (nocc * npv,) * 2
    return dH_I, dN_I, dH_II, dN_II


def coupling_blocks(s, nocc, channels):
    """(U_I, U_II): the coupling U and its block-I mirror, rows over the 2h1p /
    2p1h configurations, columns over all spin orbitals p, as columns of the
    ADC(3) U appliers (adc_u_sigma_full.apply_U_*) with the channel
    amplitudes: the ladder term from T_pp, the ring terms from T_eh,

        U_{p;abi} = <pi||ab> + sum_{jc} <pc||jb> T_{jc,ia} - (a<->b)
                    + sum_{k<l} <pi||kl> T_{kl,ab},
        U_{p;ija} = <pa||ij> + sum_{kc} <cj||pk> T_{kc,ia} - (i<->j)
                    + sum_{c<d} <pa||cd> T_{ij,cd}.

    With T -> the first-order doubles this is ADC(3)'s U^(1) + U^(2)."""
    norb = s.norb
    nv = norb - nocc
    iu, ju = pair_indices(nocc)
    au, bu = pair_indices(nv)
    t_lad = unfold_doubles(channels['pp']['T'].ravel(), nocc, norb)
    T4 = channels['eh']['T'].reshape(nocc, nv, nocc, nv)       # [j,c,i,a]
    t_ring = -T4.transpose(2, 0, 1, 3)                         # [i,k,c,a]
    zI = np.zeros((nocc, nocc, nv))
    zII = np.zeros((nocc, nv, nv))
    U_I = np.empty((len(iu) * nv, norb))
    U_II = np.empty((nocc * len(au), norb))
    for p, e in enumerate(np.eye(norb)):
        U_I[:, p] = apply_U_2h1p(s, nocc, e, zI, t2_ijcd=t_lad,
                                 t2_ring=t_ring)[0][iu, ju].ravel()
        U_II[:, p] = apply_U_2p1h(s, nocc, e, zII, t2_ijcd=t_lad,
                                  t2_ring=t_ring)[0][:, au, bu].ravel()
    return U_I, U_II


def first_order_channels(eps, g, nocc):
    """Channels with the first-order doubles as amplitudes and no phonon
    corrections (dH = dN = 0): the ADC(3) limit of the Faddeev form."""
    nso = len(eps)
    nv = nso - nocc
    o, v = slice(0, nocc), slice(nocc, nso)
    t = g[o, o, v, v] / u2_denominators(eps, nocc)[0]          # t_{ij}^{ab}
    iu, ju = pair_indices(nocc)
    au, bu = pair_indices(nv)
    nov = nocc * nv
    T_eh = t.transpose(1, 3, 0, 2).reshape(nov, nov)           # [jc, ia] = t_ij^ac
    T_pp = t[iu, ju][:, au, bu]
    npo, npv = len(iu), len(au)
    return {'eh': {'T': T_eh, 'dH': np.zeros((nov, nov)), 'dN': np.zeros((nov, nov))},
            'pp': {'T': T_pp, 'dH': np.zeros((npv, npv)), 'dN': np.zeros((npv, npv))},
            'hh': {'T': T_pp.T, 'dH': np.zeros((npo, npo)), 'dN': np.zeros((npo, npo))}}


def build_pencil(s, nocc, static_correction=None, route='phonon',
                 channels=None):
    """(H, S) of the Faddeev-ADC(3) Dyson pencil: H z = E S z with
    S = diag(1, N_I, N_II), over the ADC(3) segments (orbitals, 2h1p, 2p1h).
    H is the dense ADC(3) supermatrix (adc_u_dense_full) with the channel
    coupling U in place of ADC(3)'s and the pair corrections dH added."""
    s._require_dense_g('Faddeev-ADC(3) (dense spin-orbital reference)')
    eps, g, norb = s.eps, s.g, s.norb
    if channels is None:
        channels = pair_channels(eps, g, nocc, route=route)
    d = s.dimensions(nocc)
    off1, off2 = norb, norb + d['n2h1p']

    dH_I, dN_I, dH_II, dN_II = correction_blocks(eps, nocc, channels)
    U_I, U_II = coupling_blocks(s, nocc, channels)

    H = adc_u_dense_full.build_supermatrix(s, nocc, static_correction)
    H[off1:off2, :norb] = U_I
    H[:norb, off1:off2] = U_I.T
    H[off2:, :norb] = U_II
    H[:norb, off2:] = U_II.T
    H[off1:off2, off1:off2] += dH_I
    H[off2:, off2:] += dH_II
    S = np.eye(d['nH'])
    S[off1:off2, off1:off2] += dN_I
    S[off2:, off2:] += dN_II
    return H, S


def build_supermatrix(s, nocc, static_correction=None, route='phonon',
                      channels=None):
    """(nH, nH) symmetric Faddeev-ADC(3) supermatrix: the pencil
    of build_pencil transformed with S^{-1/2}; same segments as ADC(3)."""
    H, S = build_pencil(s, nocc, static_correction, route, channels)
    norb = s.norb
    off1 = norb + s.dimensions(nocc)['n2h1p']
    Sm = np.eye(len(S))
    Sm[norb:off1, norb:off1] = inverse_sqrt_psd(
        S[norb:off1, norb:off1], 'Faddeev-ADC(3): the 2h1p metric N')
    Sm[off1:, off1:] = inverse_sqrt_psd(
        S[off1:, off1:], 'Faddeev-ADC(3): the 2p1h metric N')
    return Sm @ H @ Sm
