"""Restricted (closed-shell, spin-free) matrix-free Faddeev-ADC(4).

The physics is that of adc_u_faddeev4 (the Faddeev pencil with the
second-order parquet kernel), assembled from two restricted operators:

- the Faddeev-ADC(3) k1 block N^{-1/2} H_F N^{-1/2} and coupling N^{-1/2} U_F
  of adc_r_faddeev, on its doublet CSFs, plus the static line dressing dC
  (a one-body operator on every line, applied on the spin-free components);
- the ADC(4) blocks outside (k1, k1) of the diagrammatic restricted Dyson
  ADC(4) (sigma_driver.restricted_dyson_operator(...).blocks) on the
  determinants of the alpha-Dyson sector: the coupling U^(<=3), the k1-k2
  coupling E3, the orbital-k2 coupling U3 and the k2 diagonal K3.

In symmetric form, over orbitals + k1 CSFs + k2 determinants,

    [[F,    C^+,                                  U3^+          ],
     [C,    N^{-1/2} (H_F + dC) N^{-1/2},          N^{-1/2} W^+ E3^+],
     [U3,   E3 W N^{-1/2},                         K3            ]],

    C = W^+ U_ADC4^(<=3) + N^{-1/2} U_F - U_an^(<=3),

with W the CSF isometry below and U_an^(<=3) = U_F(T1 + T2) + 1/2 S U^(1)
the analytic order <= 3 part of N^{-1/2} U_F. K3 is diagonal, so the k2
space is folded exactly (closed-form Schur complement, two Lanczos N^{-1/2}
per folded matvec) and the eigenproblem solved by
sigma_driver.downfold_eigensolve.

The restricted Faddeev-ADC(3) (adc_r_faddeev) carries the satellites in
doublet CSFs: 2h1p classes I (i=i; a), II (hole pair triplet-coupled), III
(singlet-coupled), 2p1h classes I' (i; a=a), II', III'. The restricted
ADC(4) operator (sigma_driver.restricted_dyson_operator) carries them as the
determinants of the alpha-Dyson sector (BuildSuperMatrixADCnRestricted.
sector_configs: 2h1p at S_z = -1/2, 2p1h at S_z = +1/2, a spatial index and
a spin per line, canonical spin-orbital order with the spin orbitals of a
spatial orbital p at 2p (alpha) and 2p + 1 (beta), occupied first). The
CSFs are orthonormal combinations of those determinants (the quartet
determinants of the sector carry no CSF):

    I'(i,a)     = |i b; a a, a b>
    II'(i,a<b)  = sqrt(2/3)|i a; a a, b a> + sqrt(1/6)(|i b; a a, b b> + |i b; a b, b a>)
    III'(i,a<b) = (|i b; a a, b b> - |i b; a b, b a>) / sqrt(2)

(|i s; a s', b s''> = a+_{a s'} a+_{b s''} a_{i s}|HF>, a = alpha, b = beta)
and the 2h1p mirror with |i s, j s'; a s''> = a+_{a s''} a_{j s'} a_{i s}|HF>.
csf_isometry returns the sparse isometry W (determinants x CSFs) of one
sector, W^T W = 1, so that a CSF-basis block is W^T (determinant block) W.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from src.Base.orbital_spaces import ov_energies
from src.SingleReference.common.eri_blocks import g_slice
from src.SingleReference.ADC import adc_r_faddeev as FR
from src.SingleReference.ADC import adc_r_sigma_df, adc_r_sigma_full
from src.SingleReference.ADC.adc_r_utils import dimensions
from src.SingleReference.ADC.sigma_driver import (downfold_eigensolve,
                                                  restricted_dyson_operator, schur_shift)
from src.SingleReference.ADC.solve import apply_inverse_sqrt
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.pp_rpa import pair_indices

_S16, _S12, _S23 = np.sqrt(1.0 / 6.0), np.sqrt(0.5), np.sqrt(2.0 / 3.0)
ALPHA, BETA = 0, 1


def _det_keys(cfg, nocc):
    """Canonical spin-orbital tuples of the determinant configs of one
    alpha-Dyson sector (cfg: hole_sp, part_sp, spins as in
    BuildSuperMatrixADCnRestricted.sector_configs) -> {tuple: index}."""
    nh = cfg['hole_sp'].shape[1]
    holes = 2 * cfg['hole_sp'] + cfg['spins'][:, :nh]
    parts = 2 * nocc + 2 * cfg['part_sp'] + cfg['spins'][:, nh:]
    keys = np.concatenate([np.sort(holes, axis=1), np.sort(parts, axis=1)], axis=1)
    return {tuple(k): n for n, k in enumerate(keys.tolist())}


def _canonical(x, y):
    """(sorted pair, sign of the sort) for spin-orbital index arrays x, y."""
    swap = x > y
    return np.where(swap, y, x), np.where(swap, x, y), np.where(swap, -1.0, 1.0)


def csf_isometry(cfg, nocc, nvir, sector):
    """Sparse W (n_det, n_csf) of `sector` ('IP' 2h1p, 'EA' 2p1h): column c
    is CSF c of adc_r_faddeev's class order (I, II, III or I', II', III') in
    the determinant configs `cfg` of the restricted ADC(4) operator."""
    keys = _det_keys(cfg, nocc)
    O, V = nocc, nvir
    rows, cols, vals = [], [], []
    col0 = [0]

    def put(terms, ncol):
        """terms: [(three spin-orbital index arrays, coefficient)], each array
        over the ncol CSFs of one class, in (h, h, p) or (h, p, p) order."""
        c = col0[0] + np.arange(ncol)
        for (x, y, z), coef in terms:
            if sector == 'IP':
                x, y, sg = _canonical(x, y)
                tup = zip(x.tolist(), y.tolist(), z.tolist())
            else:
                y, z, sg = _canonical(y, z)
                tup = zip(x.tolist(), y.tolist(), z.tolist())
            rows.append(np.fromiter((keys[t] for t in tup), dtype=np.int64, count=ncol))
            cols.append(c)
            vals.append(sg * coef)
        col0[0] += ncol

    if sector == 'IP':
        occ = lambda i, s: 2 * i + s
        vir = lambda a, s: 2 * O + 2 * a + s
        i, a = np.repeat(np.arange(O), V), np.tile(np.arange(V), O)
        put([((occ(i, ALPHA), occ(i, BETA), vir(a, BETA)), 1.0)], O * V)
        iu, ju = pair_indices(O)
        i, j = np.repeat(iu, V), np.repeat(ju, V)
        a = np.tile(np.arange(V), len(iu))
        n = len(i)
        put([((occ(i, ALPHA), occ(j, ALPHA), vir(a, ALPHA)), _S23),
             ((occ(i, ALPHA), occ(j, BETA), vir(a, BETA)), _S16),
             ((occ(i, BETA), occ(j, ALPHA), vir(a, BETA)), _S16)], n)
        put([((occ(i, ALPHA), occ(j, BETA), vir(a, BETA)), _S12),
             ((occ(i, BETA), occ(j, ALPHA), vir(a, BETA)), -_S12)], n)
    elif sector == 'EA':
        occ = lambda i, s: 2 * i + s
        vir = lambda a, s: 2 * O + 2 * a + s
        i, a = np.repeat(np.arange(O), V), np.tile(np.arange(V), O)
        put([((occ(i, BETA), vir(a, ALPHA), vir(a, BETA)), 1.0)], O * V)
        au, bu = pair_indices(V)
        i = np.repeat(np.arange(O), len(au))
        a, b = np.tile(au, O), np.tile(bu, O)
        n = len(i)
        put([((occ(i, ALPHA), vir(a, ALPHA), vir(b, ALPHA)), _S23),
             ((occ(i, BETA), vir(a, ALPHA), vir(b, BETA)), _S16),
             ((occ(i, BETA), vir(a, BETA), vir(b, ALPHA)), _S16)], n)
        put([((occ(i, BETA), vir(a, ALPHA), vir(b, BETA)), _S12),
             ((occ(i, BETA), vir(a, BETA), vir(b, ALPHA)), -_S12)], n)
    else:
        raise ValueError(f"sector={sector!r}; expected 'IP' or 'EA'")
    d = dimensions(O + V, O)
    ncsf = d['n2h1p'] if sector == 'IP' else d['n2p1h']
    assert col0[0] == ncsf
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(keys), ncsf))


# ---------------------------------------------------------------------------
# the static line dressing on the CSF segments
# ---------------------------------------------------------------------------

def line_self_energies(eps, nocc, B_aa=None, eri_chemist=None):
    """(s_lt, s_gt): the Hermitized non-resonant GF2 self-energies of the
    restricted lines, s_pq = [Sigma_pq(eps_p) + Sigma_pq(eps_q)] / 2, with

        Sigma<_ab(z) = sum_klc <ac|kl> (2 <bc|kl> - <bc|lk>) / (z + e_c - e_k - e_l)
        Sigma>_ij(z) = sum_kcd <ik|cd> (2 <jk|cd> - <jk|dc>) / (z + e_k - e_c - e_d)

    on the virtual (particle) and occupied (hole) lines; the closed-shell
    spin sums of adc_u_faddeev4.static_line_dressing's Sigma< and Sigma>."""
    eps = np.asarray(eps)
    norb = len(eps)
    o, v = slice(0, nocc), slice(nocc, norb)
    eo, ev = eps[o], eps[v]
    G = g_slice(B_aa, eri_chemist, v, v, o, o)                   # <ac|kl>
    den = (ev[:, None, None, None] + ev[None, :, None, None]
           - eo[None, None, :, None] - eo[None, None, None, :])
    X = 2.0 * G - G.transpose(0, 1, 3, 2)
    s_lt = np.einsum('ackl,bckl->ab', G / den, X, optimize=True)
    G = g_slice(B_aa, eri_chemist, o, o, v, v)                   # <ik|cd>
    den = (eo[:, None, None, None] + eo[None, :, None, None]
           - ev[None, None, :, None] - ev[None, None, None, :])
    X = 2.0 * G - G.transpose(0, 1, 3, 2)
    s_gt = np.einsum('ikcd,jkcd->ij', G / den, X, optimize=True)
    return 0.5 * (s_lt + s_lt.T), 0.5 * (s_gt + s_gt.T)


def line_dressing_Q(Q, s_lt, s_gt):
    """dC on a 2p1h doublet Q[i,a,b]: a one-body operator on every line,
    + s< on the two particles, - s> on the hole."""
    return (np.einsum('ac,icb->iab', s_lt, Q, optimize=True)
            + np.einsum('bd,iad->iab', s_lt, Q, optimize=True)
            - np.einsum('ij,jab->iab', s_gt, Q, optimize=True))


def line_dressing_R(R, s_lt, s_gt):
    """dC on a 2h1p doublet R[i,j,a]: + s> on the two holes, - s< on the
    particle."""
    return (np.einsum('ik,kja->ija', s_gt, R, optimize=True)
            + np.einsum('jl,ila->ija', s_gt, R, optimize=True)
            - np.einsum('ab,ijb->ija', s_lt, R, optimize=True))


# ---------------------------------------------------------------------------
# first- and second-order channel amplitudes, the analytic coupling
# ---------------------------------------------------------------------------

def _particle_ladder(B_aa, eri_chemist, nocc, t, mem_gb=2.0):
    """sum_ef t[k,l,e,f] <ef|cd> (the vvvv ladder of an alpha-beta pair
    amplitude), through the DF factor in auxiliary chunks or the dense tensor."""
    norb = nocc + t.shape[2]
    v = slice(nocc, norb)
    if B_aa is None:
        return np.einsum('klef,efcd->klcd', t, g_slice(None, eri_chemist, v, v, v, v),
                         optimize=True)
    Bvv = B_aa[:, v, v]
    naux = Bvv.shape[0]
    step = max(1, min(naux, int(mem_gb * 1.25e8 / max(t.size, 1))))
    out = np.zeros_like(t)
    for q in range(0, naux, step):
        Bc = Bvv[q:q + step]
        U = np.einsum('Qce,klef->Qklcf', Bc, t, optimize=True)
        out += np.einsum('Qklcf,Qdf->klcd', U, Bc, optimize=True)
    return out


def _eh_recouple(t):
    """TDHF singlet/triplet layouts [j,c,i,a] of an alpha-beta doubles
    amplitude t[i,j,a,b] (adc_r_faddeev.ccd_channels), symmetrized."""
    O, _, V, _ = t.shape
    tx = t.transpose(1, 3, 0, 2)
    ty = t.transpose(1, 2, 0, 3)
    out = {}
    for spin, T in (('singlet', 2.0 * tx - ty), ('triplet', -ty)):
        T = T.reshape(O * V, O * V)
        out[spin] = 0.5 * (T + T.T)
    return out


def second_order_doubles_ccd(mycc):
    """The complete second-order alpha-beta doubles t^(2)[i,j,a,b] from a
    pyscf RHF CC object (DF or not; its integrals, canonical orbitals):
    update_amps maps t2 to [V + L(t2) + Q(t2)] / D with the CCD residual
    quadratic, so t^(2) = u(t1) - t1 - [u(2 t1) - 2 u(t1) + u(0)] / 2,
    t1 = u(0) the first-order doubles, at zero singles."""
    eris = mycc.ao2mo()
    O = mycc.nocc
    V = mycc.nmo - O
    z1 = np.zeros((O, V))
    u = lambda t: np.asarray(mycc.update_amps(z1, t, eris)[1])
    u0 = u(np.zeros((O, O, V, V)))
    u1, u2 = u(u0), u(2.0 * u0)
    return u1 - u0 - 0.5 * (u2 - 2.0 * u1 + u0)


def channel_amplitudes(eps, nocc, B_aa=None, eri_chemist=None, route='phonon',
                       t2_second=None):
    """First- and second-order parts of the restricted channel amplitudes:
    {'eh': {spin: (T1, T2)}, 'pp': (t1, t2)}, eh in the TDHF [j,c,i,a]
    layouts, pp as the alpha-beta pair amplitude t[k,l,c,d] (hh is its
    transpose).

    RPA routes ('phonon', 'riccati'): own channel. eh per spin from the
    Casida matrices, D0 T2 + T2 D0 = -(A1 T1 + T1 A1); pp from the ladder
    Riccati equation, D t2 = -(<kl|mn> t1 + t1 <ef|cd>). 'ccd': the complete
    second-order doubles t2_second (second_order_doubles_ccd), recoupled into
    the eh channels as adc_r_faddeev.ccd_channels does."""
    eps = np.asarray(eps)
    norb = len(eps)
    o, v = slice(0, nocc), slice(nocc, norb)
    eo, ev = eps[o], eps[v]
    Vab = g_slice(B_aa, eri_chemist, o, o, v, v)                 # <kl|cd>
    D = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])
    t1 = -Vab / D
    if route == 'ccd':
        if t2_second is None:
            raise ValueError("route 'ccd' needs t2_second, the second-order doubles")
        t2 = np.asarray(t2_second)
        e1, e2 = _eh_recouple(t1), _eh_recouple(t2)
        return {'eh': {s: (e1[s], e2[s]) for s in ('singlet', 'triplet')}, 'pp': (t1, t2)}
    if route not in ('phonon', 'riccati'):
        raise ValueError(f"route={route!r}; expected 'phonon', 'riccati' or 'ccd'")
    if t2_second is not None:
        raise ValueError("t2_second belongs to route 'ccd' only")
    G = g_slice(B_aa, eri_chemist, o, o, o, o)                   # <kl|mn>
    t2 = -(np.einsum('klmn,mncd->klcd', G, t1, optimize=True)
           + _particle_ladder(B_aa, eri_chemist, nocc, t1)) / D
    d0 = ov_energies(eps, nocc)
    den = d0[:, None] + d0[None, :]
    e1 = _eh_recouple(t1)
    eh = {}
    for spin in ('singlet', 'triplet'):
        A, _ = LinearResponseSolver(eps, coeff_df=B_aa, eri_chemist=eri_chemist
                                    ).build_casida_matrices(nocc, lBSE=True,
                                                            triplet=(spin == 'triplet'))
        A1 = A - np.diag(d0)
        T1 = e1[spin]
        eh[spin] = (T1, -(A1 @ T1 + T1 @ A1) / den)
    return {'eh': eh, 'pp': (t1, t2)}


def taylor_channels(eps, nocc, amps, B_aa=None, eri_chemist=None):
    """(chan_U, chan_N, chan_0): pair channels for the analytic order <= 3
    coupling, in adc_r_faddeev's channel format (pp/hh in the alpha-beta
    form). chan_U carries T1 + T2 (its U is U_F(T1 + T2)), chan_N the leading
    metric dN = -T1^+ T1 in every channel, chan_0 no amplitude (its U is
    U^(1)); dH = 0 in all three."""
    eps = np.asarray(eps)
    norb = len(eps)
    O, V = nocc, norb - nocc
    o, v = slice(0, O), slice(O, norb)
    Vab = g_slice(B_aa, eri_chemist, o, o, v, v)
    Dm = np.zeros((O * O, O * O))
    zo, zv = np.zeros((O * V, O * V)), np.zeros((O * O, O * O))

    def chans(eh_T, t, dN):
        eh = {}
        for spin, T in eh_T.items():
            eh[spin] = {'T': T, 'dH': zo, 'dN': (-T.T @ T) if dN else zo}
        T2d = t.reshape(O * O, V * V)
        hh = {'form': 'ab', 'dH': zv, 'dN': -(T2d @ T2d.T) if dN else zv}
        return {'eh': eh, 'pp': {'form': 'ab', 't': t, 'V': Vab, 'D': Dm}, 'hh': hh}

    t1, t2 = amps['pp']
    U = chans({s: amps['eh'][s][0] + amps['eh'][s][1] for s in amps['eh']}, t1 + t2, False)
    N = chans({s: amps['eh'][s][0] for s in amps['eh']}, t1, True)
    Z = chans({s: 0.0 * amps['eh'][s][0] for s in amps['eh']}, 0.0 * t1, False)
    return U, N, Z


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

class FaddeevADC4Operator:
    """The restricted Faddeev-ADC(4) operator.

    s: the restricted Faddeev solver (ADCSolverRestricted, B_aa or eri);
    gen: the generated restricted ADC(4) solver on the same orbitals
    (BuildSuperMatrixADCnRestricted); extra_leaves: its DF leaves
    ({'B_aa', 'B_bb'}) or None. static_correction: Sigma(inf) (norb, norb),
    through fourth order for an ADC(4)-exact pencil. route/channels: the pair
    channels (adc_r_faddeev.pair_channels; route 'ccd' takes channels from
    adc_r_faddeev.ccd_channels and t2_second, the second-order doubles).
    line_dressing=False drops dC (a control, exact through third order)."""

    def __init__(self, s, gen, nocc, static_correction=None, route='phonon',
                 channels=None, t2_second=None, extra_leaves=None,
                 laplace_ntau=None, line_dressing=True):
        if route == 'ccd' and channels is None:
            raise ValueError("route 'ccd' takes its channels from "
                             "adc_r_faddeev.ccd_channels (the CCD amplitude)")
        B_aa = s.B_aa
        eri = s.eri if B_aa is None else None
        self.norb = norb = s.norb
        self.O, self.V = nocc, norb - nocc
        F = np.diag(s.eps) if static_correction is None else np.diag(s.eps) + static_correction
        self.F = F
        P = FR.Pieces(s, nocc, static_correction, route, channels)
        self.P = P
        mod = adc_r_sigma_df if B_aa is not None else adc_r_sigma_full
        self.aop3, diag3, self.d = mod.build_operator(s, nocc, static_correction,
                                                      couplings=False)
        self.nI = self.d['n2h1p']
        self.n1 = self.d['n2h1p'] + self.d['n2p1h']
        self.B = restricted_dyson_operator(
            gen, 'adc4', extra_leaves=extra_leaves, f_orb=F,
            laplace_ntau=laplace_ntau, construction='diagrammatic')[0].blocks
        self.W = {sec: csf_isometry(self.B.cfg1[sec], nocc, self.V, sec)
                  for sec in ('IP', 'EA')}
        self.K2 = np.concatenate([self.B.K2['IP'], self.B.K2['EA']])
        self.m1 = len(self.B.K2['IP'])
        self.line_dressing = line_dressing
        if line_dressing:
            self.s_lt, self.s_gt = line_self_energies(s.eps, nocc, B_aa, eri)
        amps = channel_amplitudes(s.eps, nocc, B_aa, eri, route=route,
                                  t2_second=t2_second)
        cU, cN, cZ = taylor_channels(s.eps, nocc, amps, B_aa, eri)
        PU, PN, PZ = (FR.Pieces(s, nocc, None, route, c) for c in (cU, cN, cZ))
        cols = []
        for e in np.eye(norb):
            u4 = np.concatenate([self.W['IP'].T @ self.B.u_k1('IP', e),
                                 self.W['EA'].T @ self.B.u_k1('EA', e)])
            u_an = PU.U_forward(e) - 0.5 * PN.correction(PZ.U_forward(e), 'N')
            cols.append(u4 + self.inv_sqrt_N(P.U_forward(e)) - u_an)
        self.C = np.column_stack(cols)                # (n_k1 CSF, norb)
        self.diag1 = np.concatenate([np.diag(F), diag3[norb:]])
        self.diag = np.concatenate([self.diag1, self.K2])

    # -- pieces on CSF k1 vectors ------------------------------------------
    def inv_sqrt_N(self, x):
        return apply_inverse_sqrt(lambda y: y + self.P.correction(y, 'N'), x)

    def dC(self, x):
        P = self.P
        b1, b2 = P.split(x)
        Q = FR.csf_to_Q(*b2, self.V)
        R = FR.csf_to_R(*b1, self.O)
        return P.join(FR.R_to_csf(line_dressing_R(R, self.s_lt, self.s_gt)),
                      FR.Q_to_csf(line_dressing_Q(Q, self.s_lt, self.s_gt)))

    def HF(self, w):
        """(H_F + dC) w on a CSF k1 vector (the pencil's H, before N^{-1/2})."""
        y = (self.aop3(np.concatenate([np.zeros(self.norb), w]))[self.norb:]
             + self.P.correction(w, 'H'))
        if self.line_dressing:
            y = y + self.dC(w)
        return y

    def e3(self, w):
        """E3 W w: k1 CSF -> k2 determinants (IP then EA)."""
        nI = self.nI
        return np.concatenate([self.B.e3('IP', self.W['IP'] @ w[:nI]),
                               self.B.e3('EA', self.W['EA'] @ w[nI:])])

    def e3_adj(self, v2):
        m1 = self.m1
        return np.concatenate([self.W['IP'].T @ self.B.e3_adj('IP', v2[:m1]),
                               self.W['EA'].T @ self.B.e3_adj('EA', v2[m1:])])

    def u3(self, vo):
        return np.concatenate([self.B.u_k2('IP', vo), self.B.u_k2('EA', vo)])

    def u3_adj(self, v2):
        m1 = self.m1
        return self.B.u_k2_adj('IP', v2[:m1]) + self.B.u_k2_adj('EA', v2[m1:])

    # -- operators ---------------------------------------------------------
    def full(self):
        """The LinearOperator over orbitals + k1 CSFs + k2 (IP, EA)."""
        norb, n1 = self.norb, self.n1
        n = norb + n1 + len(self.K2)

        def matvec(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            vo, x1, x2 = x[:norb], x[norb:norb + n1], x[norb + n1:]
            w = self.inv_sqrt_N(x1)
            y1 = self.inv_sqrt_N(self.HF(w) + self.e3_adj(x2)) + self.C @ vo
            yo = self.F @ vo + self.C.T @ x1 + self.u3_adj(x2)
            y2 = self.K2 * x2 + self.e3(w) + self.u3(vo)
            return np.concatenate([yo, y1, y2])

        return LinearOperator((n, n), matvec=matvec, dtype=float)

    def k1(self):
        """The k<=1 block (orbitals + k1 CSFs), the k2 space switched off."""
        norb, n1 = self.norb, self.n1

        def matvec(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            vo, x1 = x[:norb], x[norb:]
            y1 = self.inv_sqrt_N(self.HF(self.inv_sqrt_N(x1))) + self.C @ vo
            return np.concatenate([self.F @ vo + self.C.T @ x1, y1])

        return LinearOperator((norb + n1,) * 2, matvec=matvec, dtype=float)

    def folded(self, omega, clamped=None):
        """The k2 space folded in exactly at fixed omega (K3 diagonal)."""
        norb, n1 = self.norb, self.n1
        g = 1.0 / schur_shift(omega, self.K2, [] if clamped is None else clamped,
                              'FaddeevADC4Operator')

        def matvec(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            vo, x1 = x[:norb], x[norb:]
            w = self.inv_sqrt_N(x1)
            r = g * (self.e3(w) + self.u3(vo))
            y1 = self.inv_sqrt_N(self.HF(w) + self.e3_adj(r)) + self.C @ vo
            yo = self.F @ vo + self.C.T @ x1 + self.u3_adj(r)
            return np.concatenate([yo, y1])

        return LinearOperator((norb + n1,) * 2, matvec=matvec, dtype=float)

    def dense_blocks(self):
        """The symmetric-form blocks as dense arrays, for validation-scale
        systems: F, C (k1 x orbitals), A11 = N^{-1/2}(H_F + dC)N^{-1/2},
        E (k2 x k1) = E3 W N^{-1/2}, U3 (k2 x orbitals), K2."""
        eye1 = np.eye(self.n1)
        Nm = np.column_stack([self.inv_sqrt_N(e) for e in eye1])
        Nm = 0.5 * (Nm + Nm.T)
        HN = np.column_stack([self.HF(c) for c in Nm.T])
        A11 = Nm @ HN
        E = np.column_stack([self.e3(c) for c in Nm.T])
        U3 = np.column_stack([self.u3(e) for e in np.eye(self.norb)])
        return {'F': self.F, 'C': self.C, 'A11': 0.5 * (A11 + A11.T), 'E': E,
                'U3': U3, 'K2': self.K2}

    def solve(self, homo_index, v0=None, tol=1e-7, davidson_tol=None, max_outer=None,
              on_shell=False, max_cycle=200, verbose=False):
        """One root by the downfold (sigma_driver.downfold_eigensolve), followed
        from the Koopmans vector at homo_index (or v0, a k<=1 vector)."""
        clamped = []
        kw = {} if max_outer is None else {'max_outer': max_outer}
        out = downfold_eigensolve(lambda om: self.folded(om, clamped), self.k1(),
                                  self.diag1, homo_index, self.norb, v0=v0, tol=tol,
                                  davidson_tol=davidson_tol, on_shell=on_shell,
                                  max_cycle=max_cycle, verbose=verbose,
                                  where='FaddeevADC4Operator.solve', **kw)
        out.update({'n_k1': self.norb + self.n1, 'n_k2': len(self.K2),
                    'schur_pole_clamped': bool(clamped)})
        return out
