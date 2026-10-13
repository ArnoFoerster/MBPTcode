"""Faddeev-ADC(3): the own-channel, no-overlap Faddeev kernel with RPA phonons.

Two implementations are checked against each other and against ADC(3):

- the dense spin-orbital reference (ADC/adc_u_faddeev.py), and
- the restricted spin-free, matrix-free DF route (ADC/adc_r_faddeev.py) that
  production runs use.

What pins what:

1. ADC(3) is the first-order limit. With the pair channels replaced by the
   first-order doubles and no phonon corrections, both builds must give the
   ADC(3) supermatrix exactly; this gates the couplings U and every
   index convention in them.
2. The spectator embedding. Embedding each pair's TDA Hamiltonian P with its
   spectator energy and subtracting 2K must rebuild K + C: this gates the
   three-particle embedding and its antisymmetrizer signs, through which dH
   and dN enter.
3. The two pair routes. From the phonons (X^{-T} Omega X^{-1}, (X X^T)^{-1})
   and from the Riccati amplitudes alone must agree, for the spin-orbital
   build and for the restricted one, whose Riccati route is the DF-streamed
   ladder amplitude in the alpha-beta pair form.
4. Restricted against spin-orbital. The CSF isometry W, written out in closed
   form here, must reproduce the restricted ADC(3) supermatrix from the
   spin-orbital one (so W really is the restricted route's basis), and then
   the restricted Faddeev supermatrix from the spin-orbital one.
5. The matrix-free operator against the dense builder, and a Davidson root
   through the public ADCSolver API against the dense spectrum.
6. pair_route='ccd' (one CCD T2 in every channel): pyscf's DF-CCD against an
   independent spin-orbital CCD; the singlet/triplet recoupling of t2 against
   the first-order channels; restricted against spin-orbital through W; and
   v -> lam v, under which CCD-Faddeev-ADC(3) - ADC(3) must go as lam^4.
7. eh_kernel='drpa' (both eh channels by direct RPA): the triplet equals the
   non-interacting 'off' pair, the phonon and Riccati routes agree, the
   result is exact through second order only (Faddeev - ADC(3) ~ lam^3, the
   exchange part of the first-order vertex being dropped), and it exists on
   the triplet-unstable stretched H2 where TDHF is refused.

Run: python tests/test_faddeev_adc3.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf
from pyscf.cc.addons import spatial2spin

from src.Base.pyscf_interface import DFIntegrals, get_antisymmetrized_spin_eri
from src.SingleReference.ADC import (ADCSolver, ADCSolverRestricted,
                                     ADCSolverUnrestricted)
from src.SingleReference.ADC import adc_r_faddeev as FR
from src.SingleReference.ADC import adc_u_faddeev as FU
from src.SingleReference.ADC import cc_amplitudes as CA
from src.SingleReference.CC import amplitudes as CCA
from src.SingleReference.ADC.adc_u_dense_full import C_2h1p_block, C_2p1h_block
from src.SingleReference.LinearResponse.pp_rpa import build_pprpa_matrices


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build(basis='6-31g'):
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    B_aa = DFIntegrals.from_scf(mol, mf).B_aa
    eri = np.einsum('Qpq,Qrs->pqrs', B_aa, B_aa, optimize=True)
    nocc = mol.nelectron // 2
    sr = ADCSolverRestricted.from_arrays(mf.mo_energy, B_aa=B_aa, nocc=nocc)
    so = ADCSolverUnrestricted.from_arrays(np.repeat(mf.mo_energy, 2),
                                           get_antisymmetrized_spin_eri(eri))
    return mol, mf, sr, so


def csf_isometry(sr, so, nocc):
    """W (spin-orbital configurations x restricted CSFs), from the class
    definitions in adc_r_faddeev's docstring (|i s; a s', b s''> =
    a+_{a s'} a+_{b s''} a_{i s}|HF>, 2h1p mirrored):

        I'   = |i b; a a, a b>
        II'  = sqrt(2/3)|i a; a a, b a> + sqrt(1/6)(|i b; a a, b b> + |i b; a b, b a>)
        III' = (|i b; a a, b b> - |i b; a b, b a>) / sqrt(2)
    """
    n = sr.norb
    nv = n - nocc
    d, ds = sr.dimensions(nocc), so.dimensions(2 * nocc)
    i1, j1, a1 = so._configs_2h1p(2 * nocc)
    i2, a2, b2 = so._configs_2p1h(2 * nocc)
    idx1 = {c: k for k, c in enumerate(zip(i1, j1, a1))}
    idx2 = {c: k for k, c in enumerate(zip(i2, a2, b2))}
    o1, o2 = 2 * n, 2 * n + ds['n2h1p']

    def c1(si, i, sj, j, sa, a):
        I, J, A = 2 * i + si, 2 * j + sj, 2 * (nocc + a) + sa
        return (o1 + idx1[(J, I, A)], -1.0) if I > J else (o1 + idx1[(I, J, A)], 1.0)

    def c2(si, i, sa, a, sb, b):
        I, A, B = 2 * i + si, 2 * (nocc + a) + sa, 2 * (nocc + b) + sb
        return (o2 + idx2[(I, B, A)], -1.0) if A > B else (o2 + idx2[(I, A, B)], 1.0)

    W = np.zeros((ds['nH'], d['nH']))
    W[2 * np.arange(n), np.arange(n)] = 1.0
    s6, s2, s23 = np.sqrt(1 / 6), np.sqrt(0.5), np.sqrt(2 / 3)
    col = [n]

    def put(terms):
        for (r, sg), c in terms:
            W[r, col[0]] += sg * c
        col[0] += 1

    iu, ju = np.triu_indices(nocc, 1)
    au, bu = np.triu_indices(nv, 1)
    for i in range(nocc):
        for a in range(nv):
            put([(c1(0, i, 1, i, 1, a), 1.0)])
    for i, j in zip(iu, ju):
        for a in range(nv):
            put([(c1(0, i, 0, j, 0, a), s23), (c1(0, i, 1, j, 1, a), s6),
                 (c1(1, i, 0, j, 1, a), s6)])
    for i, j in zip(iu, ju):
        for a in range(nv):
            put([(c1(0, i, 1, j, 1, a), s2), (c1(1, i, 0, j, 1, a), -s2)])
    for i in range(nocc):
        for a in range(nv):
            put([(c2(1, i, 0, a, 1, a), 1.0)])
    for i in range(nocc):
        for a, b in zip(au, bu):
            put([(c2(0, i, 0, a, 0, b), s23), (c2(1, i, 0, a, 1, b), s6),
                 (c2(1, i, 1, a, 0, b), s6)])
    for i in range(nocc):
        for a, b in zip(au, bu):
            put([(c2(1, i, 0, a, 1, b), s2), (c2(1, i, 1, a, 0, b), -s2)])
    return W


def check_spin_orbital_reference(so, nocc):
    eps, g = so.eps, so.g
    ok = True
    H3 = so.build_supermatrix(nocc)
    ch1 = FU.first_order_channels(eps, g, nocc)
    Hf, S = FU.build_pencil(so, nocc, channels=ch1)
    d = np.abs(Hf - H3).max()
    ok &= check(d < 1e-12 and np.abs(S - np.eye(len(S))).max() == 0,
                'first-order channels, no phonons == ADC(3) supermatrix', f'{d:.1e}')

    A, B = FU.build_tdhf_matrices(eps, g, nocc)
    App, Bpp, Cpp = build_pprpa_matrices(eps, g, nocc)
    chP = {'eh': {'T': ch1['eh']['T'], 'dH': A, 'dN': np.eye(len(A))},
           'pp': {'T': ch1['pp']['T'], 'dH': App, 'dN': np.eye(len(App))},
           'hh': {'T': ch1['hh']['T'], 'dH': -Cpp, 'dN': np.eye(len(Cpp))}}
    dHI, _, dHII, _ = FU.correction_blocks(eps, nocc, chP)
    dd = so.dimensions(nocc)
    i2, j2, a2 = so._configs_2h1p(nocc)
    i3, a3, b3 = so._configs_2p1h(nocc)
    KI = np.diag(eps[i2] + eps[j2] - eps[a2])
    KII = np.diag(eps[a3] + eps[b3] - eps[i3])
    CI = C_2h1p_block(so, nocc, np.arange(dd['n2h1p']))
    CII = C_2p1h_block(so, nocc, np.arange(dd['n2p1h']))
    d = max(np.abs(dHI - 2 * KI - (KI + CI)).max(),
            np.abs(dHII - 2 * KII - (KII + CII)).max())
    ok &= check(d < 1e-12, 'sum over pairs of (P + s) - 2K == K + C, both blocks',
                f'{d:.1e}')

    Hp = FU.build_supermatrix(so, nocc, route='phonon')
    Hr = FU.build_supermatrix(so, nocc, route='riccati')
    d = np.abs(Hp - Hr).max()
    ok &= check(d < 1e-8, 'phonon route == Riccati route', f'{d:.1e}')
    return ok, Hp


def check_restricted_against_reference(sr, so, nocc, W, H_so):
    ok = True
    mod_H3 = sr.build_supermatrix(nocc)
    d = np.abs(W.T @ so.build_supermatrix(2 * nocc) @ W - mod_H3).max()
    ok &= check(d < 1e-12, 'W^T H_ADC(3),so W == restricted ADC(3): W is the '
                'restricted CSF basis', f'{d:.1e}')
    for route in ('phonon', 'riccati'):
        Hr = FR.build_supermatrix(sr, nocc, route=route)
        d = np.abs(W.T @ H_so @ W - Hr).max()
        ok &= check(d < 1e-8, f'restricted Faddeev ({route}) == W^T H_so W',
                    f'{d:.1e}')
    H1 = FR.build_supermatrix(sr, nocc, channels=FR.first_order_channels(
        sr.eps, nocc, B_aa=sr.B_aa))
    d = np.abs(H1 - mod_H3).max()
    ok &= check(d < 1e-12, 'restricted first-order limit == restricted ADC(3)',
                f'{d:.1e}')
    return ok


def check_eh_treatments(sr, so, nocc, W):
    """The eh_triplet variants. 'off' (a non-interacting pair: T = 0, dN = 0,
    dH = D - A) is checked in BOTH spin channels against the spin-orbital
    reference with its eh channel switched off the same way; 'first_order' is
    the ADC(3) triplet channel exactly."""
    ok = True
    eps, B_aa = sr.eps, sr.B_aa
    ch_r = FR.first_order_channels(eps, nocc, B_aa=B_aa)
    for spin in ('singlet', 'triplet'):
        ch_r['eh'][spin] = FR._eh_channel(eps, nocc, spin, B_aa, None, 'riccati', 'off')
    ch_s = FU.first_order_channels(so.eps, so.g, 2 * nocc)
    A, _ = FU.build_tdhf_matrices(so.eps, so.g, 2 * nocc)
    n = A.shape[0]
    no2 = 2 * nocc
    D = (so.eps[None, no2:] - so.eps[:no2, None]).ravel()
    ch_s['eh'] = {'T': np.zeros((n, n)), 'dN': np.zeros((n, n)),
                  'dH': np.diag(D) - A}
    Hr = FR.build_supermatrix(sr, nocc, channels=ch_r)
    Hs = FU.build_supermatrix(so, no2, channels=ch_s)
    d = np.abs(W.T @ Hs @ W - Hr).max()
    ok &= check(d < 1e-12, "eh pair 'off' in both spin channels == spin-orbital "
                "reference with the eh channel off", f'{d:.1e}')
    t1 = FR._eh_channel(eps, nocc, 'triplet', B_aa, None, 'riccati', 'first_order')
    d = np.abs(t1['T'] - FR.first_order_channels(eps, nocc, B_aa=B_aa)['eh']['triplet']['T']).max()
    ok &= check(d < 1e-14 and not t1['dH'].any(), "eh_triplet='first_order' is the "
                "ADC(3) triplet channel", f'{d:.1e}')
    return ok


def check_matrix_free(sr, nocc):
    ok = True
    for route in ('phonon', 'riccati'):
        ch = FR.pair_channels(sr.eps, nocc, B_aa=sr.B_aa, route=route)
        H = FR.build_supermatrix(sr, nocc, channels=ch)
        aop, _, d = FR.build_operator(sr, nocc, channels=ch)
        X = np.random.default_rng(7).standard_normal((d['nH'], 4))
        dev = max(np.abs(aop(x) - H @ x).max() for x in X.T)
        ok &= check(dev < 1e-9, f'aop == dense builder on random vectors ({route})',
                    f'{dev:.1e}')
    return ok


def ccd_spin_orbital(eps, g, nocc, tol=1e-13, max_iter=500):
    """Spin-orbital CCD from the generated CC doubles residual (CC/amplitudes,
    T1 = T3 = 0), Jacobi steps, independent of pyscf. Returns (t[i,j,a,b],
    E_corr)."""
    nso = len(eps)
    o, v = slice(0, nocc), slice(nocc, nso)
    nv = nso - nocc
    f = np.diag(eps)
    eo, ev = eps[o], eps[v]
    e_abij = 1.0 / (eo[None, None, :, None] + eo[None, None, None, :]
                    - ev[:, None, None, None] - ev[None, :, None, None])
    t1 = np.zeros((nv, nocc))
    t3 = np.zeros((nv, nv, nv, nocc, nocc, nocc))
    t2 = g[v, v, o, o] * e_abij
    for _ in range(max_iter):
        step = CCA.doubles_residual(t1, t2, t3, f, g, o, v) * e_abij
        t2 = t2 + step
        if np.abs(step).max() < tol:
            break
    else:
        raise RuntimeError('spin-orbital CCD not converged')
    e_corr = 0.25 * np.einsum('ijab,abij->', g[o, o, v, v], t2)
    return t2.transpose(2, 3, 0, 1), e_corr


def homo_root(H, homo):
    """(IP, weight) of the ionization root with the largest HOMO weight."""
    w, X = np.linalg.eigh(H)
    wt = X[homo] ** 2 * (w < 0)
    k = np.argmax(wt)
    return -w[k], wt[k]


def check_ccd_route(mf, sr, so, nocc, W, scaling=False):
    ok = True
    eps = mf.mo_energy
    t2, e_cc = CA.ccd_t2_restricted(mf, return_energy=True, conv_tol=1e-12,
                                    conv_tol_normt=1e-10)
    t_so = spatial2spin(t2)
    t_ref, e_ref = ccd_spin_orbital(so.eps, so.g, 2 * nocc)
    d = max(abs(e_ref - e_cc), np.abs(t_ref - t_so).max())
    ok &= check(d < 1e-7, 'pyscf DF-CCD == independent spin-orbital CCD',
                f'{d:.1e}')
    # the recoupling of t2, gated by the first-order amplitude
    o, v = slice(0, nocc), slice(nocc, None)
    ov = np.einsum('Qia,Qjb->iajb', sr.B_aa[:, o, v], sr.B_aa[:, o, v])
    t1 = ov.transpose(0, 2, 1, 3) / (
        eps[o, None, None, None] + eps[None, o, None, None]
        - eps[None, None, v, None] - eps[None, None, None, v])
    ch = FR.ccd_channels(eps, nocc, t1, sr.B_aa)
    ch1 = FR.first_order_channels(eps, nocc, sr.B_aa)
    d = max([np.abs(ch['eh'][sp]['T'] - ch1['eh'][sp]['T']).max()
             for sp in FR.SPINS]
            + [np.abs(ch['pp']['t'] - FR._alpha_beta_ladder(
                ch1['pp'], nocc, len(eps) - nocc)).max()])
    ok &= check(d < 1e-12, 'first-order t2 recouples to the first-order channels',
                f'{d:.1e}')
    chu = FU.ccd_channels(so.eps, so.g, 2 * nocc, spatial2spin(t1))
    chu1 = FU.first_order_channels(so.eps, so.g, 2 * nocc)
    d = max(np.abs(chu[k]['T'] - chu1[k]['T']).max() for k in ('eh', 'pp', 'hh'))
    ok &= check(d < 1e-12, 'spin-orbital: the same for the reference', f'{d:.1e}')
    # restricted against spin-orbital, dense and matrix-free
    chr_ = FR.ccd_channels(eps, nocc, t2, sr.B_aa)
    Hr = FR.build_supermatrix(sr, nocc, channels=chr_)
    Hs = FU.build_supermatrix(so, 2 * nocc, channels=FU.pair_channels(
        so.eps, so.g, 2 * nocc, route='ccd', t=t_so))
    d = np.abs(W.T @ Hs @ W - Hr).max()
    ok &= check(d < 1e-10, 'restricted CCD-Faddeev-ADC(3) == W^T (spin-orbital) W',
                f'{d:.1e}')
    aop, _, _ = FR.build_operator(sr, nocc, channels=chr_)
    x = np.random.default_rng(3).standard_normal(len(Hr))
    d = np.abs(aop(x) - Hr @ x).max()
    ok &= check(d < 1e-9, 'matrix-free CCD-Faddeev-ADC(3) == dense', f'{d:.1e}')
    if not scaling:
        return ok
    # v -> lam v at fixed orbital energies: CCD-Faddeev-ADC(3) - ADC(3) ~ lam^4
    diffs = []
    for lam in (0.1, 0.05):
        g = lam * so.g
        s = ADCSolverUnrestricted.from_arrays(so.eps, g)
        t, _ = ccd_spin_orbital(so.eps, g, 2 * nocc, tol=1e-15)
        Hf = FU.build_supermatrix(s, 2 * nocc,
                                  channels=FU.ccd_channels(so.eps, g, 2 * nocc, t))
        H3 = s.build_supermatrix(2 * nocc)
        diffs.append(abs(homo_root(Hf, 2 * nocc - 1)[0]
                         - homo_root(H3, 2 * nocc - 1)[0]))
    slope = np.log(diffs[0] / diffs[1]) / np.log(2.0)
    ok &= check(abs(slope - 4.0) < 0.15, 'CCD-Faddeev-ADC(3) - ADC(3) ~ lam^4',
                f'slope {slope:.2f}')
    return ok


def check_solver_api(mf, nocc):
    ok = True
    e_mf, e_d = [], None
    for route in ('phonon', 'riccati'):
        s = ADCSolver(mf, level='faddeev_adc3', df=True, pair_route=route)
        e, Z = s.solve(nroots=1, homo_index=nocc - 1, conv_tol=1e-7)
        ok &= check(bool(np.all(s.last_result['converged'])),
                    f'Davidson converged ({route})')
        e_mf.append(e[0])
    s = ADCSolver(mf, level='faddeev_adc3', df=True, pair_route='ccd')
    Hc = s.build_supermatrix(nocc)
    ch = FR.ccd_channels(s.eps, nocc, CA.ccd_t2_restricted(
        mf, conv_tol=1e-10, conv_tol_normt=1e-8), s.B_aa)
    d = np.abs(Hc - FR.build_supermatrix(s, nocc, channels=ch)).max()
    ok &= check(d < 1e-8, "pair_route='ccd' runs CCD on the mean field",
                f'{d:.1e}')
    s = ADCSolver(mf, level='faddeev_adc3', df=True, matrix_free=False)
    ed, Zd = s.solve()
    k = np.argmin(np.abs(ed - e_mf[0]))
    d = max(abs(e_mf[0] - ed[k]), abs(e_mf[1] - ed[k]))
    ok &= check(d < 1e-8, 'Davidson HOMO root == dense spectrum',
                f'IP {-ed[k] * 27.211386245988:.6f} eV, Z {Zd[k]:.4f}, dev {d:.1e}')
    for kw, label in (({'level': 'faddeev_adc3', 'en_dress': {'hh': True}},
                       'faddeev_adc3 refuses en_dress'),
                      ({'level': 'adc3', 'pair_route': 'riccati'},
                       'pair_route on adc3 is refused'),
                      ({'level': 'adc3', 'eh_triplet': 'off'},
                       'eh_triplet on adc3 is refused'),
                      ({'level': 'faddeev_adc3', 'eh_triplet': 'none'},
                       'an unknown eh_triplet is refused'),
                      ({'level': 'faddeev_adc3', 'pair_route': 'eigen'},
                       'an unknown pair_route is refused')):
        try:
            ADCSolver(mf, df=True, **kw)
            ok &= check(False, label)
        except ValueError:
            ok &= check(True, label)
    return ok


def check_drpa(mf, sr, nocc, scaling=False):
    ok = True
    eps, B = mf.mo_energy, sr.B_aa
    for route in ('riccati', 'phonon'):
        a = FR._eh_channel(eps, nocc, 'triplet', B, None, route, 'drpa')
        b = FR._eh_channel(eps, nocc, 'triplet', B, None, route, 'off')
        d = max(np.abs(a[k] - b[k]).max() for k in ('T', 'dH', 'dN'))
        ok &= check(d < 1e-12, f"dRPA triplet == non-interacting 'off' pair ({route})",
                    f'{d:.1e}')
    H = {r: FR.build_supermatrix(sr, nocc, channels=FR.pair_channels(
        eps, nocc, B, route=r, eh_kernel='drpa')) for r in ('phonon', 'riccati')}
    d = np.abs(H['phonon'] - H['riccati']).max()
    ok &= check(d < 1e-9, 'dRPA: phonon route == Riccati route', f'{d:.1e}')
    if scaling:
        diffs = []
        for lam in (0.1, 0.05):
            Bl = np.sqrt(lam) * B
            s = ADCSolverRestricted.from_arrays(eps, None, B_aa=Bl, nocc=nocc)
            Hf = FR.build_supermatrix(s, nocc, channels=FR.pair_channels(
                eps, nocc, Bl, route='riccati', eh_kernel='drpa'))
            diffs.append(abs(homo_root(Hf, nocc - 1)[0]
                             - homo_root(s.build_supermatrix(nocc), nocc - 1)[0]))
        slope = np.log(diffs[0] / diffs[1]) / np.log(2.0)
        ok &= check(abs(slope - 3.0) < 0.2,
                    'Faddeev-dRPA - ADC(3) ~ lam^3 (exact through second order)',
                    f'slope {slope:.2f}')
    return ok


def check_unstable_reference_is_refused():
    """Stretched H2 (2.5 A) is RHF triplet-unstable: the triplet TDHF channel
    has an imaginary phonon, the Faddeev pair channel does not exist, and
    both routes must say so instead of clipping omega^2."""
    mol = gto.M(atom='H 0 0 0; H 0 0 2.5', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    B_aa = DFIntegrals.from_scf(mol, mf).B_aa
    ok = True
    for route in ('phonon', 'riccati'):
        try:
            FR.pair_channels(mf.mo_energy, 1, B_aa=B_aa, route=route)
            ok &= check(False, f'unstable triplet channel refused ({route})')
        except ValueError as exc:
            ok &= check('triplet' in str(exc) and 'unstable' in str(exc),
                        f'unstable triplet channel refused ({route})',
                        str(exc).split(';')[0])
    ch = FR.pair_channels(mf.mo_energy, 1, B_aa=B_aa, route='riccati', eh_kernel='drpa')
    s = ADCSolverRestricted.from_arrays(mf.mo_energy, None, B_aa=B_aa, nocc=1)
    ip = homo_root(FR.build_supermatrix(s, 1, channels=ch), 0)[0] * 27.211386245988
    ok &= check(np.isfinite(ip) and ip > 0, 'dRPA channels exist on the unstable reference',
                f'HOMO IP {ip:.3f} eV')
    return ok


def run():
    warnings.simplefilter('ignore')
    all_ok = True
    for basis in ('sto-3g', '6-31g'):
        mol, mf, sr, so = build(basis)
        nocc = mol.nelectron // 2
        print(f'\n== H2O/{basis}')
        print('-- the spin-orbital reference')
        ok, H_so = check_spin_orbital_reference(so, 2 * nocc)
        all_ok &= ok
        print('-- restricted spin-free route against it')
        W = csf_isometry(sr, so, nocc)
        all_ok &= check(np.abs(W.T @ W - np.eye(W.shape[1])).max() < 1e-14,
                        'W orthonormal')
        all_ok &= check_restricted_against_reference(sr, so, nocc, W, H_so)
        print('-- triplet-channel treatments')
        all_ok &= check_eh_treatments(sr, so, nocc, W)
        print('-- matrix-free operator')
        all_ok &= check_matrix_free(sr, nocc)
        print('-- the CCD pair route')
        all_ok &= check_ccd_route(mf, sr, so, nocc, W,
                                  scaling=(basis == 'sto-3g'))
        print('-- the direct-RPA eh kernel')
        all_ok &= check_drpa(mf, sr, nocc, scaling=(basis == 'sto-3g'))
    print('\n-- public API (H2O/6-31G)')
    all_ok &= check_solver_api(mf, nocc)
    print('\n-- an unstable reference')
    all_ok &= check_unstable_reference_is_refused()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return all_ok


def test_faddeev_adc3_against_reference():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
