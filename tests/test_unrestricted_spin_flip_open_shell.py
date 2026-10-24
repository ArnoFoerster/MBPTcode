"""The unrestricted spin-flip channel on an open shell is the spin-orbital one.

OH radical/6-31G, UHF, exact ERIs, bare kernel. The spin-flip Casida problem
couples ab excitations (occ_a -> virt_b) with ba de-excitations and back, on
two pair grids of different size. Its (A, B), and the spin-conserving ones,
equal the blocks of the spin-orbital TDHF matrices; the spin-flip vertex equals
the exchange part of the spin-orbital electron-hole-pair vertex for any X, Y;
and PSD2 at fixed omega equals the spin-orbital electron-hole pair. The last
check runs in the Tamm-Dancoff approximation: the full spin-flip TDHF problem
of an open-shell UHF has an exact zero mode (a global spin rotation), where the
Casida omega^2 reduction is singular in both constructions.

    python tests/test_unrestricted_spin_flip_open_shell.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import ao2mo, gto, scf

from src.SingleReference.GW.self_energy import METHOD_REGISTRY
from src.SingleReference.GW import qp_energy
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

OH = 'O 0 0 0; H 0 0 0.97'


def check(ok, label, detail=''):
    msg = f"  [{'ok' if ok else 'FAIL'}] {label}"
    print(msg + (f'   ({detail})' if detail else ''))
    return bool(ok)


def radical():
    """UHF doublet, its solvers and its spin-orbital TDHF ingredients."""
    mol = gto.M(atom=OH, basis='6-31g', spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    na, nb = mf.nelec
    (Ca, Cb), (ea, eb) = mf.mo_coeff, mf.mo_energy
    nmo = Ca.shape[1]
    # spin orbitals: occupied alpha, occupied beta, virtual alpha, virtual beta
    C = np.hstack([Ca[:, :na], Cb[:, :nb], Ca[:, na:], Cb[:, nb:]])
    spin = np.array([0] * na + [1] * nb + [0] * (nmo - na) + [1] * (nmo - nb))
    es = np.concatenate([ea[:na], eb[:nb], ea[na:], eb[nb:]])
    nso, no = len(es), na + nb
    same = spin[:, None] == spin[None, :]
    chem = (ao2mo.full(mol, C, compact=False).reshape((nso,) * 4)
            * same[:, :, None, None] * same[None, None, :, :])
    g = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)   # <pq||rs>
    nv = nso - no
    o, v = slice(0, no), slice(no, None)
    A = g[v, o, o, v].transpose(2, 0, 1, 3).reshape(no * nv, no * nv).copy()
    A[np.diag_indices(no * nv)] += (es[None, v] - es[o, None]).ravel()
    B = g[v, v, o, o].transpose(2, 0, 3, 1).reshape(no * nv, no * nv).copy()

    def so(p, s):
        """Spin-orbital index of MO p of spin s (0 alpha, 1 beta)."""
        nocc = (na, nb)[s]
        if p < nocc:
            return p + s * na
        return no + p - nocc + s * (nmo - na)

    def grid(si, sa):
        return [so(i, si) * nv + so(a, sa) - no for i in range((na, nb)[si])
                for a in range((na, nb)[sa], nmo)]

    _, eri = qp_energy._two_electron_integrals(mol, mf, False, True)
    eps = tuple(mf.mo_energy)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='unrestricted')
    se = SelfEnergySolver(eps, eri_chemist=eri, spin_mode='unrestricted', eta=1e-12)
    return dict(mf=mf, na=na, nb=nb, nmo=nmo, es=es, g=g, A=A, B=B, no=no, so=so,
                lr=lr, se=se, sc=grid(0, 0) + grid(1, 1), sf=grid(0, 1) + grid(1, 0))


def eh_vertex(g, es, no, P, X, Y):
    """Spin-orbital eh-pair vertex of orbital P: (virtual r, occupied r) blocks."""
    nv = len(es) - no
    o, v = slice(0, no), slice(no, None)
    X3, Y3 = X.reshape(no, nv, -1), Y.reshape(no, nv, -1)
    Mv = (np.einsum('iab,ian->bn', g[P, o, v, v], X3)
          + np.einsum('jcb,jcn->bn', g[P, v, o, v].transpose(1, 0, 2), Y3))
    Mo = (np.einsum('iaj,ian->jn', g[P, v, o, o].transpose(1, 0, 2), X3)
          + np.einsum('kcj,kcn->jn', g[v, o, P, o].transpose(2, 0, 1), Y3))
    return Mv, Mo


def check_casida_blocks(r):
    ok = True
    nelec = r['mf'].nelec
    for label, (A, B), rows in (
            ('spin-conserving', r['lr'].build_casida_matrices(nelec, lBSE=True), r['sc']),
            ('spin-flip', r['lr'].build_spin_flip_casida_matrices(nelec, lBSE=True), r['sf'])):
        dA = np.abs(A - r['A'][np.ix_(rows, rows)]).max()
        dB = np.abs(B - r['B'][np.ix_(rows, rows)]).max()
        ok &= check(max(dA, dB) < 1e-12, f'{label} (A, B) = spin-orbital blocks',
                    f'{max(dA, dB):.1e} Ha')
    try:
        r['lr'].build_spin_flip_casida_matrices(nelec, channel='ab')
        refused = False
    except ValueError as err:
        refused = 'nocc_a == nocc_b' in str(err)
    ok &= check(refused, 'a single spin-flip channel is refused on an open shell')
    return ok


def check_spin_flip_vertex(r):
    ok = True
    rng = np.random.default_rng(7)
    X, Y = rng.standard_normal((2, len(r['sf']), 3))
    n_so = r['A'].shape[0]
    X_so, Y_so = np.zeros((n_so, 3)), np.zeros((n_so, 3))
    X_so[r['sf']], Y_so[r['sf']] = X, Y
    for spin_channel, s, t in (('alpha', 0, 1), ('beta', 1, 0)):
        nocc_t = (r['na'], r['nb'])[t]
        for p in ((r['na'], r['nb'])[s] - 1, (r['na'], r['nb'])[s]):
            chi = r['se'].get_chi_b_vertex_sf_full(r['mf'].nelec, X, Y,
                                                   spin_channel=spin_channel, p_state=p)
            Mv, Mo = eh_vertex(r['g'], r['es'], r['no'], r['so'](p, s), X_so, Y_so)
            ref = np.array([(Mo[r['so'](q, t)] if q < nocc_t else Mv[r['so'](q, t) - r['no']])
                            for q in range(r['nmo'])]).T
            d = np.abs(chi - ref).max()
            ok &= check(d < 1e-12, f'spin-flip vertex = eh-pair exchange, {spin_channel} p={p}',
                        f'{d:.1e}')
    return ok


def check_psd2_is_the_eh_pair(r):
    ok = True
    nelec = r['mf'].nelec
    om, X, Y = CasidaSolver(r['A'], r['B']).solve(tda=True)
    infos = {'PSD2': METHOD_REGISTRY['PSD2']}
    spectrum = qp_energy.casida_spectrum(r['lr'], nelec, 'TDHF', None, True, infos,
                                          ['PSD2'], True, False)
    es, no = r['es'], r['no']
    for spin_channel, s in (('alpha', 0), ('beta', 1)):
        eps = r['mf'].mo_energy[s]
        for p in (nelec[s] - 1, nelec[s]):
            Mv, Mo = eh_vertex(r['g'], es, no, r['so'](p, s), X, Y)
            amps = qp_energy._self_energy_amplitudes(
                r['se'], nelec, spectrum, infos, ['PSD2'], spin_channel, p, None, None,
                True, False)['PSD2']
            evaluator = r['se'].self_energy_evaluator(
                p, nelec, amps[0], amps[1], amps[2], eigenvalues_casida_t=amps[3],
                chiXYb_t=amps[4], spin_channel=spin_channel, vertex_mode='PSD2')
            for dw in (-0.35, 0.20):
                w = eps[p] + dw
                ref = (np.sum(Mv**2 / (w - es[no:][:, None] - om[None, :]))
                       + np.sum(Mo**2 / (w - es[:no][:, None] + om[None, :])))
                sigma = r['se'].calculate_self_energy(
                    p, w, nelec, amps[0], amps[1], amps[2], eigenvalues_casida_t=amps[3],
                    chiXYb_t=amps[4], spin_channel=spin_channel, vertex_mode='PSD2')
                ok &= check(abs(sigma - ref) < 1e-10,
                            f'PSD2 = eh pair (TDA), {spin_channel} p={p} w=eps{dw:+.2f}',
                            f'{sigma - ref:+.1e} Ha')
                ok &= check(abs(evaluator(w) - sigma) < 1e-10,
                            f'evaluator = scalar, {spin_channel} p={p} w=eps{dw:+.2f}',
                            f'{evaluator(w) - sigma:+.1e} Ha')
    return ok


def run():
    r = radical()
    ok = check_casida_blocks(r)
    ok &= check_spin_flip_vertex(r)
    ok &= check_psd2_is_the_eh_pair(r)
    return ok


def test_unrestricted_spin_flip_open_shell():
    assert run()


if __name__ == '__main__':
    passed = run()
    print('ALL PASSED' if passed else 'FAILURES DETECTED')
    sys.exit(0 if passed else 1)
