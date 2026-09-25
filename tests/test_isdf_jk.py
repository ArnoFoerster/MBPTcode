"""The ISDF J/K builder: structure first, then accuracy.

Most of what is checked here is EXACT, not approximate, because the ISDF J and
K are two contractions of one and the same approximate four-index tensor

    g_{mu nu la si} = sum_{PQ} X_{P mu} X_{P nu} Z_{PQ} X_{Q la} X_{Q si}

and everything that follows from that tensor existing has to hold to machine
precision or the implementation is wrong:

  * the blocked kernels reproduce a dense contraction of g;
  * blocking, dense/factored Z, and the symmetric/general K paths agree;
  * K is positive semidefinite for a positive semidefinite density, since Z
    and X Dm X^T are both PSD and Schur's theorem applies to their Hadamard
    product -- so the ISDF exchange energy cannot come out with the wrong sign;
  * K_SR(omega) + K_LR(omega) = K_bare EXACTLY, because Z is LINEAR in the
    two-centre metric and V_SR + V_LR = V. Range separation is not an extra
    approximation on top of the factorization, and a refitted M_omega would
    forfeit exactly this.

Only the last two blocks are tolerances rather than identities.

Run: python tests/test_isdf_jk.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, dft, gto, scf

from src.Base.isdf_jk import (ISDFJK, isdf_grid, isdf_jk, range_coulomb,
                              separable_factors_from_jk)
from src.Base.separable_ri import PUBLISHED_COUNTS
from src.SingleReference.GW.space_time import separable_factors
from src.Base.constants import HARTREE_TO_EV

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def main():
    ok = True
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    nao = mol.nao_nr()
    w = ISDFJK(mol, auxbasis='cc-pvdz-ri', j_route='isdf', check_tol=None)
    w.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    print('\n=== one tensor, two contractions (water / cc-pVDZ, M=%d) ===' % w.nk)
    V = w.auxmol.intor('int2c2e', aosym='s1')
    Z = w.M.T @ V @ w.M
    g = np.einsum('Pi,Pj,PQ,Qk,Ql->ijkl', w.X, w.X, Z, w.X, w.X, optimize=True)
    vj_ref = np.einsum('ijkl,lk->ij', g, dm)
    vk_ref = np.einsum('ijkl,jk->il', g, dm)
    vj, vk = w.get_jk(dm)
    ok &= check(np.abs(vj - vj_ref).max() < 1e-11, 'J == contraction of g',
                '%.1e' % np.abs(vj - vj_ref).max())
    ok &= check(np.abs(vk - vk_ref).max() < 1e-11, 'K == contraction of g',
                '%.1e' % np.abs(vk - vk_ref).max())

    print('\n=== the paths that must not change the answer ===')
    w.block = 7
    vj_b, vk_b = w.get_jk(dm)
    w.block = None
    ok &= check(max(np.abs(vj_b - vj).max(), np.abs(vk_b - vk).max()) < 1e-11,
                'block = 7 == one shot')
    vj_g, vk_g = w.get_jk(dm, hermi=0)          # general (non-symmetric) kernel
    ok &= check(np.abs(vk_g - vk).max() < 1e-11, 'triangular K == general K',
                '%.1e' % np.abs(vk_g - vk).max())
    wf = ISDFJK(mol, auxbasis='cc-pvdz-ri', z_mode='factored', j_route='isdf',
                check_tol=None)
    wf.coords, wf.X, wf.M, wf.auxmol, wf._built = w.coords, w.X, w.M, w.auxmol, True
    vj_f, vk_f = wf.get_jk(dm)
    ok &= check(max(np.abs(vj_f - vj).max(), np.abs(vk_f - vk).max()) < 1e-9,
                'factored Z == dense Z', '%.1e' % np.abs(vk_f - vk).max())

    # a non-symmetric density has no reason to be handled by the hermi=1 path,
    # so the kernel that claims to be general is asked to prove it
    rng = np.random.default_rng(0)
    dm_ns = rng.standard_normal((nao, nao))
    vj_n, vk_n = w.get_jk(dm_ns, hermi=0)
    ok &= check(np.abs(vk_n - np.einsum('ijkl,jk->il', g, dm_ns)).max() < 1e-10,
                'general K on a non-symmetric density')

    print('\n=== structure: K is positive semidefinite ===')
    # Z PSD and X Dm X^T PSD => their Hadamard product is PSD (Schur), so
    # K = X^T W X is PSD and Tr(K Dm) >= 0.
    ev = np.linalg.eigvalsh(0.5 * (vk + vk.T))
    ok &= check(ev.min() > -1e-10 * max(ev.max(), 1.0), 'eig(K) >= 0',
                'min %.2e, max %.2e' % (ev.min(), ev.max()))

    print('\n=== range separation is exact in the metric ===')
    omega = 0.3
    vj_lr, vk_lr = w.get_jk(dm, omega=omega)
    vj_sr, vk_sr = w.get_jk(dm, omega=-omega)
    d = max(np.abs(vk_lr + vk_sr - vk).max(), np.abs(vj_lr + vj_sr - vj).max())
    ok &= check(d < 1e-10, 'K_LR + K_SR == K_bare to machine precision', '%.1e' % d)
    with mol.with_range_coulomb(omega):
        eri_lr = mol.intor('int2e', aosym='s1')
    vk_lr_x = np.einsum('ijkl,jk->il', eri_lr, dm)
    vk_x = scf.hf.get_jk(mol, dm)[1]
    e_lr = np.abs(vk_lr - vk_lr_x).max()
    e_0 = np.abs(vk - vk_x).max()
    ok &= check(e_lr < e_0, 'attenuated K is no worse than bare K',
                'LR %.2e vs bare %.2e' % (e_lr, e_0))

    print('\n=== the two J routes leave K alone ===')
    wd = ISDFJK(mol, auxbasis='cc-pvdz-ri', check_tol=None)   # default df-direct
    wd.coords, wd.X, wd.M, wd.auxmol, wd._built = w.coords, w.X, w.M, w.auxmol, True
    vj_d, vk_d = wd.get_jk(dm)
    ok &= check(wd.j_route == 'df-direct', "default j_route is 'df-direct'")
    ok &= check(np.abs(vk_d - vk).max() < 1e-11, 'K is identical either way',
                '%.1e' % np.abs(vk_d - vk).max())
    vj_x_ref = scf.hf.get_jk(mol, dm)[0]
    ok &= check(np.abs(vj_d - vj_x_ref).max() < np.abs(vj - vj_x_ref).max(),
                'df-direct J beats ISDF J',
                'DF %.2e vs ISDF %.2e' % (np.abs(vj_d - vj_x_ref).max(),
                                          np.abs(vj - vj_x_ref).max()))

    print('\n=== the grid guard ===')
    # The probe densities are deliberately not the SCF's: a grid can look fine
    # at a physical density and still collapse the SCF. See the module
    # docstring of src/Base/isdf_jk.py.
    rel = wd.check()
    ok &= check(rel < 1e-3, 'water/cc-pVDZ grid passes the probe', '%.2e' % rel)
    ok &= check(set(wd.probe_densities()) >= {'minao', '1e'},
                'the probe set includes the diffuse 1e guess')

    print('\n=== the cderi path is closed off, not silently taken ===')
    try:
        next(w.loop())
        ok &= check(False, 'loop() raises')
    except NotImplementedError:
        ok &= check(True, 'loop() raises NotImplementedError')

    print('\n=== end to end: LRC-wPBEh SCF ===')
    ref = dft.RKS(mol, xc='lrc-wpbeh')
    ref.grids.level = 4
    ref.conv_tol = 1e-10
    ref.verbose = 0
    e_ref = ref.kernel()
    tst = isdf_jk(dft.RKS(mol, xc='lrc-wpbeh'), auxbasis='cc-pvdz-ri')  # df-direct J
    tst.grids.level = 4
    tst.conv_tol = 1e-10
    tst.verbose = 0
    tst.with_df.coords = w.coords
    e_isdf = tst.kernel()
    de = (e_isdf - e_ref) * HARTREE_TO_EV * 1e3
    ok &= check(tst.converged and abs(de) < 200.0,
                'LRC-wPBEh converges within 200 meV of exact-ERI RKS',
                'dE = %+.1f meV' % de)
    nocc = mol.nelectron // 2
    dh = (tst.mo_energy[nocc - 1] - ref.mo_energy[nocc - 1]) * HARTREE_TO_EV * 1e3
    ok &= check(abs(dh) < 50.0, 'HOMO within 50 meV', 'd = %+.1f meV' % dh)

    # A GW/BSE on top of an ISDF SCF reuses the SCF's fit through this. It has
    # to hand back the SAME factors, in the SAME gauge -- mixing two fits gives
    # a wrong answer that still looks converged, so only an equality catches it.
    print('\n=== separable_factors_from_jk: the SCF fit, reused ===')
    sf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri')
    sf.conv_tol = 1e-10
    sf.verbose = 0
    sf.kernel()
    Xmo_jk, D_jk, Xao_jk, co_jk = separable_factors_from_jk(sf)
    Xmo_st, D_st, Xao_st, co_st = separable_factors(sf, mol, auxbasis='cc-pvdz-ri')
    ok &= check(np.abs(co_jk - co_st).max() < 1e-12, 'same interpolation grid',
                '%.1e' % np.abs(co_jk - co_st).max())
    ok &= check(np.abs(Xao_jk - Xao_st).max() < 1e-12, 'same AO collocation',
                '%.1e' % np.abs(Xao_jk - Xao_st).max())
    ok &= check(np.abs(Xmo_jk - Xmo_st).max() < 1e-12, 'same MO collocation',
                '%.1e' % np.abs(Xmo_jk - Xmo_st).max())
    # D itself, not Z = D D^T: rotating the aux index leaves Z alone, so a
    # check on Z (or on any ERI) sees nothing while the gauge has changed.
    dD = np.abs(D_jk - D_st).max()
    dZ = np.abs(D_jk @ D_jk.T - D_st @ D_st.T).max()
    ok &= check(dD < 1e-12, 'same auxiliary GAUGE (D, not merely Z)',
                '|dD| = %.1e, |dZ| = %.1e' % (dD, dZ))

    try:
        separable_factors_from_jk(scf.RHF(mol).density_fit())
        ok &= check(False, 'refuses a mean field that did not fit an ISDF')
    except TypeError:
        ok &= check(True, 'refuses a mean field that did not fit an ISDF')

    # A requested grid SIZE must reach the grid, and so must the DEFAULT one.
    # cc-pVTZ used to be special: the published Duchemin-Blase tables (H/C/N/O,
    # one size each) were substituted for a caller who named no `counts`, so
    # this basis alone answered with ~320 points per atom while every other
    # basis answered with the 148 of DEFAULT_COUNTS. Those grids are ordinary
    # rows in the radii table now, at their own counts, so reaching them means
    # asking for them -- which is what the third check below pins. H2 keeps this
    # cheap: hydrogen is in the published set.
    print('\n=== grid size at cc-pVTZ: asked for, defaulted, published ===')
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvtz', verbose=0)
    m_default = len(isdf_grid(h2))
    m_small = len(isdf_grid(h2, counts={'A1': 5, 'A2': 3, 'A3': 2, 'B1': 0}))
    m_large = len(isdf_grid(h2, counts={'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}))
    # The published hydrogen grid is (5,5,4,2) with the cusp sampled: 167 points.
    m_pub = len(isdf_grid(h2, counts=dict(zip(('A1', 'A2', 'A3', 'B1'),
                                              PUBLISHED_COUNTS['H']))))
    ok &= check(m_small == 2 * 78 and m_large == 2 * 148,
                'explicit counts reach the grid at cc-pVTZ',
                f'{m_small} and {m_large} points for 78 and 148 per atom')
    ok &= check(m_default == 2 * 148,
                'omitting counts takes DEFAULT_COUNTS, not a per-basis special '
                'case', f'{m_default} points = 2 x 148')
    ok &= check(m_pub == 2 * 167,
                'the published grid is reached by asking for its counts',
                f'{m_pub} points')

    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(main())
