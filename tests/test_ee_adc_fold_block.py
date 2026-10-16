"""Block solver of the folded EE-ADC roots (ee_fold.solve_folded solver='block').

All asked roots share one Davidson subspace; every trial vector carries M v and
the folded doubles F(w_q) v at Chebyshev frequencies w_q spanning the roots,
and A_eff(w) between the nodes is their Lagrange interpolant. Each root solves
lambda_r(w) = w in the subspace model and is converged only by an exact
product at its own w. Checks:

  1. RIFoldADC2.fold_images: M U equals apply(U, None) bit for bit and each
     F[q] equals apply(U, w_q) - apply(U, None) to 1e-13 relative, singlets
     and triplets.
  2. the interpolation bound: for poles D at and above d_min the Lagrange
     interpolant of 1/(w - D) on _node_count's nodes is within the bound the
     node count is chosen from, over the whole window.
  3. method 'block' equals the dense branch to 1e-9 Ha on water's DF pieces
     (adc2 and gf2, singlets and triplets, 3 and 6 roots), every root
     converged by the block ('block' loop).
  4. a near-degenerate pair delocalized over 300 singles (roots 2.6e-4 Ha apart, a
     dense M on which the diagonal preconditioner is weak): both roots exact
     against the supermatrix to 1e-9 Ha, both by the block, in at most 30
     block iterations.
  5. roots whose window reaches the lowest doubles energy: a warning, and the
     per-root Newton gives the exact roots.
  6. over simulated MPI ranks (2 and 3) the block solve driven from rank 0
     returns the serial roots bit for bit on every rank.
  7. completeness without the count: a state last on M's diagonal that the
     fold makes the lowest root (n 800; alone, and inside a coupled block of
     50), which M's seeds miss, is found through the start's random vector,
     at nroots 1 and 3, every root exact to 1e-8 Ha and converged (a root
     whose window the block cannot interpolate goes to the per-root Newton).
  8. a block that runs out of iterations (EE_FOLD_BLOCK_MAX_ITER 2) hands its
     roots to the per-root Newton, which gives the dense roots to 1e-9 Ha.

Run: python tests/test_ee_adc_fold_block.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.ADC.eeADC import ee_fold, ee_r_sigma_df, ee_ri_fold
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.SingleReference.ADC.eeADC import ee_equations

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water_df():
    """(mf, eps, B, no): RHF water/cc-pVDZ with DF factors B[Q, p, q] (MO)."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    return mf, eps, DFIntegrals.from_scf(mol, mf).B_aa, mol.nelectron // 2


def synthetic_pieces(m, C, D):
    """Fold pieces with no = 1: M = m (symmetric) on both spin blocks, V = C
    (doubles x singles), D the doubles diagonal, the plain doubles norm; in
    the singlet channel the supermatrix is [[m, C^T], [C, diag(D)]]."""
    nv = m.shape[0]
    Ma = m.reshape(1, nv, 1, nv).copy()
    M = SB({'aaaa': Ma, 'bbbb': Ma.copy(), 'aabb': np.zeros_like(Ma),
            'bbaa': np.zeros_like(Ma)})

    def V(y1):
        return SB({s: C @ y1.get(s).ravel() for s in ('aa', 'bb')})

    def Vt(Y):
        return SB({s: (C.T @ Y.get(s)).reshape(1, nv) for s in ('aa', 'bb')})

    def dnorm2(Y):
        return float(sum(Y.get(s) @ Y.get(s) for s in ('aa', 'bb')))

    return {'M': M, 'V': V, 'Vt': Vt, 'D': np.asarray(D, float), 'dnorm2': dnorm2,
            'level': 'gw', 'no': 1, 'nv': nv, 'be': ee_equations.SPIN_BLOCKED}


def supermatrix_roots(m, C, D):
    """Eigenvalues of [[m, C^T], [C, diag(D)]], ascending."""
    ns = m.shape[0]
    H = np.zeros((ns + len(D), ns + len(D)))
    H[:ns, :ns], H[ns:, :ns], H[:ns, ns:] = m, C, C.T
    H[ns:, ns:] = np.diag(D)
    return np.linalg.eigvalsh(H)


def check_images(mf, eps, B, no):
    """Check 1."""
    ok = True
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    U = np.random.default_rng(3).standard_normal((F.no * F.nv, 2))
    nodes = [0.3, 0.55, 0.8]
    for spin in ('singlet', 'triplet'):
        op = ee_ri_fold.RIFoldADC2(F, spin)
        MU, Fq = op.fold_images(U, nodes)
        M0, _ = op.apply(U, None)
        err = max(float(np.abs(Fq[q] - (op.apply(U, [w, w])[0] - M0)).max()
                        / np.abs(Fq[q]).max()) for q, w in enumerate(nodes))
        ok &= check(np.array_equal(MU, M0) and err < 1e-13,
                    f'{spin}: fold_images equals apply at every node',
                    f'max rel diff {err:.1e}')
    return ok


def check_bound():
    """Check 2."""
    lo, hi, d_min, tol = 0.15, 0.30, 0.8, 1e-6
    p = ee_fold._node_count(lo, hi, d_min, tol, 1.0)
    nodes = ee_fold._chebyshev_nodes(lo, hi, p)
    h, d = 0.5 * (hi - lo), d_min - 0.5 * (lo + hi)
    bound = 2.0 * (h / (2.0 * (d - h))) ** p
    worst = 0.0
    for D in np.linspace(d_min, 3.0, 40):
        f = 1.0 / (nodes - D)
        for w in np.linspace(lo, hi, 101):
            l, _ = ee_fold._lagrange(nodes, w)
            worst = max(worst, abs(l @ f - 1.0 / (w - D)) * abs(w - D))
    return check(worst <= bound, f'{p} nodes: interpolated 1/(w - D) within the bound',
                 f'worst relative error {worst:.1e}, bound {bound:.1e}')


def check_dense(eps, B, no):
    """Check 3."""
    ok = True
    for level in ('adc2', 'gf2'):
        _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level=level, pieces=True)
        for spin in ('singlet', 'triplet'):
            for nr in (3, 6):
                a = ee_fold.solve_folded(P, nr, spin=spin, dense_limit=2000,
                                         tol_omega=1e-10, solver='newton')
                b = ee_fold.solve_folded(P, nr, spin=spin, dense_limit=0,
                                         solver='block', tol_omega=1e-8,
                                         tol_residual=1e-8)
                d = float(np.abs(a.omega - b.omega).max())
                ok &= check(d < 1e-9 and set(b.loop) == {'block'}
                            and b.converged.all(),
                            f'{level} {spin} {nr} roots: block equals dense',
                            f'max |dw| {d:.1e} Ha, loops {sorted(set(b.loop))}')
    return ok


def check_iteration_cap(eps, B, no):
    """Check 8."""
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    a = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=2000, tol_omega=1e-10,
                             solver='newton')
    cap = ee_fold.EE_FOLD_BLOCK_MAX_ITER
    ee_fold.EE_FOLD_BLOCK_MAX_ITER = 2
    try:
        b = ee_fold.solve_folded(P, 3, spin='singlet', dense_limit=0, solver='block',
                                 tol_omega=1e-8, tol_residual=1e-8)
    finally:
        ee_fold.EE_FOLD_BLOCK_MAX_ITER = cap
    d = float(np.abs(a.omega - b.omega).max())
    return check(d < 1e-9 and b.converged.all(),
                 'a block out of iterations hands over to the per-root Newton',
                 f'max |dw| {d:.1e} Ha, loops {sorted(set(b.loop))}')


def check_pair():
    """Check 4."""
    rng = np.random.default_rng(7)
    ns, nd, split = 300, 60, 1e-4
    e = np.concatenate([[0.5, 0.5 + split], np.linspace(0.62, 2.0, ns - 2)])
    Q = np.linalg.qr(rng.standard_normal((ns, ns)))[0]
    m = (Q * e) @ Q.T
    # doubles in equal pairs; the two pair states couple to a pair as (a, a)
    # and (a, -a): equal folds, no cross term, so A_eff keeps the split at
    # every frequency
    # the background coupling is weak: C^T C / (D - w) is a rank-nd block
    # that, at 0.03, pulls other states below the pair
    D = np.repeat(np.linspace(1.0, 3.0, nd // 2), 2)
    Ce = 0.005 * rng.standard_normal((nd, ns))
    a = 0.05 * rng.standard_normal(nd // 2)
    Ce[:, 0] = np.repeat(a, 2)
    Ce[:, 1] = np.ravel(np.column_stack([a, -a]))
    C = Ce @ Q.T
    exact = supermatrix_roots(m, C, D)[:2]
    res = ee_fold.solve_folded(synthetic_pieces(m, C, D), 2, spin='singlet',
                               dense_limit=0, solver='block', tol_omega=1e-8,
                               tol_residual=1e-6)
    d = float(np.abs(res.omega - exact).max())
    return check(d < 1e-9 and list(res.loop) == ['block', 'block']
                 and int(res.steps.max()) <= 30,
                 f'pair split {exact[1] - exact[0]:.1e} Ha over {ns} singles: both '
                 'roots by the block', f'max |dw| {d:.1e} Ha, steps '
                 f'{res.steps.tolist()}, loops {list(res.loop)}')


def check_window():
    """Check 5."""
    # M's seeds lie far above the roots near 0.5 Ha: the window from their
    # Newton steps reaches D = 1 Ha
    D = np.array([1.0, 1.4])
    C = np.array([[0.3, 0.3, 0.0], [0.3, -0.54, 0.0]])
    s = np.sum(C ** 2 / (D[:, None] - 0.5), axis=0)
    m = np.diag([0.5 + s[0], 0.5 + s[1] + 1.2e-6, 1.5])
    exact = supermatrix_roots(m, C, D)[:2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(synthetic_pieces(m, C, D), 2, spin='singlet',
                                   dense_limit=0, solver='block', tol_omega=1e-10)
    warned = any('lowest doubles energy' in str(x.message) for x in caught)
    d = float(np.abs(res.omega - exact).max())
    return check(warned and d < 1e-9 and 'block' not in res.loop,
                 'a window reaching min D warns and is solved per root',
                 f'max |dw| {d:.1e} Ha, loops {list(res.loop)}')


def check_ranks(mf, eps, B, no):
    """Check 6."""
    ok = True
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)

    def solve(o):
        r = ee_fold.solve_folded(o.pieces(), 3, spin='singlet', dense_limit=0,
                                 solver='block')
        return dict(omega=r.omega, t1=r.t1, loop=list(r.loop))

    def rank(comm):
        op = ee_ri_fold.RIFoldADC2(F, 'singlet', tile_gb=1e-9, threads=1, comm=comm)
        return ee_ri_fold.drive(op, solve)

    ser = rank(None)
    for size in (2, 3):
        outs = run_simulated(rank, size)
        same = all(np.array_equal(o['omega'], ser['omega'])
                   and np.array_equal(o['t1'], ser['t1']) for o in outs)
        ok &= check(same and set(ser['loop']) == {'block'},
                    f'{size} simulated ranks: the block roots equal the serial run '
                    'bit for bit')
    return ok


def check_hidden():
    """Check 7."""
    ok = True
    n = 800
    m = np.diag(0.5 + 0.5 * np.arange(n) / n)
    m[-1, -1] = 3.0
    C = np.zeros((1, n))
    C[0, -1] = 2.75
    rng = np.random.default_rng(7)
    na, nb = n - 50, 50
    Mf = np.zeros((n, n))
    Mf[:na, :na] = np.diag(0.5 + 0.5 * np.arange(na) / na)
    Bm = rng.standard_normal((nb, nb)) * 2e-3
    Mf[na:, na:] = np.diag(1.5 + 0.5 * np.arange(nb) / nb) + 0.5 * (Bm + Bm.T)
    Cb = np.zeros((1, n))
    Cb[0, na:] = rng.standard_normal(nb)
    Cb *= 2.2 / np.linalg.norm(Cb)
    for tag, mm, CC, DD in (('alone', m, C, [3.1]), ('coupled block', Mf, Cb, [2.2])):
        w = supermatrix_roots(mm, CC, np.asarray(DD))
        for nroots in (1, 3):
            res = ee_fold.solve_folded(synthetic_pieces(mm, CC, DD), nroots,
                                       spin='singlet', dense_limit=0, solver='block',
                                       tol_omega=1e-10)
            d = (np.abs(res.omega[:nroots] - w[:nroots]).max()
                 if res.omega.size >= nroots else np.inf)
            ok &= check(d < 1e-8 and res.converged.all(),
                        f'{tag}, nroots {nroots}: the state M misses is found',
                        f'max |dw| {d:.1e} Ha, loops {sorted(set(res.loop))}')
    return ok


def run():
    """Every check; True when all passed."""
    ok = True
    mf, eps, B, no = water_df()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ok &= check_images(mf, eps, B, no)
        ok &= check_bound()
        ok &= check_dense(eps, B, no)
        ok &= check_pair()
        ok &= check_ranks(mf, eps, B, no)
        ok &= check_hidden()
        ok &= check_iteration_cap(eps, B, no)
    ok &= check_window()
    return ok


def test_ee_adc_fold_block():
    assert run()


if __name__ == '__main__':
    good = run()
    print('\nALL PASSED' if good else '\nFAILURES DETECTED')
    sys.exit(0 if good else 1)
