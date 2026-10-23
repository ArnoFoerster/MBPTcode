"""The orbital-response solve (src.Base.pyscf.cphf) reaches its true residual,
and the relaxed densities built on it reach the finite field.

  1. RHF, water 6-31G: the CPHF equation of a random right-hand side solved
     by cphf.solve to a true residual below 1e-12; pyscf's own
     scf.cphf.solve on the same equation stops above 1e-7 whatever tol it is
     given (the floor this replaces, so the check can fail).
  2. the relaxed MP2 density (compute_mp2_density_matrix_ao, relax=True):
     its correlation dipole Tr((P_MP2 - P_HF) z) equals the 5-point
     finite-field derivative of pyscf's MP2 correlation energy (h = 1e-3, SCF
     to an orbital gradient of 1e-12) to 1e-9 a.u.; with pyscf's
     scf.cphf.solve in its place it misses by more than 1e-8.
  3. UHF, NH2 6-31G: solve_cphf_relaxation_uhf of random blocks satisfies
     its equation to 1e-9, with and without the preconditioner's level shift,
     which leaves the solution unchanged to 1e-12. (The scipy gmres it used
     before printed a warning at max_cycle and returned the vector anyway.)
  4. a solve that cannot reach tol warns (CphfNotConverged) and flags
     'cphf_not_converged' inside cphf.collecting
     (refuse inputs, warn numerics).

Run: python tests/test_cphf_residual.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, mp, scf
from pyscf.scf import cphf as pyscf_cphf

from src.Base.numerics.numerics import Numerics
from src.Base.pyscf import cphf
from src.SingleReference.DensityMatrix import density_matrix as dmm
from tests.adc_common import WATER_C2V, check, converge_newton


def tight_rhf(mol, h1=None, dm0=None):
    mf = scf.RHF(mol)
    if h1 is not None:
        mf.get_hcore = lambda *a, **k: h1
    return converge_newton(mf, dm0)


def rhf_equation(mf):
    """(fvind, e_ai, mo_occ) of the RHF CPHF equation of mf."""
    C, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    Co, Cv = C[:, occ > 0], C[:, occ == 0]
    nv, no = Cv.shape[1], Co.shape[1]

    def fvind(x):
        x = np.asarray(x).reshape(-1, nv, no)
        out = []
        for z in x:
            dm = Cv @ z @ Co.T
            dm = dm + dm.T
            vj, vk = mf.get_jk(mf.mol, dm, hermi=1)
            out.append(Cv.T @ (2.0 * vj - vk) @ Co)
        return np.array(out).reshape(x.shape)

    return fvind, eps[occ == 0][:, None] - eps[occ > 0][None, :], occ


def check_rhf_residual():
    """Check 1."""
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    mf = tight_rhf(mol)
    fvind, e_ai, occ = rhf_equation(mf)
    rhs = np.random.default_rng(5).standard_normal(e_ai.shape) * 0.05

    def residual(x):
        return np.abs(rhs - e_ai * x - fvind(x).reshape(e_ai.shape)).max()

    x = cphf.solve(fvind, e_ai, rhs, numerics=Numerics(scf={'cphf_tol': 1e-12}))
    ok = check(residual(x) < 1e-12, 'RHF: cphf.solve reaches its true residual',
               f'{residual(x):.1e}')
    worst = min(residual(pyscf_cphf.solve(fvind, mf.mo_energy, occ, -rhs, tol=t,
                                          max_cycle=200)[0]) for t in (1e-9, 1e-14))
    ok &= check(worst > 1e-7, "pyscf's scf.cphf.solve stops above 1e-7 at any tol",
                f'{worst:.1e}')
    return ok


def check_mp2_dipole():
    """Check 2."""
    ok = True
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    z = mol.intor('int1e_r')[2]
    h0 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    mf = tight_rhf(mol)
    h = 1e-3
    e = {s: mp.MP2(tight_rhf(mol, h0 + s * h * z, mf.make_rdm1())).kernel()[0]
         for s in (-2, -1, 1, 2)}
    fd = (-e[2] + 8.0 * e[1] - 8.0 * e[-1] + e[-2]) / (12.0 * h)

    def dipole():
        P = dmm.compute_mp2_density_matrix_ao(mf, mol, relax=True)
        return float(np.sum((P - mf.make_rdm1()) * z))

    an = dipole()
    ok &= check(abs(an - fd) < 1e-9, 'relaxed MP2 correlation dipole is the '
                'finite-field derivative', f'{an:+.12f} vs {fd:+.12f}, {an - fd:+.1e}')
    # the same with pyscf's solver in the shared solver's place
    keep = dmm.cphf_solve

    def pyscf_solve(fvind, diag, rhs, numerics=None, **kw):
        s = numerics.scf
        return pyscf_cphf.solve(fvind, mf.mo_energy, mf.mo_occ, -rhs,
                                max_cycle=s.cphf_max_cycle, tol=s.cphf_tol)[0]

    dmm.cphf_solve = pyscf_solve
    try:
        old = dipole()
    finally:
        dmm.cphf_solve = keep
    ok &= check(abs(old - fd) > 1e-8, "with pyscf's scf.cphf.solve it misses",
                f'{old - fd:+.1e}')
    return ok


def check_uhf():
    """Check 3."""
    ok = True
    # NH2, not OH: linear OH's degenerate pi orbitals make its orbital
    # Hessian singular (lowest eigenvalue -2.5e-10), and a random right-hand
    # side then has no solution; NH2's lowest eigenvalue is 0.08
    mol = gto.M(atom='N 0 0 0.149; H 0 0.799 -0.522; H 0 -0.799 -0.522',
                basis='6-31g', spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    na, nb = mol.nelec
    nmo = mf.mo_coeff[0].shape[1]
    rng = np.random.default_rng(3)

    def sym(n):
        a = rng.standard_normal((n, n))
        return 1e-2 * (a + a.T)

    args = (sym(na), 1e-2 * rng.standard_normal((na, nmo - na)), sym(nmo - na),
            sym(nb), 1e-2 * rng.standard_normal((nb, nmo - nb)), sym(nmo - nb))
    # the equation as the function builds it, for its true residual
    seen = {}
    keep = dmm.cphf_solve

    def spy(coupling, diag, rhs, **kw):
        x = keep(coupling, diag, rhs, **kw)
        seen['res'] = np.abs(rhs - diag * x - coupling(x)).max()
        return x

    dmm.cphf_solve = spy
    try:
        a0, b0 = dmm.solve_cphf_relaxation_uhf(mf, na, nb, *args)
        r0 = seen['res']
        # 1.0 lies above NH2's smallest gap (0.58), so it moves the preconditioner
        a1, b1 = dmm.solve_cphf_relaxation_uhf(mf, na, nb, *args, level_shift=1.0)
        r1 = seen['res']
    finally:
        dmm.cphf_solve = keep
    ok &= check(max(r0, r1) < 1e-9, 'UHF: the solve satisfies its equation',
                f'{r0:.1e}, level shift 1.0: {r1:.1e}')
    d = max(np.abs(a0 - a1).max(), np.abs(b0 - b1).max())
    ok &= check(d < 1e-12, 'the preconditioner shift leaves the solution', f'{d:.1e}')
    return ok


def check_unreachable_tol():
    """Check 4."""
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    mf = tight_rhf(mol)
    fvind, e_ai, _ = rhf_equation(mf)
    flags = []
    with warnings.catch_warnings(record=True) as caught, cphf.collecting(flags):
        warnings.simplefilter('always')
        x = cphf.solve(fvind, e_ai, np.ones_like(e_ai), numerics=Numerics(
            scf={'cphf_tol': 1e-30, 'cphf_max_cycle': 2, 'cphf_refinements': 2}))
    warned = any(issubclass(w.category, cphf.CphfNotConverged) for w in caught)
    return check(warned and [f['flag'] for f in flags] == ['cphf_not_converged']
                 and np.isfinite(x).all(),
                 'an unreachable tol warns and flags cphf_not_converged',
                 str([f['flag'] for f in flags]))


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        all_ok &= check_rhf_residual()
        all_ok &= check_mp2_dipole()
        all_ok &= check_uhf()
        all_ok &= check_unreachable_tol()
    return all_ok


def test_cphf_residual():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
