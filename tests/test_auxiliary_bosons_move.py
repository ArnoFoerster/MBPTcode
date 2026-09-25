"""The auxiliary-boson route must be the same numbers as the quasi-boson closed form.

`src.SingleReference.GW.auxiliary_bosons` replaced the quasi-boson closed form
of the dRPA class (`gradients.quasi_boson_adjoint.RPAAdjoint`, a subclass of
`SingleReference.GW.quasi_boson.RPA`) -- e^t and the eigenvectors of
Abar = [e^t (A+B) e^t + e^-t (A-B) e^-t]/2 -- by the production Casida solve.
The two are the same object: with Abar = (P^1/2 e^t)^T (P^1/2 e^t), P = A+B,

  e^t U = (A+B)^-1/2 V Om^1/2 = X + Y,   (A+B)^1/2 (A-B) (A+B)^1/2 V = V Om^2,

which is Casida's X+Y in the standard RPA normalization (X+Y)^T(A+B)(X+Y) = Om.
So the move is exact in algebra, and the gates below measure only the LAPACK
path: numpy `eigh` of Abar versus scipy `eigh` of the whitened Casida matrix
after a Cholesky (or a diagonal A-B) factorization. Nothing is bitwise, and the
gate is 1e-12; the measured worst cases are quoted at each assertion.

The sign of an eigenvector column is a gauge, and the two paths pick different
ones (3 of 15 columns flip here, 40 of 95 on water), so the boson gates compare
sign-aligned columns. Every physical consumer squares the coupling, and those
-- poles and amplitudes -- are compared with no gauge handling at all.

Each gate is shown once to fail:
  * the boson gate (Om and X+Y), on d shifted by 1e-9
  * the gauge-invariant pole/amplitude gate, on bp shifted by 1e-9
  * the "production does not import src.gradients" gate, on a module of
    src.gradients, which does
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from src.gradients.quasi_boson_adjoint import RPAAdjoint as RPA
from src.SingleReference.GW import auxiliary_bosons as prod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EPS = np.array([-0.90, -0.62, -0.35, 0.18, 0.44, 0.83, 1.25, 1.70])
NOCC = 3
N_OV = NOCC * (len(EPS) - NOCC)


def factors(naux, seed=5, scale=0.05):
    """C_ov of shape (naux, n_ov), exactly as tests/test_auxiliary_bosons.py builds it."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((naux, N_OV)) * scale


def ph_energies():
    return (EPS[NOCC:][None, :] - EPS[:NOCC][:, None]).ravel()


def inputs():
    """(bp, c_ov, d) -- the full-rank case every test in the old file uses."""
    return factors(N_OV + 6, seed=9)[:, :len(EPS)], factors(N_OV + 6), ph_energies()


def old_exact_bosons(c_ov, d):
    """Verbatim pre-move body: the dRPA bosons from the quasi-boson closed form."""
    v = np.asarray(c_ov, float).T @ np.asarray(c_ov, float)
    rpa = RPA(np.diag(np.asarray(d, float)) + 2.0 * v, 2.0 * v)
    om, u = rpa.eigAbar
    return om, rpa.exp_t @ u


def old_ab_bosons(c_ov, d, c_ab):
    """Verbatim pre-move body: the dRPA re-solved inside the AB basis, closed form."""
    cc = np.asarray(c_ov, float) @ c_ab
    v = cc.T @ cc
    a = (c_ab * np.asarray(d, float)[:, None]).T @ c_ab + 2.0 * v
    rpa = RPA(a, 2.0 * v)
    om, u = rpa.eigAbar
    return om, rpa.exp_t @ u


def old_exact_from_factors(bp, c_ov, d):
    """Verbatim pre-move body, on top of the closed-form bosons."""
    om, xy = old_exact_bosons(c_ov, d)
    w = np.sqrt(2.0) * ((np.asarray(bp, float).T @ c_ov) @ xy)
    return om, (w ** 2).T


def old_ab_from_factors(bp, c_ov, d, c_ab):
    """Verbatim pre-move body; ab_basis and ab_couplings never used RPA."""
    om, xy = old_ab_bosons(c_ov, d, c_ab)
    w = prod.ab_couplings(bp, c_ov, c_ab, xy)
    return om, (w ** 2).T


def aligned(x, y):
    """max |x - y| with each column of y put in x's sign gauge."""
    s = np.sign(np.einsum('ik,ik->k', x, y))
    s[s == 0] = 1.0
    return float(np.abs(x - y * s[None, :]).max())


def test_exact_bosons_reproduces_the_quasi_boson_closed_form():
    """Casida X+Y == e^t U: 8.4e-15 on Om, 6.4e-14 on X+Y (different eigensolver path).

    Not bitwise: the closed form diagonalizes Abar with numpy eigh, CasidaSolver
    diagonalizes (A-B)^1/2 (A+B) (A-B)^1/2 with scipy eigh. 3 of the 15 columns
    come back with the opposite sign, which is the eigenvector gauge.
    """
    _, c_ov, d = inputs()
    om_new, xy_new = prod.exact_bosons(c_ov, d)
    om_old, xy_old = old_exact_bosons(c_ov, d)
    assert np.abs(om_new - om_old).max() < 1e-12
    assert aligned(xy_old, xy_new) < 1e-12


def test_ab_bosons_reproduces_the_quasi_boson_closed_form():
    """The compressed solve, same gate: 5.3e-15 on Om, 5.2e-14 on X+Y."""
    _, c_ov, d = inputs()
    for n_bosons in (None, 12, 6):
        c_ab = prod.ab_basis(c_ov, n_bosons=n_bosons)
        om_new, xy_new = prod.ab_bosons(c_ov, d, c_ab)
        om_old, xy_old = old_ab_bosons(c_ov, d, c_ab)
        assert np.abs(om_new - om_old).max() < 1e-12, n_bosons
        assert aligned(xy_old, xy_new) < 1e-12, n_bosons


def test_the_boson_gate_can_fail():
    """A 1e-9 shift on the particle-hole energies must move Om and X+Y past 1e-12."""
    _, c_ov, d = inputs()
    om_new, xy_new = prod.exact_bosons(c_ov, d)
    om_old, xy_old = old_exact_bosons(c_ov, d + 1e-9)
    assert np.abs(om_new - om_old).max() > 1e-12
    assert aligned(xy_old, xy_new) > 1e-12


def test_the_poles_and_amplitudes_carry_no_gauge():
    """Squared couplings: 8.4e-15 on the poles, 8.0e-17 on the amplitudes, no alignment."""
    bp, c_ov, d = inputs()
    p_new, a_new = prod.exact_from_factors(bp, c_ov, d)
    p_old, a_old = old_exact_from_factors(bp, c_ov, d)
    assert np.abs(p_new - p_old).max() < 1e-12
    assert np.abs(a_new - a_old).max() < 1e-12
    for n_bosons in (None, 12, 6):
        c_ab = prod.ab_basis(c_ov, n_bosons=n_bosons)
        p_new, a_new = prod.ab_from_factors(bp, c_ov, d, c_ab=c_ab)
        p_old, a_old = old_ab_from_factors(bp, c_ov, d, c_ab)
        assert np.abs(p_new - p_old).max() < 1e-12, n_bosons
        assert np.abs(a_new - a_old).max() < 1e-12, n_bosons


def test_the_amplitude_gate_can_fail():
    """A 1e-9 shift on the three-index factors of state p must move the amplitudes."""
    bp, c_ov, d = inputs()
    _, a_new = prod.exact_from_factors(bp, c_ov, d)
    _, a_old = old_exact_from_factors(bp + 1e-9, c_ov, d)
    assert np.abs(a_new - a_old).max() > 1e-12


def test_a_real_molecule_agrees_on_the_physical_output():
    """Water/cc-pVDZ, the input the core-state test builds: 1.8e-13 poles, 7.3e-14 amplitudes.

    The eigenvectors are looser -- 3.4e-12 sign-aligned -- because the 44th
    boson sits 3.9e-4 Ha from its neighbour and an eigenvector's sensitivity to
    the eigensolver path goes as one over that gap. The poles and the squared
    couplings, which are what the self-energy sees, are unaffected.
    """
    pytest.importorskip('pyscf')
    from pyscf import gto, scf
    from src.Base.pyscf_interface import get_density_fitting_coefficients
    from src.SingleReference.base import get_occ_virt_indices
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-10)
    eps = np.asarray(mf.mo_energy, float)
    b = get_density_fitting_coefficients(mol, mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, mol.nelectron // 2)
    c_ov = b[:, occ, :][:, :, virt].reshape(b.shape[0], -1)
    d = (eps[virt][None, :] - eps[occ][:, None]).ravel()
    bp = b[:, 0, :]                                           # the oxygen 1s

    om_new, xy_new = prod.exact_bosons(c_ov, d)
    om_old, xy_old = old_exact_bosons(c_ov, d)
    assert np.abs(om_new - om_old).max() < 1e-12
    assert aligned(xy_old, xy_new) < 1e-11
    p_new, a_new = prod.exact_from_factors(bp, c_ov, d)
    p_old, a_old = old_exact_from_factors(bp, c_ov, d)
    assert np.abs(p_new - p_old).max() < 1e-12
    assert np.abs(a_new - a_old).max() < 1e-12
    c_ab = prod.ab_basis(c_ov)
    p_new, a_new = prod.ab_from_factors(bp, c_ov, d, c_ab=c_ab)
    p_old, a_old = old_ab_from_factors(bp, c_ov, d, c_ab)
    assert np.abs(p_new - p_old).max() < 1e-12
    assert np.abs(a_new - a_old).max() < 1e-12


def imports_gradients(module):
    """True if importing `module` in a fresh interpreter leaves src.gradients loaded."""
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    code = ('import sys, importlib; importlib.import_module(%r); '
            "print(any(m == 'src.gradients' or m.startswith('src.gradients.') "
            'for m in sys.modules))' % module)
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1] == 'True'


def test_production_does_not_reach_into_the_gradients_package():
    """A fresh interpreter importing the production module must not load src.gradients."""
    assert not imports_gradients('src.SingleReference.GW.auxiliary_bosons')


def test_the_sys_modules_gate_can_fail():
    """A module of src.gradients -- the one the closed form above comes from --
    must report True under the same probe."""
    assert imports_gradients('src.gradients.quasi_boson_adjoint')
