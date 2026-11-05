"""Non-Dyson IP-ADC(3) with the 1h-1h block applied on the fly (adc_r_nondyson_otf).

Checks, with an exact DF factor (eigen-decomposition of (pq|rs)) so that the dense
spin-orbital oracle sees the same integrals:
  * the restricted, streamed 1h-1h block equals Eq. (A8) of Dempwolff, Schneider, Hodecker
    and Dreuw, J. Chem. Phys. 150, 064108 (2019), evaluated densely over spin orbitals;
  * outside the 1h-1h block the operator is the precomputed one, entry for entry;
  * on the diagonal of the 1h-1h block both forms agree (they differ off it at third
    order), and the IPs of the two operators agree to a fraction of a meV;
and, with simulated ranks, the distributed product equals the serial one.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.ADC import build_static_correction
from src.SingleReference.ADC import adc_r_nondyson_otf as otf
from src.SingleReference.ADC.adc_r_nondyson_df import build_operator
from src.SingleReference.ADC.base import ADCSolverRestricted
from tests.molecules import WATER

H2O = WATER
NH3 = 'N 0 0 0.1; H 0.94 0 -0.25; H -0.47 0.81 -0.25; H -0.47 -0.81 -0.25'


def _exact_system(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    n, nocc = mf.mo_coeff.shape[1], mol.nelectron // 2
    eri = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), n)
    w, v = np.linalg.eigh(eri.reshape(n * n, n * n))
    keep = w > 1e-12
    B = (v[:, keep] * np.sqrt(w[keep])).T.reshape(-1, n, n)
    s = ADCSolverRestricted.from_arrays(mf.mo_energy, B_aa=B, nocc=nocc, level='adc3')
    sc = build_static_correction(mf, kind='mp2_unrelaxed', B_aa=B)
    return mf, eri, nocc, s, sc


def _a8_spin_orbital(eri, eps, nocc):
    """Eq. (A8) over spin orbitals (alpha-alpha block), t_ijab = <ij||ab>/(e_a+e_b-e_i-e_j)."""
    n = eps.size
    so = [(p, s) for p in range(nocc) for s in (0, 1)] + [(p, s) for p in range(nocc, n) for s in (0, 1)]
    idx = np.array([p for p, _ in so])
    spin = np.array([s for _, s in so])
    d = spin[:, None] == spin[None, :]
    g = eri[np.ix_(idx, idx, idx, idx)].transpose(0, 2, 1, 3) * d[:, None, :, None] * d[None, :, None, :]
    v = g - g.transpose(0, 1, 3, 2)
    e, no = eps[idx], 2 * nocc
    o, x = slice(0, no), slice(no, None)
    eo, ev = e[o], e[x]
    t = v[o, o, x, x] / (ev[None, None, :, None] + ev[None, None, None, :]
                         - eo[:, None, None, None] - eo[None, :, None, None])
    P = lambda m: m + m.T
    I = (-np.diag(eo)
         + 0.25 * P(np.einsum('ikab,jkab->ij', t, v[o, o, x, x]))
         - 0.125 * P(np.einsum('jkcd,ikab,abcd->ij', t, t, v[x, x, x, x], optimize=True))
         + 0.5 * P(np.einsum('jlac,ikab,kclb->ij', t, t, v[o, x, o, x], optimize=True))
         - 0.25 * P(np.einsum('klmi,klab,mjab->ij', v[o, o, o, o], t, t, optimize=True))
         - P(np.einsum('kcia,klab,jlbc->ij', v[o, x, o, x], t, t, optimize=True)))
    rho_oo = -0.5 * np.einsum('ikab,jkab->ij', t, t)
    rho_vv = 0.5 * np.einsum('ijac,ijbc->ab', t, t)
    rho_ov = ((np.einsum('ijbc,jabc->ia', t, v[o, x, x, x]) + np.einsum('jkab,jkib->ia', t, v[o, o, o, x]))
              / (2.0 * (eo[:, None] - ev[None, :])))
    I += (-0.5 * P(np.einsum('ikjl,kl->ij', v[o, o, o, o], rho_oo))
          - P(np.einsum('ikja,ka->ij', v[o, o, o, x], rho_ov))
          - 0.5 * P(np.einsum('iajb,ab->ij', v[o, x, o, x], rho_vv)))
    alpha = np.where(spin[:no] == 0)[0]
    return I[np.ix_(alpha, alpha)]


def _dense(aop, n):
    return np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])


@pytest.fixture(scope='module', params=[(H2O, '6-31g'), (NH3, 'cc-pvdz')], ids=['h2o', 'nh3'])
def operators(request):
    mf, eri, nocc, s, sc = _exact_system(*request.param)
    aop_p, diag_p, dims = build_operator(s, nocc, 'IP', sc)
    aop_o, diag_o, _ = build_operator(s, nocc, 'IP', sc, ip_block='on_the_fly')
    n = diag_p.size
    return dict(mf=mf, eri=eri, nocc=nocc, no=dims['n_orb'],
                Hp=_dense(aop_p, n), Ho=_dense(aop_o, n), diag_o=diag_o, diag_p=diag_p)


def test_block_equals_spin_orbital_a8(operators):
    no = operators['no']
    a8 = _a8_spin_orbital(operators['eri'], operators['mf'].mo_energy, operators['nocc'])
    assert np.max(np.abs(operators['Ho'][:no, :no] - a8)) < 1e-12


def test_rest_of_operator_unchanged(operators):
    no, Ho, Hp = operators['no'], operators['Ho'], operators['Hp']
    assert np.array_equal(Ho[no:], Hp[no:])
    assert np.array_equal(Ho[:no, no:], Hp[:no, no:])


def test_diagonal_and_roots_match_precomputed(operators):
    no, Ho, Hp = operators['no'], operators['Ho'], operators['Hp']
    assert np.max(np.abs(np.diag(Ho)[:no] - np.diag(Hp)[:no])) < 1e-12
    assert np.max(np.abs(Ho - Ho.T)) < 1e-12
    e_o, e_p = np.linalg.eigvalsh(Ho)[:no], np.linalg.eigvalsh(Hp)[:no]
    assert np.max(np.abs(e_o - e_p)) * HARTREE_TO_EV < 1e-3         # below 1 meV
    # the two forms are different off the diagonal: the check above is not empty
    assert np.max(np.abs(Ho[:no, :no] - Hp[:no, :no])) > 1e-7


def test_preconditioner_diagonal(operators):
    # satellites: the same zeroth-order diagonal; kept orbitals: the static part of (A8),
    # within third order of the full diagonal
    no, d_o, d_p = operators['no'], operators['diag_o'], operators['diag_p']
    assert np.array_equal(d_o[no:], d_p[no:])
    assert np.max(np.abs(d_o[:no] - np.diag(operators['Ho'])[:no])) < 0.05


@pytest.mark.parametrize('size', [2, 3])
def test_third_order_over_ranks_equals_serial(size):
    mf, eri, nocc, s, sc = _exact_system(NH3, 'cc-pvdz')
    B, eps = s.B_aa, mf.mo_energy
    x = np.random.default_rng(5).standard_normal(nocc)
    y0 = otf.third_order_apply(B, eps, nocc, x)
    S0 = otf.static_part(B, eps, nocc, sc[:nocc, :nocc])
    out = run_simulated(lambda comm: (otf.third_order_apply(B, eps, nocc, x),
                                      otf.static_part(B, eps, nocc, sc[:nocc, :nocc])), size)
    for r, (y, S) in enumerate(out):
        assert np.max(np.abs(y - y0)) < 1e-12 * np.max(np.abs(y0)), f'rank {r}'
        assert np.max(np.abs(S - S0)) < 1e-12 * np.max(np.abs(S0)), f'rank {r}'
        assert np.array_equal(y, out[0][0]), f'rank {r} differs from rank 0'
