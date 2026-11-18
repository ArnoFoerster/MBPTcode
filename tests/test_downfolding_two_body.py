"""The exact active-space integrals and the packing of g_eff.

active_eri reproduces the active block of the exact MO integrals of the
oracle. Packed spin-free and spin-orbital, the same restricted Hamiltonian of
H2/6-31G (all four spatial orbitals active, two electrons) gives the same
ground state energy from the openfermion and the pyscf_fci backends, and both
equal pyscf's FCI energy when the constant is the nuclear repulsion. An
empty active space gives zero-size blocks and a g_eff of the constant alone,
and the spin-free g_eff never aliases its inputs.

The full RPA of method='rpa' on the casida route (one run, kept for any
active space) gives the oracle's direct RPA energy on exact and on fitted
integrals, keeps X+Y in the pair order of the active-pair mask and with the
normalization xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 4K)^-1, and
refuses the modes and options it does not have.
"""
import numpy as np
import pytest
from pyscf import fci

from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.MultiReference.Downfolding.effective_hamiltonian import (
    build_effective_hamiltonian, run_low_level)
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import (active_pair_mask,
                                                      partition)
from src.MultiReference.Downfolding.two_body import active_eri, pack
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import \
    rpa_correlation_energy_casida
from src.Solvers.active_space import solve
from tests.downfolding_oracle import (drpa, h2, mo_integrals, rpa_energy,
                                      water)


def test_active_eri_is_exact():
    mf = water()
    idx = [2, 4, 5, 6]
    act = mf.mo_coeff[:, idx]
    _, eri = mo_integrals(mf)
    ref = eri[np.ix_(idx, idx, idx, idx)]
    aa, ab, bb = active_eri(mf.mol, (act, act))
    for block in (aa, ab, bb):
        assert block.shape == (4,) * 4
        assert np.max(np.abs(block - ref)) < 1e-10


def test_active_eri_ab_index_convention():
    """ab[t,u,v,w] = (t_a u_a | v_b w_b): alpha pair first."""
    mf = water()
    ia, ib = [2, 4, 5, 6], [3, 4, 5, 6]
    _, eri = mo_integrals(mf)
    _, ab, _ = active_eri(mf.mol, (mf.mo_coeff[:, ia], mf.mo_coeff[:, ib]))
    assert np.max(np.abs(ab - eri[np.ix_(ia, ia, ib, ib)])) < 1e-10


def test_pack_spin_free_contract():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 3))
    h = h + h.T
    v = rng.normal(size=(3,) * 4)
    g = pack((h, h), (v, v, v), 1.5, spin_free=True)
    assert g[0] == 1.5
    assert np.array_equal(g[1], h)
    assert np.array_equal(g[2], v.transpose(0, 2, 1, 3))
    assert g[2][0, 1, 2, 0] == v[0, 2, 1, 0]


def test_pack_spin_orbital_solves_the_same():
    mf = h2()
    h, eri = mo_integrals(mf)
    n = h.shape[0]
    e0 = mf.energy_nuc()
    args = ((h, h), (eri, eri, eri), e0)
    g_so = pack(*args, spin_free=False)
    g_sf = pack(*args, spin_free=True)
    assert g_so[0] == e0
    e_so = solve('openfermion', g_so, 2 * n, 2)
    e_sf = solve('pyscf_fci', g_sf, n, 2, spin_free=True)
    e_fci = fci.FCI(mf).kernel()[0]
    assert abs(e_so - e_sf) < 1e-9
    assert abs(e_sf - e_fci) < 1e-9


def test_pack_refuses_different_channels():
    rng = np.random.default_rng(1)
    h = np.eye(2)
    v = rng.normal(size=(2,) * 4)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h + 0.1), (v, v, v), 0.0, spin_free=True)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h), (v, v + 0.1, v), 0.0, spin_free=False)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h), (v, v, v + 0.1), 0.0, spin_free=False)


def test_empty_active_space():
    """n_act = 0: zero-size integrals without an integral transformation, and
    a g_eff that carries only the constant."""
    mf = water()
    empty = mf.mo_coeff[:, []]
    blocks = active_eri(mf.mol, (empty, empty))
    assert [b.shape for b in blocks] == [(0, 0, 0, 0)] * 3
    h = np.zeros((0, 0))
    for spin_free in (True, False):
        g = pack((h, h), blocks, -76.0, spin_free=spin_free)
        assert g[0] == -76.0
        assert g[1].shape == (0, 0) and g[2].shape == (0, 0, 0, 0)


def test_pack_spin_free_does_not_alias():
    h = np.eye(2)
    v = np.ones((2,) * 4)
    g = pack((h, h), (v, v, v), 0.0, spin_free=True)
    g[1][0, 0] = 5.0
    g[2][0, 0, 0, 0] = 5.0
    assert h[0, 0] == 1.0 and v[0, 0, 0, 0] == 1.0


@pytest.fixture(scope='module')
def wat():
    """(mf, eri, eps, nocc) of water/6-31G, eri the exact chemist MO integrals."""
    mf = water()
    _, eri = mo_integrals(mf)
    return mf, eri, mf.mo_energy, int(np.count_nonzero(mf.mo_occ))


def _pairs(eri, eps, nocc):
    """(delta, K) on the (i, a) pairs in C order, K_ia,jb = (ia|jb)."""
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    n = delta.size
    return delta, eri[:nocc, nocc:, :nocc, nocc:].reshape(n, n)


@pytest.mark.parametrize('df', [False, True])
def test_full_rpa_energy(wat, df):
    """E_RPA,corr = 1/2 (sum omega - Tr A) of the one solve, against the
    oracle on the exact integrals and against rpa_correlation_energy_casida
    on the same integrals. df=True on this non-DF RHF takes the helper's
    eigendecomposition of the exact MO integrals (eigenvalues above 1e-12
    kept, 88 of 169), exact enough for the 1e-8 against the exact oracle:
    measured |e_c - oracle| = 8.5e-14 Eh (df=True) and 0.0 (df=False).
    """
    mf, eri, eps, nocc = wat
    low = run_low_level(mf, method='rpa', mode='casida', df=df)
    assert low.method == 'rpa' and low.mf is mf and low.mode == 'casida'
    assert low.df is df and low.numerics is None and low.cache == {}
    assert low.nocc == nocc and np.array_equal(low.eps, eps)
    n_pair = nocc * (len(eps) - nocc)
    assert low.omega.shape == (n_pair,) and low.xpy.shape == (n_pair, n_pair)
    if df:
        assert low.eri is None and low.factors.shape[1:] == (len(eps),) * 2
        lr = LinearResponseSolver(eps, coeff_df=low.factors,
                                  spin_mode='restricted')
    else:
        assert low.factors is None and np.abs(low.eri - eri).max() < 1e-10
        lr = LinearResponseSolver(eps, eri_chemist=low.eri,
                                  spin_mode='restricted')
    assert abs(low.e_c_full - rpa_energy(eri, eps, nocc)) < 1e-8
    assert abs(low.e_c_full - rpa_correlation_energy_casida(lr, nocc)) < 1e-9


def test_xpy_convention(wat):
    """xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 4K)^-1 with factor 1:
    A - B = delta is diagonal, so the solver's X+Y = delta^1/2 Z omega^-1/2,
    the oracle's. The rows are in C order over (i, a), so this also pins the
    pair order. Measured max deviation 8.3e-13 (df=False)."""
    mf, eri, eps, nocc = wat
    low = run_low_level(mf, method='rpa', df=False)
    delta, K = _pairs(eri, eps, nocc)
    ref = np.linalg.inv(np.diag(delta) + 4.0 * K)
    got = low.xpy @ np.diag(1.0 / low.omega) @ low.xpy.T
    assert np.abs(got - ref).max() < 1e-8
    omega, xpy = drpa(eri, eps, nocc)
    assert np.abs(low.omega - omega).max() < 1e-10
    assert np.abs(xpy @ np.diag(1.0 / omega) @ xpy.T - ref).max() < 1e-8


@pytest.mark.parametrize('df', [False, True])
def test_casida_pair_order_is_the_mask_order(wat, df):
    """Ruling R2: the Casida pair index of build_casida_matrices is C order
    over (i, a), a counted from nocc, which is the order active_pair_mask
    reads. A - B = delta (diagonal to rounding, zero off it), and delta is
    non-degenerate on water, so a different order would show; the mask then picks the pairs whose i and a
    are both active."""
    mf, eri, eps, nocc = wat
    factors = get_density_fitting_coefficients(mf.mol, mf) if df else None
    lr = LinearResponseSolver(eps, coeff_df=factors,
                              eri_chemist=None if df else eri,
                              spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    delta, K = _pairs(eri, eps, nocc)
    assert np.unique(delta).size == delta.size
    amb = A - B
    assert np.abs(np.diag(amb) - delta).max() < 1e-12
    assert not np.any(amb - np.diag(np.diag(amb)))
    assert np.abs(np.diag(A) - (delta + 2.0 * np.diag(K))).max() < 1e-10
    aspace = ActiveSpace(len(eps), nocc, [2, 4], [6, 8])
    labels = [(i, a) for i in range(nocc) for a in range(nocc, len(eps))]
    picked = [p for p, m in zip(labels, active_pair_mask(partition(aspace, eps)))
              if m]
    assert picked == [(2, 6), (2, 8), (4, 6), (4, 8)]


def test_mode_refusals(wat):
    mf = wat[0]
    with pytest.raises(ValueError, match="'casida'"):
        run_low_level(mf, method='rpa', mode='spacetime')
    for mode in ('space-time', 'space_time', 'imagfrequency', 'imag_frequency'):
        with pytest.raises(NotImplementedError, match="'casida'"):
            run_low_level(mf, method='rpa', mode=mode)
    with pytest.raises(TypeError, match="method='rpa'.*screening"):
        run_low_level(mf, method='rpa', screening='crpa')
    assert rpa.MODES == ('casida', 'imagfrequency', 'space-time')
    assert rpa.SCREENINGS == ('bare', 'crpa', 'crpa_ph', 'mrpa')
    assert rpa.SIGMAS == (None, 'srg', 'mode_a')


def test_rpa_terms_not_built_yet(wat):
    """The full RPA runs; building an active space from it is not there yet,
    and says so."""
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', mode='Casida')
    assert low.mode == 'casida'
    aspace = ActiveSpace.from_counts(eps, nocc, 2, 2)
    with pytest.raises(NotImplementedError, match='not implemented yet'):
        build_effective_hamiltonian(low, aspace)
