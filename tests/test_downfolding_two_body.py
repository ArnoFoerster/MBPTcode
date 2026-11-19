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

The cRPA correction of an active space, the active pairs deleted from
delta + 4K and one Cholesky solve, equals the oracle's pole sum of the
deleted problem and the paper's literal zeroing of A+B and A-B on the active
block (Eq. 2.14, the Omega = 0 modes dropped), and differs from zeroing A+B
alone; it has the symmetry of a real chemist integral, only lowers the
diagonal, vanishes when every orbital is active and is the fully screened
W(0) - v when no pair is active. cRPA_ph is cRPA on the particle-hole
elements, 'bare' is zero. It also equals the cRPA written in spin
orbitals, both spins and no spin adaptation (ruling R12), which pins the
restricted factor -4 on delta + 4K.

E_c^A[v_eff], the dRPA of the active ph pairs with v_eff, and the static
defect max |W(0) - W^A[v_eff](0)| equal the oracle's for bare, crpa and
crpa_ph; the crpa defect is zero, and with no active ph pair E_c^A = 0.

mRPA (Scott and Booth's Eq. 5) equals the oracle's spin-orbital
construction, which solves every singlet and triplet mode itself; its
same-spin and opposite-spin blocks are equal (no triplet-type correction),
which fixes the restricted eta0_S as the active block of xpy xpy^T with
factor 1. It touches only the ph elements, the dRPA of the active pairs
with its v_eff has the full RPA's projected zeroth moment, it is zero with
every orbital active, and with Sigma^E its exchange energy term is zero.
"""
import numpy as np
import pytest
from pyscf import fci

from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.MultiReference.Downfolding.effective_hamiltonian import (
    build_effective_hamiltonian, run_low_level)
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import (active_pair_mask,
                                                      partition,
                                                      ph_element_mask)
from src.MultiReference.Downfolding.two_body import active_eri, pack
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import \
    rpa_correlation_energy_casida
from src.Solvers.active_space import solve
from tests.downfolding_oracle import (drpa, h2, mo_integrals,
                                      reference_energy, rpa_energy,
                                      vtilde_crpa, vtilde_crpa_so,
                                      vtilde_crpa_zeroing, vtilde_mrpa,
                                      vtilde_zero_apb_only, w0_minus_v,
                                      w_static_active, water)


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
    for mode in ('space-time', 'space_time'):
        # both spellings reach the space-time route, which needs an ISDF grid
        with pytest.raises(ValueError,
                           match="mode='space-time'.*numerics.isdf"):
            run_low_level(mf, method='rpa', mode=mode)
    with pytest.raises(TypeError, match="method='rpa'.*screening"):
        run_low_level(mf, method='rpa', screening='crpa')
    assert rpa.MODES == ('casida', 'imagfrequency', 'space-time')
    assert rpa.SCREENINGS == ('bare', 'crpa', 'crpa_ph', 'mrpa')
    assert rpa.SIGMAS == (None, 'srg', 'mode_a')


@pytest.fixture(scope='module')
def low_exact(wat):
    """The casida RPALowLevel of water/6-31G on the exact integrals."""
    return run_low_level(wat[0], method='rpa', df=False)


def _contiguous_44(eps, nocc):
    return ActiveSpace.from_counts(eps, nocc, 2, 2)


def _non_contiguous(eps, nocc):
    return ActiveSpace(len(eps), nocc, [2, 4], [5, 7])


@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_crpa_is_the_deleted_problem(wat, low_exact, aspace):
    """cRPA by deletion and one Cholesky solve equals the oracle's pole sum
    of the deleted problem and the paper's literal Eq. 2.14 (A+B and A-B
    zeroed on the active block, the Omega = 0 modes dropped), and differs
    from zeroing A+B alone (whose M is indefinite, one negative Omega^2 per
    active pair). Measured max deviation (contiguous / non-contiguous): pole
    sum 7.5e-16 / 1.6e-15, literal zeroing 3.6e-15 / 3.4e-15,
    zero-A+B-only 0.27 / 0.82."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    assert v.shape == (part.n_act,) * 4
    assert np.abs(v - vtilde_crpa(eri, eps, nocc, part)).max() < 1e-8
    assert np.abs(v - vtilde_crpa_zeroing(eri, eps, nocc, part)).max() < 1e-8
    assert np.abs(v - vtilde_zero_apb_only(eri, eps, nocc, part)).max() > 1e-4


def test_crpa_from_factors(wat):
    """df=True builds (A+B)_rr and (tu|ia) from the three-index factors; on
    this non-DF RHF they are the exact eigendecomposition of the MO
    integrals, so the oracle on the exact integrals holds to 1e-8. Measured
    max deviation 1.5e-15."""
    mf, eri, eps, nocc = wat
    low = run_low_level(mf, method='rpa', df=True)
    part = partition(_non_contiguous(eps, nocc), eps)
    v = rpa.screening_correction(low, part, 'crpa')
    assert np.abs(v - vtilde_crpa(eri, eps, nocc, part)).max() < 1e-8


def test_crpa_reduces_the_diagonal(wat, low_exact):
    """v_tu,tu = -4 x^T [(delta + 4K)_rr]^-1 x <= 0 with x_jb = (jb|tu): the
    screening only weakens the bare interaction."""
    _, _, eps, nocc = wat
    part = partition(_contiguous_44(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    diag = np.einsum('tutu->tu', v)
    assert diag.max() <= 0.0 and diag.min() < -1e-2


@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_crpa_symmetry(wat, low_exact, aspace):
    """(tu|vw) = (ut|vw) = (vw|tu) for real orbitals."""
    _, _, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    assert np.abs(v - v.transpose(1, 0, 2, 3)).max() < 1e-10
    assert np.abs(v - v.transpose(2, 3, 0, 1)).max() < 1e-10


def test_crpa_ph_is_the_masked_crpa(wat, low_exact):
    """crpa_ph keeps the cRPA correction on the particle-hole elements and
    nothing elsewhere, where cRPA itself is not zero."""
    _, _, eps, nocc = wat
    part = partition(_non_contiguous(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    v_ph = rpa.screening_correction(low_exact, part, 'crpa_ph')
    mask = ph_element_mask(part)
    assert np.array_equal(v_ph, v * mask)
    assert np.abs(v[~mask]).max() > 1e-3


def test_bare_is_zero(wat, low_exact):
    _, _, eps, nocc = wat
    part = partition(_non_contiguous(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'Bare')
    assert v.shape == (4,) * 4 and not np.any(v)


def test_all_active_gives_zero(monkeypatch):
    """H2/6-31G with every orbital active: no pair is left to screen, so the
    correction is zero for every screening, without a solve."""
    mf = h2()
    low = run_low_level(mf, method='rpa', df=False)
    part = partition(ActiveSpace(4, 1, [0], [1, 2, 3]), mf.mo_energy)

    def no_solve(*args, **kwargs):
        raise AssertionError('a solve on an empty residual space')
    monkeypatch.setattr(rpa, 'cho_factor', no_solve)
    for screening in ('bare', 'crpa', 'crpa_ph'):
        v = rpa.screening_correction(low, part, screening)
        assert v.shape == (4,) * 4 and not np.any(v)


@pytest.mark.parametrize('occ, virt', [([3, 4], []), ([], [5, 6])],
                         ids=['no_active_virtual', 'no_active_occupied'])
def test_no_active_virtual(wat, low_exact, occ, virt):
    """No active virtual (or no active occupied) orbital: no pair has both
    ends active, nothing is deleted, and the correction is the fully
    screened W(0) - v on those orbitals (oracle pole sum over every pair).
    Measured max deviation 2.4e-15 / 3.6e-16."""
    mf, eri, eps, nocc = wat
    part = partition(ActiveSpace(len(eps), nocc, occ, virt), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    assert np.abs(v - w0_minus_v(eri, eps, nocc, part.act)).max() < 1e-8


def test_screening_refusals(wat, low_exact):
    _, _, eps, nocc = wat
    part = partition(_contiguous_44(eps, nocc), eps)
    with pytest.raises(ValueError, match="'crpa'"):
        rpa.screening_correction(low_exact, part, 'rpa')


@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_crpa_is_the_spin_orbital_crpa(wat, low_exact, aspace):
    """Ruling R12: the restricted -4 on delta + 4K against the cRPA written
    in spin orbitals (both spins, the active pairs of both deleted,
    -2 V (A+B)_so,rr^-1 V^T), its alpha-alpha block. Measured max deviation
    (contiguous / non-contiguous): 2.6e-16 / 1.9e-16."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'crpa')
    assert np.abs(v - vtilde_crpa_so(eri, eps, nocc, part)).max() < 1e-8


def _v_eff(eri, part, screening, low):
    a = part.act
    return (eri[np.ix_(a, a, a, a)]
            + rpa.screening_correction(low, part, screening))


def _active_w_static(v_eff, eps, part):
    """W^A[v_eff](0) of the oracle: the restricted direct RPA of v_eff alone,
    on the active orbitals (occupied first) and their orbital energies."""
    n = part.n_act
    return v_eff + w0_minus_v(v_eff, eps[part.act], part.n_act_occ,
                              np.arange(n))


@pytest.mark.parametrize('screening', ['bare', 'crpa', 'crpa_ph'])
@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_active_correlation_energy(wat, low_exact, aspace, screening):
    """E_c^A[v_eff], the dense dRPA of the active ph pairs with v_eff (exact
    v + vtilde here, df=False), against the oracle's dRPA energy of v_eff on
    the active orbitals. Measured deviation 0.0 for every case listed (both
    run the same eigensolve of delta^1/2 (delta + 4K) delta^1/2)."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v_eff = _v_eff(eri, part, screening, low_exact)
    e = rpa.active_correlation_energy(low_exact, part, v_eff)
    ref = rpa_energy(v_eff, eps[part.act], part.n_act_occ)
    assert e < 0.0 and abs(e - ref) < 1e-8


@pytest.mark.parametrize('screening', ['bare', 'crpa', 'crpa_ph'])
@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_static_defect(wat, low_exact, aspace, screening):
    """max |W(0) - W^A[v_eff](0)| over the active (pq|rs), against the
    oracle's full W(0) (`w_static_active`) and its dRPA of v_eff alone. For
    crpa it is zero: W^A[W^r](0) = W(0) exactly. Measured defect
    (contiguous / non-contiguous): crpa 2.9e-15 / 2.0e-15, bare 0.143 /
    0.163, crpa_ph 0.142 / 0.163; deviation from the oracle at most 1.1e-15."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v_eff = _v_eff(eri, part, screening, low_exact)
    d = rpa.static_defect(low_exact, part, v_eff)
    ref = np.abs(w_static_active(eri, eps, nocc, part)
                 - _active_w_static(v_eff, eps, part)).max()
    assert abs(d - ref) < 1e-8
    if screening == 'crpa':
        assert d < 1e-8
    else:
        assert d > 1e-4


@pytest.mark.parametrize('occ, virt', [([3, 4], []), ([], [5, 6])],
                         ids=['no_active_virtual', 'no_active_occupied'])
def test_no_active_pair_correlation_and_defect(wat, low_exact, occ, virt):
    """No active ph pair: E_c^A = 0 and W^A = v_eff, so the crpa defect is
    |W(0) - v - vtilde| with vtilde the fully screened W(0) - v: zero."""
    mf, eri, eps, nocc = wat
    part = partition(ActiveSpace(len(eps), nocc, occ, virt), eps)
    v_eff = _v_eff(eri, part, 'crpa', low_exact)
    assert rpa.active_correlation_energy(low_exact, part, v_eff) == 0.0
    assert rpa.static_defect(low_exact, part, v_eff) < 1e-8


# --- mRPA (spec §3.3, §3.5; Scott and Booth, PRL 132, 076401, Eq. 5) --------

@pytest.mark.parametrize('df', [False, True])
@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_mrpa_against_the_spin_orbital_oracle(wat, aspace, df):
    """The restricted mRPA, (ia|jb)_eff = 1/4 [eta0_S^-1 Delta_A eta0_S^-1 -
    Delta_A] with eta0_S the active block of xpy xpy^T (factor 1), against
    Eq. 5 in spin orbitals on the spin-blocked ph space, from the oracle's own
    solve of every singlet and triplet mode: its alpha-alpha block, every
    element. df=True (the exact eigendecomposition on this non-DF RHF) also.
    Measured max deviation (contiguous / non-contiguous): df=False 3.5e-14 /
    1.8e-13, df=True 2.6e-14 / 4.8e-14; max |vt| 7.1e-3 / 6.7e-3. Factors
    1/2 or 2 on eta0_S miss by 0.18 to 1.5.
    """
    mf, eri, eps, nocc = wat
    low = run_low_level(mf, method='rpa', df=df)
    part = partition(aspace(eps, nocc), eps)
    v = rpa.screening_correction(low, part, 'mrpa')
    aa, _ = vtilde_mrpa(eri, eps, nocc, part)
    assert v.shape == (part.n_act,) * 4
    assert np.abs(v - aa).max() < 1e-8
    assert np.abs(aa).max() > 1e-3


@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_mrpa_is_spin_independent(wat, aspace):
    """Spec §3.5: in spin orbitals the same-spin and the opposite-spin blocks
    of the mRPA correction are equal, i.e. the triplet-type channel (alpha
    minus beta) gives no correction; that is what lets the restricted form
    carry one interaction for every spin block. Measured 3.9e-14 / 3.7e-14
    (contiguous / non-contiguous)."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    aa, ab = vtilde_mrpa(eri, eps, nocc, part)
    assert np.abs(aa - ab).max() < 1e-10


def test_mrpa_only_touches_ph_elements(wat, low_exact):
    """mRPA changes the particle-hole elements alone, (ia|jb), (ai|jb),
    (ia|bj) and (ai|bj) all equal, symmetric as a real chemist integral, and
    is exactly zero on every other element. Measured symmetry deviation
    4.7e-17, max |vt| 6.7e-3."""
    _, _, eps, nocc = wat
    part = partition(_non_contiguous(eps, nocc), eps)
    v = rpa.screening_correction(low_exact, part, 'mrpa')
    mask = ph_element_mask(part)
    assert not np.any(v[~mask])
    assert np.abs(v[mask]).max() > 1e-3
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.abs(v - v.transpose(perm)).max() < 1e-12


@pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                         ids=['contiguous_44', 'non_contiguous'])
def test_active_rpa_with_mrpa_reproduces_the_moments(wat, low_exact, aspace):
    """The point of mRPA: the dRPA of the active ph pairs alone with v_eff =
    v + vtilde has the zeroth moment eta0 = (X+Y)(X+Y)^T of the full RPA
    projected on those pairs (and eta1 = Delta_A by construction). Both sides
    from the oracle's `drpa`, the full one on every pair, the active one on
    v_eff and the active orbital energies. Measured (contiguous /
    non-contiguous): 4.8e-14 / 2.1e-13; the bare v misses by 1.6e-2 /
    6.1e-3."""
    mf, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    v_eff = _v_eff(eri, part, 'mrpa', low_exact)
    _, xpy = drpa(eri, eps, nocc)
    mask = active_pair_mask(part)
    eta0 = (xpy @ xpy.T)[np.ix_(mask, mask)]
    _, xpy_a = drpa(v_eff, eps[part.act], part.n_act_occ)
    assert np.abs(xpy_a @ xpy_a.T - eta0).max() < 1e-8
    # the bare v does not: the moments are what the correction is for
    _, xpy_v = drpa(_v_eff(eri, part, 'bare', low_exact), eps[part.act],
                    part.n_act_occ)
    assert np.abs(xpy_v @ xpy_v.T - eta0).max() > 1e-4


def test_all_active_mrpa_is_bare():
    """H2/6-31G with every orbital active: eta0 is the full moment, and
    eta0^-1 Delta eta0^-1 = Delta + 4K gives back the bare (ia|jb), so the
    correction is zero to rounding (computed, not short-cut). Measured
    2.4e-16."""
    mf = h2()
    low = run_low_level(mf, method='rpa', df=False)
    part = partition(ActiveSpace(4, 1, [0], [1, 2, 3]), mf.mo_energy)
    v = rpa.screening_correction(low, part, 'mrpa')
    assert v.shape == (4,) * 4 and np.abs(v).max() < 1e-9


@pytest.fixture(scope='module')
def wat_df():
    """(mf, low, aspace): water/6-31G RHF, its default casida RPA (df=True)
    and the contiguous (4,4) active space."""
    mf = water()
    return mf, run_low_level(mf), ActiveSpace.from_counts(mf.mo_energy, 5,
                                                          2, 2)


def test_reference_determinant_energy_mrpa(wat_df):
    """As for the other screenings: the reference determinant of H_eff is
    E_HF + E_c[v] - E_c^A[v_eff]; vtilde has no (kk|ll) or (kl|lk) element,
    so E_HF,ee[vtilde] = 0. Measured 5.7e-14 (tolerance 1e-9)."""
    mf, low, aspace = wat_df
    g, _, en = build_effective_hamiltonian(low, aspace, screening='mrpa')
    assert abs(reference_energy(g, 2)
               - (mf.e_tot + en.e_c_full - en.e_c_active)) < 1e-9
    assert en.e_hf_ee_vtilde == 0.0
    assert en.e_c_active < 0.0 and en.e_c_full < en.e_c_active


@pytest.mark.parametrize('sigma', ['srg', 'mode_a'])
def test_sigma_with_mrpa(wat_df, sigma):
    """mRPA with Sigma^E: vtilde has no (kl|lk) element, so the exchange
    energy term is exactly 0, and switching Sigma^E on keeps the energy of
    the reference determinant. Measured 0.0 for both forms (tolerance
    1e-9); Sigma_trace -0.0777."""
    mf, low, aspace = wat_df
    g_plain, _, _ = build_effective_hamiltonian(low, aspace, screening='mrpa')
    g, _, en = build_effective_hamiltonian(low, aspace, screening='mrpa',
                                           sigma=sigma)
    assert en.e_exchange_term == 0.0 and abs(en.e_sigma_trace) > 1e-3
    assert abs(reference_energy(g, 2) - reference_energy(g_plain, 2)) < 1e-9
    assert np.array_equal(g[2], g_plain[2])


def test_one_restricted_channel_of_the_two_body_term():
    """A restricted reference passes one channel: `active_eri` returns one
    block from one integral transformation, and `pack` of the one channel
    is `pack` of two identical channels, spin free and in spin orbitals."""
    mf = water()
    idx = [2, 4, 5, 6]
    act = mf.mo_coeff[:, idx]
    one = active_eri(mf.mol, (act,))
    assert len(one) == 1
    assert np.array_equal(one[0], active_eri(mf.mol, (act, act))[0])
    rng = np.random.default_rng(5)
    h = rng.normal(size=(4, 4))
    h = h + h.T
    for spin_free in (True, False):
        g_one = pack((h,), one, 1.5, spin_free=spin_free)
        g_two = pack((h, h), one * 3, 1.5, spin_free=spin_free)
        assert g_one.keys() == g_two.keys() and g_one[0] == 1.5
        assert all(np.array_equal(g_one[k], g_two[k]) for k in (1, 2))
    g = pack((h,), one, 0.0, spin_free=True)
    g[1][0, 0] = 5.0
    assert h[0, 0] != 5.0
    empty = active_eri(mf.mol, (mf.mo_coeff[:, []],))
    assert [b.shape for b in empty] == [(0, 0, 0, 0)]
