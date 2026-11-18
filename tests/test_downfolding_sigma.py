"""The static embedding self-energy Sigma_bar^E on the casida route.

Sigma_bar^E is the static reduction of iG^E(W - v) on the active orbitals,
the internal orbital r running over the environment (spec §3.6): 'srg' is
Eq. 44 of Marie and Loos (s = None: Eq. 42), 'mode_a' their Eq. 12 at
eta = 0, both at the mean-field energies.

The residues of W (`screened_residues`) carry the restricted factor sqrt(2):
with r over every orbital, 'srg' is the active block of the qsGW SRG kernel
(`SelfEnergySolver.static_self_energy_matrix`), which pins that factor to
the existing code. On the environment both reductions equal the numpy
oracle's (its own dRPA residues), on a contiguous and a non-contiguous
space; the s = inf diagonal of 'srg' is that of 'mode_a'; Eq. 42 is the real
part of the exact pole sum S^E at one complex frequency per element; 'srg'
at finite s stays within its damping bound of s = inf. 'mode_a' refuses an
active space whose orbital energy leaves the pole-free window (or comes
within the margin of its edge), where 'srg' stays finite and equal to the
oracle. On the prototype's own orbitals 'mode_a' is its Sigma_bar. The
`downfolding` numerics section holds srg_s and pole_margin.
"""
import json
import pathlib

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import (DOWNFOLDING_POLE_MARGIN,
                                         DOWNFOLDING_SRG_S)
from src.Base.numerics.numerics import DownfoldingNumerics, Numerics
from src.MultiReference.Downfolding import run_low_level
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import partition
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from tests.downfolding_oracle import (env_residues, mo_integrals,
                                      sigma_env, sigma_mode_a, sigma_srg,
                                      water)

PINS = json.loads((pathlib.Path(__file__).parent / 'data'
                   / 'downfolding_prototype_pins.json').read_text())['cases']


@pytest.fixture(scope='module')
def wat():
    """(mf, eri, eps, nocc) of water/6-31G, eri the exact chemist MO integrals."""
    mf = water()
    _, eri = mo_integrals(mf)
    return mf, eri, mf.mo_energy, int(np.count_nonzero(mf.mo_occ))


@pytest.fixture(scope='module')
def low_exact(wat):
    """The casida RPALowLevel of water/6-31G on the exact integrals."""
    return run_low_level(wat[0], method='rpa', df=False)


@pytest.fixture(scope='module')
def low_infinite_s(wat):
    """The same, with srg_s = None (s = inf)."""
    return run_low_level(wat[0], method='rpa', df=False,
                         numerics=Numerics(downfolding={'srg_s': None}))


def _contiguous_44(eps, nocc):
    return ActiveSpace.from_counts(eps, nocc, 2, 2)


def _non_contiguous(eps, nocc):
    """Occupied 2 and 4 around the inactive 3; the virtuals stay the lowest
    two, so every active energy is inside the pole-free window."""
    return ActiveSpace(len(eps), nocc, [2, 4], [5, 6])


def _high_virtual(eps, nocc):
    """The virtual 7 active, the lower virtual 6 in the environment: eps_7 =
    1.057 lies beyond eps_6 + Omega_min = 1.028, the window's upper edge."""
    return ActiveSpace(len(eps), nocc, [3, 4], [5, 7])


SPACES = pytest.mark.parametrize('aspace', [_contiguous_44, _non_contiguous],
                                 ids=['contiguous_44', 'non_contiguous'])


def _pure_args(low, part):
    """(w, eps_t, eps_r, occupied_r, omega) of the environment of `part`."""
    env = part.env
    return (rpa.screened_residues(low, part), low.eps[part.act],
            low.eps[env], env < low.nocc, low.omega)


def test_srg_matches_the_qsgw_kernel(wat):
    """With r over every orbital and the DF factors, srg_self_energy at
    s = 100 is the active block of the qsGW SRG kernel, built from its own
    Casida solve and transition density, and mode_a_self_energy is the
    kernel's mode A at eta = 1e-12. This pins the restricted factor of
    `screened_residues`: sqrt(2) on sum_ia (tr|ia) (X+Y)_ia, the kernel's
    2 sum chi chi (measured element ratio 1 - 3.7e-9 to 1 + 2.7e-9). The
    SRG deviation is the kernel's Laplace quadrature (relative 1e-7 per
    term, QSGW_SRG_QUAD_TOL, the tightest its rule reaches here): measured
    1.9e-10, mode A 3.1e-17 (tolerance 1e-8)."""
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', df=True)
    part = partition(_contiguous_44(eps, nocc), eps)
    nmo = len(eps)
    every = np.arange(nmo)
    w = rpa.screened_residues(low, part, orbitals=every)
    assert w.shape == (len(low.omega), part.n_act, nmo)
    ours = rpa.srg_self_energy(w, eps[part.act], eps, every < nocc,
                               low.omega, 100.0)
    lr = LinearResponseSolver(eps, coeff_df=low.factors,
                              spin_mode='restricted')
    omega, x, y = CasidaSolver(*lr.build_casida_matrices(nocc,
                                                         lBSE=False)).solve()
    se = SelfEnergySolver(eps, df_coeff=low.factors, spin_mode='restricted',
                          eta=1e-12)
    rho = se.rho_a_df(nocc, x, y)
    block = np.ix_(part.act, part.act)
    ref = se.static_self_energy_matrix(nocc, omega, rho, flow=100.0)[block]
    assert np.abs(ours - ref).max() < 1e-8
    assert np.abs(ref).max() > 1e-2
    mode_a = rpa.mode_a_self_energy(w, eps[part.act], eps, every < nocc,
                                    low.omega)
    ref = se.static_self_energy_matrix(nocc, omega, rho)[block]
    assert np.abs(mode_a - ref).max() < 1e-8


@SPACES
def test_srg_and_mode_a_against_the_oracle(wat, low_exact, aspace):
    """embedding_self_energy 'srg' (s = DOWNFOLDING_SRG_S, the default) and
    'mode_a' against the oracle's Eq. 44 and Eq. 12 on its own residues,
    both symmetric. Measured max deviation (contiguous / non-contiguous):
    srg 1.5e-16 / 3.0e-16, mode_a 1.5e-16 / 3.0e-16 (tolerance 1e-8)."""
    _, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    srg = rpa.embedding_self_energy(low_exact, part, 'srg')
    mode_a = rpa.embedding_self_energy(low_exact, part, 'mode_a')
    for ours, ref in ((srg, sigma_srg(eri, eps, nocc, part, DOWNFOLDING_SRG_S)),
                      (mode_a, sigma_mode_a(eri, eps, nocc, part))):
        assert ours.shape == (part.n_act, part.n_act)
        assert np.array_equal(ours, ours.T)
        assert np.abs(ours - ref).max() < 1e-8
        assert np.abs(ref).max() > 1e-3


@SPACES
def test_diagonals_agree_at_infinite_s(low_infinite_s, wat, aspace):
    """srg_s = None: (a + b)/(a^2 + b^2) at a = b is 1/a, so the diagonal of
    'srg' is that of 'mode_a' (spec §3.6); off the diagonal they differ
    (by up to 1.9e-4 / 1.7e-4). Measured max diagonal deviation 2.2e-19 /
    1.7e-18 (tolerance 1e-10)."""
    _, _, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    srg = rpa.embedding_self_energy(low_infinite_s, part, 'srg')
    mode_a = rpa.embedding_self_energy(low_infinite_s, part, 'mode_a')
    assert np.abs(np.diag(srg) - np.diag(mode_a)).max() < 1e-10
    assert np.abs(srg - mode_a).max() > 1e-6


@SPACES
def test_eq42_is_one_complex_frequency(wat, low_infinite_s, aspace):
    """srg_s = None: Sigma_bar_tu = Re S^E_tu(z_tu), z_tu = (eps_t + eps_u)/2
    + i (eps_u - eps_t)/2, S^E the oracle's exact pole sum, since
    (a + b)/(a^2 + b^2) = Re[(1 + i)/(a + i b)]. Measured max deviation
    1.5e-16 / 3.0e-16 (tolerance 1e-10)."""
    _, eri, eps, nocc = wat
    part = partition(aspace(eps, nocc), eps)
    srg = rpa.embedding_self_energy(low_infinite_s, part, 'srg')
    e = eps[part.act]
    ref = np.array([[sigma_env(0.5 * (e[t] + e[u]) + 0.5j * (e[u] - e[t]),
                               eri, eps, nocc, part)[t, u].real
                     for u in range(part.n_act)] for t in range(part.n_act)])
    assert np.abs(srg - ref).max() < 1e-10


@pytest.mark.parametrize('space', [_contiguous_44, _non_contiguous,
                                   _high_virtual],
                         ids=['contiguous_44', 'non_contiguous',
                              'high_virtual'])
@pytest.mark.parametrize('s', [DOWNFOLDING_SRG_S, 2.0])
def test_srg_equals_infinite_s_away_from_poles(wat, low_exact, space, s):
    """|srg(s) - srg(inf)| <= C exp(-2 s d^2), C = sum of |terms| of Eq. 42
    and d = min |D_tr| over the terms (each term is damped by
    exp(-(D_tr^2 + D_ur^2) s) <= exp(-2 s d^2)); d is at least the active
    orbitals' distance to the pole-free window where they are inside it.
    At s = 500 the bound is below rounding on the two spaces inside the
    window (d = 0.88 / 0.58, the difference exactly 0) and 0.41 C on the
    high virtual (d = 0.030); s = 2 makes it bite on all three. Measured
    ratio |difference| / bound at most 0.045."""
    _, eri, eps, nocc = wat
    part = partition(space(eps, nocc), eps)
    w, e_t, e_r, occ_r, omega = _pure_args(low_exact, part)
    diff = np.abs(rpa.srg_self_energy(w, e_t, e_r, occ_r, omega, s)
                  - rpa.srg_self_energy(w, e_t, e_r, occ_r, omega, None))
    _, w_ref, c = env_residues(eri, eps, nocc, part)
    d_all = eps[part.act][None, :, None] - c[:, None, :]       # (n, t, r)
    d = np.abs(d_all).min()
    lo, hi = rpa.pole_window(low_exact, part)
    inside = (e_t > lo) & (e_t < hi)
    if inside.all():
        assert d >= np.minimum(e_t - lo, hi - e_t).min() - 1e-12
    a, b = d_all[:, :, None, :], d_all[:, None, :, :]
    terms = (np.abs((a + b) / (a**2 + b**2))
             * np.abs(w_ref[:, :, None, :] * w_ref[:, None, :, :]))
    bound = terms.sum(axis=(0, 3)) * np.exp(-2.0 * s * d * d)
    assert np.all(diff <= bound + 1e-12)


def test_pole_window(wat, low_exact):
    """(max eps_I - Omega_min, min eps_Ev + Omega_min), an empty side
    infinite: every orbital active leaves no environment, the window is the
    whole axis and Sigma_bar^E is zero."""
    _, _, eps, nocc = wat
    part = partition(_contiguous_44(eps, nocc), eps)
    lo, hi = rpa.pole_window(low_exact, part)
    om = low_exact.omega.min()
    assert lo == eps[part.inactive].max() - om
    assert hi == eps[part.env_virt].min() + om
    every = partition(ActiveSpace.from_counts(eps, nocc, nocc,
                                              len(eps) - nocc), eps)
    assert rpa.pole_window(low_exact, every) == (-np.inf, np.inf)
    for sigma in ('srg', 'mode_a'):
        out = rpa.embedding_self_energy(low_exact, every, sigma)
        assert out.shape == (len(eps),) * 2 and not out.any()


def test_near_pole_space(wat, low_exact):
    """The virtual 7 made active over the environment virtual 6 lies beyond
    the window's upper edge (asserted here before the refusal): 'mode_a'
    refuses, naming orbital 7 and its distance; 'srg' is finite and equal to
    the oracle (measured 2.7e-16, tolerance 1e-8), and its damping matters
    there (|srg(500) - srg(inf)| = 1.5e-4, the nearest pole 0.030 Ha away)."""
    _, eri, eps, nocc = wat
    part = partition(_high_virtual(eps, nocc), eps)
    lo, hi = rpa.pole_window(low_exact, part)
    margin = DOWNFOLDING_POLE_MARGIN / np.sqrt(2.0 * DOWNFOLDING_SRG_S)
    assert eps[7] > hi - margin and eps[5] < hi - margin
    with pytest.raises(ValueError, match=r'orbital 7 \(') as err:
        rpa.embedding_self_energy(low_exact, part, 'mode_a')
    assert 'orbital 5 ' not in str(err.value)
    srg = rpa.embedding_self_energy(low_exact, part, 'srg')
    assert np.all(np.isfinite(srg))
    assert np.abs(srg - sigma_srg(eri, eps, nocc, part,
                                  DOWNFOLDING_SRG_S)).max() < 1e-8
    assert np.abs(srg - sigma_srg(eri, eps, nocc, part, None)).max() > 5e-5


def test_mode_a_margin_uses_srg_s(wat):
    """The margin is pole_margin / sqrt(2 s), with the default s when
    srg_s = None: a margin wide enough to reach the active LUMO refuses it;
    the default does not; an unknown sigma is refused."""
    mf, _, eps, nocc = wat
    part = partition(_contiguous_44(eps, nocc), eps)
    _, hi = rpa.pole_window(run_low_level(mf, method='rpa', df=False), part)
    dist = hi - eps[6]
    for srg_s in (None, 2.0):
        s = DOWNFOLDING_SRG_S if srg_s is None else srg_s
        wide = 1.01 * dist * np.sqrt(2.0 * s)
        low = run_low_level(mf, method='rpa', df=False, numerics=Numerics(
            downfolding={'srg_s': srg_s, 'pole_margin': wide}))
        with pytest.raises(ValueError, match=r'orbital 6 \('):
            rpa.embedding_self_energy(low, part, 'mode_a')
        assert np.all(np.isfinite(rpa.embedding_self_energy(low, part,
                                                            'srg')))
    with pytest.raises(ValueError, match='sigma'):
        rpa.embedding_self_energy(run_low_level(mf, method='rpa', df=False),
                                  part, 'mode_b')


def _pinned(molecule):
    """(low, part) of a prototype pin: an RHF on the pinned geometry and
    basis carrying the prototype's mo_coeff, mo_energy and mo_occ (no SCF),
    its casida RPA on exact integrals, and the pinned active orbitals (as in
    tests/test_downfolding_reference.py)."""
    c = PINS[molecule]
    mol = gto.M(atom=[(a[0], a[1:]) for a in c['atoms']], basis=c['basis'],
                unit='A', charge=c['charge'], spin=c['spin'], cart=c['cart'],
                verbose=0)
    mf = scf.RHF(mol)
    mf.mo_coeff = np.array(c['mo_coeff'])
    mf.mo_energy = np.array(c['mo_energy'])
    mf.mo_occ = np.where(np.arange(len(mf.mo_energy)) < c['nocc'], 2.0, 0.0)
    low = run_low_level(mf, method='rpa', mode='casida', df=False)
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), c['nocc'],
                                      np.array(c['act_occ']),
                                      np.array(c['act_virt']))
    return low, partition(aspace, mf.mo_energy)


@pytest.mark.parametrize('case', sorted(PINS))
def test_prototype_sigma_bar(case):
    """'mode_a' on the prototype's own orbitals is its Sigma_bar (1/2 [S(e_t)
    + S(e_u)], environment sum, full W), evaluated at its sigma_energies,
    which are the mean-field active energies (both well inside the
    pole-free window). Measured max deviation ethylene 5.2e-17, benzene
    2.4e-16 (tolerance 1e-7)."""
    c = PINS[case]
    low, part = _pinned(case)
    assert np.abs(low.eps[part.act] - np.array(c['sigma_energies'])).max() \
        < 1e-12
    ours = rpa.embedding_self_energy(low, part, 'mode_a')
    assert np.abs(ours - np.array(c['sigma_bar'])).max() < 1e-7


def test_numerics_section():
    """Numerics.downfolding: the defaults from constants, a dict built into
    the section, srg_s = None allowed (s = inf), srg_s <= 0 and
    pole_margin < 0 refused."""
    assert Numerics().downfolding == DownfoldingNumerics(
        srg_s=DOWNFOLDING_SRG_S, pole_margin=DOWNFOLDING_POLE_MARGIN)
    assert Numerics(downfolding={'srg_s': 250.0}).downfolding.srg_s == 250.0
    assert Numerics(downfolding={'srg_s': None}).downfolding.srg_s is None
    assert DownfoldingNumerics(pole_margin=0.0).pole_margin == 0.0
    for bad in ({'srg_s': -1.0}, {'srg_s': 0.0}, {'pole_margin': -1.0},
                {'srg_s': float('inf')}, {'srg_s': float('nan')}):
        with pytest.raises(ValueError):
            Numerics(downfolding=bad)
