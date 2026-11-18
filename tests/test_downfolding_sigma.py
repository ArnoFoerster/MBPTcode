"""The static embedding self-energy Sigma_bar^E on the casida, the
imagfrequency and the space-time route.

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
`downfolding` numerics section holds srg_s, pole_margin and
continuation_margin.

A term of Eq. 44 exactly at a pole is 0, not NaN (ruling R17). The
'sigma_poles' diagnostic of `terms` (window, pole distances, Z_t against
the oracle's S^E, the s applied, the SRG damping) warns on the near-pole
space only; along a short scan through a pole crossing 'srg' stays as
smooth as on a space far from the poles, where Eq. 42 jumps.

On imagfrequency S^E(mu + i w) comes from the Lorentzian fit of W and is
continued by Pade to one complex frequency per element. The Pade error
alone (the exact casida S^E through the continuation, water and benzene
(6,6)) is far below the 1e-4 Eh target; the imaginary-axis S^E matches the
casida pole sum; Omega_min is the lowest RPA root; the near-pole space is
refused for both forms, pointing to casida; the 'sigma_continuation'
diagnostic is present and warns on a too-coarse node set; on benzene the
route matches casida.

On space-time S^E(mu + i w) is the space-time GW kernel with the
environment's G: plus the same call with the active orbitals' G it is the GW
route's full Sigma_c on the active block, and Omega_min (dense or Davidson,
ruling R21) is the lowest root on the ISDF v. Its agreement with casida is
in tests/test_downfolding_routes.py.
"""
import json
import pathlib
import re
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import (DOWNFOLDING_CONTINUATION_MARGIN,
                                         DOWNFOLDING_POLE_MARGIN,
                                         DOWNFOLDING_SRG_S)
from src.Base.numerics.numerics import (DownfoldingNumerics, Numerics,
                                        numerics_or_default)
from src.Base.orbital_spaces import particle_hole_window
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.MultiReference.Downfolding.methods import rpa
from src.MultiReference.Downfolding.partition import partition
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.GW.imaginary_time import \
    self_energy_matrix_imaginary_time
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.space_time import self_energy_matrix_space_time
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from tests.downfolding_oracle import (env_residues, mo_integrals,
                                      sigma_env, sigma_mode_a, sigma_srg,
                                      water)
from tests.gw_density_routes import mean_field as nh3_mean_field
from tests.isdf_test_grid import g1_numerics

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


def _pinned_mean_field(molecule, density_fit=False):
    """(mf, part) of a prototype pin: an RHF on the pinned geometry and basis
    (density fitted if asked) carrying the prototype's mo_coeff, mo_energy
    and mo_occ (no SCF), and the partition of the pinned active orbitals (as
    in tests/test_downfolding_reference.py)."""
    c = PINS[molecule]
    mol = gto.M(atom=[(a[0], a[1:]) for a in c['atoms']], basis=c['basis'],
                unit='A', charge=c['charge'], spin=c['spin'], cart=c['cart'],
                verbose=0)
    mf = scf.RHF(mol).density_fit() if density_fit else scf.RHF(mol)
    mf.mo_coeff = np.array(c['mo_coeff'])
    mf.mo_energy = np.array(c['mo_energy'])
    mf.mo_occ = np.where(np.arange(len(mf.mo_energy)) < c['nocc'], 2.0, 0.0)
    aspace = ActiveSpace.from_indices(len(mf.mo_energy), c['nocc'],
                                      np.array(c['act_occ']),
                                      np.array(c['act_virt']))
    return mf, partition(aspace, mf.mo_energy)


def _pinned(molecule):
    """(low, part) of a prototype pin (`_pinned_mean_field`): its casida
    RPA on exact integrals."""
    mf, part = _pinned_mean_field(molecule)
    return run_low_level(mf, method='rpa', mode='casida', df=False), part


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
    the section, srg_s = None allowed (s = inf), srg_s <= 0, pole_margin < 0
    and continuation_margin < 0 refused."""
    assert Numerics().downfolding == DownfoldingNumerics(
        srg_s=DOWNFOLDING_SRG_S, pole_margin=DOWNFOLDING_POLE_MARGIN,
        continuation_margin=DOWNFOLDING_CONTINUATION_MARGIN)
    assert DownfoldingNumerics(
        continuation_margin=0.0).continuation_margin == 0.0
    assert Numerics(downfolding={'srg_s': 250.0}).downfolding.srg_s == 250.0
    assert Numerics(downfolding={'srg_s': None}).downfolding.srg_s is None
    assert DownfoldingNumerics(pole_margin=0.0).pole_margin == 0.0
    for bad in ({'srg_s': -1.0}, {'srg_s': 0.0}, {'pole_margin': -1.0},
                {'srg_s': float('inf')}, {'srg_s': float('nan')},
                {'continuation_margin': -0.1}):
        with pytest.raises(ValueError):
            Numerics(downfolding=bad)


def test_srg_term_at_an_exact_pole_is_zero():
    """Ruling R17: a term with D_tr = D_ur = 0 exactly is 0/0 in Eq. 44; its
    limit at finite s is 0 (the kernel is about 2 D s near D = 0), and that
    is what it gives, without a NaN or a floating-point warning. The other
    term is untouched. Energies chosen exactly representable: D = 1 - 1/2 -
    1/2 = 0 for r = 0, D = 1 - 1/4 - 1/2 = 1/4 for r = 1."""
    w = np.array([[[0.3, 0.2]]])
    args = (np.array([1.0]), np.array([0.5, 0.25]), np.array([False, False]),
            np.array([0.5]))
    with np.errstate(all='raise'):
        out = rpa.srg_self_energy(w, *args, 500.0)
        kept = rpa.srg_self_energy(w[:, :, 1:], args[0], args[1][1:],
                                   args[2][1:], args[3], 500.0)
    assert np.all(np.isfinite(out))
    assert out[0, 0] == kept[0, 0] and kept[0, 0] != 0.0


def _poles(low, aspace, sigma='srg', **opts):
    """energies.diagnostics['sigma_poles'] of `aspace` on `low` (crpa)."""
    return build_effective_hamiltonian(low, aspace, sigma=sigma,
                                       **opts)[2].diagnostics['sigma_poles']


@pytest.mark.parametrize('space', [_contiguous_44, _high_virtual],
                         ids=['contiguous_44', 'high_virtual'])
def test_sigma_poles_diagnostic_and_warning(wat, low_exact, space):
    """'sigma_poles': the pole-free window, each active orbital's distance to
    its nearer edge (negative outside), Z_t = 1/(1 - S'_tt(eps_t)) against
    a central difference of the oracle's exact pole sum S^E (h = 1e-5;
    measured 3.1e-13 / 1.1e-9, tolerance 1e-7; Z from 0.963 at orbital 7
    to 0.995), the s applied and the SRG damping max |Sigma_bar(s) -
    Sigma_bar(inf)|. On the contiguous (4,4) the damping is exactly 0 and
    nothing warns; on the near-pole space it is 1.5e-4 and a
    SigmaDampingWarning names orbital 7 alone. 'mode_a' applies no s:
    srg_s None, damping 0. With diagnostics=() there is no record and no
    warning."""
    _, eri, eps, nocc = wat
    aspace = space(eps, nocc)
    part = partition(aspace, eps)
    near = space is _high_virtual
    if near:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always', rpa.SigmaDampingWarning)
            poles = _poles(low_exact, aspace)
        rec = [r for r in rec if r.category is rpa.SigmaDampingWarning]
        assert len(rec) == 1
        assert re.search(r'orbital 7\b', str(rec[0].message))
        assert 'orbital 5' not in str(rec[0].message)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            poles = _poles(low_exact, aspace)
            mode_a = _poles(low_exact, aspace, sigma='mode_a')
        assert mode_a['srg_s'] is None and mode_a['srg_damping'] == 0.0
        assert np.array_equal(mode_a['z'], poles['z'])
    assert set(poles) == {'window', 'pole_distance', 'z', 'srg_s',
                          'srg_damping'}
    lo, hi = rpa.pole_window(low_exact, part)
    e = eps[part.act]
    assert poles['window'] == (lo, hi)
    assert np.array_equal(poles['pole_distance'], np.minimum(e - lo, hi - e))
    assert (poles['pole_distance'] < 0).tolist() == [False, False, False, near]
    assert poles['srg_s'] == DOWNFOLDING_SRG_S
    w, e_t, e_r, occ_r, omega = _pure_args(low_exact, part)
    damping = np.abs(rpa.srg_self_energy(w, e_t, e_r, occ_r, omega,
                                         DOWNFOLDING_SRG_S)
                     - rpa.srg_self_energy(w, e_t, e_r, occ_r, omega,
                                           None)).max()
    assert poles['srg_damping'] == damping
    assert (damping > 1e-6) == near
    h = 1e-5
    slope = np.array([(sigma_env(x + h, eri, eps, nocc, part)[t, t]
                       - sigma_env(x - h, eri, eps, nocc, part)[t, t]).real
                      / (2 * h) for t, x in enumerate(e)])
    assert np.abs(poles['z'] - 1.0 / (1.0 - slope)).max() < 1e-7
    assert np.all((poles['z'] > 0) & (poles['z'] <= 1))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, en = build_effective_hamiltonian(low_exact, aspace,
                                               sigma='srg', diagnostics=())
    assert en.diagnostics == {}


def _stretched_water(r_oh):
    """Converged RHF of water/6-31G with both O-H bonds at r_oh Angstrom,
    the angle and the O position of `water()`."""
    o = np.array([0.0, 0.0, 0.1173])
    hs = np.array([[0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    u = (hs - o) / np.linalg.norm(hs - o, axis=1)[:, None]
    mol = gto.M(atom=[('O', o)] + [('H', o + r_oh * x) for x in u],
                basis='6-31g', cart=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


SCAN = (1.08, 1.09, 1.10, 1.11, 1.12)


def test_srg_is_smooth_through_a_pole_crossing():
    """Along a symmetric O-H stretch of water, 0.01 A steps, the near-pole
    space (occupied 3, 4; virtual 5, 7, with 6 in the environment) crosses a
    pole: eps_7 - eps_6 - Omega_1 changes sign between 1.10 and 1.11 A
    (-1.2e-3 -> +2.7e-3 Ha, residue |w| = 8.5e-3). The active orbitals'
    signs are fixed to the first geometry (orbitals 6 and 7 flip in the
    SCF along the scan). Each element's largest second difference of 'srg'
    (s = 500) stays below 10 times that element's on the plain (4,4) space
    (occupied 3, 4; virtual 5, 6) along the same scan, with a floor of
    1e-12 for the elements that symmetry makes zero (both at 1e-15 noise):
    the largest ratio is 7.0, at the pole's own element (7, 7): 3.9e-5
    against 5.6e-6. The undamped Eq. 42 (s = inf) on the same scan jumps
    through the pole (1.3e-1 on (7, 7)), so the scan has teeth."""
    srg, infinite, plain, gaps = [], [], [], []
    ref = None
    for r_oh in SCAN:
        mf = _stretched_water(r_oh)
        eps, nocc = mf.mo_energy, 5
        act = [3, 4, 5, 6, 7]
        c = mf.mo_coeff
        if ref is None:
            ref = c[:, act]
        sign = np.sign(np.einsum('ap,ab,bp->p', ref, mf.get_ovlp(),
                                 c[:, act]))
        mf.mo_coeff = c.copy()
        mf.mo_coeff[:, act] *= sign
        low = run_low_level(mf, method='rpa', df=False)
        near = partition(_high_virtual(eps, nocc), eps)
        srg.append(rpa.embedding_self_energy(low, near, 'srg'))
        args = _pure_args(low, near)
        infinite.append(rpa.srg_self_energy(*args, None))
        plain.append(rpa.embedding_self_energy(
            low, partition(_contiguous_44(eps, nocc), eps), 'srg'))
        gaps.append(eps[7] - eps[6] - low.omega[1])

    def second(xs):
        """Each element's largest |second difference| along the scan."""
        xs = np.array(xs)
        return np.abs(xs[2:] - 2 * xs[1:-1] + xs[:-2]).max(axis=0)

    assert gaps[2] < 0 < gaps[3]
    assert np.all(np.isfinite(srg))
    baseline = np.maximum(second(plain), 1e-12)
    assert np.all(second(srg) < 10 * baseline)
    assert second(infinite).max() > 1e3 * second(plain).max()


# --- the imagfrequency route: S^E(mu + i w), Pade to z_tu -----------------


def _pole_sum(low, part, z):
    """S^E_tu(z) (nz, n_act, n_act), the exact pole sum on the casida route:
    sum_rn w_tr w_ur / (z - eps_r - sgn(eps_r - eps_F) Omega_n), the
    residues of `screened_residues` (pinned to the oracle above)."""
    env = part.env
    sign = np.where(env < low.nocc, -1.0, 1.0)
    poles = low.eps[env][None, :] + sign[None, :] * low.omega[:, None]
    w = rpa.screened_residues(low, part)
    g = 1.0 / (np.atleast_1d(z)[:, None, None] - poles[None])
    return np.einsum('ntr,nur,qnr->qtu', w, w, g)


def _pade_points(low):
    """(nodes, mu): the Pade nodes of the default numerics and the mid-gap."""
    return (pade_nodes(numerics_or_default(None).quadrature),
            particle_hole_window(low.eps, low.nocc)[2])


def _pade_alone_cases(wat, low_exact):
    _, _, eps, nocc = wat
    for space in (_contiguous_44, _non_contiguous):
        yield low_exact, partition(space(eps, nocc), eps)
    yield _pinned('benzene')


def test_pade_alone(wat, low_exact):
    """The Pade error alone: the exact casida S^E(mu + i w) on the Pade nodes
    (16 Gauss-Legendre points of scale 1 Ha, `pade_nodes`) through
    `continued_self_energy` against the exact Eq. 42 (Re S^E(z_tu)) and
    mode A (S^E on the real axis): water/6-31G contiguous (4,4) and
    non-contiguous, benzene (6,6) pi space cc-pVDZ on the prototype's
    orbitals. Target 1e-4 Eh per element; measured max (srg / mode_a)
    water 4.5e-10 / 4.5e-10 and 2.0e-9 / 2.0e-9, benzene 1.8e-7 / 1.8e-7
    (the largest at a diagonal element, where both forms coincide)."""
    for low, part in _pade_alone_cases(wat, low_exact):
        nodes, mu = _pade_points(low)
        samples = _pole_sum(low, part, mu + 1j * nodes)
        args = _pure_args(low, part)
        e = low.eps[part.act]
        for sigma, exact in (('srg', rpa.srg_self_energy(*args, None)),
                             ('mode_a', rpa.mode_a_self_energy(*args))):
            got = rpa.continued_self_energy(mu + 1j * nodes, samples, e,
                                            sigma)
            assert np.array_equal(got, got.T)
            assert np.abs(exact).max() > 1e-2
            assert np.abs(got - exact).max() < 1e-4


def test_sigma_bar_matches_casida_benzene():
    """Benzene (6,6) pi space cc-pVDZ on the prototype's orbitals, density
    fitted: Sigma_bar^E on imagfrequency against casida on the same
    three-index object at srg_s = None, the case spec §3.6 fixes the route
    tolerance on. Target 1e-4 Eh per element; measured 1.6e-7 for both
    forms (|Sigma_bar| up to 4.5e-2; the Pade error alone is 1.8e-7,
    test_pade_alone), the lowest root (Davidson) 2.1e-14 from casida's."""
    mf, part = _pinned_mean_field('benzene', density_fit=True)
    infinite = Numerics(downfolding={'srg_s': None})
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=infinite)
    low_cas = run_low_level(mf, method='rpa', mode='casida',
                            numerics=infinite)
    assert abs(rpa.lowest_excitation(low_if, solver='davidson')
               - low_cas.omega.min()) < 1e-8
    for sigma in ('srg', 'mode_a'):
        got = rpa.embedding_self_energy(low_if, part, sigma)
        want = rpa.embedding_self_energy(low_cas, part, sigma)
        assert np.abs(want).max() > 1e-2
        assert np.abs(got - want).max() < 1e-4


def test_sigma_env_imag_axis_against_the_pole_sum(wat):
    """sigma_env_imag_axis on imagfrequency against the casida pole sum on the
    same three-index object (water/6-31G, df), on the Pade nodes, with the
    internal orbital over the environment alone: the Lorentzian fit of W on
    the minimax W grid against the exact residues. Measured max deviation
    2.2e-8 / 2.7e-8 (contiguous / non-contiguous; target 1e-5, |S| up to
    2.3e-2). W is cached on the first
    request (low.cache['w_grid']) and reused. A casida result (df or
    exact integrals) is refused, pointing to imagfrequency."""
    mf, _, eps, nocc = wat
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency')
    low_cas = run_low_level(mf, method='rpa', mode='casida')
    assert 'w_grid' not in low_if.cache
    for space in (_contiguous_44, _non_contiguous):
        part = partition(space(eps, nocc), eps)
        nodes, mu = _pade_points(low_if)
        got = rpa.sigma_env_imag_axis(low_if, part, nodes)
        assert got.shape == (len(nodes), part.n_act, part.n_act)
        ref = _pole_sum(low_cas, part, mu + 1j * nodes)
        assert np.abs(got - ref).max() < 1e-5
    w_grid = low_if.cache['w_grid']
    rpa.sigma_env_imag_axis(low_if, part, nodes)
    assert low_if.cache['w_grid'] is w_grid
    for low in (low_cas, run_low_level(mf, method='rpa', df=False)):
        with pytest.raises(NotImplementedError,
                           match="use mode='imagfrequency'"):
            rpa.sigma_env_imag_axis(low, part, nodes)


def test_lowest_excitation(wat):
    """Omega_min on imagfrequency (no eigenvectors kept) is the lowest RPA
    root, by the dense Casida solve when the pair space is small and by the
    Davidson otherwise; both equal the casida route's lowest omega (measured
    0.0 dense, the same solver, and 1.7e-15 Davidson), cached in
    low.cache['omega_min']; so the pole-free windows agree. Without an
    environment the window is the whole axis and no root is solved for."""
    mf, _, eps, nocc = wat
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency')
    low_cas = run_low_level(mf, method='rpa', mode='casida')
    ref = low_cas.omega.min()
    assert rpa.lowest_excitation(low_cas) == ref
    for solver in ('dense', 'davidson'):
        assert abs(rpa.lowest_excitation(low_if, solver=solver) - ref) < 1e-8
    assert 'omega_min' not in low_if.cache
    om = rpa.lowest_excitation(low_if)
    assert low_if.cache['omega_min'] == om
    part = partition(_contiguous_44(eps, nocc), eps)
    assert np.allclose(rpa.pole_window(low_if, part),
                       rpa.pole_window(low_cas, part), atol=1e-8, rtol=0)
    fresh = run_low_level(mf, method='rpa', mode='imagfrequency')
    every = partition(ActiveSpace.from_counts(eps, nocc, nocc,
                                              len(eps) - nocc), eps)
    assert rpa.pole_window(fresh, every) == (-np.inf, np.inf)
    assert 'omega_min' not in fresh.cache


def test_pole_refusal_points_to_casida(wat):
    """On imagfrequency both 'srg' (s = inf there) and 'mode_a' refuse the
    near-pole space of the casida tests (virtual 7 beyond the window's edge),
    naming orbital 7 alone and pointing to mode='casida', where 'srg' is
    finite; the contiguous (4,4) is accepted."""
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    near = partition(_high_virtual(eps, nocc), eps)
    for sigma in ('srg', 'mode_a'):
        with pytest.raises(ValueError, match=r'orbital 7 \(') as err:
            rpa.embedding_self_energy(low, near, sigma)
        assert "mode='casida'" in str(err.value)
        assert 'orbital 5 ' not in str(err.value)
        with pytest.raises(ValueError, match="mode='casida'"):
            build_effective_hamiltonian(low, _high_virtual(eps, nocc),
                                        sigma=sigma)
        ok = rpa.embedding_self_energy(low, partition(
            _contiguous_44(eps, nocc), eps), sigma)
        assert np.all(np.isfinite(ok))


def test_continuation_diagnostic_present(wat):
    """'sigma_continuation' on imagfrequency: per element, |Pade on the even
    nodes - Pade on the odd nodes| at the element's evaluation point, below
    the 1e-4 Eh warning level on the contiguous (4,4) (measured max
    6.7e-6), without a warning; absent on casida. 'sigma_poles' there records
    the window of the lowest root, s = inf (srg_s None, srg_damping 0.0) and
    Z_t from the continued fraction (against casida's, measured 9.1e-9).
    With every orbital active (no environment) Sigma_bar and the spread are
    zero. With npade = 4 the two halves are 2-point fits, the spread exceeds the
    level (measured 1.1e-3) and a SigmaContinuationWarning names the
    elements."""
    mf, _, eps, nocc = wat
    a = _contiguous_44(eps, nocc)
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, en = build_effective_hamiltonian(low, a, sigma='srg')
    spread = en.diagnostics['sigma_continuation']
    assert spread.shape == (4, 4)
    assert np.all(spread >= 0) and spread.max() < 1e-4
    poles = en.diagnostics['sigma_poles']
    _, _, en_c = build_effective_hamiltonian(
        run_low_level(mf, method='rpa', mode='casida'), a, sigma='srg')
    assert 'sigma_continuation' not in en_c.diagnostics
    ref = en_c.diagnostics['sigma_poles']
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    assert np.allclose(poles['window'], ref['window'], atol=1e-8, rtol=0)
    assert np.abs(poles['z'] - ref['z']).max() < 1e-4
    every = ActiveSpace.from_counts(eps, nocc, nocc, len(eps) - nocc)
    _, _, en = build_effective_hamiltonian(low, every, sigma='mode_a')
    assert not en.diagnostics['sigma_continuation'].any()
    assert not rpa.embedding_self_energy(low, partition(every, eps),
                                         'srg').any()
    coarse = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=Numerics(quadrature={'npade': 4}))
    with pytest.warns(rpa.SigmaContinuationWarning, match=r'\(\d+, \d+\)'):
        _, _, en = build_effective_hamiltonian(coarse, a, sigma='srg')
    assert en.diagnostics['sigma_continuation'].max() > 1e-4


def _scaled_water(scale):
    """Converged RHF of water/6-31G with both H positions of `water()`
    scaled by `scale` towards the origin (the O-H bonds shortened and the
    window edges moved, the reviewer's scan of ruling R22)."""
    y, z = 0.7572 * scale, -0.4692 * scale
    mol = gto.M(atom=f'O 0 0 0.1173; H 0 {y} {z}; H 0 {-y} {z}',
                basis='6-31g', cart=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def _edge_distance(low, part):
    lo, hi = rpa.pole_window(low, part)
    e = low.eps[part.act]
    return np.minimum(e - lo, hi - e).min()


def test_continuation_margin_near_the_edge(wat):
    """Ruling R22: on imagfrequency Sigma^E is refused within
    max(pole_margin/sqrt(2 s), continuation_margin) = 0.26 Ha of the
    pole-free window's edge, the Pade continuation's accurate range by
    measurement (the scan of distance against error is in the comment of
    DOWNFOLDING_CONTINUATION_MARGIN: the last case above 1e-4 Eh at 0.162 Ha,
    every case beyond 0.171 below it, times 1.5). With the H positions at 0.8
    (srg_s = None, casida on the same factors): occupied 3, 4 and virtual 5,
    6, 11, 0.277 Ha from the edge, is accepted and meets 1e-4 (measured
    6.9e-5 for both forms; its spread, 1.9e-4, warns there); virtual 12 in place of 11, 0.171 Ha from the
    edge, is refused, naming orbital 12. At equilibrium the same space with
    virtual 12 lies 0.089 Ha from the edge: refused by default, and with
    both margins 0 it is continued and misses the target (measured 7.0e-4)
    while the 'sigma_continuation' spread (3.6e-5, a lower-bound-type
    indicator) stays below its 1e-4 warning level, without a warning."""
    infinite = Numerics(downfolding={'srg_s': None})
    mf = _scaled_water(0.8)
    eps = mf.mo_energy
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=infinite)
    low_cas = run_low_level(mf, method='rpa', mode='casida',
                            numerics=infinite)
    outside = partition(ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 11]),
                        eps)
    inside = partition(ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 12]),
                       eps)
    margin = DOWNFOLDING_CONTINUATION_MARGIN
    assert margin < _edge_distance(low_cas, outside) < 1.1 * margin
    assert _edge_distance(low_cas, inside) < margin
    for sigma in ('srg', 'mode_a'):
        got = rpa.embedding_self_energy(low_if, outside, sigma)
        want = rpa.embedding_self_energy(low_cas, outside, sigma)
        assert np.abs(got - want).max() < 1e-4
        with pytest.raises(ValueError, match=r'orbital 12 \(.*'
                           r"continuation_margin.*mode='casida'"):
            rpa.embedding_self_energy(low_if, inside, sigma)

    mf, _, eps, nocc = wat
    a = ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 12])
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        numerics=infinite)
    with pytest.raises(ValueError, match=r'orbital 12 \('):
        rpa.embedding_self_energy(low, partition(a, eps), 'srg')
    bare = Numerics(downfolding={'srg_s': None, 'pole_margin': 0.0,
                                 'continuation_margin': 0.0})
    low = run_low_level(mf, method='rpa', mode='imagfrequency', numerics=bare)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, en = build_effective_hamiltonian(low, a, sigma='srg')
    assert en.diagnostics['sigma_continuation'].max() < 1e-4
    part = partition(a, eps)
    want = rpa.embedding_self_energy(run_low_level(
        mf, method='rpa', mode='casida', numerics=bare), part, 'srg')
    assert np.abs(rpa.embedding_self_energy(low, part, 'srg')
                  - want).max() > 1e-4


# --- the space-time route: S^E from the imaginary-time GW kernel ----------


@pytest.fixture(scope='module')
def nh3_space_time():
    """The space-time RPA of NH3/cc-pVDZ HF (density fitted on cc-pvdz-ri)
    on the G1 ISDF grid."""
    mf = nh3_mean_field('HF')
    return run_low_level(mf, method='rpa', mode='space-time',
                         numerics=g1_numerics(mf))


def _nh3_part(mf, occ, virt):
    return partition(ActiveSpace.from_indices(len(mf.mo_energy), 5, occ,
                                              virt), mf.mo_energy)


def test_environment_only_g_is_what_the_kernel_sees(nh3_space_time):
    """S^E on space-time (G of the environment orbitals alone, the full
    system's mu and fit ranges) plus S^A (the same kernel call with the
    active orbitals' G) is the GW route's full Sigma_c on the active block,
    `self_energy_matrix_space_time` on the same numerics, at the Pade nodes:
    the orbital restriction is the only change, and the W and the axes of
    `_screening_grid` are the GW route's. Target 1e-10; measured 1.1e-16 on
    the contiguous (6,6) and the non-contiguous (2,3) of NH3 (|Sigma| up to
    6.8e-2, |S^E| and |S^A| each above 4e-2). W is built on the first
    request and kept in low.cache['w_grid']."""
    low = nh3_space_time
    mf, eps, nmo = low.mf, low.eps, len(low.eps)
    nodes = pade_nodes(low.numerics.quadrature)
    full = self_energy_matrix_space_time(mf, mf.mol, low.nocc, nodes,
                                         low.numerics)[0]
    mu = particle_hole_window(eps, low.nocc)[2]
    assert 'w_grid' not in low.cache
    for occ, virt in (([2, 3, 4], [5, 6, 7]), ([1, 4], [5, 8, 9])):
        part = _nh3_part(mf, occ, virt)
        act = part.act
        s_e = rpa.sigma_env_imag_axis(low, part, nodes)
        axes, w = low.cache['w_grid']
        x, d = low.cache['factors']
        s_a = self_energy_matrix_imaginary_time(
            x, d, w, np.eye(nmo)[:, act], eps[act], part.n_act_occ,
            axes['tau_points'], axes['freq_points'], nodes, mu=mu,
            ranges=axes['ranges'])[:, act][:, :, act]
        assert min(np.abs(s_e).max(), np.abs(s_a).max()) > 1e-2
        assert np.abs(s_e + s_a - full[:, act][:, :, act]).max() < 1e-10
    assert low.cache['w_grid'][1] is w


def test_lowest_excitation_space_time(nh3_space_time, monkeypatch):
    """Omega_min on space-time is the lowest root of the Casida problem on
    the ISDF v: the dense solve and the Davidson (ISDF block action, one
    root) agree (measured 3.8e-14) and both are casida's on the RI factors
    to the ISDF error (2.7e-6; target 1e-5). Ruling R21: dense up to
    DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS pairs (NH3 has 120), the Davidson
    above; the window is the route's own, no factors needed."""
    low = nh3_space_time
    dense = rpa.lowest_excitation(low, solver='dense')
    davidson = rpa.lowest_excitation(low, solver='davidson')
    assert abs(dense - davidson) < 1e-10
    cas = run_low_level(low.mf, method='rpa', mode='casida')
    assert abs(dense - cas.omega.min()) < 1e-5
    fresh = run_low_level(low.mf, method='rpa', mode='space-time',
                          numerics=low.numerics)
    assert rpa.lowest_excitation(fresh) == dense
    assert fresh.cache['omega_min'] == dense
    part = _nh3_part(low.mf, [2, 3, 4], [5, 6, 7])
    assert np.allclose(rpa.pole_window(fresh, part),
                       rpa.pole_window(cas, part), atol=1e-5, rtol=0)
    monkeypatch.setattr(rpa, 'DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS', 100)
    fresh = run_low_level(low.mf, method='rpa', mode='space-time',
                          numerics=low.numerics)
    assert rpa.lowest_excitation(fresh) == davidson
