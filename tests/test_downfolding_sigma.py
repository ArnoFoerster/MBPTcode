"""The static environment self-energy Sigma_bar^E on the casida, the
imagfrequency and the space-time route.

Sigma_bar^E is the static reduction of iG^E(W - v) on the active orbitals,
the internal orbital r running over the environment: 'srg' is
Eq. 44 of Marie and Loos (s = None: Eq. 42), 'averaged' their Eq. 12 at
eta = 0, both at the mean-field energies.

The residues of W (`screened_residues`) carry the restricted factor sqrt(2):
with r over every orbital, 'srg' is the active block of the qsGW SRG kernel
(`SelfEnergySolver.static_self_energy_matrix`), which pins that factor to
the existing code. On the environment both reductions equal the numpy
oracle's (its own dRPA residues), on a contiguous and a non-contiguous
space; the s = inf diagonal of 'srg' is that of 'averaged'; Eq. 42 is the real
part of the exact pole sum S^E at one complex frequency per element; 'srg'
at finite s stays within its damping bound of s = inf. 'averaged' warns on an
active space whose orbital energy leaves the pole-free window (or comes
within the margin of its edge) and is computed all the same; 'srg' stays
finite and equal to the oracle there. A Sigma_bar that is not finite is
refused. On the prototype's own orbitals 'averaged' is its Sigma_bar. The
`downfolding` numerics section holds srg_s, pole_margin and
continuation_margin.

A term of Eq. 44 exactly at a pole is 0, not NaN. The
'sigma_poles' diagnostic of `downfold` (window, pole distances, Z_t against
the oracle's S^E, the s applied, the SRG damping) warns on the near-pole
space only; along a short scan through a pole crossing 'srg' stays as
smooth as on a space far from the poles, where Eq. 42 jumps.

On imagfrequency S^E(mu + i w) comes from the Lorentzian fit of W and is
continued by Pade to one complex frequency per element. The Pade error
alone (the exact casida S^E through the continuation, water and benzene
(6,6)) is far below the 1e-4 Eh target; the imaginary-axis S^E matches the
casida pole sum; Omega_min is the lowest RPA root; the near-pole space
warns for both forms, pointing to casida; the 'sigma_continuation'
diagnostic is present and warns on a too-coarse node set; on benzene the
route matches casida.

On space-time S^E(mu + i w) is sampled as on imagfrequency, on the ISDF
factors and their W: against the Casida problem on the same factors it agrees
on the nodes and after continuation, also for virtuals far from the gap, and
the 0.26 Ha margin holds there too. Omega_min is the lowest root on the
ISDF v. Its agreement with casida on the RI factors is
in tests/test_downfolding_routes.py.
"""
import json
import pathlib
import re
import warnings
import numpy as np
import pytest
from pyscf import gto, scf
from scipy.sparse.linalg import ArpackNoConvergence

from src.Base.numerics.constants import (DOWNFOLDING_CONTINUATION_MARGIN,
                                         DOWNFOLDING_POLE_MARGIN,
                                         DOWNFOLDING_SRG_DAMPING_WARN,
                                         DOWNFOLDING_SRG_S)
from src.Base.numerics.numerics import (DownfoldingNumerics, Numerics,
                                        numerics_or_default)
from src.Base.orbital_spaces import particle_hole_window
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.SingleReference.GW.self_energy import SelfEnergySolver
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


def _pure_args(low, active_space):
    """(w, eps_t, eps_r, occupied_r, omega) of the environment of
    `active_space`."""
    env = active_space.env
    return (rpa.screened_residues(low, active_space),
            low.eps[active_space.act],
            low.eps[env], env < low.nocc, low.omega)


def test_srg_matches_the_qsgw_kernel(wat):
    """With r over every orbital and the DF factors, srg_self_energy at
    s = 100 is the active block of the qsGW SRG kernel, built from its own
    Casida solve and transition density, and averaged_self_energy is the
    kernel's mode A at eta = 1e-12. This pins the restricted factor of
    `screened_residues`: sqrt(2) on sum_ia (tr|ia) (X+Y)_ia, the kernel's
    2 sum chi chi (measured element ratio 1 - 3.7e-9 to 1 + 2.7e-9). The
    SRG deviation is the kernel's Laplace quadrature (relative 1e-7 per
    term, QSGW_SRG_QUAD_TOL, the tightest its rule reaches here): measured
    1.9e-10, mode A 3.1e-17 (tolerance 1e-8)."""
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', df=True)
    active_space = _contiguous_44(eps, nocc)
    nmo = len(eps)
    every = np.arange(nmo)
    w = rpa.screened_residues(low, active_space, orbitals=every)
    assert w.shape == (len(low.omega), active_space.n_act, nmo)
    ours = rpa.srg_self_energy(w, eps[active_space.act], eps, every < nocc,
                               low.omega, 100.0)
    lr = LinearResponseSolver(eps, coeff_df=low.factors,
                              spin_mode='restricted')
    omega, x, y = CasidaSolver(*lr.build_casida_matrices(nocc,
                                                         lBSE=False)).solve()
    se = SelfEnergySolver(eps, df_coeff=low.factors, spin_mode='restricted',
                          eta=1e-12)
    rho = se.rho_a_df(nocc, x, y)
    block = np.ix_(active_space.act, active_space.act)
    ref = se.static_self_energy_matrix(nocc, omega, rho, flow=100.0)[block]
    assert np.abs(ours - ref).max() < 1e-8
    assert np.abs(ref).max() > 1e-2
    mode_a = rpa.averaged_self_energy(w, eps[active_space.act], eps,
                                      every < nocc,
                                    low.omega)
    ref = se.static_self_energy_matrix(nocc, omega, rho)[block]
    assert np.abs(mode_a - ref).max() < 1e-8


@SPACES
def test_srg_and_mode_a_against_the_oracle(wat, low_exact, aspace):
    """environment_self_energy 'srg' (s = DOWNFOLDING_SRG_S, the default) and
    'averaged' against the oracle's Eq. 44 and Eq. 12 on its own residues,
    both symmetric. Measured max deviation (contiguous / non-contiguous):
    srg 1.5e-16 / 3.0e-16, mode_a 1.5e-16 / 3.0e-16 (tolerance 1e-8)."""
    _, eri, eps, nocc = wat
    active_space = aspace(eps, nocc)
    srg = rpa.environment_self_energy(low_exact, active_space, 'srg')
    mode_a = rpa.environment_self_energy(low_exact, active_space, 'averaged')
    for ours, ref in ((srg, sigma_srg(eri, eps, nocc, active_space,
                                      DOWNFOLDING_SRG_S)),
                      (mode_a, sigma_mode_a(eri, eps, nocc, active_space))):
        assert ours.shape == (active_space.n_act, active_space.n_act)
        assert np.array_equal(ours, ours.T)
        assert np.abs(ours - ref).max() < 1e-8
        assert np.abs(ref).max() > 1e-3


@SPACES
def test_diagonals_agree_at_infinite_s(low_infinite_s, wat, aspace):
    """srg_s = None: (a + b)/(a^2 + b^2) at a = b is 1/a, so the diagonal of
    'srg' is that of 'averaged'; off the diagonal they differ
    (by up to 1.9e-4 / 1.7e-4). Measured max diagonal deviation 2.2e-19 /
    1.7e-18 (tolerance 1e-10)."""
    _, _, eps, nocc = wat
    active_space = aspace(eps, nocc)
    srg = rpa.environment_self_energy(low_infinite_s, active_space, 'srg')
    mode_a = rpa.environment_self_energy(low_infinite_s, active_space,
                                         'averaged')
    assert np.abs(np.diag(srg) - np.diag(mode_a)).max() < 1e-10
    assert np.abs(srg - mode_a).max() > 1e-6


@SPACES
def test_eq42_is_one_complex_frequency(wat, low_infinite_s, aspace):
    """srg_s = None: Sigma_bar_tu = Re S^E_tu(z_tu), z_tu = (eps_t + eps_u)/2
    + i (eps_u - eps_t)/2, S^E the oracle's exact pole sum, since
    (a + b)/(a^2 + b^2) = Re[(1 + i)/(a + i b)]. Measured max deviation
    1.5e-16 / 3.0e-16 (tolerance 1e-10)."""
    _, eri, eps, nocc = wat
    active_space = aspace(eps, nocc)
    srg = rpa.environment_self_energy(low_infinite_s, active_space, 'srg')
    e = eps[active_space.act]
    ref = np.array([[sigma_env(0.5 * (e[t] + e[u]) + 0.5j * (e[u] - e[t]),
                               eri, eps, nocc, active_space)[t, u].real
                     for u in range(active_space.n_act)]
                    for t in range(active_space.n_act)])
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
    active_space = space(eps, nocc)
    w, e_t, e_r, occ_r, omega = _pure_args(low_exact, active_space)
    diff = np.abs(rpa.srg_self_energy(w, e_t, e_r, occ_r, omega, s)
                  - rpa.srg_self_energy(w, e_t, e_r, occ_r, omega, None))
    _, w_ref, c = env_residues(eri, eps, nocc, active_space)
    d_all = eps[active_space.act][None, :, None] - c[:, None,
                                                     :]       # (n, t, r)
    d = np.abs(d_all).min()
    lo, hi = rpa.pole_window(low_exact, active_space)
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
    active_space = _contiguous_44(eps, nocc)
    lo, hi = rpa.pole_window(low_exact, active_space)
    om = low_exact.omega.min()
    assert lo == eps[active_space.env_occ].max() - om
    assert hi == eps[active_space.env_virt].min() + om
    every = ActiveSpace.from_counts(eps, nocc, nocc,
                                              len(eps) - nocc)
    assert rpa.pole_window(low_exact, every) == (-np.inf, np.inf)
    for sigma in ('srg', 'averaged'):
        out = rpa.environment_self_energy(low_exact, every, sigma)
        assert out.shape == (len(eps),) * 2 and not out.any()


def test_near_pole_space(wat, low_exact):
    """The virtual 7 made active over the environment virtual 6 has its
    energy above hi - margin, within the 'averaged' pole margin of the
    window's upper edge hi (asserted here first, as is the active virtual 5
    below hi - margin; eps_7 in fact lies 0.030 Ha beyond hi): 'averaged'
    warns (SigmaNearPoleWarning, the named flag 'sigma_near_pole'), naming
    orbital 7 and its distance, and is computed all the same, equal to the
    oracle's undamped pole sum; the Hamiltonian is built and the flag is in
    its energies, also with diagnostics=(). 'srg' does not warn of the pole,
    is finite and equal to the oracle (measured 2.7e-16, tolerance 1e-8),
    and its damping matters
    there (|srg(500) - srg(inf)| = 1.5e-4, the nearest pole 0.030 Ha away)."""
    _, eri, eps, nocc = wat
    active_space = _high_virtual(eps, nocc)
    lo, hi = rpa.pole_window(low_exact, active_space)
    margin = DOWNFOLDING_POLE_MARGIN / np.sqrt(2.0 * DOWNFOLDING_SRG_S)
    assert eps[7] > hi - margin and eps[5] < hi - margin
    with pytest.warns(rpa.SigmaNearPoleWarning, match=r'orbital 7 \(') as rec:
        mode_a = rpa.environment_self_energy(low_exact, active_space,
                                             'averaged')
    text = str(rec[0].message)
    assert 'orbital 5 ' not in text and "sigma='srg'" in text
    assert np.all(np.isfinite(mode_a))
    assert np.abs(mode_a - sigma_mode_a(eri, eps, nocc,
                                        active_space)).max() < 1e-8
    assert rec[0].filename == __file__
    with pytest.warns(rpa.SigmaNearPoleWarning) as rec:
        ham = build_effective_hamiltonian(
            low_exact, _high_virtual(eps, nocc), sigma='averaged',
            diagnostics=())
        g, en = ham.g_eff, ham.energies
    # the warning points at the line that asked for the Hamiltonian
    assert rec[0].filename == __file__
    assert np.all(np.isfinite(g[1]))
    assert ham.numerical_flags == [{'flag': 'sigma_near_pole',
                                   'message': text}]
    with warnings.catch_warnings():
        warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
        srg = rpa.environment_self_energy(low_exact, active_space, 'srg')
    assert np.all(np.isfinite(srg))
    assert np.abs(srg - sigma_srg(eri, eps, nocc, active_space,
                                  DOWNFOLDING_SRG_S)).max() < 1e-8
    assert np.abs(srg - sigma_srg(eri, eps, nocc, active_space,
                                  None)).max() > 5e-5


def test_undamped_srg_warns_near_a_pole(wat, low_infinite_s):
    """At srg_s = None 'srg' is Eq. 42: undamped, with the diagonal of
    'averaged'. On casida it therefore warns of an active orbital near a pole
    as 'averaged' does (the same orbital, the named flag 'sigma_near_pole' in
    the energies), and neither message calls 'srg' finite everywhere: what
    is bounded near a pole is 'srg' with a finite srg_s, which does not warn
    (`test_near_pole_space`)."""
    _, _, eps, nocc = wat
    active_space = _high_virtual(eps, nocc)
    found = {}
    for sigma in ('srg', 'averaged'):
        with pytest.warns(rpa.SigmaNearPoleWarning,
                          match=r'orbital 7 \(') as rec:
            found[sigma] = rpa.environment_self_energy(low_infinite_s,
                                                       active_space,
                                                       sigma)
        text = str(rec[0].message)
        assert text.startswith(f"sigma={sigma!r}:")
        assert 'finite everywhere' not in text and 'finite srg_s' in text
        assert rec[0].filename == __file__
    assert np.abs(np.diag(found['srg'])
                  - np.diag(found['averaged'])).max() < 1e-12
    with pytest.warns(rpa.SigmaNearPoleWarning, match="'srg' at srg_s = None"):
        ham = build_effective_hamiltonian(low_infinite_s, active_space,
                                          sigma='srg', diagnostics=())
        en = ham.energies
    assert [entry['flag'] for entry in ham.numerical_flags] == [
        'sigma_near_pole']


def test_mode_a_margin_uses_srg_s(wat):
    """The margin is pole_margin / sqrt(2 s), with the default s when
    srg_s = None: a margin wide enough to reach the active LUMO warns of it
    and changes no number; the default does not warn; an unknown sigma is
    refused. 'srg' warns within that margin at srg_s = None, where it is
    undamped, and not at a finite srg_s."""
    mf, _, eps, nocc = wat
    active_space = _contiguous_44(eps, nocc)
    _, hi = rpa.pole_window(run_low_level(mf, method='rpa', df=False),
                            active_space)
    dist = hi - eps[6]
    with warnings.catch_warnings():
        warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
        quiet = rpa.environment_self_energy(
            run_low_level(mf, method='rpa',
                          df=False), active_space, 'averaged')
    for srg_s in (None, 2.0):
        s = DOWNFOLDING_SRG_S if srg_s is None else srg_s
        wide = 1.01 * dist * np.sqrt(2.0 * s)
        low = run_low_level(mf, method='rpa', df=False, numerics=Numerics(
            downfolding={'srg_s': srg_s, 'pole_margin': wide}))
        with pytest.warns(rpa.SigmaNearPoleWarning, match=r'orbital 6 \('):
            got = rpa.environment_self_energy(low, active_space, 'averaged')
        assert np.array_equal(got, quiet)
        if srg_s is None:
            with pytest.warns(rpa.SigmaNearPoleWarning,
                              match=r'orbital 6 \('):
                srg = rpa.environment_self_energy(low, active_space, 'srg')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
                srg = rpa.environment_self_energy(low, active_space, 'srg')
        assert np.all(np.isfinite(srg))
    with pytest.raises(ValueError, match='sigma'):
        rpa.environment_self_energy(run_low_level(mf, method='rpa', df=False),
                                    active_space, 'mode_b')


def _pinned_mean_field(molecule, density_fit=False):
    """(mf, active_space) of a prototype pin: an RHF on the pinned geometry
    and basis (density fitted if asked) carrying the prototype's mo_coeff,
    mo_energy and mo_occ (no SCF), and the partition of the pinned active
    orbitals (as in tests/test_downfolding_reference.py)."""
    c = PINS[molecule]
    mol = gto.M(atom=[(a[0], a[1:]) for a in c['atoms']], basis=c['basis'],
                unit='A', charge=c['charge'], spin=c['spin'], cart=c['cart'],
                verbose=0)
    mf = scf.RHF(mol).density_fit() if density_fit else scf.RHF(mol)
    mf.mo_coeff = np.array(c['mo_coeff'])
    mf.mo_energy = np.array(c['mo_energy'])
    mf.mo_occ = np.where(np.arange(len(mf.mo_energy)) < c['nocc'], 2.0, 0.0)
    active_space = ActiveSpace.from_indices(len(mf.mo_energy), c['nocc'],
                                      np.array(c['act_occ']),
                                      np.array(c['act_virt']))
    return mf, active_space


def _pinned(molecule):
    """(low, active_space) of a prototype pin (`_pinned_mean_field`): its
    casida RPA on exact integrals."""
    mf, active_space = _pinned_mean_field(molecule)
    return run_low_level(mf, method='rpa', mode='casida',
                         df=False), active_space


@pytest.mark.parametrize('case', sorted(PINS))
def test_prototype_sigma_bar(case):
    """'averaged' on the prototype's own orbitals is its Sigma_bar (1/2 [S(e_t)
    + S(e_u)], environment sum, full W), evaluated at its sigma_energies,
    which are the mean-field active energies (both well inside the
    pole-free window). Measured max deviation ethylene 5.2e-17, benzene
    2.4e-16 (tolerance 1e-7)."""
    c = PINS[case]
    low, active_space = _pinned(case)
    eps_act = low.eps[active_space.act]
    assert np.abs(eps_act - np.array(c['sigma_energies'])).max() < 1e-12
    ours = rpa.environment_self_energy(low, active_space, 'averaged')
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


def test_the_settings_of_sigma_can_change_per_hamiltonian(wat, low_exact):
    """srg_s, pole_margin and continuation_margin act after the RPA. A build
    takes them from the `numerics` it is given (its downfolding section) on
    the same low-level result, and gives what a new run with those numerics
    gives, bit for bit: h_eff, E_0 and the 'sigma_poles' record at a second
    srg_s on the near-pole space, where the damping matters (h_eff moves by
    4.9e-4 Ha between s = 500 and 50); on imagfrequency, both margins at
    zero silence the near-pole warning and leave Sigma_bar as it is. The
    low-level result keeps its own settings for the next build. Numerics
    derived from the run's count as theirs but for the change; any other
    difference from the run's numerics is refused: the run used those."""
    mf, _, eps, nocc = wat
    active_space = _high_virtual(eps, nocc)
    other = Numerics(downfolding={'srg_s': 50.0})
    fresh = run_low_level(mf, method='rpa', df=False, numerics=other)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rpa.SigmaDampingWarning)
        ham = build_effective_hamiltonian(low_exact, active_space,
                                          sigma='srg', numerics=other)
        g, en = ham.g_eff, ham.energies
        ham_new = build_effective_hamiltonian(fresh, active_space,
                                              sigma='srg')
        g_new, en_new = ham_new.g_eff, ham_new.energies
        ham_own = build_effective_hamiltonian(low_exact, active_space,
                                              sigma='srg')
        g_own, en_own = ham_own.g_eff, ham_own.energies
    assert g[0] == g_new[0] and np.array_equal(g[1], g_new[1])
    poles, poles_new = (h.diagnostics['sigma_poles'] for h in (ham, ham_new))
    assert poles['srg_s'] == poles_new['srg_s'] == 50.0
    assert poles['srg_damping'] == poles_new['srg_damping']
    assert ham_own.diagnostics['sigma_poles']['srg_s'] == DOWNFOLDING_SRG_S
    assert np.abs(g[1] - g_own[1]).max() > 1e-4
    assert low_exact.numerics is None
    derived = Numerics().derive('a second srg_s',
                                downfolding={'srg_s': 50.0})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rpa.SigmaDampingWarning)
        ham_derived = build_effective_hamiltonian(
            low_exact, active_space, sigma='srg', numerics=derived)
        g_derived = ham_derived.g_eff
    assert np.array_equal(g_derived[1], g[1])
    assert np.array_equal(
        rpa.environment_self_energy(low_exact, active_space, 'srg',
                                    numerics=other),
        rpa.environment_self_energy(fresh, active_space, 'srg'))

    near = ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 12])
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency')
    with pytest.warns(rpa.SigmaNearPoleWarning, match=r'orbital 12 \('):
        ham_warned = build_effective_hamiltonian(low_if, near,
                                                 sigma='srg')
        g_warned = ham_warned.g_eff
    quiet = Numerics(downfolding={'pole_margin': 0.0,
                                  'continuation_margin': 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
        ham_quiet = build_effective_hamiltonian(
            low_if, near, sigma='srg', numerics=quiet)
        g_quiet, en_quiet = ham_quiet.g_eff, ham_quiet.energies
    assert np.array_equal(g_quiet[1], g_warned[1])
    assert 'sigma_near_pole' not in [f['flag']
                                     for f in ham_quiet.numerical_flags]

    for wrong in (Numerics(quadrature={'npade': 12}),
                  Numerics(downfolding={'srg_s': 50.0},
                           imagfreq={'nfreq': 24})):
        with pytest.raises(ValueError, match='numerics.downfolding'):
            build_effective_hamiltonian(low_exact, active_space, sigma='srg',
                                        numerics=wrong)
    with pytest.raises(TypeError, match='takes no options'):
        build_effective_hamiltonian(run_low_level(mf, method='hf'),
                                    active_space,
                                    numerics=other)


def test_srg_term_at_an_exact_pole_is_zero():
    """A term with D_tr = D_ur = 0 exactly is 0/0 in Eq. 44; its
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


def test_srg_kernel_blocked_over_excitations(wat, low_exact, monkeypatch):
    """The (nexc, n_act, n_act, n_r) kernel of Eq. 44 is
    built a block of excitations at a time, the block from the byte budget
    tile_gb, so the working set is bounded and only the order of the sum
    over n changes. One excitation per block (tile_gb 1e-12) against all of
    them in one (tile_gb 1e3, the unblocked formula), at s = 500 and s = inf,
    on the environment of the contiguous (4,4) and the near-pole space of
    water/6-31G (40 excitations): measured 0.0, bitwise (bar 1e-14, the
    rounding a reordered sum may bring on a larger case). The damping diagnostic
    reads Sigma_bar(inf) from the same block loop as Sigma_bar(s), without
    a second srg_self_energy, bitwise the separate call."""
    _, _, eps, nocc = wat
    for space in (_contiguous_44, _high_virtual):
        args = _pure_args(low_exact, space(eps, nocc))
        nexc = len(args[-1])
        assert len(rpa._excitation_blocks(nexc, 4, len(args[2]),
                                          1e-12)) == nexc
        assert len(rpa._excitation_blocks(nexc, 4, len(args[2]),
                                          1e3)) == 1
        for s in (DOWNFOLDING_SRG_S, None):
            one = rpa.srg_self_energy(*args, s, tile_gb=1e-12)
            whole = rpa.srg_self_energy(*args, s, tile_gb=1e3)
            assert np.abs(whole).max() > 1e-2
            assert np.abs(one - whole).max() < 1e-14
    calls = []
    srg = rpa.srg_self_energy
    monkeypatch.setattr(rpa, 'srg_self_energy',
                        lambda *a, **k: calls.append(a) or srg(*a, **k))
    active_space = _high_virtual(eps, nocc)
    with pytest.warns(rpa.SigmaDampingWarning):
        poles = _poles(low_exact, active_space)
    assert calls == []
    args = _pure_args(low_exact, active_space)
    s_bar = srg(*args, DOWNFOLDING_SRG_S)
    assert poles['srg_damping'] == np.abs(s_bar - srg(*args, None)).max()


def _poles(low, aspace, sigma='srg', **opts):
    """The 'sigma_poles' diagnostic of `aspace` on `low` (crpa)."""
    return build_effective_hamiltonian(low, aspace, sigma=sigma,
                                       **opts).diagnostics['sigma_poles']


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
    SigmaDampingWarning names orbital 7 alone. 'averaged' applies no s:
    srg_s None, damping 0. The warning does not depend on the record:
    with diagnostics=() nothing is stored and the near-pole space still
    warns, since a damped Sigma_bar goes into H_eff either way."""
    _, eri, eps, nocc = wat
    active_space = space(eps, nocc)
    near = space is _high_virtual
    if near:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always', rpa.SigmaDampingWarning)
            poles = _poles(low_exact, active_space)
        rec = [r for r in rec if r.category is rpa.SigmaDampingWarning]
        assert len(rec) == 1
        assert re.search(r'orbital 7\b', str(rec[0].message))
        assert 'orbital 5' not in str(rec[0].message)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            poles = _poles(low_exact, active_space)
            mode_a = _poles(low_exact, active_space, sigma='averaged')
        assert mode_a['srg_s'] is None and mode_a['srg_damping'] == 0.0
        assert np.array_equal(mode_a['z'], poles['z'])
    assert set(poles) == {'window', 'pole_distance', 'z', 'srg_s',
                          'srg_damping'}
    lo, hi = rpa.pole_window(low_exact, active_space)
    e = eps[active_space.act]
    assert poles['window'] == (lo, hi)
    assert np.array_equal(poles['pole_distance'], np.minimum(e - lo, hi - e))
    assert (poles['pole_distance'] < 0).tolist() == [False, False, False, near]
    assert poles['srg_s'] == DOWNFOLDING_SRG_S
    w, e_t, e_r, occ_r, omega = _pure_args(low_exact, active_space)
    damping = np.abs(rpa.srg_self_energy(w, e_t, e_r, occ_r, omega,
                                         DOWNFOLDING_SRG_S)
                     - rpa.srg_self_energy(w, e_t, e_r, occ_r, omega,
                                           None)).max()
    assert poles['srg_damping'] == damping
    assert (damping > DOWNFOLDING_SRG_DAMPING_WARN) == near
    h = 1e-5
    def s_tt(x, t):
        return sigma_env(x, eri, eps, nocc, active_space)[t, t]

    slope = np.array([(s_tt(x + h, t) - s_tt(x - h, t)).real / (2 * h)
                      for t, x in enumerate(e)])
    assert np.abs(poles['z'] - 1.0 / (1.0 - slope)).max() < 1e-7
    assert np.all((poles['z'] > 0) & (poles['z'] <= 1))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always', RuntimeWarning)
        ham = build_effective_hamiltonian(low_exact, active_space,
                                          sigma='srg', diagnostics=())
        en = ham.energies
    assert ham.diagnostics == {}
    damped = [r for r in rec if r.category is rpa.SigmaDampingWarning]
    assert len(damped) == (1 if near else 0)
    assert not [r for r in rec if issubclass(r.category, RuntimeWarning)
                and r.category is not rpa.SigmaDampingWarning]


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
        near = _high_virtual(eps, nocc)
        srg.append(rpa.environment_self_energy(low, near, 'srg'))
        args = _pure_args(low, near)
        infinite.append(rpa.srg_self_energy(*args, None))
        plain.append(rpa.environment_self_energy(
            low, _contiguous_44(eps, nocc), 'srg'))
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


def _pole_sum(low, active_space, z):
    """S^E_tu(z) (nz, n_act, n_act), the exact pole sum on the casida route:
    sum_rn w_tr w_ur / (z - eps_r - sgn(eps_r - eps_F) Omega_n), the
    residues of `screened_residues` (pinned to the oracle above)."""
    env = active_space.env
    sign = np.where(env < low.nocc, -1.0, 1.0)
    poles = low.eps[env][None, :] + sign[None, :] * low.omega[:, None]
    w = rpa.screened_residues(low, active_space)
    g = 1.0 / (np.atleast_1d(z)[:, None, None] - poles[None])
    return np.einsum('ntr,nur,qnr->qtu', w, w, g)


def _pade_points(low):
    """(nodes, mu): the Pade nodes of the default numerics and the mid-gap."""
    return (pade_nodes(numerics_or_default(None).quadrature),
            particle_hole_window(low.eps, low.nocc)[2])


def _pade_alone_cases(wat, low_exact):
    _, _, eps, nocc = wat
    for space in (_contiguous_44, _non_contiguous):
        yield low_exact, space(eps, nocc)
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
    for low, active_space in _pade_alone_cases(wat, low_exact):
        nodes, mu = _pade_points(low)
        samples = _pole_sum(low, active_space, mu + 1j * nodes)
        args = _pure_args(low, active_space)
        e = low.eps[active_space.act]
        for sigma, exact in (('srg', rpa.srg_self_energy(*args, None)),
                             ('averaged', rpa.averaged_self_energy(*args))):
            got = rpa.continued_self_energy(mu + 1j * nodes, samples, e,
                                            sigma)
            assert np.array_equal(got, got.T)
            assert np.abs(exact).max() > 1e-2
            assert np.abs(got - exact).max() < 1e-4


def test_sigma_bar_matches_casida_benzene():
    """Benzene (6,6) pi space cc-pVDZ on the prototype's orbitals, density
    fitted: Sigma_bar^E on imagfrequency against casida on the same
    three-index object at srg_s = None, the case the route's tolerance is
    set on. Target 1e-4 Eh per element; measured 1.6e-7 for both
    forms (|Sigma_bar| up to 4.5e-2; the Pade error alone is 1.8e-7,
    test_pade_alone), the lowest root 2.0e-14 from casida's."""
    mf, active_space = _pinned_mean_field('benzene', density_fit=True)
    infinite = Numerics(downfolding={'srg_s': None})
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=infinite)
    low_cas = run_low_level(mf, method='rpa', mode='casida',
                            numerics=infinite)
    assert abs(rpa.lowest_excitation(low_if) - low_cas.omega.min()) < 1e-10
    for sigma in ('srg', 'averaged'):
        got = rpa.environment_self_energy(low_if, active_space, sigma)
        want = rpa.environment_self_energy(low_cas, active_space, sigma)
        assert np.abs(want).max() > 1e-2
        assert np.abs(got - want).max() < 1e-4


def test_environment_self_energy_samples_against_the_pole_sum(wat):
    """environment_self_energy_samples on imagfrequency against the casida
    pole sum on the same three-index object (water/6-31G, df), on the Pade
    nodes, with the internal orbital over the environment alone: the
    Lorentzian fit of W on
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
        active_space = space(eps, nocc)
        nodes, mu = _pade_points(low_if)
        got = rpa.environment_self_energy_samples(low_if, active_space, nodes)
        assert got.shape == (len(nodes), active_space.n_act,
                             active_space.n_act)
        ref = _pole_sum(low_cas, active_space, mu + 1j * nodes)
        assert np.abs(got - ref).max() < 1e-5
    w_grid = low_if.cache['w_grid']
    rpa.environment_self_energy_samples(low_if, active_space, nodes)
    assert low_if.cache['w_grid'] is w_grid
    for low in (low_cas, run_low_level(mf, method='rpa', df=False)):
        with pytest.raises(NotImplementedError,
                           match="use mode='imagfrequency'"):
            rpa.environment_self_energy_samples(low, active_space, nodes)


def test_lowest_excitation(wat):
    """Omega_min on imagfrequency (no eigenvectors kept) is the lowest RPA
    root, found without building the Casida problem (`lowest_excitation`):
    it equals the casida route's lowest omega (measured 1.8e-15; bar
    1e-10), is cached in low.cache['omega_min'], and is the same number on a
    second run, the start vector being fixed; so the pole-free windows
    agree. Without an environment the window is the whole axis and no root
    is solved for."""
    mf, _, eps, nocc = wat
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency')
    low_cas = run_low_level(mf, method='rpa', mode='casida')
    ref = low_cas.omega.min()
    assert rpa.lowest_excitation(low_cas) == ref
    assert 'omega_min' not in low_if.cache
    om = rpa.lowest_excitation(low_if)
    assert abs(om - ref) < 1e-10
    assert low_if.cache['omega_min'] == om
    assert rpa.lowest_excitation(
        run_low_level(mf, method='rpa', mode='imagfrequency')) == om
    active_space = _contiguous_44(eps, nocc)
    assert np.allclose(rpa.pole_window(low_if, active_space),
                       rpa.pole_window(low_cas,
                                       active_space), atol=1e-10, rtol=0)
    fresh = run_low_level(mf, method='rpa', mode='imagfrequency')
    every = ActiveSpace.from_counts(eps, nocc, nocc,
                                              len(eps) - nocc)
    assert rpa.pole_window(fresh, every) == (-np.inf, np.inf)
    assert 'omega_min' not in fresh.cache


def test_near_pole_warning_points_to_casida(wat):
    """On imagfrequency both 'srg' (s = inf there) and 'averaged' warn on the
    near-pole space of the casida tests (virtual 7 beyond the window's edge),
    naming orbital 7 alone and pointing to mode='casida', where 'srg' is
    finite; Sigma_bar is continued all the same, the Hamiltonian is built
    and 'sigma_near_pole' is among the flags of its energies. The contiguous
    (4,4) does not warn of the poles."""
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    near = _high_virtual(eps, nocc)
    for sigma in ('srg', 'averaged'):
        with pytest.warns(rpa.SigmaNearPoleWarning,
                          match=r'orbital 7 \(') as rec:
            got = rpa.environment_self_energy(low, near, sigma)
        text = str([w.message for w in rec
                    if w.category is rpa.SigmaNearPoleWarning][0])
        assert "mode='casida'" in text and 'orbital 5 ' not in text
        assert np.all(np.isfinite(got))
        with pytest.warns(rpa.SigmaNearPoleWarning, match="mode='casida'"):
            ham = build_effective_hamiltonian(
                low, _high_virtual(eps, nocc), sigma=sigma)
            g, en = ham.g_eff, ham.energies
        assert np.all(np.isfinite(g[1]))
        assert {'flag': 'sigma_near_pole',
                'message': text} in ham.numerical_flags
        with warnings.catch_warnings():
            warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
            ok = rpa.environment_self_energy(low, _contiguous_44(eps, nocc),
                                             sigma)
        assert np.all(np.isfinite(ok))


def test_a_sigma_that_is_not_finite_is_refused(wat, low_exact, monkeypatch):
    """Near a pole Sigma_bar^E is computed with a warning; exactly on one
    'averaged' divides by zero. A Sigma_bar with an element that is not finite
    cannot go into H_eff and is refused (FloatingPointError), naming the
    orbital pairs and pointing to sigma='srg' on casida."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # D = eps_t - eps_r - Omega = 1 - 0.5 - 0.5 = 0 for the empty r
        on_pole = rpa.averaged_self_energy(
            np.ones((1, 1, 1)), np.array([1.0]), np.array([0.5]),
            np.array([False]), np.array([0.5]))
    assert not np.isfinite(on_pole).any()

    _, _, eps, nocc = wat
    active_space = _contiguous_44(eps, nocc)
    exact = rpa.averaged_self_energy

    def with_a_pole(*args):
        out = exact(*args)
        out[-1, -1] = np.inf
        return out

    monkeypatch.setattr(rpa, 'averaged_self_energy', with_a_pole)
    act = active_space.act
    with pytest.raises(FloatingPointError,
                       match=rf"\({act[-1]}, {act[-1]}\).*sigma='srg'"):
        rpa.environment_self_energy(low_exact, active_space, 'averaged')


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
    elements, also with diagnostics=(), which stores nothing: the warning
    does not depend on the record."""
    mf, _, eps, nocc = wat
    a = _contiguous_44(eps, nocc)
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ham = build_effective_hamiltonian(low, a, sigma='srg')
        en = ham.energies
    spread = ham.diagnostics['sigma_continuation']
    assert spread.shape == (4, 4)
    assert np.all(spread >= 0) and spread.max() < 1e-4
    poles = ham.diagnostics['sigma_poles']
    ham_c = build_effective_hamiltonian(
        run_low_level(mf, method='rpa', mode='casida'), a, sigma='srg')
    en_c = ham_c.energies
    assert 'sigma_continuation' not in ham_c.diagnostics
    ref = ham_c.diagnostics['sigma_poles']
    assert poles['srg_s'] is None and poles['srg_damping'] == 0.0
    assert np.allclose(poles['window'], ref['window'], atol=1e-8, rtol=0)
    assert np.abs(poles['z'] - ref['z']).max() < 1e-4
    every = ActiveSpace.from_counts(eps, nocc, nocc, len(eps) - nocc)
    ham = build_effective_hamiltonian(low, every, sigma='averaged')
    en = ham.energies
    assert not ham.diagnostics['sigma_continuation'].any()
    assert not rpa.environment_self_energy(low, every,
                                           'srg').any()
    coarse = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=Numerics(quadrature={'npade': 4}))
    with pytest.warns(rpa.SigmaContinuationWarning, match=r'\(\d+, \d+\)'):
        ham = build_effective_hamiltonian(coarse, a, sigma='srg')
        en = ham.energies
    assert ham.diagnostics['sigma_continuation'].max() > 1e-4
    with pytest.warns(rpa.SigmaContinuationWarning, match=r'\(\d+, \d+\)'):
        ham = build_effective_hamiltonian(coarse, a, sigma='srg',
                                          diagnostics=())
        en = ham.energies
    assert ham.diagnostics == {}


def _scaled_water(scale):
    """Converged RHF of water/6-31G with both H positions of `water()`
    scaled by `scale` towards the origin (the O-H bonds shortened and the
    window edges moved: the scan behind DOWNFOLDING_CONTINUATION_MARGIN)."""
    y, z = 0.7572 * scale, -0.4692 * scale
    mol = gto.M(atom=f'O 0 0 0.1173; H 0 {y} {z}; H 0 {-y} {z}',
                basis='6-31g', cart=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def _edge_distance(low, active_space):
    lo, hi = rpa.pole_window(low, active_space)
    e = low.eps[active_space.act]
    return np.minimum(e - lo, hi - e).min()


def test_continuation_margin_near_the_edge(wat):
    """On imagfrequency Sigma^E warns (SigmaNearPoleWarning) within
    max(pole_margin/sqrt(2 s), continuation_margin) = 0.26 Ha of the
    pole-free window's edge, beyond which the Pade continuation is accurate
    by measurement (the scan of distance against error is in the comment of
    DOWNFOLDING_CONTINUATION_MARGIN: the last case above 1e-4 Eh at 0.162 Ha,
    every case beyond 0.171 below it, times 1.5). With the H positions at 0.8
    (srg_s = None, casida on the same factors): occupied 3, 4 and virtual 5,
    6, 11, 0.277 Ha from the edge, does not warn of the poles and meets 1e-4
    (measured 6.9e-5 for both forms; its spread, 1.9e-4, warns there);
    virtual 12 in place of 11, 0.171 Ha from the edge, warns, naming orbital
    12, and is continued. At equilibrium the same space with virtual 12 lies
    0.089 Ha from the edge: it warns by default and its value is the one
    with both margins 0, which is quiet, misses the target (measured 7.0e-4)
    and has a 'sigma_continuation' spread (3.6e-5, a lower-bound-type
    indicator) below its 1e-4 warning level: the margin is the only guard
    there."""
    infinite = Numerics(downfolding={'srg_s': None})
    mf = _scaled_water(0.8)
    eps = mf.mo_energy
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=infinite)
    low_cas = run_low_level(mf, method='rpa', mode='casida',
                            numerics=infinite)
    outside = ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 11])
    inside = ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 12])
    margin = DOWNFOLDING_CONTINUATION_MARGIN
    assert margin < _edge_distance(low_cas, outside) < 1.1 * margin
    assert _edge_distance(low_cas, inside) < margin
    for sigma in ('srg', 'averaged'):
        with warnings.catch_warnings():
            warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
            got = rpa.environment_self_energy(low_if, outside, sigma)
        want = rpa.environment_self_energy(low_cas, outside, sigma)
        assert np.abs(got - want).max() < 1e-4
        with pytest.warns(rpa.SigmaNearPoleWarning, match=r'orbital 12 \(.*'
                          r"continuation_margin.*mode='casida'"):
            got = rpa.environment_self_energy(low_if, inside, sigma)
        assert np.all(np.isfinite(got))

    mf, _, eps, nocc = wat
    a = ActiveSpace.from_indices(13, 5, [3, 4], [5, 6, 12])
    active_space = a
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        numerics=infinite)
    with pytest.warns(rpa.SigmaNearPoleWarning, match=r'orbital 12 \('):
        warned = rpa.environment_self_energy(low, active_space, 'srg')
    bare = Numerics(downfolding={'srg_s': None, 'pole_margin': 0.0,
                                 'continuation_margin': 0.0})
    low = run_low_level(mf, method='rpa', mode='imagfrequency', numerics=bare)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ham = build_effective_hamiltonian(low, a, sigma='srg')
        en = ham.energies
    assert ham.diagnostics['sigma_continuation'].max() < 1e-4
    want = rpa.environment_self_energy(run_low_level(
        mf, method='rpa', mode='casida', numerics=bare), active_space, 'srg')
    quiet = rpa.environment_self_energy(low, active_space, 'srg')
    assert np.array_equal(warned, quiet)
    assert np.abs(quiet - want).max() > 1e-4


# --- the space-time route: S^E from the same fit, on the ISDF factors ------


@pytest.fixture(scope='module')
def nh3_space_time():
    """The space-time RPA of NH3/cc-pVDZ HF (density fitted on cc-pvdz-ri)
    on the G1 ISDF grid."""
    mf = nh3_mean_field('HF')
    return run_low_level(mf, method='rpa', mode='space-time',
                         numerics=g1_numerics(mf))


def _nh3_part(mf, occ, virt):
    return ActiveSpace.from_indices(len(mf.mo_energy), 5, occ,
                                              virt)


def _casida_on_the_isdf_factors(low):
    """A casida low-level result on the three-index object of the space-time
    result's own ISDF factors, B_Q,pq = sum_P D_PQ X_Pp X_Pq: the same fitted
    integrals, with explicit poles, so the continuation can be judged apart
    from the fit."""
    x, d = low.cache['factors']
    nmo = x.shape[1]
    b = (d.T @ (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)).reshape(
        d.shape[1], nmo, nmo)
    lr = LinearResponseSolver(low.eps, coeff_df=b, spin_mode='restricted')
    omega, xpy, y = CasidaSolver(*lr.build_casida_matrices(
        low.nocc, lBSE=False)).solve()
    return rpa.RPALowLevel(
        mf=low.mf, mode='casida', df=True, eps=low.eps, nocc=low.nocc,
        e_c_full=0.0, omega=omega, xpy=xpy + y, factors=b, eri=None,
        numerics=Numerics(downfolding={'srg_s': None}))


def test_space_time_samples_the_fitted_self_energy(nh3_space_time):
    """S^E on space-time is sampled as on imagfrequency, the Lorentzian fit
    of W convolved exactly with G, on the route's own ISDF factors and its
    own W; not through the imaginary-time quadrature of the space-time GW
    self-energy, whose 1e-7 of noise on the nodes the continuation amplified
    to above 1e-4 for a virtual far from the gap.

    Against the Casida problem on the SAME ISDF factors (explicit poles), on
    NH3/cc-pVDZ, G1 grid: S^E on the Pade nodes agrees with the exact pole
    sum (measured 2.7e-8 and 2.8e-8, target 1e-5), and Sigma_bar (Eq. 42)
    with its exact value, also for the space 4 | 5, 6, 7, 12, 13, whose
    virtuals 12 and 13 lie 1.19 Ha above mu and which the default margin
    accepts (0.28 Ha from the window edge): measured 5.6e-9 and 6.4e-7,
    target 1e-5. The imaginary-time sampler gave 2.2e-4 on that space
    against the RI casida; this one gives 3.2e-6 there, the ISDF fit. W is
    built on the first request, on the minimax grid of imagfrequency, and
    kept."""
    low = nh3_space_time
    mf, eps = low.mf, low.eps
    exact = _casida_on_the_isdf_factors(low)
    nodes, mu = _pade_points(low)
    assert 'w_grid' not in low.cache
    for occ, virt in (([2, 3, 4], [5, 6, 7]), ([4], [5, 6, 7, 12, 13])):
        active_space = _nh3_part(mf, occ, virt)
        s_e = rpa.environment_self_energy_samples(low, active_space, nodes)
        assert np.abs(s_e).max() > 1e-2
        assert np.abs(s_e - _pole_sum(exact, active_space,
                                      mu + 1j * nodes)).max() < 1e-5
        got = rpa.continued_self_energy(mu + 1j * nodes, s_e,
                                        eps[active_space.act],
                                        'srg')
        want = rpa.environment_self_energy(exact, active_space, 'srg')
        assert np.abs(got - want).max() < 1e-5
    points, w = low.cache['w_grid']
    assert len(points) == low.numerics.imagfreq.nfreq == len(w)
    rpa.environment_self_energy_samples(low, active_space, nodes)
    assert low.cache['w_grid'][1] is w


def _no_convergence(*args, **kwargs):
    """Stands in for the Lanczos solver: the iteration gives up."""
    raise ArpackNoConvergence('ARPACK error -1: No convergence (stood in)',
                              np.zeros(0), np.zeros((0, 0)))


def _n2():
    """N2/6-31G RHF, density fitted, in symmetry-adapted orbitals: pi_y and
    pi*_y are pi_x and pi*_x turned by 90 degrees, with the same phases."""
    mf = scf.RHF(gto.M(atom='N 0 0 0; N 0 0 1.1', basis='6-31g', verbose=0,
                       symmetry=True)).density_fit()
    mf.conv_tol = 1e-11
    return mf.run()


def _rhf(atom, basis):
    """RHF without a density fit: the three-index factors of both routes
    are then the eigendecomposition of the exact integrals."""
    mf = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0))
    mf.conv_tol = 1e-11
    return mf.run()


def test_omega_min_builds_no_casida_problem(wat, monkeypatch):
    """Omega_min on a numerical route comes from applications of the
    occupied-virtual factors alone: no Casida matrix is built or
    diagonalized for it (a stand-in for the dense solver that fails when
    called is never called), and it is the casida route's lowest root."""
    mf = wat[0]
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    ref = run_low_level(mf, method='rpa', mode='casida').omega.min()

    def refuse(*args, **kwargs):
        raise AssertionError('a Casida problem was built for Omega_min')

    monkeypatch.setattr(rpa, 'CasidaSolver', refuse)
    assert abs(rpa.lowest_excitation(low) - ref) < 1e-10


@pytest.mark.parametrize('case', ['two pairs', 'three pairs', 'n2'])
def test_lowest_excitation_of_small_and_degenerate_pair_spaces(case):
    """The solver at its edges, against the casida route on the same
    factors (bar 1e-10): two particle-hole pairs (He/6-311G) and three
    (H2/6-31G), where the Lanczos space is the whole space; and N2 in
    symmetry-adapted orbitals. Its lowest root is the combination
    pi_x -> pi*_y minus pi_y -> pi*_x, whose transition density vanishes, so
    it lies at the orbital-energy gap itself (0.770970 Ha), and a start
    vector with equal weights on those two pairs is orthogonal to it: with
    such a start the iteration returns the next root, 0.056 Ha too high,
    which would widen the pole-free window (tried; this test fails then).
    The pseudo-random start finds it. Measured: 0.0, 1.1e-16 and 1.4e-14."""
    mf = {'two pairs': lambda: _rhf('He', '6-311g'),
          'three pairs': lambda: _rhf('H 0 0 0; H 0 0 0.74', '6-31g'),
          'n2': _n2}[case]()
    ref = run_low_level(mf, method='rpa', mode='casida').omega.min()
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    assert abs(rpa.lowest_excitation(low) - ref) < 1e-10


def test_lowest_excitation_of_one_pair():
    """One particle-hole pair is its own root, Omega^2 = delta (delta +
    4 K), with nothing to iterate. The imagfrequency run itself needs more
    than one transition energy for its grid, so the result is put together
    by hand from the casida one of H2 in a minimal basis: its factors and
    P(0) = -4 B B^T / delta. Measured: 0.0."""
    cas = run_low_level(_rhf('H 0 0 0; H 0 0 0.74', 'sto-3g'), method='rpa',
                        mode='casida')
    assert cas.omega.shape == (1,)
    b = cas.factors[:, :1, 1:].reshape(-1, 1)
    p0 = -4.0 * (b @ b.T) / (cas.eps[1] - cas.eps[0])
    low = rpa.RPALowLevel(
        mf=cas.mf, mode='imagfrequency', df=True, numerics=None, eps=cas.eps,
        nocc=1, e_c_full=cas.e_c_full, omega=None, xpy=None,
        factors=cas.factors, eri=None, cache={'p0': p0})
    assert abs(rpa.lowest_excitation(low) - cas.omega[0]) < 1e-12


def test_omega_min_falls_back_to_the_gap_bound(wat, monkeypatch):
    """A Lanczos iteration that does not finish does not stop the build
    and leaves no estimate that could lie above Omega_min: the exact lower
    bound of the direct RPA, the smallest orbital-energy gap, takes its
    place, with one SigmaNearPoleWarning and the named flag
    'sigma_pole_window_bound' in the energies of every build on that
    result. The window is then narrower than the true one (water: the gap
    is 0.0228 Ha below Omega_min), the safe side; Sigma_bar and h_eff
    do not read it."""
    mf, _, eps, nocc = wat
    space = _contiguous_44(eps, nocc)
    gap = eps[nocc] - eps[nocc - 1]
    true = run_low_level(mf, method='rpa', mode='casida').omega.min()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ham_ok = build_effective_hamiltonian(
            run_low_level(mf, method='rpa', mode='imagfrequency'), space,
            sigma='srg')
        g_ok, en_ok = ham_ok.g_eff, ham_ok.energies
    monkeypatch.setattr(rpa, 'eigsh', _no_convergence)
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    with pytest.warns(rpa.SigmaNearPoleWarning,
                      match='did not finish.*smallest orbital-energy '
                            'gap') as rec:
        om = rpa.lowest_excitation(low)
    assert len(rec) == 1
    assert om == gap and 1e-3 < true - om < 0.1
    fresh = run_low_level(mf, method='rpa', mode='imagfrequency')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ham = build_effective_hamiltonian(fresh, space, sigma='srg')
        g, en = ham.g_eff, ham.energies
        ham_again = build_effective_hamiltonian(fresh, space, sigma='srg')
        again = ham_again.energies
    names = [f['flag'] for f in ham.numerical_flags]
    assert 'sigma_pole_window_bound' in names
    assert [f['flag'] for f in ham_again.numerical_flags] == names
    assert 'sigma_pole_window_bound' not in [
        f['flag'] for f in ham_ok.numerical_flags]
    assert np.array_equal(g[1], g_ok[1])
    (lo, hi), (lo_ok, hi_ok) = (h.diagnostics['sigma_poles']['window']
                                for h in (ham, ham_ok))
    assert lo_ok < lo and hi < hi_ok


def test_lowest_excitation_space_time(nh3_space_time):
    """Omega_min on space-time is the lowest root of the Casida problem on
    the ISDF v, with the factors applied through the collocation (the
    three-index object is never built): it is the lowest root of the dense
    problem on those same factors (measured 3.8e-14; bar 1e-10, which
    the quadrature error of the route's P(0), 2e-9 in the inverse the
    iteration runs on, would miss without the Rayleigh quotient that ends
    it), and casida's on the RI factors to the ISDF error (2.7e-6; target
    1e-5). Cached; the window is the route's own."""
    low = nh3_space_time
    om = rpa.lowest_excitation(low)
    assert abs(om - _casida_on_the_isdf_factors(low).omega.min()) < 1e-10
    cas = run_low_level(low.mf, method='rpa', mode='casida')
    assert abs(om - cas.omega.min()) < 1e-5
    fresh = run_low_level(low.mf, method='rpa', mode='space-time',
                          numerics=low.numerics)
    assert rpa.lowest_excitation(fresh) == om
    assert fresh.cache['omega_min'] == om
    active_space = _nh3_part(low.mf, [2, 3, 4], [5, 6, 7])
    assert np.allclose(rpa.pole_window(fresh, active_space),
                       rpa.pole_window(cas, active_space), atol=1e-5, rtol=0)


def test_space_time_continuation_margin_near_the_edge(nh3_space_time):
    """The continuation margin on space-time is imagfrequency's, 0.26 Ha,
    because the two routes now sample S^E the same way (the scan repeated on
    space-time: water, NH3 and benzene in cc-pVDZ, G1 grid, 160 spaces,
    against the RI casida; beyond the margin the worst difference is 4.2e-5
    on water, 8.2e-6 on NH3 and 8.5e-5 on benzene, which holds the ISDF fit
    of the integrals. Against casida on the same ISDF factors, checked on 5
    spaces, three of benzene and two of NH3, the continuation error
    is at most 9.3e-7).

    NH3 with default numerics: occupied 4 and virtuals 5, 6, 7, 12, 13,
    0.280 Ha from the edge with two virtuals 1.19 Ha above the mid-gap, is
    quiet about the poles and agrees with the casida route on the RI factors
    to 1e-5 (measured 3.2e-6 'srg', the ISDF fit; the imaginary-time sampler
    this replaced gave 2.2e-4); occupied 1, 4 and virtuals 5, 8, 9, 0.12 Ha
    from the edge, warns, naming the margin and pointing to casida, and is
    continued."""
    low = nh3_space_time
    mf = low.mf
    cas = run_low_level(mf, method='rpa', mode='casida',
                        numerics=Numerics(downfolding={'srg_s': None}))
    margin = DOWNFOLDING_CONTINUATION_MARGIN
    outside = _nh3_part(mf, [4], [5, 6, 7, 12, 13])
    inside = _nh3_part(mf, [1, 4], [5, 8, 9])
    assert margin < _edge_distance(cas, outside) < 1.1 * margin
    assert _edge_distance(cas, inside) < margin
    assert (low.eps[outside.act] - particle_hole_window(
        low.eps, low.nocc)[2]).max() > 1.0
    for sigma in ('srg', 'averaged'):
        with warnings.catch_warnings():
            warnings.simplefilter('error', rpa.SigmaNearPoleWarning)
            warnings.simplefilter('ignore', rpa.SigmaContinuationWarning)
            got = rpa.environment_self_energy(low, outside, sigma)
        want = rpa.environment_self_energy(cas, outside, sigma)
        assert np.abs(got - want).max() < 1e-5
        with pytest.warns(rpa.SigmaNearPoleWarning,
                          match="continuation_margin.*mode='casida'"):
            got = rpa.environment_self_energy(low, inside, sigma)
        assert np.all(np.isfinite(got))


def test_sigma_warnings_are_named_flags_in_the_energies(wat, low_exact):
    """The two Sigma^E warnings reach `h.numerical_flags`: the SRG
    damping of the near-pole space on casida ('sigma_srg_damping', its
    message the warning's), and the Pade spread of a too-coarse node set on
    imagfrequency ('sigma_continuation_spread'); also with diagnostics=().
    A space clear of the poles has none."""
    mf, _, eps, nocc = wat
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always', rpa.SigmaDampingWarning)
        ham = build_effective_hamiltonian(
            low_exact, _high_virtual(eps, nocc), sigma='srg', diagnostics=())
        en = ham.energies
    flags = ham.numerical_flags
    assert [f['flag'] for f in flags] == ['sigma_srg_damping']
    assert flags[0]['message'] == str(rec[-1].message)
    assert rec[-1].filename == __file__
    ham = build_effective_hamiltonian(
        low_exact, _contiguous_44(eps, nocc), sigma='srg')
    en = ham.energies
    assert ham.numerical_flags == []
    coarse = run_low_level(mf, method='rpa', mode='imagfrequency',
                           numerics=Numerics(quadrature={'npade': 4}))
    with pytest.warns(rpa.SigmaContinuationWarning):
        ham = build_effective_hamiltonian(
            coarse, _contiguous_44(eps, nocc), sigma='srg')
        en = ham.energies
    assert [f['flag'] for f in ham.numerical_flags] == [
        'sigma_continuation_spread']


def test_flags_of_a_cached_quantity_reach_every_active_space(wat, monkeypatch):
    """One low-level result serves several active spaces, and W on the
    frequency grid is built by the first Sigma^E request alone. A named flag
    raised while building it is in the energies of every later active space
    that reuses it, not only of the first; the warning is issued once."""
    from src.Base.numerical_flags import flag
    mf, _, eps, nocc = wat
    low = run_low_level(mf, method='rpa', mode='imagfrequency')
    build, built = rpa._build_screening_grid, []

    def flagged(*args):
        built.append(args)
        flag('probe_grid', 'raised while building W', RuntimeWarning)
        return build(*args)

    monkeypatch.setattr(rpa, '_build_screening_grid', flagged)
    probe = {'flag': 'probe_grid', 'message': 'raised while building W'}
    with pytest.warns(RuntimeWarning, match='raised while building W'):
        ham_first = build_effective_hamiltonian(
            low, _contiguous_44(eps, nocc), sigma='srg')
        first = ham_first.energies
    assert ham_first.numerical_flags == [probe]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ham_later = build_effective_hamiltonian(
            low, _contiguous_44(eps, nocc), sigma='averaged')
        later = ham_later.energies
    assert ham_later.numerical_flags == [probe] and len(built) == 1
    assert low.numerical_flags == []
    ham_without = build_effective_hamiltonian(low, _contiguous_44(eps, nocc),
                                              sigma=None)
    without = ham_without.energies
    assert ham_without.numerical_flags == []


def test_both_sigma_warnings_share_one_threshold():
    """The SRG damping warning (casida) and the continuation warning
    (imagfrequency, space-time) never fire for the same run, and both mean
    that Sigma_bar is uncertain, or depends on a setting, at one level:
    1e-4 Ha, the accuracy the route tests aim for. A micro-Hartree damping
    is the regularization at work, not a warning."""
    from src.Base.numerics import constants
    assert (constants.DOWNFOLDING_SRG_DAMPING_WARN
            == constants.DOWNFOLDING_SIGMA_CONTINUATION_WARN == 1e-4)


@pytest.mark.parametrize('mode', ['casida', 'imagfrequency'])
def test_empty_environment_or_active_space_gives_zero_on_every_route(
        wat, low_exact, mode):
    """Every orbital active (no environment) or no orbital active: each spin
    channel's record holds a zero Sigma_bar, on imagfrequency without a Pade
    fit, and nothing warns of the continuation. Space-time takes the same
    path as imagfrequency once its samples are zero."""
    mf, _, eps, nocc = wat
    low = (low_exact if mode == 'casida'
           else run_low_level(mf, method='rpa', mode=mode))
    for space in (ActiveSpace.from_counts(eps, nocc, nocc, len(eps) - nocc),
                  ActiveSpace(len(eps), nocc, [], [])):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always', RuntimeWarning)
            records, _ = rpa._environment_self_energy(low, space, 'averaged')
        assert [type(r) for r in records] == [rpa._SigmaE]
        (record,) = records
        n = space.n_act
        assert record.sigma_bar.shape == (n, n)
        assert not record.sigma_bar.any()
        assert np.array_equal(record.eps_act, eps[space.act])
        if mode == 'casida':
            assert isinstance(record.evaluation, rpa._PoleSum)
        else:
            assert isinstance(record.evaluation, rpa._Continued)
            assert record.evaluation.fit is None
        assert not [r for r in rec
                    if r.category is rpa.SigmaContinuationWarning]


def _coarse_imagfrequency(mf):
    """The imagfrequency run of `mf` with npade = 4, whose two half fits
    differ by more than the continuation warning's level."""
    return run_low_level(mf, method='rpa', mode='imagfrequency',
                         numerics=Numerics(quadrature={'npade': 4}))


def _recorded(category, call):
    """The warnings of `category` that `call()` raises, each one kept."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always', RuntimeWarning)
        call()
    return [r for r in rec if r.category is category]


def test_public_function_warns_like_the_driver(wat, low_exact):
    """`environment_self_energy` gives the warnings
    `build_effective_hamiltonian` gives for the same Sigma_bar: the SRG
    damping of the near-pole space on casida and the Pade spread of a
    too-coarse node set on imagfrequency."""
    mf, _, eps, nocc = wat
    with pytest.warns(rpa.SigmaDampingWarning, match=r'orbital 7\b'):
        rpa.environment_self_energy(low_exact, _high_virtual(eps, nocc), 'srg')
    with pytest.warns(rpa.SigmaContinuationWarning, match=r'\(\d+, \d+\)'):
        rpa.environment_self_energy(_coarse_imagfrequency(mf),
                                    _contiguous_44(eps, nocc), 'srg')


def test_driver_warns_once(wat, low_exact):
    """Through `build_effective_hamiltonian` each Sigma^E warning comes
    once: one SigmaDampingWarning on the near-pole space, one
    SigmaContinuationWarning on the coarse node set."""
    mf, _, eps, nocc = wat
    damped = _recorded(rpa.SigmaDampingWarning,
                       lambda: build_effective_hamiltonian(
                           low_exact, _high_virtual(eps, nocc), sigma='srg'))
    spread = _recorded(rpa.SigmaContinuationWarning,
                       lambda: build_effective_hamiltonian(
                           _coarse_imagfrequency(mf),
                           _contiguous_44(eps, nocc), sigma='srg'))
    assert len(damped) == 1 and len(spread) == 1


def test_warnings_point_at_the_caller(wat, low_exact):
    """The Sigma^E warnings of both entry points point at the line that
    called them (this file), not into rpa.py."""
    mf, _, eps, nocc = wat
    near, coarse = _high_virtual(eps, nocc), _coarse_imagfrequency(mf)
    space = _contiguous_44(eps, nocc)
    calls = [
        (rpa.SigmaDampingWarning,
         lambda: rpa.environment_self_energy(low_exact, near, 'srg')),
        (rpa.SigmaDampingWarning,
         lambda: build_effective_hamiltonian(low_exact, near, sigma='srg')),
        (rpa.SigmaContinuationWarning,
         lambda: rpa.environment_self_energy(coarse, space, 'srg')),
        (rpa.SigmaContinuationWarning,
         lambda: build_effective_hamiltonian(coarse, space, sigma='srg')),
        (rpa.SigmaNearPoleWarning,
         lambda: rpa.environment_self_energy(low_exact, near, 'averaged')),
        (rpa.SigmaNearPoleWarning,
         lambda: build_effective_hamiltonian(low_exact, near,
                                             sigma='averaged')),
    ]
    for category, call in calls:
        (rec,) = _recorded(category, call)
        assert rec.filename == __file__, (category.__name__, rec.filename)
