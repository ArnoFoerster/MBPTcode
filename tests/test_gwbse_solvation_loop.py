"""Self-consistent state-specific solvation of GW-BSE states:
`src.gradients.gwbse.state_model.GWBSEState` in the medium-independent loop
(`src.environments.solvation_loop`) against the continuum adapter
(`src.environments.reaction_field.ContinuumField`), with the gates and bars of the
ADC(2) twin (test_solvation_loop, test_solvation_force,
test_solvation_gas_limit, test_solvation_loop_checkpoint).

Water (bent, non-symmetric), cc-pVDZ, Hartree-Fock reference (density fitted,
cc-pvdz-jkfit), G0W0-BSE with the dense Casida solve on the G1 grid; toluene
IEF-PCM unless stated. S0 is the reference itself (E_HF, D_HF), S1 carries the
linear-response term (exchange at eps_inf), T1 none.

  1. the reference equilibrated by the loop is pyscf's equilibrium PCM-SCF:
     energy to 1e-10 Ha, force to 1e-8 Ha/Bohr.
  2. the contract: Tr(D_eff O) is the five-point finite-field derivative of
     the state's energy in h + lam O (h the operator of the converged
     charges), O = mu_z and a random symmetric matrix, S1 and T1, 1e-8.
     The same for the Tamm-Dancoff kernel (mu_z), S1 and T1.
  3. the first loop cycle is the first-order term: in the reference's
     equilibrium charges Omega is scheme B's omega_B on the PCM-SCF
     (1e-9 Ha), D_eff - D_ref is its relaxed frozen-field density (1e-8) and
     its E_SS in Q_s is the equilibrium E_SS (1e-10 Ha).
  4. stationarity: G has no first derivative in q at convergence (below 1e-7
     Ha per unit |dq|) and has one at the start (above 1e-5).
  5. a state in its own slow charges: G_neq(X | X) = G_eq(X), S0 and S1, 1e-9.
  6. eps_s = eps_inf: G_neq(S1 | S0) = G_eq(S1), 1e-9.
  7. G_neq is stationary in the fast charges (< 1e-7) and not at the start.
  8. weak continuum (eps_s 1.1, eps_inf 1.05): absorption = Omega(q_0) +
     (1/2) dV Q_f dV and absorption - emission = -dV (Q_s - Q_f) dV > 0, each
     within 10%; the equilibrium excitation is omega_B + E_SS(eq) within 10%,
     the remainder shrinking 1.5x or more from eps_s 1.2.
  9. eps = 1: G, E(q), the relaxed density, the cycle count and the force of
     G_eq equal the gas phase (`NoMedium`) to a few ulp (tests/pins.py
     assert_same: two loops stand on two sets of SCFs) for S0, S1 and T1, and
     S1's absorption and emission too (an eps_inf = 1 continuum answers with
     exact zeros and the BSE is the gas phase's, `LinearResponseOnly.vanishes`);
     at eps_s = 1 + delta (eps_inf = 1 + delta / 2) the shift of G_eq(S1)
     scales with delta: ratio at 1e-2 / 1e-3 within 10% of 10, nonzero at
     1e-3. The LR term reads eps_inf: at eps_s 30, eps_inf 2 the first
     cycle's omega is that of an optical continuum at 2, not at 30.
 10. forces of G_eq along random directions against central differences of
     converged loops (h = 5e-4 Bohr, each displaced point a whole loop on its
     own cavity, on the reference point's surface: `GWBSEState.at` with the
     displaced field's transition environment), S1 and T1 in toluene, S0, S1
     and T1 in the gas phase, 1e-6 Ha/Bohr; `at` refuses a field whose
     continuum is not the chain's, and a model without Numerics is refused.
 11. a Kohn-Sham reference (LRC-wPBEh): the S1 contract (mu_z) and its force
     along one direction.
 12. simulated ranks (each with its own molecule, as an MPI process): G_eq,
     q, D_eff, the cycle count and the force of S1 are rank 0's bits on every
     rank and the same on two and three ranks to a few ulp; against one rank 1e-10 (the
     ranks sum chunked kernels in pieces).
 13. one rank's SCF unconverged: every rank takes rank 0's mean field (the
     lockstep before the flag) and the answer is the uncapped run's to a few
     ulp; rank 0's unconverged SCF is flagged alike on every rank.
 14. on `SolvatedSurface`: a make_state building a fresh model per geometry
     (another surface) is refused; a walk's commit reaches the chain
     (conventions re-keyed, followed state moved); the surface's force is the
     central difference of its energy (toluene S1, 1e-6 Ha/Bohr).
 15. checkpoint: a loop stopped after three cycles and resumed by a fresh
     model and medium gives the uninterrupted G, history, charges, density and
     force bit for bit (S0 and S1 with its LR term); a checkpoint of another
     geometry or state is refused; a model moved with `at` restores only on
     its chain.
 16. refreeze (`SolvatedSurface.refreeze`): S1 in toluene frozen at R0 and
     rebuilt at R1 (0.05 Bohr off): a new surface whose G_eq at R1 is the
     vertical's (a model built afresh at R1) to a few ulp (two solves, 1e-15
     relative) and differs from the
     old surface's by more than 1e-8 Ha (the old no-op returned the surface
     itself); the rebuilt surface moves its one model on. The reference is
     its own refreeze. A T1 walk in toluene, refreeze=1 (formaldehyde, whose
     T1 is bound -- water's S1 and T1 dissociate --; TDA, Cartesian, nudged;
     loops at the driver's 1e-9 e / 1e-11 Ha on its SCF, 1e-12 / grad
     1e-10): the drift is measured at the minimum (0.16 meV), above 1e-8 Ha
     and within REFREEZE_TOL_MEV; it is the rebuilt surface's G_eq at R*
     (the vertical's, to a few ulp) minus the old one's; the record holds the
     passes; the final G_eq is the vertical's at the final minimum within
     REFREEZE_TOL_MEV (9e-13 Ha when recorded).

Run: python -m pytest tests/test_gwbse_solvation_loop.py
"""
import os
import sys

import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
from src.environments import solvation_loop as sl
from src.environments.environment import solvent_scheme
from src.Base.geometry.geometry import molecule_at
from src.gradients.rpa.chain_settings import with_setting
from src.Base.numerics.constants import (BOHR_TO_ANGSTROM, HARTREE_TO_MEV,
                                         REFREEZE_TOL_MEV)
from src.Base.numerics.numerics import Numerics
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_model import GWBSEState
from src.gradients.scf_loop_response import loop_density_response
from src.properties.optimize import relax
from src.gradients.solvated_surface import SolvatedSurface
from src.gradients.surface import commit
from tests.adc_common import (FD_STEP_SOLVATION, Stop, central, five_point,
                              stop_after)
from tests.isdf_test_grid import g1
from tests.pins import assert_same

ATOMS = [('O', (0.02, -0.01, 0.1173)), ('H', (0.0, 0.7672, -0.4592)),
         ('H', (0.01, -0.7472, -0.4792))]
JKFIT = 'cc-pvdz-jkfit'
#: Step (e, along a unit direction of the charges) of the slopes of G in q.
CHARGE_STEP = 1e-4


def molecule(shift=None):
    coords = np.array([xyz for _, xyz in ATOMS])
    if shift is not None:
        coords = coords + shift * BOHR_TO_ANGSTROM
    return gto.M(atom=[(s, tuple(c)) for (s, _), c in zip(ATOMS, coords)],
                 basis='cc-pvdz', verbose=0)


def numerics(mol):
    """The one `Numerics` of a test: the chain's (dense Casida, G1) and the
    loop's (tol_q 1e-10 e, tol_energy 1e-12 Ha)."""
    return Numerics(bse={'solver': 'dense'}, isdf={'counts': g1(mol)},
                    solvent={'loop_tol_q': 1e-10, 'loop_tol_energy': 1e-12})


def loop(state, field, **kw):
    """equilibrate at the test's tolerances (the state's own Numerics); a
    model whose first call freezes its chain starts from S0's equilibrium
    charges (`s0_charges`) unless q0 is given."""
    if ('q0' not in kw and field.nsites
            and getattr(state, 'freezes_on_first_call', False)):
        kw['q0'] = s0_charges(state, field)
    return sl.equilibrate(state, field, numerics=state.numerics, **kw)


def ground_of(state):
    """The reference (S0) model of `state`'s mean field, nuclei and
    Numerics."""
    return GWBSEState(state.mol, state.make_scf, numerics=state.numerics)


def s0_charges(state, field):
    """S0's equilibrium charges in `field` (the start of an excited loop)."""
    return loop(ground_of(state), field).q


def hcore_shift(model):
    """What the mean field the chain of `model` froze its conventions on
    added to the gas-phase core Hamiltonian: the operator of the charges its
    first call was in."""
    mf0, mol = model.chain.mf0, model.mol
    return mf0.get_hcore() - (mol.intor('int1e_kin') + mol.intor('int1e_nuc'))


def make_scf(mol, xc='hf'):
    mf = (scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)).density_fit(JKFIT)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-15, 1e-12, 200, 0
    return mf


def screening(mol, eps_static=None, eps_inf=None):
    if eps_static is None:
        return SolventScreening(mol, solvent='toluene')
    return SolventScreening(mol, eps=eps_inf, eps_static=eps_static,
                            allow_static_eps=True)


def setup(mol, kind, field=None, xc='hf', num=None, scf_factory=None, tda=False, **eps):
    """(field, state) with kind 'ground', 'singlet' or 'triplet'."""
    field = field or ContinuumField.from_environment(screening(mol, **eps))
    lr = field.transition_environment() if kind == 'singlet' else None
    state = GWBSEState(mol, scf_factory or (lambda m: make_scf(m, xc)),
                       None if kind == 'ground' else kind, 0, bse_tda=tda,
                       lr_environment=lr, numerics=num or numerics(mol))
    return field, state


@pytest.fixture(scope='module')
def mol():
    return molecule()


@pytest.fixture(scope='module')
def loops(mol):
    """{kind: (field, state, converged loop)} in toluene."""
    out = {}
    for kind in ('ground', 'singlet', 'triplet'):
        field, state = setup(mol, kind)
        r = loop(state, field)
        assert r.converged, kind
        out[kind] = (field, state, r)
    return out


def test_the_reference_loop_is_the_pcm_scf(mol, loops):
    field, state, r = loops['ground']
    mf = SolventScreening(mol, solvent='toluene').mean_field(
        mol, lambda m: make_scf(m).run())
    assert abs(r.energy - mf.e_tot) < 1e-10, (r.energy, mf.e_tot)
    g = mf.nuc_grad_method()
    g.verbose = 0
    d = np.abs(sl.force(state, field, r) - g.kernel()).max()
    assert d < 1e-8, d


def test_the_ground_loop_response_is_its_references(loops):
    """S0's loop response is `HFState.loop_response` of its reference on the
    mean field of its last call, the same bits as the response formed on
    that mean field directly; an excited state has none."""
    field, state, _ = loops['ground']
    mu = 1e-3 * np.cos(np.arange(field.nsites))
    h = field.operator(mu)[0]
    want = loop_density_response(state.mf, state.reference._h0,
                                 field.scf_continuum(), h, state.numerics)
    assert np.array_equal(state.loop_response(h, field, state.numerics), want)
    with pytest.raises(ValueError, match='reference'):
        loops['singlet'][1].loop_response(h, field, state.numerics)


@pytest.mark.parametrize('kind', ['singlet', 'triplet'])
def test_the_density_is_the_field_derivative(mol, loops, kind):
    field, state, r = loops[kind]
    h, e = field.operator(r.q)
    rng = np.random.default_rng(3)
    a = rng.standard_normal((mol.nao, mol.nao))
    ops = {'mu_z': mol.intor('int1e_r')[2], 'random': 0.05 * (a + a.T)}
    dm = state(h, e).dm_eff
    for name, op in ops.items():
        fd = five_point(lambda lam: state(h + lam * op, e).energy)
        an = float(np.einsum('mn,mn', dm, op))
        assert abs(fd - an) < 1e-8, (kind, name, an, fd)


@pytest.mark.parametrize('kind', ['singlet', 'triplet'])
def test_the_tda_density_is_the_field_derivative(mol, kind):
    """The Tamm-Dancoff kernel (what walks run on): its loop converges in
    toluene and Tr(D_eff mu_z) is the five-point derivative of its energy."""
    field, state = setup(mol, kind, tda=True)
    r = loop(state, field)
    h, e = field.operator(r.q)
    rz = mol.intor('int1e_r')[2]
    fd = five_point(lambda lam: state(h + lam * rz, e).energy)
    an = float(np.einsum('mn,mn', state(h, e).dm_eff, rz))
    assert abs(fd - an) < 1e-8, (kind, an, fd)


def test_the_first_cycle_is_the_first_order_term(mol, loops):
    """In S0's equilibrium charges the S1 state model IS scheme B on the
    PCM-SCF: same Omega, same relaxed density, same E_SS."""
    field, _, r0 = loops['ground']
    _, state = setup(mol, 'singlet', field=field)
    first = state(*field.operator(r0.q))
    optical = SolventScreening(mol, solvent='toluene')
    env = solvent_scheme(optical, 'classical-ss', ss_equilibrium=True)
    mf = env.mean_field(mol, lambda m: make_scf(m).run())
    chain = ExcitedStateChain(mol, None, mf=mf, spin='singlet',
                              environment=env, numerics=numerics(mol))
    rec = chain.state_specific_correction(mol, mf)
    assert abs(first.info['omega'] - rec['omega_B']) < 1e-9
    drho = chain.excitation_density(mol, mf)
    assert np.abs(first.info['delta_density'] - drho).max() < 1e-8
    e_ss = env.static.state_specific_energy(mol, first.info['delta_density'])[0]
    assert abs(e_ss - rec['E_SS']) < 1e-10, (e_ss, rec['E_SS'])
    # the same Q: the loop's static response is the continuum's
    assert np.abs(field.response_matrix('static')
                  - env.static.response_matrix()).max() < 1e-12


@pytest.mark.parametrize('kind', ['singlet', 'triplet'])
def test_g_is_stationary_at_convergence(mol, loops, kind):
    field, state, r = loops[kind]
    d = np.random.default_rng(5).standard_normal(r.q.shape)
    d /= np.linalg.norm(d)
    at = central(lambda t: sl.free_energy(state, field, r.q + t * d), CHARGE_STEP)
    q0 = loops['ground'][2].q      # S0's charges: not S1's stationary point
    start = central(lambda t: sl.free_energy(state, field, q0 + t * d), CHARGE_STEP)
    assert abs(at) < 1e-7, at
    assert abs(start) > 1e-5, start


@pytest.mark.parametrize('kind', ['ground', 'singlet'])
def test_a_state_in_its_own_slow_charges(mol, loops, kind):
    field, state, r = loops[kind]
    neq = loop(state, field, slow=sl.slow_charges(field, r.dm_eff))
    assert neq.converged and abs(neq.energy - r.energy) < 1e-9


def test_no_slow_charges_when_eps_static_is_eps_inf(mol):
    field, _ = setup(mol, 'ground', eps_static=2.0, eps_inf=2.0)
    _, s0 = setup(mol, 'ground', field=field)
    g0 = loop(s0, field)
    _, sx = setup(mol, 'singlet', field=field)
    gx = loop(sx, field)
    neq = loop(sx, field, slow=sl.slow_charges(field, g0.dm_eff))
    assert abs(neq.energy - gx.energy) < 1e-9


def test_g_neq_is_stationary_in_the_fast_charges(mol, loops):
    r0 = loops['ground'][2]
    field, state, _ = loops['singlet']
    slow = sl.slow_charges(field, r0.dm_eff)
    neq = loop(state, field, slow=slow)
    d = np.random.default_rng(7).standard_normal(neq.q.shape)
    d /= np.linalg.norm(d)
    at = central(lambda t: sl.free_energy(state, field, neq.q + t * d, slow), CHARGE_STEP)
    start = central(lambda t: sl.free_energy(state, field, r0.q + t * d, slow), CHARGE_STEP)
    assert abs(at) < 1e-7, at
    assert abs(start) > 1e-5, start


def first_order(mol, eps_static):
    """(equilibrium excitation, omega_B + E_SS(eq)) at eps_s, eps_inf =
    1 + (eps_s - 1) / 2, and the vertical energies with their first-order
    estimates."""
    eps = dict(eps_static=eps_static, eps_inf=1 + 0.5 * (eps_static - 1))
    field, s0 = setup(mol, 'ground', **eps)
    g0 = loop(s0, field)
    _, sx = setup(mol, 'singlet', field=field)
    first = sx(*field.operator(g0.q))
    dv = field.electronic_potential(first.info['delta_density'])
    Qs, Qf = field.response_matrix('static'), field.response_matrix('fast')
    omega = first.info['omega']
    vert = sl.vertical_energies(lambda: setup(mol, 'ground', field=field)[1],
                                lambda: setup(mol, 'singlet', field=field)[1],
                                field, numerics=numerics(mol))
    eq = vert['G']['eq_state'] - vert['G']['eq_ground']
    return dict(eq=eq, eq_1=omega + 0.5 * dv @ Qs @ dv,
                absorption=vert['absorption'],
                absorption_1=omega + 0.5 * dv @ Qf @ dv,
                stokes=vert['absorption'] - vert['emission'],
                stokes_1=-dv @ (Qs - Qf) @ dv, omega=omega)


def test_a_weak_continuum_is_first_order(mol):
    weak, weaker = first_order(mol, 1.2), first_order(mol, 1.1)
    shift = abs(weaker['eq_1'] - weaker['omega'])
    assert abs(weaker['eq'] - weaker['eq_1']) < 0.1 * shift
    assert (abs(weak['eq'] - weak['eq_1'])
            > 1.5 * abs(weaker['eq'] - weaker['eq_1']))
    a_shift = abs(weaker['absorption_1'] - weaker['omega'])
    assert abs(weaker['absorption'] - weaker['absorption_1']) < 0.1 * a_shift
    assert weaker['stokes'] > 0
    assert abs(weaker['stokes'] - weaker['stokes_1']) < 0.1 * weaker['stokes_1']


@pytest.mark.parametrize('kind', ['ground', 'singlet', 'triplet'])
def test_eps_one_is_the_gas_phase(mol, kind):
    one = ContinuumField.from_environment(
        SolventScreening(mol, eps=1.0, eps_static=1.0, allow_static_eps=True))
    assert not one.response_matrix('static').any()
    assert not one.response_matrix('fast').any()
    _, s1 = setup(mol, kind, field=one)
    r1 = loop(s1, one)
    gas = NoMedium(mol)
    _, sg = setup(mol, kind, field=gas)
    rg = loop(sg, gas)
    # two loops on two sets of SCFs: the same numbers to a few ulp, never
    # bitwise across two SCF runs
    assert len(r1.history) == len(rg.history)
    assert_same(r1.energy, rg.energy)
    assert_same(r1.state_energy, rg.state_energy)
    assert_same(r1.dm_eff, rg.dm_eff, unit='1', quantity='vector')
    assert_same(sl.force(s1, one, r1), sl.force(sg, gas, rg), quantity='gradient')
    if kind == 'singlet':
        vert = lambda field: sl.vertical_energies(
            lambda: setup(mol, 'ground', field=field)[1],
            lambda: setup(mol, 'singlet', field=field)[1], field,
            numerics=numerics(mol))
        a, b = vert(one), vert(gas)
        assert_same(a['absorption'], b['absorption'])
        assert_same(a['emission'], b['emission'])


def test_the_continuum_leaves_the_gas_phase_linearly(mol):
    """eps_s = 1 + delta, eps_inf = 1 + delta / 2: the shift of G_eq(S1)
    against the gas phase scales with delta (the LR path running, unlike at
    eps = 1 itself)."""
    gas = NoMedium(mol)
    _, sg = setup(mol, 'singlet', field=gas)
    g_gas = loop(sg, gas).energy

    def shift(delta):
        field, s = setup(mol, 'singlet', eps_static=1 + delta,
                         eps_inf=1 + delta / 2)
        return loop(s, field).energy - g_gas
    big, small = shift(1e-2), shift(1e-3)
    assert small != 0
    assert abs(big / small - 10) < 1.0, big / small


@pytest.mark.parametrize('kind,medium', [('singlet', 'toluene'),
                                         ('triplet', 'toluene'),
                                         ('ground', 'gas'),
                                         ('singlet', 'gas'),
                                         ('triplet', 'gas')])
def test_the_force_of_g_eq(mol, loops, kind, medium):
    if medium == 'toluene':
        field, state, r = loops[kind]
        fields = lambda m: ContinuumField.from_environment(screening(m))
    else:
        field, state = setup(mol, kind, field=NoMedium(mol))
        r = loop(state, field)
        fields = NoMedium
    F = sl.force(state, field, r)
    d = np.random.default_rng(17).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)

    def G(shift):
        m = molecule(shift * d)
        f = fields(m)
        lr = f.transition_environment() if kind == 'singlet' else None
        out = loop(state.at(m, make_scf, lr), f)
        assert out.converged
        return out.energy
    fd = central(G, FD_STEP_SOLVATION)
    an = float(np.sum(F * d))
    assert abs(fd - an) < 1e-6, (an, fd)


def test_a_kohn_sham_reference(mol):
    xc = 'lrc-wpbeh'
    field, state = setup(mol, 'singlet', xc=xc)
    r = loop(state, field)
    assert r.converged
    h, e = field.operator(r.q)
    rz = mol.intor('int1e_r')[2]
    fd = five_point(lambda lam: state(h + lam * rz, e).energy)
    an = float(np.einsum('mn,mn', state(h, e).dm_eff, rz))
    assert abs(fd - an) < 2e-8, (an, fd)
    F = sl.force(state, field, r)
    d = np.random.default_rng(19).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)

    def G(shift):
        m = molecule(shift * d)
        f = ContinuumField.from_environment(screening(m))
        out = loop(state.at(m, lambda mm: make_scf(mm, xc), f.transition_environment()),
                   f)
        return out.energy
    fd = central(G, FD_STEP_SOLVATION)
    assert abs(fd - float(np.sum(F * d))) < 1e-6


def test_a_moved_model_refuses_another_continuum(mol, loops):
    """`at` takes the displaced field's transition environment and refuses one
    that is not the chain's continuum (another solvent, or none)."""
    _, state, _ = loops['singlet']
    m = molecule(np.full((3, 3), 1e-3))
    water = ContinuumField.from_environment(SolventScreening(m, solvent='water'))
    for lr in (water.transition_environment(), None):
        with pytest.raises(ValueError, match="chain answers in"):
            state.at(m, make_scf, lr)


def test_a_model_without_numerics_is_refused(mol):
    with pytest.raises(ValueError, match='needs its Numerics'):
        GWBSEState(mol, make_scf, 'singlet')


def test_the_loop_takes_one_numerics(mol):
    field, state = setup(mol, 'ground', field=NoMedium(mol))
    with pytest.raises(ValueError, match='one object'):
        sl.equilibrate(state, field, numerics=Numerics())


def test_the_lr_term_reads_eps_inf(mol):
    """eps_s 30, eps_inf 2: in S0's equilibrium charges the S1 model's omega
    is that of a model whose LR term reads an optical continuum at 2, and not
    that of one at 30 (the shifts are linear in delta near eps = 1, so only a
    large gap between the two constants tells them apart)."""
    eps = dict(eps_static=30.0, eps_inf=2.0)
    field, s0 = setup(mol, 'ground', **eps)
    h, e = field.operator(loop(s0, field).q)
    _, sx = setup(mol, 'singlet', field=field)
    omega = sx(h, e).info['omega']

    def at_eps(value):
        optical = SolventScreening(mol, eps=value, allow_static_eps=True)
        model = GWBSEState(mol, make_scf, 'singlet', lr_environment=optical,
                           numerics=numerics(mol))
        return model(h, e).info['omega']
    assert abs(omega - at_eps(2.0)) < 1e-9
    assert abs(omega - at_eps(30.0)) > 1e-4


def test_the_state_cannot_move_before_its_surface_is_fixed(mol):
    _, state = setup(mol, 'singlet', field=NoMedium(mol))
    with pytest.raises(ValueError, match='first evaluation'):
        state.at(molecule(), make_scf)


# --- every chain is built in S0's equilibrium charges ----------------------

def test_a_loop_refuses_to_freeze_a_chain_at_no_charges(mol):
    """A loop of a model whose first call freezes its chain, started from no
    charges in a medium with sites: refused (the chain would freeze its
    conventions on the zero-charge mean field)."""
    field, state = setup(mol, 'singlet')
    assert state.freezes_on_first_call
    with pytest.raises(ValueError, match='freezes its surface'):
        sl.equilibrate(state, field, numerics=state.numerics)


def test_a_verticals_chain_is_built_in_s0_charges(mol):
    """The chain of a vertical's state (absorption, emission) froze its
    conventions on the mean field converged in S0's equilibrium charges: its
    core Hamiltonian is the gas one plus the operator of those charges (it
    was the gas one, the loop's first call at no charges)."""
    field = ContinuumField.from_environment(screening(mol))
    models = []

    def state():
        models.append(setup(mol, 'singlet', field=field)[1])
        if len(models) > 1:
            # the second loop of the state stands on the first's chain
            models[-1] = models[0].at(mol, make_scf, field.transition_environment())
        return models[-1]
    r = sl.vertical_energies(lambda: setup(mol, 'ground', field=field)[1],
                             state, field, numerics=numerics(mol))
    h = field.operator(r['loops']['eq_ground'].q)[0]
    assert np.abs(h).max() > 1e-4
    assert np.abs(hcore_shift(models[0]) - h).max() < 1e-12


def test_a_surfaces_chain_is_built_in_s0_charges(mol):
    """The solvated surface's first geometry and a refrozen surface's: the
    chain stands on the mean field in S0's equilibrium charges there."""
    surf, ref = _toluene_surface(mol, 'singlet')
    surf.total_energy(mol)
    field = ContinuumField.from_environment(screening(mol))
    h = field.operator(s0_charges(ref[0], field))[0]
    assert np.abs(hcore_shift(ref[0]) - h).max() < 1e-10
    r1 = molecule(_offset(31))
    rebuilt = surf.refreeze(r1)
    rebuilt.total_energy(r1)
    f1 = ContinuumField.from_environment(screening(r1))
    model = rebuilt.last['state']
    h1 = f1.operator(s0_charges(model, f1))[0]
    assert np.abs(hcore_shift(model) - h1).max() < 1e-10


def test_a_surface_without_its_ground_is_refused(mol):
    """A solvated surface of a GW-BSE excited state with no S0 model to
    start its first loop from: the loop refuses it."""
    ref = []

    def make_state(m, field):
        ref.append(setup(m, 'singlet', field=field)[1])
        return ref[0]
    surf = SolvatedSurface(
        mol, lambda m: ContinuumField.from_environment(screening(m)), make_state,
        label='S1 toluene', numerics=numerics(mol))
    with pytest.raises(ValueError, match='freezes its surface'):
        surf.total_energy(mol)


def _on_rank(comm):
    """S1 in toluene on one rank, with its own molecule, medium and state, as
    an MPI process has: simulated ranks share one process, and a `Mole`
    shared between them shares the caches hung on it."""
    m = molecule()
    field, state = setup(m, 'singlet')
    r = loop(state, field)
    return (r.energy, r.q, r.dm_eff, len(r.history),
            sl.force(state, field, r))


def test_ranks_reproduce_one_rank(loops):
    """Every rank holds rank 0's bits; two ranks equal three; one rank's
    serial kernels sum in one piece where the ranks sum in chunks, so the
    distributed answer is held to 1e-10 against serial, the cycle
    count to one (measured 19 vs 18: the round-off of the chunked sums tips
    the last cycle at tol_energy 1e-12)."""
    field0, state0, serial = loops['singlet']
    f_serial = sl.force(state0, field0, serial)
    runs = {size: run_simulated(_on_rank, size) for size in (2, 3)}
    for size, out in runs.items():
        for rank in out[1:]:
            assert rank[0] == out[0][0] and rank[3] == out[0][3]
            for a, b in zip(rank[1:], out[0][1:]):
                assert np.array_equal(a, b), size
    two, three = runs[2][0], runs[3][0]
    # two runs, each on its own SCFs: to a few ulp, not bitwise
    assert_same(two[0], three[0])
    assert_same(two[4], three[4], quantity='gradient')
    assert abs(two[0] - serial.energy) < 1e-10
    # both loops start from the same charges (S0's, bitwise equal) and stop
    # within loop_tol_q = 1e-10 of the fixed point, at charges that differ
    # by 3e-11 (cluster) between chunked and serial sums; the force follows
    # the charges: 1e-11 on a workstation, 1.7e-10 on the cluster
    assert np.abs(two[4] - f_serial).max() < 5e-10
    # the chunked sums' round-off can tip the last cycle at tol_energy 1e-12
    assert abs(two[3] - len(serial.history)) <= 1


# --- the reference's lockstep comes before its flag ------------------------

def _capped_on_rank(comm, capped_rank):
    """The reference in toluene on one rank, the SCF of `capped_rank` stopped
    after two cycles (unconverged there, converged on the others)."""
    m = molecule()

    def capped(mm):
        mf = make_scf(mm)
        if comm.Get_rank() == capped_rank:
            mf.max_cycle = 2
        return mf
    field, state = setup(m, 'ground', scf_factory=capped)
    r = loop(state, field)
    return r.energy, r.q, r.dm_eff, len(r.history), [f['flag'] for f in r.flags]


def test_one_ranks_unconverged_scf_takes_rank_zeros_answer():
    """Rank 1's SCF alone fails: every rank holds rank 0's converged mean
    field (lockstep, then the flag), so no rank flags while the others carry
    on, and the answer is the uncapped run's."""
    capped = run_simulated(_capped_on_rank, 2, 1)
    free = run_simulated(_capped_on_rank, 2, -1)
    # rank 1 holds rank 0's copy: bitwise
    for a, b in zip(capped[1][:4], capped[0][:4]):
        assert np.array_equal(a, b)
    # the uncapped run is another run on its own SCFs: to a few ulp
    e, q, dm, n = capped[0][:4]
    e_free, q_free, dm_free, n_free = free[0][:4]
    assert n == n_free
    assert_same(e, e_free)
    assert_same(q, q_free, unit='1', quantity='vector')
    assert_same(dm, dm_free, unit='1', quantity='vector')
    assert capped[0][4] == capped[1][4] == free[0][4] == []


def test_rank_zeros_unconverged_scf_is_flagged_on_every_rank():
    """Rank 0's SCF stops short: a numerical event (it warns,
    it does not refuse), flagged alike on every rank, the loop
    carrying on in step."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = run_simulated(_capped_on_rank, 2, 0)
    assert 'scf_not_converged' in out[0][4] and out[0][4] == out[1][4]
    assert out[0][0] == out[1][0] and out[0][3] == out[1][3]


# --- one surface on SolvatedSurface ---------------------------------

def _gas_surface(mol, fresh):
    """S1 in the gas phase as a SolvatedSurface (NoMedium: the loop is cheap)
    whose make_state moves one reference model, or builds a fresh one per
    geometry (`fresh`); track='overlap' so the chain follows a state."""
    num = with_setting(numerics(mol), 'bse.track', 'overlap')
    ref = []

    def make_state(m, field):
        if fresh or not ref:
            ref.append(setup(m, 'singlet', field=field, num=num)[1])
            return ref[-1]
        return ref[0].at(m, make_scf, field.transition_environment())
    return SolvatedSurface(mol, NoMedium, make_state, label='S1 gas',
                           numerics=num), ref


def test_a_fresh_model_per_geometry_is_refused(mol):
    """A make_state that builds a GW-BSE model afresh at each geometry is
    another surface: refused before its loop runs."""
    surf, _ = _gas_surface(mol, fresh=True)
    surf.total_energy(mol)
    with pytest.raises(ValueError, match='another surface'):
        surf.total_energy(molecule(np.full((3, 3), 1e-3)))


def test_a_walks_commit_reaches_the_chain(mol):
    """The walkers' commit(surface, mol) reaches the chain through the state
    model: its conventions are re-keyed and its followed state
    moves to the accepted geometry."""
    surf, ref = _gas_surface(mol, fresh=False)
    surf.total_gradient(mol)
    chain = ref[0].chain
    moved = molecule(np.full((3, 3), 2e-3))
    surf.total_gradient(moved)
    assert chain.conventions.commits == 0
    commit(surf, moved)
    assert chain.conventions.commits == 1
    assert np.array_equal(chain.followed[0].atom_coords(), moved.atom_coords())


def test_the_solvated_surface_force_is_its_energy_derivative(mol):
    """GW-BSE S1 in toluene on the SolvatedSurface, its make_state the
    reference model moved (`at`): the surface's force along a random unit
    direction against the central difference of its energy (each point a
    converged loop on its own cavity), 1e-6 Ha/Bohr."""
    ref = []

    def make_state(m, field):
        if not ref:
            ref.append(setup(m, 'singlet', field=field)[1])
            return ref[0]
        return ref[0].at(m, make_scf, field.transition_environment())
    surf = SolvatedSurface(
        mol, lambda m: ContinuumField.from_environment(screening(m)), make_state,
        label='S1 toluene', numerics=numerics(mol),
        make_ground=lambda m, f: setup(m, 'ground', field=f)[1])
    F = surf.total_gradient(mol)[0]
    d = np.random.default_rng(23).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)
    fd = central(lambda t: surf.total_energy(molecule(t * d)), FD_STEP_SOLVATION)
    assert abs(fd - float(np.sum(F * d))) < 1e-6


# --- a loop resumed from its checkpoint ------------------------

def same_history(got, want):
    """Two loops' cycle histories [(cycle, G, max |dq|)]: the cycles exactly,
    G and max |dq| to a few ulp (two computations)."""
    assert [c[0] for c in got] == [c[0] for c in want]
    assert_same([c[1] for c in got], [c[1] for c in want])
    assert_same([c[2] for c in got], [c[2] for c in want], unit='1')

@pytest.mark.parametrize('kind', ['ground', 'singlet'])
def test_a_resumed_loop_is_the_uninterrupted_one(mol, tmp_path, kind):
    """A loop stopped after three cycles and resumed by a fresh state model and
    medium (as a new process would build them) gives the uninterrupted run's
    G, charges, density, cycle history and force to a few ulp (a resumed run
    solves its SCFs again: not bitwise); the GW-BSE model
    rebuilds its chain from the first call the checkpoint holds."""
    field, state = setup(mol, kind)
    # one start for every run: the checkpoint's key holds it
    start = {} if kind == 'ground' else {'q0': s0_charges(state, field)}
    whole = loop(state, field, **start)
    path = str(tmp_path / f'{kind}.json')
    f1, s1 = setup(mol, kind)
    with pytest.raises(Stop):
        loop(s1, f1, checkpoint=path, log=stop_after(3), **start)
    f2, s2 = setup(mol, kind)
    part = loop(s2, f2, checkpoint=path, **start)
    assert len(part.history) == len(whole.history) > 4
    same_history(part.history, whole.history)
    assert_same(part.energy, whole.energy)
    assert_same(part.q, whole.q, unit='1', quantity='vector')
    assert_same(part.dm_eff, whole.dm_eff, unit='1', quantity='vector')
    assert_same(sl.force(s2, f2, part), sl.force(state, field, whole),
                quantity='gradient')


def test_a_checkpoint_of_another_loop_is_refused(mol, tmp_path):
    path = str(tmp_path / 'ground.json')
    field, state = setup(mol, 'ground')
    with pytest.raises(Stop):
        loop(state, field, checkpoint=path, log=stop_after(1))
    m = molecule(np.full((3, 3), 1e-3))
    f2, s2 = setup(m, 'ground')
    with pytest.raises(ValueError, match='another loop'):
        loop(s2, f2, checkpoint=path)


def test_a_checkpoint_of_another_state_is_refused(mol, tmp_path):
    """The singlet's checkpoint resumed by the triplet: refused (the state's
    identity is in the checkpoint's key). In toluene: a gas-phase loop ends in
    its first cycle and writes no checkpoint."""
    path = str(tmp_path / 'singlet.json')
    field, state = setup(mol, 'singlet')
    with pytest.raises(Stop):
        loop(state, field, checkpoint=path, log=stop_after(1))
    _, other = setup(mol, 'triplet', field=field)
    with pytest.raises(ValueError, match='another loop'):
        loop(other, field, checkpoint=path)


def test_a_moved_model_restores_only_on_its_chain(mol, tmp_path):
    """A model moved with `at` holds no chain of its own: its checkpoint
    cannot rebuild one, and a fresh model is refused. In toluene: a gas-phase
    loop ends in its first cycle and writes no checkpoint."""
    field, ref = setup(mol, 'singlet')
    loop(ref, field)
    m = molecule(np.full((3, 3), 1e-3))
    f2 = ContinuumField.from_environment(screening(m))
    lr = f2.transition_environment()
    moved = ref.at(m, make_scf, lr)
    path = str(tmp_path / 'moved.json')
    _, fresh = setup(m, 'singlet', field=f2)
    q0 = s0_charges(fresh, f2)   # the checkpoint key holds q0: one start for all
    with pytest.raises(Stop):
        loop(moved, f2, q0=q0, checkpoint=path, log=stop_after(1))
    with pytest.raises(ValueError, match='no chain of its own'):
        loop(fresh, f2, q0=q0, checkpoint=path)
    again = loop(ref.at(m, make_scf, lr), f2, q0=q0, checkpoint=path)
    same_history(again.history, loop(ref.at(m, make_scf, lr), f2, q0=q0).history)


# --- refreeze on SolvatedSurface -----------------------------------

#: Off the reference geometry (Bohr, seeded): far enough that the conventions
#: frozen there are not those of the geometry walked to.
REFREEZE_OFFSET = 0.05


#: Two independent solves of one quantity agree to a few ulp, not bitwise
#: (BLAS reduction order varies run to run).
SAME_SOLVE_RTOL = 1e-15


def same_solve(a, b):
    return abs(a - b) <= SAME_SOLVE_RTOL * abs(b)


def _offset(seed):
    d = np.random.default_rng(seed).standard_normal((3, 3))
    return REFREEZE_OFFSET * d / np.abs(d).max()


def walk_scf(mol):
    """The driver's SCF (conv_tol 1e-12, conv_tol_grad 1e-10): at 1e-15 an SCF
    at a walk's trial step can miss its tolerance."""
    mf = make_scf(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    return mf


def walk_numerics(mol):
    """`numerics` at the solvated photophysics driver's loop tolerances
    (tol_q 1e-9 e, tol_energy 1e-11 Ha), which a walk's trial steps reach
    (at 1e-10 a loop at a trial step stalls near 7e-9 e)."""
    return Numerics(bse={'solver': 'dense'}, isdf={'counts': g1(mol)},
                    solvent={'loop_tol_q': 1e-9, 'loop_tol_energy': 1e-11})


def _toluene_surface(m0, kind):
    """kind in toluene as a SolvatedSurface whose make_state builds one model
    at its first geometry and moves it (`at`), as the driver's walks do."""
    ref = []

    def make_state(m, field):
        if not ref:
            ref.append(setup(m, kind, field=field)[1])
            return ref[0]
        lr = field.transition_environment() if kind == 'singlet' else None
        return ref[0].at(m, make_scf, lr)
    return SolvatedSurface(
        m0, lambda m: ContinuumField.from_environment(screening(m)), make_state,
        label=f'{kind} toluene', numerics=numerics(m0),
        make_ground=lambda m, f: setup(m, 'ground', field=f)[1]), ref


def _vertical_g(m, kind):
    """G_eq of kind at m on a model built afresh there (a vertical's)."""
    field, state = setup(m, kind)
    return loop(state, field).energy


def test_a_refreeze_is_the_vertical_surface(mol):
    surf, ref = _toluene_surface(mol, 'singlet')
    surf.total_energy(mol)
    r1 = molecule(_offset(31))
    old = surf.total_energy(r1)
    rebuilt = surf.refreeze(r1)
    assert rebuilt is not surf
    new = rebuilt.total_energy(r1)
    assert same_solve(new, _vertical_g(r1, 'singlet'))
    assert abs(new - old) > 1e-8
    chain = rebuilt.last['state'].chain
    assert chain is not ref[0].chain
    rebuilt.total_energy(molecule(_offset(31) + 1e-3))
    assert rebuilt.last['state'].chain is chain


def test_the_reference_is_its_own_refreeze(mol):
    surf, _ = _toluene_surface(mol, 'ground')
    surf.total_energy(mol)
    assert surf.refreeze(molecule(_offset(31))) is surf


def ao_numerics(mol):
    """`numerics` with the AO-product fit on 200 points pivoted from the G1
    parent (the solvated driver's --isdf-test-set / --pivot-points)."""
    return Numerics(bse={'solver': 'dense'},
                    isdf={'counts': g1(mol), 'test_set': 'ao_products',
                          'pivot_points': 200},
                    solvent={'loop_tol_q': 1e-10, 'loop_tol_energy': 1e-12})


def _ao_surface(m0):
    """S1 in toluene on the AO-product fit, one model moved as a walk's."""
    ref = []

    def make_state(m, field):
        if not ref:
            ref.append(setup(m, 'singlet', field=field, num=ao_numerics(m0))[1])
            return ref[0]
        return ref[0].at(m, make_scf, field.transition_environment())
    return SolvatedSurface(
        m0, lambda m: ContinuumField.from_environment(screening(m)), make_state,
        label='S1 toluene', numerics=ao_numerics(m0),
        make_ground=lambda m, f: setup(m, 'ground', field=f,
                                       num=ao_numerics(m0))[1]), ref


def test_the_ao_fit_solvated_force_is_its_energy_derivative(mol):
    """The SolvatedSurface force on the AO-product fit (points pivoted once,
    carried by their atoms) against the central difference of its energy along
    a random unit direction, 1e-6 Ha/Bohr, as on the tabulated grid."""
    surf, _ = _ao_surface(mol)
    F = surf.total_gradient(mol)[0]
    d = np.random.default_rng(29).standard_normal((mol.natm, 3))
    d /= np.linalg.norm(d)
    fd = central(lambda t: surf.total_energy(molecule(t * d)), FD_STEP_SOLVATION)
    assert abs(fd - float(np.sum(F * d))) < 1e-6


def test_an_ao_fit_refreeze_pivots_again(mol):
    """A refreeze on the AO-product fit is the vertical surface at the new
    geometry: its points are pivoted there, not the moved reference's."""
    surf, ref = _ao_surface(mol)
    surf.total_energy(mol)
    r1 = molecule(_offset(31))
    old = surf.total_energy(r1)
    rebuilt = surf.refreeze(r1)
    new = rebuilt.total_energy(r1)
    field, state = setup(r1, 'singlet', num=ao_numerics(mol))
    assert same_solve(new, loop(state, field).energy)
    assert abs(new - old) > 1e-8
    chain = rebuilt.last['state'].chain
    assert chain is not ref[0].chain
    assert chain.factorization.test_set == 'ao_products'


#: The row fit's tile edge of the rows-vs-whole gate: 200 points in four
#: tiles.
ROW_TILE = 64
#: Rows vs whole, relative to the largest force component: the two
#: realizations of this fit differ by 1.0e-10 already in the gas phase (the
#: S1 force here, its tiled against its whole Cholesky); scheme B measured
#: 0.8-1.0e-10, and 0.78 without the bare root's adjoint.
ROWS_VS_WHOLE_RTOL = 2e-10


def fit_numerics(mol, fit):
    """`ao_numerics` with the fit realized `fit` ('rows' or 'replicated'),
    sliced, the rows in tiles of ROW_TILE points."""
    layout = {'sliced': True}
    if fit == 'rows':
        layout['fit_block'] = ROW_TILE
    return Numerics(bse={'solver': 'dense'},
                    isdf={'counts': g1(mol), 'test_set': 'ao_products',
                          'pivot_points': 200, 'fit': fit},
                    layout=layout,
                    solvent={'loop_tol_q': 1e-10, 'loop_tol_energy': 1e-12})


def _scheme_b_singlet(fit):
    """(energy, relaxed density, force) of the scheme-B singlet (its
    linear-response gauge beside the bare one) in toluene, in S0's
    equilibrium charges, on the AO-product fit realized `fit`; one molecule,
    medium and model per call (per simulated rank)."""
    m = molecule()
    field, state = setup(m, 'singlet', num=fit_numerics(m, fit))
    h, e = field.operator(s0_charges(state, field))
    r = state(h, e)
    F = state.force(h, e)[0]
    return r.energy, np.array(r.dm_eff), np.array(F)


def test_scheme_b_runs_on_the_row_fit():
    """The scheme-B singlet's relaxed density and force on the row fit (the
    bare gauge's root adjoint in the row tiles) are the whole fit's to
    ROWS_VS_WHOLE_RTOL of their scale, at one and three simulated ranks,
    every rank rank 0's bits (refused on the rows before)."""
    whole = _scheme_b_singlet('replicated')
    rows = _scheme_b_singlet('rows')
    three = run_simulated(lambda comm: _scheme_b_singlet('rows'), 3)
    for rank in three[1:]:
        for a, b in zip(rank, three[0]):
            assert np.array_equal(a, b)
    for got in (rows, three[0]):
        assert abs(got[0] - whole[0]) < 1e-10 * abs(whole[0])
        for a, b in zip(got[1:], whole[1:]):
            assert np.abs(a - b).max() < ROWS_VS_WHOLE_RTOL * np.abs(b).max(), \
                np.abs(a - b).max() / np.abs(b).max()


#: Formaldehyde (Angstrom, planar): its T1 is bound, water's S1 and T1 are
#: dissociative (a walk ends in a collapsed radius at the BSE instability).
H2CO = [('C', (0.0, 0.0, -0.528)), ('O', (0.0, 0.0, 0.677)),
        ('H', (0.0, 0.937, -1.116)), ('H', (0.0, -0.937, -1.116))]


def test_a_solvated_walk_refreezes():
    start = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    num = walk_numerics(start)
    ref = []

    def make_state(m, field):
        if not ref:
            ref.append(GWBSEState(m, walk_scf, 'triplet', 0, bse_tda=True, numerics=num))
            return ref[0]
        return ref[0].at(m, walk_scf, None)

    def vertical_g(m):
        field = ContinuumField.from_environment(screening(m))
        return loop(GWBSEState(m, walk_scf, 'triplet', 0, bse_tda=True,
                               numerics=num), field).energy
    surf = SolvatedSurface(
        start, lambda m: ContinuumField.from_environment(screening(m)), make_state,
        label='T1 toluene', numerics=num,
        make_ground=lambda m, f: GWBSEState(m, walk_scf, numerics=num))
    states = []
    m, info = relax(surf, start, engine='cartesian', refreeze=1, verbose=False,
                    checkpoint=states.append)
    passes = info['refreeze_passes']
    print('refreeze passes', passes, 'cycles', info['first_walk_cycles'],
          info['cycles'])
    assert info['refreeze'] == 'measured' and info['converged']
    assert info['refreeze_converged'] and passes[-1]['converged']
    assert abs(passes[-1]['drift_meV']) <= REFREEZE_TOL_MEV
    assert abs(passes[0]['drift_meV']) > 1e-8 * HARTREE_TO_MEV
    final = info['final_surface']
    assert final is not surf and final.last['state'].chain is not ref[0].chain
    # pass 1 starts at the first walk's minimum R*: its first state is the
    # rebuilt surface there, which is the vertical's, and the drift is that
    # minus the old surface at R*
    first = next(st for st in states if st['pass'] == 1)
    r_star = molecule_at(start, first['minimum'])
    assert same_solve(first['energy'], vertical_g(r_star))
    assert passes[0]['drift_meV'] == (
        first['energy'] - first['first_walk']['energy']) * HARTREE_TO_MEV
    gap = final.total_energy(m) - vertical_g(m)
    print('walk vs vertical at the final minimum (Ha)', gap)
    assert abs(gap) * HARTREE_TO_MEV <= REFREEZE_TOL_MEV

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
