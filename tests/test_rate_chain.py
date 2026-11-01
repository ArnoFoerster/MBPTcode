"""Gates for src/photophysics/rate_chain.py: the ISC/RISC rate model of one
method's minima, each of its ablation switches, and its limits.

1. Each switch moves the rate the way its physics says: a shared coupling
   twice the size gives four times the rate; Condon removes exactly the
   Herzberg-Teller part; a T2 within reach with a large <S1|H_SO|T2> raises
   RISC over T1 alone; the Duschinsky rotation and frequency change move it;
   switching the zero-point difference off is the same density read at the
   gap shifted by that difference; an assumed lambda equal to the computed
   one changes nothing.
2. Limits: with no derivative the spin-vibronic model is the Condon one,
   bitwise; with every other state out of thermal reach the manifold is the
   T1 pair, bitwise; the last rung of the ladder is
   `rates.marcus_levich_jortner_rate`, bitwise; one Hessian for both states
   makes the Duschinsky density the displaced oscillator's (1e-6); one
   displaced mode at 3000 cm^-1 makes the multimode density MLJ's (1e-5,
   the hot-band weight e^{-omega/kT}); the gradient route equals the
   displacement route on a harmonic surface (1e-8).
3. The protocol is method-neutral: minima built by `minimum.from_adc2` with
   a derivative made by `spin_vibronic.second_order_derivative` from
   spin-orbit vectors and derivative couplings run through the same chain.
4. Wrong inputs are refused by name.

Synthetic formaldehyde: Hessians built from chosen frequencies and modes, no
electronic structure.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto
from scipy.linalg import expm

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (AMU_TO_ME, HARTREE_TO_CM,
                                         HARTREE_TO_EV, RATE_MLJ_N_MAX,
                                         TRIPLET_SUBLEVELS)
from src.photophysics import rates
from src.photophysics.minimum import StateMinimum, from_adc2
from src.photophysics.rate_chain import (ABLATION_LADDER,
                                         FCDensityAtRoundoff, RateInputs,
                                         RateModel, ablation_ladder,
                                         rate_chain, solvent_reorganization)
from src.photophysics.rate_models import RateRefused
from src.properties.duschinsky import CartesianHessian, mode_sets
from src.properties.optimize import translation_rotation_basis
from src.properties.spin_vibronic import (mixed_weight, path_ratio,
                                          second_order_derivative)

EV = 1.0 / HARTREE_TO_EV
CM = 1.0 / HARTREE_TO_CM
T = 300.0
GEO = 'C 0 0 0; O 0 0 1.21; H 0 0.94 -0.59; H 0 -0.94 -0.59'
W_T1 = np.array([900., 1150., 1250., 1500., 2900., 3000.]) * CM
W_S1 = np.array([800., 1050., 1300., 1350., 2850., 3050.]) * CM
LAM_SOLV = 0.08 * EV
#: The S1 mode set is the T1 one turned by this angle (radians): the two
#: states of one rigid molecule, whose Duschinsky model the guard accepts.
S1_TURN = 0.3


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=GEO, basis='sto-3g', verbose=0)


def vibrations(mol):
    """(masses in m_e, orthonormal mass-weighted vibrational basis)."""
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True)) * AMU_TO_ME
    w = np.repeat(masses, 3) ** -0.5
    tr = translation_rotation_basis(mol.atom_coords(), masses) / w[None, :]
    q, _ = np.linalg.qr(tr.T)
    u, _, _ = np.linalg.svd(np.eye(3 * mol.natm) - q @ q.T)
    return masses, u[:, :3 * mol.natm - q.shape[1]]


def rotation(n, seed, near=None):
    """A random rotation of n modes (`seed`); with `near` = (seed, angle),
    the rotation of that seed turned further by `angle` radians about a
    random axis of the mode space, as two states of one molecule are."""
    rot, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    if near is None:
        return rot
    a = np.random.default_rng(near[0]).normal(size=(n, n))
    a = (a - a.T) / np.linalg.norm(a - a.T, 2)
    return rot @ expm(near[1] * a)


def hessian(mol, omega, seed, source, near=None, like=None):
    """A Cartesian Hessian at `mol` whose vibrations are `omega`, its modes
    a random rotation (`seed`, `near`: `rotation`) of the vibrational
    space; with `like` (another geometry) that space's basis is the one
    closest to `like`'s, so one seed gives nearly the same modes at both."""
    masses, v = vibrations(mol)
    if like is not None:
        u, _, wt = np.linalg.svd(v.T @ vibrations(like)[1])
        v = v @ (u @ wt)
    if seed is None:
        rot = np.eye(v.shape[1])
    else:
        rot = rotation(v.shape[1], seed, near)
    modes = v @ rot
    hm = modes @ np.diag(omega ** 2) @ modes.T
    s = np.repeat(masses, 3) ** 0.5
    return CartesianHessian(mol, hm * s[:, None] * s[None, :], source)


def displaced(mol, hess, dq):
    """`mol` moved by the mass-weighted displacement dq on `hess`'s modes."""
    sg = mode_sets(hess, hess)[0]
    dx = (sg.modes @ dq) * np.repeat(sg.masses, 3) ** -0.5
    return molecule_at(mol, mol.atom_coords() + dx.reshape(-1, 3))


def state_minimum(where, coords, e_s1, e_t1, splits, socme, dv,
                  method='synthetic', solvation='equilibrium', dipole=None):
    """A `StateMinimum` with S1/T1 at e_s1/e_t1 and the higher states
    `splits` = {label: energy above its spin's lowest} (Hartree)."""
    energy = {'S1': e_s1, 'T1': e_t1}
    for lab, d in splits.items():
        energy[lab] = (e_s1 if lab[0] == 'S' else e_t1) + d
    e0 = -114.0
    return StateMinimum(
        method=method, where=where, coords_bohr=np.asarray(coords),
        e0=e0, energy=energy,
        omega={k: v - e0 for k, v in energy.items()}, emitter='S1',
        reservoir='T1', socme=np.asarray(socme, float),
        soc_dv_cart=None if dv is None else np.asarray(dv, float),
        soc_dv_paths={'T2': np.asarray(dv)} if dv is not None else None,
        dipole=dipole, f_osc=None if dipole is None else np.ones(3),
        d_ground=None, solvation=solvation)


SOCME = np.array([[0.0, 0.0, 0.0],
                  [0.4, 6.0, 0.2],
                  [0.3, 0.1, 0.1],
                  [0.2, 0.1, 0.1]]) * CM          # rows S0..S3, cols T1..T3
SPLITS = {'S2': 0.6 * EV, 'S3': 0.9 * EV, 'T2': 0.05 * EV, 'T3': 0.7 * EV}
FAR = {'S2': 3.0 * EV, 'S3': 3.5 * EV, 'T2': 3.0 * EV, 'T3': 3.5 * EV}


def build(mol, *, splits=SPLITS, socme=SOCME, dv_scale=1.0, same=False,
          dq=None, solvent=True, gradients=True, method='synthetic',
          rotate_s1=False):
    """RateInputs of a synthetic method whose minima ARE the two Hessians'
    minima (so the harmonic gradient is exact). dq: the S1 minimum on the
    T1 modes (mass-weighted); same: S1 takes T1's Hessian."""
    h_t = hessian(mol, W_T1, 11, 'synthetic T1')
    if dq is None:
        dq = np.array([6.0, -4.0, 3.0, 2.0, -1.0, 0.5])
    mol_s = displaced(mol, h_t, dq)
    # same: S1's mode set is T1's own Hessian (the mode sets need not sit at
    # the method's minima)
    h_s = h_t if same else hessian(mol_s, W_S1, 11, 'synthetic S1',
                                   near=(12, S1_TURN), like=mol)
    s_coords = mol_s.atom_coords()
    rng = np.random.default_rng(7)
    dv = rng.normal(size=(mol.natm, 3, 3)) * 2.0 * CM * dv_scale
    dv_s = dv
    if rotate_s1:
        # the S1 minimum in another frame, its derivative turned with it
        q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        s_coords = s_coords @ q + np.array([0.2, -0.4, 1.0])
        dv_s = np.einsum('axe,xy->aye', dv, q)
    # each state's energy at the other minimum on its own harmonic surface
    dx = (mol.atom_coords() - mol_s.atom_coords()).ravel()
    lam_s = 0.5 * dx @ h_s.hessian @ dx
    lam_t = 0.5 * dx @ h_t.hessian @ dx
    e_s1, e_t1 = -113.8, -113.8 - 0.12 * EV
    s1 = state_minimum('S1', s_coords, e_s1, e_t1 + lam_t, splits,
                       socme, dv_s, method=method,
                       solvation='equilibrium' if solvent else 'gas')
    t1 = state_minimum('T1', mol.atom_coords(), e_s1 + lam_s, e_t1,
                       splits, socme, dv, method=method,
                       solvation='equilibrium' if solvent else 'gas')
    grads = {}
    if gradients:
        # S1's harmonic gradient at the T1 minimum, in the T1 frame
        hs = h_s.hessian
        dx = (mol.atom_coords() - mol_s.atom_coords()).ravel()
        grads['S1@T1'] = (hs @ dx).reshape(-1, 3)
    sol = {}
    if solvent:
        sol = {'T1@S1': {'equilibrium': e_t1, 'nonequilibrium':
                         e_t1 + LAM_SOLV},
               'S1@T1': {'equilibrium': e_s1, 'nonequilibrium':
                         e_s1 + LAM_SOLV}}
    return RateInputs(s1=s1, t1=t1, hessians={'S1': h_s, 'T1': h_t},
                      gradients=grads, solvent=sol)


FULL = RateModel.full('curvilinear', 0.0)


# ------------------------------------------------------------------- switches
def test_shared_coupling_twice_the_size_is_four_times_the_rate(mol):
    own = build(mol)
    shared = build(mol, socme=2.0 * SOCME, dv_scale=2.0, method='gw-bse')
    a = rate_chain(own, FULL, T)
    b = rate_chain(own, replace(FULL, soc='shared'), T, shared=shared)
    assert b['coupling_from'] == 'gw-bse'
    for side in ('ISC', 'RISC'):
        assert b[side]['k_pair'] / a[side]['k_pair'] == pytest.approx(
            4.0, rel=1e-12)
        assert b[side]['k_eff'] / a[side]['k_eff'] == pytest.approx(
            4.0, rel=1e-12)


def test_condon_removes_exactly_the_herzberg_teller_part(mol):
    inp = build(mol)
    pert = replace(FULL, mixing='perturbative')
    full = rate_chain(inp, pert, T)
    condon = rate_chain(inp, replace(pert, coupling='condon'), T)
    for side in ('ISC', 'RISC'):
        assert full[side]['k_ht'] > 0.0
        assert condon[side]['k_pair'] == full[side]['k_condon']
        assert condon[side]['k_ht'] == 0.0
        assert full[side]['k_pair'] > condon[side]['k_pair']


def test_a_reachable_t2_with_its_coupling_raises_risc(mol):
    inp = build(mol)
    man = rate_chain(inp, FULL, T)
    one = rate_chain(inp, replace(FULL, triplets='T1'), T)
    assert one['RISC']['k_eff'] == one['RISC']['k_pair']
    assert set(man['RISC']['manifold']['populations']) == {'T1', 'T2'}
    pairs = {p['pair'] for p in man['RISC']['manifold']['pairs']}
    assert 'T2->S1' in pairs
    # <S1|H_SO|T2> is 15x <S1|H_SO|T1>: the T2 channel dominates
    assert man['RISC']['k_eff'] > 5.0 * one['RISC']['k_eff']
    # and ISC gains S1 -> T2 below S1
    assert man['ISC']['k_eff'] > one['ISC']['k_eff']


def test_duschinsky_and_frequency_change_move_the_rate(mol):
    inp = build(mol)
    on = rate_chain(inp, FULL, T)
    off = rate_chain(inp, replace(FULL, duschinsky=False), T)
    for side in ('ISC', 'RISC'):
        ratio = on[side]['k_pair'] / off[side]['k_pair']
        assert abs(ratio - 1.0) > 1e-2


def test_zero_point_off_is_the_density_at_the_shifted_gap(mol):
    """Without Duschinsky the zero-point difference is a shift of the gap:
    the model with it equals the model without it on minima whose S1 sits
    that much higher, to rounding."""
    inp = build(mol)
    m = replace(FULL, duschinsky=False, coupling='condon', triplets='T1')
    on = rate_chain(inp, m, T)
    zpe = on['zero_point_shift_eV'] * EV            # ZPE_S1 - ZPE_T1
    assert abs(zpe) > 1e-3 * EV
    lifted = replace(inp, s1=replace(inp.s1, energy={
        k: v + (zpe if k[0] == 'S' else 0.0)
        for k, v in inp.s1.energy.items()}), t1=replace(inp.t1, energy={
            k: v + (zpe if k[0] == 'S' else 0.0)
            for k, v in inp.t1.energy.items()}))
    off = rate_chain(lifted, replace(m, zero_point=False), T)
    for side in ('ISC', 'RISC'):
        assert off[side]['k_pair'] == pytest.approx(on[side]['k_pair'],
                                                    rel=1e-9)
    # with Duschinsky the phase does it: the switch moves the rate
    d_on = rate_chain(inp, replace(m, duschinsky=True), T)
    d_off = rate_chain(inp, replace(m, duschinsky=True, zero_point=False), T)
    assert abs(d_on['RISC']['k_pair'] / d_off['RISC']['k_pair'] - 1) > 1e-2


def test_assumed_lambda_equal_to_the_computed_one_changes_nothing(mol):
    inp = build(mol)
    a = rate_chain(inp, FULL, T)
    b = rate_chain(inp, replace(FULL, classical_lambda=LAM_SOLV
                                * HARTREE_TO_EV), T)
    c = rate_chain(inp, replace(FULL, classical_lambda=0.2), T)
    for side in ('ISC', 'RISC'):
        assert b[side]['k_eff'] == pytest.approx(a[side]['k_eff'], rel=1e-9)
        assert abs(c[side]['k_eff'] / a[side]['k_eff'] - 1.0) > 1e-2
    assert a['ISC']['lambda_solvent_eV'] == pytest.approx(0.08, rel=1e-12)


def test_the_ladder_records_every_switch(mol):
    inp = build(mol)
    ladder = ablation_ladder(inp, FULL, T, shared=inp)
    assert [r['step'] for r in ladder] == [s for s, _ in ABLATION_LADDER]
    assert all('table' in r for r in ladder), [r.get('refused')
                                              for r in ladder]
    last = ladder[-1]['model']
    assert last == {'soc': 'shared', 'coupling': 'condon', 'triplets': 'T1',
                    'duschinsky': False,
                    'duschinsky_coordinates': 'curvilinear',
                    'classical_below_cm': 100.0, 'zero_point': False,
                    'density': 'mlj', 'classical_lambda': 0.2,
                    'huang_rhys_route': 'curvilinear',
                    'gaussian_sigma_ev': 0.0, 'mixing': 'perturbative'}
    for r in ladder:
        assert r['table']['model'] == r['model']


# --------------------------------------------------------------------- limits
def test_no_derivative_is_the_condon_model_bitwise(mol):
    inp = build(mol, dv_scale=0.0)
    a = rate_chain(inp, FULL, T)
    b = rate_chain(inp, replace(FULL, coupling='condon'), T)
    for side in ('ISC', 'RISC'):
        assert a[side]['k_pair'] == b[side]['k_pair']
        assert a[side]['k_eff'] == b[side]['k_eff']


def test_out_of_reach_states_leave_the_t1_pair_bitwise(mol):
    inp = build(mol, splits=FAR)
    a = rate_chain(inp, FULL, T)
    b = rate_chain(inp, replace(FULL, triplets='T1'), T)
    for side in ('ISC', 'RISC'):
        assert a[side]['k_eff'] == b[side]['k_eff'] == a[side]['k_pair']


def test_the_last_rung_is_the_marcus_levich_jortner_rate_bitwise(mol):
    inp = build(mol)
    ladder = ablation_ladder(inp, FULL, T, shared=inp)
    last = ladder[-1]['table']
    sg = mode_sets(inp.hessians['T1'], inp.hessians['S1'], 'curvilinear')
    lam = 0.2 * EV
    de = inp.s1.energy['S1'] - inp.t1.energy['T1']
    # the S1 minimum on the T1 modes, as the chain builds it
    from src.properties.duschinsky import displacement_from_minima
    k = displacement_from_minima(sg[0], sg[1],
                                 molecule_at(mol, inp.t1.coords_bohr),
                                 molecule_at(mol, inp.s1.coords_bohr),
                                 'curvilinear')
    # through the curvilinear J's inverse, as the full model maps it
    inv = np.linalg.inv(sg[2])
    s_g = sg[0].omega * (-inv @ k) ** 2 / 2.0
    s_sum = float(s_g.sum())
    om = float((s_g * sg[0].omega).sum() / s_sum)
    v = abs(SOCME[1, 0])
    risc = rates.marcus_levich_jortner_rate(
        v / np.sqrt(TRIPLET_SUBLEVELS), de, lam, s_sum, om, T,
        n_max=RATE_MLJ_N_MAX)
    isc = rates.marcus_levich_jortner_rate(v, -de, lam, s_sum, om, T,
                                           n_max=RATE_MLJ_N_MAX)
    assert last['RISC']['k_eff'] == risc
    assert last['ISC']['k_eff'] == isc


def test_one_hessian_makes_duschinsky_the_displaced_oscillator(mol):
    inp = build(mol, same=True)
    m = replace(FULL, coupling='condon', triplets='T1')
    a = rate_chain(inp, m, T)
    b = rate_chain(inp, replace(m, duschinsky=False), T)
    assert abs(a['zero_point_shift_eV']) < 1e-12
    for side in ('ISC', 'RISC'):
        assert a[side]['k_pair'] == pytest.approx(b[side]['k_pair'],
                                                  rel=1e-6)


def test_one_high_mode_makes_multimode_the_mlj_density(mol):
    """Only the 3000 cm^-1 mode displaced, one Hessian: the multimode
    density is the Poisson sum of MLJ up to the hot-band weight
    e^{-omega/kT} = 5.6e-7 at 300 K."""
    dq = np.zeros(6)
    dq[5] = 1.0
    inp = build(mol, same=True, dq=dq)
    m = replace(FULL, coupling='condon', triplets='T1', duschinsky=False,
                zero_point=False)
    a = rate_chain(inp, m, T)
    b = rate_chain(inp, replace(m, density='mlj'), T)
    for side in ('ISC', 'RISC'):
        assert a[side]['k_pair'] == pytest.approx(b[side]['k_pair'],
                                                  rel=1e-5)


def test_gradient_route_equals_the_cartesian_route_when_harmonic(mol):
    """The S1 gradient at the T1 minimum of a harmonic S1 surface gives the
    straight-line displacement; the curvilinear one is another measure."""
    inp = build(mol, rotate_s1=True)
    cart = rate_chain(inp, replace(FULL, huang_rhys_route='cartesian'), T)
    grad = rate_chain(inp, replace(FULL, huang_rhys_route='gradient'), T)
    assert grad['gradient_from'] == 'S1@T1'
    s = cart['huang_rhys_sum']
    assert s['gradient'] == pytest.approx(s['cartesian'], rel=1e-8)
    assert s['curvilinear'] != s['cartesian']
    for side in ('ISC', 'RISC'):
        assert grad[side]['k_eff'] == pytest.approx(cart[side]['k_eff'],
                                                    rel=1e-7)


def test_a_rotated_s1_minimum_gives_the_same_rates(mol):
    a = rate_chain(build(mol), FULL, T)
    b = rate_chain(build(mol, rotate_s1=True), FULL, T)
    for side in ('ISC', 'RISC'):
        assert b[side]['k_eff'] == pytest.approx(a[side]['k_eff'], rel=1e-8)


# ------------------------------------------------------------ method-neutral
def test_minima_of_another_method_run_through_the_same_chain(mol):
    """ADC(2)-shaped minima (`from_adc2`) whose derivative is built from
    spin-orbit vectors and derivative couplings by
    `second_order_derivative`: the T2 path alone, so the derivative scales
    with the coupling and the Herzberg-Teller rate with its square."""
    base = build(mol, gradients=False)
    rng = np.random.default_rng(3)
    v_s1t2 = np.array([0.0, 4.0, 1.0]) * CM          # <S1|H_SO|T2>
    d_t2t1 = rng.normal(size=(mol.natm, 3)) * 0.05     # <T2|d/dR|T1>

    def adc2(where, src, scale):
        dv = second_order_derivative({}, {'T2': v_s1t2}, {},
                                     {'T2': scale * d_t2t1})
        sing = [src.omega[f'S{i}'] for i in (1, 2, 3)]
        trip = [src.omega[f'T{i}'] for i in (1, 2, 3)]
        return from_adc2(where, src.coords_bohr, src.e0, sing, trip,
                         socme=SOCME, soc_dv_cart=dv,
                         solvation='equilibrium')

    out = {}
    for scale in (1.0, 2.0):
        inp = replace(base, s1=adc2('S1', base.s1, scale),
                      t1=adc2('T1', base.t1, scale))
        out[scale] = rate_chain(inp, replace(FULL, triplets='T1'), T)
    assert out[1.0]['method'] == 'adc2'
    for side in ('ISC', 'RISC'):
        assert out[2.0][side]['k_condon'] == out[1.0][side]['k_condon']
        assert out[2.0][side]['k_ht'] / out[1.0][side]['k_ht'] == \
            pytest.approx(4.0, rel=1e-10)


def test_an_unresolved_duschinsky_density_is_flagged(mol):
    """RISC across 1.5 eV with lambda 0.08 eV sits near e^-300 below its
    band: the real-time integral cannot resolve it, and the record says so."""
    inp = build(mol)
    # the singlets lifted at both minima, so each surface keeps its shape
    far = replace(inp, **{where: replace(m, energy={
        k: v + (1.4 * EV if k[0] == 'S' else 0.0)
        for k, v in m.energy.items()})
        for where, m in (('s1', inp.s1), ('t1', inp.t1))})
    with pytest.warns(FCDensityAtRoundoff):
        out = rate_chain(far, replace(FULL, triplets='T1'), T)
    assert out['numerical_flags'] == ['fc_density_at_roundoff']
    assert rate_chain(inp, FULL, T)['numerical_flags'] == []


def test_kinetics_with_a_dipole(mol):
    inp = build(mol)
    dip = np.array([[0.3, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    inp = replace(inp, s1=replace(inp.s1, dipole=dip))
    out = rate_chain(inp, FULL, T)
    assert out['k_r'] > 0.0
    # a yield of one to rounding: MKL lands 3e-14 above it
    assert 0.0 < out['pl']['phi_total'] <= 1.0 + 1e-12


# ------------------------------------------------- the diabatic T1/T2 mixing
def test_mixed_weight_is_the_thermal_average_of_the_two_state_mixing():
    """Against a direct quadrature over the Gaussian coupling u; second
    order (r^2) below, 1/2 (full mixing) above; the series and the closed
    form meet at r = 1e-3."""
    from scipy.integrate import quad
    for r in (0.02, 0.1, 0.5, 3.0):
        ref = quad(lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * 0.5
                   * (1 - 1 / np.sqrt(1 + 4 * r * r * z * z)),
                   -np.inf, np.inf, epsabs=1e-14, epsrel=1e-12)[0]
        assert mixed_weight(r) == pytest.approx(ref, rel=1e-9)
    assert mixed_weight(1e-5) == pytest.approx(1e-10, rel=1e-8)
    assert mixed_weight(0.1) == pytest.approx(0.009221864644, rel=1e-9)
    assert mixed_weight(1e4) == pytest.approx(0.5, abs=1e-3)
    assert mixed_weight(np.inf) == 0.5
    lo, hi = mixed_weight(1e-3 * (1 - 1e-12)), mixed_weight(1e-3)
    assert hi == pytest.approx(lo, rel=1e-8)
    with pytest.raises(ValueError):
        mixed_weight(-0.1)


def test_weak_coupling_is_the_perturbative_rate(mol):
    """The diabatic model at a path ratio r differs from the perturbative
    one by O(r^2): 1e-6 at r ~ 1e-3, and a tenth of the ratio is a
    hundredth of the difference."""
    dev = {}
    for scale in (1e-3, 1e-2):
        inp = build(mol, dv_scale=scale)
        d = rate_chain(inp, replace(FULL, mixing='diabatic'), T)
        p = rate_chain(inp, replace(FULL, mixing='perturbative'), T)
        r = d['ISC']['mixing']['paths']['T2']['ratio']
        assert p['ISC']['mixing']['model'] == 'perturbative'
        assert d['ISC']['mixing']['model'] == 'diabatic'
        dev[scale] = max(abs(d[s]['k_eff'] / p[s]['k_eff'] - 1.0)
                         for s in ('ISC', 'RISC'))
        assert dev[scale] < 10.0 * r * r
    assert dev[1e-3] < 1e-6
    assert dev[1e-2] / dev[1e-3] == pytest.approx(100.0, rel=0.05)


def test_auto_switches_at_the_stated_ratio(mol):
    from src.Base.numerics.constants import RATE_MIXING_RATIO_MAX
    weak = rate_chain(build(mol, dv_scale=1e-3), FULL, T)
    strong = rate_chain(build(mol, dv_scale=10.0), FULL, T)
    for side in ('ISC', 'RISC'):
        rec = weak[side]['mixing']
        assert rec['model'] == 'perturbative'
        assert rec['paths']['T2']['ratio'] < RATE_MIXING_RATIO_MAX
        rec = strong[side]['mixing']
        assert rec['model'] == 'diabatic'
        assert rec['paths']['T2']['ratio'] > RATE_MIXING_RATIO_MAX
        assert rec['ratio_max'] == RATE_MIXING_RATIO_MAX
        assert rec['declared'] == 'auto'


def test_a_closing_gap_saturates_the_diabatic_rate(mol):
    """A path derivative 10x larger (its gap 10x smaller): the perturbative
    ISC grows as the square, the diabatic one with the path state's exact
    weight s, which saturates at full mixing, 1/2."""
    k = {}
    for scale in (100.0, 1000.0):
        inp = build(mol, dv_scale=scale)
        k[scale] = {m: rate_chain(inp, replace(FULL, mixing=m,
                                               triplets='T1'), T)['ISC']
                    for m in ('perturbative', 'diabatic')}
    grow = k[1000.0]['perturbative']['k_pair'] / k[100.0]['perturbative'][
        'k_pair']
    assert grow == pytest.approx(100.0, rel=0.02)
    sat = k[1000.0]['diabatic']['k_pair'] / k[100.0]['diabatic']['k_pair']
    assert sat < 1.1
    # the borrowed rate is the path state's exact weight s times a fixed one
    s = {sc: k[sc]['diabatic']['mixing']['paths']['T2']['sin2']
         for sc in k}
    assert k[1000.0]['diabatic']['k_ht'] / k[100.0]['diabatic']['k_ht'] == \
        pytest.approx(s[1000.0] / s[100.0], rel=1e-9)
    assert 0.45 < s[1000.0] <= 0.5


def test_the_mixed_manifold_keeps_the_coupling(mol):
    """Diabatic, the T2 channel of ISC carries (1 - s) |V(S1,T2)|^2 +
    s |V(S1,T1)|^2: coupling moves between the pair, none is created."""
    inp = build(mol, dv_scale=10.0)
    out = rate_chain(inp, replace(FULL, mixing='diabatic'), T)['ISC']
    s = out['mixing']['paths']['T2']['sin2']
    pair = {p['pair']: p for p in out['manifold']['pairs']}
    v11, v12 = SOCME[1, 0] * HARTREE_TO_CM, SOCME[1, 1] * HARTREE_TO_CM
    assert pair['S1->T2']['v_cm'] == pytest.approx(
        np.sqrt((1 - s) * v12 ** 2 + s * v11 ** 2), rel=1e-12)
    assert out['mixing']['condon_factor'] == pytest.approx(1 - s, rel=1e-12)


def test_path_ratio_reads_the_coupling_over_the_gap():
    """dV/dq_k = kappa_k V / dE: r^2 = sum (n_k + 1/2) kappa_k^2 / dE^2."""
    w = np.array([500.0, 1500.0]) * CM
    kappa = np.array([20.0, 60.0]) * CM
    de, v = 30.0 * 1e-3 * EV, 2.0 * CM
    from src.properties.franck_condon import bose
    from src.Base.numerics.constants import BOLTZMANN_HARTREE_PER_KELVIN
    n = bose(w, BOLTZMANN_HARTREE_PER_KELVIN * T)
    ref = np.sqrt(((n + 0.5) * kappa ** 2).sum()) / de
    assert path_ratio(kappa * v / de, w, v, T) == pytest.approx(ref,
                                                                rel=1e-13)
    assert path_ratio(np.zeros(2), w, v, T) == 0.0
    assert path_ratio(kappa, w, 0.0, T) == np.inf


# ------------------------------------------------------------------ refusals
def test_refusals(mol):
    inp = build(mol)
    with pytest.raises(ValueError, match='MLJ density needs duschinsky'):
        replace(FULL, density='mlj')
    with pytest.raises(ValueError, match='no inhomogeneous width'):
        replace(FULL, density='mlj', duschinsky=False, gaussian_sigma_ev=0.01)
    with pytest.raises(ValueError, match="'computed' or a positive"):
        replace(FULL, classical_lambda=0.0)
    with pytest.raises(RateRefused, match='reference RateInputs'):
        rate_chain(inp, replace(FULL, soc='shared'), T)
    with pytest.raises(RateRefused, match='gradient route needs'):
        rate_chain(build(mol, gradients=False),
                   replace(FULL, huang_rhys_route='gradient'), T)
    with pytest.raises(RateRefused, match='no derivative'):
        rate_chain(replace(inp, s1=replace(inp.s1, soc_dv_cart=None)),
                   FULL, T)
    gas = build(mol, solvent=False)
    with pytest.raises(RateRefused, match='needs a width'):
        rate_chain(gas, FULL, T)
    assert rate_chain(gas, replace(FULL, gaussian_sigma_ev=0.02),
                      T)['ISC']['k_eff'] > 0.0
    with pytest.raises(RateRefused, match='MLJ density needs a positive'):
        rate_chain(gas, replace(FULL, duschinsky=False, density='mlj'), T)
    bare = RateInputs(s1=inp.s1, t1=inp.t1, hessians=inp.hessians,
                      gradients={}, solvent={})
    with pytest.raises(RateRefused, match='no solvent energies'):
        rate_chain(bare, FULL, T)
    assumed = rate_chain(bare, replace(FULL, classical_lambda=0.1), T)
    assert assumed['RISC']['lambda_solvent_eV'] is None
    with pytest.raises(RateRefused, match='both'):
        RateInputs(s1=inp.s1, t1=inp.t1, hessians=inp.hessians,
                   gradients={}, solvent={'T1@S1': inp.solvent['T1@S1']})
    with pytest.raises(RateRefused, match='not of one state'):
        solvent_reorganization({'equilibrium': -1.0,
                                'nonequilibrium': -1.001})
    with pytest.raises(RateRefused, match='one method'):
        replace(inp, t1=replace(inp.t1, method='other'))
