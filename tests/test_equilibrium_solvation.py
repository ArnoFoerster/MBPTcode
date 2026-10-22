"""Equilibrium solvation of a charged state, and the solvent's reorganization.

A quasiparticle level in a continuum is VERTICAL: the new charge polarizes
only the fast (electronic, eps_inf) response, the slow one staying as the
neutral left it. Once the solvent has reoriented, the level moves further by

    Delta eps_p^eq = Eq18_p(eps_s) - Eq18_p(eps_inf),

Duchemin et al.'s Eq. (18) in the static response less the one in the optical
response on the same cavity -- the screened form -- and the ion's energy
E_0 -/+ eps_p drops by Delta E_ss = -/+ Delta eps^eq <= 0; lambda_s = -Delta
E_ss is the outer-sphere reorganization energy.

  1. The screened form sits on the Born-sphere ratio (1/eps_inf - 1/eps_s) /
     (1 - 1/eps_inf) against the optical shift of a compact level, vanishes
     when eps_s = eps_inf, and lowers the ion for removal and attachment
     alike.
  2. `calc_qp_energy(equilibrium=True)` adds exactly that shift on every
     route; it is refused without a static response; an unrestricted
     reference gets one per spin.
  3. The ISDF chain's own shift agrees with the density-fitted one, and its
     analytic gradient matches a finite difference of it.
  4. The charged surfaces carry it (`RPAQPSurface`, `MeanFieldQPSurface`),
     the four-point driver turns the vertical and equilibrium surfaces into
     lambda_s, takes energies from outside, and a mean-field ground state
     relaxes in its continuum.
  5. A charge well inside a sphere is Born's: the optical Eq. (18) of neon's
     2p hole is (1 - 1/eps_inf)/2a and the equilibrium extra
     (1/eps_inf - 1/eps_s)/2a, a the cavity radius.
  6. The assembled forces -- ground state, quasiparticle and the solvent's
     relaxation -- of both charged surfaces match a finite difference of their
     own energies.

Run as a script (`python tests/test_equilibrium_solvation.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.solvent.pcm import modified_Bondi

from src.Base.numerics import Numerics
from src.Base.constants import HARTREE_TO_EV, HARTREE_TO_MEV
from src.Base.declaration import ChargedExcitation, Excitation, GroundState
from src.Base.environment import (NoEnvironment, PointCharges, attach_environment,
                                  equilibrium_environment)
from src.Base.solvent_screening import SolventScreening, attach_solvent_screening
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import (
    environment_quasiparticle_shift, equilibrium_level_shift,
    equilibrium_solvation_energy)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import MeanFieldQPSurface, RPAQPSurface
from src.properties.characters import ct_character, roots
from src.properties.excitations import SurfaceSpec, surface_of
from src.properties.optimize import MeanFieldSurface
from src.properties.surfaces import potential_energy_surface
from src.properties.vibronic import reorganization_four_point
from tests.blas_bar import assert_same
from tests.isdf_test_grid import g1, g1_numerics
from tests.reference_scf import rhf_df_ri_1e12, rhf_tight

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
AUX = 'cc-pvdz-ri'
EPS_INF, EPS_S = 1.776, 78.355
#: the Born ratio of the equilibrium extra to the optical shift
BORN_RATIO = (1 / EPS_INF - 1 / EPS_S) / (1 - 1 / EPS_INF)


def _water_df():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    return scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)


def _factory(mol):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    return mf


def test_screened_form_is_born_like_and_lowers_the_ion():
    """Measured on water/cc-pVDZ in water: HOMO 1.2607 against Born's 1.2594
    (the solute's own screening moves it by 0.1 %)."""
    mf = _water_df()
    nocc = mf.mol.nelectron // 2
    attach_solvent_screening(mf, solvent='water')
    level = equilibrium_level_shift(mf)
    optical = environment_quasiparticle_shift(mf)
    ratio = level[nocc - 1] / optical[nocc - 1]
    assert abs(ratio / BORN_RATIO - 1.0) < 1e-2, ratio
    for p in (nocc - 1, nocc):
        assert equilibrium_solvation_energy(level, p, nocc) < 0.0, p
    attach_environment(mf, SolventScreening(mf.mol, eps=2.0, eps_static=2.0))
    assert np.abs(equilibrium_level_shift(mf)).max() < 1e-14


def test_every_route_adds_it_and_refusals():
    mf = _water_df()
    nocc = mf.mol.nelectron // 2
    states = [nocc - 1, nocc]
    attach_solvent_screening(mf, solvent='water')
    level = equilibrium_level_shift(mf)[states] * HARTREE_TO_EV
    for kw in (dict(mode='casida'), dict(mode='imagfrequency'),
               dict(mode='space-time', counts=g1(mf)),
               dict(mode='space-time', continuation='cd', counts=g1(mf))):
        vertical = calc_qp_energy(mf, state=states, **kw)
        relaxed = calc_qp_energy(mf, state=states, equilibrium=True, **kw)
        if isinstance(vertical, dict):
            vertical = [vertical[p]['GW'] for p in states]
            relaxed = [relaxed[p]['GW'] for p in states]
        assert np.abs(np.subtract(relaxed, vertical) - level).max() < 1e-10, kw
    attach_environment(mf, SolventScreening(mf.mol, eps=EPS_INF))
    with pytest.raises(ValueError, match='eps_static'):
        calc_qp_energy(mf, state='homo', equilibrium=True)
    attach_environment(mf, None)
    with pytest.raises(ValueError, match='static response'):
        calc_qp_energy(mf, state='homo', equilibrium=True)


def test_unrestricted_reference_per_spin():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    attach_solvent_screening(uhf, solvent='water')
    level = equilibrium_level_shift(uhf)
    assert level.shape == (2, mol.nao)
    for spin, n in enumerate(uhf.nelec):
        assert level[spin, n - 1] > 0 and level[spin, n] < 0


@pytest.fixture(scope='module')
def solvated_chain():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='water')
    return mol, ExcitedStateChain(
        mol, _factory, environment=env,
        numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))


def test_chain_shift_and_its_gradient(solvated_chain):
    """The chain's shift is formed on the ISDF factors, the density-fitted one
    on the RI of the mean field: they agree to the fit (measured below 5 meV).
    The gradient is checked like the solvated excitation's -- one chain at
    displaced molecules, a 4-point central difference -- against the ISDF
    gradient reproducibility floor, 1e-8 Ha/Bohr absolute: measured 2.2e-9 on
    components up to 1.8e-3."""
    mol, chain = solvated_chain
    _, mf = chain.mean_field(mol)
    nocc = mol.nelectron // 2
    df = scf.RHF(mol).density_fit(auxbasis=AUX)
    df.mo_coeff, df.mo_energy, df.mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    attach_environment(df, chain.environment_at(mol))
    reference = equilibrium_level_shift(df)[nocc - 1]
    shift = chain.equilibrium_shift(0, mol)
    assert abs(shift - reference) * HARTREE_TO_EV < 5e-3, (shift, reference)

    analytic, _ = chain.equilibrium_shift_gradient(0, mol)
    h = 1e-4
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                m = mol.copy()
                d = np.zeros((mol.natm, 3))
                d[ia, x] = k * h
                m.set_geom_(mol.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(chain.equilibrium_shift(0, m))
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    assert np.abs(analytic - fd).max() < 1e-8, (analytic, fd)


def test_surfaces_and_the_four_point_driver(solvated_chain):
    mol, chain = solvated_chain
    env = chain.environment
    vertical = MeanFieldQPSurface(mol, _factory, excited=chain, state=0)
    relaxed = MeanFieldQPSurface(mol, _factory, excited=chain, state=0,
                                 equilibrium=True)
    e_v, e_ks, qp = vertical.energy(mol)
    e_r, _, qp_r = relaxed.energy(mol)
    assert abs(e_v - (e_ks - qp)) < 1e-12
    shift = chain.equilibrium_shift(0, mol)
    assert abs((qp_r - qp) - shift) < 1e-12
    assert e_r < e_v, 'the relaxed cation sits below the vertical one'
    assert relaxed.physics.excitation == ChargedExcitation(
        mol.nelectron // 2 - 1, -1, equilibrium=True)

    out = reorganization_four_point(
        energies={'a_at_a': -1.0, 'a_at_b': -0.99, 'b_at_a': -0.6,
                  'b_at_b': -0.62},
        mol_b=mol, outer_surfaces=(vertical, relaxed))
    assert abs(out['lambda_inner'] - 0.03) < 1e-12
    assert abs(out['lambda_outer'] - (e_v - e_r)) < 1e-12
    assert abs(out['lambda_outer'] - shift) < 1e-12, 'lambda_s = -Delta E_ss'
    assert out['sources']['a_at_b'] == 'given'
    with pytest.raises(ValueError, match='neither given nor computable'):
        reorganization_four_point(energies={'a_at_a': 0.0})
    with pytest.raises(ValueError, match='static response'):
        MeanFieldQPSurface(mol, _factory, state=0, equilibrium=True,
                           numerics=g1_numerics(mol))
    with pytest.raises(NotImplementedError, match='dense route'):
        potential_energy_surface(mol, _factory, ground_state=GroundState('rpa', 'hf'),
                                 excitation=ChargedExcitation(4, -1,
                                                              equilibrium=True),
                                 chi0='dense-qb', factorization='four-index')
    assert env is chain.environment


def test_mean_field_surface_in_its_continuum():
    """The ground state a UKS geometry of an ion relaxes on: its SCF is the
    continuum's own (PCM at eps_s), declared, and its force is the energy's."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    env = SolventScreening(mol, solvent='water')
    surface = potential_energy_surface(mol, _factory,
                                       ground_state=GroundState('dft', 'hf'),
                                       environment=env)
    assert isinstance(surface, MeanFieldSurface)
    assert surface.physics.environment == repr(env)
    e0 = surface.total_energy(mol)
    assert abs(e0 - env.mean_field(mol, _factory).e_tot) < 1e-9
    g, _, _ = surface.total_gradient(mol)
    h = 1e-4
    for ia, x in ((0, 2), (1, 1)):
        v = []
        for k in (-1, 1):
            m = mol.copy()
            d = np.zeros((mol.natm, 3))
            d[ia, x] = k * h
            m.set_geom_(mol.atom_coords() + d, unit='Bohr')
            m.build(False, False)
            v.append(surface.total_energy(m))
        assert abs((v[1] - v[0]) / (2 * h) - g[ia, x]) < 1e-6, (ia, x)


def test_born_sphere_limit():
    """Measured on Ne/cc-pVDZ: 1.0028 and 1.0036 of Born's (the atom's own
    screening and the discretization), Lebedev-converged; aug-cc-pVDZ 0.996."""
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-11)
    attach_environment(mf, SolventScreening(mol, eps=EPS_INF, eps_static=EPS_S))
    radius = 1.2 * modified_Bondi[10]                  # pyscf's default rule
    homo = mol.nelectron // 2 - 1
    optical = environment_quasiparticle_shift(mf)[homo]
    extra = equilibrium_level_shift(mf)[homo]
    assert abs(optical / (0.5 * (1 - 1 / EPS_INF) / radius) - 1) < 1e-2
    assert abs(extra / (0.5 * (1 / EPS_INF - 1 / EPS_S) / radius) - 1) < 1e-2


def _central_gradient(energy, mol, h=1e-4):
    """The 4-point central difference of `energy` over every coordinate."""
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                m = mol.copy()
                d = np.zeros((mol.natm, 3))
                d[ia, x] = k * h
                m.set_geom_(mol.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(energy(m))
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    return fd


@pytest.mark.parametrize('kind', ['mean-field gas', 'mean-field water',
                                  'rpa water'])
def test_whole_surface_gradients(kind):
    """Each whole force against its own energy; measured below 2e-9 Ha/Bohr on
    the O z-component with a 2-point stencil, gated at the ISDF floor."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='water')
    if kind == 'mean-field gas':
        surface = MeanFieldQPSurface(
            mol, _factory, state=0,
            numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    elif kind == 'mean-field water':
        surface = MeanFieldQPSurface(
            mol, _factory, state=0, environment=env, equilibrium=True,
            numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    else:
        surface = RPAQPSurface(
            mol, _factory, state=0, environment=env, equilibrium=True,
            numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    analytic, _, _ = surface.total_gradient(mol)
    fd = _central_gradient(surface.total_energy, mol)
    assert np.abs(np.asarray(analytic) - fd).max() < 1e-8, (analytic, fd)


# Equilibrium solvation of a relaxed neutral excited state (the response at
# eps_static, `SolventScreening(equilibrium=True)`, `equilibrium_partner`).
WATER_EQ = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: ethylene stacked over F2 at 3.5 A; ethylene pi -> F2 sigma* is the CT root
C2H4_F2 = ('C 0.0000 0.6695 0.0000; C 0.0000 -0.6695 0.0000; '
           'H 0.9289 1.2321 0.0000; H -0.9289 1.2321 0.0000; '
           'H 0.9289 -1.2321 0.0000; H -0.9289 -1.2321 0.0000; '
           'F 0.0000 0.0000 3.5000; F 0.0000 0.0000 4.9120')
F2_ATOMS = (6, 7)
NROOTS = 8


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER_EQ, basis='cc-pvdz', verbose=0)


@pytest.mark.parametrize('spin', ('singlet', 'triplet'))
def test_equal_constants_are_the_non_equilibrium_object(water, spin):
    eps = 1.7764
    kw = dict(eps=eps, eps_static=eps)
    out = []
    for equilibrium in (False, True):
        env = SolventScreening(water, equilibrium=equilibrium, **kw)
        chain = ExcitedStateChain(
            water, rhf_tight, environment=env, spin=spin,
            numerics=g1_numerics(water, Numerics(bse={'solver': 'dense'})))
        out.append(chain.excitation_gradient(water)[0])
        out.append(np.atleast_1d(chain.excitation(water)))
    # Bitwise on the recording BLAS (two chains, two SCF solves); on another
    # BLAS the gradient and the excitation energy hold to their class bars.
    for a, b, quantity in zip(out[:2], out[2:], ('gradient', 'energy')):
        assert_same(a, b, quantity=quantity, label=quantity)


def test_the_partner_moves_one_constant(water):
    env = SolventScreening(water, solvent='dichloromethane')
    eq = env.equilibrium_partner()
    assert eq is env.equilibrium_partner() and eq.equilibrium_partner() is eq
    assert eq.equilibrium and not env.equilibrium
    assert eq.eps == env.eps_static == eq.eps_static
    assert eq.eps_optical == env.eps == env.eps_optical
    assert eq.ngrids == env.ngrids and eq.method == env.method
    assert 'equilibrium' in repr(eq) and 'equilibrium' not in repr(env)
    moved = water.copy()
    moved.set_geom_(water.atom_coords() + 0.01, unit='Bohr')
    moved.build(False, False)
    assert eq.for_geometry(moved).equilibrium
    assert not env.for_geometry(moved).equilibrium
    with pytest.raises(ValueError, match='correlation'):
        eq.dynamic_factor(np.ones(3))
    with pytest.raises(ValueError, match='eps_static'):
        SolventScreening(water, eps=2.0, equilibrium=True)
    assert equilibrium_environment(env, 'full') is eq
    split = equilibrium_environment(env)
    assert split.equilibrium == 'split' and split.static_response() is eq
    assert equilibrium_environment(split) is split
    assert equilibrium_environment(split, 'full') is eq
    assert equilibrium_environment(eq, 'full') is eq
    with pytest.raises(ValueError, match='NON-equilibrium'):
        equilibrium_environment(eq)
    assert equilibrium_environment(None) is None
    gas, charges = NoEnvironment(), PointCharges([[0.0, 0.0, 9.0]], [0.5])
    assert equilibrium_environment(gas) is gas
    assert equilibrium_environment(charges) is charges


@pytest.mark.parametrize('spin', ('singlet', 'triplet'))
def test_equilibrium_gradient_is_the_derivative_of_its_energy(water, spin):
    """Water in water with the response at eps_static, 5-point stencil."""
    env = SolventScreening(water, solvent='water').equilibrium_partner()
    chain = ExcitedStateChain(
        water, rhf_tight, environment=env, spin=spin,
        numerics=g1_numerics(water, Numerics(bse={'solver': 'dense'})))
    analytic, _ = chain.excitation_gradient(water)
    h = 1e-4
    fd = np.zeros((water.natm, 3))
    for ia in range(water.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                m = water.copy()
                d = np.zeros((water.natm, 3))
                d[ia, x] = k * h
                m.set_geom_(water.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(chain.excitation(m))
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-6


def ct_shift(mol, solvent):
    """(Omega_eq - Omega_neq, classical estimate) of the CT root, Hartree."""
    env = SolventScreening(mol, solvent=solvent)
    eq = env.equilibrium_partner()
    spectra = {}
    for name, e in (('neq', env), ('eq', eq)):
        chain = ExcitedStateChain(
            mol, rhf_df_ri_1e12, environment=e, numerics=g1_numerics(mol, Numerics(bse={'nroots': NROOTS})))
        mol_, mf = chain.mean_field()
        om, x, y, nocc = roots(chain, mol_, mf)
        spectra[name] = (om, x, y, mf, ct_character(chain, mf, x, y,
                                                    F2_ATOMS))
    om, x, y, mf, ct = spectra['neq']
    i = int(np.argmax(ct[:NROOTS]))
    assert ct[i] > 0.95
    om_e, x_e, _, _, _ = spectra['eq']
    ovl = np.abs(x[:, i] @ x_e)
    j = int(np.argmax(ovl))
    assert ovl[j] > 0.99
    # the unrelaxed density difference of the CT root, and its reaction field
    c = mf.mo_coeff
    xx, yy = x[:, i].reshape(nocc, -1), y[:, i].reshape(nocc, -1)
    drho = (c[:, nocc:] @ (xx.T @ xx + yy.T @ yy) @ c[:, nocc:].T
            - c[:, :nocc] @ (xx @ xx.T + yy @ yy.T) @ c[:, :nocc].T)
    v = np.einsum('mnk,mn->k', env.ao_grid_potential(mol), drho)
    classical = 0.5 * v @ (eq.response_matrix() - env.response_matrix()) @ v
    return om_e[j] - om[i], classical


def test_charge_transfer_shift_toluene_against_dichloromethane():
    mol = gto.M(atom=C2H4_F2, basis='cc-pvdz', verbose=0)
    tol, tol_cl = ct_shift(mol, 'toluene')
    dcm, dcm_cl = ct_shift(mol, 'dichloromethane')
    print(f'CT equilibrium shift: toluene {tol * HARTREE_TO_MEV:+.1f} meV '
          f'(classical {tol_cl * HARTREE_TO_MEV:+.1f}), dichloromethane '
          f'{dcm * HARTREE_TO_MEV:+.1f} meV '
          f'(classical {dcm_cl * HARTREE_TO_MEV:+.1f})')
    assert tol < 0.0 and dcm < 0.0
    assert dcm < 10.0 * tol
    for ours, ref in ((tol, tol_cl), (dcm, dcm_cl)):
        assert 0.75 < ours / ref < 1.0
    assert np.isclose(dcm / tol, dcm_cl / tol_cl, rtol=0.1)


def test_the_switch_reaches_a_declared_surface(water):
    """SurfaceSpec(environment=equilibrium_environment(env)) is the flag."""
    env = SolventScreening(water, solvent='dichloromethane')
    energies = {}
    for e in (env, equilibrium_environment(env)):
        spec = SurfaceSpec(GroundState('dft', 'hf'), environment=e,
                           numerics={'nroots': 3})
        surface = surface_of(spec, Excitation('triplet', root=1), water, rhf_tight)
        named = 'equilibrium' in surface.physics.environment
        assert named == bool(e.equilibrium)
        energies[bool(e.equilibrium)] = surface.total_gradient(water)[1]
    # the slow polarization relaxes around T1: lower, by about 0.1 eV here
    # (-114 meV, water/cc-pVDZ HF in dichloromethane). With Eq. (18) inside
    # the G0W0 equation only Z x its eps_s - eps_inf change was realized and
    # the kernel's whole: -34 meV, which this window rejects.
    assert -0.2 < (energies[True] - energies[False]) * HARTREE_TO_EV < -0.06


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
