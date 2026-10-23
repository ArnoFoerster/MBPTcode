"""Equilibrium solvation of a relaxed excited state: the excitation's response
at eps_static (`SolventScreening(equilibrium=True)`, `equilibrium_partner`).

WHAT EACH GATE IS FOR:

- eps_static = eps_inf makes the equilibrium object the non-equilibrium one,
  bit for bit, in the energy AND the force: the switch moves one constant and
  nothing else;
- the partner keeps the solvent, the cavity and the ground state, stays in
  equilibrium when the cavity follows the atoms, and refuses the one consumer
  it has no meaning for (a correlation energy's frequency integral);
- the switch is the environment a `SurfaceSpec` declares: the surface's
  physics names it, and T1 of water in dichloromethane drops by tens of meV;
- the equilibrium excitation gradient is the derivative of its own energy
  (water in water, eps_static = 78.4 in the response): the force needs no new
  code because every adjoint is the continuum's own;
- the physics, on the ethylene/F2 charge-transfer pair (CT weight 0.99): the
  equilibrium correction Omega_eq - Omega_neq of the CT root is negative,
  fifteen times larger in dichloromethane (eps 8.93 / 2.03) than in toluene
  (2.37 / 2.24), and 86-87% of the classical estimate
  1/2 v[drho] (Q(eps_static) - Q(eps_inf)) v[drho] of its own unrelaxed
  density difference in the same cavity -- the PCM expectation, the shortfall
  being the solute's screening of Eq. (18) and the root's relaxation.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import HARTREE_TO_EV, HARTREE_TO_MEV
from src.Base.declaration import Excitation, GroundState
from src.environments.environment import PointCharges, equilibrium_environment
from src.Base.environment_seam import NoEnvironment
from src.environments.solvent_screening import SolventScreening
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.characters import ct_character, roots
from src.properties.excitations import SurfaceSpec, surface_of
from src.Base.numerics.numerics import numerics_of
from tests.blas_bar import assert_same
from tests.isdf_test_grid import g1
from tests.reference_scf import rhf_df_ri_1e12, rhf_tight

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: ethylene stacked over F2 at 3.5 A; ethylene pi -> F2 sigma* is the CT root
C2H4_F2 = ('C 0.0000 0.6695 0.0000; C 0.0000 -0.6695 0.0000; '
           'H 0.9289 1.2321 0.0000; H -0.9289 1.2321 0.0000; '
           'H 0.9289 -1.2321 0.0000; H -0.9289 -1.2321 0.0000; '
           'F 0.0000 0.0000 3.5000; F 0.0000 0.0000 4.9120')
F2_ATOMS = (6, 7)
NROOTS = 8


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


@pytest.mark.parametrize('spin', ('singlet', 'triplet'))
def test_equal_constants_are_the_non_equilibrium_object(water, spin):
    eps = 1.7764
    kw = dict(eps=eps, eps_static=eps)
    out = []
    for equilibrium in (False, True):
        env = SolventScreening(water, equilibrium=equilibrium, **kw)
        chain = ExcitedStateChain(water, rhf_tight, environment=env, spin=spin, numerics=numerics_of({'solver': 'dense', 'counts': g1(water)}))
        out.append(chain.excitation_gradient(water)[0])
        out.append(np.atleast_1d(chain.excitation(water)))
    # The same object: bitwise on the recording BLAS (two chains, two SCF
    # solves); on another BLAS the gradient and the excitation energy each hold
    # to their class bar.
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
    chain = ExcitedStateChain(water, rhf_tight, environment=env, spin=spin, numerics=numerics_of({'solver': 'dense', 'counts': g1(water)}))
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
        chain = ExcitedStateChain(mol, rhf_df_ri_1e12, environment=e, numerics=numerics_of({'nroots': NROOTS, 'counts': g1(mol)}))
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
                           numerics=numerics_of({'nroots': 3}))
        surface = surface_of(spec, Excitation('triplet', root=1), water, rhf_tight)
        named = 'equilibrium' in surface.physics.environment
        assert named == bool(e.equilibrium)
        energies[bool(e.equilibrium)] = surface.total_gradient(water)[1]
    # the slow polarization relaxes around T1: lower, by about 0.1 eV here
    # (-114 meV, water/cc-pVDZ HF in dichloromethane). With Eq. (18) inside
    # the G0W0 equation only Z x its eps_s - eps_inf change was realized and
    # the kernel's whole: -34 meV, which this window rejects.
    assert -0.2 < (energies[True] - energies[False]) * HARTREE_TO_EV < -0.06
