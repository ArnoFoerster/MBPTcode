"""Split equilibrium solvation: the state-specific response at eps_static, the
transition-density exchange at eps_inf (`src/Base/split_equilibrium.py`).

WHAT EACH GATE IS FOR:

- eps_static = eps_inf: split, full and non-equilibrium are one object, bit
  for bit in the energy and the force;
- a triplet has no exchange term, so split and full agree (to the roundoff of
  carrying W_s through T = R_inf^-1 R_s into the optical gauge) in the energy
  AND the force -- the gate that T and both roots' adjoints are right;
- the split force is the derivative of its own energy (water in water);
- full - split is the transition-density term alone: to first order
  kappa v[rho_T] (Q_s - Q_inf) v[rho_T] with rho_T = sum (X + Y)_ia phi_i phi_a,
  the classical LR estimate in the same cavity;
- a bright local singlet (ethylene pi -> pi*, f large, no dipole change): the
  split shift is the small state-specific one and the full shift is
  dominated by the transition term it should not carry;
- a charge-transfer root shifts like the full equilibrium: its transition
  density is small.
"""
import numpy as np
import pytest
from pyscf import gto

from tests.pins import assert_same
from src.Base.numerics import Numerics
from src.Base.constants import HARTREE_TO_MEV, KAPPA
from src.Base.environment import equilibrium_environment
from src.Base.solvent_screening import SolventScreening
from src.gradients.excited_state import ExcitedStateChain
from src.properties.characters import ct_character, roots
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_ri_1e12, rhf_tight

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
ETHYLENE = ('C 0.0000 0.0000 0.6695; C 0.0000 0.0000 -0.6695; '
            'H 0.0000 0.9289 1.2321; H 0.0000 -0.9289 1.2321; '
            'H 0.0000 0.9289 -1.2321; H 0.0000 -0.9289 -1.2321')
C2H4_F2 = ('C 0.0000 0.6695 0.0000; C 0.0000 -0.6695 0.0000; '
           'H 0.9289 1.2321 0.0000; H -0.9289 1.2321 0.0000; '
           'H 0.9289 -1.2321 0.0000; H -0.9289 -1.2321 0.0000; '
           'F 0.0000 0.0000 3.5000; F 0.0000 0.0000 4.9120')
F2_ATOMS = (6, 7)


def environments(env):
    return {'neq': env, 'full': equilibrium_environment(env, 'full'),
            'split': equilibrium_environment(env)}


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def test_equal_constants_collapse_every_mode_bitwise(water):
    env = SolventScreening(water, eps=1.7764, eps_static=1.7764)
    out = {}
    for name, e in environments(env).items():
        chain = ExcitedStateChain(
            water, rhf_tight, environment=e,
            numerics=g1_numerics(water, Numerics(bse={'solver': 'dense'})))
        out[name] = (chain.excitation_gradient(water)[0],
                     chain.excitation(water))
    for name in ('full', 'split'):
        # a few ulp on the recording BLAS (separate SCF solves per mode); the
        # gradient and energy class bars on another
        assert_same(out[name][0], out['neq'][0], quantity='gradient',
                    label=name)
        assert_same(out[name][1], out['neq'][1], label=name)


def test_a_triplet_has_no_exchange_term(water):
    env = SolventScreening(water, solvent='dichloromethane')
    out = {}
    for name in ('full', 'split'):
        chain = ExcitedStateChain(
            water, rhf_tight, spin='triplet', environment=environments(env)[name],
            numerics=g1_numerics(water, Numerics(bse={'solver': 'dense'})))
        out[name] = (chain.excitation_gradient(water)[0],
                     chain.excitation(water))
    assert abs(out['split'][1] - out['full'][1]) < 1e-12
    assert np.abs(out['split'][0] - out['full'][0]).max() < 1e-10


def test_split_gradient_is_the_derivative_of_its_energy(water):
    env = equilibrium_environment(SolventScreening(water, solvent='water'))
    chain = ExcitedStateChain(
        water, rhf_tight, environment=env,
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


def spectra(mol, env, factory, nroots=None):
    """{mode: (Omega, X, Y, mf, nocc)} of every root, in each mode."""
    out = {}
    for name, e in environments(env).items():
        numerics = g1_numerics(mol, Numerics() if nroots is None
                               else Numerics(bse={'nroots': nroots}))
        chain = ExcitedStateChain(mol, factory, environment=e,
                                  numerics=numerics)
        mol_, mf = chain.mean_field()
        om, x, y, nocc = roots(chain, mol_, mf)
        out[name] = (om, x, y, mf, nocc)
    return out


def followed(spec, i):
    """Root i of the neq spectrum in every mode, by amplitude overlap."""
    _, x0, _, _, _ = spec['neq']
    idx = {}
    for name, (_, x, _, _, _) in spec.items():
        ovl = np.abs(x0[:, i] @ x)
        idx[name] = int(np.argmax(ovl))
        assert ovl[idx[name]] > 0.95, (name, ovl[idx[name]])
    return {name: spec[name][0][idx[name]] for name in spec}


def grid_terms(env, mol, mf, nocc, x, y):
    """(transition, state-specific) classical estimates of the equilibrium
    change, Hartree: kappa v[rho_T] dQ v[rho_T] and 1/2 v[drho] dQ v[drho],
    dQ = Q(eps_s) - Q(eps_inf) on the shared cavity."""
    c = mf.mo_coeff
    co, cv = c[:, :nocc], c[:, nocc:]
    xx, yy = x.reshape(nocc, -1), y.reshape(nocc, -1)
    rho_t = co @ (xx + yy) @ cv.T
    drho = (cv @ (xx.T @ xx + yy.T @ yy) @ cv.T
            - co @ (xx @ xx.T + yy @ yy.T) @ co.T)
    v_ao = env.ao_grid_potential(mol)
    dq = env.equilibrium_partner().response_matrix() - env.response_matrix()
    vt = np.einsum('mnk,mn->k', v_ao, 0.5 * (rho_t + rho_t.T))
    vd = np.einsum('mnk,mn->k', v_ao, drho)
    return KAPPA['singlet'] * vt @ dq @ vt, 0.5 * vd @ dq @ vd


def test_bright_local_singlet_keeps_its_transition_term_optical():
    mol = gto.M(atom=ETHYLENE, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='dichloromethane')
    spec = spectra(mol, env, rhf_tight, nroots=None)
    om, x, y, mf, nocc = spec['neq']
    # the bright pi -> pi*: largest transition dipole among the low roots
    dip = mol.intor('int1e_r')
    c = mf.mo_coeff
    mu = [np.linalg.norm(np.einsum(
        'xmn,mn->x', dip,
        c[:, :nocc] @ (x[:, n] + y[:, n]).reshape(nocc, -1) @ c[:, nocc:].T))
        for n in range(min(8, x.shape[1]))]
    i = int(np.argmax(mu))
    e = followed(spec, i)
    lr, ss = grid_terms(env, mol, mf, nocc, x[:, i], y[:, i])
    full, split = e['full'] - e['neq'], e['split'] - e['neq']
    print(f'bright root {i}: full {full * HARTREE_TO_MEV:+.1f} split '
          f'{split * HARTREE_TO_MEV:+.1f} meV; LR estimate '
          f'{lr * HARTREE_TO_MEV:+.1f}, SS {ss * HARTREE_TO_MEV:+.1f}')
    # full - split IS the transition term, to first order
    assert np.isclose(full - split, lr, rtol=0.1)
    # and it is most of the full shift of a bright local state
    assert abs(full - split) > 2.0 * abs(split)


def test_charge_transfer_root_shifts_like_the_full_equilibrium():
    mol = gto.M(atom=C2H4_F2, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='dichloromethane')
    spec = spectra(mol, env, rhf_df_ri_1e12, nroots=8)
    om, x, y, mf, nocc = spec['neq']
    chain = ExcitedStateChain(
        mol, rhf_df_ri_1e12, environment=env, numerics=g1_numerics(mol, Numerics(bse={'nroots': 8})))
    ct = ct_character(chain, mf, x, y, F2_ATOMS)
    i = int(np.argmax(ct[:8]))
    assert ct[i] > 0.95
    e = followed(spec, i)
    lr, _ = grid_terms(env, mol, mf, nocc, x[:, i], y[:, i])
    full, split = e['full'] - e['neq'], e['split'] - e['neq']
    print(f'CT root {i}: full {full * HARTREE_TO_MEV:+.1f} split '
          f'{split * HARTREE_TO_MEV:+.1f} meV; LR estimate '
          f'{lr * HARTREE_TO_MEV:+.1f}')
    assert abs(full - split) < 0.05 * abs(full)
    assert split < 0.0
