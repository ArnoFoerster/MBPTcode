"""Scheme B: a solvated BSE whose state-specific part is the classical
reaction-field energy of the excitation's own density difference.

`environment.solvent_scheme(env, 'classical-ss')` keeps the ground state at
eps_static and the transition-density exchange at eps_inf, as the split
equilibrium does, puts Eq. (18) and the direct W on the bare interaction, and
adds E_SS = 1/2 dV^T Q dV of the orbital-relaxed (or unrelaxed) density
difference after the excitation is solved (`split_equilibrium`). The gates:

  * E_SS is pyscf's PCM energy functional: half its second difference along
    the density, 1e-10 Ha.
  * The relaxed density is d Omega / d h: Tr[drho z] against a five-point
    finite field of the whole calculation (SCF and excitation re-solved), in
    the gas phase (singlet and triplet, full and Tamm-Dancoff) and in
    toluene under scheme B, with the continuum responding and with the
    ground state's surface charges frozen (pyscf's `frozen` PCM), 1e-8 a.u.;
    on a hybrid and a range-separated Kohn-Sham reference (the xc correction's
    orbital terms ride the same capture), 2e-8 a.u. at a 1e-3 step, where the
    SCF's round-off over the step (1e-9 to 1e-8) dominates.
  * Scheme B is another surface than Duchemin's: no quasiparticle level moves
    (Delta W = 0), the excitation differs, and its force is the derivative of
    its energy (five-point, 1e-6 Ha/Bohr).
  * With the unrelaxed density, the equilibrium-minus-vertical E_SS is the
    classical state-specific estimate of `test_split_equilibrium.grid_terms`.
  * In a continuum of eps = 1 scheme B is the gas phase.
  * A chain in any other environment refuses the classical term.
  * The surface entry declares the scheme: its environment label names the
    scheme, the density and vertical/equilibrium, and a vertical record
    carries E_SS and E_B beside omega_B (None under Duchemin's scheme).

Water/cc-pVDZ, Hartree-Fock, the G1 grid, the dense Casida solve.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
from src.Base.declaration import Excitation, GroundState
from src.environments.environment import solvent_scheme
from src.Base.numerics.numerics import numerics_of
from src.environments.solvent_screening import SolventScreening
from src.environments.split_equilibrium import ClassicalStateSpecific
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.characters import roots
from src.properties.excitations import SurfaceSpec, calc_vertical_excitation
from tests.adc_common import five_point
from tests.isdf_test_grid import g1
from tests.test_split_equilibrium import grid_terms

WATER = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Field step (a.u.) of the five-point finite field.
FIELD_STEP = 2e-4
#: Tr[drho z] against the finite field, a.u.: the five-point truncation at
#: FIELD_STEP is 7e-10.
DENSITY_TOL = 1e-8


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def field_factory(mol, field):
    """An SCF factory whose core Hamiltonian carries field['lam'] * z."""
    rz = mol.intor('int1e_r')[2]

    def factory(m):
        mf = scf.RHF(m).density_fit('cc-pvdz-ri')
        h = mf.get_hcore()
        lam = field['lam']
        mf.get_hcore = lambda *a: h + lam * rz
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
        mf.kernel()
        return mf
    return factory


def chain_on(mol, mf, factory, spin='singlet', tda=False, environment=None):
    return ExcitedStateChain(
        mol, factory, mf=mf, spin=spin, bse_tda=tda, environment=environment,
        numerics=numerics_of({'solver': 'dense', 'counts': g1(mol)}))


def test_the_ss_energy_is_the_pcm_functional(mol):
    env = SolventScreening(mol, solvent='toluene')
    d0 = scf.RHF(mol).run().make_rdm1()
    a = np.random.default_rng(1).standard_normal(d0.shape) * 1e-2
    delta = a + a.T
    energy = lambda dm: env._pcm._get_vind(dm)[0]
    second = (energy(d0 + delta) + energy(d0 - delta) - 2 * energy(d0)) / 2
    e_ss, dv = env.state_specific_energy(mol, delta)
    assert e_ss < 0 and dv.shape == (env.ngrids,)
    assert abs(e_ss - second) < 1e-10, (e_ss, second)


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
@pytest.mark.parametrize('tda', [False, True])
def test_the_relaxed_density_is_the_field_derivative_in_the_gas(mol, spin,
                                                                tda):
    field = {'lam': 0.0}
    factory = field_factory(mol, field)
    mf0 = factory(mol)
    chain = chain_on(mol, mf0, factory, spin, tda)
    rz = mol.intor('int1e_r')[2]
    analytic = np.einsum('mn,mn', chain.excitation_density(mol, mf0), rz)

    def omega(lam):
        field['lam'] = lam
        return chain.excitation(mol, factory(mol))
    fd = five_point(omega, FIELD_STEP)
    assert abs(analytic - fd) < DENSITY_TOL, (analytic, fd)
    unrelaxed = chain.excitation_density(mol, mf0, relaxed=False)
    assert abs(np.trace(unrelaxed @ mol.intor('int1e_ovlp'))) < 1e-12


#: Field step and bar on a Kohn-Sham reference: the finite field's round-off
#: goes as the SCF's over the step (3.6e-8 at 2e-4, 2e-9 at 1e-3, PBE0).
KS_FIELD_STEP = 1e-3
KS_DENSITY_TOL = 2e-8


@pytest.mark.parametrize('xc,spin', [('pbe0', 'singlet'),
                                     ('lrc-wpbeh', 'singlet'),
                                     ('lrc-wpbeh', 'triplet')])
def test_the_relaxed_density_on_a_kohn_sham_reference(mol, xc, spin):
    rz = mol.intor('int1e_r')[2]
    field = {'lam': 0.0}

    def factory(m):
        mf = dft.RKS(m, xc=xc).density_fit('cc-pvdz-jkfit')
        h = mf.get_hcore()
        lam = field['lam']
        mf.get_hcore = lambda *a: h + lam * rz
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        return mf.run()
    mf0 = factory(mol)
    chain = chain_on(mol, mf0, factory, spin)
    analytic = np.einsum('mn,mn', chain.excitation_density(mol, mf0), rz)

    def omega(lam):
        field['lam'] = lam
        return chain.excitation(mol, factory(mol))
    fd = five_point(omega, KS_FIELD_STEP)
    assert abs(analytic - fd) < KS_DENSITY_TOL, (analytic, fd)


@pytest.mark.parametrize('frozen', [True, False])
def test_the_relaxed_density_in_scheme_b(mol, frozen):
    """frozen: the ground state's surface charges held (pyscf's frozen PCM)
    against `frozen_field=True`; else the continuum responds on both sides."""
    field = {'lam': 0.0}
    factory = field_factory(mol, field)
    env = solvent_scheme(SolventScreening(mol, solvent='toluene'),
                         'classical-ss')
    assert isinstance(env, ClassicalStateSpecific)
    mf0 = env.mean_field(mol, factory)
    chain = chain_on(mol, mf0, factory, environment=env)
    rz = mol.intor('int1e_r')[2]
    analytic = np.einsum('mn,mn', chain.excitation_density(
        mol, mf0, frozen_field=frozen), rz)

    def omega(lam):
        field['lam'] = lam
        if not frozen:
            return chain.excitation(mol, env.mean_field(mol, factory))
        mf = mf0.copy()
        mf.with_solvent = mf0.with_solvent.copy()
        mf.with_solvent.frozen = True
        mf.with_solvent._dm = mf0.make_rdm1()
        h = mf0.get_hcore()
        mf.get_hcore = lambda *a: h + lam * rz
        mf.kernel(dm0=mf0.make_rdm1())
        return chain.excitation(mol, mf)
    fd = five_point(omega, FIELD_STEP)
    assert abs(analytic - fd) < DENSITY_TOL, (analytic, fd)


@pytest.fixture(scope='module')
def schemes(mol):
    """The Duchemin (split) and the scheme B chain on one ground state."""
    warnings.simplefilter('ignore')
    factory = field_factory(mol, {'lam': 0.0})
    optical = SolventScreening(mol, solvent='toluene')
    a = solvent_scheme(optical, 'duchemin')
    b = solvent_scheme(optical, 'classical-ss')
    mf = b.mean_field(mol, factory)
    return (chain_on(mol, mf, factory, environment=a),
            chain_on(mol, mf, factory, environment=b), mf, factory, optical)


def test_scheme_b_moves_no_quasiparticle_level(mol, schemes):
    chain_a, chain_b, mf, _, _ = schemes
    nocc = mol.nelectron // 2
    shift_a = np.asarray(chain_a.shared_forward(mol, mf).shift)
    shift_b = np.asarray(chain_b.shared_forward(mol, mf).shift)
    assert np.abs(shift_a).max() > 1e-3           # Eq. (18) moves the levels
    assert np.abs(shift_b).max() < 1e-12, np.abs(shift_b).max()
    assert chain_b.excitation(mol, mf) != chain_a.excitation(mol, mf)
    rec = chain_b.state_specific_correction(mol, mf)
    assert rec['E_SS'] < 0
    assert rec['E_B'] == rec['omega_B'] + rec['E_SS']
    assert np.linalg.norm(rec['delta_dipole']) > 0.1
    print(f"\nwater S1 in toluene: Duchemin {chain_a.excitation(mol, mf):.6f}, "
          f"scheme B omega {rec['omega_B']:.6f} + E_SS {rec['E_SS']:.6f} Ha "
          f"({nocc} occupied)")


def test_the_scheme_b_force_is_the_derivative_of_its_energy(mol, schemes):
    _, chain_b, mf, _, _ = schemes
    g, _ = chain_b.excitation_gradient(mol, mf)
    coords = mol.atom_coords()

    def omega(h):
        c = coords.copy()
        c[0, 2] += h
        return chain_b.excitation(mol.set_geom_(c, unit='Bohr', inplace=False))
    h = 1e-3
    fd = (omega(-2 * h) - 8 * omega(-h) + 8 * omega(h) - omega(2 * h)) / (12 * h)
    assert abs(g[0, 2] - fd) < 1e-6, (g[0, 2], fd)


def test_the_unrelaxed_term_is_the_classical_estimate(mol, schemes):
    _, chain_b, mf, factory, optical = schemes
    om, x, y, nocc = roots(chain_b, mol, mf)
    root = int(np.argmin(np.abs(om - chain_b.excitation(mol, mf))))
    eq = ClassicalStateSpecific(optical, 'unrelaxed', ss_equilibrium=True)
    vert = ClassicalStateSpecific(optical, 'unrelaxed', ss_equilibrium=False)
    drho = chain_b.excitation_density(mol, mf, relaxed=False)
    e_eq = eq.ss_continuum().state_specific_energy(mol, drho)[0]
    e_vert = vert.ss_continuum().state_specific_energy(mol, drho)[0]
    _, ss = grid_terms(optical, mol, mf, nocc, x[:, root], y[:, root])
    assert abs((e_eq - e_vert) - ss) < 1e-12 * max(1.0, abs(ss)), (e_eq - e_vert, ss)


def test_scheme_b_in_a_vacuum_continuum_is_the_gas_phase(mol):
    """eps = 1 at both frequencies: no reaction field anywhere, so omega_B is
    the gas-phase excitation (two SCF solves, so to 1e-11 Ha and not in the
    bits) and E_SS is exactly zero."""
    factory = field_factory(mol, {'lam': 0.0})
    env = solvent_scheme(SolventScreening(mol, eps=1.0, eps_static=1.0,
                                          allow_static_eps=True),
                         'classical-ss')
    mf_b, mf_g = env.mean_field(mol, factory), factory(mol)
    rec = chain_on(mol, mf_b, factory,
                   environment=env).state_specific_correction(mol, mf_b)
    gas = chain_on(mol, mf_g, factory).excitation(mol, mf_g)
    assert rec['E_SS'] == 0.0
    assert abs(rec['omega_B'] - gas) < 1e-11, rec['omega_B'] - gas


def test_another_environment_refuses_the_classical_term(mol, schemes):
    chain_a, _, mf, _, _ = schemes
    with pytest.raises(ValueError, match='scheme B'):
        chain_a.state_specific_correction(mol, mf)
    with pytest.raises(ValueError, match='scheme'):
        solvent_scheme(SolventScreening(mol, solvent='toluene'), 'pte')


@pytest.mark.parametrize('equilibrium', [False, True])
def test_the_surface_entry_records_the_scheme(mol, equilibrium):
    factory = field_factory(mol, {'lam': 0.0})
    optical = SolventScreening(mol, solvent='toluene')
    state = Excitation('singlet', root=1, kernel='bse')
    recs = {}
    for scheme in ('classical-ss', 'duchemin'):
        env = solvent_scheme(optical, scheme, ss_equilibrium=equilibrium)
        spec = SurfaceSpec(GroundState('dft', 'hf'), environment=env,
                           solver='dense', numerics=numerics_of({'grid_accuracy': 'G1'}))
        recs[scheme] = calc_vertical_excitation(spec, state, mol, factory,
                                                gradient=False)
    b = recs['classical-ss']
    label = b['physics']['record'].environment
    assert 'ClassicalStateSpecific' in label and 'relaxed density' in label
    assert ('equilibrium' if equilibrium else 'vertical') in label
    ss = b['state_specific']
    assert ss['ss_density'] == 'relaxed'
    assert ss['ss_equilibrium'] is equilibrium
    assert ss['E_SS_eV'] < 0
    assert abs(ss['omega_B_eV'] - b['omega_eV']) < 1e-10
    assert abs(ss['E_B_eV'] - ss['omega_B_eV'] - ss['E_SS_eV']) < 1e-12
    assert recs['duchemin']['state_specific'] is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
