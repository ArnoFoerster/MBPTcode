"""The dRPA ground state in a continuum, in both solvation modes.

E_0 = E_HF[PCM(eps_static)] + E_c^dRPA with E_c screened by the solvent's
ELECTRONIC response only (eps_inf, and its pole g(iw) under the fold). The
equilibrium switch (`SolventScreening(equilibrium=True)`) moves an
excitation's response to eps_static; the ground state is in equilibrium in
both modes already, so `RPAGroundStateChain` reads an equilibrium continuum
through its `non_equilibrium_partner` and E_0 is one function.

WHAT EACH GATE IS FOR (water/cc-pVDZ RHF in water unless stated):
- E_c in a continuum against an independent route: dense ov-space ACFDT and
  plasmon formulas on the RI matrices (no ISDF, no metric root, no fold
  assembly), for the bare (A), dressed (B) and fold (F) functionals;
- an equilibrium continuum leaves E_0 and its force those of the
  non-equilibrium one, B and F (two chains evaluated apart: to a few ulp,
  `matches` of tests/pins.py, not bitwise), while the declared environment keeps its
  name (reading eps_static inside E_c instead would be a different
  functional, +9.0 mHa here);
- the equilibrium surface's force (F) is the derivative of its own energy;
- the composed `RPABSESurface`: equilibrium = non-equilibrium to a few ulp
  where eps_static == eps_inf (energy and force, F ground state), and at
  water's constants E_0 is unmoved while Omega moves.

Judge by OUTPUT: every check prints its measured number.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import df, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (HARTREE_TO_EV, SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.environment import equilibrium_environment
from src.Base.numerics import Numerics
from src.Base.solvent_screening import SolventScreening
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.solvated_rpa_energy import solvated_correlation_energies
from tests.isdf_test_grid import g1
from tests.pins import RECORDING_BLAS, matches

BASIS = 'cc-pvdz'
AUXBASIS = 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Ha. The chain's ISDF factorization against the same RI matrices taken
#: whole: measured 3.3e-5 to 3.5e-5 on the absolute A/B/F at the shipped
#: radii of the default counts (they are chosen on BSE roots, not on the
#: correlation energy), 1.4e-6 on their differences.
ISDF_ENERGY_TOL = 5e-5
ISDF_SHIFT_TOL = 5e-6
#: Ha. The dense frequency quadrature against the plasmon (Furche trace)
#: formula on one set of matrices: measured 1.7e-8.
QUADRATURE_TOL = 1e-7
#: Ha/Bohr. Four-point stencil at FD_STEP on a PCM-relaxed surface.
FD_STEP = 4e-3
FD_FORCE_TOL = 1e-6
#: eV. Water's S1 in water moves -102.5 meV with the slow field; anything
#: below this would mean the switch never reached the excited half.
OMEGA_SHIFT_FLOOR_EV = 1e-3


def check(ok, label, detail=''):
    """Print a named check with its measured number and return the verdict."""
    print(f'  [{"ok" if ok else "FAIL"}] {label}' + (f'   {detail}' if detail else ''))
    return bool(ok)


def rhf(mol):
    """Density-fitted RHF converged tightly enough for a frozen factorization."""
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def solvated(water):
    """The non-equilibrium chain in water, whose mean field and factorization
    every other chain below is handed, so only the environment differs."""
    env = SolventScreening(water, solvent='water')
    return RPAGroundStateChain(water, rhf, numerics=Numerics(isdf={'auxbasis': AUXBASIS, 'counts': g1(water)}), environment=env)


def twin_numerics(chain, fold=False):
    """The `Numerics` of `twin`: `chain`'s quadrature and `fold`."""
    return Numerics(quadrature={'ntau_rpa': chain.ntau, 'nfreq_rpa': len(chain.grid.omega_points)},
                    solvent={'fold': fold})


def twin(chain, environment, fold=False, numerics=None):
    """`chain`'s mean field, factorization and quadrature in `environment`;
    `numerics` replaces the twin's own when the surface it joins is built
    from the same object."""
    mol, mf = chain.mean_field()
    return RPAGroundStateChain(mol, rhf, mf=mf, environment=environment,
                               factorization=chain.factorization,
                               numerics=numerics or twin_numerics(chain, fold))


def ri_ov_matrices(chain):
    """(Delta_ia, (ia|v|jb), (ia|vtilde|jb)) on the chain's auxiliary basis,
    taken whole: c = J^-1 (P|ia), v = (ia|P) c, vtilde = c^T vtilde_PQ c."""
    mol, mf = chain.mean_field()
    nocc = chain.nocc
    auxmol = chain.auxmol(mol)
    c_occ, c_vir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1')
    p_ia = np.einsum('mnP,mi,na->iaP', int3c, c_occ, c_vir, optimize=True)
    p_ia = p_ia.reshape(-1, auxmol.nao_nr())
    fit = np.linalg.solve(auxmol.intor('int2c2e'), p_ia.T)
    kernel = chain.environment_at(mol).aux_kernel(auxmol)
    eps = mf.mo_energy
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    return delta, p_ia @ fit, fit.T @ kernel @ fit


def dense_acfdt(delta, interaction, counter_term, grid):
    """(1/2pi) sum_w W_w [ln det(1 + M U(w) M) - Tr(M C M)] in Hartree, with
    M^2 = 4 Delta/(Delta^2 + w^2) the closed-shell ov-space -chi0(iw)."""
    total = 0.0
    for w, weight in zip(grid.omega_points, grid.omega_weights):
        m = np.sqrt(4.0 * delta / (delta ** 2 + w ** 2))
        log = np.linalg.slogdet(np.eye(len(delta))
                                + m[:, None] * interaction(w) * m[None, :])[1]
        total += weight * (log - np.sum(m * m * np.diag(counter_term)))
    return total / (2.0 * np.pi)


def dense_plasmon(delta, k):
    """Furche's trace formula 1/2 (sum Omega - Tr A), A = Delta + 2K, B = 2K."""
    root = np.sqrt(delta)
    omega2 = np.linalg.eigvalsh(np.diag(delta ** 2)
                                + 4.0 * root[:, None] * k * root[None, :])
    return 0.5 * (np.sqrt(omega2).sum() - delta.sum() - 2.0 * np.trace(k))


def test_the_continuum_reaches_E_c_as_the_dense_route_has_it(solvated):
    """A, B and F on the chain against dense ACFDT on the RI matrices.

    The dense route shares only the auxiliary representation of v and vtilde
    with the chain: no ISDF, no metric root, no fold assembly. F's g(iw) is
    the solvent's own `dynamic_factor`; the dense quadrature is itself pinned
    to the plasmon formula, which has no frequency integral at all.
    """
    res = solvated_correlation_energies(solvated)
    delta, v, vt = ri_ov_matrices(solvated)
    g = solvated.environment.dynamic_factor
    grid = solvated.grid
    dense = dict(
        A=dense_acfdt(delta, lambda w: v, v, grid),
        B=dense_acfdt(delta, lambda w: v + vt, v + vt, grid),
        F=dense_acfdt(delta, lambda w: v + g(w) * vt, v, grid))
    chain = dict(A=res.bare, B=res.dressed, F=res.fold)
    ok = True
    for name, k in (('A', v), ('B', v + vt)):
        err = abs(dense[name] - dense_plasmon(delta, k))
        ok &= check(err < QUADRATURE_TOL, f'dense {name}: quadrature = plasmon',
                    f'|diff| {err:.1e} Ha')
    for name in 'ABF':
        err = abs(chain[name] - dense[name])
        ok &= check(err < ISDF_ENERGY_TOL, f'{name}: chain = dense',
                    f'{chain[name]:.9f} vs {dense[name]:.9f}, |diff| {err:.1e} Ha')
    for name in 'BF':
        shift_c, shift_d = chain[name] - chain['A'], dense[name] - dense['A']
        err = abs(shift_c - shift_d)
        ok &= check(err < ISDF_SHIFT_TOL, f'{name} - A: chain = dense',
                    f'{shift_c:+.9f} vs {shift_d:+.9f}, |diff| {err:.1e} Ha')
    assert ok


def equilibrium_objects(env):
    """The continuum's equilibrium forms: the all-static partner, and whatever
    `equilibrium_environment` declares by default."""
    return {'partner': env.equilibrium_partner(),
            'declared': equilibrium_environment(env),
            'split': equilibrium_environment(env, 'split')}


@pytest.mark.parametrize('mode', ['partner', 'declared', 'split'])
@pytest.mark.parametrize('fold', [False, True])
def test_equilibrium_leaves_E_0_and_its_force_alone(solvated, fold, mode):
    """E_0 and dE_0/dR in an equilibrium continuum are the non-equilibrium
    ones (to a few ulp: two chains evaluated apart); the chain still declares
    the environment it was given.

    Water's eps_static (78.4) is far from its eps_inf (1.78), so a chain that
    read the equilibrium object's kernel would move E_c by +9.0 mHa (B) or
    refuse (F, whose g(iw) has no meaning for the static response).
    """
    env = solvated.environment
    neq = twin(solvated, env, fold=fold)
    eq = twin(solvated, equilibrium_objects(env)[mode], fold=fold)
    ok = check('equilibrium' in eq.physics.environment,
               'the declared environment is the equilibrium one',
               eq.physics.environment)
    e_neq, e_eq = neq.energy(), eq.energy()
    d_e = np.abs(np.subtract(e_eq, e_neq)).max()
    ok &= check(matches(e_eq, e_neq, RECORDING_BLAS), 'E_0, E_HF, E_c the same',
                f'E_c {e_eq[2]:.12f} / {e_neq[2]:.12f}, max |d| {d_e:.1e} Ha')
    g_neq, _, _ = neq.total_gradient()
    g_eq, _, _ = eq.total_gradient()
    d_g = np.abs(g_eq - g_neq).max()
    ok &= check(matches(g_eq, g_neq, RECORDING_BLAS, quantity='gradient'),
                'dE_0/dR the same',
                f'max |dE_0/dR| {np.abs(g_neq).max():.3e} Ha/Bohr, '
                f'max |d| {d_g:.1e}')
    assert ok


def test_the_equilibrium_force_is_the_derivative_of_its_energy(water):
    """dE_0/dR of the folded surface in an equilibrium continuum against a
    four-point difference of E_0 itself.

    Each displaced energy rebuilds the equilibrium cavity around the moved
    atoms (`for_geometry` keeps the mode) and re-relaxes the PCM(eps_static)
    mean field, so the difference answers whether the partner is taken at the
    DISPLACED geometry -- a partner frozen at the reference cavity would give
    a force with the cavity's motion missing from E_c alone.
    """
    env = SolventScreening(water, solvent='water').equilibrium_partner()
    chain = RPAGroundStateChain(water, rhf, numerics=Numerics(isdf={'auxbasis': AUXBASIS, 'counts': g1(water)},
                                                              solvent={'fold': True}),
                                environment=env)
    grad, _, _ = chain.total_gradient()
    x0 = water.atom_coords()

    def energy_at(ia, k, dx):
        mol = water.copy()
        coords = x0.copy()
        coords[ia, k] += dx
        mol.set_geom_(coords, unit='Bohr')
        mol.build(False, False)
        return chain.total_energy(mol)

    ok, worst = True, 0.0
    for ia, k in ((0, 2), (1, 1)):
        f = [energy_at(ia, k, s * FD_STEP) for s in (1, -1, 2, -2)]
        fd = (8.0 * (f[0] - f[1]) - (f[2] - f[3])) / (12.0 * FD_STEP)
        err = abs(fd - grad[ia, k])
        worst = max(worst, err)
        ok &= check(err < FD_FORCE_TOL, f'atom {ia} component {k}',
                    f'analytic {grad[ia, k]:+.9f}  FD {fd:+.9f}  '
                    f'|diff| {err:.2e} Ha/Bohr')
    assert ok and worst < FD_FORCE_TOL


def composed(mol, environment):
    """E_0^dRPA (fold) + Omega_S1 in `environment`, built from scratch."""
    numerics = Numerics(isdf={'auxbasis': AUXBASIS, 'counts': g1(mol)},
                        solvent={'fold': True})
    ground = RPAGroundStateChain(mol, rhf, numerics=numerics, environment=environment)
    return RPABSESurface(mol, rhf, environment=environment, ground=ground, numerics=numerics)


def test_the_composed_surface_at_equal_constants_is_one_object(water):
    """eps_static == eps_inf: the equilibrium E_0 + Omega surface IS the
    non-equilibrium one, energy and force to a few ulp (two surfaces built
    from scratch), with the folded dRPA ground state read through the
    non-equilibrium partner."""
    eps = SolventScreening(water, solvent='toluene').eps
    omega_p = SolventScreening(water, solvent='toluene').omega_p
    env = SolventScreening(water, eps=eps, eps_static=eps, omega_p=omega_p)
    g_neq, e_neq, d_neq = composed(water, env).total_gradient()
    g_eq, e_eq, d_eq = composed(water, env.equilibrium_partner()).total_gradient()
    ok = check(matches(e_eq, e_neq, RECORDING_BLAS), 'E_0 + Omega the same',
               f'{e_eq:.12f} / {e_neq:.12f} Ha, Omega {d_eq["omega"]:.6f} Ha')
    ok &= check(matches(d_eq['e_c'], d_neq['e_c'], RECORDING_BLAS),
                'E_c the same', f'{d_eq["e_c"]:.12f}')
    ok &= check(matches(g_eq, g_neq, RECORDING_BLAS, quantity='gradient'),
                'dE/dR the same',
                f'max |dE/dR| {np.abs(g_neq).max():.3e} Ha/Bohr')
    assert ok


def test_equilibrium_moves_omega_and_not_E_0(solvated):
    """Water in water: the composed surface's E_0 is unmoved by the switch
    while the excitation relaxes in the slow field."""
    env = solvated.environment
    mol, mf = solvated.mean_field()
    parts = {}
    for name, e in (('neq', env), ('eq', env.equilibrium_partner())):
        numerics = twin_numerics(solvated, fold=True)
        surface = RPABSESurface(mol, rhf, mf=mf, environment=e,
                                ground=twin(solvated, e, numerics=numerics), numerics=numerics)
        parts[name] = surface.energy()
    (_, e0_n, ehf_n, ec_n, om_n), (_, e0_e, ehf_e, ec_e, om_e) = (
        parts['neq'], parts['eq'])
    shift = (om_e - om_n) * HARTREE_TO_EV
    ok = check(matches((e0_e, ehf_e, ec_e), (e0_n, ehf_n, ec_n),
                       RECORDING_BLAS),
               'E_0, E_HF, E_c the same', f'E_c {ec_e:.12f}')
    ok &= check(abs(shift) > OMEGA_SHIFT_FLOOR_EV,
                'Omega moves with the slow field',
                f'Omega_eq - Omega_neq {shift * 1e3:+.1f} meV')
    assert ok


