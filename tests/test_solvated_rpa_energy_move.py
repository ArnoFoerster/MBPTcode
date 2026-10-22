"""The solvated dRPA fold: production's forward functional against an
independent spelling, to a few ulp.

`SingleReference/LinearResponse/solvated_rpa_energy.py` holds the forward
functional -- the reaction-field matrix N in the dressed auxiliary gauge, the
(screening, counter-term) pair of the fold, and the solute-solvent dispersion
term `solute_solvent_dispersion_energy` (not `Base.dispersion`'s empirical
D3/D4 correction). `gradients/solvated_rpa_energy.py` holds the reverse pass
(`fold_bar_to_gauge`) and the four-sweep diagnostic that drives it
(`solvated_correlation_energies`).

Every gate is a few ulp (`same`, `pins.matches`), not a method tolerance,
against the reference routines below, written out independently of
production. What breaks each:

  * the bare interaction dropped from `fold_terms`' counter-term (I - N
    spelled as I): the fold gate and the two-chain gate. The uncancelled first
    order moves `RPAGroundStateChain(fold=True)` from -0.244861130356 to
    -0.198584458853 Ha (29 kcal/mol) and turns the embedded fold's shift from
    -0.019901360 into +0.026374989 Ha. The three-energies gate does not see
    it, because the diagnostic builds its own counter-term out of N.
  * the symmetrization dropped from `screening_matrix`: N is 1.1e-13 from
    symmetric here, so every energy still prints the same twelve decimals and
    differs only in the last digits, which a few-ulp gate sees.
  * the sign of `solute_solvent_dispersion_energy` flipped: the dispersion
    gate; A, B and F do not read it, while F - A moves from +0.003208806543 to
    -0.032524897887 Ha.
  * production importing the reverse pass: the layering gate (a fresh
    interpreter; its positive control is the reverse module, which must bring
    `src.gradients` in).

H2O/cc-pVDZ in water.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.SingleReference.LinearResponse.solvated_rpa_energy as production
import src.gradients.solvated_rpa_energy as reverse
from src.Base.constants import SCF_DIFFERENTIABLE_GRAD_TOL, SCF_TIGHT_CONV_TOL
from src.Base.environment import NoEnvironment, dresses_interaction
from src.Base.numerics import Numerics
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import (SolventScreening,
                                        dynamic_screening_factor)
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.space_time_adjoint import (rpa_energy_and_adjoint,
                                              rpa_frequency_traces)
from tests.isdf_test_grid import g1
from tests.pins import RECORDING_BLAS, matches


def same(got, want, quantity='root', unit='1'):
    """Production and the reference spelling, two computations: a few ulp
    under the recording BLAS, the class bar elsewhere.
    """
    return matches(got, want, RECORDING_BLAS, unit=unit, quantity=quantity)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Hartree. The adiabatic limit and a physical pole energy, so the dispersion
#: gate sees both ends of g(iw) rather than one point of it.
POLE_ENERGIES = (1e8, 3.67e-2, 1e-4)


def check(ok, label, detail=''):
    """Print a named check with its measured number and return the verdict."""
    print(f'  [{"ok" if ok else "FAIL"}] {label}' + (f'   {detail}' if detail else ''))
    return bool(ok)


# ---------------------------------------------------------------------------
# the reference spelling every numeric gate below compares against
# ---------------------------------------------------------------------------

def reference_screening_matrix(auxmol, environment, V=None):
    """N = V_d^(-1/2) vtilde V_d^(-1/2), spelled out here."""
    kernel = environment.aux_kernel(auxmol)
    if kernel is None:
        return None
    V = auxmol.intor('int2c2e', aosym='s1') if V is None else V
    root = aux_metric_sqrt(auxmol, environment, V=V)
    n_mat = np.linalg.solve(root, np.linalg.solve(root, kernel).T).T
    return 0.5 * (n_mat + n_mat.T)


def reference_fold_terms(environment, auxmol, omega_points):
    """(screening, counter_term), spelled out here."""
    if not dresses_interaction(environment, auxmol):
        return None, None
    n_mat = reference_screening_matrix(auxmol, environment)
    if n_mat is None:
        return None, None
    g = np.asarray(environment.dynamic_factor(np.asarray(omega_points, float)),
                   float)
    return (n_mat, g), np.eye(n_mat.shape[0]) - n_mat


def reference_dispersion_energy(traces, grid, omega_p):
    """-(1/2pi) sum_w W_w g(w) Tr[vtilde P_1(iw)], spelled out here."""
    g = dynamic_screening_factor(grid.omega_points, omega_p)
    return -float(grid.omega_weights @ (g * np.asarray(traces, float))) / (
        2.0 * np.pi)


def scf_factory(mol):
    """A density-fitted RHF converged tightly enough for a frozen factorization."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    """(chain, auxmol, environment, factors) for H2O/cc-pVDZ in water."""
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    chain = RPAGroundStateChain(mol, scf_factory,
                                environment=SolventScreening(mol, solvent='water'), numerics=Numerics(isdf={'counts': g1(mol)}))
    m, mf = chain.mean_field()
    x_mo, d, eps, auxmol, _, _ = chain.factors_at(m, mf)
    return chain, auxmol, chain.environment_at(m), (x_mo, d, eps)


def test_the_callers_take_their_fold_terms_from_production():
    """The chain that folds reads the one production pair, so no route can
    assemble a counter-term of its own."""
    from src.gradients import rpa_ground_state
    ok = check(rpa_ground_state.fold_terms is production.fold_terms,
               'RPAGroundStateChain._fold_terms is the production pair')
    assert ok


def test_the_screening_matrix_is_the_old_one(water):
    """N is what the gradient module built, to a few ulp, with and without a V."""
    _, auxmol, env, _ = water
    V = auxmol.intor('int2c2e', aosym='s1')
    new = production.screening_matrix(auxmol, env)
    old = reference_screening_matrix(auxmol, env)
    ok = check(same(new, old), 'N is the reference N, to a few ulp',
               f'{new.shape} max |N| {np.abs(new).max():.6e}')
    ok &= check(same(production.screening_matrix(auxmol, env, V=V),
                     reference_screening_matrix(auxmol, env, V=V)),
                'and so is N built on a supplied metric')
    ok &= check(production.screening_matrix(auxmol, NoEnvironment()) is None,
                'nothing screens in the gas phase')
    assert ok


def test_the_fold_terms_are_the_old_ones(water):
    """(screening, counter_term) to a few ulp, and (None, None) where nothing
    dresses."""
    chain, auxmol, env, _ = water
    w = chain.grid.omega_points
    (n_new, g_new), c_new = production.fold_terms(env, auxmol, w)
    (n_old, g_old), c_old = reference_fold_terms(env, auxmol, w)
    ok = check(same(n_new, n_old),
               'the rescaled interaction is the reference one')
    ok &= check(same(g_new, g_old), 'and so is g(iw)',
                f'g in [{g_new.min():.6f}, {g_new.max():.6f}]')
    ok &= check(same(c_new, c_old), 'and so is the BARE counter-term',
                f'max |I - N - C_old| {np.abs(c_new - c_old).max():.1e}')
    ok &= check(production.fold_terms(NoEnvironment(), auxmol, w) == (None, None),
                'the gas phase has no fold to take')
    assert ok


def test_the_dispersion_term_is_the_old_one(water):
    """The solute-solvent dispersion at three pole energies, to a few ulp."""
    chain, auxmol, env, (x_mo, d, eps) = water
    n_mat = production.screening_matrix(auxmol, env)
    traces = rpa_frequency_traces(x_mo, d, eps, chain.nocc, chain.grid, n_mat)
    ok = True
    for w_p in POLE_ENERGIES:
        new = production.solute_solvent_dispersion_energy(traces, chain.grid, w_p)
        old = reference_dispersion_energy(traces, chain.grid, w_p)
        ok &= check(same(new, old, 'energy', 'Ha'), f'Omega_p = {w_p:.2e} Ha',
                    f'{new:.12f} vs {old:.12f} Ha')
    assert ok


def test_the_three_energies_are_the_old_ones(water):
    """A, B, F and the split of F - A, against the reference spelling on the
    same factors.

    The diagnostic stays in the gradient package because it drives
    `rpa_energy_and_adjoint` four times, but everything it folds with comes
    from production: this is the gate that a dropped or unmatched term in
    either half of the pair breaks.
    """
    chain, auxmol, env, (x_mo, d, eps) = water
    grid = chain.grid
    res = reverse.solvated_correlation_energies(chain)
    (n_old, _), c_old = reference_fold_terms(env, auxmol, grid.omega_points)
    g_old = dynamic_screening_factor(grid.omega_points, res.omega_p)

    def energy(**kw):
        return float(rpa_energy_and_adjoint(x_mo, d, eps, chain.nocc, grid,
                                            want_grad=False, **kw))

    old = {'dressed': energy(),
           'bare': energy(screening=(n_old, np.zeros(grid.nfreq)),
                          counter_term=c_old),
           'fold': energy(screening=(n_old, g_old), counter_term=c_old)}
    traces = rpa_frequency_traces(x_mo, d, eps, chain.nocc, grid, n_old)
    old['dispersion'] = reference_dispersion_energy(traces, grid, res.omega_p)
    old['beyond_first_order'] = old['fold'] - old['bare'] - old['dispersion']
    ok = check(same(res.traces, traces),
               'Tr[vtilde P_1(iw)] is the reference trace')
    for name, value in old.items():
        ok &= check(same(getattr(res, name), value, 'energy', 'Ha'),
                    f'{name} is the reference {name}',
                    f'{getattr(res, name):.12f} vs {value:.12f} Ha')
    ok &= check(type(res) is production.SolvatedCorrelation,
                'and the record is the production dataclass',
                type(res).__module__)
    assert ok


def test_the_folding_chain_is_the_diagnostic_fold(water):
    """`RPAGroundStateChain(fold=True)` folds to the diagnostic's F, to a few
    ulp."""
    chain, _, _, _ = water
    folded = RPAGroundStateChain(chain.mol0, scf_factory, numerics=Numerics(isdf={'auxbasis': BASIS + '-ri', 'counts': g1(chain.mol0)},
                                                                            solvent={'fold': True}),
                                 environment=chain.environment)
    e_fold = folded.correlation_energy()
    ok = check(same(e_fold, reverse.solvated_correlation_energies(chain).fold,
                    'energy', 'Ha'),
               'RPAGroundStateChain(fold=True) is the diagnostic F',
               f'{e_fold:.12f} Ha')
    assert ok


def test_production_does_not_import_the_gradient_package():
    """Importing the functional must not drag the reverse pass in with it.

    A fresh interpreter, because `sys.modules` in this one already carries the
    gradient package: the positive control below is the same probe importing
    the reverse module, which must bring `src.gradients` in.
    """
    probe = (
        'import sys; sys.path.insert(0, {0!r});'
        'import {1};'
        'print(sorted(m for m in sys.modules if m.startswith("src.gradients")))')

    def loaded(module):
        out = subprocess.run([sys.executable, '-c', probe.format(REPO, module)],
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr[-2000:]
        return eval(out.stdout.strip().splitlines()[-1])

    forward = loaded('src.SingleReference.LinearResponse.solvated_rpa_energy')
    ok = check(not forward, 'production pulls in no src.gradients module',
               f'{forward}' if forward else 'none')
    backward = loaded('src.gradients.solvated_rpa_energy')
    ok &= check(bool(backward), 'while the reverse module does, so the probe '
                'can see one', f'{len(backward)} modules')
    assert not forward
    assert ok


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-q']))
