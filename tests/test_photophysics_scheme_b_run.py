"""The stage pipeline under solvent scheme B, evaluated.

  1. forces: G_eq of S0, S1 and T1 on the pipeline's own scheme-B surfaces
     (`Run.loop_surface`: the declared chain inside the loop, water
     aug-cc-pVDZ in toluene, the AO-product fit on 600 points, TDA, SCF to 1e-12 / 1e-10,
     loops to 1e-10 e / 1e-12 Ha) against five-point differences of G_eq
     along a random direction on the same surface (h = 2e-3 Bohr), within
     1e-8 Ha/Bohr. Scheme B has no density-fitted route: its forces are
     gated against their own energy.
  2. the stages (formaldehyde cc-pVDZ in toluene, the AO-product fit on 900
     points, the frontier set, the outside scissor by tiers, TDA, geomeTRIC): s0, vert, s1, t1 and rate run;
     every record names the scheme; vert's omega of S1 and T1 are their
     absorption and the S1 (T1) minimum's its emission, its energy G_eq
     there, the walk-versus-vertical gap flagged above the refreeze bar; the rate table's adiabatic energies are differences of G_eq; a
     second launch reads every stage back and evaluates nothing.
  3. restart (no refreeze, scheme-b-noneq): a T1 walk stopped after its
     first accepted step is resumed by the next launch as a new walk from
     that step's geometry on a surface built there (`restarted_from` in the
     record), and reaches the minimum; its energies are the non-equilibrium
     ones (T1 in S0's slow charges at its minimum, S0 in its own).
  4. the saddle check on a scheme-B surface (water cc-pVDZ, S0 at its loop
     minimum): a minimum, its displaced forces cached.

Run: python -m pytest tests/test_photophysics_scheme_b_run.py
"""
import numpy as np
import pytest

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (HARTREE_TO_EV, HARTREE_TO_MEV,
                                         REFREEZE_TOL_MEV)
from src.photophysics import stages
from src.photophysics.stages import run_tadf
from src.photophysics.store import Store
from tests.photophysics_cases import Stop, declaration, refusing, run_in

WATER = ('--system', 'water', '--atom',
         'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
         '--basis', 'aug-cc-pvdz', '--pivot-points', '600')
SCHEME_B = ('--environment', 'toluene', '--relaxed-solvation', 'equilibrium',
            '--solvent-scheme', 'scheme-b-eq', '--scf', 'df', '--tda',
            '--grid-accuracy', 'G1', '--isdf-test-set', 'ao_products',
            '--pivot-points', '900', '--engine', 'auto', '--qp-states',
            'frontier')
TIGHT = ('--scf-conv-tol', '1e-12', '--scf-grad-tol', '1e-10',
         '--loop-tol-q', '1e-10', '--loop-tol-energy', '1e-12')
H = 2e-3
FORCE_BAR = 1e-8


@pytest.mark.parametrize('kind', ['S0', 'singlet', 'triplet'])
def test_the_force_is_the_derivative_of_g_eq(tmp_path, kind):
    run = run_in(tmp_path, declaration(*SCHEME_B, *WATER, *TIGHT))
    mol = run.mol_in
    surface = run.loop_surface(kind, mol, 'force gate')
    gradient = surface.total_gradient(mol)[0]
    direction = np.random.default_rng(7).standard_normal(gradient.shape)
    direction /= np.linalg.norm(direction)

    def g(step):
        return surface.total_energy(molecule_at(
            mol, mol.atom_coords() + step * direction))
    fd = (-g(2 * H) + 8 * g(H) - 8 * g(-H) + g(-2 * H)) / (12 * H)
    analytic = float((gradient * direction).sum())
    print(f'{kind}: dG/dx.u {analytic:.12f}, differences {fd:.12f}, '
          f'|diff| {abs(analytic - fd):.2e} Ha/Bohr')
    assert abs(analytic - fd) < FORCE_BAR


#: the outside scissor by tiers: on formaldehyde/cc-pVDZ the virtual
#: outside curve overshoots its knots at some walk geometries
#: (CurveOvershoots, which refuses), and each scheme-B surface fits its own
FORMALDEHYDE_B = SCHEME_B + ('--hessian', 'analytic-df', '--outside-scissor',
                             'tiers')


def test_the_stages_under_scheme_b(tmp_path, monkeypatch):
    d = declaration(*FORMALDEHYDE_B)
    out = run_tadf(d, Store(tmp_path, d), max_memory=4000)
    for name in ('s0', 'hess', 'vert', 's1', 't1', 'rate'):
        assert out[name]['solvent_scheme'] == 'scheme-b-eq'
    vert, s1, t1 = out['vert'], out['s1'], out['t1']
    for key in ('S1', 'T1'):
        assert vert['props']['omega'][key] * HARTREE_TO_EV == pytest.approx(
            vert['absorption_eV'][key], abs=1e-12)
    assert s1['props']['energy']['S1'] == s1['G_eq']
    assert s1['props']['omega']['S1'] * HARTREE_TO_EV == pytest.approx(
        s1['emission_eV'], abs=1e-12)
    assert t1['props']['energy']['T1'] == t1['G_eq']
    # the walk's G_eq against the vertical's at the minimum, flagged above the
    # refreeze bar (both walks are refrozen)
    for rec in (s1, t1):
        assert rec['walk_vs_vertical_flag'] == (
            abs(rec['walk_vs_vertical_G_Ha']) > REFREEZE_TOL_MEV / HARTREE_TO_MEV)
        assert ('walk_vs_vertical' in rec['numerical_flags']) == \
            rec['walk_vs_vertical_flag']
    # adiabatic dE_ST of the table is the difference of G_eq at each minimum
    adiabatic = (s1['G_eq'] - t1['G_eq']) * HARTREE_TO_EV
    rate = out['rate']
    assert rate['dE_ST_adiabatic_eV'] == pytest.approx(adiabatic, abs=1e-9)
    assert rate['S1_adiabatic_eV'] == pytest.approx(
        (s1['G_eq'] - vert['props']['e0']) * HARTREE_TO_EV, abs=1e-9)

    # a second launch evaluates nothing
    monkeypatch.setattr(stages.state_specific, 'vertical', refusing(
        'a stored vertical was evaluated again'))
    monkeypatch.setattr(stages.SolvatedSurface, 'total_energy', refusing(
        'a stored walk was evaluated again'))
    again = run_tadf(d, Store(tmp_path, d), max_memory=4000)
    for name in ('vert', 's1', 't1'):
        assert again[name]['props']['e0'] == out[name]['props']['e0']


def test_a_stopped_walk_restarts_from_its_last_step(tmp_path, monkeypatch):
    d = declaration(*FORMALDEHYDE_B, '--refreeze', '0', '--solvent-scheme',
                    'scheme-b-noneq', '--relaxed-solvation', 'nonequilibrium')
    run_tadf(d, Store(tmp_path, d), max_memory=4000,
             stages=('s0', 'hess', 'vert', 's1'))
    original = stages.Run.write

    def stop_after_one(self, name, record):
        original(self, name, record)
        if name == 't1_progress' and len(record['state']['history']) >= 2:
            raise Stop
    monkeypatch.setattr(stages.Run, 'write', stop_after_one)
    with pytest.raises(Stop):
        run_tadf(d, Store(tmp_path, d), max_memory=4000, stages=('t1',))
    monkeypatch.setattr(stages.Run, 'write', original)
    t1 = run_tadf(d, Store(tmp_path, d), max_memory=4000,
                  stages=('t1',))['t1']
    assert t1['restarted_from']['accepted_steps'] >= 2
    assert t1['converged']
    # scheme-b-noneq: T1 at its minimum in S0's slow charges, S0 in its own
    # equilibrium there, so omega(T1) is the absorption at the T1 minimum
    G = t1['emission_G']
    assert t1['props']['energy']['T1'] == G['neq_state']
    assert t1['props']['e0'] == G['eq_ground']
    assert t1['solvent_scheme'] == 'scheme-b-noneq'
    assert t1['solvation'] == 'nonequilibrium'


def test_a_saddle_check_on_a_scheme_b_surface(tmp_path):
    """The saddle check (6N differenced forces of the surface) runs on the
    loop's surface: water cc-pVDZ in toluene, its scheme-B S0 minimum is a
    minimum, and every displaced force is kept in the stage's cache."""
    d = declaration(*SCHEME_B, *WATER, '--basis', 'cc-pvdz',
                    '--saddle-check', '1')
    run = run_in(tmp_path, d)
    stages.ground_minimum(run)
    surface = run.loop_surface('S0', run.mol0, 'saddle check')
    info = {'converged': True, 'numerical_flags': {}}
    mol, info, _, _, record = run.saddle_checked('s1', run.mol0, info, None,
                                                 surface, 0.0)
    assert record['status'] == 'minimum', record['status']
    assert not record['first']['saddle']
    assert 'saddle_check' not in info['numerical_flags']
    assert any(run.store.workdir('hess_s1').iterdir())
