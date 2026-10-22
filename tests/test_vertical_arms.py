"""The vertical arms and the production walk declare ONE chain.

The walk of a run and
the vertical arms (`photophysics.vertical_arms`) of the same declaration build
the same `SurfaceSpec` (the quasiparticle declaration aside, which an arm names
itself) and the same Kohn-Sham mean field through `chain.chain_spec` and
`chain.mean_field_factory`; a force-taking walk converges its mean field for
forces, the arms for an energy; neither keeps a private copy of the continuum,
auxiliary-set, element or span helpers. Checked on the library functions, the
mean field factories recorded and not run. The last test runs one arm (water,
cc-pVDZ, 'sop-admitted') end to end; it is cheap enough to run always.
"""
import ast
import pathlib
from dataclasses import replace

import pytest

from src.Base.constants import (SCF_ENERGY_CONV_TOL, SCF_ENERGY_GRAD_TOL,
                                SCF_OPTIMIZATION_CONV_TOL,
                                SCF_OPTIMIZATION_GRAD_TOL)
from src.photophysics import chain, vertical_arms
from src.photophysics.cli import declaration_of, parser
from tests.photophysics_cases import (calls, halves_electrons,
                                      occupied_counts_by_hand, run_in)

WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'
PACKAGE = pathlib.Path(vertical_arms.__file__).resolve().parent
#: Options the arms must read from the declaration, all set.
SHARED = ['--grid-accuracy', 'G1', '--nroots', '5', '--n-states', '2',
          '--outside-scissor', 'tiers', '--track', 'overlap',
          '--qp-hamiltonian', '--bse-conv-tol', '1e-6',
          '--reference', 'pbe0', '--scf', 'isdf']


def declaration(out, *extra):
    argv = ['--atom', WATER, '--system', 'water', '--basis', 'cc-pvdz',
            '--environment', 'gas', '--temperature', '300',
            '--lambda-env', '0.05', '--nudge', '0', '--no-far-start',
            '--refreeze', '0', '--saddle-check', '0', '--df-check', '0',
            '--hessian', 'analytic-df', '--relaxed-solvation', 'equilibrium',
            '--span', 'full', '--manifold-gaps', 'stacked',
            '--qp-states', 'admitted', '--out', str(out),
            *SHARED, *extra]
    return declaration_of(parser().parse_args(argv))


@pytest.fixture
def one(tmp_path):
    d = declaration(tmp_path)
    return d, run_in(tmp_path, d)


def test_walk_and_arms_declare_the_same_chain(one):
    d, run = one
    for env in (None, object()):
        arm = chain.chain_spec(
            vertical_arms.arm_declaration(d, 'sop-admitted'), env)
        assert arm == chain.chain_spec(d, env)
    assert run.spec == chain.chain_spec(
        vertical_arms.arm_declaration(d, 'ac-all'), run.env)
    numerics = dict(run.spec.numerics)
    assert numerics['track'] == 'overlap'
    assert numerics['qp_hamiltonian'] is True
    assert numerics['bse_conv_tol'] == 1e-6


def test_the_arms_name_only_the_quasiparticle_declaration(one):
    d, _ = one
    adaptive = vertical_arms.arm_declaration(d, 'sop-adaptive')
    assert adaptive.qp_states.kind == 'adaptive'
    assert tuple(adaptive.qp_states.targets) == (('singlet', 0),
                                                 ('triplet', 0))
    assert replace(adaptive, qp_states=d.qp_states) == d
    assert vertical_arms.arm_declaration(d, 'ac-all').qp_states.kind == \
        'admitted'
    with pytest.raises(ValueError):
        vertical_arms.arm_declaration(d, 'no-such-arm')


def test_one_mean_field_factory(one, monkeypatch):
    """Both are `chain.mean_field_factory`: the same functional, route, grid,
    span and finish; the walk converges for forces (SCF_OPTIMIZATION_*), the
    arms for an energy (SCF_ENERGY_*), derived from the walk's numerics."""
    d, run = one
    calls = []
    monkeypatch.setattr(chain, 'mean_field_factory',
                        lambda *a, **k: calls.append((a, k)))
    run.mean_field_factory(d.numerics)
    vertical_arms.arm_mean_field_factory(
        d, 4000, [], (run.span['n_kept'], run.span['cut']))
    (a_run, k_run), (a_arm, k_arm) = calls
    assert a_run[0] == a_arm[0] == 'pbe0'
    walk, arms = a_run[1], a_arm[1]
    assert (walk.scf.route, walk.scf.grid_level) == (arms.scf.route,
                                                     arms.scf.grid_level)
    assert (walk.scf.conv_tol, walk.scf.conv_tol_grad) == (
        SCF_OPTIMIZATION_CONV_TOL, SCF_OPTIMIZATION_GRAD_TOL)
    assert (arms.scf.conv_tol, arms.scf.conv_tol_grad) == (
        SCF_ENERGY_CONV_TOL, SCF_ENERGY_GRAD_TOL)
    assert arms.parent == d.numerics.identity()
    assert 'energies only' in arms.reason
    for key in ('finish', 'span', 'surface_route'):
        assert k_run[key] == k_arm[key], key


def test_no_private_copies_of_the_chain_helpers():
    """The arms and the acceptance check read `chain`: they define none of
    its helpers, and count occupied orbitals and move molecules through
    `record_io`."""
    helpers = {n.name for n in ast.parse(
        (PACKAGE / 'chain.py').read_text()).body
        if isinstance(n, ast.FunctionDef)}
    for name in ('vertical_arms.py', 'df_check.py'):
        tree = ast.parse((PACKAGE / name).read_text())
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        assert not defined & helpers, (name, defined & helpers)
        assert calls(tree, 'set_geom_') == [], name
        assert halves_electrons(tree) == [], name
        assert occupied_counts_by_hand(tree) == [], name


@pytest.mark.parametrize('arm', vertical_arms.ARMS)
def test_each_arm_runs(tmp_path, arm):
    """Each arm end to end on the library function: water, cc-pVDZ, the
    sum-over-poles residues on the admitted window ('sop-admitted'), on the
    adaptive set ('sop-adaptive'), and every orbital by the analytic
    continuation ('ac-all'), which also records the admitted spectrum of the
    same forward pass."""
    d = declaration(tmp_path)
    rec = vertical_arms.vertical_arm(d, arm, max_memory=4000)
    assert rec['arm'] == arm and 'refusal' not in rec
    s = rec['spectrum']
    assert 5.0 < s['S1'] < 12.0 and s['T1'] < s['S1']
    assert s['dEST'] == pytest.approx(s['S1'] - s['T1'])
    assert s['gap_eV'] > 0
    assert rec['n_explicit'] <= rec['n_orbitals']
    if arm == 'ac-all':
        assert rec['ac']['ac_failed'] == []
        # AC and SOP differ by the residue treatment and the scissor alone
        assert abs(rec['ac_minus_sop_same_forward_eV']['S1']) < 0.2
