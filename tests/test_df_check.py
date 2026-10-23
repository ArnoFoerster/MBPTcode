"""The ISDF-vs-DF acceptance check (`photophysics.df_check`).

After the vertical stage the check of the stage's geometry runs once, its output is kept as
'isdf_df_check_vert' and its verdict is written into the stage record as
`isdf_df_check`; a record holding one is not checked again; and the rate stage
of a run any of whose checked stages failed warns and flags its table
(refuse inputs, warn numerics). Formaldehyde, cc-pVDZ, gas phase,
the vertical record produced by `stages.vertical` on a small `Run` from an S0
record at the input geometry (no walk).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Base.numerics.constants import (DF_CHECK_ORBITAL_MATCH_TOL_EV, EV_TO_MEV,
                                         HARTREE_TO_EV, ISDF_GRID_ACCURACY_TARGET_EV)
from src.photophysics import df_check, stages
from src.photophysics.cli import declaration_of, parser
from tests.photophysics_cases import FORMALDEHYDE, run_in


def declaration(out):
    argv = ['--atom', FORMALDEHYDE, '--system', 'formaldehyde', '--basis', 'cc-pvdz',
            '--environment', 'gas', '--temperature', '300',
            '--lambda-env', '0.05', '--nudge', '0', '--no-far-start',
            '--refreeze', '0', '--saddle-check', '0', '--df-check', '1',
            '--hessian', 'analytic-df', '--relaxed-solvation', 'equilibrium',
            '--span', 'full', '--manifold-gaps', 'stacked',
            '--qp-states', 'admitted', '--out', str(out)]
    return declaration_of(parser().parse_args(argv))


@pytest.fixture(scope='module')
def vert(tmp_path_factory):
    """(run, vertical record after accept) of formaldehyde at its input geometry."""
    out = tmp_path_factory.mktemp('df_check')
    d = declaration(out)
    run = run_in(out, d)
    run.write('s0', {'coords_bohr': run.mol_in.atom_coords()})
    stages.ground_minimum(run)
    return run, stages.vertical(run)


def test_accept_records_the_check(vert):
    """The check passes at the level's target and its verdict and output sit
    in the stage record."""
    run, record = vert
    check = record['isdf_df_check']
    assert check['verdict'] == 'pass', check
    assert check['output'] == 'isdf_df_check_vert'
    out = run.read('isdf_df_check_vert')
    target = ISDF_GRID_ACCURACY_TARGET_EV['G1']
    assert out['target_eV'] == target and out['reason'] is None
    assert set(out['spins']) == {'singlet', 'triplet'}
    for spin in out['spins'].values():
        assert abs(spin['isdf_minus_df_meV']) <= EV_TO_MEV * target
        assert spin['pass'] is True
    assert out['orbital_match_eV'] < DF_CHECK_ORBITAL_MATCH_TOL_EV


def test_a_stored_check_is_not_run_again(vert, monkeypatch):
    run, _ = vert
    monkeypatch.setattr(df_check, 'check_stage', lambda *a: 1 / 0)
    run.accept('vert')


def test_a_failed_check_flags_the_rate(vert, monkeypatch):
    """A stage record whose acceptance check failed is warned about and
    named in the rate table's flags; the table is computed."""
    run, _ = vert
    for name in ('hess', 's1', 't1'):
        run.write(name, {'isdf_df_check': {'verdict': 'pass'}})
    run.store.update('vert', run.stamp('vert'),
                     isdf_df_check={'verdict': 'fail', 'reason': 'planted'})
    # the table itself is not this test's: a stand-in on the records
    monkeypatch.setattr(stages, 'GroundModes', SimpleNamespace(
        from_record=lambda r: r))
    monkeypatch.setattr(stages, 'from_gw_bse_record', lambda r, w: w)
    rules = stages.RATE_RULES['tadf']
    monkeypatch.setitem(stages.RATE_RULES, 'tadf', rules._replace(
        table=lambda run, minima, records: {'k': 1.0},
        printer=lambda *a: None))
    with pytest.warns(stages.AcceptanceCheckFailed,
                      match='acceptance check failed at vert'):
        table = stages.rates(run)
    assert table['numerical_flags']['vert']['isdf_df_check_failed'] == 'fail'
    assert run.read('rate')['numerical_flags'] == table['numerical_flags']


def test_the_diagonal_is_the_chains_on_its_set_and_scissored_outside():
    eps = np.array([-0.5, -0.3, 0.1, 0.4])
    mf = SimpleNamespace(mo_energy=eps)
    book = {'n_orbitals': 4, 'qp_root_eV': {1: -0.35 * HARTREE_TO_EV},
            'eps_mean_field_eV': {1: -0.3 * HARTREE_TO_EV + 2e-3},
            'scissor_eV': {3: 1.0}, 'inside_scissor_eV': {}}
    diag, dev = df_check.chain_diagonal(mf, book)
    assert diag[1] == pytest.approx(-0.35)
    assert diag[0] == eps[0] and diag[2] == eps[2]
    assert diag[3] == pytest.approx(0.4 + 1.0 / HARTREE_TO_EV)
    assert dev == pytest.approx(2e-3)
    with pytest.raises(RuntimeError):
        df_check.chain_diagonal(mf, dict(book, n_orbitals=5))
