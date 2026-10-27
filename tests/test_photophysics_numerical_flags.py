"""The photophysics stages warn about a numerical event and name it in the
record instead of refusing the run (a bad input is refused; a numerical
shortfall warns and carries a named flag).

1. A ground-state walk that did not converge: s0 carries on where it stopped
   and its record says 'walk_not_converged'.
2. An excited walk that did not converge is kept as '<stage>_walked' with the
   flag in its info, which the stage record copies (`walk_fields`).
3. A mean field whose SCF stopped short is warned about by the factory
   (`chain.SCFNotConverged`) and the stage that built it is flagged
   'scf_not_converged', and a named flag raised inside a stage is named in
   its record.
4. The rate table gathers every stage record's flags by stage, and a failed
   ISDF-vs-DF acceptance check is warned about and flagged.
5. A Franck-Condon integrand that had not decayed is named in the rate
   table's flags ('fc_integrand_not_decayed').

Stand-ins for the walks and the table; one formaldehyde SCF of one cycle.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerical_flags import flag
from src.photophysics import chain, stages
from src.photophysics.records import walk_fields
from src.properties.excitations import WalkNotRelaxed
from src.photophysics import rates
from src.properties import franck_condon
from tests.photophysics_cases import run_of

UNCONVERGED = {'converged': False, 'status': 'max cycles', 'cycles': 3,
               'energy': -1.0, 'opt_grad_max': 3e-3, 'history': [{}] * 3}


def test_an_unconverged_ground_walk_is_flagged_not_refused(tmp_path,
                                                          monkeypatch):
    run = run_of(tmp_path)
    monkeypatch.setattr(stages, 'MeanFieldSurface', lambda *a, **k: None)
    monkeypatch.setattr(stages, 'relax',
                        lambda surface, mol, **kw: (mol, dict(UNCONVERGED)))
    with pytest.warns(WalkNotRelaxed, match='S0 did not relax: max cycles'):
        record = stages.ground_minimum(run)
    assert record['numerical_flags'] == {'walk_not_converged': 'max cycles'}
    assert run.read('s0')['numerical_flags'] == record['numerical_flags']
    assert run.mol0 is not None


def test_an_unconverged_excited_walk_is_kept_and_flagged(tmp_path,
                                                        monkeypatch):
    run = run_of(tmp_path)
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    kept = {}
    monkeypatch.setattr(run, 'walk', lambda *a, **k: (mol, dict(UNCONVERGED),
                                                      0, 'surface'))
    monkeypatch.setattr(run, 'write_walked',
                        lambda stage, *a: kept.setdefault(stage, a))
    with pytest.warns(WalkNotRelaxed, match='S1 did not relax'):
        _, info, root, final = stages.walked_minimum(run, 's1', 'surface',
                                                     mol, None)
    assert 's1' in kept, 'the walk is stored as walked, not refused'
    assert info['numerical_flags'] == {'walk_not_converged': 'max cycles'}
    assert walk_fields(info)['numerical_flags'] == info['numerical_flags']


def test_a_converged_walk_carries_no_flag():
    info = dict(UNCONVERGED, converged=True, status='converged')
    with warnings.catch_warnings():
        warnings.simplefilter('error', WalkNotRelaxed)
        assert stages.relaxation_flags('t1', info) == {}
    assert walk_fields(info)['numerical_flags'] == {}


def test_an_unconverged_scf_is_warned_and_flags_its_stage(tmp_path):
    run = run_of(tmp_path)
    numerics = run.declaration.numerics.derive(
        'one SCF cycle: an SCF that stops short',
        scf={'route': 'df', 'grid_level': None, 'max_cycle': 1})
    factory = run.mean_field_factory(numerics)

    def stage(run_):
        with pytest.warns(chain.SCFNotConverged, match='did not converge'):
            factory(run_.mol_in)
        run_.write('s0', {'coords_bohr': run_.mol_in.atom_coords(),
                          'numerical_flags': {}})
        return run_.read('s0')
    record = stages.flag_scfs(run, 's0', stage)
    assert run.unconverged_scfs() == 1
    assert record['numerical_flags'] == {'scf_not_converged': 1}


def test_a_flag_raised_inside_a_stage_is_named_in_its_record(tmp_path):
    run = run_of(tmp_path)

    def stage(run_):
        flag('planted', 'a shortfall inside the stage', RuntimeWarning)
        run_.write('s0', {'coords_bohr': run_.mol_in.atom_coords(),
                          'numerical_flags': {}})
        return run_.read('s0')
    with pytest.warns(RuntimeWarning, match='inside the stage'):
        record = stages.flag_scfs(run, 's0', stage)
    assert record['numerical_flags'] == {
        'planted': ['a shortfall inside the stage']}


def stub_run(tmp_path, records):
    """A run whose store holds `records` and writes into `written`."""
    written = {}
    return SimpleNamespace(
        declaration=SimpleNamespace(
            rate_model=SimpleNamespace(name='tadf'), system=SimpleNamespace(
                name='stub'), environment='gas'),
        reusable=lambda n: n in records, read=lambda n: records[n],
        write=written.__setitem__, store=SimpleNamespace(dir=str(tmp_path)),
        written=written)


def test_the_rate_table_gathers_every_stage_flag(tmp_path, monkeypatch):
    passed = {'isdf_df_check': {'verdict': 'pass'}}
    records = {'s0': {'numerical_flags': {'scf_not_converged': 2}},
               'hess': {}, 'vert': dict(passed),
               's1': dict(passed, numerical_flags={
                   'walk_not_converged': 'max cycles'}),
               't1': {'isdf_df_check': {'verdict': 'fail'},
                      'numerical_flags': {}}}
    run = stub_run(tmp_path, records)
    monkeypatch.setattr(stages, 'GroundModes',
                        SimpleNamespace(from_record=lambda r: r))
    monkeypatch.setattr(stages, 'from_gw_bse_record', lambda r, w: w)
    rules = stages.RATE_RULES['tadf']
    monkeypatch.setitem(stages.RATE_RULES, 'tadf', rules._replace(
        table=lambda run, minima, records: {
            'numerical_flags': {'fc_integrand_not_decayed': ['S1-T1']}},
        printer=lambda *a: None))
    with pytest.warns(stages.AcceptanceCheckFailed, match='failed at t1'):
        table = stages.rates(run)
    assert table['numerical_flags'] == {
        's0': {'scf_not_converged': 2},
        's1': {'walk_not_converged': 'max cycles'},
        't1': {'isdf_df_check_failed': 'fail'},
        'rate': {'fc_integrand_not_decayed': ['S1-T1']}}
    assert run.written['rate'] is table


def test_an_undecayed_fc_integrand_warns_with_its_class():
    # one mode and no bath recurs forever
    with pytest.warns(franck_condon.FCIntegrandNotDecayed, match='had not decayed'):
        franck_condon.fc_weighted_dos(-0.02, [1.0], [0.01], 300.0, t_decay=0.5)


def test_the_rate_table_names_an_undecayed_density(monkeypatch):
    real = franck_condon.fc_weighted_dos
    omega = np.array([0.004, 0.008, 0.015])

    def truncated(e, s, w, temperature, **kw):
        if s[0] < 0.2:      # only the S0 <-> S1 density
            warnings.warn('planted: had not decayed',
                          franck_condon.FCIntegrandNotDecayed)
        return real(e, s, w, temperature, **kw)

    def minimum(where):
        s1 = -0.900 + (0.006 if where == 'T1' else 0.0)
        t1 = -0.904 + (0.006 if where == 'S1' else 0.0)
        return rates.Minimum(emitter='S1', reservoir='T1',
                             energy={'S1': s1, 'T1': t1},
                             socme=np.array([[1e-6], [2e-6]]),
                             emission=s1 + 1.0, dipole=np.array([0.5, 0, 0]),
                             f_osc=0.1, e_ground=-0.995)
    s_k = 0.006 / omega.sum() * np.ones(3)
    s1, t1 = minimum('S1'), minimum('T1')
    clean = rates.rate_table(s1, t1, -1.0, omega, s_k, 300.0, 0.002,
                             s_ic=10 * s_k / 12, d_ic=np.ones(3) * 1e-3)
    assert clean['numerical_flags'] == {}
    monkeypatch.setattr(rates, 'fc_weighted_dos', truncated)
    with pytest.warns(franck_condon.FCIntegrandNotDecayed, match='planted'):
        flagged = rates.rate_table(s1, t1, -1.0, omega, s_k, 300.0, 0.002,
                                   s_ic=10 * s_k / 12,
                                   d_ic=np.ones(3) * 1e-3)
    assert flagged['numerical_flags'] == {
        'fc_integrand_not_decayed': ['S0-S1']}
    assert flagged['k_ic'] == clean['k_ic']
