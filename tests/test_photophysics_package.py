"""The photophysics package's own contracts: what a declaration must name,
what the store refuses, what a relaunch reuses, the command line, and a
method-neutral minimum through the rate table.

  * a declaration that leaves a physics policy out is refused (no field has
    a default), and so is a numerical choice the chain could not read;
  * a record written under one declaration is refused under another, by the
    fields it differs in, and read back to the bit under its own;
  * `run_tadf` on a store that holds s0 and hess runs neither again: no walk,
    no Hessian, the records as written;
  * the command line refuses a run without its policies and, given them, makes
    the declaration and stops before the first SCF (`--dry-run`); without
    --stages it runs every stage of the declared rate model;
  * ADC(2) excitation energies at three geometries fill `StateMinimum`
    records that `tadf_rate_table` reads like GW-BSE ones; the channels whose
    ingredients ADC(2) leaves out are refused by name, and R0, S1 and T1 of
    two methods are not mixed in one table.
"""
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.photophysics import cli, stages
from src.photophysics.declaration import DeclarationRefused, Policies
from src.photophysics.minimum import (GroundModes, StateMinimum, from_adc2,
                                      from_gw_bse_record)
from src.photophysics.rate_models import RateRefused, tadf_rate_table
from src.photophysics.stages import Run, run_tadf
from src.photophysics.store import Store, StoreRefused
from src.properties.vibronic import normal_modes
from tests.photophysics_cases import (REQUIRED, argv, declaration, refusing,
                                      run_in)

#: Water, its two displaced stand-in minima (Angstrom), small enough for a
#: dense ADC(2) and an analytic Hessian in seconds.
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
S1_AT = 'O 0 0 0.1273; H 0 0.7872 -0.4892; H 0 -0.7872 -0.4892'
T1_AT = 'O 0 0 0.1223; H 0 0.7772 -0.4792; H 0 -0.7772 -0.4792'


# ----------------------------------------------------------- declaration
def test_a_declaration_without_a_policy_is_refused():
    d = declaration()
    fields = dict(d.policies.__dict__)
    fields.pop('nudge_bohr')
    with pytest.raises(TypeError, match='nudge_bohr'):
        Policies(**fields)
    with pytest.raises(DeclarationRefused, match='refreeze'):
        replace(d, policies=replace(d.policies, refreeze=None))
    with pytest.raises(DeclarationRefused, match='temperature'):
        replace(d, temperature=None)
    with pytest.raises(DeclarationRefused, match='hessian_numerics'):
        replace(d, policies=replace(d.policies, hessian='numerical'))


def test_a_choice_the_chain_cannot_read_is_refused():
    d = declaration()
    numerics = d.numerics.derive('a tau target the flat names cannot carry',
                                 quadrature={'tau_target': 1e-9})
    run_decl = replace(d, numerics=numerics)
    with pytest.raises(ValueError, match='quadrature.tau_target'):
        stages.chain.chain_spec(run_decl, None)


def test_the_identity_moves_with_every_policy():
    d = declaration()
    assert d.identity() == declaration().identity()
    for change in ({'nudge_bohr': 0.0}, {'df_check': True},
                   {'saddle_check': True}, {'refreeze': False}):
        other = replace(d, policies=replace(d.policies, **change))
        assert other.identity() != d.identity(), change


# ----------------------------------------------------------------- store
def test_a_record_of_another_declaration_is_refused(tmp_path):
    d = declaration()
    run = run_in(tmp_path, d)
    record = {'coords_bohr': run.mol_in.atom_coords(), 'x': np.arange(3.0)}
    run.write('s0', record)
    back = run.read('s0')
    assert np.array_equal(back['coords_bohr'], record['coords_bohr'])
    assert back['x'].dtype == record['x'].dtype
    held = run.store.held('s0')
    assert {'commit', 'blas', 'host'} <= set(held['provenance'])
    other = declaration('--scf-grid', 'G3')
    run_other = run_in(tmp_path, other)
    with pytest.raises(StoreRefused, match='mean_field'):
        run_other.read('s0')
    # a policy the ground state does not read leaves its record reusable
    nudged = declaration('--nudge', '0')
    assert run_in(tmp_path, nudged).reusable('s0')
    run.write('s1', {'k': 1})
    with pytest.raises(StoreRefused, match='nudge_bohr'):
        run_in(tmp_path, nudged).read('s1')


# ---------------------------------------------------------------- resume
def test_run_tadf_reuses_what_the_store_holds(tmp_path, monkeypatch):
    d = declaration()
    run = run_in(tmp_path, d)
    run.write('s0', {'coords_bohr': run.mol_in.atom_coords()})
    run.write('hess', {'omega': np.ones(6), 'route': 'analytic-df'})

    refused = refusing('a stored stage ran again')
    monkeypatch.setattr(stages, 'relax', refused)
    monkeypatch.setattr(stages, 'normal_modes', refused)
    monkeypatch.setattr(Run, 'analytic_df_hessian', refused)
    out = run_tadf(d, Store(tmp_path, d), stages=('s0', 'hess'),
                   max_memory=4000)
    assert np.array_equal(out['s0']['coords_bohr'], run.mol_in.atom_coords())
    assert np.array_equal(out['hess']['omega'], np.ones(6))
    with pytest.raises(stages.StageRefused, match='run s0 first'):
        run_tadf(d, Store(tmp_path / 'empty', d), stages=('hess',),
                 max_memory=4000)


# ----------------------------------------------------------------- cli
def test_the_command_line_needs_every_policy(tmp_path, capsys):
    full = argv(out=tmp_path)
    for flag in ('--nudge', '--refreeze', '--hessian', '--temperature',
                 '--span', '--manifold-gaps'):
        k = full.index(flag)
        with pytest.raises(SystemExit):
            cli.parser().parse_args(full[:k] + full[k + 2:])
    assert '--no-far-start' in REQUIRED


def test_the_command_line_dry_run(tmp_path, capsys):
    assert cli.main(argv('--dry-run', '--threads', '2', '--max-memory',
                         '2000', out=tmp_path)) == 0
    said = capsys.readouterr().out
    assert 'formaldehyde: 4 atoms' in said and 'stored: nothing' in said
    assert (tmp_path / 'formaldehyde' / 'gas').is_dir()


def test_the_default_stages_are_the_rate_model_s(tmp_path):
    """Without --stages a run takes every stage of its rate model: 'vib'
    under the UISC model, which its rate stage reads, and not under the TADF
    model, whose vib stage refuses."""
    for model, vib in (('tadf', False), ('uisc', True)):
        args = cli.parser().parse_args(argv('--rate-model', model,
                                            out=tmp_path))
        assert args.stages is None
        default = stages.stages_of(cli.declaration_of(args).rate_model)
        assert ('vib' in default) is vib
        assert default == tuple(s for s in stages.STAGE_ORDER
                                if vib or s != 'vib')


# ------------------------------------------------- ADC(2) through the table
def adc2_at(atom, where):
    """`StateMinimum` of ADC(2) (folded, DF) on RHF/6-31G at `atom`: E_0 the
    MP2 energy there, the two lowest singlets and triplets."""
    mol = gto.M(atom=atom, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol = 1e-11
    mf.kernel()
    mp2 = mf.MP2().run()
    omega = {spin: np.asarray(solve_ee_adc(
        mf, level='adc2', nroots=3, spin=spin, fold=True, frozen=1,
        auxbasis='cc-pvdz-ri', conv_tol=1e-9)[0])
        for spin in ('singlet', 'triplet')}
    rng = np.random.default_rng(len(where))
    return from_adc2(where, mol.atom_coords(), mp2.e_tot, omega['singlet'],
                     omega['triplet'],
                     socme=1e-6 * np.abs(rng.standard_normal((4, 3))),
                     provenance={'socme': 'a test table, not ADC(2)'})


@pytest.fixture(scope='module')
def adc2():
    mol0 = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol0).run(conv_tol=1e-11)
    omega, modes, masses, _ = normal_modes(None, mol0,
                                           hess=mf.Hessian().kernel())
    return (mol0, GroundModes(omega, modes, masses, 'analytic'),
            adc2_at(WATER, 'R0'), adc2_at(S1_AT, 'S1'), adc2_at(T1_AT, 'T1'))


def test_an_adc2_minimum_goes_through_the_rate_table(adc2):
    mol0, modes, vert, s1, t1 = adc2
    table = tadf_rate_table(modes, vert, s1, t1, mol0, 300.0, 0.05,
                            'stacked')
    assert table['method'] == 'adc2'
    assert table['dE_ST_adiabatic_eV'] == pytest.approx(
        (s1.energy['S1'] - t1.energy['T1']) * HARTREE_TO_EV, rel=1e-14)
    assert table['S1_adiabatic_eV'] == pytest.approx(
        (s1.energy['S1'] - vert.e0) * HARTREE_TO_EV, rel=1e-14)
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        ch = table['channels'][name]
        assert ch['dv_dq_rms_cm'] is None, 'Condon only: no derivative'
        # the test geometries are not minima (lambda_T comes out negative):
        # the displacement model is refused by its guard, by name, not run
        fc = ch['fc_lambda']
        assert fc['status'] == 'refused' and ch['A3_fc_condon'] is None
        assert np.isfinite(fc['lambda_sum_rule_eV'])
    # what ADC(2) left out is refused by name, not read as zero
    assert table['k_r'] is None and table['k_ic'] is None
    assert table['pl'] is None
    assert set(table['channels_refused']['S1']) == {'soc_dv_cart', 'dipole',
                                                    'd_ground'}


def test_two_methods_are_not_mixed_in_one_table(adc2):
    mol0, modes, vert, s1, t1 = adc2
    gw = StateMinimum(**{**t1.to_record(), 'method': 'gw-bse'})
    with pytest.raises(RateRefused, match='one method'):
        tadf_rate_table(modes, vert, s1, gw, mol0, 300.0, 0.05, 'stacked')
    bare = replace(s1, socme=None)
    with pytest.raises(RateRefused, match='spin-orbit'):
        tadf_rate_table(modes, vert, bare, t1, mol0, 300.0, 0.05,
                        'stacked')


def test_a_gw_bse_record_reads_into_the_same_record():
    props = {'coords_bohr': np.zeros((3, 3)), 'e0': -1.0,
             'energy': {'S1': -0.8, 'T1': -0.85}, 'omega': {'S1': 0.2,
                                                           'T1': 0.15},
             'emitter': 'S1', 'reservoir': 'T1', 'f_osc': np.ones(2),
             'dipole': np.ones((2, 3)), 'socme': np.ones((3, 2)),
             'd_ground': np.zeros((3, 3)), 'soc_dv_cart': np.zeros((3, 3)),
             'soc_dv_paths': {}}
    m = from_gw_bse_record({'props': props, 'solvation': 'gas'}, 'S1')
    assert m.method == 'gw-bse' and m.where == 'S1'
    assert type(m) is StateMinimum
