"""One rate table: both rate models call `rates.rate_table`.

The rate ladder (Marcus, MLJ, every-mode Franck-Condon, Condon plus
Herzberg-Teller, k_r, k_IC, k_nr(T1), the T2 channel, the kinetics) is
written once. Both rate models (`rate_models.tadf_rate_table`,
`uisc.uisc_rate_table`) hand their two minima to the library, which reads ISC
at the S1 minimum and RISC at the T1 minimum.

Synthetic inputs, no SCF: water's three atoms at three geometries, five
modes drawn, of which the four vibrations are passed (the imaginary one is
refused now, and was dropped before with nothing of it in any
number), random spin-orbit tables and derivatives. The frozen
numbers are what the two former script drivers printed or wrote, bit for bit
(`float.hex`), with every crossing out of a triplet re-recorded once at a
third of its rate (the sublevel average); the TADF table reproduces the TADF
driver's on both inputs, and the UISC table the UISC driver's on the inputs
where its one-geometry reading coincides with the rule (the same table,
derivative and lambda at both minima), with the per-mode derivative its vib
record held. Two runs are not bitwise: every frozen
number holds to PIN_SAME_ULPS ulp of itself under the recording BLAS and to
PIN_CROSS_BLAS_BAR of itself, relative, under another (`same_numbers`), where
the random modes' QR and the Franck-Condon sums round differently (a cluster's
MKL: 28 of 160 numbers moved, k_ic most, 1.2e-14 relative).

The frozen numbers predate the Franck-Condon and decay guards
(`rates.fc_lambda_guard`), which refuse this synthetic displacement (its sum
rule is a few percent of lambda, and no minimum carries S0's energy): the
ladder is compared with the guards passing (`guards_pass`), and
`test_the_guards_refuse_the_synthetic_displacement` holds the refusal.
"""
import json
import pathlib

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (AMU_TO_ME, BOHR_TO_ANGSTROM, HARTREE_TO_CM,
                                         HARTREE_TO_EV, PIN_CROSS_BLAS_BAR,
                                         PIN_SAME_ULPS)
from src.photophysics import rate_models, uisc
from src.photophysics.declaration import UiscModel
from src.photophysics.minimum import GroundModes, from_gw_bse_record
from src.photophysics import rates
from src.properties.vibronic import huang_rhys_from_displacement
from tests.pins import RECORDING_BLAS, blas_identity

WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'
#: Hartree; dyadic so that every energy difference below is exact
OMEGA = np.array([-0.001, 0.004, 0.0075, 0.012, 0.018])
#: the modes the tables are handed: the vibrations
VIBRATIONS = OMEGA > 0
E0, E_S1, E_T1 = -76.25, -76.0, -76.0078125
LAMBDA_T, LAMBDA_S, T2_GAP = 0.015625, 0.01171875, 0.0078125
TEMPERATURE = 300.0
LAMBDA_ENV = 0.05 / HARTREE_TO_EV
#: The former drivers' numbers (float.hex strings), the crossings out of a
#: triplet re-recorded at a third (sublevel average): the TADF table on
#: `synthetic()` and on the uisc-shaped inputs, and the UISC table on the
#: uisc-shaped inputs (`dv` is the per-mode derivative its vib record held).
FROZEN = json.loads((pathlib.Path(__file__).parent / 'data'
                     / 'one_rate_table_frozen.json').read_text())


def through_xyz(mol, coords):
    """`coords` as an xyz checkpoint reads them back: 10 decimals of Angstrom."""
    text = '\n'.join(f'{mol.atom_pure_symbol(ia):2s} {c[0]:16.10f} '
                     f'{c[1]:16.10f} {c[2]:16.10f}'
                     for ia, c in enumerate(coords * BOHR_TO_ANGSTROM))
    out = mol.copy()
    out.set_geom_(text, unit='Angstrom')
    out.build(False, False)
    return out.atom_coords()


def synthetic(lambda_s=LAMBDA_S, same_tables=False):
    """Every input of both tables: the S0 modes, three geometries, and each
    minimum's energies, spin-orbit table, its element's Cartesian
    derivative, the S1 -> S0 coupling and the emission. `same_tables` is
    the UISC driver's reading: one table and derivative at both minima, and the
    geometries as its xyz checkpoints hold them."""
    rng = np.random.default_rng(20261016)
    mol0 = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    x0 = mol0.atom_coords()
    modes = np.linalg.qr(rng.standard_normal((9, len(OMEGA))))[0]
    modes = modes[:, VIBRATIONS]
    masses = np.array([15.999, 1.008, 1.008]) * AMU_TO_ME
    theta = 0.05
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    x_s1 = x0 + 0.03 * rng.standard_normal(x0.shape)
    x_t1 = (x0 + 0.03 * rng.standard_normal(x0.shape)) @ rot + 0.1
    socme = {k: 1e-6 * rng.standard_normal((4, 4)) for k in ('s1', 't1')}
    dv = {k: 1e-6 * rng.standard_normal((3, 3, 3)) for k in ('s1', 't1')}
    if same_tables:
        socme['s1'], dv['s1'] = socme['t1'], dv['t1']
        mol0.set_geom_(through_xyz(mol0, x0), unit='Bohr')
        x_s1, x_t1 = through_xyz(mol0, x_s1), through_xyz(mol0, x_t1)
    return {
        'mol0': mol0, 'x_s1': x_s1, 'x_t1': x_t1, 'modes': modes,
        'masses': masses, 'omega': OMEGA[VIBRATIONS], 'socme': socme,
        'dv': dv,
        'paths': {'direct': 0.5 * dv['s1']},
        'd_ground': 0.1 * rng.standard_normal((3, 3)),
        'dipole': rng.standard_normal((3, 3)), 'f_osc': [0.3, 0.1, 0.05],
        'energy_s1': {'S1': E_S1, 'T1': E_T1 + LAMBDA_T},
        'energy_t1': {'S1': E_S1 + lambda_s, 'T1': E_T1,
                      'T2': E_T1 + T2_GAP},
        'emission': 0.125}


def minima(data):
    """(modes, R0, S1 minimum, T1 minimum) of `data` as the stages' records
    read back."""
    mol0 = data['mol0']
    modes = GroundModes(data['omega'], data['modes'], data['masses'], 'df')

    def props(coords, energy, key):
        return {'coords_bohr': np.asarray(coords), 'e0': E0,
                'energy': energy,
                'omega': {k: e - E0 for k, e in energy.items()} | (
                    {'S1': data['emission']} if key == 's1' else {}),
                'emitter': 'S1', 'reservoir': 'T1',
                'f_osc': data['f_osc'], 'dipole': data['dipole'],
                'socme': data['socme'].get(key, data['socme']['t1']),
                'd_ground': data['d_ground'],
                'soc_dv_cart': data['dv'].get(key, data['dv']['t1']),
                'soc_dv_paths': dict(data['paths'])}

    return (modes,) + tuple(
        from_gw_bse_record({'props': props(coords, energy, key)}, where)
        for where, coords, energy, key in (
            ('R0', mol0.atom_coords(), {'S1': -76.1, 'T1': -76.15}, 'vert'),
            ('S1', data['x_s1'], data['energy_s1'], 's1'),
            ('T1', data['x_t1'], data['energy_t1'], 't1')))


def tadf_table(data):
    return rate_models.tadf_rate_table(*minima(data), data['mol0'],
                                       TEMPERATURE, 0.05, 'stacked')


#: The UISC model of the frozen numbers (the A0 rows are not frozen).
MODEL = UiscModel(n_modes=4, dq=0.1, s_block=2, t_block=2,
                  lambda_samanta=(0.10, 0.20))


def uisc_table(data):
    """The UISC table on `data`: the per-mode derivative the frozen run's vib
    record held, at both minima, and `data`'s spin-orbit tables."""
    dv = np.array(unhex(FROZEN['uisc']['dv']))
    vib = {'modes': np.arange(VIBRATIONS.sum()), 'omega': OMEGA[VIBRATIONS],
           'dv_dq_isc': dv, 'dv_dq_risc': dv}
    modes, vert, s1, t1 = minima(data)
    return uisc.uisc_rate_table(modes, vert, s1, t1, vib, data['mol0'],
                                TEMPERATURE, 0.05, MODEL, 'stacked')


@pytest.fixture
def guards_pass(monkeypatch):
    """The readings the frozen numbers were taken on: every Franck-Condon and
    decay guard of `rates.rate_table` passes, and the Huang-Rhys factors are
    the Cartesian displacement's, as the former drivers read them (both
    models read them through `rate_models`)."""
    monkeypatch.setattr(rates, 'fc_lambda_guard', lambda lam_rule, lam: {
        'status': 'computed', 'reason': 'guard passed by the test'})
    monkeypatch.setattr(
        rate_models, 'huang_rhys_curvilinear',
        lambda ref, a, b, *rest: huang_rhys_from_displacement(a, b, *rest))


def test_the_guards_refuse_the_synthetic_displacement():
    got = tadf_table(synthetic())
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        assert got['channels'][name]['fc_lambda']['status'] == 'refused'
        assert got['channels'][name]['A3_fc_condon'] is None
    assert got['k_nr_t1'] is None and got['pl'] is None


def unhex(value):
    if isinstance(value, dict):
        return {k: unhex(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unhex(v) for v in value]
    if isinstance(value, str) and value.startswith(('0x', '-0x')):
        return float.fromhex(value)
    return value


def same_numbers(got, want, where=''):
    """Every number of `want` in `got`: to PIN_SAME_ULPS ulp of it under the
    recording BLAS, to PIN_CROSS_BLAS_BAR of it, relative, under another (the
    rates span 1e-28 to 1e6 s^-1, so the bar scales with each)."""
    if isinstance(want, dict):
        for k, v in want.items():
            same_numbers(got[k], v, f'{where}.{k}')
    elif isinstance(want, list):
        assert len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want)):
            same_numbers(g, w, f'{where}[{i}]')
    elif isinstance(want, float):
        bar = (PIN_SAME_ULPS * np.spacing(abs(want))
               if RECORDING_BLAS == blas_identity()
               else PIN_CROSS_BLAS_BAR * abs(want))
        assert abs(float(got) - want) <= bar, (where, float(got).hex(),
                                               want.hex())
    else:
        assert got == want, where


def test_the_tadf_table_is_the_frozen_one(guards_pass):
    for case, kw in (('general', {}),
                     ('uisc_shaped', dict(lambda_s=LAMBDA_T,
                                          same_tables=True))):
        got = tadf_table(synthetic(**kw))
        want = unhex(FROZEN[f'tadf_{case}'])
        # the frozen run's synthetic records named no solvation; a stage
        # record always does, and a StateMinimum carries it
        assert want.pop('relaxed_solvation') is None
        assert got['relaxed_solvation'] == 'gas'
        same_numbers(got, want, case)
        assert got['pl_omits'] == []


def test_the_uisc_table_is_the_frozen_one(guards_pass):
    data = synthetic(lambda_s=LAMBDA_T, same_tables=True)
    got = uisc_table(data)
    want = unhex(FROZEN['uisc'])
    for name, key in (('ISC S1->T1', 'ISC'), ('RISC T1->S1', 'RISC')):
        ch, old = got['channels'][name], want[key]
        same_numbers([ch['A1_marcus'], ch['A2_mlj'], ch['A3_fc_condon'],
                   ch['A4_fc_condon_ht'], ch['A4_ht_part']],
                  [old['A1'], old['A2'], old['A3'], old['A4'], old['HT']],
                  name)
        same_numbers(ch['marcus_mlj'], old['lams'], name)
    same_numbers([got['k_ic'], got['k_nr_t1'], got['k_r']],
              [want['k_ic'], want['k_nr_t'], want['k_r']])
    same_numbers(got['pl'], want['pl'])
    # the two models agree wherever their readings coincide
    tadf = unhex(FROZEN['tadf_uisc_shaped'])
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        for k in ('A1_marcus', 'A2_mlj', 'A3_fc_condon', 'marcus_mlj'):
            same_numbers(got['channels'][name][k], tadf['channels'][name][k])


def test_uisc_reads_isc_at_the_s1_minimum():
    """Two different tables at the two minima: ISC takes the S1 minimum's
    <S1|H_SO|T1>, RISC and T1 -> S0 the T1 minimum's."""
    data = synthetic()
    got = uisc_table(data)
    cm = HARTREE_TO_CM
    assert got['channels']['ISC S1->T1']['v_condon_cm'] == (
        abs(data['socme']['s1'][1, 0]) * cm)
    assert got['channels']['RISC T1->S1']['v_condon_cm'] == (
        abs(data['socme']['t1'][1, 0]) * cm)
    assert got['v_s0t1_cm'] == abs(data['socme']['t1'][0, 0]) * cm
    assert got['lambda_S_eV'] == pytest.approx(LAMBDA_S * HARTREE_TO_EV)
    assert got['lambda_T_eV'] == pytest.approx(LAMBDA_T * HARTREE_TO_EV)
    assert got['rate_model'] == 'uisc'


def test_both_models_take_the_one_table(monkeypatch):
    """Neither model keeps a rate ladder of its own; both call
    `rates.rate_table`."""
    for module in (rate_models, uisc):
        assert not hasattr(module, 'marcus_rows')
    calls = []
    real = rates.rate_table
    monkeypatch.setattr(rates, 'rate_table',
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    data = synthetic()
    tadf_table(data)
    uisc_table(data)
    assert len(calls) == 2


def test_a_crossing_out_of_a_triplet_takes_a_third(monkeypatch, guards_pass):
    """RISC from T1 and T2 and T1 -> S0 average over the three thermalized
    initial sublevels, |V|^2 / 3; ISC sums over the final ones and keeps it."""
    data = synthetic()
    averaged = tadf_table(data)
    monkeypatch.setattr(rates, 'TRIPLET_SUBLEVELS', 1)
    summed = tadf_table(data)
    rows = ('A1_marcus', 'A2_mlj', 'A3_fc_condon', 'A4_fc_condon_ht',
            'A4_ht_part')
    for k in rows:
        same_numbers(averaged['channels']['ISC S1->T1'][k],
                  summed['channels']['ISC S1->T1'][k], k)
        assert averaged['channels']['RISC T1->S1'][k] == pytest.approx(
            summed['channels']['RISC T1->S1'][k] / 3, rel=1e-13), k
    for k in ('k_risc_t2', 'k_nr_t1'):
        assert averaged[k] == pytest.approx(summed[k] / 3, rel=1e-13), k
    same_numbers(averaged['k_isc_t2'], summed['k_isc_t2'])
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        assert (averaged['channels'][name]['v_condon_cm']
                == summed['channels'][name]['v_condon_cm'])


def test_both_models_read_curvilinear_huang_rhys_factors(monkeypatch):
    """The S1 <-> T1, S0 <-> S1 and S0 <-> T1 factors of both tables are the
    displacements in valence internals linearized at R0, where the S0 modes
    were taken (`huang_rhys_curvilinear`)."""
    calls = []
    real = rate_models.huang_rhys_curvilinear
    monkeypatch.setattr(rate_models, 'huang_rhys_curvilinear',
                        lambda *a: calls.append(a) or real(*a))
    data = synthetic()
    tadf_table(data)
    uisc_table(data)
    assert len(calls) == 6
    x0 = data['mol0'].atom_coords()
    for ref, *_ in calls:
        assert np.allclose(ref.atom_coords(), x0, atol=1e-12)
