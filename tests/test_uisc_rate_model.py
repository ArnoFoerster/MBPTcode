"""The UISC rate model reads ONE rate table, the one the TADF model reads.

The rate ladder (Marcus, Marcus-Levich-Jortner, every-mode Franck-Condon,
Condon plus Herzberg-Teller, k_r, k_IC, k_nr(T1), the T2 channel, the
kinetics) is written once. Both rate models hand their two minima to
`properties.rates.rate_table`, which reads intersystem crossing at the S1
minimum and reverse crossing at the T1 minimum. The claims kept here:

- (`test_one_rate_table` holds that both rate models call `rates.rate_table`
  and which minimum's spin-orbit table each crossing reads;)
- the derivative of each crossing is the vib record's (`dv_dq_isc` at S1,
  `dv_dq_risc` at T1) and replaces the analytic one a minimum may carry;
  without a vib record both crossings are Condon only;
- the Samanta A0 rows are the Marcus rate with S = 0 at each reorganization
  energy of the protocol (Samanta, Kim, Coropceanu and Bredas, J. Am. Chem.
  Soc. 139, 4042 (2017)), with the spin-orbit element at the crossing's own
  minimum and the downhill sign of the gap.

Synthetic inputs, no SCF: water's three atoms at three geometries, four
vibrational modes (an imaginary one is refused), random spin-orbit
tables and derivatives.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import AMU_TO_ME, HARTREE_TO_CM, HARTREE_TO_EV
from src.Base.geometry.geometry import molecule_at
from src.photophysics import uisc
from src.photophysics.declaration import UiscModel
from src.photophysics.minimum import GroundModes, StateMinimum
from src.photophysics import rates
from src.properties import franck_condon
from src.properties.curvilinear import huang_rhys_curvilinear
from src.properties.vibronic import align_to

WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'
#: Hartree; dyadic, so every energy difference below is exact
OMEGA = np.array([0.004, 0.0075, 0.012, 0.018])
E0, E_S1, E_T1 = -76.25, -76.0, -76.0078125
#: lambda_T and lambda_S near the displacement's sum rule (6.1e-4 Ha), so the
#: Franck-Condon rows pass their guard (`rates.fc_lambda_guard`)
LAMBDA_T, LAMBDA_S, T2_GAP = 0.0009765625, 0.0010986328125, 0.0078125
TEMPERATURE = 300.0
LAMBDA_ENV_EV = 0.05
MODEL = UiscModel(n_modes=3, dq=0.1, s_block=2, t_block=2,
                  lambda_samanta=(0.10, 0.20))


def synthetic():
    """(template, modes, vert, s1, t1, tables): the S0 modes and the three
    `StateMinimum` records, each minimum with its own spin-orbit table and its
    own analytic derivative (which the UISC table must not read)."""
    rng = np.random.default_rng(20261016)
    template = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    x0 = template.atom_coords()
    # five modes drawn, as when one of them was imaginary: the first is dropped
    basis_modes = np.linalg.qr(rng.standard_normal((9, len(OMEGA) + 1)))[0][:, 1:]
    masses = np.array([15.999, 1.008, 1.008]) * AMU_TO_ME
    modes = GroundModes(OMEGA, basis_modes, masses, 'df')
    x_s1 = x0 + 0.03 * rng.standard_normal(x0.shape)
    x_t1 = x0 + 0.03 * rng.standard_normal(x0.shape) + 0.1
    socme = {k: 1e-6 * rng.standard_normal((4, 4)) for k in ('s1', 't1')}
    dv = {k: 1e-6 * rng.standard_normal((3, 3)) for k in ('s1', 't1')}
    d_ground = 0.1 * rng.standard_normal((3, 3))
    dipole = rng.standard_normal((3, 3))
    f_osc = np.array([0.3, 0.1, 0.05])

    def minimum(where, coords, energy, omega, table, dv_cart):
        return StateMinimum(
            method='gw-bse', where=where, coords_bohr=coords, e0=E0,
            energy=energy, omega=omega, emitter='S1', reservoir='T1',
            socme=table, soc_dv_cart=dv_cart, soc_dv_paths=None,
            dipole=dipole, f_osc=f_osc, d_ground=d_ground, solvation='gas')

    vert = minimum('R0', x0, {'S1': -76.1, 'T1': -76.15},
                   {'S1': 0.15, 'T1': 0.1}, socme['t1'], dv['t1'])
    s1 = minimum('S1', x_s1, {'S1': E_S1, 'T1': E_T1 + LAMBDA_T},
                 {'S1': 0.125, 'T1': 0.1}, socme['s1'], dv['s1'])
    t1 = minimum('T1', x_t1, {'S1': E_S1 + LAMBDA_S, 'T1': E_T1,
                              'T2': E_T1 + T2_GAP},
                 {'S1': 0.125, 'T1': 0.1}, socme['t1'], dv['t1'])
    return template, modes, vert, s1, t1, socme


def vib_record(seed=3):
    """A vib record on the three modes of largest index but the last:
    distinct derivatives for the two crossings."""
    rng = np.random.default_rng(seed)
    picked = np.array([2, 1, 0])
    return {'modes': picked, 'omega': OMEGA[picked],
            'dv_dq_isc': 1e-7 * (1 + rng.random(3)),
            'dv_dq_risc': 1e-6 * (1 + rng.random(3)),
            'v0_isc': 1e-6, 'v0_risc': 2e-6,
            'overlap_isc': np.ones(3), 'overlap_risc': np.ones(3)}


def table(vib=None):
    template, modes, vert, s1, t1, _ = synthetic()
    return uisc.uisc_rate_table(modes, vert, s1, t1, vib, template,
                                TEMPERATURE, LAMBDA_ENV_EV, MODEL, 'stacked')


def test_the_vib_derivative_replaces_the_analytic_one():
    """Each crossing's |dV/dq| is the vib record's; the minima's analytic
    derivative is not read; the Herzberg-Teller rate is the library's on
    exactly those numbers."""
    template, modes, vert, s1, t1, _ = synthetic()
    assert s1.soc_dv_cart is not None and t1.soc_dv_cart is not None
    vib = vib_record()
    got = table(vib)
    for name, key in (('ISC S1->T1', 'dv_dq_isc'),
                      ('RISC T1->S1', 'dv_dq_risc')):
        want = float(np.sqrt((vib[key] ** 2).mean()) * HARTREE_TO_CM)
        assert got['channels'][name]['dv_dq_rms_cm'] == pytest.approx(
            want, rel=1e-14)
        assert got['channels'][name]['dv_dq_rms_by_path_cm'] == {}
    assert got['channels']['ISC S1->T1']['dv_dq_rms_cm'] != \
        got['channels']['RISC T1->S1']['dv_dq_rms_cm']
    assert 'soc_dv_cart' not in got['channels_refused']['S1']
    # the rate on those numbers, from the library directly
    rho = franck_condon.fc_weighted_dos
    isc = got['channels']['ISC S1->T1']
    k_ht, _, _ = rates.spin_vibronic_rate(
        -(E_S1 - E_T1), isc['v_condon_cm'] / HARTREE_TO_CM,
        vib['dv_dq_isc'], vib['omega'], lambda e: rho(
            e, np.asarray(_s_isc(template, modes, s1, t1)), OMEGA,
            TEMPERATURE,
            lambda_classical=LAMBDA_ENV_EV / HARTREE_TO_EV), TEMPERATURE)
    assert isc['A4_fc_condon_ht'] == pytest.approx(k_ht, rel=1e-12)


def _s_isc(template, modes, s1, t1):
    mol0 = molecule_at(template, np.asarray(template.atom_coords()))
    a = align_to(mol0, molecule_at(template, s1.coords_bohr))
    b = align_to(mol0, molecule_at(template, t1.coords_bohr))
    return huang_rhys_curvilinear(mol0, a, b, modes.omega, modes.modes,
                                  modes.masses)[0]


def test_without_a_vib_record_both_crossings_are_condon_only():
    got = table(None)
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        assert got['channels'][name]['dv_dq_rms_cm'] is None
    assert 'Condon only' in got['channels_refused']['S1']['soc_dv_cart']


def test_a_vib_record_of_other_modes_is_refused():
    template, modes, vert, s1, t1, _ = synthetic()
    vib = vib_record()
    vib['omega'] = vib['omega'] * 1.01
    with pytest.raises(uisc.RateRefused):
        uisc.uisc_rate_table(modes, vert, s1, t1, vib, template, TEMPERATURE,
                             LAMBDA_ENV_EV, MODEL, 'stacked')


def test_samanta_rows_are_the_marcus_rate_with_s_zero():
    """A0: Marcus at each lambda of the protocol, |V| at the crossing's own
    minimum, Delta_E = E_final - E_initial (downhill is negative)."""
    _, _, _, _, _, socme = synthetic()
    got = table(vib_record())
    rows = got['samanta']
    assert set(rows) == {'ISC S1->T1', 'RISC T1->S1'}
    want = {'ISC S1->T1': (abs(socme['s1'][1, 0]), E_T1 - E_S1),
            'RISC T1->S1': (abs(socme['t1'][1, 0]), E_S1 - E_T1)}
    for name, (v, de) in want.items():
        row = rows[name]
        assert row['lambda_eV'] == list(MODEL.lambda_samanta)
        assert row['delta_E_eV'] == pytest.approx(de * HARTREE_TO_EV)
        for lam, k in zip(MODEL.lambda_samanta, row['k']):
            assert k == pytest.approx(
                rates.marcus_rate(v, de, lam / HARTREE_TO_EV, TEMPERATURE),
                rel=1e-12)
            assert k > 0.0


def test_the_table_prints(capsys):
    uisc.print_uisc_table('cbp', 'gas', table(vib_record()))
    out = capsys.readouterr().out
    assert 'A0  Samanta (lam=0.10)' in out and 'Samanta 0.88 calc' in out
