"""The Marcus and MLJ rows refuse a displacement model that contradicts lambda.

Marcus-Levich-Jortner treats the Huang-Rhys sum rule sum_k S_k omega_k of the
displaced modes quantum mechanically and the remainder lambda - S omega_eff
classically, lambda being the accepting state's reorganization off the
energies. On a torsion the Cartesian projection inflates S until the sum rule
exceeds lambda; a clamped negative remainder would print a finite rate. The
rows are refused instead, with both lambdas in the record, while every other
row of the table is judged on its own (the Franck-Condon rows by
`rates.fc_lambda_guard`). Both rate models
(`rate_models.tadf_rate_table`, `uisc.uisc_rate_table`) read the one
`rates.rate_table` and its `marcus_rows`.

Synthetic data, no SCF: three modes, the S1-T1 displacement given as Huang-Rhys
factors, the four energies chosen so that lambda_T (ISC) is 0.025 Ha and
lambda_S (RISC) 0.010 Ha.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (HARTREE_TO_CM, HARTREE_TO_EV,
                                         HARTREE_TO_MEV, RATE_LAMBDA_REMAINDER_MIN_MEV,
                                         RATE_MLJ_N_MAX)
from src.photophysics import rate_models
from src.photophysics.minimum import from_gw_bse_record
from src.photophysics import rates
from tests.photophysics_cases import gw_bse_props, unit_modes

WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'
OMEGA = np.array([0.01, 0.02, 0.03])
#: Hartree: lambda_T = E_T1(R_S1) - E_T1(R_T1), lambda_S = E_S1(R_T1) - E_S1(R_S1)
LAMBDA_T, LAMBDA_S = 0.025, 0.010
ISC, RISC = 'ISC S1->T1', 'RISC T1->S1'


#: K: the table's temperature.
TEMPERATURE = 300.0


def props(mol, energy):
    """A minimum's record, S1 and T1 at `energy` and T2 above them."""
    energy = dict(energy, T2=energy['T1'] + 0.05)
    return gw_bse_props(mol, energy, {k: e + 1.0 for k, e in energy.items()},
                        (0.01 * np.ones((3, 3))).tolist())


def table_on(monkeypatch, s_k):
    """The TADF table on the synthetic minima with Huang-Rhys factors
    `s_k`."""
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    n = len(OMEGA)
    modes = unit_modes(mol, OMEGA)
    e_s1, e_t1 = -0.900, -0.905
    minima = [from_gw_bse_record({'props': props(mol, energy)}, where)
              for where, energy in (
                  ('R0', {'S1': -0.8, 'T1': -0.85}),
                  ('S1', {'S1': e_s1, 'T1': e_t1 + LAMBDA_T}),
                  ('T1', {'S1': e_s1 + LAMBDA_S, 'T1': e_t1}))]
    zeros = np.zeros(n)
    for name, fn in {
            'align_to': lambda mol0, m, return_rotation=False: (m, np.eye(3)),
            'huang_rhys_curvilinear': lambda *a: (np.asarray(s_k), None),
            'mode_derivatives': lambda *a, **k: (None, np.zeros((n, 3))),
            'project_coupling': lambda *a: zeros}.items():
        monkeypatch.setattr(rate_models, name, fn)
    for name, fn in {
            'spin_vibronic_rate': lambda *a: (1.0, 1.0, 0.0),
            'radiative_rate': lambda *a: 1.0,
            'internal_conversion_rate': lambda *a: (0.0, zeros),
            'photoluminescence': lambda *a, **k: dict.fromkeys(
                ('k_prompt', 'phi_prompt', 'k_delayed', 'phi_delayed',
                 'phi_total'), 0.0)}.items():
        monkeypatch.setattr(rates, name, fn)
    return rate_models.tadf_rate_table(modes, *minima, mol, TEMPERATURE,
                                       0.05, 'stacked')


def sum_rule(s_k):
    return float((np.asarray(s_k) * OMEGA).sum())


def test_both_crossings_refused_when_the_sum_rule_exceeds_lambda(
        monkeypatch):
    s_k = [1.0, 1.0, 1.0]                     # sum rule 0.06 Ha, above both
    out = table_on(monkeypatch, s_k)
    for name, lam in ((ISC, LAMBDA_T), (RISC, LAMBDA_S)):
        ch = out['channels'][name]
        assert ch['A1_marcus'] is None and ch['A2_mlj'] is None
        rec = ch['marcus_mlj']
        assert rec['status'] == 'refused'
        assert rec['lambda_energy_eV'] == pytest.approx(lam * HARTREE_TO_EV)
        assert rec['lambda_sum_rule_eV'] == pytest.approx(
            sum_rule(s_k) * HARTREE_TO_EV)
        # the Franck-Condon rows are judged by their own guard: a sum rule
        # beyond RATE_FC_LAMBDA_RATIO_MAX times lambda refuses them too
        assert ch['A3_fc_condon'] is None
        assert ch['fc_lambda']['status'] == 'refused'
    rate_models.print_rate_table('synthetic', dict(out, emission_eV=0.0,
                                                   f_emission=0.0))


def test_a_remainder_inside_the_tolerance_is_refused(monkeypatch):
    """RISC keeps a remainder of half the tolerance and is refused; ISC keeps
    a wide one and is computed off it, unclamped."""
    half = 0.5 * RATE_LAMBDA_REMAINDER_MIN_MEV / HARTREE_TO_MEV
    s_k = np.array([1.0, 1.0, 1.0]) * (LAMBDA_S - half) / OMEGA.sum()
    out = table_on(monkeypatch, s_k)
    risc, isc = out['channels'][RISC], out['channels'][ISC]
    assert risc['marcus_mlj']['status'] == 'refused'
    assert risc['A2_mlj'] is None
    assert isc['marcus_mlj']['status'] == 'computed'
    assert isc['marcus_mlj']['lambda_classical_eV'] == pytest.approx(
        (LAMBDA_T - sum_rule(s_k)) * HARTREE_TO_EV)
    s_sum = float(s_k.sum())
    expected = rates.marcus_levich_jortner_rate(
        isc['v_condon_cm'] / HARTREE_TO_CM, -0.005,
        LAMBDA_T - sum_rule(s_k), s_sum, sum_rule(s_k) / s_sum,
        TEMPERATURE, n_max=RATE_MLJ_N_MAX)
    assert isc['A2_mlj'] == pytest.approx(expected, rel=1e-12)
    assert isc['A1_marcus'] > 0


def test_the_library_rows_refuse_alike():
    """`rates.marcus_rows`, which both rate models read."""
    v, de, lam = 1e-5, -0.005, LAMBDA_S
    rows = rates.marcus_rows(v, de, lam, 1.0, 2.0 * lam, 300.0)
    lams = rows['marcus_mlj']
    assert rows['A1_marcus'] is None and rows['A2_mlj'] is None
    assert lams['status'] == 'refused'
    assert lams['lambda_sum_rule_eV'] == pytest.approx(2 * lam * HARTREE_TO_EV)
    rows = rates.marcus_rows(v, de, lam, 1.0, 0.5 * lam, 300.0)
    assert rows['marcus_mlj']['status'] == 'computed'
    assert rows['A1_marcus'] > 0 and rows['A2_mlj'] > 0
