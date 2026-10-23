"""The rate table's Franck-Condon refusal and its thermal state manifold.

1. The Franck-Condon rows (A3, A4) share one density built from the S1-T1
   displacement. When its harmonic sum rule disagrees with a crossing's
   energy-based lambda by more than RATE_FC_LAMBDA_RATIO_MAX the rows are
   refused, with both lambdas recorded, and so is every kinetics on them.
2. RISC from T1 runs into every singlet within thermal reach at the T1 minimum,
   each with its own gap and coupling: a dark S2 a few k_B T above S1 that
   carries the spin-orbit coupling adds its own channel, at |V|^2 / 3.
3. With the same splittings and couplings at both minima and one density, the
   two thermalized reservoirs obey detailed balance:
   Z_S k_ISC,eff = TRIPLET_SUBLEVELS Z_T k_RISC,eff, Z the Boltzmann sums
   over each spin's states at their energies.
4. The decays to S0 (internal conversion, T1 -> S0) are refused alike against
   S0's own relaxation E_S0(R_min) - E_S0(R0), and refuse the kinetics; with
   no S0 energy at the minimum they are refused, not run unchecked.
5. A followed emitter that is not the lowest singlet at its own minimum stays
   the manifold's initial state.
6. The gap rule is declared (`ManifoldGaps`), with no default: a table with
   none computes no manifold; 'stacked' shifts the adiabatic gap by the
   vertical splittings at the initial minimum, 'adiabatic' puts every state at
   its own minimum's energy and leaves out one without a minimum.
7. The T2 channels stand on the S1 <-> T1 density and are refused with its
   Franck-Condon rows, and pl_manifold says whether its manifold was
   complete: under 'adiabatic' with S1 and T1 alone relaxed it is not.

Synthetic minima, no electronic structure.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN, HARTREE_TO_EV,
                                         RATE_FC_LAMBDA_RATIO_MAX, TRIPLET_SUBLEVELS)
from src.photophysics import rates
from src.properties import franck_condon

TEMPERATURE = 300.0
LAMBDA_ENV = 0.002
OMEGA = np.array([0.004, 0.008, 0.015])
#: Hartree: adiabatic S1 and T1 and the vertical splittings at both minima
E_S1, E_T1, E0 = -0.900, -0.904, -1.0
LAMBDA = 0.006


def minimum(where, split_s, split_t, socme, e_ground=None, emitter='S1'):
    """A minimum with S1/T1 at their adiabatic energies plus `LAMBDA` on the
    other state, and higher roots at the given splittings."""
    s1 = E_S1 + (LAMBDA if where == 'T1' else 0.0)
    t1 = E_T1 + (LAMBDA if where == 'S1' else 0.0)
    energy = {f'S{n + 1}': s1 + x for n, x in enumerate(split_s)}
    energy.update({f'T{m + 1}': t1 + x for m, x in enumerate(split_t)})
    return rates.Minimum(emitter=emitter, reservoir='T1', energy=energy,
                         socme=np.asarray(socme, float),
                         emission=s1 - E0, dipole=np.array([0.5, 0, 0]),
                         f_osc=0.1, e_ground=e_ground)


def huang_rhys(lam):
    """S_k on OMEGA whose sum rule is `lam`."""
    return lam / OMEGA.sum() * np.ones_like(OMEGA)


STACKED = rates.ManifoldGaps('stacked', None)


def table(s1, t1, lam_rule=LAMBDA, gaps=STACKED):
    return rates.rate_table(s1, t1, E0, OMEGA, huang_rhys(lam_rule),
                            TEMPERATURE, LAMBDA_ENV, manifold_gaps=gaps)


SOC = [[0, 0], [1e-6, 2e-6], [8e-6, 1e-6]]


def test_fc_rows_refuse_a_sum_rule_off_the_energies():
    s1 = minimum('S1', [0.0, 0.01], [0.0, 0.003], SOC)
    t1 = minimum('T1', [0.0, 0.01], [0.0, 0.003], SOC)
    ok = table(s1, t1)
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        ch = ok['channels'][name]
        assert ch['fc_lambda']['status'] == 'computed'
        assert ch['A4_fc_condon_ht'] > 0
    assert ok['pl'] is not None and ok['pl_manifold'] is not None
    bad = table(s1, t1, lam_rule=1.1 * RATE_FC_LAMBDA_RATIO_MAX * LAMBDA)
    for name in ('ISC S1->T1', 'RISC T1->S1'):
        ch = bad['channels'][name]
        fc = ch['fc_lambda']
        assert fc['status'] == 'refused'
        assert fc['lambda_energy_eV'] == pytest.approx(LAMBDA * HARTREE_TO_EV)
        assert fc['ratio'] == pytest.approx(1.1 * RATE_FC_LAMBDA_RATIO_MAX)
        assert ch['A3_fc_condon'] is None and ch['A4_fc_condon_ht'] is None
    assert bad['pl'] is None and bad['pl_manifold'] is None
    assert bad['manifold']['RISC']['status'] == 'refused'


def test_risc_runs_through_a_dark_singlet_within_reach():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    split = 3 * kt
    soc = SOC + [[0, 0]]
    # S3 at 0.02 Ha is beyond the window and closes it
    s1 = minimum('S1', [0.0, 0.02, 0.03], [0.0, 0.02], soc)
    t1 = minimum('T1', [0.0, split, 0.02], [0.0, 0.02], soc)
    out = table(s1, t1)
    risc = out['manifold']['RISC']
    pairs = {p['pair']: p for p in risc['pairs']}
    assert set(pairs) == {'T1->S1', 'T1->S2'}
    assert pairs['T1->S1']['k'] == out['channels']['RISC T1->S1'][
        'A4_fc_condon_ht']
    rho = (lambda e: franck_condon.fc_weighted_dos(e, huang_rhys(LAMBDA), OMEGA,
                                           TEMPERATURE,
                                           lambda_classical=LAMBDA_ENV))
    de = (E_S1 - E_T1) + split
    want = rates.spin_vibronic_rate(de, 8e-6 / np.sqrt(TRIPLET_SUBLEVELS),
                                    [], [], rho, TEMPERATURE)[0]
    assert pairs['T1->S2']['dE_eV'] == pytest.approx(de * HARTREE_TO_EV)
    assert pairs['T1->S2']['k'] == pytest.approx(want, rel=1e-12)
    # the dark state's 8x coupling outweighs its 3 k_B T higher gap
    assert pairs['T1->S2']['k'] > pairs['T1->S1']['k']
    assert risc['k_eff'] == pytest.approx(
        pairs['T1->S1']['k'] + pairs['T1->S2']['k'], rel=1e-12)
    assert risc['complete'] == {'initial': True, 'final': True}


def test_the_two_reservoirs_obey_detailed_balance():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    split_s, split_t = [0.0, 2 * kt], [0.0, 1 * kt, 4 * kt]
    soc = [[0, 0, 0], [1e-6, 3e-6, 2e-6], [5e-6, 1e-6, 4e-6]]
    s1 = minimum('S1', split_s, split_t, soc)
    t1 = minimum('T1', split_s, split_t, soc)
    man = table(s1, t1)['manifold']
    # Boltzmann sums with energies from the adiabatic T1
    z_s = sum(np.exp(-(E_S1 - E_T1 + x) / kt) for x in split_s)
    z_t = sum(np.exp(-x / kt) for x in split_t)
    assert len(man['ISC']['pairs']) == len(man['RISC']['pairs']) == 6
    assert z_s * man['ISC']['k_eff'] == pytest.approx(
        TRIPLET_SUBLEVELS * z_t * man['RISC']['k_eff'], rel=1e-6)


def test_a_truncated_manifold_is_marked_incomplete():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    # the highest computed singlet at the T1 minimum is inside the window
    s1 = minimum('S1', [0.0, 0.02], [0.0, 0.02], SOC)
    t1 = minimum('T1', [0.0, 2 * kt], [0.0, 0.02], SOC)
    risc = table(s1, t1)['manifold']['RISC']
    assert risc['complete']['final'] is False
    assert risc['k_eff'] > 0


def decays(e_ground):
    """The table with both decays to S0 asked for, S0 at `e_ground` at both
    minima and every displacement the LAMBDA spectrum."""
    soc = [[3e-6, 1e-6], [1e-6, 2e-6], [8e-6, 1e-6]]
    s1 = minimum('S1', [0.0, 0.02], [0.0, 0.02], soc, e_ground=e_ground)
    t1 = minimum('T1', [0.0, 0.02], [0.0, 0.02], soc, e_ground=e_ground)
    s = huang_rhys(LAMBDA)
    return rates.rate_table(s1, t1, E0, OMEGA, s, TEMPERATURE, LAMBDA_ENV,
                            s_ic=s, d_ic=np.full(len(OMEGA), 0.01), s_t=s,
                            manifold_gaps=STACKED)


def test_decays_to_s0_refuse_a_sum_rule_off_the_ground_state():
    ok = decays(E0 + LAMBDA)
    for guard in ('ic_fc_lambda', 't1_s0_fc_lambda'):
        assert ok[guard]['status'] == 'computed'
        assert ok[guard]['ratio'] == pytest.approx(1.0)
    assert ok['k_ic'] > 0 and ok['k_nr_t1'] > 0
    assert ok['pl'] is not None and ok['pl_refused_by'] == []
    bad = decays(E0 + LAMBDA / (1.1 * RATE_FC_LAMBDA_RATIO_MAX))
    for guard in ('ic_fc_lambda', 't1_s0_fc_lambda'):
        assert bad[guard]['status'] == 'refused'
        assert bad[guard]['lambda_energy_eV'] == pytest.approx(
            LAMBDA / (1.1 * RATE_FC_LAMBDA_RATIO_MAX) * HARTREE_TO_EV)
    assert bad['k_ic'] is None and bad['k_nr_t1'] is None
    assert bad['manifold']['k_nr_t_eff'] is None
    assert bad['pl'] is None and bad['pl_manifold'] is None
    assert bad['pl_refused_by'] == ['k_ic', 'k_nr_t1']
    # the S1/T1 crossings do not depend on it
    assert bad['channels']['ISC S1->T1']['A4_fc_condon_ht'] == ok[
        'channels']['ISC S1->T1']['A4_fc_condon_ht']


def test_decays_to_s0_without_its_energy_are_refused():
    out = decays(None)
    for guard in ('ic_fc_lambda', 't1_s0_fc_lambda'):
        assert out[guard]['status'] == 'refused'
        assert out[guard]['lambda_energy_eV'] is None
    assert out['k_ic'] is None and out['k_nr_t1'] is None
    assert out['pl'] is None


def test_zero_lambdas_agree_and_a_negative_one_is_refused():
    assert rates.fc_lambda_guard(0.0, 0.0)['status'] == 'computed'
    assert rates.fc_lambda_guard(0.0, -0.001)['status'] == 'refused'
    assert rates.fc_lambda_guard(0.001, -0.001)['status'] == 'refused'


def test_a_followed_emitter_above_a_lower_singlet_stays_initial():
    # S2 is the emitter at the S1 minimum, 0.3 eV above S1 there
    soc = SOC + [[0, 0]]
    s1 = minimum('S1', [-0.011, 0.0, 0.03], [0.0, 0.02], soc, emitter='S2')
    t1 = minimum('T1', [0.0, 0.0, 0.03], [0.0, 0.02], soc)
    isc = table(s1, t1)['manifold']['ISC']
    assert 'S2' in isc['populations']
    assert any(p['pair'] == 'S2->T1' and p['herzberg_teller']
               for p in isc['pairs'])


def test_no_declared_gap_rule_computes_no_manifold():
    s1 = minimum('S1', [0.0, 0.01], [0.0, 0.003], SOC)
    t1 = minimum('T1', [0.0, 0.01], [0.0, 0.003], SOC)
    out = table(s1, t1, gaps=None)
    assert out['manifold']['status'] == 'not declared'
    assert out['pl_manifold'] is None
    assert out['pl'] is not None and out['pl_with_t2'] is not None


def test_a_gap_rule_is_declared_in_full():
    with pytest.raises(ValueError):
        rates.ManifoldGaps('vertical', None)
    with pytest.raises(ValueError):
        rates.ManifoldGaps('stacked', {'S2': -0.8})
    with pytest.raises(ValueError):
        rates.ManifoldGaps('adiabatic', None)


def test_adiabatic_gaps_stand_on_each_state_s_own_minimum():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    soc = SOC + [[0, 0]]
    s1 = minimum('S1', [0.0, 0.02, 0.03], [0.0, 0.02], soc)
    # S2 vertically 3 k_B T above S1 at the T1 minimum, relaxed 2 k_B T above
    # it; S3 and T2 have no minimum
    t1 = minimum('T1', [0.0, 3 * kt, 0.03], [0.0, 0.02], soc)
    e_s2 = E_S1 + 2 * kt
    gaps = rates.ManifoldGaps('adiabatic', {'S2': e_s2})
    risc = table(s1, t1, gaps=gaps)['manifold']['RISC']
    pairs = {p['pair']: p for p in risc['pairs']}
    assert set(pairs) == {'T1->S1', 'T1->S2'}
    assert pairs['T1->S2']['dE_eV'] == pytest.approx(
        (e_s2 - E_T1) * HARTREE_TO_EV)
    assert risc['gap_rule'] == 'adiabatic'
    assert risc['without_level'] == ['T2', 'S3']
    assert risc['complete'] == {'initial': False, 'final': False}
    stacked = table(s1, t1)['manifold']['RISC']
    assert {p['pair']: p['dE_eV'] for p in stacked['pairs']}['T1->S2'] == (
        pytest.approx((E_S1 - E_T1 + 3 * kt) * HARTREE_TO_EV))


def test_adiabatic_reservoirs_obey_detailed_balance():
    kt = BOLTZMANN_HARTREE_PER_KELVIN * TEMPERATURE
    soc = [[0, 0, 0], [1e-6, 3e-6, 2e-6], [5e-6, 1e-6, 4e-6]]
    # the vertical splittings differ from the relaxed ones on purpose
    s1 = minimum('S1', [0.0, 5 * kt], [0.0, 6 * kt], soc)
    t1 = minimum('T1', [0.0, 7 * kt], [0.0, 3 * kt], soc)
    e_s2, e_t2 = E_S1 + 2 * kt, E_T1 + 1 * kt
    gaps = rates.ManifoldGaps('adiabatic', {'S2': e_s2, 'T2': e_t2})
    man = table(s1, t1, gaps=gaps)['manifold']
    z_s = sum(np.exp(-(e - E_T1) / kt) for e in (E_S1, e_s2))
    z_t = sum(np.exp(-(e - E_T1) / kt) for e in (E_T1, e_t2))
    assert len(man['ISC']['pairs']) == len(man['RISC']['pairs']) == 4
    assert z_s * man['ISC']['k_eff'] == pytest.approx(
        TRIPLET_SUBLEVELS * z_t * man['RISC']['k_eff'], rel=1e-6)


def test_t2_channels_are_refused_with_the_fc_rows():
    s1 = minimum('S1', [0.0, 0.01], [0.0, 0.003], SOC)
    t1 = minimum('T1', [0.0, 0.01], [0.0, 0.003], SOC)
    ok = table(s1, t1)
    assert ok['t2_status'] == 'computed' and ok['t2_refused_by'] == []
    assert ok['k_risc_t2'] > 0 and ok['k_isc_t2'] > 0
    bad = table(s1, t1, lam_rule=1.1 * RATE_FC_LAMBDA_RATIO_MAX * LAMBDA)
    assert bad['k_risc_t2'] is None and bad['k_isc_t2'] is None
    assert bad['t2_status'] == 'refused'
    assert bad['t2_refused_by'] == ['ISC S1->T1', 'RISC T1->S1']
    assert bad['pl_with_t2'] is None
    # the splitting is a vertical energy, not a rate on the density
    assert bad['dE_T2T1_vertical_eV'] == ok['dE_T2T1_vertical_eV']


def test_pl_manifold_says_when_its_manifold_is_incomplete():
    s1 = minimum('S1', [0.0, 0.01], [0.0, 0.003], SOC)
    t1 = minimum('T1', [0.0, 0.01], [0.0, 0.003], SOC)
    # 'adiabatic' with S1 and T1 alone: S2 and T2 have no level
    lone = table(s1, t1, gaps=rates.ManifoldGaps('adiabatic', {}))
    assert lone['pl_manifold'] is not None
    assert lone['pl_manifold_complete'] is False
    assert lone['manifold']['RISC']['without_level']
    # stacked, every root within reach and beyond the window computed
    far = minimum('S1', [0.0, 0.05], [0.0, 0.05], SOC)
    far_t = minimum('T1', [0.0, 0.05], [0.0, 0.05], SOC)
    full = table(far, far_t)
    assert full['pl_manifold_complete'] is True
    assert table(s1, t1, gaps=None)['pl_manifold_complete'] is None
