"""The radiative rate and the two-level kinetics a PL experiment reports.

Everything here has a limit with a known answer, which is the only way to test
a rate expression without a second implementation to compare against:

- an oscillator strength of one at 3 eV is a nanosecond-scale lifetime, the
  textbook number for a fully allowed transition;
- spontaneous emission is CUBIC in the emission energy, so the same dipole at
  twice the energy is eight times the rate -- the single most likely place to
  lose a power, and the reason the emission energy and not the vertical one
  belongs in it;
- with no intersystem crossing the yield collapses to k_r / (k_r + k_nr) and
  the delayed component vanishes;
- a triplet that neither decays nor returns traps everything that crosses, so
  the yield is k_r / k_S and the delayed lifetime is infinite rather than a
  division by zero;
- prompt and delayed yields must sum to the closed-form total, which is a
  different expression -- one is a 2x2 matrix inverse, the other a split over
  two exponentials, and they agree only if both are right.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.constants import HARTREE_TO_EV
from src.properties.rates import photoluminescence, radiative_rate


def dipole_for(f, omega):
    """|mu| giving oscillator strength f at omega, from f = (2/3) w |mu|^2."""
    return np.sqrt(1.5 * f / omega)


def test_allowed_transition_is_nanoseconds():
    w = 3.0 / HARTREE_TO_EV
    tau = 1.0 / radiative_rate(w, dipole_for(1.0, w))
    assert 1e-9 < tau < 5e-9


def test_rate_is_cubic_in_the_emission_energy():
    assert np.isclose(radiative_rate(0.2, 1.0) / radiative_rate(0.1, 1.0), 8.0,
                      rtol=1e-12)


def test_vector_and_magnitude_agree():
    v = np.array([0.3, -0.4, 1.2])
    assert np.isclose(radiative_rate(0.1, v),
                      radiative_rate(0.1, np.linalg.norm(v)), rtol=1e-14)


def test_emission_energy_must_be_positive():
    with pytest.raises(ValueError):
        radiative_rate(-0.1, 1.0)


def test_no_crossing_gives_the_bare_yield():
    d = photoluminescence(k_r=1e7, k_isc=0.0, k_risc=0.0, k_nr_s=1e7)
    assert np.isclose(d['phi_total'], 0.5)
    assert d['phi_delayed'] == 0.0
    assert np.isclose(d['k_prompt'], 2e7)


def test_trapped_triplet_does_not_divide_by_zero():
    d = photoluminescence(k_r=1e7, k_isc=1e8, k_risc=0.0)
    assert np.isclose(d['phi_total'], 1e7 / 1.1e8)
    assert np.isinf(d['tau_delayed'])
    assert d['phi_delayed'] == 0.0


@pytest.mark.parametrize('k_risc', (1e4, 1e6, 1e8))
def test_prompt_and_delayed_sum_to_the_closed_form(k_risc):
    d = photoluminescence(k_r=1e7, k_isc=1e8, k_risc=k_risc, k_nr_t=1e3)
    assert np.isclose(d['phi_prompt'] + d['phi_delayed'], d['phi_total'],
                      rtol=1e-10)
    assert d['k_prompt'] >= d['k_delayed'] > 0.0


def test_delayed_component_tracks_the_reservoir_not_k_r():
    """k_delayed is k_risc times the singlet's branching ratio, so a tenfold
    slower reverse crossing is a NEARLY tenfold longer delayed lifetime --
    9.92x here, not 10, because the branching shifts with it. The prompt
    component barely moves, which is what separates the two channels.
    """
    fast = photoluminescence(k_r=1e7, k_isc=1e8, k_risc=1e6)
    slow = photoluminescence(k_r=1e7, k_isc=1e8, k_risc=1e5)
    ratio = slow['tau_delayed'] / fast['tau_delayed']
    assert 9.0 < ratio < 10.0
    assert np.isclose(slow['tau_prompt'], fast['tau_prompt'], rtol=1e-2)


def test_slow_reverse_crossing_does_not_lose_the_delayed_root():
    """lambda_- FROM THE PRODUCT, NOT FROM THE DIFFERENCE.

    The two decay constants satisfy lambda_+ lambda_- = det exactly, and the
    subtracted form 0.5 (k_S + k_T - disc) loses every significant digit once
    4 k_isc k_risc << k_S^2 -- ordinary whenever the reverse crossing is slow.
    These numbers are formaldehyde's, where the ratio is 1e-20: the subtracted
    form returns lambda_- = 0 exactly, so tau_delayed reads infinite and
    phi_delayed is 0/0. The triplet is NOT trapped here (det > 0), so the
    trapped branch does not catch it and the nan reaches the report.
    """
    d = photoluminescence(k_r=1.9244e-2, k_isc=1.5597e-8, k_risc=7.9401e-23,
                          k_nr_s=1.6941e-49, k_nr_t=3.4654e-27)
    assert d['k_delayed'] > 0.0
    assert np.isfinite(d['tau_delayed'])
    assert np.isfinite(d['phi_delayed'])
    # the reservoir empties at its own rate, which is what lambda_- must be
    assert np.isclose(d['k_delayed'], 7.9401e-23 + 3.4654e-27, rtol=1e-6)
    assert np.isclose(d['phi_prompt'] + d['phi_delayed'], d['phi_total'],
                      rtol=1e-10)
