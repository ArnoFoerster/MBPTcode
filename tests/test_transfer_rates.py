"""Golden-rule rates of excitation transfer between molecules
(`src.photophysics.transfer_rates`) against independent references.

WHAT EACH GATE IS FOR:

- rho_FC of a two-molecule process is the overlap of the donor's band with the
  acceptor's: the joined Huang-Rhys spectrum at dE = dE_D + dE_A equals the
  numerical convolution of the two molecules' own densities. With the
  acceptor's energy change of the wrong sign the identity fails.
- detailed balance: forward over reverse is exp(-dE / k_B T).
- the classical limit: with only soft modes the rate is the Marcus rate.
- the linear coupling model: the spectrum between two excited states reads
  their gradient difference, and a mode that is not a vibration is refused.
- the spin weight scales the rate, and the triplet transfer refuses
  transitions that are not T1 -> S0 and S0 -> T1.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         HARTREE_TO_CM)
from src.photophysics.transfer_rates import (TTA_SPIN_WEIGHTS, Transition,
                                             dexter_rate, transfer_rate,
                                             transition_from_gradients)
from src.properties.franck_condon import TRAPEZOID, fc_weighted_dos

ROOM = 298.15
CM = 1.0 / HARTREE_TO_CM


def donor_acceptor():
    donor = Transition('D', 'T1', 'S0', -0.060,
                       np.array([0.4, 0.8, 0.3]),
                       np.array([300.0, 1200.0, 1600.0]) * CM)
    acceptor = Transition('A', 'S0', 'T1', 0.045,
                          np.array([0.6, 0.2]),
                          np.array([500.0, 1500.0]) * CM)
    return donor, acceptor


def test_the_joined_spectrum_is_the_overlap_of_the_two_bands():
    donor, acceptor = donor_acceptor()
    sigma = 150.0 * CM
    de = donor.delta_e + acceptor.delta_e
    joined = transfer_rate(1e-4, (donor, acceptor), ROOM,
                           broadening=np.sqrt(2.0) * sigma)['rho_fc']
    x = np.linspace(-0.25, 0.15, 6001)
    # each band is a density in the electronic energy its molecule takes;
    # the process takes their sum
    rho_d = fc_weighted_dos(x, donor.s_k, donor.omega_k, ROOM,
                            broadening=sigma)
    rho_a = fc_weighted_dos(de - x, acceptor.s_k, acceptor.omega_k, ROOM,
                            broadening=sigma)
    overlap = TRAPEZOID(rho_d * rho_a, x)
    assert abs(joined - overlap) < 1e-6 * overlap
    flipped = Transition('A', 'S0', 'T1', -acceptor.delta_e, acceptor.s_k,
                         acceptor.omega_k)
    wrong = transfer_rate(1e-4, (donor, flipped), ROOM,
                          broadening=np.sqrt(2.0) * sigma)['rho_fc']
    assert abs(wrong - overlap) > 0.5 * overlap


def test_detailed_balance_and_the_classical_limit():
    donor, acceptor = donor_acceptor()
    out = transfer_rate(1e-4, (donor, acceptor), ROOM, lambda_env=0.01)
    assert abs(out['k'] / out['k_reverse'] / out['detailed_balance'] - 1.0) \
        < 1e-8
    back = transfer_rate(1e-4, (donor.reverse(), acceptor.reverse()), ROOM,
                         lambda_env=0.01)
    assert abs(back['k'] / out['k_reverse'] - 1.0) < 1e-10
    kt = BOLTZMANN_HARTREE_PER_KELVIN * ROOM
    soft = Transition('D', 'T1', 'S0', -0.02, np.full(20, 30.0),
                      np.full(20, 0.02 * kt))
    out = transfer_rate(1e-4, (soft,), ROOM)
    assert abs(out['k'] / out['k_marcus'] - 1.0) < 2e-2


def test_the_linear_coupling_spectrum():
    rng = np.random.default_rng(3)
    natm = 4
    modes = np.linalg.qr(rng.normal(size=(3 * natm, 5)))[0]
    masses = np.full(natm, 1822.9)
    omega = np.array([400.0, 800.0, 1100.0, 1500.0, 3000.0]) * CM
    g_t, g_s = rng.normal(size=(2, natm, 3)) * 1e-3
    st = transition_from_gradients('M', 'T1', 'S1', 0.05, omega, modes,
                                   masses, grad_initial=g_t, grad_final=g_s)
    diff = transition_from_gradients('M', 'T1', 'S1', 0.05, omega, modes,
                                     masses, grad_final=g_s - g_t)
    assert np.array_equal(st.s_k, diff.s_k)
    omega[0] = -50.0 * CM
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        transition_from_gradients('M', 'S0', 'T1', 0.1, omega, modes, masses,
                                  grad_final=g_t)


def test_spin_weight_and_refusals():
    donor, acceptor = donor_acceptor()
    one = transfer_rate(1e-4, (donor, acceptor), ROOM, lambda_env=0.005)['k']
    ninth = transfer_rate(1e-4, (donor, acceptor), ROOM, lambda_env=0.005,
                          spin_weight=TTA_SPIN_WEIGHTS['singlet'])['k']
    assert one > 0.0
    assert abs(ninth * 9.0 / one - 1.0) < 1e-14
    assert abs(sum(TTA_SPIN_WEIGHTS.values()) - 1.0) < 1e-15
    assert dexter_rate(1e-4, donor, acceptor, ROOM, lambda_env=0.005)['k'] == one
    with pytest.raises(ValueError, match='donor T1 -> S0'):
        dexter_rate(1e-4, acceptor, donor, ROOM, lambda_env=0.005)
    with pytest.raises(ValueError, match='omega > 0'):
        Transition('X', 'T1', 'S0', -0.1, [1.0], [-1e-3])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
