"""Vibronic and rate models read vibrations only.

An imaginary mode (negative from `vibronic.normal_modes`) is a saddle, and a
mode below VIBRONIC_MODE_FLOOR_CM is a rigid-body remnant or a flat
direction whose Huang-Rhys factor g^2 / (2 omega^3) has no bound. Both were
dropped silently (S = 0, or the mode skipped), which changes the sum rule
unseen; every entry now refuses them.
"""
import numpy as np
import pytest

from src.Base.numerics.constants import HARTREE_TO_CM, VIBRONIC_MODE_FLOOR_CM
from src.photophysics import rates
from src.properties import franck_condon
from src.properties.spin_vibronic import mode_derivatives
from src.properties.vibronic import (huang_rhys_from_gradient,
                                     reorganization_from_huang_rhys)

#: Hartree: two vibrations and one bad mode, imaginary or below the floor
GOOD = np.array([0.004, 0.012])
BAD = {'imaginary': -50.0 / HARTREE_TO_CM,
       'below the floor': 0.5 * VIBRONIC_MODE_FLOOR_CM / HARTREE_TO_CM}


def spectrum(bad):
    return np.append(GOOD, BAD[bad])


@pytest.mark.parametrize('bad', sorted(BAD))
def test_huang_rhys_and_lambda_refuse(bad):
    omega = spectrum(bad)
    modes, masses = np.eye(3), np.ones(1)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        huang_rhys_from_gradient(np.ones(3), omega, modes, masses)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        reorganization_from_huang_rhys(np.ones(3), omega)


@pytest.mark.parametrize('bad', sorted(BAD))
def test_the_rate_routines_refuse(bad):
    omega = spectrum(bad)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        rates.internal_conversion_rate(-0.1, np.ones(3), omega,
                                       lambda e: 1.0, 300.0)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        mode_derivatives(np.ones((1, 3, 3)), np.eye(3), np.ones(1), omega)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        rates.rate_table(None, None, 0.0, omega, np.ones(3), 300.0, 0.0)


def test_vibrations_pass():
    franck_condon.require_vibrational_modes(GOOD, 'test')
    s, _ = huang_rhys_from_gradient(np.ones(3), GOOD, np.eye(3)[:, :2],
                                    np.ones(1))
    assert s == pytest.approx(1.0 / (2.0 * GOOD ** 3))
