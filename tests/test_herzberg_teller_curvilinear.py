"""Gate for the Herzberg-Teller band of a non-orthogonal (curvilinear)
Duschinsky matrix: the dipole term of the correlation function with J^-T where
J multiplies the initial frequencies, and J^-1 for J^T.

The reference is the sum over vibronic states of test_herzberg_teller.py on
the non-orthogonal two-mode model of test_curvilinear_duschinsky.py: the
final states on the grid normalized with the Jacobian |det J|, the dipole
mu(q) = mu0 + m^T q_g expanded about the ground minimum and carried into the
excited coordinates through q_g = J^-1 (q_e - K).
"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import pytest
from scipy.stats import norm

from src.properties.duschinsky import DuschinskyOscillators
from src.properties.herzberg_teller import TransitionDipole
from test_curvilinear_duschinsky import CURVED
from test_duschinsky import overlap_lines, rotation
from test_herzberg_teller import DE, DMU, MU0, SIGMA


def dipole_on_grid(model, mu0, m, direction):
    """mu_a(q) on the grid of the initial coordinates of `direction`, for a
    dipole expanded about the ground minimum."""
    inv = np.linalg.inv(model.J)

    def op(q):
        qg = q if direction == 'absorption' else np.einsum(
            'ij,jab->iab', inv, q - model.K[:, None, None])
        return [mu0[a] + np.einsum('k,kab->ab', m[:, a], qg)
                for a in range(3)]
    return op


@pytest.mark.parametrize('temperature, direction, n_init',
                         [(1.0, 'absorption', 1), (1.0, 'emission', 1),
                          (150.0, 'absorption', 9), (150.0, 'emission', 9)])
def test_curvilinear_herzberg_teller_band_matches_the_sum_over_states(
        temperature, direction, n_init):
    dip = TransitionDipole(MU0, DMU, 'ground')
    wi, wf, j, k = CURVED.oriented(direction)
    x, w = overlap_lines(wi, wf, j, k, temperature, n_init, 40,
                         operator=dipole_on_grid(CURVED, MU0, DMU, direction))
    w = w * abs(np.linalg.det(j))
    ref = np.array([(w * norm.pdf(-d - x, scale=SIGMA)).sum() for d in DE])
    got = DuschinskyOscillators(CURVED, dip).density(
        DE, temperature, direction, broadening=SIGMA)
    assert np.abs(got - ref).max() < 1e-8 * ref.max()
    assert dip.strength(CURVED, temperature, direction) == pytest.approx(
        w.sum(), rel=1e-8)


def test_the_expansion_point_does_not_change_the_curvilinear_band():
    """The same linear dipole about the ground or the excited minimum: about
    the excited one, mu0' = mu0 + m^T q_g(e) and m' = J^-T m."""
    inv = np.linalg.inv(CURVED.J)
    mu_e = MU0 + DMU.T @ (-inv @ CURVED.K)
    dmu_e = inv.T @ DMU
    for direction in ('absorption', 'emission'):
        a = DuschinskyOscillators(CURVED, TransitionDipole(
            MU0, DMU, 'ground')).density(DE, 150.0, direction,
                                         broadening=SIGMA)
        b = DuschinskyOscillators(CURVED, TransitionDipole(
            mu_e, dmu_e, 'excited')).density(DE, 150.0, direction,
                                             broadening=SIGMA)
        assert np.abs(a - b).max() < 1e-10 * a.max()


def test_an_orthogonal_curvilinear_dipole_band_is_the_rectilinear_one():
    rect = replace(CURVED, J=rotation(0.35), coordinates='rectilinear')
    curv = replace(rect, coordinates='curvilinear')
    dip = TransitionDipole(MU0, DMU, 'ground')
    for direction in ('absorption', 'emission'):
        a = DuschinskyOscillators(rect, dip).density(DE, 150.0, direction,
                                                     broadening=SIGMA)
        b = DuschinskyOscillators(curv, dip).density(DE, 150.0, direction,
                                                     broadening=SIGMA)
        assert np.abs(a - b).max() < 1e-11 * a.max()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
