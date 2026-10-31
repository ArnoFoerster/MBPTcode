"""Gates for `duschinsky.cumulants_of` on wide bands.

The first two cumulants of a band are read off ln G at two short times. A band
whose soft modes carry a large thermal amplitude into stiff ones is wide, and
its short-time expansion needs steps far below the slowest period: on the
rectilinear 4-(dimethylamino)benzonitrile S1/T1 model (TDA-wB97X-D3/def2-SVP
Hessians) the first step reads kappa_1 4.2 eV for RISC against the closed
form 46.7 eV. The step is halved until the estimates settle; a narrow band
settles at the first step and keeps its bits.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import pytest

from src.Base.numerics.constants import HARTREE_TO_EV
from src.properties import duschinsky as dm
from src.properties.duschinsky import (DuschinskyOscillators, cumulants_of,
                                       mean_energy_transfer)
from test_curvilinear_duschinsky import models
from test_duschinsky import TWO_MODE
from test_duschinsky_guard import dmabn  # noqa


@pytest.mark.parametrize('coordinates', ['rectilinear', 'curvilinear'])
@pytest.mark.parametrize('direction', ['absorption', 'emission'])
def test_a_wide_band_reads_the_closed_form_mean(dmabn, coordinates,
                                                direction):
    _, hs, ht, _, _ = dmabn
    m = models(hs, ht)[coordinates]
    exact = mean_energy_transfer(m, 300.0, direction)
    k1, k2 = DuschinskyOscillators(m).cumulants(300.0, direction)
    assert k1 == pytest.approx(exact, rel=1e-5)
    assert k2 > 0.0
    if coordinates == 'rectilinear' and direction == 'absorption':
        # the first step alone is ten times off
        h = dm.CUMULANT_STEP * 2.0 * np.pi / float(
            min(m.omega_g.min(), m.omega_e.min()))
        first, _ = dm._richardson(m.correlation(300.0, direction), h)
        assert first * HARTREE_TO_EV < 5.0
        assert exact * HARTREE_TO_EV == pytest.approx(46.69, abs=0.01)


@pytest.mark.parametrize('direction', ['absorption', 'emission'])
def test_a_narrow_band_keeps_the_first_step_bitwise(direction):
    corr = TWO_MODE.correlation(300.0, direction)
    wmin = float(min(TWO_MODE.omega_g.min(), TWO_MODE.omega_e.min()))
    h = dm.CUMULANT_STEP * 2.0 * np.pi / wmin
    assert cumulants_of(corr, wmin) == dm._richardson(corr, h)


def test_a_band_that_never_settles_is_refused(monkeypatch):
    monkeypatch.setattr(dm, 'DUSCHINSKY_CUMULANT_MAX_HALVINGS', 2)
    rng = np.random.default_rng(1)

    def noisy(t):
        return -1j * t + rng.normal(size=len(t)) * 1e-3 * (t > 0)
    with pytest.raises(RuntimeError, match='did not settle'):
        cumulants_of(noisy, 1.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
