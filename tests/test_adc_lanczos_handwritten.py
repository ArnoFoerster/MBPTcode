"""The Lanczos spectral solve (solve(method='lanczos')) of the hand-written
matrix-free ADC routes, RHF and UHF: the peak nearest the Davidson root lies
within the frequency-grid spacing of it, with the same pole strength up to the
broadening.

H2O/STO-3G, IP window (-0.45, -0.2) Ha; each case runs in under a second.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.SingleReference.ADC.base import ADCSolver

WINDOW = (-0.45, -0.2)
# lanczos_spectral's default grid: 2000 points over the window
GRID_STEP = (WINDOW[1] - WINDOW[0]) / 2000


@pytest.fixture(scope='module')
def water():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                 basis='sto-3g', verbose=0)


@pytest.fixture(scope='module')
def rhf(water):
    return scf.RHF(water).run(conv_tol=1e-12)


@pytest.fixture(scope='module')
def uhf(water):
    return scf.UHF(water).run(conv_tol=1e-12)


def lanczos_matches_davidson(mf, level):
    solver = ADCSolver(mf, level=level, matrix_free=True)
    e, Z = solver.solve(nroots=1)
    peaks, weights = solver.solve(method='lanczos', omega_range=WINDOW)
    i = np.argmin(abs(peaks - e[0]))
    assert abs(peaks[i] - e[0]) < 2 * GRID_STEP
    assert abs(weights[i] - Z[0]) < 0.05
    return peaks[i]


@pytest.mark.parametrize('level', ['adc2x', 'adc3', 'faddeev_adc3'])
def test_restricted_handwritten_lanczos(rhf, level):
    lanczos_matches_davidson(rhf, level)


def test_unrestricted_handwritten_lanczos(rhf, uhf):
    # the spin-orbital hand-written route is ADC(3) only; on a closed shell
    # its spectrum is the restricted one
    peak_u = lanczos_matches_davidson(uhf, 'adc3')
    peak_r = lanczos_matches_davidson(rhf, 'adc3')
    assert abs(peak_u - peak_r) < 1e-6
