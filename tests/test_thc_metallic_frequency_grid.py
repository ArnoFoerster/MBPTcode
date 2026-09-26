"""The THC frequency grid on a metal.

`beta` reaches the molecular GW route and the periodic RI route, but not the
THC route, which therefore built a T = 0 grid for a Fermi-smeared mean field.

The grid is scaled by the smallest particle-hole transition. In a metal that is
the level spacing at mu, so it collapses as the k-mesh refines and the
quadrature ends up spanning a small fraction of the transition spectrum while
still returning a plausible energy. Refining nw does not recover it: a
quadrature on the wrong interval has nothing to converge to.

Checks are response curves over the k-mesh, not pinned energies. Any single
mesh looks reasonable on its own; what identifies the defect is coverage
falling as the mesh improves. Windows below are measured on bulk Li
(gth-dzv, PBE, sigma = 0.01).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.grids import gauss_legendre_grid              # noqa: E402
from src.Base.utils.matsubara import thermal_e_min                # noqa: E402
from src.SingleReference.Periodic.pbc_isdf_rpa import (           # noqa: E402
    frequency_grid_occ, rpa_ecorr_thc, rpa_ecorr_thc_streaming)

# (kmesh, e_min, e_max), measured on bulk Li:
# gth-dzv, PBE, Fermi smearing at sigma = 0.01 Ha, default FFT grid.
LI_WINDOWS = [(2, 5.208e-02, 6.189), (3, 7.087e-03, 6.188),
              (4, 5.263e-03, 6.247), (5, 1.873e-03, 6.275),
              (6, 3.154e-04, 6.281)]
BETA = 100.0                      # 1/sigma for sigma = 0.01 Ha


def _synthetic(e_min, nmo=6):
    """(mo_energy, mo_occ, kplus) with a prescribed smallest transition.

    Synthetic because the defect is in the GRID SCALE, which depends on the
    transition window and nothing else; driving an SCF to reproduce a particular
    e_min would make the test slow and its input harder to see.
    """
    occ = np.array([[2.0, 2.0, 1.0, 1.0, 0.0, 0.0]])
    e = np.array([[-1.0, -0.5, 0.0, e_min, 3.0, 6.0]])
    return e[:, :nmo], occ[:, :nmo], np.array([[0]])


def test_floor_restores_spectrum_coverage():
    """Every measured Li window must span its spectrum once floored.

    The raw column is the defect and the floored column is the fix; asserting
    both keeps the test honest about which half it is checking.
    """
    raw_fail = 0
    for k, e_min, e_max in LI_WINDOWS:
        raw, _ = gauss_legendre_grid(32, w0=0.5 * e_min)
        flo, _ = gauss_legendre_grid(32, w0=0.5 * thermal_e_min(BETA, e_min))
        if raw[-1] < e_max:
            raw_fail += 1
        assert flo[-1] > e_max, (
            f'{k}x{k}x{k}: floored grid still stops at {flo[-1]:.2f} Ha, short '
            f'of e_max = {e_max:.2f}')
    assert raw_fail >= 4, (
        'the unfloored grid misses the spectrum at 4 of 5 meshes; if it stops '
        'doing so, this test no longer measures the defect')


def test_coverage_degrades_with_the_mesh_without_the_floor():
    """The signature that a pinned energy could never have caught.

    Coverage should not get WORSE as the calculation gets better. It does,
    monotonically, because e_min is an accident of level positions in a metal.
    """
    frac = []
    for k, e_min, e_max in LI_WINDOWS:
        raw, _ = gauss_legendre_grid(32, w0=0.5 * e_min)
        frac.append(min(1.0, raw[-1] / e_max))
    assert frac == sorted(frac, reverse=True), f'expected decreasing, got {frac}'
    assert frac[0] == 1.0 and frac[-1] < 0.2, (
        f'coarsest mesh should cover and finest should not: {frac}')


@pytest.mark.parametrize('k,e_min,e_max', LI_WINDOWS)
def test_frequency_grid_occ_applies_the_floor(k, e_min, e_max):
    """The helper both drivers call, on each measured window."""
    mo_e, mo_occ, kplus = _synthetic(e_min)
    _, _, raw_min, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 32, beta=None)
    freqs, _, flo_min, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 32, beta=BETA)
    assert flo_min >= np.pi / BETA - 1e-12
    assert flo_min >= raw_min
    assert freqs[-1] > e_max


def test_a_gap_above_the_floor_is_untouched():
    """A gapped system keeps its own window: max(gap, pi/beta) is the gap."""
    gap = 0.5                                  # well above pi/beta = 0.0314
    mo_e, mo_occ, kplus = _synthetic(gap)
    f_raw, w_raw, min_raw, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 24, beta=None)
    f_flo, w_flo, min_flo, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 24, beta=BETA)
    assert min_flo == min_raw
    assert np.allclose(f_raw, f_flo, rtol=0, atol=0), 'gapped grid must be bit-identical'
    assert np.allclose(w_raw, w_flo, rtol=0, atol=0)


def test_beta_zero_reproduces_the_unfloored_grid():
    """The escape hatch for comparing against pre-fix numbers.

    thermal_e_min rejects beta <= 0, so this has to be handled before the call
    rather than passed through -- otherwise reproducing an old number raises.
    """
    mo_e, mo_occ, kplus = _synthetic(1.873e-03)
    f_none, _, _, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 32, beta=None)
    f_zero, _, _, _ = frequency_grid_occ(mo_e, mo_occ, kplus, 32, beta=0.0)
    assert np.allclose(f_none, f_zero, rtol=0, atol=0)


@pytest.mark.parametrize('fn', [rpa_ecorr_thc, rpa_ecorr_thc_streaming])
def test_both_drivers_accept_and_default_beta(fn):
    """Wiring, not physics: the parameter has to exist and default from the mf.

    `rpa_ecorr_thc_streaming` had no beta parameter AT ALL, so a smeared mean
    field could not have been given the right grid even by a caller who knew to
    ask. Checked by signature so it costs no SCF.
    """
    import inspect
    sig = inspect.signature(fn)
    assert 'beta' in sig.parameters, f'{fn.__name__} takes no beta'
    assert sig.parameters['beta'].default is None, (
        f'{fn.__name__} must default beta to None so it can be read off the '
        f'mean field; a hard default would silently ignore the smearing')
    src = inspect.getsource(fn)
    assert 'beta_from_mf' in src, (
        f'{fn.__name__} does not default beta from the mean field, so a smeared '
        f'SCF still gets a T = 0 grid unless the caller restates the temperature')

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
