"""W(i.omega) is sampled on frequencies built over the range its way back to
imaginary time is fitted over.

The space-time GW route transforms W - I from the frequency axis to the
self-energy's tau axis with a least-squares fit (`COSINE_WT`) of e^{-x tau},
x in rW = [0.3 e_min, 3 e_max], so its input frequencies are the minimax grid
of rW. The minimax grid of the bare window [e_min, e_max] does not reach the
fit's padding: on a chlorophyll-like spectrum (bare e_max/e_min = 337, rW
ratio 3370) that fit stalls at 5.9e-4 to 1.0e-3 from 26 to 34 points, growing
with the point count, while on the rW-built grid it falls to 2.5e-7 at 34.

The first test gates the residual on the production axes at that range, the
second that the driver samples W there (its fit is handed the minimax grid of
the fit's own range).
"""
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.grids import minimax_frequency_grid, minimax_time_grid
from src.Base.utils.time_frequency import COSINE_WT, minimax_transform_weights
from src.SingleReference.GW import imaginary_time, space_time
from src.SingleReference.GW.imaginary_time import (self_energy_fit_ranges,
                                                   screening_frequency_grid)

SIZES = list(range(22, 35, 2))
#: bare e_max/e_min of the chlorophyllide dimer, whose rW ratio is 3.4e3
BARE_RATIO = 337.0
#: the W fit's worst residual allowed from 26 points up
W_FIT_BOUND = 1e-5


def chlorophyll_like_spectrum(w_lo=0.15):
    """(eps, nocc, mu): a frontier pair about mu = 0, a deep core level and a
    high virtual spanning a bare window of `BARE_RATIO`."""
    w_hi = BARE_RATIO * w_lo
    core = -0.95 * w_hi
    return np.array([core, -w_lo / 2, w_lo / 2, w_hi + core]), 2, 0.0


def w_fit_residual(n, eps, nocc, mu):
    """Max residual of the route's omega -> tau fit of W - I at n points."""
    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau = 0.5 * minimax_time_grid(n, *rS)[0]
    freq = screening_frequency_grid(n, eps, nocc, mu=mu)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return minimax_transform_weights(COSINE_WT, tau, freq, *rW)[1]


def test_the_w_fit_converges_at_a_chlorophyll_like_range():
    """Below `W_FIT_BOUND` from 26 points, and falling over every four-point
    step and against the best smaller grid to within a factor two.

    Not strictly monotone: at 32 points it is 2.2e-6 against 1.2e-6 at 30,
    the floor of the per-point Tikhonov fit.
    """
    eps, nocc, mu = chlorophyll_like_spectrum()
    err = {n: w_fit_residual(n, eps, nocc, mu) for n in SIZES}
    table = ' '.join(f'{n}:{err[n]:.1e}' for n in SIZES)
    assert all(err[n] < W_FIT_BOUND for n in SIZES if n >= 26), table
    assert all(err[n + 4] < err[n] for n in SIZES if n + 4 in err), table
    assert all(err[n] <= 2.0 * min(err[m] for m in SIZES if m < n)
               for n in SIZES[1:]), table
    assert err[34] < 0.1 * err[26], table


def test_the_route_samples_w_on_the_range_its_fit_runs_over():
    """solve_qp_energy_space_time hands its W fit the minimax grid of the
    fit's own range, rW. The grid-row and blocked paths read the same axis
    from the driver."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    rW = self_energy_fit_ranges(eps, nocc, mu=mu)[0]
    ntau = 22
    seen = []

    def recording(kind, tau, omega, e_min, e_max, **kw):
        if kind == COSINE_WT:
            seen.append((np.array(omega), e_min, e_max))
        return minimax_transform_weights(kind, tau, omega, e_min, e_max, **kw)

    saved = imaginary_time.minimax_transform_weights
    imaginary_time.minimax_transform_weights = recording
    try:
        space_time.solve_qp_energy_space_time(mf, mol, nocc, nocc - 1,
                                              ntau=ntau, distribute=False)
    finally:
        imaginary_time.minimax_transform_weights = saved
    assert len(seen) == 1
    omega, lo, hi = seen[0]
    assert (lo, hi) == rW
    assert np.array_equal(omega, minimax_frequency_grid(ntau, lo, hi)[0])
    assert np.array_equal(omega,
                          screening_frequency_grid(ntau, eps, nocc, mu=mu)[0])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
