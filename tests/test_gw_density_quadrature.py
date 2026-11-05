"""The outer imaginary-frequency quadrature of the GW density and its
`gwdensity` Numerics section.

- The dynamic blocks equal the direct full-axis integral of G Sigma G
  (scipy quad) for a model pole self-energy, and assemble a symmetric matrix.
- The `gwdensity` section is a later section: its defaults leave the identity
  what it was without it; a changed setting changes it.
- The outer grid: Gauss-Legendre or tabulated minimax; refusals name the choice.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from scipy.integrate import quad

from src.Base.numerics.numerics import GWDensityNumerics, Numerics
from src.Base.utils import record_compat as rc
from src.Base.utils.grids.grids import minimax_supported_sizes
from src.SingleReference.GW.density_matrix import (dynamic_density_blocks,
                                                   outer_frequency_grid)

NMO, NOCC = 6, 3
EPS = np.array([-0.95, -0.6, -0.3, 0.25, 0.55, 1.0])


def _model():
    rng = np.random.default_rng(7)
    poles = np.array([-0.8, -0.4, 0.35, 0.9])
    a = rng.normal(scale=0.2, size=(len(poles), NMO))

    def sigma(z):
        return np.einsum('kp,kq,k->pq', a, a, 1.0 / (z - poles))
    return sigma


def _blocks_and_matrix():
    sigma = _model()
    numerics = Numerics(gwdensity={'nfreq': 200})
    points, weights = outer_frequency_grid(EPS, NOCC, numerics)
    sig = np.array([sigma(1j * w) for w in points])
    oo, ov, vv = dynamic_density_blocks(EPS, sig, points, weights, NOCC)
    full = np.zeros((NMO, NMO))
    full[:NOCC, :NOCC], full[:NOCC, NOCC:] = oo, ov
    full[NOCC:, :NOCC], full[NOCC:, NOCC:] = ov.T, vv
    return sigma, full


def test_dynamic_blocks_match_direct_quadrature():
    sigma, full = _blocks_and_matrix()
    ref = np.zeros((NMO, NMO))
    for p in range(NMO):
        for q in range(NMO):
            def f(w):
                z = 1j * w
                return (sigma(z)[p, q] / ((z - EPS[p]) * (z - EPS[q]))).real
            neg = quad(f, -np.inf, 0, limit=500, epsabs=1e-12)[0]
            pos = quad(f, 0, np.inf, limit=500, epsabs=1e-12)[0]
            ref[p, q] = (neg + pos) / (2 * np.pi)
    assert np.max(np.abs(full - ref)) < 1e-8


def test_dynamic_blocks_are_symmetric_and_real():
    _, full = _blocks_and_matrix()
    assert np.isrealobj(full)
    assert np.max(np.abs(full - full.T)) < 1e-12


def test_gwdensity_section_defaults_and_identity():
    assert Numerics().gwdensity == GWDensityNumerics()
    assert 'gwdensity' in rc.NUMERICS_LATER_SECTIONS
    now = Numerics().to_record()
    old = {k: v for k, v in now.items() if k != 'gwdensity'}
    assert Numerics.from_record(old).identity() == Numerics().identity()
    assert (Numerics(gwdensity={'nfreq': GWDensityNumerics().nfreq + 16}).identity()
            != Numerics().identity())


def test_outer_grid_choices():
    n = Numerics().gwdensity.nfreq
    points, weights = outer_frequency_grid(EPS, NOCC, Numerics())
    assert len(points) == len(weights) == n and np.all(points > 0)
    size = minimax_supported_sizes()[0]
    points, _ = outer_frequency_grid(
        EPS, NOCC, Numerics(gwdensity={'nfreq': size, 'grid': 'minimax'}))
    assert len(points) == size
    unsupported = max(minimax_supported_sizes()) + 1
    with pytest.raises(ValueError, match='supported sizes'):
        outer_frequency_grid(
            EPS, NOCC, Numerics(gwdensity={'nfreq': unsupported, 'grid': 'minimax'}))
    with pytest.raises(ValueError, match='simpson'):
        outer_frequency_grid(EPS, NOCC, Numerics(gwdensity={'grid': 'simpson'}))
