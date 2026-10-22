"""One Pade node rule, one node count, every route passing its nodes explicitly.

The space-time route samples Sigma_c(i omega) at 16 Gauss-Legendre nodes on
w0 = 1.0 (`constants.DEFAULT_NPADE` and `GL_FREQUENCY_SCALE`, the defaults of
`QuadratureNumerics`), the imaginary-frequency route at its 20 minimax W nodes,
scGW at its first 30 Matsubara points. `analyticalContinuation.pade_nodes` is
the one rule, on
PADE_NPADE and PADE_W0; the space-time default is pinned to it here until that
route reads it, and the imaginary-frequency route takes it through pade_freq.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import (DEFAULT_NPADE, GL_FREQUENCY_SCALE,
                                PADE_NPADE, PADE_W0)
from src.Base.numerics import Numerics
from src.Base.utils.analyticalContinuation import pade_nodes
from src.Base.utils.grids import gauss_legendre_grid
import src.SingleReference.GW.imaginary_axis as ia
from tests.pins import assert_same


def test_the_space_time_default_is_the_rule():
    """The space-time route's default Pade nodes are `pade_nodes()`."""
    quadrature = Numerics().quadrature
    assert DEFAULT_NPADE == PADE_NPADE == quadrature.npade
    assert GL_FREQUENCY_SCALE == PADE_W0 == quadrature.w0_gl
    assert_same(gauss_legendre_grid(quadrature.npade, w0=quadrature.w0_gl)[0],
                pade_nodes(), unit='1', quantity='root', label='Pade nodes')


def test_the_imaginary_frequency_route_samples_the_nodes_it_is_given(
        monkeypatch):
    """pade_freq reaches the sampling of Sigma_c; None keeps the W nodes."""
    mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    seen = []
    real = ia.imaginary_axis_sample_points

    def spy(freq, nocc, p, mu):
        seen.append(np.array(freq))
        return real(freq, nocc, p, mu)

    monkeypatch.setattr(ia, 'imaginary_axis_sample_points', spy)
    nocc = mol.nelectron // 2
    ia.solve_qp_energy_imaginary_axis(mf, mol, nocc, nocc - 1,
                                      pade_freq=pade_nodes())
    ia.solve_qp_energy_imaginary_axis(mf, mol, nocc, nocc - 1)
    assert_same(seen[0], pade_nodes(), unit='1', quantity='root',
                label='sampled nodes')
    assert len(seen[1]) == 20 and not np.array_equal(seen[1][:PADE_NPADE],
                                                     pade_nodes())
