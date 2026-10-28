"""The state models' force of a Kohn-Sham reference (`df_correlation_force.
hf_force`, which `GWBSEState` and `HFState` take) carries the motion of the
XC quadrature (pyscf's `grid_response`, off by default): it is translation
invariant to 1e-9 Ha/Bohr on water/cc-pVDZ LRC-wPBEh on grid 1, where
without it the forces sum to 1.6e-4 Ha/Bohr, and the scheme-B forces of
`test_photophysics_scheme_b_run` miss their own differences by 3e-7.

Run: python -m pytest tests/test_state_model_ks_force.py
"""
import numpy as np
from pyscf import dft, gto

from src.gradients.df_correlation_force import hf_force


def test_a_kohn_sham_state_force_carries_its_grid():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol, xc='lrc-wpbeh').density_fit(auxbasis='cc-pvdz-jkfit')
    mf.grids.level, mf.conv_tol, mf.conv_tol_grad = 1, 1e-12, 1e-9
    mf.kernel()
    assert np.abs(hf_force(mf).sum(axis=0)).max() < 1e-9
