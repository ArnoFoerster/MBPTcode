"""The fast twin of tests/test_metallic_grid_wiring.py.

The nightly file's Li monolayer in gth-dzvp costs a 96 s SCF and its RPA
ladder (six frequency integrations up to nw = 32) another ten minutes. This
twin keeps every check that needs neither, on the same layer in gth-szv:

  1. `beta_from_mf` reads the Fermi-Dirac width and refuses a Gaussian one;
  2. the silent half of the integer window (gth-szv: a positive, fictitious
     e_min, no guard fires) and its repair by mo_occ; the loud half
     (gth-dzvp, negative e_min) stays nightly;
  3. the T = 0 path is bit-identical on a gapped slab;
  4. the Pade order cap and that it is sized from Sigma's range.

The end-to-end metal (the periodic RPA correlation energy on the thermal
grid, how it differs from the beta-free one and its convergence in nw) stays
nightly: building the damped integrals alone costs 88 s even in gth-szv.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.Periodic.pbc_occupations import (split_transition_window,
                                                          transition_window)
from src.SingleReference.Periodic.pbc_rpa import build_freq_grid
from tests.test_metallic_grid_wiring import (check_beta_from_mf,
                                             check_gapped_is_untouched,
                                             check_pade_cap_is_sized_from_sigma_not_from_w,
                                             check_pade_order_cap, li_layer)

KMESH = (3, 3, 1)


@pytest.fixture(scope='module')
def szv():
    return li_layer(kmesh=KMESH, basis='gth-szv')


def test_beta_is_read_off_the_smearing(szv):
    assert check_beta_from_mf(szv[1])


def test_the_silent_half_of_the_integer_window(szv):
    _, mf = szv
    e = np.asarray(mf.mo_energy)
    f = np.asarray(mf.mo_occ, dtype=float)
    n0 = int((f > 1e-8).sum(axis=1)[0])
    e_int, _ = split_transition_window(list(e), n0)
    honest, _ = transition_window(e, f)
    build_freq_grid(list(e), n0, 16)            # no guard fires
    assert 0 < honest < e_int, (honest, e_int)
    ref = build_freq_grid(list(e), n0, 16)
    fixed = build_freq_grid(list(e), n0, 16, mo_occ=f)
    assert fixed[0].min() < ref[0].min()


def test_the_zero_temperature_path_is_untouched():
    assert check_gapped_is_untouched()


def test_the_pade_order_cap():
    assert check_pade_order_cap()
    assert check_pade_cap_is_sized_from_sigma_not_from_w()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
