"""The quasiparticle gradient kernels refuse what they cannot honour.

`qp_set_gradient` and `qp_gradient_space_time` are called directly by tests
and tools as well as by the chain, which passes every frozen argument. Water/
cc-pVDZ Hartree-Fock on its ISDF factors:

  * a residue route the kernels do not know is refused on both, before any
    work; the set kernel must not run the Newton with no residue backend;
  * residue_route has no default: 'auto', a backend decided per geometry, is
    not a mode a surface may take by omission;
  * per-state static terms must be one per state of the set.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.qp_space_time import qp_gradient_space_time, qp_set_gradient
from tests.isdf_test_grid import g1

NTAU, NFREQ = 8, 16


@pytest.fixture(scope='module')
def kernel_args():
    """(X, D, eps, nocc, grid, nu, wt) of water and the states nocc-2..nocc."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    return (X, D, eps, nocc, grid, nu, wt), [nocc - 2, nocc - 1, nocc]


def test_an_unknown_residue_route_is_refused_on_both_kernels(kernel_args):
    args, states = kernel_args
    with pytest.raises(ValueError, match="residue_route 'laplac'"):
        qp_set_gradient(*args, states, np.ones(len(states)),
                        residue_route='laplac')
    with pytest.raises(ValueError, match="residue_route 'laplac'"):
        qp_gradient_space_time(*args, states[1], residue_route='laplac')


def test_residue_route_has_no_default(kernel_args):
    args, states = kernel_args
    with pytest.raises(TypeError, match='residue_route'):
        qp_set_gradient(*args, states, np.ones(len(states)))
    with pytest.raises(TypeError, match='residue_route'):
        qp_gradient_space_time(*args, states[1])


def test_static_terms_of_another_length_are_refused(kernel_args):
    """One static term per state: a list of the wrong length would be read
    by position, silently."""
    args, states = kernel_args
    with pytest.raises(ValueError, match='static terms'):
        qp_set_gradient(*args, states, np.zeros(len(states)),
                        xc_correction=np.zeros(len(states) + 1),
                        residue_route='sop')
