"""The quasiparticle-set gradient holds one frequency's B_p_bar, and the
single-state gradient is the set gradient of one state.

`qp_set_gradient`'s reverse pass over the contour-deformation frequencies
must not keep every owned frequency's (naux, nmo) B_p_bar for every weighted
state until the loop ends, nfreq x nstates arrays (87 GB on one rank for a
four-state set at the chlorophyllide dimer/cc-pVTZ with 64 frequencies). Each
frequency's terms join the running sum as soon as the frequency is solved, in
frequency order through `ordered_chain_sum`, so the bits are those of the
whole-array sum at every rank count. Water/cc-pVDZ Hartree-Fock
on its ISDF factors, one frequency per block:

  * the tracemalloc peak of a three-state solve grows by less than
    GROWTH_MAX of one frequency's per-state holding when the frequencies go
    from NFREQS[0] to NFREQS[1] (keeping every frequency grows by about one);
  * `qp_gradient_space_time` returns the root, Z and the three adjoints of
    `qp_set_gradient` on that one state at weight one, to a few ulp, serially and
    on two simulated ranks, on the explicit and the pole-model route.
"""
import tracemalloc
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import distributed, run_simulated
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.qp_space_time import qp_gradient_space_time, qp_set_gradient
from tests.isdf_test_grid import g1
from tests.pins import assert_same

NTAU = 8
#: two contour-deformation quadratures, three times apart
NFREQS = (16, 48)
#: the peak's growth over NFREQS, in units of the per-frequency holding there
GROWTH_MAX = 0.2
WEIGHTS = np.array([0.3, -1.1, 0.8])


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy, float)
    return dict(X=X, D=D, eps=eps, nocc=nocc,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]),
                states=[nocc - 2, nocc - 1, nocc])


def quadrature(w, nfreq):
    """(grid, nu, wt) of an nfreq-point contour-deformation quadrature."""
    eps, nocc = w['eps'], w['nocc']
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(nfreq, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    return grid, nu, wt


def peak_of(w, nfreq):
    """tracemalloc peak, bytes, of one weighted three-state solve."""
    grid, nu, wt = quadrature(w, nfreq)
    naux = w['D'].shape[1]
    one_frequency = 3 * naux ** 2 * 8 / 1e9
    with distributed(None):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        qp_set_gradient(w['X'], w['D'], w['eps'], w['nocc'], grid, nu, wt,
                        w['states'], WEIGHTS, mu=w['mu'],
                        residue_route='explicit', tile_gb=one_frequency)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
    return peak


def test_peak_is_flat_in_the_number_of_frequencies(water):
    w = water
    naux, nmo = w['D'].shape[1], w['X'].shape[1]
    held = (NFREQS[1] - NFREQS[0]) * len(w['states']) * naux * nmo * 8
    growth = (peak_of(w, NFREQS[1]) - peak_of(w, NFREQS[0])) / held
    print(f'[info] peak growth {NFREQS[0]} -> {NFREQS[1]} frequencies: '
          f'{growth:.3f} of the per-frequency B_p_bar holding')
    assert growth < GROWTH_MAX


@pytest.mark.parametrize('size', [1, 2])
@pytest.mark.parametrize('route', ['explicit', 'sop'])
def test_single_state_is_the_set_of_one(water, route, size):
    w = water
    grid, nu, wt = quadrature(w, NFREQS[0])
    p = w['nocc'] - 1
    args = (w['X'], w['D'], w['eps'], w['nocc'], grid, nu, wt)
    kw = dict(mu=w['mu'], residue_route=route, xc_correction=0.01)

    def both(comm):
        single = qp_gradient_space_time(*args, p, comm=comm, **kw)
        ro = {}
        roots, *adjoints = qp_set_gradient(*args, [p], [1.0], route_out=ro,
                                           comm=comm, **kw)
        return single, (roots[0], ro['z'][0], *adjoints)

    for single, of_set in run_simulated(both, size):
        assert len(single) == len(of_set)
        # the root (Ha), Z (1), then the adjoints
        for i, (a, b) in enumerate(zip(single, of_set)):
            assert_same(a, b, unit='1' if i == 1 else 'Ha',
                        quantity='energy' if i < 2 else 'root', label=i)
