"""The frequency split keeps the serial chi0 rows, on any BLAS.

`owned_frequency_blocks` hands each rank the chi0(i.nu) rows of the
frequencies it owns, and every frequency-split route factorizes those rows:
the contour-deformation contraction of both quasiparticle gradients, the dRPA
energy and its adjoint, and the screened interaction in imaginary time and
its adjoint. The quasiparticle gradients' wc is an output partition -- a rank
writes its own rows and leaves the rest exact zeros, and the sum over ranks
adds nothing else -- so their roots are the serial ones bitwise PROVIDED each
row a rank computes is the serial row. The rows are one GEMM over tau, and a
GEMM row depends on the call's shape: OpenBLAS 0.3.18 moves 30% of wc's
elements by 1 to 224 ulp when a rank transforms its 3 of 24 rows alone
(water/cc-pVDZ), and on a cluster's OpenBLAS nodes that failed
`qp_set_gradient`'s bitwise root gate over 8 real ranks. MKL keeps such rows
bitwise, so a gate reading whichever BLAS this machine has may not be able to
fail. Here
the `np` that `owned_frequency_blocks` transforms with is replaced by one
whose tensordot scales its result by 1 + nrows * `ROW_COUNT_SKEW`: rows then
depend on how many rows the call has, as on OpenBLAS, but by far more than a
last bit, so neither a Newton root nor a re-associated sum can hide the
difference.

Gated, water/cc-pVDZ Hartree-Fock:
  * serially, every block `owned_frequency_blocks` yields is the serial call
    -- the tensordot of the whole block -- bitwise, on the real BLAS, for one
    block of all 24 frequencies and for blocks of five;
  * on the row-count-sensitive BLAS at 2, 3, 8 and 16 simulated ranks, every
    rank's rows are the serial block's rows bitwise, each owned frequency is
    yielded once, and the ranks' wc sum to the serial wc bitwise;
  * on the same BLAS at 2, 3 and 8 ranks, `qp_set_gradient`'s four
    explicit-route roots and the single-state root and Z on the
    contour-deformation and the pole-model route are bitwise the serial ones,
    the adjoints within `REL`;
  * on the same BLAS at 2, 3 and 8 ranks, the routes whose frequency rows feed
    a SUM over frequencies -- the dRPA energy and its adjoint, Wt(tau) and its
    adjoint -- within `SUM_REL` of serial.

Shown to fail: `owned_frequency_blocks` transforming only the rows a rank owns
(the code before this gate) fails all 20 distributed gates on the
row-count-sensitive BLAS and none of the 2 serial ones.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import ISDF_TILE_GB
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import partition, run_simulated
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.contour_deformation import \
    cd_screening_contraction_multi
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    frequency_blocks, owned_frequency_blocks)
from src.gradients.qp_space_time import qp_gradient_space_time, qp_set_gradient
from src.gradients.space_time_adjoint import (polarizability_tau,
                                              rpa_energy_and_adjoint,
                                              screened_interaction_tau,
                                              screened_interaction_tau_backward,
                                              sigma_transforms,
                                              three_index_slice)

NTAU, NFREQ = 8, 24
SIZES = [2, 3, 8, 16]
ROUTE_SIZES = [2, 3, 8]
#: The relative change per row of the model BLAS's tensordot: 3 rows against
#: 24 move chi0 by 3.1e-7, wc, the roots and every frequency sum with it.
ROW_COUNT_SKEW = 2.0 ** -26
#: The quasiparticle adjoints are partial sums over the frequency partition,
#: reduced.
REL = 1e-11
#: The frequency sums, reduced: at 2, 3 and 8 ranks on the row-count BLAS they
#: sit at most 2.4e-16 (E_c), 5.3e-16 (its adjoints), 3.6e-12 (Wt) and
#: 1.1e-10 (Wt's adjoint, a cancelling sum over the minimax weights) from
#: serial, and 3.2e-7, 1.3e-7, 1.6e-7 and 2.6e-8 at the least when a rank
#: transforms only its own rows.
SUM_REL = 1e-9
#: One block of all 24 frequencies (the default budget) and blocks of five,
#: where a block holds some of a rank's frequencies and some of another's.
FIVE_PER_BLOCK_GB = 5 * 3 * 84 ** 2 * 8 / 1e9
TILES = [None, FIVE_PER_BLOCK_GB]


class RowCountSensitiveNumpy:
    """numpy, save a tensordot whose rows depend on the call's row count."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def tensordot(a, b, axes=2):
        return (np.tensordot(a, b, axes)
                * (1.0 + np.shape(a)[0] * ROW_COUNT_SKEW))


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')[:2]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    states = [nocc - 2, nocc - 1, nocc, nocc + 1]
    proj = polarizability_tau(X, D, eps, nocc, grid, mu=mu)
    assert proj.shape[-1] == 84, 'FIVE_PER_BLOCK_GB is sized for naux = 84'
    return dict(X=X, D=D, eps=eps, nocc=nocc, grid=grid, nu=nu, wt=wt, mu=mu,
                states=states, weights=np.array([0.3, -1.1, 0.8, 0.45]),
                proj=proj, Bps=[three_index_slice(X, D, p) for p in states])


@pytest.fixture
def row_count_sensitive(monkeypatch):
    monkeypatch.setattr(ls_space_time, 'np', RowCountSensitiveNumpy())


def tile_of(tile):
    return {} if tile is None else dict(tile_gb=tile)


@pytest.mark.parametrize('tile', TILES)
def test_serial_path_is_the_serial_call(water, tile):
    w = water
    cosft = w['grid'].cosft_wt
    budget = ISDF_TILE_GB if tile is None else tile
    blocks = frequency_blocks(NFREQ, 84, budget, 3)
    if tile is not None:
        assert len(blocks) == 5
    got = list(owned_frequency_blocks(w['proj'], cosft, budget))
    assert [ks for ks, _ in got] == [list(range(k0, k1)) for k0, k1 in blocks]
    for (k0, k1), (_, blk) in zip(blocks, got):
        ks = list(range(k0, k1))
        assert np.array_equal(blk, np.tensordot(cosft[ks], w['proj'],
                                                axes=(1, 0)))


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('tile', TILES)
def test_owned_rows_are_the_serial_rows(water, row_count_sensitive, size, tile):
    w = water
    cosft = w['grid'].cosft_wt
    budget = ISDF_TILE_GB if tile is None else tile
    serial = {k: row
              for ks, blk in owned_frequency_blocks(w['proj'], cosft, budget)
              for k, row in zip(ks, blk)}
    wc_serial = cd_screening_contraction_multi(w['proj'], cosft, w['Bps'],
                                               **tile_of(tile))
    wc_sum = [np.zeros_like(a) for a in wc_serial]
    for rank in range(size):
        mine = partition(NFREQ, rank, size)
        got = list(owned_frequency_blocks(w['proj'], cosft, budget, mine))
        assert sorted(k for ks, _ in got for k in ks) == list(mine)
        for ks, blk in got:
            for k, row in zip(ks, blk):
                assert np.array_equal(row, serial[k]), (rank, k)
        for acc, part in zip(wc_sum, cd_screening_contraction_multi(
                w['proj'], cosft, w['Bps'], freq_indices=mine,
                **tile_of(tile))):
            acc += part
    for a, b in zip(wc_sum, wc_serial):
        assert np.array_equal(a, b)


def worst_rel(ref, got):
    """The largest relative deviation over a tuple of arrays."""
    return max(np.abs(np.asarray(a) - np.asarray(b)).max()
               / max(np.abs(np.asarray(a)).max(), 1e-300)
               for a, b in zip(ref, got))


@pytest.mark.parametrize('size', ROUTE_SIZES)
def test_qp_set_gradient_roots(water, row_count_sensitive, size):
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'], w['nu'], w['wt'],
            w['states'], w['weights'])
    kw = dict(mu=w['mu'], residue_route='explicit')
    serial = qp_set_gradient(*args, **kw)
    for roots, *adjoints in run_simulated(
            lambda comm: qp_set_gradient(*args, comm=comm, **kw), size):
        assert np.array_equal(roots, serial[0]), roots - serial[0]
        assert worst_rel(serial[1:], adjoints) <= REL


@pytest.mark.parametrize('size', ROUTE_SIZES)
@pytest.mark.parametrize('route', ['explicit', 'sop'])
def test_single_state_root_and_z(water, row_count_sensitive, size, route):
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'], w['nu'], w['wt'],
            w['nocc'] - 1)
    kw = dict(mu=w['mu'], residue_route=route)
    serial = qp_gradient_space_time(*args, **kw)
    for w_star, z, *adjoints in run_simulated(
            lambda comm: qp_gradient_space_time(*args, comm=comm, **kw), size):
        assert w_star == serial[0] and z == serial[1], (w_star - serial[0],
                                                        z - serial[1])
        assert worst_rel(serial[2:], adjoints) <= REL


@pytest.mark.parametrize('size', ROUTE_SIZES)
def test_frequency_sums(water, row_count_sensitive, size):
    w = water
    grid = w['grid']
    with warnings.catch_warnings():
        # 8 minimax points under-resolve this range; serial and split share it
        warnings.simplefilter('ignore')
        Ctw = sigma_transforms(w['eps'], w['nocc'], grid.tau_points,
                               grid.omega_points, grid.omega_points,
                               mu=w['mu'])[0]
    wt_bar = np.random.default_rng(5).normal(size=(Ctw.shape[0],)
                                             + w['proj'].shape[1:])
    rpa_args = (w['X'], w['D'], w['eps'], w['nocc'], grid)

    def routes(comm=None):
        e_c, *adjoints, _ = rpa_energy_and_adjoint(*rpa_args, mu=w['mu'],
                                                   comm=comm)
        return ([e_c], adjoints,
                [screened_interaction_tau(w['proj'], grid, Ctw, comm=comm)],
                [screened_interaction_tau_backward(wt_bar, w['proj'], grid, Ctw,
                                                   comm=comm)])

    serial = routes()
    for got in run_simulated(routes, size):
        for name, a, b in zip(('E_c', 'dRPA adjoints', 'Wt', 'Wt adjoint'),
                              serial, got):
            assert worst_rel(a, b) <= SUM_REL, (name, worst_rel(a, b))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
