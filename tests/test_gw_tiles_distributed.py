"""The quasiparticle solve's M^2 sweeps on fixed grid tiles: no rank holds a
factor whole, what a rank holds falls with the rank count, and the sums the
code adds in tile order are the same bits at every rank count.

Gated over 1, 2 and 3 simulated ranks (`run_simulated`, threads of this
process with the real collectives) on water/cc-pVDZ Hartree-Fock, with a
tile budget small enough that the grid is cut into tens of tiles:

  * the tile layer (`GridTileRows.from_rows`, `fetch`, `stream`): rows cut
    from every rank's contiguous block, in the grid's order or a permuted
    one, arrive verbatim, the same as tiles cut from the whole array;
  * W(0) (`static_screening_matrix`): the one-rank bits at every rank count,
    from whole factors and from `SlicedFactors` alike (beside their own D
    or a whole D in another gauge), and the old realization's to a few ulp;
    a running sum moved as it is handed between ranks breaks the bits (the
    planted defect);
  * the self-energy's branch sums (`self_energy_branch_sums_tiles`): the
    same bits at every rank count, sliced or whole, with and without the
    far-field screen (which drops tile pairs on a stretched water dimer),
    and the AO-pair sweep's (`SigmaPairs`) to a few ulp;
  * THE MEMORY ASSERTION: the tiles a rank holds (`tile_ledger`, read off
    the arrays) are about 1/ranks of the factors -- at most one tile more
    than its share -- and a sweep's pass is bounded by the budget, not by
    the grid; the in-core quasiparticle window on slices gathers no factor
    whole (tests/test_sliced_factors.py counts the gathers);
  * the in-core quasiparticle window on slices is the window on whole
    factors at the same rank count, bitwise, and every rank's is rank 0's.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.factorization.sliced_factors import (GridTileRows, SlicedFactors,
                                                   column_tiles, tile_ledger)
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import contiguous_block
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW import imaginary_time
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse import space_time
from src.SingleReference.LinearResponse.space_time import ProjRows
from tests.isdf_test_grid import g1
from tests.pins import RECORDING_BLAS, matches

SIZES = [1, 2, 3]
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Two waters 12 Angstrom apart: tile pairs across the gap are far.
DIMER = WATER + '; O 0 0 12.1173; H 0 0.7572 11.5308; H 0 -0.7572 11.5308'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
#: GB per tile pass: a few tens of grid rows per tile on these grids.
TILE_GB = 1e-4
NUM = Numerics().derive('tens of grid tiles on a small molecule',
                        layout={'tile_gb': TILE_GB})


def _factors(atom):
    mol = gto.M(atom=atom, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    return mol, mf, mol.nelectron // 2, separable_factors(
        mf, mol, auxbasis=AUXBASIS, counts=g1(mol))


@pytest.fixture(scope='module')
def water():
    return _factors(WATER)


@pytest.fixture(scope='module')
def dimer():
    return _factors(DIMER)


def sliced(factors, comm):
    """This rank's rows of whole factors, as `SlicedFactors`; the whole
    factors themselves on one rank."""
    X, D, X_ao, coords = factors
    if comm.Get_size() == 1:
        return factors
    r0, r1 = contiguous_block(len(coords), comm.Get_rank(), comm.Get_size())
    return SlicedFactors(X[r0:r1].copy(), D[r0:r1].copy(),
                         X_ao[r0:r1].copy(), coords, comm)


def few_ulp(a, b):
    return matches(np.asarray(a), np.asarray(b), RECORDING_BLAS, unit='1',
                   quantity='root')


@pytest.mark.parametrize('size', SIZES)
def test_tile_layer_moves_rows_verbatim(size):
    rng = np.random.default_rng(7)
    A, B = rng.normal(size=(53, 6)), rng.normal(size=(53, 2))
    order = rng.permutation(53)

    def rank(comm):
        r0, r1 = contiguous_block(53, comm.Get_rank(), size)
        many = comm if size > 1 else None
        out = []
        for o in (None, order):
            want = np.hstack([A, B]) if o is None else np.hstack([A, B])[o]
            tiles = column_tiles([A[r0:r1], B[r0:r1]], 53, 5, many, o,
                                 whole=size == 1)
            cut = GridTileRows.from_columns([A, B], 5, many, o)
            asks = [(comm.Get_rank() + 2 * j) % len(tiles.bounds)
                    for j in range(comm.Get_rank() + 1)]
            got = tiles.fetch(asks)
            out.append(np.array_equal(tiles.rows, cut.rows)
                       and np.array_equal(tiles.gather(), want)
                       and all(np.array_equal(got[t],
                                              want[slice(*tiles.bounds[t])])
                               for t in asks)
                       and all(np.array_equal(v, want[slice(*tiles.bounds[t])])
                               for t, v in tiles.stream()))
        return out

    assert all(all(r) for r in run_simulated(rank, size))


def test_static_w_is_the_one_rank_bits(water):
    mol, mf, nocc, factors = water
    X, D = factors[:2]
    one = space_time.static_screening_matrix(X, D, mf.mo_energy, nocc,
                                             comm=None,
                                             tile_memory_gb=TILE_GB)
    grid = space_time.static_screening_grid(mf.mo_energy, nocc)
    old = space_time.dyson_screening(space_time.chi0_imaginary_frequency(
        X, D, mf.mo_energy, nocc, grid)[0])
    assert few_ulp(one, old)

    def rank(comm, cut):
        f = sliced(factors, comm) if cut else X
        with tile_ledger() as led:
            W = space_time.static_screening_matrix(
                f, None if cut else D, mf.mo_energy, nocc,
                tile_memory_gb=TILE_GB)
        return W, dict(led)

    whole = (X.nbytes + D.nbytes)
    for size in SIZES[1:]:
        for cut in (False, True):
            res = run_simulated(rank, size, cut)
            assert all(np.array_equal(W, one) for W, _ in res), (size, cut)
            for _, led in res:
                assert led['tiles'] < whole / size * 1.5, (size, led)


def test_static_w_reads_a_whole_d_beside_sliced_factors(water):
    """A reaction field screens the bare factor D T beside sliced factors
    that hold the dressed D (`reaction_field_adjoint.static_screening`): W(0)
    is the one-rank W of (X, D T), not of the factors' own D."""
    mol, mf, nocc, factors = water
    X, D = factors[:2]
    T = np.eye(D.shape[1]) + 0.05 * np.random.default_rng(5).normal(
        size=(D.shape[1],) * 2)
    bare = D @ T
    one = space_time.static_screening_matrix(X, bare, mf.mo_energy, nocc,
                                             comm=None,
                                             tile_memory_gb=TILE_GB)
    for size in SIZES[1:]:
        res = run_simulated(lambda comm: space_time.static_screening_matrix(
            sliced(factors, comm), bare, mf.mo_energy, nocc,
            tile_memory_gb=TILE_GB), size)
        assert all(np.array_equal(W, one) for W in res), size


def test_static_w_fails_a_moved_running_sum(water, monkeypatch):
    """A running sum moved by a few thousand ulp as it is handed from one
    rank to the next is no longer the one-rank W: the chain carries the sum
    and nothing re-derives it. Two ranks more than tau points split every
    point's tiles over the ranks, so hand-overs happen."""
    mol, mf, nocc, factors = water
    X, D = factors[:2]
    one = space_time.static_screening_matrix(X, D, mf.mo_energy, nocc,
                                             comm=None,
                                             tile_memory_gb=TILE_GB)
    real, moved = space_time._hand_over_round, []

    def plant(chains, h, totals, rank, comm, naux):
        got = real(chains, h, totals, rank, comm, naux)
        for a in got.values():
            a.flat[np.argmax(np.abs(a))] *= 1 + 1e-12
            moved.append(rank)
        return got

    monkeypatch.setattr(space_time, '_hand_over_round', plant)
    res = run_simulated(lambda comm: space_time.static_screening_matrix(
        X, D, mf.mo_energy, nocc, tile_memory_gb=TILE_GB),
        space_time.static_screening_grid(mf.mo_energy, nocc).ntau + 2)
    assert moved, 'no running sum was handed over'
    assert not all(np.array_equal(W, one) for W in res)


def sigma_sums(comm, factors, mf, nocc, Wt, tau, cut, slice_them):
    eps = mf.mo_energy
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    naux = factors[1].shape[1]
    b0, b1 = contiguous_block(naux, comm.Get_rank(), comm.Get_size())
    many = comm if comm.Get_size() > 1 else None
    a, d = ((sliced(factors, comm), None)
            if slice_them and comm.Get_size() > 1 else (factors[2], factors[1]))
    with tile_ledger() as led:
        tiled, far = imaginary_time.sigma_factor_tiles(
            a, d, mf.mo_coeff, eps, nocc, mu, comm, coords=factors[3],
            screen_r_cut=cut, numerics=NUM)
        out = imaginary_time.self_energy_branch_sums_tiles(
            tiled, ProjRows(Wt[:, b0:b1].copy(), naux, many),
            np.arange(len(eps)), tau, far)
    return np.array(out), dict(led), 0 if far is None else int(far.sum())


def model_wt(naux, ntau=5):
    rng = np.random.default_rng(3)
    Wt = rng.normal(size=(ntau, naux, naux)) * 1e-2
    return Wt + Wt.transpose(0, 2, 1), np.linspace(0.1, 3.0, ntau)


@pytest.mark.parametrize('cut', [None, 3.0])
def test_self_energy_sums_are_the_same_bits_at_every_rank_count(dimer, cut):
    mol, mf, nocc, factors = dimer
    Wt, tau = model_wt(factors[1].shape[1])
    one, _, nfar = run_simulated(sigma_sums, 1, factors, mf, nocc, Wt, tau,
                                 cut, False)[0]
    if cut is None:
        eps = mf.mo_energy
        pairs = imaginary_time.SigmaPairs(
            factors[2], factors[1], mf.mo_coeff, eps, nocc,
            np.arange(len(eps)), 0.5 * (eps[nocc - 1] + eps[nocc]))
        old = imaginary_time.self_energy_branch_sums(
            pairs, ProjRows(Wt, Wt.shape[1]), tau)
        assert few_ulp(one, np.array(old))
    else:
        assert nfar > 0, 'the screen dropped no tile pair'
    whole = sum(a.nbytes for a in factors[1:3])
    for size in SIZES[1:]:
        for slice_them in (False, True):
            res = run_simulated(sigma_sums, size, factors, mf, nocc, Wt, tau,
                                cut, slice_them)
            for out, led, _ in res:
                assert np.array_equal(out, one), (size, slice_them)
                assert led['tiles'] < whole / size * 1.5, (size, led)
                assert led['pass'] < 3 * TILE_GB * 1e9, led


def test_quasiparticle_window_on_tiles(water):
    mol, mf, nocc, factors = water
    num = NUM
    window = np.arange(nocc - 2, nocc + 2)

    def rank(comm, cut):
        f = sliced(factors, comm) if cut else factors
        extras = {}
        with tile_ledger() as led:
            qp = solve_qp_energy_space_time(mf, mol, nocc, window,
                                            numerics=num, factors=f,
                                            extras=extras)
        return qp, extras['w_static'], dict(led)

    for size in SIZES[1:]:
        whole, part = run_simulated(rank, size, False), run_simulated(
            rank, size, True)
        for r, ((qw, ww, _), (qs, ws, led)) in enumerate(zip(whole, part)):
            assert np.array_equal(qs, qw) and np.array_equal(ws, ww), r
            assert np.array_equal(qs, part[0][0]), f'rank {r} != rank 0'
            assert led['tiles'] < (factors[0].nbytes + factors[1].nbytes) \
                / size * 1.5, (size, led)
