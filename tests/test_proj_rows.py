"""No MPI call carries more than `MPI_COUNT_MAX` elements, and a collective
cut into windows gives the bits of the one call.

mpi4py hands the library a C int as the count wherever the library lacks the
MPI-4 large-count routines -- Open MPI 5.0.x has none -- and raises
MPI_ERR_ARG past 2^31 - 1, so a count past it is not a slow message but a
failed run. `Base.utils.mpi_grid` therefore takes every collective in windows
of at most `MPI_COUNT_MAX` float64: the elementwise reductions and broadcasts
over the flat buffer, the row-typed gathers, broadcasts and exchanges in
rounds of row windows. Here the limit is set to `LIMIT` elements, far below
the arrays, and on 2, 3 and 8 simulated ranks every collective is checked to

  * give the bits the unchunked call gives (the sums are elementwise and the
    gathers verbatim, so a window is the same arithmetic on fewer elements),
  * make more than one call, none of them past the limit (read off the
    simulated communicator's own methods, which see each window), and
  * refuse, in `exchange_blocks`, a single Alltoallv past the limit rather
    than hand MPI a count it cannot take.

proj(tau) BY AUXILIARY ROWS (`LinearResponse.space_time.ProjRows`), what the
quasiparticle solves hold instead of the whole (ntau, naux, naux) array.
Water/cc-pVDZ Hartree-Fock, 8 tau points, 24 frequencies, at 2, 3 and 8
simulated ranks:

  * each rank's rows are bitwise the rows of the zero-padded Allreduce the
    solves ran before (`polarizability_tau` split over tau, `reduce_sum`);
  * `gather_slices` hands every rank the whole slices of its own list, in
    its order, whatever the other ranks ask for -- the serial slices bitwise;
  * every frequency a rank owns comes out of `ProjRows.blocks` as the serial
    ProjRows' chi0, bitwise, for one block of all frequencies and blocks of
    five; the real-frequency transform is the serial one on every rank; the
    adjoint folded into projbar rows is the serial fold's rows bitwise, and
    the tau slices the reverse sweep gathers are the serial slices;
  * the same, and the quasiparticle roots and pole strengths of both solves
    on the explicit, Laplace and pole-model routes, on a SHAPE-SENSITIVE BLAS:
    a tensordot whose result moves with the row count of its first operand
    and the row length of its second, as a GEMM's rows do on OpenBLAS, far
    past a last bit. Each transform is one auxiliary row per call, a shape
    no rank count changes, so nothing moves; a ProjRows transforming a rank's
    whole row block in one call (planted) moves every distributed gate and
    no serial one.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

import src.Base.utils.mpi_grid as mpi_grid
from src.Base.constants import ISDF_TILE_GB
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import (SimulatedComm, allgather_blocks,
                                     allgather_ranges, allgather_rows,
                                     broadcast_rows, contiguous_block,
                                     exchange_blocks, exchange_rows, lockstep,
                                     partition, reduce_max, reduce_sum,
                                     replicate, run_simulated)
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, polarizability_projected_rows)
from src.gradients.qp_space_time import qp_gradient_space_time, qp_set_gradient
from src.gradients.space_time_adjoint import polarizability_tau

SIZES = [2, 3, 8]
#: The artificial count limit: every test array is several windows of it, and
#: it holds a row of `ROW` doubles from each of eight ranks, the least a
#: row-typed round can move.
LIMIT = 50
ROW = 3
NROWS = 37
NTAU, NFREQ = 8, 24
#: One block of all 24 frequencies (the default budget) and blocks of five.
FIVE_PER_BLOCK_GB = 5 * 3 * 84 ** 2 * 8 / 1e9
TILES = [None, FIVE_PER_BLOCK_GB]
#: The shape-sensitive BLAS's relative skew per row of the first operand and
#: per element of a row of the second: far past a last bit, so neither a
#: Newton root nor a sum can hide a call whose shape followed the rank count.
ROW_SKEW, COLUMN_SKEW = 2.0 ** -30, 2.0 ** -34
#: What the 8-point grid's bare quadrature carries at the deeper states'
#: residue frequencies (4e-4): this is a test of bits, not of the Laplace
#: backend's accuracy, which needs the finer grids the chains build.
LAPLACE_TOL_OF_THE_GRID = 1e-2


class CallSizes:
    """Elements each simulated collective call carries, per method."""

    def __init__(self, monkeypatch):
        self.sizes = {}
        for name, count in (
                ('allreduce_sum', lambda buf: buf.size),
                ('allreduce_max', lambda buf: buf.size),
                ('bcast_into', lambda buf, root=0: buf.size),
                ('allgather_blocks', lambda buf, blocks: sum(
                    (s1 - s0) for s0, s1 in blocks)
                    * mpi_grid._row_length(buf)),
                ('allgather_ranges', lambda buf, ranges: sum(
                    (s1 - s0) for own in ranges for s0, s1 in own)
                    * mpi_grid._row_length(buf)),
                ('alltoall_blocks', lambda blocks: sum(
                    int(np.size(b)) for b in blocks))):
            self._wrap(monkeypatch, name, count)

    def _wrap(self, monkeypatch, name, count):
        method = getattr(SimulatedComm, name)
        sizes = self.sizes.setdefault(name, [])

        def recorded(comm, *args, **kwargs):
            sizes.append(count(*args, **kwargs))
            return method(comm, *args, **kwargs)
        monkeypatch.setattr(SimulatedComm, name, recorded)


def rank_array(rank, shape, salt=0):
    """A different float64 array on every rank."""
    return np.random.default_rng(100 * rank + salt).normal(size=shape)


def cyclic_ranges(n, size, block=4):
    """Block-cyclic row ranges of n rows, tile t on rank t % size."""
    out = [[] for _ in range(size)]
    for t, s0 in enumerate(range(0, n, block)):
        out[t % size].append((s0, min(s0 + block, n)))
    return out


def tau_slice(k):
    """Tau slice k of a (ntau, NROWS, ROW) sweep, the same on every rank."""
    return np.random.default_rng(1000 + k).normal(size=(NROWS, ROW))


def collectives(comm):
    """{name: array} of every windowed collective of `mpi_grid`, one rank."""
    size, rank = comm.Get_size(), comm.Get_rank()
    out = {}
    out['reduce_sum'] = reduce_sum(rank_array(rank, (NROWS, ROW, 2)), comm)
    out['reduce_max'] = reduce_max(rank_array(rank, (NROWS, ROW), 1), comm)
    blocks = [contiguous_block(NROWS, r, size) for r in range(size)]
    r0, r1 = blocks[rank]
    for name, gather in (('allgather_blocks', allgather_blocks),
                         ('allgather_rows', allgather_rows)):
        a = np.full((NROWS, ROW), np.nan)
        a[r0:r1] = rank_array(rank, (r1 - r0, ROW), 2)
        out[name] = gather(a, comm)
    ranges = cyclic_ranges(NROWS, size)
    a = np.full((NROWS, ROW), np.nan)
    for s0, s1 in ranges[rank]:
        a[s0:s1] = rank_array(rank, (s1 - s0, ROW), 3 + s0)
    out['allgather_ranges'] = allgather_ranges(a, ranges, comm)
    out['broadcast_rows'] = broadcast_rows(rank_array(rank, (NROWS, ROW), 4),
                                           size - 1, comm)
    locked = lockstep({'x': rank_array(rank, (NROWS, ROW), 5),
                       'y': rank_array(rank, (7,), 6)}, comm)
    out['lockstep.x'], out['lockstep.y'] = locked['x'], locked['y']
    out['replicate'] = replicate(rank_array(rank, (NROWS, 2), 7), comm=comm)
    # tau slices owned round-robin, turned into every rank's rows of each
    ntau = 5
    rows = np.zeros((ntau * (r1 - r0), ROW))
    for j in range(-(-ntau // size)):
        k = rank + j * size
        send = tau_slice(k) if k < ntau else np.empty((0, ROW))
        sends = [blocks[s] if k < ntau else (0, 0) for s in range(size)]
        recvs = [((r + j * size) * (r1 - r0), (r + j * size + 1) * (r1 - r0))
                 if r + j * size < ntau else (0, 0) for r in range(size)]
        exchange_rows(send, sends, rows, recvs, comm)
    out['exchange_rows'] = rows
    return out


#: The simulated communicator's methods a windowed collective reaches.
WINDOWED = ('allreduce_sum', 'allreduce_max', 'bcast_into',
            'allgather_ranges', 'alltoall_blocks')


@pytest.mark.parametrize('size', SIZES)
def test_windows_give_the_bits_of_one_call(size, monkeypatch):
    whole = run_simulated(collectives, size)
    monkeypatch.setattr(mpi_grid, 'MPI_COUNT_MAX', LIMIT)
    calls = CallSizes(monkeypatch)
    cut = run_simulated(collectives, size)
    for rank, (a, b) in enumerate(zip(whole, cut)):
        assert sorted(a) == sorted(b)
        for name in a:
            assert a[name].tobytes() == b[name].tobytes(), (size, rank, name)
    for name in WINDOWED:
        sizes = calls.sizes[name]
        # more calls than one per collective and rank: the windows were cut
        assert len(sizes) > 2 * size, (name, len(sizes))
        assert max(sizes) <= LIMIT, (name, max(sizes))


@pytest.mark.parametrize('size', SIZES)
def test_the_one_call_results(size):
    """What the unchunked collectives must return, once."""
    blocks = [contiguous_block(NROWS, r, size) for r in range(size)]
    gathered = np.concatenate([rank_array(r, (s1 - s0, ROW), 2)
                               for r, (s0, s1) in enumerate(blocks)])
    total = rank_array(0, (NROWS, ROW, 2))
    for r in range(1, size):
        total = total + rank_array(r, (NROWS, ROW, 2))
    for rank, res in enumerate(run_simulated(collectives, size)):
        assert np.array_equal(res['allgather_rows'], gathered)
        assert np.array_equal(res['allgather_blocks'], gathered)
        assert np.array_equal(res['reduce_sum'], total)
        assert np.array_equal(res['broadcast_rows'],
                              rank_array(size - 1, (NROWS, ROW), 4))
        r0, r1 = blocks[rank]
        assert np.array_equal(res['exchange_rows'], np.concatenate(
            [tau_slice(k)[r0:r1] for k in range(5)]))


def test_exchange_blocks_refuses_one_call_past_the_limit(monkeypatch):
    monkeypatch.setattr(mpi_grid, 'MPI_COUNT_MAX', LIMIT)

    def one(comm):
        size = comm.Get_size()
        send = [np.zeros((LIMIT // size + 1, 1)) for _ in range(size)]
        return exchange_blocks(send, [b.shape for b in send], comm)

    with pytest.raises(ValueError, match='MPI_COUNT_MAX'):
        run_simulated(one, 2)


# ---------------------------------------------------------------- proj rows

class ShapeSensitiveNumpy:
    """numpy, save a tensordot whose result follows the call's shape."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def tensordot(a, b, axes=2):
        out = np.tensordot(a, b, axes)
        if not isinstance(b, np.ndarray):
            return out
        rows = np.shape(a)[0] if np.ndim(a) > 1 else 1
        width = np.size(b) // max(np.shape(b)[0], 1)
        return out * (1.0 + rows * ROW_SKEW + width * COLUMN_SKEW)


def one_call_per_block(c, rows):
    """`transform_rows` as one call over a rank's whole row block, through the
    module's BLAS: its shape follows the rank count."""
    return ls_space_time.np.tensordot(c, rows, axes=(c.ndim - 1, 0))


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
    assert D.shape[1] == 84, 'FIVE_PER_BLOCK_GB is sized for naux = 84'
    return dict(X=X, D=D, eps=eps, nocc=nocc, grid=grid, nu=nu, wt=wt, mu=mu,
                states=[nocc - 3, nocc - 2, nocc - 1, nocc, nocc + 1],
                weights=np.array([0.2, 0.3, -1.1, 0.8, 0.45]))


def chi0_bar_of(k, naux):
    """A fixed adjoint for frequency k, the same on every rank."""
    return np.random.default_rng(500 + k).normal(size=(naux, naux))


def rows_routes(w, tile, comm=None):
    """Everything a ProjRows hands its consumers, on this rank."""
    proj = polarizability_projected_rows(w['X'], w['D'], w['eps'], w['nocc'],
                                         w['grid'].tau_points, mu=w['mu'],
                                         comm=comm)
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    nu_mine = partition(NFREQ, rank, size) if size > 1 else None
    tau_mine = partition(NTAU, rank, size) if size > 1 else None
    cosft = w['grid'].cosft_wt
    chi0, proj_bar = {}, proj.zeros_like()
    for fb in proj.blocks(cosft, ISDF_TILE_GB if tile is None else tile,
                          nu_mine):
        for m, k in enumerate(fb.ks):
            chi0[k] = fb.chi0[m].copy()
            fb.chi0[m] = chi0_bar_of(k, proj.naux)
        proj_bar.fold(cosft, fb)
    slices = {k: slab.copy() for k, slab in proj_bar.tau_slices(tau_mine)}
    # each rank asks for its own list, of its own length, repeats and all
    wants = [(3 * rank + 5 * j) % NTAU for j in range(rank % 3 + 1)]
    gathered = [(k, slab.copy()) for k, slab in proj.gather_slices(wants)]
    c = np.cos(np.arange(NTAU) + 0.5)
    return dict(rows=proj.rows.copy(), r0=proj.r0, chi0=chi0,
                bar_rows=proj_bar.rows.copy(), slices=slices,
                real=np.tensordot(c, proj, axes=(0, 0)), wants=wants,
                gathered=gathered)


def distributed_rows(w, tile, size):
    return run_simulated(lambda comm: rows_routes(w, tile, comm), size)


def same_as_serial(serial, ranks):
    """(name, rank) of every piece a rank holds that is not serial's bits."""
    naux = serial['rows'].shape[1]
    moved = []
    for rank, got in enumerate(ranks):
        r0, r1 = got['r0'], got['r0'] + got['rows'].shape[1]
        checks = [('rows', got['rows'], serial['rows'][:, r0:r1]),
                  ('bar_rows', got['bar_rows'], serial['bar_rows'][:, r0:r1]),
                  ('real', got['real'], serial['real'])]
        checks += [(f'chi0[{k}]', v, serial['chi0'][k])
                   for k, v in got['chi0'].items()]
        checks += [(f'slice[{k}]', v, serial['slices'][k])
                   for k, v in got['slices'].items()]
        assert [k for k, _ in got['gathered']] == got['wants']
        checks += [(f'gathered[{k}]', v, serial['rows'][k])
                   for k, v in got['gathered']]
        moved += [(name, rank) for name, a, b in checks
                  if a.tobytes() != b.tobytes()]
        assert got['rows'].shape == (NTAU, r1 - r0, naux)
    return moved


@pytest.mark.parametrize('size', SIZES)
def test_rows_are_the_allreduced_rows(water, size):
    """Each rank's rows against the rows of the zero-padded Allreduce."""
    w = water

    def both(comm):
        rank = comm.Get_rank()
        proj = polarizability_projected_rows(w['X'], w['D'], w['eps'],
                                             w['nocc'], w['grid'].tau_points,
                                             mu=w['mu'], comm=comm)
        whole = polarizability_tau(w['X'], w['D'], w['eps'], w['nocc'],
                                   w['grid'], mu=w['mu'],
                                   tau_indices=partition(NTAU, rank, size))
        reduce_sum(whole, comm)
        return proj.rows, whole[:, proj.r0:proj.r1], proj.nbytes

    naux = w['D'].shape[1]
    for rows, cut, held in run_simulated(both, size):
        assert rows.tobytes() == cut.tobytes()
        assert held <= NTAU * -(-naux // size) * naux * 8


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('tile', TILES)
def test_rows_give_the_serial_bits(water, size, tile):
    serial = rows_routes(water, tile)
    ranks = distributed_rows(water, tile, size)
    owned = sorted(k for got in ranks for k in got['chi0'])
    swept = sorted(k for got in ranks for k in got['slices'])
    assert owned == list(range(NFREQ)) and swept == list(range(NTAU))
    assert same_as_serial(serial, ranks) == []


def test_serial_rows_are_the_whole_sweep(water):
    """One rank holds every row: the sweep itself, and chi0 the whole call's
    to rounding on any BLAS."""
    w = water
    serial = rows_routes(w, None)
    whole = polarizability_tau(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                               mu=w['mu'])
    assert serial['rows'].tobytes() == whole.tobytes()
    blk = np.tensordot(w['grid'].cosft_wt, whole, axes=(1, 0))
    for k, chi0 in serial['chi0'].items():
        assert np.allclose(chi0, blk[k], rtol=1e-13, atol=1e-15)
    with pytest.raises(TypeError):
        np.asarray(ProjRows(whole, whole.shape[-1]))
    with pytest.raises(TypeError):
        np.zeros_like(ProjRows(whole, whole.shape[-1]))


@pytest.fixture
def shape_sensitive(monkeypatch):
    monkeypatch.setattr(ls_space_time, 'np', ShapeSensitiveNumpy())


def solves(w, comm=None):
    """(roots and Z of both solves on three routes) on this rank."""
    out = {}
    for route in ('explicit', 'laplace', 'sop'):
        ro = {}
        kw = dict(mu=w['mu'], residue_route=route, comm=comm,
                  laplace_tol=LAPLACE_TOL_OF_THE_GRID)
        got = qp_set_gradient(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                              w['nu'], w['wt'], w['states'], w['weights'],
                              route_out=ro, **kw)
        out[f'set.{route}'] = np.concatenate([got[0], ro['z']])
        single = qp_gradient_space_time(w['X'], w['D'], w['eps'], w['nocc'],
                                        w['grid'], w['nu'], w['wt'],
                                        w['nocc'] - 3, **kw)
        out[f'single.{route}'] = np.array(single[:2])
    return out


@pytest.mark.parametrize('size', SIZES)
def test_shape_sensitive_blas_moves_no_rank(water, shape_sensitive, size):
    serial = rows_routes(water, FIVE_PER_BLOCK_GB)
    assert same_as_serial(serial, distributed_rows(
        water, FIVE_PER_BLOCK_GB, size)) == []
    roots = solves(water)
    for got in run_simulated(lambda comm: solves(water, comm), size):
        for name, value in roots.items():
            assert got[name].tobytes() == value.tobytes(), (size, name)


def test_a_rank_count_shaped_transform_is_caught(water, shape_sensitive,
                                                 monkeypatch):
    """Planted: a rank's whole row block in one call. Every distributed rank
    moves; the serial reference cannot tell."""
    monkeypatch.setattr(ls_space_time, 'transform_rows', one_call_per_block)
    serial = rows_routes(water, FIVE_PER_BLOCK_GB)
    for size in SIZES:
        moved = same_as_serial(serial, distributed_rows(
            water, FIVE_PER_BLOCK_GB, size))
        assert {rank for _, rank in moved} == set(range(size)), (size, moved)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
