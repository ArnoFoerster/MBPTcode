"""Primitive collectives of `mpi_grid`, under `simulated_world` at 2, 3 and 8
ranks so the uneven-block and empty-block edges (rank count exceeding item
count) are both exercised.

mpirun cannot start in the sandboxed test runner, so every check here runs
through `simulated_world`: ranks are threads of one process and the
collectives move real bytes through the barriers `SimulatedComm` implements,
not a no-op. tests/test_mpi_grid_distribution.py under mpirun is the wire
check for the MPI path these share.

Gated:
  * `contiguous_block` covers every index of n exactly once, and hands out
    empty blocks rather than raising when ranks outnumber items;
  * `allgather_blocks` of the `contiguous_block` row blocks reproduces the
    serial array BITWISE, for a 1-D array, an uneven split, an empty block,
    and a 3-D array with the partitioned axis leading;
  * `exchange_blocks` transposes a per-rank block layout: what rank j reads
    back for i is the block rank i sent it, for every ordered pair;
  * `broadcast` and `replicate` put rank 0's object -- a picklable one and an
    array -- on every rank;
  * `reduce_sum` agrees with `np.sum` over the ranks' contributions to 1e-12
    relative (summation order differs, so not bitwise).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.utils.mpi_grid import (allgather_blocks, broadcast,
                                     contiguous_block, exchange_blocks,
                                     reduce_sum, replicate, run_simulated)

SIZES = [2, 3, 8]


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('n', [0, 1, 5, 17, 40])
def test_contiguous_block_covers_every_index_exactly_once(size, n):
    blocks = [contiguous_block(n, r, size) for r in range(size)]
    covered = []
    for start, stop in blocks:
        assert start <= stop
        covered.extend(range(start, stop))
    assert covered == list(range(n))
    if size > n:
        empty = [b for b in blocks if b[0] == b[1]]
        assert len(empty) == size - n


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('shape', [(1,), (23,), (23, 4), (23, 3, 2)])
def test_allgather_blocks_is_bitwise_the_serial_array(size, shape):
    rng = np.random.default_rng(hash(('allgather', size, shape)) % 2**32)
    serial = rng.standard_normal(shape)

    def one_rank(comm):
        start, stop = contiguous_block(serial.shape[0], comm.Get_rank(),
                                       comm.Get_size())
        buf = np.zeros_like(serial)
        buf[start:stop] = serial[start:stop]
        allgather_blocks(buf, comm)
        return buf

    for out in run_simulated(one_rank, size):
        assert np.array_equal(out, serial)


@pytest.mark.parametrize('size', SIZES)
def test_exchange_blocks_is_the_transpose_of_a_block_layout(size):
    h, w = 3, 2
    # full[i, j]: the block rank i sends to rank j.
    full = np.arange(size * size * h * w, dtype=np.float64).reshape(size, size, h, w)

    def one_rank(comm):
        i = comm.Get_rank()
        send_blocks = [np.ascontiguousarray(full[i, j]) for j in range(size)]
        recv_shapes = [(h, w)] * size
        return exchange_blocks(send_blocks, recv_shapes, comm)

    for j, got in enumerate(run_simulated(one_rank, size)):
        for i in range(size):
            assert np.array_equal(got[i], full[i, j])


@pytest.mark.parametrize('size', SIZES)
def test_broadcast_gives_every_rank_root_object(size):
    payload = {'sentinel': 'root', 'value': [1, 2, 3]}

    def one_rank(comm):
        obj = payload if comm.Get_rank() == 0 else None
        out = broadcast(obj, comm, root=0)
        assert out is not payload or comm.Get_rank() == 0  # a copy off-root
        return out

    for out in run_simulated(one_rank, size):
        assert out == payload


@pytest.mark.parametrize('size', SIZES)
def test_replicate_gives_every_rank_root_array(size):
    root_array = np.arange(30, dtype=np.float64).reshape(5, 6)

    def one_rank(comm):
        rank = comm.Get_rank()
        arr = root_array.copy() if rank == 0 else np.full_like(root_array, -1.0)
        return replicate(arr, comm=comm, root=0)

    for out in run_simulated(one_rank, size):
        assert np.array_equal(out, root_array)


@pytest.mark.parametrize('size', SIZES)
def test_reduce_sum_matches_np_sum(size):
    rng = np.random.default_rng(hash(('reduce', size)) % 2**32)
    parts = rng.standard_normal((size, 7, 3))
    expected = np.sum(parts, axis=0)

    def one_rank(comm):
        buf = parts[comm.Get_rank()].copy()
        reduce_sum(buf, comm)
        return buf

    for out in run_simulated(one_rank, size):
        rel = np.abs(out - expected) / np.maximum(np.abs(expected), 1e-300)
        assert np.all(rel < 1e-12)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
