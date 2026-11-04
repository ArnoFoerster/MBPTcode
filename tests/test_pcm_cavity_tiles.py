"""The continuum's surface potential on tiles, split over the SCF's ranks.

pyscf's `PCM._get_v` and `PCM._get_vmat` evaluate (nao, nao, k) blocks of
three-centre integrals with k sized by `max_memory` alone, so a large memory
share puts the whole (nao, nao, n_surf) array in one block on every rank, in
every SCF cycle (16 GB for a 62-atom emitter in aug-cc-pVDZ).
`pcm_factorization.tile_cavity` evaluates both on `cavity_tiles`, and inside
`distributed_fock` each rank only its share of them (`owned_blocks`).

Gates (water/cc-pVDZ in toluene, the tile budget lowered to a few points
where the gate needs several tiles):
  * at nao 2000 and 2600 surface points (a 118-atom emitter in aug-cc-pVDZ),
    on one rank and over 8, every tile holds at most LIBCINT_BLOCK_LIMIT
    integrals and at most PCM_CAVITY_BLOCK_GB, there are at least
    PCM_CAVITY_MIN_TILES of them, and the ranks' tiles cover the cavity once;
  * a cavity inside one tile is evaluated in pyscf's own block, bit for bit;
  * on tiles of a few points the potential and the reaction-field matrix are
    pyscf's to round-off, and no block exceeds the tile budget;
  * over 2 and 3 simulated ranks with the communicator attached, each rank
    evaluates only its tiles (together the cavity once), and every rank
    returns the one-rank result bit for bit;
  * a rank keeps its tiles for the next call at the same geometry, within
    PCM_CAVITY_CACHE_GB (bit for bit what it evaluates afresh with nothing
    kept), and evaluates them again once the atoms move;
  * a solvated SCF over 3 ranks evaluates each cycle's cavity once in total,
    not once per rank, ends on the serial mean field, and leaves no
    communicator on the continuum after the block.
"""
import threading
import warnings

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf import solvent as pyscf_solvent
from pyscf.df import incore
from pyscf.solvent.pcm import PCM

from src.Base.environment_seam import attach_cavity_comm, cavity_comm
from src.Base.factorization.distributed_df import distributed_mean_field
from src.Base.numerics.constants import (BYTES_PER_GB, PCM_CAVITY_BLOCK_GB, PCM_CAVITY_MIN_TILES,
                                         RANK_SPLIT_ENERGY_TOL, SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.utils.parallel import mpi_grid
from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.environments import pcm_factorization
from src.environments.pcm_derivatives import LIBCINT_BLOCK_LIMIT, owned_blocks
from src.environments.pcm_factorization import _cavity_share, cavity_tiles, factorize_once
from tests.molecules import WATER_DISTORTED

WATER = WATER_DISTORTED

#: Surface points per tile where a gate lowers the budget.
PER_TILE = 7


def water():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def continuum(mol):
    pcm = PCM(mol)
    pcm.eps = 2.38
    pcm.build()
    return pcm


def density(mol):
    rng = np.random.default_rng(7)
    a = rng.standard_normal((mol.nao, mol.nao))
    return (a + a.T)[None] / mol.nao


def small_tiles(monkeypatch, nao):
    """Lower the tile budget to PER_TILE surface points."""
    monkeypatch.setattr(pcm_factorization, 'PCM_CAVITY_BLOCK_GB', PER_TILE * 8.0 * nao * nao / 1e9)


class Census:
    """The surface points each thread asks libcint for, per call."""

    def __init__(self, monkeypatch):
        self.points, self._lock = {}, threading.Lock()
        own = incore.aux_e2

        def counted(mol, auxmol, *args, **kwargs):
            with self._lock:
                self.points.setdefault(threading.current_thread().name, []).append(auxmol.natm)
            return own(mol, auxmol, *args, **kwargs)
        monkeypatch.setattr(incore, 'aux_e2', counted)

    def total(self):
        return sum(sum(v) for v in self.points.values())


class _Ranks:
    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


@pytest.mark.parametrize('size', [1, 8])
def test_tiles_fit_libcint_at_emitter_size(size):
    nao, ngrids = 2000, 2600
    assert len(cavity_tiles(nao, ngrids)) >= PCM_CAVITY_MIN_TILES
    covered = []
    for rank in range(size):
        cavity = type('Cavity', (), {})()
        cavity.surface = {'grid_coords': np.zeros((ngrids, 3))}
        attach_cavity_comm(cavity, _Ranks(rank, size))
        tiles, _ = _cavity_share(cavity, nao)
        for _, (p0, p1) in tiles:
            assert nao * nao * (p1 - p0) <= LIBCINT_BLOCK_LIMIT
            assert 8.0 * nao * nao * (p1 - p0) <= PCM_CAVITY_BLOCK_GB * BYTES_PER_GB
            covered.extend(range(p0, p1))
    assert sorted(covered) == list(range(ngrids))


def test_one_tile_is_pyscf_bit_for_bit():
    mol = water()
    reference = continuum(mol)
    dms = density(mol)
    v_ref = PCM._get_v(reference, dms)
    q = np.random.default_rng(3).standard_normal(v_ref.shape)
    tiled = factorize_once(continuum(mol))
    assert len(cavity_tiles(mol.nao, v_ref.shape[1])) == 1
    np.testing.assert_array_equal(tiled._get_v(dms), v_ref)
    np.testing.assert_array_equal(tiled._get_vmat(q), PCM._get_vmat(reference, q))


def test_tiles_are_pyscf_and_bounded(monkeypatch):
    mol = water()
    reference = continuum(mol)
    dms = density(mol)
    v_ref = PCM._get_v(reference, dms)
    q = np.random.default_rng(3).standard_normal(v_ref.shape)
    vmat_ref = PCM._get_vmat(reference, q)

    ngrids = reference.surface['grid_coords'].shape[0]
    small_tiles(monkeypatch, mol.nao)
    census = Census(monkeypatch)
    tiled = factorize_once(continuum(mol))
    v = tiled._get_v(dms)
    vmat = tiled._get_vmat(q)

    np.testing.assert_allclose(v, v_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(vmat, vmat_ref, rtol=1e-12, atol=1e-13)
    blocks = [n for calls in census.points.values() for n in calls]
    # the potential evaluates every tile and keeps it; the matrix reads them
    assert max(blocks) <= PER_TILE and len(blocks) >= ngrids // PER_TILE
    assert census.total() == ngrids


@pytest.mark.parametrize('size', [2, 3])
def test_ranks_evaluate_their_tiles(size, monkeypatch):
    mol = water()
    dms = density(mol)
    small_tiles(monkeypatch, mol.nao)
    with distributed(None):
        serial = factorize_once(continuum(mol))
        v_serial = serial._get_v(dms)
        q = np.random.default_rng(3).standard_normal(v_serial.shape)
        vmat_serial = serial._get_vmat(q)
    ngrids = serial.surface['grid_coords'].shape[0]
    tiles = cavity_tiles(mol.nao, ngrids)
    census = Census(monkeypatch)

    def one_rank(comm):
        pcm = factorize_once(continuum(water()))
        attach_cavity_comm(pcm, comm)
        return threading.current_thread().name, pcm._get_v(dms), pcm._get_vmat(q)

    out = run_simulated(one_rank, size)
    for rank, (name, v, vmat) in enumerate(out):
        mine = sum(tiles[k][1] - tiles[k][0]
                   for k in owned_blocks(len(tiles), _Ranks(rank, size)))
        assert sum(census.points[name]) == mine
        assert max(census.points[name]) <= PER_TILE
        np.testing.assert_array_equal(v, v_serial)
        np.testing.assert_array_equal(vmat, vmat_serial)
    assert census.total() == ngrids


def solvated_mean_field(mol):
    warnings.simplefilter('ignore')
    mf = pyscf_solvent.PCM(dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-jkfit'))
    mf.with_solvent.eps = 2.38
    factorize_once(mf.with_solvent)
    mf.conv_tol, mf.conv_tol_grad = SCF_DIFFERENTIABLE_CONV_TOL, SCF_DIFFERENTIABLE_GRAD_TOL
    return mf


def test_scf_over_ranks_evaluates_the_cavity_once(monkeypatch):
    size = 3
    small_tiles(monkeypatch, water().nao)
    with distributed(None):
        serial = solvated_mean_field(water()).run()
    ngrids = serial.with_solvent.surface['grid_coords'].shape[0]
    tiles = cavity_tiles(water().nao, ngrids)
    census = Census(monkeypatch)

    def one_rank(comm):
        mf = solvated_mean_field(water())
        distributed_mean_field(mf)
        return (threading.current_thread().name, mf.e_tot, mf.converged,
                cavity_comm(mf.with_solvent) is not None)

    out = run_simulated(one_rank, size)
    shares = []
    for rank, (name, e_tot, converged, attached) in enumerate(out):
        assert converged and not attached
        assert abs(e_tot - serial.e_tot) < RANK_SPLIT_ENERGY_TOL
        shares.append(sum(tiles[k][1] - tiles[k][0]
                          for k in owned_blocks(len(tiles), _Ranks(rank, size))))
        assert max(census.points[name]) <= PER_TILE
    # every rank evaluates the same number of cavity passes, each over its own tiles
    passes = {sum(census.points[name]) / share for (name, *_), share in zip(out, shares)}
    assert len(passes) == 1 and float(passes.pop()).is_integer()


def test_tiles_are_kept_between_calls_at_one_geometry(monkeypatch):
    mol = water()
    dms = density(mol)
    small_tiles(monkeypatch, mol.nao)
    census = Census(monkeypatch)
    pcm = factorize_once(continuum(mol))
    ngrids = pcm.surface['grid_coords'].shape[0]
    v = pcm._get_v(dms)
    q = np.random.default_rng(3).standard_normal(v.shape)
    vmat = pcm._get_vmat(q)
    for _ in range(3):
        np.testing.assert_array_equal(pcm._get_v(dms), v)
        np.testing.assert_array_equal(pcm._get_vmat(q), vmat)
    assert census.total() == ngrids

    # nothing kept: every call evaluates afresh, to the same bits
    monkeypatch.setattr(pcm_factorization, 'PCM_CAVITY_CACHE_GB', 0.0)
    fresh = factorize_once(continuum(mol))
    np.testing.assert_array_equal(fresh._get_v(dms), v)
    np.testing.assert_array_equal(fresh._get_vmat(q), vmat)
    assert census.total() == 3 * ngrids

    # the atoms move: the kept tiles belong to the old cavity
    monkeypatch.setattr(pcm_factorization, 'PCM_CAVITY_CACHE_GB', 2.0)
    moved = mol.set_geom_(mol.atom_coords() + 0.01, unit='Bohr', inplace=False)
    pcm.reset(moved)
    pcm.build()
    n_moved = pcm.surface['grid_coords'].shape[0]
    pcm._get_v(dms)
    assert census.total() == 3 * ngrids + n_moved


@pytest.mark.parametrize('size', [2, 3])
def test_reaction_field_matrix_moves_slices_not_every_tile(size, monkeypatch):
    """Over ranks the reaction-field matrix is summed in tile order on element
    slices (`ordered_sliced_sum`): a rank receives each tile's term only on
    its 1 / size slice, and the slices once, instead of every tile's whole
    term (`ordered_sum`). The result is the serial matrix bit for bit."""
    mol = water()
    dms = density(mol)
    small_tiles(monkeypatch, mol.nao)
    with distributed(None):
        serial = factorize_once(continuum(mol))
        q = np.random.default_rng(3).standard_normal(serial._get_v(dms).shape)
        vmat_serial = serial._get_vmat(q)
    ntiles = len(cavity_tiles(mol.nao, serial.surface['grid_coords'].shape[0]))
    assert ntiles > 4 * size
    received, lock = {}, threading.Lock()

    def note(n):
        with lock:
            name = threading.current_thread().name
            received[name] = received.get(name, 0) + n
    exchange, gather = mpi_grid.exchange_blocks, mpi_grid.allgather_ranges

    def counted_exchange(send, shapes, comm):
        note(sum(int(np.prod(s)) for s in shapes))
        return exchange(send, shapes, comm)

    def counted_gather(a, ranges, comm):
        note(sum(s1 - s0 for own in ranges for s0, s1 in own) * a.shape[1])
        return gather(a, ranges, comm)
    monkeypatch.setattr(mpi_grid, 'exchange_blocks', counted_exchange)
    monkeypatch.setattr(mpi_grid, 'allgather_ranges', counted_gather)

    def one_rank(comm):
        pcm = factorize_once(continuum(water()))
        attach_cavity_comm(pcm, comm)
        return pcm._get_vmat(q)

    for vmat in run_simulated(one_rank, size):
        np.testing.assert_array_equal(vmat, vmat_serial)
    assert len(received) == size
    whole = vmat_serial.size
    for n in received.values():
        # its slice of the start (zeros) and of every tile, then the sum
        assert n <= (ntiles + 1) * -(-whole // size) + whole, (n, whole, ntiles)
