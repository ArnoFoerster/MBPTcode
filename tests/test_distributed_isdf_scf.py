"""The ISDF-K SCF over simulated ranks (`Base.distributed_isdf_jk`) against
the serial ISDFJK SCF, and the rules its handle is written to.

Gated, cc-pVDZ at 148 points per atom, tiles of `TILE` points so that every
rank count owns a different set of them (water 7 tiles, ethylene 14):
  * one rank and no region is the serial ISDFJK SCF itself: the factory
    hands back None and `distributed_mean_field` is `mf.kernel()`, bitwise;
  * at 2, 3 and 8 ranks, water and ethylene, PBE0 and LRC-wPBEh: the
    converged energy within the SCF's own `CONV_TOL` of the serial ISDF-K
    energy and the same on every rank, the orbitals one set of bits across
    the ranks, the mean field's own ISDFJK back on it afterwards;
  * J, K and the long-range K of one fixed density at 2, 3 and 8 ranks
    against the one-rank handle (the same tiles, no collective): K and K_lr,
    sums over row tiles, with every rank's partial its tiles' one-rank
    addends added in tile order, bitwise, and the reduced K within the
    rounding bound of the addends' exact sum at every element
    (`rounding_bound`: half an ulp of each partial sum a rank forms and of
    each join of two ranks' partials, the most any order of the reduction
    can move it, floored at one ulp of |K|max); J within COMPOSED_GRAD_K
    times the ranks' partials reversed, floored at one ulp of its largest
    element;
  * tests/test_distributed_isdf_scf_mpi.py's own checks at 8 ranks, where
    rank 7 owns none of water's 7 tiles: a pass on every rank;
  * the bytes every rank holds, read off the handle (`memory_faults` of
    tests/test_distributed_isdf_scf_mpi.py): its
    tiles of X and M^T and its rows of each operator's Z are its own tiles'
    rows and they tile the grid; nothing grid-indexed is whole -- not X, M^T,
    Z or X Dm X^T (one (tile, tile) block), T (the rank's rows), a streamed
    tile, a slab of the metric -- and neither is any array of the fit;
  * on a BLAS whose GEMM bits follow the call's shape AND the thread count
    of the calling rank's BLAS pool (a subprocess: every GEMM under src on
    the shape-sensitive stand-in of tests/test_force_serial_shaped.py, its
    scaling keyed on the pool too, and `Base.utils.threads` reading and
    limiting each simulated rank's own pool of `NODE_THREADS` threads
    (`on_the_node_pool`), so `blas_single_threaded` drops it to one and
    `blas_full_pool` gives it back as on a rank of a real node): every
    rank's tiles of Z (and of G, factored) bitwise the one-rank tiles, J on
    its anchored bar and K on its rounding bound; each rank's K partials
    built inside the SCF's wrap bitwise the ones outside it; after a
    distributed SCF whose handle was built inside it (`build=False`), every
    rank's tiles bitwise the one-rank tiles and its record at NODE_THREADS
    for the fit, the interaction and K -- while a GEMM over a rank's
    concatenated
    rows, a GEMM inside the wrap and a one-rank reference fitted at one
    pool thread, all three shown moved, would not be;
  * under the same emulated node in this process, the pool every stage of
    that SCF runs on: the fit's Cholesky, G and Z's rows, the K builds and
    the DF-J metric's factor and solve at NODE_THREADS, and `nr_rks`, the
    DF-J passes, the fit's three-centre integrals and the metric slabs at
    one; the density-fitted SCF's K and `nr_rks` at one, as before;
  * under the pyscf race (`racing_dgemm` of tests/test_force_serial_shaped.py,
    shown live on each run): every rank's converged energy and orbitals are
    rank 0's;
  * anthracene/cc-pVDZ/PBE0 at 3 ranks on the default tile, once.

SHOWN TO FAIL, the pooled gate: with `distributed_isdf_jk` building the
handle inside `blas_single_threaded` (the factory before this gate), every
rank fitted its tiles at one pool thread and the one-rank handle at the
process's own count, and at 2, 3 and 8 ranks every tile of both operators' Z
differed from the one-rank tile, K at 7.1e3 and K_lr at 5.2e4 to 6.1e4
times its rounding bound (0.2-0.7 without the defect) -- what 16-thread
OpenBLAS nodes showed, and what no laptop
gate could: the real `blas_single_threaded` is a no-op below
BLAS_WRAP_MIN_THREADS (two threads here) and off the main thread (every
simulated rank). On the shape stand-in alone, the pool left out, the same
code passed: no GEMM of the handle or of the row fit is rank-shaped. With
`blas_full_pool` a no-op (the handle's stages inside the SCF's wrap at one
thread, as they ran on the 8-node pentacene run), the pooled gate fails on
every rank's wrapped K partials and on every tile of the SCF-built handles,
and the per-stage test reads one thread for every GEMM stage. The script's
checks at 8 ranks, with the check of the SCF-built handle asking every rank
for a tile: rank 7 failed it for both functionals, as on the 8-node run.
"""
import collections
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest
import scipy.linalg
from pyscf import dft, gto, lib, scf
from pyscf.df import df_jk
from pyscf.dft import numint
from pyscf.lib import numpy_helper
from pyscf.scf import jk as pyscf_jk

from src.Base import distributed_df as dist_df
from src.Base import distributed_isdf_jk as dist_isdf
from src.Base import separable_ri
from src.Base.constants import BLAS_WRAP_MIN_THREADS, COMPOSED_GRAD_K
from src.Base.distributed_df import distributed_mean_field, release_distributed
from src.Base.distributed_isdf_jk import (ISDF_BLAS_KEYS, distributed_isdf_jk,
                                          distributed_isdf_storage)
from src.Base.isdf_jk import ISDFJK, isdf_jk
from src.Base.utils import threads
from src.Base.utils.mpi_grid import run_simulated
from tests import test_force_serial_shaped as shape_blas
from tests.reduction_bounds import exact_offset, regrouped, rounding_bound
from tests.test_distributed_isdf_scf_mpi import main as mpi_main
from tests.test_distributed_isdf_scf_mpi import memory_faults, tile_addends
from tests.test_force_serial_shaped import racing_dgemm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
#: Anthracene, the idealized D2h acene of three rings (C-C 1.40, C-H
#: 1.09 Angstrom).
ANTHRACENE = '\n'.join([
    'C      1.212440     0.700000     0.000000',
    'C      0.000000     1.400000     0.000000',
    'C     -1.212440     0.700000     0.000000',
    'C     -1.212440    -0.700000     0.000000',
    'C     -0.000000    -1.400000     0.000000',
    'C      1.212440    -0.700000     0.000000',
    'C      3.637310     0.700000     0.000000',
    'C      2.424870     1.400000     0.000000',
    'C      2.424870    -1.400000     0.000000',
    'C      3.637310    -0.700000     0.000000',
    'C      6.062180     0.700000     0.000000',
    'C      4.849740     1.400000     0.000000',
    'C      4.849740    -1.400000     0.000000',
    'C      6.062180    -0.700000     0.000000',
    'H     -0.000000     2.490000     0.000000',
    'H     -2.156400     1.245000     0.000000',
    'H     -2.156400    -1.245000     0.000000',
    'H     -0.000000    -2.490000     0.000000',
    'H      2.424870     2.490000     0.000000',
    'H      2.424870    -2.490000     0.000000',
    'H      7.006150     1.245000     0.000000',
    'H      4.849740     2.490000     0.000000',
    'H      4.849740    -2.490000     0.000000',
    'H      7.006150    -1.245000     0.000000',
])
SIZES = [2, 3, 8]
MOLECULES = {'water': WATER, 'ethylene': ETHYLENE}
FUNCTIONALS = ['pbe0', 'lrc-wpbeh']
#: LRC-wPBEh's range-separation parameter in pyscf.
OMEGA = 0.2
#: Grid points per tile: water's 444 points in 7 tiles, ethylene's 888 in 14.
TILE = 64
#: The SCF's own thresholds: the energy gate is CONV_TOL itself.
CONV_TOL = 1e-10
CONV_TOL_GRAD = 1e-6
#: Metric slab for the memory gate: water's 84 auxiliary functions in three
#: slabs, where the production slab would take them in one.
SMALL_SLAB = 32
ULP = np.finfo(float).eps
#: The BLAS threads of the node every simulated rank stands for under
#: `on_the_node_pool`: at BLAS_WRAP_MIN_THREADS or more, so every wrap arms.
NODE_THREADS = 8
#: Each simulated rank's BLAS pool under `on_the_node_pool`: a thread of this
#: process is a rank, so the pool it models is the thread's.
POOL = threading.local()
#: One simulated rank at a time into `note`'s sets.
_NOTE_LOCK = threading.Lock()
#: The handle's GEMM stages of a distributed ISDF-K SCF, by the call `note`
#: keys them on: each must run on the whole node pool.
GEMM_STAGES = {
    ('dpotrf', 'src.Base.separable_ri', 'fit_rows'): "the fit's Cholesky",
    ('_trsm', 'src.Base.separable_ri', 'fit_rows'): "the fit's trsm",
    ('dot', 'src.Base.distributed_isdf_jk', '_interaction_rows'):
        "G and Z's rows",
    ('dot', 'src.Base.distributed_isdf_jk', '_exchange_rows'): 'the K builds',
    ('cho_factor', 'src.Base.distributed_isdf_jk', '__init__'):
        "the DF-J metric's factor",
    ('cho_solve', 'src.Base.distributed_isdf_jk', '__call__'):
        "the DF-J metric's solve"}
#: Its stages of pyscf's OpenMP and libcint: each must run at one BLAS thread.
OPENMP_STAGES = {
    ('nr_rks', 'src.Base.distributed_df', 'partial_xc'): 'nr_rks on the grid',
    ('get_jk', 'src.Base.distributed_isdf_jk', '__call__'): 'the DF-J passes',
    ('getints3c', 'src.Base.separable_ri', '__call__'):
        "the fit's three-centre integrals",
    ('intor', 'src.Base.distributed_isdf_jk', '_interaction_rows'):
        'the metric slabs'}


class NodePoolLimit:
    """threadpoolctl's `threadpool_limits` on the calling rank's own pool:
    the count set when made, the one it found put back on exit."""

    def __init__(self, limits=None, user_api=None):
        self.found = node_pool_threads()
        POOL.threads = limits

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        POOL.threads = self.found
        return False


class CountingNumpy:
    """numpy, save that `dot` notes the calling rank's pool under
    ('dot', the caller's module, the caller's name) in `seen`."""

    def __init__(self, seen):
        self._seen = seen

    def __getattr__(self, name):
        return getattr(np, name)

    def dot(self, *args, **kwargs):
        note(self._seen, 'dot', sys._getframe(1))
        return np.dot(*args, **kwargs)


def node_pool_threads():
    """The calling rank's BLAS threads: NODE_THREADS until a limit moves it."""
    return getattr(POOL, 'threads', NODE_THREADS)


def on_the_node_pool(set_attr):
    """`Base.utils.threads` reading and limiting the calling rank's pool in
    place of the process's, every simulated rank the main thread of its own
    process on a node of NODE_THREADS: the real wraps are no-ops below
    BLAS_WRAP_MIN_THREADS (two on a two-thread machine) and off the main thread (every
    simulated rank), so no laptop run moves a pool without this.

    set_attr: `monkeypatch.setattr`, or `setattr` in a subprocess."""
    set_attr(threads, 'threadpool_limits', NodePoolLimit)
    set_attr(threads, 'blas_threads', node_pool_threads)
    set_attr(threads, '_pool_is_ours', lambda: True)


def pooled(shape_factor):
    """The shape-sensitive BLAS's scaling keyed on the pool as well: a GEMM
    on fewer than NODE_THREADS takes other bits, as on a BLAS whose sums
    block by its thread count -- a two-thread MKL fits water's M^T 2.4e-3
    apart at one and at two threads."""
    def factor(op, shapes):
        count = node_pool_threads()
        return shape_factor(op, shapes if count == NODE_THREADS
                            else (shapes, count))
    return factor


def note(seen, label, frame):
    """The calling rank's pool under (label, the frame's module, its name)."""
    key = (label, frame.f_globals.get('__name__'), frame.f_code.co_name)
    with _NOTE_LOCK:
        seen[key].add(node_pool_threads())


def noting(seen, label, func):
    """`func`, noting the calling rank's pool on every call."""
    def call(*args, **kwargs):
        note(seen, label, sys._getframe(1))
        return func(*args, **kwargs)
    return call


def fresh(atom, xc):
    """An unrun ISDF mean field on `atom`/cc-pVDZ, the default grid."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=atom, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
    mf = isdf_jk(mf, auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = CONV_TOL, CONV_TOL_GRAD
    return mf


def digest(a):
    """The bytes of an array, hashed."""
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def converge(size, atom, xc, tile=TILE):
    """Every rank's converged ISDF-K SCF over `size` simulated ranks: its
    energy, orbital digest, what its with_df was after, and its storage."""
    mfs = [fresh(atom, xc) for _ in range(size)]

    def one(comm):
        mf = mfs[comm.Get_rank()]
        distributed_isdf_jk(mf, comm, tile=tile)
        distributed_mean_field(mf)
        out = dict(e=mf.e_tot, mo=digest(mf.mo_coeff), cycles=mf.cycles,
                   with_df=type(mf.with_df).__name__,
                   storage=distributed_isdf_storage(mf, comm),
                   timings=dict(mf._distributed_timings))
        release_distributed(mf)
        return out
    return run_simulated(one, size)


@pytest.fixture(scope='module')
def serial():
    """The serial ISDF-K SCF of every (molecule, functional), converged."""
    out = {}
    for name, atom in MOLECULES.items():
        for xc in FUNCTIONALS:
            mf = fresh(atom, xc)
            mf.kernel()
            out[name, xc] = mf
    return out


def test_one_rank_is_the_serial_isdfjk():
    """No communicator: no handle, and the driver is `mf.kernel()` bit for
    bit on the mean field's own ISDFJK."""
    ref = fresh(WATER, 'pbe0')
    ref.kernel()
    mf = fresh(WATER, 'pbe0')
    assert distributed_isdf_jk(mf, None) is None
    distributed_mean_field(mf, comm=None)
    assert isinstance(mf.with_df, ISDFJK)
    assert mf.e_tot == ref.e_tot
    assert np.array_equal(mf.mo_coeff, ref.mo_coeff)
    assert getattr(mf, '_distributed', None) is None


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('xc', FUNCTIONALS)
@pytest.mark.parametrize('name', sorted(MOLECULES))
def test_scf_matches_serial(serial, name, xc, size):
    """The energy within CONV_TOL of serial and one number on every rank,
    the orbitals one set of bits, pyscf's cycle count the same, the mean
    field's own ISDFJK back on it."""
    ref = serial[name, xc]
    res = converge(size, MOLECULES[name], xc)
    energies = {r['e'] for r in res}
    assert len(energies) == 1, energies
    dE = abs(res[0]['e'] - ref.e_tot)
    assert dE <= CONV_TOL, (name, xc, size, dE)
    assert len({r['mo'] for r in res}) == 1
    assert len({r['cycles'] for r in res}) == 1
    assert all(r['with_df'] == 'ISDFJK' for r in res)
    assert all(r['timings']['scf_requests_k'] > 0 for r in res)
    # built before the SCF, the handle records the interaction and K only
    outside = threads.blas_threads() or 0
    assert all(r['timings'][key] == outside for r in res
               for key in ISDF_BLAS_KEYS[1:]), [r['timings'] for r in res]


def fixed_density(mf):
    """The converged density of `mf`, tagged with its orbitals, a copy."""
    dm = mf.make_rdm1()
    return lib.tag_array(np.array(dm), mo_coeff=mf.mo_coeff.copy(),
                         mo_occ=mf.mo_occ.copy())


def one_rank_jk(mf, factored=False):
    """J, K and K_lr of `mf`'s density from the handle on one rank (the same
    tiles, no collective), its interaction tiles, and K's and K_lr's tile
    addends."""
    source = fresh(WATER, 'lrc-wpbeh').with_df
    if factored:
        source.z_mode = 'factored'
    handle = dist_isdf.DistributedISDFJK(source, comm=None,
                                         tile=TILE).build()
    vj, vk = handle.get_jk(fixed_density(mf))
    klr = handle.get_jk(fixed_density(mf), with_j=False, omega=OMEGA)[1]
    addends = tuple(tile_addends(handle, fixed_density(mf), omega)
                    for omega in (None, OMEGA))
    return (vj, vk, klr), interaction_tiles(handle), addends


def interaction_tiles(handle):
    """{(operator, tile): rows} of every interaction the handle holds."""
    return {(key, t): rows.copy()
            for key, kernel in handle.omega_kernels.items()
            for t, rows in kernel.items()}


def ranked_partials(mf, size, factored=False):
    """Over `size` ranks on `mf`'s density: every rank's reduced (J, K, K_lr),
    its own partials of the three before the reduction, its interaction
    tiles, (a rank with two tiles or more) the first tile's rows of one
    GEMM over all its M^T rows beside that tile's own GEMM, its K and K_lr
    partials again inside the SCF's `blas_single_threaded`, and its tiles."""
    mfs = [fresh(WATER, 'lrc-wpbeh') for _ in range(size)]

    def one(comm):
        mine = mfs[comm.Get_rank()]
        if factored:
            mine.with_df.z_mode = 'factored'
        handle = distributed_isdf_jk(mine, comm, tile=TILE)
        dm = fixed_density(mf)
        stack = np.asarray(dm).reshape(1, *dm.shape)
        factors = dist_isdf._occupied_factors(dm)
        pj = handle._coulomb_engine(None, 1e-13)(stack)[0]
        pk = handle.exchange_partial(stack, factors)[0]
        plr = handle.exchange_partial(stack, factors, OMEGA)[0]
        with threads.blas_single_threaded():
            wrapped = (handle.exchange_partial(stack, factors)[0],
                       handle.exchange_partial(stack, factors, OMEGA)[0])
        vj, vk = handle.get_jk(fixed_density(mf))
        klr = handle.get_jk(fixed_density(mf), with_j=False, omega=OMEGA)[1]
        joined = None
        if len(handle.mine) > 1:
            # on the shape-sensitive BLAS's own GEMM, as a tile formed so
            # would take it
            mm = shape_blas.shape_mm
            first = handle.MT[handle.mine[0]]
            rows = np.vstack([handle.MT[t] for t in handle.mine])
            joined = (mm(rows, first.T)[:len(first)], mm(first, first.T))
        tiles = interaction_tiles(handle)
        mine = list(handle.mine)
        handle.release()
        return (vj, vk, klr), (pj, pk, plr), tiles, joined, wrapped, mine
    return run_simulated(one, size)


def anchored_faults(mf, factored=False, one=None):
    """J, K and K_lr at every size of SIZES against the one-rank handle. J
    on COMPOSED_GRAD_K times the largest reassociation response measured on
    the run: one ulp of its largest element, and every size's partials added
    in reverse rank order. K and K_lr on their rounding bound: every rank's
    partial its tiles' one-rank addends added in tile order, bitwise, and
    the reduced K within `rounding_bound` of the addends' exact sum at every
    element, floored at one ulp of |K|max. The interaction tiles bitwise the
    one-rank tiles; every rank's results one set of bits; every rank's K and
    K_lr partials inside the SCF's wrap bitwise the ones outside it. Returns
    (faults, ratios, live): each result's distance over its bar, and whether
    a joined GEMM's rows differed from the tile's own.

    one: `one_rank_jk(mf, factored)`, made here when None."""
    ref, ref_tiles, addends = (one_rank_jk(mf, factored=factored)
                               if one is None else one)
    runs = {size: ranked_partials(mf, size, factored) for size in SIZES}
    anchor = ULP * np.abs(ref[0]).max()
    for res in runs.values():
        parts = [r[1][0] for r in res]
        forward, backward = parts[0].copy(), parts[-1].copy()
        for p in parts[1:]:
            forward = forward + p
        for p in parts[-2::-1]:
            backward = backward + p
        anchor = max(anchor, np.abs(forward - backward).max())
    faults, ratios, live = [], {}, False
    faults += [f'the tile addends do not sum to the one-rank {name}'
               for name, adds, whole in zip(('K', 'K_lr'), addends, ref[1:])
               if not np.array_equal(regrouped(adds, [range(len(adds))]),
                                     whole)]
    for size, res in runs.items():
        tiles = {}
        for rank, r in enumerate(res):
            tiles.update(r[2])
            if r[3] is not None:
                live |= not np.array_equal(*r[3])
            faults += [f'{size} ranks: rank {rank} {name} inside the '
                       "SCF's wrap is not its partial outside it"
                       for name, inside, outside in zip(
                           ('K', 'K_lr'), r[4], r[1][1:])
                       if not np.array_equal(inside, outside)]
        if set(tiles) != set(ref_tiles):
            faults.append(f'{size} ranks: the tiles do not tile the grid')
        faults += [f'{size} ranks: tile {key} is not the one-rank tile'
                   for key, rows in tiles.items()
                   if not np.array_equal(rows, ref_tiles.get(key))]
        owners = [r[5] for r in res]
        for n, name in enumerate(('J', 'K', 'K_lr')):
            got = [r[0][n] for r in res]
            if len({digest(g) for g in got}) != 1:
                faults.append(f'{size} ranks: {name} differs between ranks')
            if n == 0:
                ratios[size, name] = (np.abs(got[0] - ref[n]).max()
                                      / (COMPOSED_GRAD_K * anchor))
            else:
                adds = addends[n - 1]
                faults += [f'{size} ranks: rank {rank} {name} partial is not '
                           "its tiles' one-rank addends"
                           for rank, r in enumerate(res)
                           if not np.array_equal(r[1][n],
                                                 regrouped(adds, [r[5]]))]
                bound = np.maximum(rounding_bound(adds, owners),
                                   np.spacing(np.abs(ref[n]).max()))
                ratios[size, name] = float(
                    (np.abs(exact_offset(got[0], adds)) / bound).max())
            if ratios[size, name] > 1:
                faults.append(f'{size} ranks: {name} at '
                              f'{ratios[size, name]:.2f} of its bar')
    return faults, ratios, live


def test_jk_reduction_is_anchored(serial):
    """J at 2, 3 and 8 ranks on its anchored bar and K and K_lr on their
    rounding bound, every rank's K partials its tiles' one-rank addends, the
    tiles of both operators' Z bitwise the one-rank tiles, every rank the
    same."""
    faults, ratios, _ = anchored_faults(serial['water', 'lrc-wpbeh'])
    assert not faults, (faults, ratios)


@pytest.mark.parametrize('size', SIZES)
def test_no_rank_holds_a_grid_array_whole(monkeypatch, size):
    """Every rank's held bytes, read off its handle after a LRC-wPBEh SCF:
    its own tiles' rows of X, M^T and both operators' Z, the rows tiling the
    grid, and no grid-indexed array or metric whole anywhere."""
    monkeypatch.setattr(dist_isdf, 'ISDF_SCF_METRIC_SLAB',
                        SMALL_SLAB)
    res = converge(size, WATER, 'lrc-wpbeh')
    nocc = gto.M(atom=WATER, basis='cc-pvdz').nelectron // 2
    storages = [r['storage'] for r in res]
    assert sum(s['rows_here'] for s in storages) == storages[0]['M']
    for rank, storage in enumerate(storages):
        faults = memory_faults(storage, nocc, slab=SMALL_SLAB)
        assert not faults, (size, rank, faults)
        assert len([k for k in storage['held_now']
                    if k.startswith('interaction_')]) == 2


def test_the_mpi_script_where_a_rank_owns_no_tile():
    """tests/test_distributed_isdf_scf_mpi.py's own checks over 8 simulated
    ranks, where water's 7 tiles leave rank 7 without one: every rank's
    verdict a pass, the checks of a handle built inside the SCF included."""
    codes = run_simulated(mpi_main, 8)
    assert codes == [0] * 8, codes


@pytest.mark.parametrize('factored', [False, True])
def test_rows_are_serial_shaped(factored):
    """On the shape- and pool-sensitive BLAS, every GEMM under src on it and
    every rank on its own node pool: every rank's tiles of Z (of G,
    factored) are the one-rank tiles bitwise at 2, 3 and 8 ranks, built
    before the SCF or inside it, J sits on its anchored bar measured on the
    same BLAS and K, K_lr on their rounding bound; K built inside the SCF's
    wrap is K outside
    it, bitwise; a GEMM over a rank's joined rows, one inside the wrap and a
    reference fitted at one pool thread are moved by it, so a tile formed
    any of those ways would not pass."""
    got = on_the_pooled_blas(factored)
    print(f"factored={factored}: {got['ratios']}")
    assert got['perturbed'] > 0, 'the shape-sensitive BLAS scaled nothing'
    assert got['rows_live'], 'the shape-sensitive BLAS moved no joined GEMM'
    assert got['pool_live'], 'the pool stand-in moved no GEMM'
    assert got['reference_live'], ('a reference fitted at one pool thread '
                                   'took the tiles of the whole pool')
    assert not got['faults'], (got['faults'], got['ratios'])


def on_the_pooled_blas(factored):
    """`pooled_main(factored)` in a fresh interpreter, its verdict."""
    env = dict(os.environ, MPI4PY_RC_INITIALIZE='0',
               PYTHONPATH=os.pathsep.join(
                   [REPO] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
    out = subprocess.run([sys.executable, os.path.abspath(__file__),
                          'factored' if factored else 'dense'], env=env,
                         cwd=REPO, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith('RESULT '))
    return json.loads(line[len('RESULT '):])


def pooled_main(factored, pool=True):
    """In the subprocess: src imported afresh with every GEMM on the
    shape-sensitive BLAS, this file imported again on it, and -- `pool` --
    the BLAS keyed on the pool and every rank on its own node pool."""
    shape_blas.import_src_on_the_shape_sensitive_blas()
    for name in [n for n in sys.modules if n.startswith('tests.')
                 and n != shape_blas.__name__]:
        del sys.modules[name]
    here = importlib.import_module('tests.test_distributed_isdf_scf')
    if pool:
        shape_blas.shape_factor = here.pooled(shape_blas.shape_factor)
        here.on_the_node_pool(setattr)
    return here.pooled_verdict(factored)


def pooled_verdict(factored):
    """`anchored_faults` and `scf_tile_faults` on the BLAS this process
    runs, and whether each of its sensitivities moved a product: a GEMM over
    a rank's joined rows against the tile's own, one GEMM inside the SCF's
    wrap against outside, a one-rank reference fitted at one pool thread
    against the one fitted on the whole pool."""
    mf = fresh(WATER, 'lrc-wpbeh')
    mf.kernel()
    one = one_rank_jk(mf, factored=factored)
    faults, ratios, live = anchored_faults(mf, factored=factored, one=one)
    faults += scf_tile_faults(one[1], factored)
    a = np.random.default_rng(5).normal(size=(TILE, TILE))
    with threads.blas_single_threaded():
        inside = shape_blas.shape_mm(a, a.T)
    with threads.threadpool_limits(limits=1, user_api='blas'):
        narrow = one_rank_tiles(factored)
    return dict(faults=faults, rows_live=bool(live),
                pool_live=not np.array_equal(inside, shape_blas.shape_mm(a, a.T)),
                reference_live=any(not np.array_equal(rows, one[1][key])
                                   for key, rows in narrow.items()),
                ratios={f'{size} {name}': float(r)
                        for (size, name), r in ratios.items()},
                perturbed=shape_blas.SHAPE_STATE['perturbed'])


def one_rank_tiles(factored):
    """The one-rank handle's tiles of the bare operator's interaction."""
    source = fresh(WATER, 'lrc-wpbeh').with_df
    if factored:
        source.z_mode = 'factored'
    handle = dist_isdf.DistributedISDFJK(source, comm=None, tile=TILE).build()
    handle.interaction(None)
    return interaction_tiles(handle)


def scf_tile_faults(ref_tiles, factored):
    """At every size of SIZES, a LRC-wPBEh SCF whose handle is built inside
    it (`build=False`): every rank's tiles of both operators
    bitwise the one-rank tiles `ref_tiles`, and the BLAS threads its record
    says the fit, the interaction and K ran on the count outside the SCF."""
    outside = threads.blas_threads() or 0
    faults = []
    for size in SIZES:
        mfs = [fresh(WATER, 'lrc-wpbeh') for _ in range(size)]

        def one(comm):
            mf = mfs[comm.Get_rank()]
            if factored:
                mf.with_df.z_mode = 'factored'
            distributed_isdf_jk(mf, comm, tile=TILE, build=False)
            distributed_mean_field(mf)
            out = (interaction_tiles(mf._distributed[0]),
                   {key: mf._distributed_timings.get(key)
                    for key in ISDF_BLAS_KEYS})
            release_distributed(mf)
            return out
        covered = set()
        for rank, (tiles, record) in enumerate(run_simulated(one, size)):
            covered |= set(tiles)
            faults += [f'SCF {size} ranks: rank {rank} tile {key} is not the '
                       'one-rank tile' for key, rows in tiles.items()
                       if not np.array_equal(rows, ref_tiles.get(key))]
            faults += [f'SCF {size} ranks: rank {rank} {key} = {count}, not '
                       f'{outside}' for key, count in record.items()
                       if count != outside]
        if covered != set(ref_tiles):
            faults.append(f'SCF {size} ranks: the tiles of both operators do '
                          'not tile the grid')
    return faults


@pytest.mark.skipif(threads.threadpool_limits is None
                    or (threads.blas_threads() or 1) < 2,
                    reason='a pool of one thread: nothing to take or give back')
def test_the_full_pool_is_the_count_the_wrap_took():
    """On this machine's own BLAS (`min_threads=1` arms the wrap at two
    threads): one thread inside `blas_single_threaded`, the count it found
    inside `blas_full_pool` -- and one again in a wrap nested there -- and
    every count back on the way out; outside any wrap, the pool untouched."""
    ambient = threads.blas_threads()
    with threads.blas_full_pool() as count:
        assert count == ambient == threads.blas_threads()
    with threads.blas_single_threaded(min_threads=1):
        assert threads.blas_threads() == 1
        with threads.blas_full_pool() as count:
            assert count == ambient == threads.blas_threads()
            with threads.blas_single_threaded(min_threads=1):
                assert threads.blas_threads() == 1
            assert threads.blas_threads() == ambient
        assert threads.blas_threads() == 1
    assert threads.blas_threads() == ambient


def test_each_stage_runs_on_its_pool(monkeypatch):
    """On an emulated node of NODE_THREADS threads a rank (`on_the_node_pool`),
    a LRC-wPBEh SCF over two ranks whose handle is built inside it: every
    GEMM stage of the handle on the whole pool, every pass of pyscf's OpenMP
    and libcint at one thread, and every rank's record at NODE_THREADS for
    the fit, the interaction and K."""
    assert NODE_THREADS >= BLAS_WRAP_MIN_THREADS
    mfs = [fresh(WATER, 'lrc-wpbeh') for _ in range(2)]
    seen = collections.defaultdict(set)
    on_the_node_pool(monkeypatch.setattr)
    monkeypatch.setattr(dist_isdf, 'np', CountingNumpy(seen))
    for owner, name in ((separable_ri, '_trsm'),
                        (scipy.linalg.lapack, 'dpotrf'),
                        (scipy.linalg, 'cho_factor'),
                        (scipy.linalg, 'cho_solve'), (numint, 'nr_rks'),
                        (pyscf_jk, 'get_jk'), (gto.moleintor, 'getints3c'),
                        (gto.Mole, 'intor')):
        monkeypatch.setattr(owner, name,
                            noting(seen, name, getattr(owner, name)))

    def one(comm):
        mf = mfs[comm.Get_rank()]
        distributed_isdf_jk(mf, comm, tile=TILE, build=False)
        distributed_mean_field(mf)
        record = {key: mf._distributed_timings.get(key)
                  for key in ISDF_BLAS_KEYS}
        release_distributed(mf)
        return record
    records = run_simulated(one, 2)
    table = {label: sorted(seen.get(key, ())) for key, label in
             list(GEMM_STAGES.items()) + list(OPENMP_STAGES.items())}
    print(f'BLAS threads per stage on a {NODE_THREADS}-thread node: {table}; '
          f'records {records}')
    faults = [f'{label} on {table[label]}, not [{NODE_THREADS}]'
              for label in GEMM_STAGES.values()
              if table[label] != [NODE_THREADS]]
    faults += [f'{label} on {table[label]}, not [1]'
               for label in OPENMP_STAGES.values() if table[label] != [1]]
    faults += [f'a GEMM of the handle in {name} on {sorted(counts)}'
               for (label, module, name), counts in seen.items()
               if label == 'dot' and counts != {NODE_THREADS}]
    faults += [f'rank {rank} recorded {record}'
               for rank, record in enumerate(records)
               if record != dict.fromkeys(ISDF_BLAS_KEYS, NODE_THREADS)]
    assert not faults, faults


def test_the_df_scf_keeps_blas_at_one_thread(monkeypatch):
    """On the same emulated node, the density-fitted PBE0 SCF over two ranks:
    its J/K (pyscf's contraction of the fitted rows) and `nr_rks` at one
    BLAS thread -- the ISDF handle's stages are the only ones given the pool
    back."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mfs = [dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
           for _ in range(2)]
    seen = collections.defaultdict(set)
    on_the_node_pool(monkeypatch.setattr)
    for owner, name in ((df_jk, 'get_jk'), (numint, 'nr_rks')):
        monkeypatch.setattr(owner, name,
                            noting(seen, name, getattr(owner, name)))

    def one(comm):
        mf = mfs[comm.Get_rank()]
        distributed_mean_field(mf)
        release_distributed(mf)
    run_simulated(one, 2)
    table = {f'{label} in {name}': sorted(counts)
             for (label, module, name), counts in seen.items()}
    assert ('get_jk', dist_df.__name__, 'partial_jk') in seen, table
    assert ('nr_rks', dist_df.__name__, 'partial_xc') in seen, table
    assert all(counts == {1} for counts in seen.values()), table


@pytest.mark.parametrize('size', [2, 3])
def test_every_rank_holds_rank_0s_scf_under_the_pyscf_race(monkeypatch,
                                                           size):
    """With pyscf's threaded GEMM adding its partials in arrival order (live:
    one product taken twice differs), the converged energy and orbitals are
    rank 0's on every rank."""
    racing = racing_dgemm(numpy_helper._dgemm)
    monkeypatch.setattr(numpy_helper, '_dgemm', racing)
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(8, 4096)), rng.normal(size=(4096, 8))
    live = any(not np.array_equal(lib.dot(a, b), lib.dot(a, b))
               for _ in range(20))
    assert live, 'the race stand-in moved no product'
    res = converge(size, WATER, 'lrc-wpbeh')
    assert len({r['e'] for r in res}) == 1
    assert len({r['mo'] for r in res}) == 1


def test_anthracene_once():
    """Anthracene/cc-pVDZ/PBE0 at 3 ranks on the production tile: the
    energy within CONV_TOL of the serial ISDF-K SCF, the same on every rank,
    one set of orbitals."""
    atom = ANTHRACENE
    ref = fresh(atom, 'pbe0')
    ref.kernel()
    res = converge(3, atom, 'pbe0', tile=None)
    assert len({r['e'] for r in res}) == 1
    assert abs(res[0]['e'] - ref.e_tot) <= CONV_TOL
    assert len({r['mo'] for r in res}) == 1


if __name__ == '__main__':
    print('RESULT ' + json.dumps(pooled_main(
        sys.argv[1] == 'factored', pool=sys.argv[2:] != ['nopool'])),
        flush=True)
