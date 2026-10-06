"""The SCF's Fock build divided over the ranks: the auxiliary index of the
density fit, and the points of the exchange-correlation grid.

Every other stage of a GW/BSE run divides over the ranks (tau, frequency, the
rows of Zt, the screened kernel). Undivided, the mean field is the same SCF on
the same data on every rank, with `cderi`, naux x nao(nao+1)/2 doubles, whole
on every rank, and spilled to disk and streamed back once per cycle where it
exceeds max_memory (`pyscf/df/df.py::DF.build`).

What divides. J and K are sums over the auxiliary index,

    J_{mu nu} = sum_P L^P_{mu nu} sum_{la si} L^P_{la si} D_{la si}
    K_{mu nu} = sum_P sum_{la si} L^P_{mu la} D_{la si} L^P_{si nu}

with L^P the fitted three-index tensor, so a contiguous block of P gives a
partial of both and the partials add. Rank r holds only its block of rows
(`contiguous_block(naux, r, nranks)`, ceil(naux/nranks) of them, tiling the
auxiliary index), and its partial comes from pyscf's own contraction: this
object is a `pyscf.df.DF` whose stored tensor is the slice, so
`pyscf/df/df_jk.py::get_jk` reads it through `DF.loop` and sums over the rows
that are there, with nothing re-implemented. One reduction adds the ranks'
partials. Per-rank storage is naux_r x nao(nao+1)/2 x 8 bytes.

The grid. On a hybrid, `nr_rks` costs more than the fitted exchange it
accompanies, so dividing J and K alone divides a small share of the time. The
grid integral is a sum over points,

    nelec = sum_g w_g rho(r_g),   E_xc = sum_g w_g rho(r_g) eps_xc(r_g)
    V_g^{mu nu} = sum_g w_g v_xc(r_g) chi_mu(r_g) chi_nu(r_g)

so a contiguous block of points is a partial of all three and the partials
add, as the auxiliary rows do. Rank 0 builds `mf.grids` (pyscf's own
`Grids.build`, with pyscf's own pruning, and the density-dependent
`prune_small_rho_grids_` its SCF applies on the first Fock build) and
broadcasts the coordinates and weights once, 32 bytes a point. Every rank then
takes its block, rebuilds the screening mask for it (`make_mask` on its own
points, since `non0tab` is indexed in blocks of the array it was made for) and
runs pyscf's own `numint.nr_rks` on a `Grids` carrying nothing else. The blocks
are cut in units of `ALIGNMENT_UNIT` points, because `NumInt.block_loop` turns
the sparse AO kernels off on a grid whose length is not a multiple of it.

The long-range operator is a second tensor. A range-separated hybrid asks for
the exchange of erf(omega r)/r beside the bare one (LRC-wPBEh: 0.2 of the bare
K and 0.8 of the long-range one at omega = 0.2), and pyscf answers it from a
second fitted tensor, built from the attenuated three- and two-centre
integrals (`pyscf/df/df.py::range_coulomb`). Here that is a second
`DistributedDF` for that omega, built the same way inside the attenuated
operator the first time the SCF asks for it and cut by the same rule into the
rows of its metric factor (the attenuated metric is singular, so it takes
pyscf's eigen-replacement and has fewer rows than the bare one). Its partials
reduce as the bare one's. It doubles a rank's slice.

The split does not reduce the resident grid memory: `block_loop` already
streams the AO values in blocks capped at 1200 x BLKSIZE points and by
max_memory, whatever the rank count, until a rank's slice falls below one
block. What falls with the rank count is the number of blocks a rank walks,
which is the time.

Not split: the NLC grid (`nr_nlc_vxc` on `nlcgrids`), integrated on rank 0
alone because VV10 is a double sum over the points and not a sum of blocks;
the density probe `prune_small_rho_grids_` reads, because what it prunes is
the grid rank 0 hands out; and with `split_grid=False` the whole quadrature,
which is the fit's division timed apart from the grid's. Rank 0's result is
shared through the same reduction, the other ranks adding exact zeros.

How a rank gets its rows: the tensor is cut twice. cderi = L^-1 (P|mu nu) with
L the Cholesky factor of the auxiliary metric (P|Q), so row P of the result
reads raw rows 0..P: no factorization of the metric lets a rank evaluate its
own rows from its own auxiliary shells. What a row slice does not need is the
whole raw tensor, only the whole auxiliary extent of the AO-pair columns in
flight, which is the block `pyscf/df/incore.py::cholesky_eri` forms in its
buffer. So the evaluation is cut by column and the storage by row, and one
exchange turns the first into the second: each rank takes its share of
pyscf's own block list (round-robin, so a round is one block per rank),
evaluates those blocks whole and solves the metric against them, and each
round the ranks trade the row slices of the blocks they just built. Every
column is evaluated by one rank, the whole tensor crosses the wire once, and
no rank holds more of it than its slice plus one round. (A replicated
evaluation, every rank walking every block, would need no communication but
would not divide in time.)

The last bit of the tensor belongs to the blocking, not the split. How many
columns go into one `solve_triangular` decides how LAPACK blocks the solve, so
cderi is not bit-stable in the width of the blocks: `incore.cholesky_eri` on
ethylene/cc-pVTZ differs from itself by 2.5e-17 between max_memory 4000 and
20. The ranks' slices reassemble bitwise into the same block list walked by
one rank, which is what says the split is exact; against pyscf's default
blocking they sit that same last bit away.

Every rank runs the driver; only the two handles communicate. pyscf's SCF
loop runs unchanged on every rank. It takes discrete decisions from its own
arithmetic (the DIIS extrapolation, the occupation, when it has converged),
and ranks deciding them from last-bit-different numbers fall out of step and
then call the same collective a different number of times. So both handles
`lockstep` their input at entry: `get_jk` and the quadrature write rank 0's
density matrix, and the orbitals and occupations tagging it (both
`df_jk.get_jk` and the rho evaluator read them), into every rank's own arrays
in place before any partial is formed. The partials are then partials of one
density, the all-reduced J, K and V_xc are the same bits on every rank, and
because the lockstep wrote into the loop's own density and orbitals, whatever
pyscf decides next (the energy, the orbital gradient, the DIIS error vector)
is computed from the same inputs on every rank. What no per-call lockstep
reaches is the loop's exit, so `conv_tol`, `conv_tol_grad`, `max_cycle` and
`conv_check` are rank 0's from the start, and the converged spectrum,
orbitals, occupations, energy and flag are locked once more at the end, so
every rank returns rank 0's mean field bit for bit even where a rank's
eigensolver differed in a last bit. The grid is the one piece not run
everywhere: while the quadrature handle is installed only rank 0's
`initialize_grids` builds anything, and its points reach the others inside the
quadrature, once per epoch. At the end rank 0's whole grid (the points pruned
against its density) is written onto every rank's `mf.grids` (and
`nlcgrids`), so no rank builds and prunes its own afterwards: a rank pruning
against the converged density keeps another point count, and every later
consumer of the grid (a Fock build, a response kernel, a grid-response
skeleton, pyscf's gradient) would integrate another quadrature on each rank.

Per Fock build that is two locksteps of the density and its orbitals,
2 x (2 nao^2 + nmo) doubles, one all-reduce of 2 nao^2 for (J, K) and one of
nao^2 for (nelec, E_xc, V_xc). The two locksteps are checked
(`mpi_grid.lockstep(check=True)`), so where the ranks agree they send 8 bytes
an array and only the all-reduces carry volume. One-off on top of that: the
metric factor, the grid and the fitted tensor's one crossing.

What the SCF leaves behind. The slices outlive the SCF that built them, on
`mf._distributed`, because the static exchange every quasiparticle route
builds next is the same two operators over the same ranks: one K and one
v_xc on the DFT grid. They are not left installed (a downstream
`with_df.loop()` or a response kernel would read one rank's block as the
whole), so `distributed_fock` is the bracket that installs them for a stage
and takes them off again. What they hold is this rank's slice;
`release_distributed` gives it back.

Where the wall clock goes. A distributed run leaves `TIMING_KEYS` on
`mf._distributed_timings`, filled on every rank, because the question a rank
count raises is not how long the SCF took but which part of it did not divide.
The loop's own work, `scf_driver`, is replicated and the same on every rank;
what a rank cannot spend on its own share it spends waiting inside the
reductions and the locksteps for the slowest rank, `scf_reduce` and
`scf_lockstep`.

What the numbers do. The reductions re-associate the sums over P and over the
grid points, so J, K and V_xc differ from the serial ones in the last bits,
and the SCF converges to a mean field that differs at the convergence
tolerance and not more, because every rank takes every decision from the same
numbers. Gated on water/cc-pVDZ and ethylene/cc-pVTZ at RHF, PBE0 and UKS with
conv_tol 1e-12: energies within 1e-10 Ha of serial, densities within 1e-8,
orbital energies within 1e-8 Ha, and the orbitals identical bit for bit across
ranks.

pyscf internals. Three of the names imported below are private to pyscf:
`df.incore._eig_decompose` (the metric factor's eigen-fallback),
`df.outcore._guess_shell_ranges` (the AO-pair column blocks `cholesky_eri`
walks) and `dft.gen_grid.ALIGNMENT_UNIT` (the point count `block_loop` keeps
its sparse kernels on for), because the split is exact only if it walks
pyscf's own blocking. They were checked against pyscf 2.12.1, the version the
tests here run on; a pyscf that renames one fails at import, not in a result.
"""
import contextlib
import time

import numpy as np
import scipy.linalg
from pyscf import df, gto, lib
from pyscf.df import addons, df_jk
from pyscf.df.incore import _eig_decompose
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.dft import gen_grid, numint
from pyscf.dft.gen_grid import ALIGNMENT_UNIT
from pyscf.lib import logger

from src.Base.constants import DF_EXCHANGE_TRANSIENT_FRACTION
from src.Base.isdf_jk import ISDFJK, range_coulomb
from src.Base.utils import memory
from src.Base.utils.mpi_grid import (broadcast, contiguous_block,
                                     current_comm, exchange_blocks, lockstep,
                                     lockstep_stats, partition, reduce_sum,
                                     replicate)
from src.Base.utils.threads import blas_full_pool, blas_single_threaded

#: What a distributed run leaves on `mf._distributed_timings`, on every rank.
#: Seconds, except the counts in `_COUNT_KEYS` and the two `_mb` volumes. The
#: stages: `scf_build_slices` is this rank's rows of cderi, start to finish --
#: the metric broadcast plus `scf_build_integrals`, the column blocks this rank
#: evaluated and solved against the metric factor (the solve is most of it),
#: plus `scf_build_exchange`, the transposition of those columns
#: into everyone's rows; `scf_guess` is pyscf's initial guess; `scf_grid` is
#: the grid -- rank 0's build and density pruning, and on every rank the wait
#: for a new grid's header, its share of the broadcast and its own mask;
#: `scf_fock_jk` and `scf_fock_xc` are this rank's own partials, summed over
#: every Fock piece it contributed to; `scf_reduce` is the time inside the
#: reductions and `scf_lockstep` inside the locksteps and the per-quadrature
#: header, both largely the wait for the slowest rank; `scf_driver` is the
#: rest of pyscf's loop -- DIIS, the eigenproblem, the density, the energy, the
#: convergence test -- which every rank runs. A build made inside the loop (the
#: ISDF handle fits on its first request) is `scf_build_slices`, not driver.
#: Inside `scf_driver`, `_DRIVER_STAGES`: `scf_driver_eig` the generalized
#: eigenproblems, `scf_driver_fock` the Fock assembly with its DIIS
#: extrapolation, `scf_driver_rdm1` the densities, `scf_driver_grad` the
#: orbital gradients; `scf_driver_other` is the rest. No rank idles and none
#: runs the loop alone, so no key is one rank's. They sum to no more than
#: `scf_total`, the whole call. `scf_lockstep_mb` is what this rank moved
#: through `lockstep` in the call and `scf_lockstep_skipped_mb` what the checked
#: locksteps of the density and the result did not move because every rank's
#: digest agreed. `scf_requests_jk` and `scf_requests_xc` count the
#: J/K builds and the quadratures this rank contributed a partial to.
TIMING_KEYS = ('scf_build_slices', 'scf_build_integrals',
               'scf_build_exchange', 'scf_guess', 'scf_grid', 'scf_fock_jk',
               'scf_fock_xc', 'scf_reduce', 'scf_lockstep', 'scf_driver',
               'scf_lockstep_mb', 'scf_lockstep_skipped_mb', 'scf_cycles',
               'scf_requests_jk', 'scf_requests_xc', 'scf_build_blocks',
               'scf_driver_eig', 'scf_driver_fock', 'scf_driver_rdm1',
               'scf_driver_grad', 'scf_driver_other', 'scf_total')

#: The keys of TIMING_KEYS that count rather than measure.
_COUNT_KEYS = ('scf_cycles', 'scf_requests_jk', 'scf_requests_xc',
               'scf_build_blocks')

#: The stages of pyscf's own SCF driver that are timed around rather than
#: inside a handle. They are timed by wrapping the bound method, not by calling
#: it here: pyscf passes its own arguments to both (`get_init_guess(mol,
#: mf.init_guess, s1e=s1e, **kwargs)`), and a copy of that call could drift (a
#: guess taken with the wrong key lands on a different SCF solution).
#: `initialize_grids` carries `Grids.build` and the `prune_small_rho_grids_`
#: pass, which evaluates the density on the whole grid on rank 0 and builds
#: nothing on the others.
_TIMED_STAGES = (('get_init_guess', 'scf_guess'),
                 ('initialize_grids', '_grid_build'))

#: The dense linear algebra of pyscf's loop, which every rank runs whole on a
#: nao x nao problem: the generalized eigensolve, DIIS's error vector
#: C^T (S D F - F D S) C (four nao^3 GEMMs) and extrapolation, the density and
#: the orbital gradient. Wrapped like `_TIMED_STAGES`, and each takes back the
#: BLAS pool the driver's wrap took (`blas_full_pool`): held at one thread
#: they run a rank's whole nao^3 share of every cycle on one of its cores.
#: A call that would build a Fock piece (`get_fock` without `vhf`, a solvent's
#: without its `v_solvent`; `get_grad` without `fock`) runs as it came, on one
#: thread and off these clocks.
_DRIVER_STAGES = (('eig', 'scf_driver_eig'),
                  ('get_fock', 'scf_driver_fock'),
                  ('make_rdm1', 'scf_driver_rdm1'),
                  ('get_grad', 'scf_driver_grad'))

#: The keys a stage inside `mf.kernel()` adds to; what is left of the kernel's
#: wall is `scf_driver`.
_KERNEL_STAGES = ('scf_build_slices', 'scf_guess', '_grid_build', 'scf_grid',
                  'scf_fock_jk', 'scf_fock_xc', 'scf_reduce', 'scf_lockstep')

#: A mean field's quadrature grids, each rank 0's on every rank after the SCF.
_GRID_ATTRIBUTES = ('grids', 'nlcgrids')

#: What a built and pruned pyscf `Grids` holds: the points and their Becke
#: weights, each point's atom and unpartitioned weight (the grid response
#: reads both), and the AO screening mask of those points.
_GRID_ARRAYS = ('coords', 'weights', 'atm_idx', 'quadrature_weights',
                'non0tab', 'screen_index')


def _spent(timings, key, since):
    """Add the seconds since `since` to `key`. No dict, no timing."""
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + (time.time() - since)


def _counted(timings, key, n=1):
    """`n` more of whatever `key` counts."""
    if timings is not None:
        timings[key] = timings.get(key, 0) + n


def _timed(func, timings, key):
    """`func` with the clock read around it and nothing else changed."""
    def call(*args, **kwargs):
        t0 = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            _spent(timings, key, t0)
    return call


def _time_stages(mf, timings):
    """Clock reads around `_TIMED_STAGES` on `mf`; what to put back."""
    saved = []
    for name, key in _TIMED_STAGES:
        if not hasattr(mf, name):
            continue                       # a Hartree-Fock mean field: no grid
        saved.append((name, mf.__dict__.get(name)))
        setattr(mf, name, _timed(getattr(mf, name), timings, key))
    return saved


def _driver_stages(mf, timings):
    """`_DRIVER_STAGES` on `mf` timed and on the full BLAS pool; what to put
    back."""
    saved = []
    for name, key in _DRIVER_STAGES:
        saved.append((name, mf.__dict__.get(name)))
        setattr(mf, name, _pooled(getattr(mf, name), mf, name, timings, key))
    return saved


def _pooled(func, mf, name, timings, key):
    """`func` on the pool the driver's wrap took, timed under `key`, where
    the call is linear algebra only; as it came otherwise."""
    def call(*args, **kwargs):
        if not _builds_nothing(mf, name, args, kwargs):
            return func(*args, **kwargs)
        t0 = time.time()
        try:
            with blas_full_pool():
                return func(*args, **kwargs)
        finally:
            _spent(timings, key, t0)
    return call


def _builds_nothing(mf, name, args, kwargs):
    """Whether this call of a `_DRIVER_STAGES` method needs no Fock piece:
    `get_fock` handed h1e and vhf (with a solvent's v_solvent on it), and
    `get_grad` handed the Fock matrix."""
    if name == 'get_fock':
        h1e = kwargs.get('h1e', args[0] if len(args) > 0 else None)
        vhf = kwargs.get('vhf', args[2] if len(args) > 2 else None)
        solvated = getattr(mf, 'with_solvent', None) is not None
        return (h1e is not None and vhf is not None
                and (not solvated
                     or getattr(vhf, 'v_solvent', None) is not None))
    if name == 'get_grad':
        fock = kwargs.get('fock', args[2] if len(args) > 2 else None)
        return fock is not None
    return True


def _restore_methods(mf, saved):
    """The bound methods back as they were."""
    for name, original in saved:
        if original is None:
            mf.__dict__.pop(name, None)    # the class's method again
        else:
            setattr(mf, name, original)


def _metric_factor(mol, auxmol):
    """(L, mode) with (P|Q) = L L^T, or the eigen-replacement where it is not
    positive definite -- the choice `pyscf/df/incore.py::cholesky_eri` makes.

    Rank 0's alone, because `mode` is a discrete decision taken from threaded
    dense arithmetic: a rank that fell back to the eigendecomposition holds a
    factor with fewer rows than one that did not, and the ranks would then be
    slicing auxiliary indices of different lengths.
    """
    j2c = auxmol.intor('int2c2e', hermi=1)
    try:
        return scipy.linalg.cholesky(j2c, lower=True), 'cd'
    except scipy.linalg.LinAlgError:
        return _eig_decompose(mol, j2c), 'eig'


def _replicated_metric_factor(mol, auxmol, comm):
    """Rank 0's metric factor and its mode on every rank.

    L^-1 mixes the whole auxiliary index into every row, so rows built against
    two factors that differ in the last bits are rows of two different tensors
    and their partials are not partials of one sum. (naux, naux) doubles, one
    broadcast, once.
    """
    if comm.Get_rank() == 0:
        low, mode = _metric_factor(mol, auxmol)
        header = (mode, low.shape, low.dtype.str)
    else:
        low, header = None, None
    mode, shape, dtype = broadcast(header, comm)
    if low is None:
        low = np.empty(shape, dtype=dtype)
    replicate(low, comm=comm)
    return low, mode


def _column_blocks(mol, nao_pair, naux, size):
    """The AO-pair column blocks the ranks divide: (bstart, bend, width,
    offset), in order, tiling the pair index.

    pyscf's own boundaries (`pyscf/df/outcore.py::_guess_shell_ranges`, the
    list `incore.cholesky_eri` walks), so a block is a whole number of AO
    shells and every column of cderi is evaluated by one rank with the
    arithmetic a serial walk of the same list gives it.

    A pure function of the problem and the rank count: a width read off the
    free memory a rank happens to have would differ between ranks, and a
    block list that does not tile is columns nobody evaluates. It is sized
    against the slice instead: four buffers of (naux, width) are live at once
    (the two integral buffers, the piece this rank sends, the pieces it
    receives), and `DF_EXCHANGE_TRANSIENT_FRACTION` is what they may add up to
    beside the slice being built. `build` refuses upfront if that
    peak does not fit, rather than shrinking the blocks until the exchange is
    all the run does.
    """
    width = (DF_EXCHANGE_TRANSIENT_FRACTION * max(naux // size, 1)
             * nao_pair / (4 * naux))
    buflen = min(max(int(width), 8), nao_pair)
    blocks, offset = [], 0
    for bstart, bend, block in _guess_shell_ranges(mol, buflen, 's2ij'):
        blocks.append((bstart, bend, block, offset))
        offset += block
    return blocks


def _round_block(blocks, cycle, rank, size):
    """(width, offset) of the block rank `rank` owns in this exchange round.

    Round-robin, `partition`'s rule: rank r owns blocks r, r + size, ..., so
    round k is one block per rank, in rank order. A round past the end
    of the list leaves the last ranks with nothing to send, which is a legal
    round of zero-width pieces rather than a special case.
    """
    index = cycle * size + rank
    if index >= len(blocks):
        return 0, 0
    return blocks[index][2], blocks[index][3]


def _solve_column_block(env, low, mode, block, bufs):
    """One AO-pair column block of cderi = L^-1 (P|mu nu), (naux, width).

    `pyscf/df/incore.py::cholesky_eri` (lines 188-217 in pyscf 2.12), one
    iteration of it: the three-centre integrals of those AO pairs against the
    whole auxiliary index, and the metric factor solved against them. The
    same branch on the contiguity of the integral block, because that decides
    which LAPACK call runs and the serial tensor is what these columns must
    equal.

    The result aliases the integral buffer the solve overwrote, so the next
    block destroys it: consume it before asking for another.

    Two pools: the integrals are libcint's, OpenMP over the block's (i, P)
    shell jobs, and run inside the caller's `blas_single_threaded`; the solve
    is BLAS's, naux^2 flops per column against naux for the integrals, and
    runs on the pool that wrap took (`blas_full_pool`), since held at one
    thread it would dominate the build.
    """
    mol, auxmol, int3c, atm, bas, benv, ao_loc, cintopt = env
    bufs1, bufs2 = bufs
    naoaux = low.shape[1]
    bstart, bend = block[0], block[1]
    shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    ints = gto.moleintor.getints3c(int3c, atm, bas, benv, shls_slice, 1,
                                   's2ij', ao_loc, cintopt, out=bufs1)
    if ints.ndim == 3 and ints.flags.f_contiguous:
        ints = lib.transpose(ints.T, axes=(0, 2, 1),
                             out=bufs2).reshape(naoaux, -1)
        bufs[0], bufs[1] = bufs2, bufs1
    else:
        ints = ints.reshape((-1, naoaux)).T
    with blas_full_pool():
        return _metric_solve(low, mode, ints)


def _metric_solve(low, mode, ints):
    """L^-1 (P|mu nu) of one column block, or the eigen-replacement's product,
    overwriting `ints` where LAPACK solves in place."""
    if mode != 'cd':
        return lib.dot(low, ints)
    if ints.flags.c_contiguous:
        trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, ints))
        return trsm(1.0, low, ints.T, lower=True, trans_a=1, side=1,
                    overwrite_b=True).T
    return scipy.linalg.solve_triangular(low, ints, lower=True,
                                         overwrite_b=True, check_finite=False)


def _int3c_env(mol, auxmol):
    """The libcint environment the three-centre blocks are evaluated in."""
    int3c = gto.moleintor.ascint3(mol._add_suffix('int3c2e'))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    return mol, auxmol, int3c, atm, bas, env, ao_loc, cintopt


def _distributed_cderi(mol, auxmol, low, mode, blocks, comm, timings=None):
    """This rank's contiguous auxiliary rows of cderi, built by every rank at
    once and transposed from columns to rows in one exchange.

    The integrals are cut by AO-pair column, the only index the evaluation
    divides along: the metric factor is triangular, so a row of the result
    reads the whole auxiliary extent of the columns in flight. The result is
    stored cut by auxiliary row, the index J and K sum over. So each rank
    walks its own blocks (round-robin over pyscf's block list, one block per
    rank per round), and each round the ranks exchange the row-slices of the
    blocks they just built, every rank's piece of every block going straight
    into its own slice. The whole tensor crosses the wire once and no rank
    holds more of it than its slice plus one round.
    """
    naux = low.shape[0]
    rank, size = comm.Get_rank(), comm.Get_size()
    row_blocks = [contiguous_block(naux, r, size) for r in range(size)]
    start, stop = row_blocks[rank]
    nao_pair = blocks[-1][2] + blocks[-1][3]
    rows = np.empty((stop - start, nao_pair))
    widest = max(width for _, _, width, _ in blocks)
    bufs = [np.empty((widest, low.shape[1])), np.empty((widest, low.shape[1]))]
    env = _int3c_env(mol, auxmol)
    for cycle in range(-(-len(blocks) // size)):
        t0 = time.time()
        index = cycle * size + rank
        dat = (_solve_column_block(env, low, mode, blocks[index], bufs)
               if index < len(blocks) else np.empty((naux, 0)))
        _spent(timings, 'scf_build_integrals', t0)
        t0 = time.time()
        pieces = exchange_blocks(
            [dat[lo:hi] for lo, hi in row_blocks],
            [(stop - start, _round_block(blocks, cycle, r, size)[0])
             for r in range(size)], comm)
        for r, piece in enumerate(pieces):
            offset = _round_block(blocks, cycle, r, size)[1]
            rows[:, offset:offset + piece.shape[1]] = piece
        dat = pieces = None
        _spent(timings, 'scf_build_exchange', t0)
    return rows


def _lockstep_density(dms, comm, timings=None):
    """Rank 0's density matrix, and the orbitals and occupations tagging it,
    written into this rank's own arrays; the same object back.

    The tag travels with the density because both `df_jk.get_jk` and
    `numint._gen_rho_evaluator` read it -- K and rho from the orbitals, not
    from D -- so a density locked without its orbitals would still hand the
    partial this rank's own. In place, so the SCF loop that owns these arrays
    decides its next step from rank 0's copy too.

    Checked (`mpi_grid.lockstep(check=True)`): every rank diagonalized the
    same all-reduced Fock matrix, so digests prove the density rank 0's and
    only a drifted array is broadcast.
    """
    if np.iscomplexobj(dms):
        raise NotImplementedError(
            'the reduction moves float64; a complex density matrix would need '
            'its halves reduced separately')
    t0 = time.time()
    lockstep((dms, getattr(dms, 'mo_coeff', None),
              getattr(dms, 'mo_occ', None)), comm, check=True)
    _spent(timings, 'scf_lockstep', t0)
    return dms


def _reduce_parts(arrays, comm):
    """The ranks' partials added, in one reduction for the whole Fock piece.

    The sum is re-associated (rank boundaries first, then whatever order a
    rank used inside its own block), so the result differs from the serial
    one in the last bits. `reduce_sum` adds in rank order, so it differs in
    the same last bits on every rank and in every repeat.
    """
    shapes = [np.shape(a) for a in arrays]
    buf = np.concatenate([np.ravel(np.asarray(a, dtype=float))
                          for a in arrays])
    reduce_sum(buf, comm)
    out, off = [], 0
    for shape in shapes:
        n = int(np.prod(shape, dtype=int))
        out.append(buf[off:off + n].reshape(shape))
        off += n
    return out


def _grid_slice(mol, coords, weights, cutoff, rank, size, make_mask):
    """A `Grids` carrying this rank's points and its own screening mask.

    Cut in units of ALIGNMENT_UNIT points: `NumInt.block_loop` turns the
    sparse AO kernels off on a grid whose length is not a multiple of it
    (`pyscf/dft/numint.py:2816`), and pyscf's own grid is padded to that
    multiple, so a rank's block should be too.

    `non0tab` is rebuilt rather than sliced: it is indexed in blocks of
    BLKSIZE rows of the coordinate array it was made for, so rank r's block
    starts at the wrong place in rank 0's mask. Nothing calls `build` on the
    result (`block_loop` rebuilds a grid only when `coords is None`), and it
    carries no `atm_idx` or `quadrature_weights`, which the quadrature does
    not read and only a grid-response gradient would.
    """
    n = weights.size
    unit = ALIGNMENT_UNIT if n % ALIGNMENT_UNIT == 0 else 1
    start, stop = contiguous_block(n // unit, rank, size)
    start, stop = start * unit, stop * unit
    grids = gen_grid.Grids(mol)
    grids.coords = np.ascontiguousarray(coords[start:stop])
    grids.weights = np.ascontiguousarray(weights[start:stop])
    grids.cutoff = cutoff
    grids.non0tab = grids.screen_index = make_mask(mol, grids.coords)
    return grids, (start, stop)


def _builds_no_grid(mf):
    """`initialize_grids` for a rank other than 0 while the quadrature handle
    is installed: the grid is rank 0's to build and prune, and it arrives
    inside the quadrature."""
    def initialize_grids(mol=None, dm=None):
        return mf
    return initialize_grids


class DistributedDF(df.df.DF):
    """A `with_df` holding one contiguous block of the auxiliary index.

    Subclasses pyscf's DF for the plumbing `_DFHF` expects and for `DF.loop`,
    which yields the stored tensor in blocks: that tensor being this rank's
    rows is the whole mechanism, and `df_jk.get_jk` then sums over them
    without knowing it is computing a partial.

    Every rank calls `get_jk` at the same point of the same SCF loop: it
    locksteps the density, adds this rank's partial to the others' and
    returns the total, which is what pyscf's SCF asked for.
    """

    _keys = {'comm', 'row_slice', 'column_blocks', 'timings', 'omega'}

    def __init__(self, mol, auxbasis=None, comm=None, timings=None,
                 omega=None):
        super().__init__(mol, auxbasis=auxbasis)
        comm = current_comm() if comm is None else comm
        if comm is None or comm.Get_size() == 1:
            raise ValueError(
                'DistributedDF divides the auxiliary index over ranks and '
                'needs a communicator of more than one. Without one use '
                'pyscf.df.DF, which is what distributed_mean_field falls back '
                'to: a one-rank world must not touch the serial path at all.')
        self.comm = comm
        #: [start, stop) of the auxiliary rows this rank owns, once built.
        self.row_slice = None
        #: Indices, in pyscf's own block list, of the AO-pair column blocks
        #: this rank evaluated. They tile the list across the ranks.
        self.column_blocks = ()
        #: Filled at stage boundaries when a caller wants the wall clock.
        self.timings = timings
        #: The operator the rows are fitted with, pyscf's `range_coulomb`
        #: convention: 0 bare, > 0 long-range erf(omega r)/r, < 0 its
        #: short-range complement.
        self.omega = 0.0 if omega is None else float(omega)
        self._blocks = ()
        self._replaced = None
        self._range_separated = {}

    # -- construction --------------------------------------------------------

    def build(self):
        """This rank's rows of cderi, from the column blocks it evaluates."""
        if self._cderi is not None:
            return self
        # The attenuated operator is read from both molecules' environments
        # while the integrals are evaluated, and from nowhere after.
        with (range_coulomb(self.mol, None, self.omega) if self.omega
              else contextlib.nullcontext()):
            return self._build()

    def _build(self):
        """`build` under whichever operator the molecule carries now."""
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        t_build = time.time()
        mol = self.mol
        auxmol = self.auxmol = addons.make_auxmol(mol, self.auxbasis)
        low, mode = _replicated_metric_factor(mol, auxmol, self.comm)
        naux = low.shape[0]
        rank, size = self.comm.Get_rank(), self.comm.Get_size()
        start, stop = contiguous_block(naux, rank, size)
        self.row_slice = (start, stop)
        nao = mol.nao_nr()
        nao_pair = nao * (nao + 1) // 2
        left = self.max_memory - memory.current_memory_mb()
        need = (stop - start) * nao_pair * 8 / 1e6
        peak = need * (1 + DF_EXCHANGE_TRANSIENT_FRACTION)
        whole = naux * nao_pair * 8 / 1e6
        if peak > left:
            fits = int(np.ceil(whole / left)) if left > 0 else None
            raise MemoryError(
                f"this rank's {stop - start} of {naux} auxiliary rows are "
                f'{need:.0f} MB, {peak:.0f} MB while the exchange that fills '
                f'them is in flight, and {left:.0f} MB of max_memory are '
                f'left; the whole tensor is {whole:.0f} MB and {fits} ranks '
                'or more would hold it in core. This route keeps its slice in '
                'core by construction and will not spill it to disk, which is '
                'the cost it exists to remove.')
        blocks = self._blocks = _column_blocks(mol, nao_pair, naux, size)
        self.column_blocks = tuple(int(i) for i
                                   in partition(len(blocks), rank, size))
        self._cderi = _distributed_cderi(mol, auxmol, low, mode, blocks,
                                         self.comm, self.timings)
        low = None
        _counted(self.timings, 'scf_build_blocks', len(self.column_blocks))
        _spent(self.timings, 'scf_build_slices', t_build)
        log.timer('cderi rows %d:%d of %d from %d of %d column blocks '
                  '(%.0f MB here, %.0f MB whole)'
                  % (start, stop, naux, len(self.column_blocks), len(blocks),
                     need, whole), *t0)
        return self

    def reset(self, mol=None):
        super().reset(mol)
        if mol is not None:
            self.row_slice = None
            self.column_blocks = ()
            self._range_separated = {}
        return self

    def range_separated(self, omega):
        """This rank's rows of the tensor fitted with the attenuated operator
        `omega`, one object per omega, built on the first request -- which
        every rank makes at the same point of its SCF, since the build is
        collective."""
        key = '%.6f' % omega
        rows = self._range_separated.get(key)
        if rows is None:
            rows = DistributedDF(self.mol, auxbasis=self.auxbasis,
                                 comm=self.comm, timings=self.timings,
                                 omega=omega)
            rows.max_memory = self.max_memory
            rows.stdout, rows.verbose = self.stdout, self.verbose
            with blas_single_threaded():
                rows.build()
            self._range_separated[key] = rows
        rows.timings = self.timings
        return rows

    # -- the interface _DFHF calls ------------------------------------------

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        """The whole J and K of rank 0's density, from every rank's rows of
        the tensor fitted with the operator `omega` (None or 0 the bare one).
        Collective: every rank calls it at the same point of its SCF."""
        dms = _lockstep_density(dm, self.comm, self.timings)
        rows = self.range_separated(omega) if omega else self
        vj, vk = rows.partial_jk(dms, hermi, with_j, with_k, direct_scf_tol)
        return self._reduce_jk(vj, vk, with_j, with_k)

    def partial_jk(self, dm, hermi=1, with_j=True, with_k=True,
                   direct_scf_tol=1e-13):
        """This rank's rows' contribution to J and K.

        `df_jk.get_jk` on this object: the contraction is pyscf's, and it sums
        over the rows `DF.loop` yields, which are the ones this rank owns. A
        rank that owns none contributes zeros (surplus ranks are a legitimate
        configuration, and pyscf's accumulator would meet no block at all).
        """
        if self._cderi is None:
            self.build()
        t0 = time.time()
        if self._cderi.shape[0] == 0:
            zero = np.zeros(np.shape(dm))
            vj, vk = zero, zero.copy()
        else:
            vj, vk = df_jk.get_jk(self, dm, hermi, with_j, with_k,
                                  direct_scf_tol)
        _spent(self.timings, 'scf_fock_jk', t0)
        _counted(self.timings, 'scf_requests_jk')
        return (vj if with_j else None), (vk if with_k else None)

    def _reduce_jk(self, vj, vk, with_j, with_k):
        """The ranks' partials added, in one reduction of both matrices.

        The sum over the auxiliary index is re-associated here (rank
        boundaries first, then rows within a rank), so J and K differ from the
        serial ones in the last bits. `reduce_sum` adds in rank order, so they
        differ in the same last bits on every rank and in every repeat.
        """
        t0 = time.time()
        out = iter(_reduce_parts([x for x, want in ((vj, with_j), (vk, with_k))
                                  if want], self.comm))
        _spent(self.timings, 'scf_reduce', t0)
        return (next(out) if with_j else None), (next(out) if with_k else None)

    # -- installation --------------------------------------------------------

    def install(self, mf):
        """Answer this mean field's J/K until `uninstall`."""
        if mf.with_df is not self:
            self._replaced = (mf, mf.with_df)
            mf.with_df = self
        return self

    def uninstall(self):
        """Put the mean field's own DF object back, keeping the slice.

        A row slice is not a cderi: a downstream caller that iterates
        `mf.with_df.loop()` (the GW fit, the solvent screening) would read
        one rank's rows and call them the tensor. So this object answers J/K
        only while a distributed stage is running, and what is left on the
        mean field between stages is the DF the caller handed in.
        """
        pair, self._replaced = self._replaced, None
        if pair is not None:
            mf, with_df = pair
            mf.with_df = with_df
        return self

    def release(self):
        """Drop this rank's rows of every tensor it holds, the attenuated
        operators' included, and the object with them."""
        self.uninstall()
        for rows in self._range_separated.values():
            rows.release()
        self._range_separated = {}
        self._cderi = None
        self.row_slice, self.column_blocks, self._blocks = None, (), ()
        return self


class DistributedNumInt(numint.NumInt):
    """The numint whose quadrature runs on this rank's block of grid points.

    `pyscf/dft/rks.py::get_veff` reaches the grid through `ks._numint.nr_rks`,
    so this object standing in for `mf._numint` is the whole hook. Everything
    it does not override (`eval_xc_eff`, `get_rho`, the response kernels a
    later TDDFT or gradient asks for) is pyscf's and runs on whatever grid it
    is handed, which on rank 0 is still the whole one.

    Every rank calls the quadrature at the same point of the same SCF loop: it
    locksteps the density, adds this rank's points' partial to the others' and
    returns the total. The grid itself travels once per epoch, rank 0's count
    of how many times the grid it was handed has changed: the SCF's first
    Fock build prunes it by density, and that is a decision, so the other
    ranks take rank 0's points rather than repeat it.

    split: False gives rank 0 every point and the other ranks none, which is
           the quadrature whole on rank 0 with its result shared through the
           same reduction.
    """

    _keys = {'comm', 'mol', 'point_slice', 'timings', 'split'}

    def __init__(self, ni, mol, comm=None, timings=None, split=True):
        self.__dict__.update(ni.__dict__)
        comm = current_comm() if comm is None else comm
        if comm is None or comm.Get_size() == 1:
            raise ValueError(
                'DistributedNumInt divides the grid over ranks and needs a '
                'communicator of more than one; without one the mean '
                "field's own numint is the quadrature.")
        self.mol = mol
        self.comm = comm
        self.split = bool(split)
        #: [start, stop) of rank 0's grid this rank integrates, once known.
        self.point_slice = None
        #: Filled at stage boundaries when a caller wants the wall clock.
        self.timings = timings
        self._grids = None
        self._source = self._source_coords = None
        self._epoch = 0
        self._replaced = None

    # -- the interface get_veff calls ---------------------------------------

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        """(nelec, E_xc, V_xc) over the whole grid, from every rank's block."""
        return self._quadrature('nr_rks', mol, grids, xc_code, dms,
                                relativity, hermi, max_memory, verbose)

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        """The same for a spin-polarized density; the partials still add."""
        return self._quadrature('nr_uks', mol, grids, xc_code, dms,
                                relativity, hermi, max_memory, verbose)

    def nr_nlc_vxc(self, mol, grids, xc_code, dm, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
        """The non-local correlation on rank 0's whole NLC grid, shared.

        VV10 is a double sum over the points, not a sum of blocks, so rank 0
        integrates it alone and the others add zeros to the reduction: their
        own `nlcgrids` was never built while this handle was installed.
        """
        t0 = time.time()
        if self.comm.Get_rank() == 0:
            partial = numint.nr_nlc_vxc(self, mol, grids, xc_code, dm,
                                        relativity, hermi, max_memory,
                                        verbose)
        else:
            partial = (0.0, 0.0, np.zeros(np.shape(dm)[-2:]))
        _spent(self.timings, 'scf_fock_xc', t0)
        return self._reduce_xc(partial)

    def _quadrature(self, method, mol, grids, xc_code, dms, relativity, hermi,
                    max_memory, verbose):
        """One collective quadrature: rank 0's grid decision, the density
        locked, this rank's points integrated, the partials added.

        `max_memory` is rank 0's, since it sets how `block_loop` blocks the
        points and so the order of the sums inside a rank's partial.
        """
        max_memory = self._follow_grid(mol, grids, max_memory)
        dms = _lockstep_density(dms, self.comm, self.timings)
        return self._reduce_xc(self.partial_xc(method, mol, xc_code, dms,
                                               relativity, hermi, max_memory,
                                               verbose))

    def _follow_grid(self, mol, grids, max_memory):
        """Rank 0's grid decision on every rank, this rank's points of it
        taken where it is new; rank 0's `max_memory`. Collective."""
        t0 = time.time()
        header = None
        if self.comm.Get_rank() == 0 and grids.coords is not None:
            fresh = (self._source is not grids or self._grids is None
                     or self._source_coords is not grids.coords)
            header = (fresh, grids.weights.size, float(grids.cutoff),
                      max_memory)
        header = broadcast(header, self.comm)
        if header is None:
            raise ValueError(
                "rank 0's grid has no points: rank 0 builds the grid "
                "(pyscf's own Grids.build, with its pruning) and broadcasts "
                'it, so it must exist before the quadrature is asked for.')
        fresh, npoints, cutoff, max_memory = header
        if fresh:
            self._take_grid(mol, grids, npoints, cutoff)
        _spent(self.timings, 'scf_grid' if fresh else 'scf_lockstep', t0)
        return max_memory

    # -- the response quadrature an orbital-response solve calls -------------

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        """(rho0, v_xc, f_xc) at this rank's points of rank 0's grid.

        Pointwise quantities, so this rank's block of them is what its
        partial of `nr_rks_fxc` reads and nothing is reduced; the
        orbitals are rank 0's. Collective (the grid decision).
        """
        max_memory = self._follow_grid(mol, grids, max_memory)
        mo_coeff, mo_occ = lockstep((np.asarray(mo_coeff), np.asarray(mo_occ)),
                                    self.comm, check=True)
        grids_here, npoints = self._grids, self._grids.weights.size
        if npoints == 0:
            # libxc on no points: the shapes from one zero-weight point
            grids_here = _grid_slice(mol, np.zeros((1, 3)), np.zeros(1),
                                     self._grids.cutoff, 0, 1,
                                     self.make_mask)[0]
        out = numint.cache_xc_kernel(self, mol, grids_here, xc_code, mo_coeff,
                                     mo_occ, spin, max_memory)
        return tuple(np.asarray(a)[..., :npoints] for a in out)

    def nr_rks_fxc(self, mol, grids, xc_code, dm0, dms, relativity=0,
                   hermi=0, rho0=None, vxc=None, fxc=None, max_memory=2000,
                   verbose=None):
        """f_xc . dms over the whole grid, from every rank's block of points
        and of the kernel `cache_xc_kernel` returned. Collective.

        A sum over grid points like `nr_rks`'s V_xc, so the partials add in
        one reduction; the GGA hermitization inside is linear and commutes
        with it. `max_memory` is rank 0's (it blocks the points).
        """
        max_memory = self._follow_grid(mol, grids, max_memory)
        npoints = self._grids.weights.size
        if fxc is not None and np.shape(fxc)[-1] != npoints:
            raise ValueError(
                f'the response kernel holds {np.shape(fxc)[-1]} points and '
                f'this rank integrates {npoints}: it was cached on another '
                'grid than the one rank 0 now holds, so it would be read '
                'against the wrong points.')
        dms = _lockstep_density(dms, self.comm, self.timings)
        t0 = time.time()
        partial = numint.nr_rks_fxc(self, mol, self._grids, xc_code, dm0, dms,
                                    relativity, hermi, rho0, vxc, fxc,
                                    max_memory, verbose)
        _spent(self.timings, 'scf_fock_xc', t0)
        t0 = time.time()
        out = _reduce_parts([partial], self.comm)[0]
        _spent(self.timings, 'scf_reduce', t0)
        return out

    def _take_grid(self, mol, grids, npoints, cutoff):
        """This rank's points of rank 0's new grid, with a mask built for
        them; rank 0's whole grid itself where rank 0 owns every point."""
        rank, size = self.comm.Get_rank(), self.comm.Get_size()
        self._epoch += 1
        if rank == 0:
            self._source, self._source_coords = grids, grids.coords
        if not self.split:
            if rank == 0:
                self._grids, self.point_slice = grids, (0, npoints)
            else:
                empty = np.empty((0, 3)), np.empty(0)
                self._grids = _grid_slice(mol, *empty, cutoff, 0, 1,
                                          self.make_mask)[0]
                self.point_slice = (npoints, npoints)
            return
        if rank == 0:
            coords = np.ascontiguousarray(grids.coords)
            weights = np.ascontiguousarray(grids.weights)
        else:
            coords, weights = np.empty((npoints, 3)), np.empty(npoints)
        replicate(coords, weights, comm=self.comm)
        self._grids, self.point_slice = _grid_slice(
            mol, coords, weights, cutoff, rank, size, self.make_mask)

    def partial_xc(self, method, mol, xc_code, dms, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
        """This rank's points' contribution to (nelec, E_xc, V_xc).

        pyscf's own quadrature on a `Grids` that holds only those points: the
        module-level `numint.nr_rks`, because the bound one is this object's
        collective. A rank whose block is empty meets no block in `block_loop`
        and returns zeros, which is what a surplus rank should contribute.
        """
        t0 = time.time()
        out = getattr(numint, method)(self, mol, self._grids, xc_code, dms,
                                      relativity, hermi, max_memory, verbose)
        _spent(self.timings, 'scf_fock_xc', t0)
        _counted(self.timings, 'scf_requests_xc')
        return out

    def _reduce_xc(self, partial):
        """(nelec, excsum, vmat) added over the ranks in one reduction.

        All three are sums over grid points (the electron count, the
        exchange-correlation energy and its potential matrix), so a block of
        points is a partial of each. The GGA hermitization inside `nr_rks` is
        linear (V + V^T), so it commutes with the sum over blocks.
        """
        nelec, excsum, vmat = partial
        t0 = time.time()
        out = _reduce_parts([nelec, excsum, vmat], self.comm)
        _spent(self.timings, 'scf_reduce', t0)
        scalar = [float(x) if np.ndim(x) == 0 else x for x in out[:2]]
        return scalar[0], scalar[1], out[2]

    # -- installation --------------------------------------------------------

    def install(self, mf):
        """Answer this mean field's quadrature until `uninstall`; on a rank
        other than 0 its `initialize_grids` builds nothing meanwhile."""
        if mf._numint is not self:
            grids = None
            if self.comm.Get_rank() != 0:
                grids = [('initialize_grids',
                          mf.__dict__.get('initialize_grids'))]
                mf.initialize_grids = _builds_no_grid(mf)
            self._replaced = (mf, mf._numint, grids)
            mf._numint = self
        return self

    def uninstall(self):
        """Put the mean field's own numint back, keeping this rank's points.

        Everything this object does not override runs on whatever grid it is
        handed, so a response kernel a later TDDFT asks for would quietly
        integrate one rank's block: it answers the quadrature only while a
        distributed stage is running.
        """
        entry, self._replaced = self._replaced, None
        if entry is not None:
            mf, ni, grids = entry
            mf._numint = ni
            if grids is not None:
                _restore_methods(mf, grids)
        return self

    def release(self):
        """Drop this rank's points, and the object with them."""
        self.uninstall()
        self._grids = self._source = self._source_coords = None
        self.point_slice = None
        return self


def distributed_df_jk(mf, comm=None, timings=None):
    """Give `mf` a `with_df` holding one block of the auxiliary index per rank.

    Returns the object, built and installed, or None on a one-rank world,
    where `mf` is left as it came in. From here every rank's
    `mf.get_jk` is the collective; `uninstall()` puts back the mean field's
    own DF object while the slice stays. `distributed_fock` is that bracket
    and `distributed_mean_field` the whole sequence for an SCF.

    Only pyscf's own `DF` is replaced by fitted rows: its J/K is the
    contraction of the fitted three-centre tensor that the slices divide. An
    `ISDFJK` answers J/K from interpolated factors, and gets the handle that
    divides those (`distributed_isdf_jk.DistributedISDFJK`: its grid tiles
    and auxiliary shells). Any other subclass answers J/K its own way, and
    swapping in plain density-fitted rows would converge a different mean
    field without a word, so it is refused on every rank.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        return None
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError(
            'distributed J/K replaces a density fit; mf carries no with_df. '
            'Build it with mf.density_fit(), identically on every rank.')
    if isinstance(with_df, DistributedDF):
        return with_df
    if (isinstance(with_df, ISDFJK)
            or isinstance(getattr(with_df, 'source', None), ISDFJK)):
        # cycle: distributed_isdf_jk imports this module's lockstep and sum
        from src.Base.distributed_isdf_jk import distributed_isdf_jk
        return distributed_isdf_jk(mf, comm, timings)
    if type(with_df) is not df.df.DF:
        raise NotImplementedError(
            f"{type(with_df).__name__} is not pyscf's DF: the distributed SCF "
            'slices the rows of a density-fitted three-centre tensor and '
            'answers J/K from those rows only, so the J/K this with_df '
            'computes would be replaced by plain density fitting. Converge '
            'this mean field serially (comm=None, or inside '
            'distributed(None)), or build it with mf.density_fit().')
    dist = DistributedDF(mf.mol, auxbasis=with_df.auxbasis, comm=comm,
                         timings=timings)
    dist.max_memory = with_df.max_memory
    dist.stdout, dist.verbose = with_df.stdout, with_df.verbose
    dist.install(mf)
    with blas_single_threaded():
        dist.build()
    return dist


def distributed_numint(mf, comm=None, timings=None, split=True):
    """Give `mf` a `_numint` that integrates only this rank's grid points.

    Returns the object, or None where there is nothing to divide: a one-rank
    world, or a mean field with no exchange-correlation grid at all (a
    Hartree-Fock one, whose whole Fock build is already J and K). The grid
    itself is not touched here: rank 0 builds it inside its own SCF, where
    pyscf prunes it against the first density, and it travels on the first
    quadrature. `split=False` keeps every point on rank 0.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        return None
    ni = getattr(mf, '_numint', None)
    if ni is None:
        return None
    if isinstance(ni, DistributedNumInt):
        return ni
    if type(ni) is not numint.NumInt:
        raise NotImplementedError(
            f'{type(ni).__name__} is not the one-component quadrature this '
            'splits. A two- or four-component numint integrates a spinor '
            'density whose partials this reduction does not describe.')
    return DistributedNumInt(ni, mf.mol, comm, timings=timings,
                             split=split).install(mf)


def distributed_handles(mf, comm=None):
    """`mf`'s (J/K, quadrature, comm) handles under `comm`, or None.

    What a converged distributed SCF leaves on `mf._distributed`: the objects
    holding this rank's slice of the fitted tensor and of the grid, so a later
    stage that wants the same Fock pieces (the static exchange the
    quasiparticle routes build once) reuses them instead of paying the slice
    build again. They are not installed on the mean field between stages;
    `distributed_fock` is the bracket that installs them.

    None where the handles belong to another communicator, which is a mean
    field carried over from a run with a different rank count: rows cut for
    four ranks are not a partition over eight.
    """
    comm = current_comm() if comm is None else comm
    handles = getattr(mf, '_distributed', None)
    if comm is None or comm.Get_size() == 1 or handles is None:
        return None
    return handles if handles[2] is comm else None


def release_distributed(mf):
    """Drop `mf`'s handles and the slices they hold.

    The memory they hold is the slice: this rank's rows of the fitted tensor,
    naux/nranks x nao(nao+1)/2 x 8 bytes, plus this rank's grid points and
    their screening mask, and on rank 0 a reference to the whole grid.
    That is the SCF's own working set kept alive after it; a caller that will
    not build another Fock matrix from this mean field should say so here.
    """
    handles = getattr(mf, '_distributed', None)
    if handles is None:
        return mf
    for part in handles[:2]:
        if part is not None:
            part.release()
    del mf._distributed
    return mf


@contextlib.contextmanager
def distributed_fock(mf, comm=None, split_grid=True, timings=None, build=True):
    """`mf` answering J/K and the quadrature over the ranks inside the block.

    Yields the (J/K, quadrature, comm) handles, or None where there is nothing
    distributed (no communicator, a world of one, or `build=False` on a mean
    field that carries no handles). Inside the block every rank's Fock pieces
    are collectives, so every rank runs the same calls. On the way out the
    mean field has pyscf's own objects back and the handles stay on
    `mf._distributed`.

    `build=False` is for a stage that is only worth distributing if the SCF
    already paid for the slices: building them here costs the whole slice
    build again, which can exceed what the stage saves.

    Blocks nest: an inner block leaves installed what an enclosing one
    installed, so a routine may bracket its own Fock pieces whoever calls it.
    """
    comm = current_comm() if comm is None else comm
    handles = distributed_handles(mf, comm)
    if comm is not None and comm.Get_size() > 1 and not build:
        # Whether this stage is distributed at all is rank 0's to decide: one
        # rank entering the collective while another builds locally is a
        # deadlock rather than a wrong number.
        if not broadcast(handles is not None, comm):
            handles = None
        elif handles is None:
            raise RuntimeError(
                'rank 0 holds the distributed Fock handles for this mean '
                'field and this rank does not, so it cannot take part in the '
                'collectives rank 0 is about to enter. They are made and '
                'released together; release_distributed on some ranks only '
                'is what leaves them like this.')
    if handles is None and (comm is None or comm.Get_size() == 1 or not build):
        yield None
        return
    if handles is None:
        handles = (distributed_df_jk(mf, comm, timings),
                   distributed_numint(mf, comm, timings, split=split_grid),
                   comm)
        mf._distributed = handles
        parts = [part for part in handles[:2] if part is not None]
    else:
        # An enclosing block's handles stay installed when this one ends.
        parts = [part for part in handles[:2]
                 if part is not None and part._replaced is None]
        for part in parts:
            part.install(mf)
    for part in parts:
        part.timings = timings
    try:
        yield handles
    finally:
        for part in parts:
            # A handle between stages measures nothing: the next stage says
            # where its own numbers go, and a stale dict would take the
            # static exchange's Fock pieces into the SCF's record.
            part.timings = None
            part.uninstall()


def distributed_df_storage(mf, comm=None):
    """What one rank holds of the fitted tensor under this split, in GB.

    `Base.utils.memory.describe_df_storage` reads the mean field's own DF
    object, which is the whole tensor and is not what a rank stores here; and
    after `distributed_mean_field` has put that object back, unbuilt, there is
    nothing resident for it to read. This answers the same question for
    the split: how many of the naux rows this rank owns, what they weigh, and
    what the tensor would have weighed on every rank.
    """
    comm = current_comm() if comm is None else comm
    auxmol = addons.make_auxmol(mf.mol, getattr(mf.with_df, 'auxbasis', None))
    naux, nao = auxmol.nao_nr(), mf.mol.nao_nr()
    nao_pair = nao * (nao + 1) // 2
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    start, stop = contiguous_block(naux, rank, size)
    # The attenuated operators' rows, read off the handles a range-separated
    # SCF left on the mean field: their metric has its own rank.
    handles = distributed_handles(mf, comm)
    attenuated = ([] if handles is None or handles[0] is None
                  else list(getattr(handles[0], '_range_separated',
                                    {}).values()))
    return dict(naux=naux, ranks=size, rows=stop - start,
                slice_gb=(stop - start) * nao_pair * 8 / 1e9,
                whole_gb=naux * nao_pair * 8 / 1e9,
                range_separated_gb=sum(int(np.size(r._cderi)) * 8 / 1e9
                                       for r in attenuated
                                       if r._cderi is not None))


def distributed_mean_field(mf, comm=None, dm0=None, split_grid=True):
    """`mf` converged, on every rank, with its Fock build divided over them.

    Takes an unconverged density-fitted mean field that every rank built
    identically, and returns rank 0's converged one on all of them. Every rank
    runs pyscf's SCF driver unchanged against the reduced J/K and the reduced
    quadrature, whose entry locksteps keep the loops in step; the settings
    that end the loop are rank 0's from the start, and the converged spectrum,
    orbitals, occupations, energy, flag and grid are rank 0's at the end, so
    the ranks agree bitwise before anything downstream decides a grid size or
    a point count from them.

    comm: defaults to `current_comm()`.

    `split_grid=False` leaves the exchange-correlation quadrature whole on rank
    0 and divides the fit alone, which is the two effects timed apart rather
    than a cheaper route: it is the same answer, and slower.

    A distributed run leaves its wall clock, stage by stage, on
    `mf._distributed_timings` (`TIMING_KEYS`, on every rank) and its
    handles on `mf._distributed`, so the next stage to want a Fock piece over
    the same ranks reuses this rank's slices instead of building them again.
    Both travel on the mean field rather than in a return value because the
    callers keep the mean field and drop the return, and under an underscore
    because that is what pyscf's `check_sanity` leaves alone. The handles hold
    the slice: `release_distributed` is how a caller gives it back.

    Without a communicator, or on a world of one, this is `mf.kernel()` and
    nothing else: the serial path is pyscf's, bit for bit, and carries
    neither timings nor handles.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        mf.kernel(dm0=dm0)
        return mf
    timings = {}
    t_start = time.time()
    stats0 = lockstep_stats()
    _lockstep_exit_settings(mf, comm, timings)
    with distributed_fock(mf, comm, split_grid, timings):
        saved = _time_stages(mf, timings) + _driver_stages(mf, timings)
        # The checkpoint is rank 0's to write: two ranks writing one HDF5
        # file corrupt it.
        chkfile = mf.chkfile
        if comm.Get_rank() != 0:
            mf.chkfile = None
        before = {key: timings.get(key, 0.0) for key in _KERNEL_STAGES}
        t_kernel = time.time()
        try:
            # pyscf's own OpenMP runs the Fock build (the DF contraction,
            # `nr_rks` on the DFT grid), so BLAS is held at one thread and the
            # two pools stop spinning against each other; the ISDF handle's
            # GEMM stages (its row fit, Z, K) and the loop's own dense linear
            # algebra (`_DRIVER_STAGES`) take the pool back inside
            # (`blas_full_pool`).
            with blas_single_threaded():
                mf.kernel(dm0=dm0)
        finally:
            mf.chkfile = chkfile
            _restore_methods(mf, saved)
        _driver_time(timings, before, t_kernel)
    _lockstep_result(mf, comm, timings)
    stats = lockstep_stats()
    timings['scf_lockstep_mb'] = (stats['bytes'] - stats0['bytes']) / 1e6
    timings['scf_lockstep_skipped_mb'] = (stats['skipped_bytes']
                                          - stats0['skipped_bytes']) / 1e6
    return _attach_timings(mf, timings, t_start)


def _lockstep_exit_settings(mf, comm, timings):
    """Rank 0's convergence thresholds and cycle cap on every rank.

    The loop's exit is the one decision the per-call locksteps do not reach:
    the energy and the orbital gradient are computed from locked inputs, but
    the thresholds they are compared against are each rank's own, and a rank
    that stopped a cycle early would leave the others waiting in a collective
    it never enters.
    """
    t0 = time.time()
    (mf.conv_tol, mf.conv_tol_grad, mf.max_cycle,
     mf.conv_check) = lockstep((mf.conv_tol, mf.conv_tol_grad, mf.max_cycle,
                                mf.conv_check), comm)
    _spent(timings, 'scf_lockstep', t0)


def _lockstep_result(mf, comm, timings):
    """Rank 0's converged mean field onto every rank, in place.

    The loop kept the ranks' densities and orbitals locked, but the final
    eigensolve after the last Fock build and the energy it reports are each
    rank's own arithmetic; this makes them rank 0's bit for bit. Checked, as
    the density is: the arrays move only where a digest shows a rank apart.
    The quadrature grid, which only rank 0 built, follows (`_lockstep_grids`).
    """
    t0 = time.time()
    (mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.e_tot,
     mf.converged) = lockstep((mf.mo_energy, mf.mo_coeff, mf.mo_occ,
                               mf.e_tot, mf.converged), comm, check=True)
    _spent(timings, 'scf_lockstep', t0)
    t0 = time.time()
    moved = _lockstep_grids(mf, comm)
    _spent(timings, 'scf_grid' if moved else 'scf_lockstep', t0)


def _lockstep_grids(mf, comm):
    """Rank 0's built grids written onto every rank's `mf`, bit for bit.

    Rank 0 built and pruned them inside the SCF, against its own density;
    the other ranks' were never built (`_builds_no_grid`), and building them
    now would prune against the converged density and keep another point
    count. The arrays move verbatim, the screening mask with them, so no rank
    builds or prunes anything afterwards. A mean field without a built grid
    on rank 0 (Hartree-Fock, or no VV10 grid) moves nothing. True where a
    grid moved, on every rank.
    """
    rank, moved = comm.Get_rank(), False
    for name in _GRID_ATTRIBUTES:
        grids = getattr(mf, name, None)
        header = None
        if rank == 0 and grids is not None and grids.coords is not None:
            fields = []
            for field in _GRID_ARRAYS:
                a = getattr(grids, field, None)
                if field == 'screen_index' and a is grids.non0tab:
                    fields.append((field, 'non0tab'))
                elif a is None:
                    fields.append((field, None))
                else:
                    fields.append((field, (a.shape, a.dtype.str)))
            header = (float(grids.cutoff), tuple(fields))
        header = broadcast(header, comm)
        if header is None:
            continue
        moved = True
        if grids is None:
            raise RuntimeError(
                f'rank 0 holds a built mf.{name} and rank {rank} has none: '
                'the ranks were handed different mean fields.')
        cutoff, fields = header
        moving = [(field, spec) for field, spec in fields
                  if isinstance(spec, tuple)]
        arrays = [getattr(grids, field) if rank == 0
                  else np.empty(shape, dtype=np.dtype(dtype))
                  for field, (shape, dtype) in moving]
        if arrays:
            received = replicate(*arrays, comm=comm)
            arrays = [received] if len(arrays) == 1 else list(received)
        if rank == 0:
            continue
        values = dict(zip((field for field, _ in moving), arrays))
        for field, spec in fields:
            if spec is None:
                values[field] = None
            elif spec == 'non0tab':
                values[field] = values['non0tab']
        for field in _GRID_ARRAYS:
            setattr(grids, field, values[field])
        grids.cutoff = cutoff
    return moved


def _driver_time(timings, before, t_kernel):
    """`scf_driver`: the kernel's wall minus every stage timed inside it,
    which is pyscf's own loop (DIIS, the eigenproblem, the density, the
    energy, the convergence test), and the grid build folded into
    `scf_grid`; `scf_driver_other` is the driver less `_DRIVER_STAGES`."""
    inside = sum(timings.get(key, 0.0) - before[key] for key in _KERNEL_STAGES)
    timings['scf_driver'] = max(time.time() - t_kernel - inside, 0.0)
    timings['scf_driver_other'] = max(
        timings['scf_driver'] - sum(timings.get(key, 0.0)
                                    for _, key in _DRIVER_STAGES), 0.0)
    timings['scf_grid'] = (timings.get('scf_grid', 0.0)
                           + timings.pop('_grid_build', 0.0))


def _attach_timings(mf, timings, t_start):
    """`TIMING_KEYS` on `mf._distributed_timings`, JSON-clean, every rank.

    Plain floats and ints, no numpy scalars: a caller that writes the dict into
    a JSON record would otherwise find, at the last step, a record that cannot
    be serialized and a run whose timings were thrown away.
    """
    timings.pop('_grid_build', None)
    timings['scf_cycles'] = int(getattr(mf, 'cycles', 0))
    for key in TIMING_KEYS:
        timings.setdefault(key, 0 if key in _COUNT_KEYS else 0.0)
    for key, value in timings.items():
        timings[key] = int(value) if isinstance(value, int) else float(value)
    timings['scf_total'] = time.time() - t_start
    mf._distributed_timings = timings
    return mf
