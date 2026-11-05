"""Space-time RPA screening: the polarizability in imaginary time, then one
cosine transform to the imaginary-frequency axis.

With a separable (ISDF) ERI,

    (ia|jb) = sum_PQ X_o[P,i] X_v[P,a] Z[P,Q] X_o[Q,j] X_v[Q,b]

the particle-hole bubble separates into an occupied and a virtual half that each
carry only one orbital index,

    Pi_PQ(i.tau) = G^o_PQ(tau) * G^v_PQ(tau)          (elementwise)
    G^o_PQ(tau)  = sum_i X_o[P,i] X_o[Q,i] e^{+eps_i tau}
    G^v_PQ(tau)  = sum_a X_v[P,a] X_v[Q,a] e^{-eps_a tau}

so a tau point costs O(M^2 (n_occ + n_vir)) -- two GEMMs and a Hadamard product
-- against O(naux^2 n_occ n_vir) per frequency for the direct summation in
`imaginary_frequency.py`. That is the N^3-against-N^4 step, crossing over near
120 basis functions.

The transform is used one way only, on the model space it is fitted for
(Pi(i.tau) is a sum of e^{-Delta_ia tau} with Delta_ia in [e_min, e_max]), so
the minimax transform's lack of matrix duality costs nothing.

Sign convention follows `imaginary_frequency.f_rpa`: f(i.w) = -2 d/(d^2 + w^2)
and chi0 = 2 (C_ov f) C_ov^T; the cosine transform maps e^{-d tau} to
2d/(d^2 + w^2), hence the leading -2 below.

One kernel serves both sides: `polarizability_projected_tau` is the N^3
sweep and its only implementation. The energy routes stream
it into chi0 here, and the gradient chain
(`gradients.rpa.space_time_adjoint.polarizability_tau`) runs it over every tau
point through `polarizability_projected_sweep` and keeps the result whole,
because every contour-deformation frequency is a fixed linear combination of
proj(tau) and the adjoint lands back on an array of that shape. The reverse
pass mirrors this kernel's tiling block for block. A change to the arithmetic,
the tiling or the tau partition therefore reaches both, and
tests/test_space_time_shared_kernel.py pins the two callers to each other.
`polarizability_projected_rows` is the same sweep held by auxiliary rows over
the ranks (`ProjRows`), which is how the quasiparticle solves keep it: each
rank its rows of every tau slice, every transform over tau made one auxiliary
row per call so that a row is the same bits at every rank count.

Grid-row split: a tau split leaves every rank past ntau idle and hands each
rank whole (naux, naux) slices of every frequency. `chi0_frequency_rows`
splits the (tau point, grid-row tile) pairs of the sweep instead, the tiles
being the serial kernel's (`polarizability_tiles`), so the work divides past
ntau ranks and a rank forms only its tiles' (rows, M) Green's-function blocks.
proj(tau) is then a sum over the ranks' tiles, reduce-scattered to auxiliary
rows (re-associated, the one reduced sum of the sweep), and chi0(i.nu) is
accumulated on those rows with the serial elementwise update, so every row of
chi0 is the serial update of its proj rows, bitwise.

Grid tiles: the factors themselves are whole on no rank when the sweep runs
on fixed grid tiles (`grid_factor_tiles`, `polarizability_tile_rows`): each
(tau point, p tile) item holds its p rows while every q tile streams past
from its owner, so a rank holds its tiles and one pass's working set, which
falls with the rank count. The tiles are set by the grid and the budget, not
by the rank count, so each item is the same calls at every rank count; the
static screening adds the items in tile order along a chain of the ranks
holding them and is the one-rank W bit for bit, while the GW chi0
reduce-scatters the ranks' partials as above.

`three_index_slice`, `b_block` and `three_index_ov` sit here for the same
reason: one bra state's pair density, an arbitrary index block and the
particle-hole block are the forward halves of the three-index chain, read by
the contour-deformation energy route and the gradient chain alike, while their
adjoints stay in `gradients.rpa.space_time_adjoint` and `gradients.gwbse.bse_isdf`
beside the rest of the reverse pass.

`rpa_correlation_energy_space_time` is the dRPA energy's own forward sweep, and
`want_tape=True` hands back what a reverse pass through it consumes --
proj(tau) and the two partitions it was built on, so a reverse pass over ranks
splits as the forward sweep did. The gradient's
`space_time_adjoint.rpa_energy_and_adjoint` calls it rather than sweeping
again, so there is one E_c^dRPA in the code and the adjoint cannot drift from
the energy it differentiates.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- building the
response in imaginary time, where the occupied and virtual sums decouple.
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) -- the
minimax imaginary-time/frequency quadrature the transform below runs on.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) -- the separable RI
supplying X_o and X_v.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.Base.numerics.constants import (DEFAULT_TAU_TARGET,
                                         ISDF_SQUARE_TILE_ROWS, ISDF_TILE_GB)
from src.Base.numerical_flags import flag
from src.Base.factorization.sliced_factors import SlicedFactors, column_tiles
from src.Base.numerics.numerics import Numerics, numerics_or_default
from src.Base.utils.parallel.memory import fits
from src.Base.utils.parallel.mpi_grid import (agreement, allgather_rows,
                                              contiguous_block, current_comm,
                                              exchange_rows, ordered_chain_sum,
                                              ordered_sum, partition,
                                              reduce_scatter_rows, reduce_sum)
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.Base.orbital_spaces import (get_occ_virt_indices,
                                     particle_hole_window, transition_range)


@dataclass(frozen=True)
class RPAEnergyTape:
    """The forward sweep of E_c^dRPA, as its reverse pass reads it.

    proj_tau is the one whole object of the N^3 route, (ntau, naux, naux), from
    which every frequency is a fixed linear combination; the two index sets are
    the tau points and the frequencies this rank owns, so the reverse pass
    splits as the forward did and its reductions carry the same
    disjoint slots. Both are None in a serial run.
    """

    proj_tau: np.ndarray
    tau_indices: np.ndarray
    freq_indices: np.ndarray


class FrequencyBlock(NamedTuple):
    """One serial block, or one round, of the frequency axis as
    `ProjRows.blocks` yields it.

    block:  every frequency of the block, the serial block's own list, or of
            the round, one frequency per rank (`frequency_rounds`)
    owners: the rank that factorizes each of them, or None where every rank
            takes every frequency whole
    ks:     the frequencies this rank takes, in block order
    chi0:   their chi0(i.nu), whole, (len(ks), naux, naux); a reverse pass
            overwrites them with their adjoints before `ProjRows.fold`
    serial: for a round, the pass's `SerialFolds`, which `ProjRows.fold`
            folds the adjoint rows through by serial block; None for a block
    """

    block: list
    owners: list
    ks: list
    chi0: np.ndarray
    serial: object = None


class SerialFolds:
    """The adjoint rows of one reverse frequency pass taken by rounds, held
    until their serial block is whole: each serial block is then folded into
    projbar with the serial call on the serial block's rows, in block order,
    so the fold is the serial fold bitwise whatever the rounds."""

    def __init__(self, spans):
        self.spans = list(spans)
        self.rows = {}

    def take(self, k, rows):
        """Frequency k's adjoint rows, (nrows, naux), this rank's."""
        if k in self.rows:
            raise ValueError(f'frequency {k} handed back twice in one pass')
        self.rows[k] = rows

    def whole(self):
        """((k0, k1), its rows stacked (k1 - k0, nrows, naux)) for every
        serial block whose rows are all here, in block order, each once."""
        while self.spans and all(k in self.rows
                                 for k in range(*self.spans[0])):
            k0, k1 = self.spans.pop(0)
            yield (k0, k1), np.stack([self.rows.pop(k)
                                      for k in range(k0, k1)])


class ProjRows:
    """proj(tau) held by auxiliary rows: this rank's rows [r0, r1) of every tau
    slice, (ntau, r1 - r0, naux), the `contiguous_block` of naux it owns.

    Every consumer of proj(tau) is linear in it and elementwise in (P, Q) --
    chi0(i.nu) = sum_tau cosft[nu, tau] proj(tau), W at a real frequency
    below the gap likewise -- so a rank transforms its own rows, and what
    needs a whole matrix gets one frequency or one tau point at a time: the
    rank that factorizes a frequency gathers that chi0, the rank that sweeps a
    tau point gathers that slice. The adjoints come back the same way, so
    projbar lives in the same rows (`zeros_like`, `fold`). The grid-row split
    of the GW stage holds chi0(i.nu) and Wt(i.tau) the same way, the first
    axis a frequency or a tau point (`gather_slices` hands a rank any slices
    it names).

    The rows are the same bits at every rank count. A GEMM's rows depend on
    the call's shape (`owned_frequency_blocks`), and a rank's row block is a
    shape that changes with the rank count, so every transform over tau here
    is made one auxiliary row at a time (`transform_rows`,
    `fold_frequency_rows`): the same (c, (ntau, naux)) call for a row whether
    one rank holds all of them or eight share them. A serial ProjRows is the
    whole sweep, its rows every row, and takes the same calls.

    One whole-matrix read is answered: np.tensordot(c, proj,
    axes=(0, 0)) for a weight vector c, the real-frequency screening's, which
    comes back whole on every rank (`transform`). Every other numpy function
    raises rather than gather the whole array behind the caller's back.
    """

    def __init__(self, rows, naux, comm=None):
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        self.rows = rows
        self.naux = int(naux)
        self.comm = comm if size > 1 else None
        self.bounds = [contiguous_block(self.naux, r, size)
                       for r in range(size)]
        self.r0, self.r1 = self.bounds[rank]
        if rows.shape[1:] != (self.r1 - self.r0, self.naux):
            raise ValueError(f'rows {rows.shape} are not rank {rank} of {size}'
                             f"'s rows of naux {self.naux}")

    @property
    def shape(self):
        """The whole array's shape, (ntau, naux, naux)."""
        return (self.rows.shape[0], self.naux, self.naux)

    @property
    def nbytes(self):
        """What this rank holds."""
        return self.rows.nbytes

    def __array__(self, *args, **kwargs):
        raise TypeError('proj(tau) is held by auxiliary rows; a whole-array '
                        'read has to name the collective it needs')

    def __array_function__(self, func, types, args, kwargs):
        # by name: numpy hands over its own function, whatever `np` is here
        if (getattr(func, '__name__', None) == 'tensordot' and len(args) == 2
                and args[1] is self and np.ndim(args[0]) == 1
                and tuple(kwargs.get('axes', ())) == (0, 0)):
            return self.transform(args[0])
        return NotImplemented

    def zeros_like(self):
        """An adjoint accumulator in the same rows, zeroed."""
        return ProjRows(np.zeros_like(self.rows), self.naux, self.comm)

    def size_rank(self):
        return ((1, 0) if self.comm is None
                else (self.comm.Get_size(), self.comm.Get_rank()))

    def transform(self, c):
        """sum_tau c[tau] proj(tau), (naux, naux), whole on every rank: each
        rank transforms its rows and the rows are gathered verbatim."""
        own = transform_rows(np.asarray(c, float), self.rows)
        if self.comm is None:
            return own
        out = np.empty((self.naux, self.naux))
        out[self.r0:self.r1] = own
        return allgather_rows(out, self.comm)

    def blocks(self, cosft_wt, tile_gb, freq_indices=None, live=3):
        """`FrequencyBlock` for every serial block of the frequency axis, or
        with owners for every round of it, on every rank and in the same
        order, since every rank holds rows of every frequency.

        freq_indices: this rank's frequencies, the round-robin `partition`;
        each is gathered whole to it alone, by rounds (`_rounds`): one
        exchange hands each rank its frequency of the round, so every owner
        factorizes at once. None gives every rank every frequency, gathered to
        all. Every rank transforms its rows of the whole serial block, one
        auxiliary row at a time, so a frequency's rows are the same bits
        whoever owns it.
        """
        size, rank = self.size_rank()
        nfreq = cosft_wt.shape[0]
        spans = frequency_blocks(nfreq, self.naux, tile_gb, live=live)
        if freq_indices is not None and size > 1:
            mine = sorted(int(k) for k in np.atleast_1d(freq_indices))
            if mine != list(partition(nfreq, rank, size)):
                raise ValueError('proj(tau) by rows routes each frequency to '
                                 'its round-robin owner; this rank was '
                                 f'handed {mine}')
            yield from self._rounds(cosft_wt, spans)
            return
        nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
        for k0, k1 in spans:
            block = list(range(k0, k1))
            rows = transform_rows(cosft_wt[block], self.rows)
            if size == 1:
                yield FrequencyBlock(block, None, block, rows)
                continue
            chi0 = np.empty((len(block), self.naux, self.naux))
            for m in range(len(block)):
                exchange_rows(rows[m], [(0, nrows)] * size, chi0[m],
                              self.bounds, self.comm)
            rows = None
            yield FrequencyBlock(block, None, block, chi0)

    def _rounds(self, cosft_wt, spans):
        """`FrequencyBlock` for every round of the frequency axis: rank j
        owns the round's j-th frequency, which is its round-robin one.

        Each serial block is transformed with the serial call, as `blocks`
        does, and its rows wait in `ready` until their round; one exchange
        per round sends every frequency's rows to its owner, so the owners'
        factorizations run concurrently between two collectives instead of
        one owner at a time. A round holds one (naux, naux) of rows a rank
        on top of the serial block."""
        size, rank = self.size_rank()
        nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
        todo, ready = iter(spans), {}
        serial = SerialFolds(spans)
        for rnd in frequency_rounds(cosft_wt.shape[0], size):
            while not all(k in ready for k in rnd):
                k0, k1 = next(todo)
                block = list(range(k0, k1))
                rows = transform_rows(cosft_wt[block], self.rows)
                ready.update(zip(block, rows))
            send = np.empty((len(rnd) * nrows, self.naux))
            for j, k in enumerate(rnd):
                send[j * nrows:(j + 1) * nrows] = ready[k]
            for k in rnd:
                ready.pop(k, None)
            ks = rnd[rank:rank + 1]
            chi0 = np.empty((len(ks), self.naux, self.naux))
            exchange_rows(send, round_ranges(len(rnd), nrows, size),
                          chi0[0] if ks else empty,
                          [b if ks else (0, 0) for b in self.bounds],
                          self.comm)
            send = None
            yield FrequencyBlock(rnd, list(range(len(rnd))), ks, chi0, serial)

    def fold(self, cosft_wt, fb):
        """self += sum_{k in fb.block} cosft_wt[k, :] fb.chi0(k), on this
        rank's rows: the adjoint of `blocks`' transform, fb.chi0 holding each
        owned frequency's adjoint whole. The owners hand every rank its rows,
        and each rank adds the whole serial block into its rows, one row at a
        time -- the serial association, at every rank count. A round's
        adjoints come back in one exchange and wait in `fb.serial` until
        their serial block is whole, which is then folded as one."""
        size, rank = self.size_rank()
        if fb.serial is not None:
            nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
            rows = np.empty((len(fb.block) * nrows, self.naux))
            exchange_rows(fb.chi0[0] if fb.ks else empty,
                          [b if fb.ks else (0, 0) for b in self.bounds], rows,
                          round_ranges(len(fb.block), nrows, size), self.comm)
            for j, k in enumerate(fb.block):
                fb.serial.take(k, rows[j * nrows:(j + 1) * nrows])
            for (k0, k1), blk in fb.serial.whole():
                fold_frequency_rows(self.rows, cosft_wt[list(range(k0, k1))],
                                    blk)
            return
        if size == 1 or fb.owners is None:
            rows = fb.chi0[:, self.r0:self.r1]
        else:
            nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
            rows = np.empty((len(fb.block), nrows, self.naux))
            for m, (k, owner) in enumerate(zip(fb.block, fb.owners)):
                here = owner == rank
                exchange_rows(
                    fb.chi0[fb.ks.index(k)] if here else empty,
                    [b if here else (0, 0) for b in self.bounds], rows[m],
                    [(0, nrows) if r == owner else (0, 0)
                     for r in range(size)], self.comm)
        fold_frequency_rows(self.rows, cosft_wt[fb.block], rows)

    def tau_slices(self, tau_indices=None):
        """(k, slice k whole) for this rank's tau points, gathered from every
        rank's rows one tau point a rank per round, so every rank takes part
        in every round. tau_indices: the round-robin `partition`, or None for
        every point on every rank. The slice buffer is reused: read it before
        asking for the next."""
        size, rank = self.size_rank()
        ntau = self.rows.shape[0]
        if size == 1:
            which = range(ntau) if tau_indices is None else tau_indices
            for k in np.atleast_1d(np.asarray(list(which), dtype=int)):
                yield int(k), self.rows[k]
            return
        nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
        flat = self.rows.reshape(ntau * nrows, self.naux)
        slab = np.empty((self.naux, self.naux))
        if tau_indices is None:
            rounds = [[k] * size for k in range(ntau)]
        else:
            mine = sorted(int(k) for k in np.atleast_1d(tau_indices))
            if mine != list(partition(ntau, rank, size)):
                raise ValueError('proj(tau) by rows gathers each tau point to '
                                 'its round-robin owner; this rank was handed '
                                 f'{mine}')
            rounds = [[r + j * size for r in range(size)]
                      for j in range(-(-ntau // size))]
        for ks in rounds:
            exchange_rows(
                flat, [(k * nrows, (k + 1) * nrows) if k < ntau else (0, 0)
                       for k in ks],
                slab if ks[rank] < ntau else empty,
                [b if ks[rank] < ntau else (0, 0) for b in self.bounds],
                self.comm)
            if ks[rank] < ntau:
                yield ks[rank], slab

    def gather_slices(self, wants):
        """(k, slice k whole) for every k of `wants`, this rank's own list in
        its order, whatever the other ranks ask for: round t hands each rank
        the t-th slice of its list from every rank's rows, verbatim. The
        lists travel first (one small allgather), so the rounds pair. The
        slice buffer is reused: read it before asking for the next."""
        size, rank = self.size_rank()
        wants = [int(k) for k in wants]
        if size == 1:
            for k in wants:
                yield k, self.rows[k]
            return
        lists = self.comm.allgather(wants)
        nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
        flat = self.rows.reshape(self.rows.shape[0] * nrows, self.naux)
        slab = np.empty((self.naux, self.naux))
        for t in range(max(len(own) for own in lists)):
            ks = [own[t] if t < len(own) else None for own in lists]
            here = ks[rank] is not None
            exchange_rows(
                flat, [(k * nrows, (k + 1) * nrows) if k is not None
                       else (0, 0) for k in ks],
                slab if here else empty,
                [b if here else (0, 0) for b in self.bounds], self.comm)
            if here:
                yield ks[rank], slab


def tile_rows(M, tile_memory_gb, per_row_bytes):
    """Grid rows per tile so that `per_row_bytes` x rows fits in
    tile_memory_gb: at least one row, at most all M of them."""
    return fits(tile_memory_gb, per_row_bytes, cap=M)


def square_tile_rows(rows):
    """`rows` of a (rows, M) tile of an (M, M) intermediate, at most
    ISDF_SQUARE_TILE_ROWS, so that no rank forms the (M, M) array whole
    where M exceeds it, whatever the memory budget allows."""
    return max(1, min(int(rows), ISDF_SQUARE_TILE_ROWS))


def polarizability_work(M, tile_memory_gb=ISDF_TILE_GB):
    """The two (rows, M) Green's-function tiles `polarizability_projected_tau`
    works in, sized for this M and budget, to be reused across tau points."""
    rows = square_tile_rows(tile_rows(M, tile_memory_gb, 3 * M * 8))
    return np.empty((rows, M)), np.empty((rows, M))


def polarizability_tiles(M, tile_memory_gb=ISDF_TILE_GB):
    """[(p0, p1)] grid-row tiles of `polarizability_projected_tau`, set by M
    and the budget alone: the addends of proj(tau), whoever sums them."""
    rows = square_tile_rows(tile_rows(M, tile_memory_gb, 3 * M * 8))
    return [(p0, min(p0 + rows, M)) for p0 in range(0, M, rows)]


def sweep_waves(ntau, size):
    """[(k0, k1)] windows of at most `size` tau points: a (tau, item) sweep
    split over the ranks one window at a time, so a rank's contiguous share of
    a window spans at most two tau points (one alone serially)."""
    return [(k0, min(k0 + size, ntau)) for k0 in range(0, ntau, size)]


def wave_items(k0, k1, nitems, rank, size):
    """This rank's (k, j) of the window [k0, k1) with `nitems` items per tau
    point: the `contiguous_block` of the tau-major pairs, so the ranks' work
    differs by at most one item whatever divides what."""
    i0, i1 = contiguous_block((k1 - k0) * nitems, rank, size)
    return [(k0 + i // nitems, i % nitems) for i in range(i0, i1)]


def split_branches(X, eps, nocc, mu=None, branches=None):
    """(X_o, X_v, e_o, e_v, mu, occ, virt): the collocation split at the gap.

    The energies come back shifted by mu, which defaults to the middle of the
    gap. Any chemical potential inside the gap makes every e_v - e_o positive,
    which is what keeps both exponentials in the Green's functions decaying.
    The occupied and virtual blocks are made contiguous here, once, so the
    GEMMs downstream never copy them.

    X may be `SlicedFactors`: the two blocks are then gathered whole from the
    ranks' grid rows of X_mo, one collective each, and the whole X_mo never
    exists -- every kernel that reads the collocation only through this split
    takes sliced factors in its place.

    branches: (X_o, X_v) a caller already split from X, returned in their
    place, so two sweeps on one route gather a sliced X once.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = particle_hole_window(eps, nocc)[2]
    if branches is not None:
        X_o, X_v = branches
    elif isinstance(X, SlicedFactors):
        X_o, X_v = X.branches(occ, virt)
    else:
        X_o, X_v = np.ascontiguousarray(X[:, occ]), np.ascontiguousarray(X[:, virt])
    return X_o, X_v, eps[occ] - mu, eps[virt] - mu, mu, occ, virt


def polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                 tile_memory_gb=ISDF_TILE_GB, out=None,
                                 work=None, tiles=None):
    """chi0 at one imaginary time, already projected to the auxiliary basis:

        proj_ab(tau) = -2 sum_PQ D[P,a] (Go_PQ Gv_PQ) D[Q,b]

    Tiled over grid rows, which is exact because the expression is a sum over
    them, so the M x M object never exists: rows set by the budget and at most
    ISDF_SQUARE_TILE_ROWS (`square_tile_rows`). Contract the Pi block with D first:
    the other order builds an (naux, M) intermediate instead.

    The one kernel of the N^3 sweep, for the energy routes and the gradient
    chain alike (see the module docstring). The reverse pass in
    `gradients.rpa.space_time_adjoint.polarizability_backward` mirrors this tiling,
    so the row budget here, 3 * M * 8 bytes per row, is a convention and not a
    private detail.

    out:  (naux, naux) buffer the result is written into, so a sweep over many
          tau points allocates nothing per point. Zeroed here; allocated when
          None.
    work: the two (rows, M) tiles from `polarizability_work`, reused across
          calls for the same reason. Their contents are never read.
    tiles: the `polarizability_tiles` to sum, in order; None is all of them,
          the whole proj(tau), and a subset is that subset's partial sum.
    """
    M = X_o.shape[0]
    naux = D.shape[1]
    rows = square_tile_rows(tile_rows(M, tile_memory_gb, 3 * M * 8))
    if tiles is None:
        tiles = polarizability_tiles(M, tile_memory_gb)
    if work is None:
        work = polarizability_work(M, tile_memory_gb)
    Go_buf, Gv_buf = work
    for buf in (Go_buf, Gv_buf):
        if buf.shape[0] < rows or buf.shape[1] != M:
            raise ValueError(f'work tile is {buf.shape}; this M and budget '
                             f'need at least ({rows}, {M})')
    if out is None:
        out = np.zeros((naux, naux))
    else:
        if out.shape != (naux, naux):
            raise ValueError(f'out is {out.shape}, need ({naux}, {naux})')
        out[...] = 0.0
    eo_t, ev_t = np.exp(e_o * tau), np.exp(-e_v * tau)
    for p0, p1 in tiles:
        b = p1 - p0
        Go, Gv = Go_buf[:b], Gv_buf[:b]
        np.matmul(X_o[p0:p1] * eo_t, X_o.T, out=Go)   # (b, M)
        np.matmul(X_v[p0:p1] * ev_t, X_v.T, out=Gv)   # (b, M)
        Go *= Gv                                       # Pi block, in Go's tile
        out += D[p0:p1].T @ (Go @ D)                   # (naux, naux)
    out *= -2.0
    return out


def polarizability_projected_sweep(X, D, eps, nocc, tau_points, mu=None,
                                   tau_indices=None,
                                   tile_memory_gb=ISDF_TILE_GB):
    """proj(tau) on every time point, (ntau, naux, naux): the one whole object
    of the N^3 route.

    `chi0_imaginary_frequency` folds each point into chi0 as it is built and
    never holds this array. The gradient chain needs it whole, since every
    contour-deformation frequency is a fixed linear combination of it and the
    adjoint comes back with the same shape. Same kernel, kept instead of
    streamed.

    tau_indices: the points to compute; the other slots stay zero, so the sum
    over disjoint subsets is the full sweep. That is the partition
    `Base.utils.parallel.mpi_grid` hands out, and an all-reduce of the result is exact.
    """
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    tau_points = np.asarray(tau_points, float)
    naux = D.shape[1]
    proj_tau = np.zeros((len(tau_points), naux, naux))
    work = polarizability_work(X.shape[0], tile_memory_gb)
    which = (range(len(tau_points)) if tau_indices is None
             else np.atleast_1d(tau_indices))
    for k in which:
        polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau_points[k],
                                     tile_memory_gb=tile_memory_gb,
                                     out=proj_tau[k], work=work)
    return proj_tau


def polarizability_projected_rows(X, D, eps, nocc, tau_points, mu=None,
                                  tile_memory_gb=ISDF_TILE_GB, comm=None):
    """proj(tau) as `ProjRows`: this rank's auxiliary rows of every tau slice.

    The same sweep as `polarizability_projected_sweep`, split over tau the
    same way, but no rank holds the (ntau, naux, naux) array: in each round
    every rank computes its next tau point into one (naux, naux) buffer and
    hands each rank that point's rows (`exchange_rows`). Each slice is its
    tau owner's verbatim, so every row is the one a zero-padded all-reduce of
    the sweep gives. Serially the result wraps the whole sweep.

    comm: None is `current_comm()`. X and D whole.
    """
    comm = current_comm() if comm is None else comm
    size = 1 if comm is None else comm.Get_size()
    naux = D.shape[1]
    if size == 1:
        return ProjRows(polarizability_projected_sweep(
            X, D, eps, nocc, tau_points, mu=mu,
            tile_memory_gb=tile_memory_gb), naux)
    rank = comm.Get_rank()
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    tau_points = np.asarray(tau_points, float)
    ntau = len(tau_points)
    bounds = [contiguous_block(naux, r, size) for r in range(size)]
    nrows = bounds[rank][1] - bounds[rank][0]
    rows = np.empty((ntau, nrows, naux))
    flat = rows.reshape(ntau * nrows, naux)
    slab, empty = np.empty((naux, naux)), np.empty((0, naux))
    work = polarizability_work(X_o.shape[0], tile_memory_gb)
    for j in range(-(-ntau // size)):
        k = rank + j * size
        if k < ntau:
            polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau_points[k],
                                         tile_memory_gb=tile_memory_gb,
                                         out=slab, work=work)
        lands = [((r + j * size) * nrows, (r + j * size + 1) * nrows)
                 if r + j * size < ntau else (0, 0) for r in range(size)]
        exchange_rows(slab if k < ntau else empty,
                      [b if k < ntau else (0, 0) for b in bounds], flat,
                      lands, comm)
    return ProjRows(rows, naux, comm)


def chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=None,
                             tau_indices=None, tile_memory_gb=ISDF_TILE_GB):
    """chi0(i.omega) in the DF auxiliary basis, shape (nfreq, naux, naux).

    X :    (M, norb) collocation in the MO basis, or `SlicedFactors`, whose
           occupied and virtual columns `split_branches` gathers whole.
    D :    (M, naux) Coulomb factor, Z = D D^T; carries the result back to the
           auxiliary basis every DF consumer speaks.
    grid : TimeFrequencyGrid with an imaginary-time axis.

    Same chi0 as `solve_rpa_screening_df`, so W follows as [I - chi0]^-1.

    chi0(i.w) = -2 sum_tau cosft_wt[w,tau] (D^T Pi(tau) D), each Pi(i.tau)
    projected and accumulated as it is built: the peak is chi0 itself, one
    (nfreq, naux, naux) update of it and the two (rows, M) tiles of the
    kernel; every rank of a tau split holds all of that, which
    `chi0_frequency_rows` does not.
    """
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    naux = D.shape[1]
    chi0 = np.zeros((grid.nfreq, naux, naux))
    for _, term in chi0_tau_terms(X_o, X_v, e_o, e_v, D, grid, tau_indices,
                                  tile_memory_gb):
        chi0 += term
    return chi0


def chi0_tau_terms(X_o, X_v, e_o, e_v, D, grid, tau_indices=None,
                   tile_memory_gb=ISDF_TILE_GB):
    """(k, cosft_wt[:, k] proj(tau_k)) for the tau points `tau_indices`
    (every point by default), in order: the addends of
    `chi0_imaginary_frequency`'s sum over tau, one (nfreq, naux, naux)
    each."""
    naux = D.shape[1]
    # One projected point and the two Green's-function tiles, allocated once
    # for the whole sweep and handed to the kernel every time.
    proj = np.empty((naux, naux))
    work = polarizability_work(X_o.shape[0], tile_memory_gb)
    which = (range(grid.ntau) if tau_indices is None
             else np.atleast_1d(tau_indices))
    for k in which:
        polarizability_projected_tau(X_o, X_v, e_o, e_v, D, grid.tau_points[k],
                                     tile_memory_gb=tile_memory_gb, out=proj,
                                     work=work)
        yield int(k), grid.cosft_wt[:, k, None, None] * proj


def polarizability_rows_sweep(X_o, X_v, e_o, e_v, D, tau_points, comm=None,
                              tile_memory_gb=ISDF_TILE_GB):
    """(k, this rank's auxiliary rows of proj(tau_k)) for every tau point in
    order, (r1 - r0, naux), the rows its `contiguous_block` of naux owns.

    The (tau point, `polarizability_tiles` tile) pairs go over the ranks in
    `sweep_waves` windows (`wave_items`): a rank sums its tiles of each of its
    at most two tau points of the window into one (naux, naux) partial, and
    each point's partials are reduce-scattered to rows (a rank without tiles
    of it hands in zeros). So the work divides past ntau ranks, a rank holds
    two partials and one zero slice besides its (rows, M) tiles, and proj is
    a sum re-associated over the ranks' tiles. Serially the window is one tau
    point and its whole kernel call, bitwise. The row buffer is reused: read
    it before asking for the next point.
    """
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    M, naux = X_o.shape[0], D.shape[1]
    tiles = polarizability_tiles(M, tile_memory_gb)
    r0, r1 = contiguous_block(naux, rank, size)
    rows = np.empty((r1 - r0, naux))
    work = polarizability_work(M, tile_memory_gb)
    zero = None
    for k0, k1 in sweep_waves(len(tau_points), size):
        mine = wave_items(k0, k1, len(tiles), rank, size)
        partial = {}
        for k in sorted({k for k, _ in mine}):
            partial[k] = polarizability_projected_tau(
                X_o, X_v, e_o, e_v, D, tau_points[k],
                tile_memory_gb=tile_memory_gb, work=work,
                tiles=[tiles[j] for kk, j in mine if kk == k])
        for k in range(k0, k1):
            send = partial.pop(k, None)
            if send is None:
                if zero is None:
                    zero = np.zeros((naux, naux))
                send = zero
            reduce_scatter_rows(send, comm, out=rows)
            del send
            yield k, rows


def chi0_frequency_rows(X, D, eps, nocc, grid, mu=None, comm=None,
                        tile_memory_gb=ISDF_TILE_GB, tiles=None):
    """chi0(i.omega) held by auxiliary rows, `ProjRows` over the frequency
    axis: this rank's rows of every frequency, (nfreq, r1 - r0, naux).

    The M^2 sweep is split over grid-row tiles as well as tau points; each
    point's proj rows are folded in with `chi0_imaginary_frequency`'s own
    update, elementwise, so a rank's chi0 rows are the serial update of the
    reduced proj rows bitwise and no rank holds a whole (naux, naux) slice of
    chi0.

    With `tiles` (`grid_factor_tiles`), or X as `SlicedFactors`, the sweep
    runs on the fixed grid tiles (`polarizability_tile_rows`): a rank holds
    its tiles of X and D and never either array whole. Otherwise X and D are
    whole, the tiles are the serial kernel's (`polarizability_rows_sweep`)
    and serially it is `chi0_imaginary_frequency`, bitwise. comm: None is
    `current_comm()`.
    """
    comm = current_comm() if comm is None else comm
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    if tiles is None and isinstance(X, SlicedFactors):
        tiles = grid_factor_tiles(X, D, eps, nocc, mu, comm,
                                  layout_numerics(tile_memory_gb))
    if tiles is not None:
        naux = tiles.naux
        r0, r1 = contiguous_block(naux, rank, size)
        chi0 = np.zeros((grid.nfreq, r1 - r0, naux))
        for k, rows in polarizability_tile_rows(tiles, grid.tau_points, comm):
            chi0 += grid.cosft_wt[:, k, None, None] * rows
        return ProjRows(chi0, naux, comm)
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    naux = D.shape[1]
    r0, r1 = contiguous_block(naux, rank, size)
    chi0 = np.zeros((grid.nfreq, r1 - r0, naux))
    for k, rows in polarizability_rows_sweep(X_o, X_v, e_o, e_v, D,
                                             grid.tau_points, comm,
                                             tile_memory_gb):
        chi0 += grid.cosft_wt[:, k, None, None] * rows
    return ProjRows(chi0, naux, comm)


def layout_tile_gb(numerics):
    """GB of one pass of a grid-tile sweep: `numerics.layout.tile_gb`,
    ISDF_TILE_GB where it is None."""
    gb = numerics_or_default(numerics).layout.tile_gb
    return ISDF_TILE_GB if gb is None else float(gb)


def layout_numerics(gb):
    """Numerics whose layout budget is `gb`, for a caller that was handed
    the budget alone: the defaults where it is theirs."""
    base = Numerics()
    if gb is None or float(gb) == layout_tile_gb(base):
        return base
    return base.derive('the caller\'s working-set budget',
                       layout={'tile_gb': float(gb)})


def grid_tile_rows(npts, ncol, numerics=None):
    """Rows of one fixed tile of a distributed grid sweep: four (rows, ncol)
    blocks within the layout's budget, at most ISDF_SQUARE_TILE_ROWS
    (`square_tile_rows`). Set by the grid, the column count and the budget
    alone and never by the rank count, so a sweep made of tiles computes
    every tile the same way at every rank count."""
    return square_tile_rows(tile_rows(int(npts), layout_tile_gb(numerics),
                                      4 * 8 * int(ncol)))


class TiledFactors(NamedTuple):
    """The separable factors as the distributed M^2 sweeps read them:
    `tiles`, [X_mo | D] in fixed grid tiles (`GridTileRows`), the occupied
    columns first; `nocc` and `nmo` split the columns, X_o = [:, :nocc],
    X_v = [:, nocc:nmo], D = [:, nmo:]; e_o and e_v the energies shifted by
    the chemical potential, as `split_branches` hands them out."""
    tiles: object
    nocc: int
    nmo: int
    e_o: np.ndarray
    e_v: np.ndarray

    @property
    def naux(self):
        return self.tiles.rows.shape[1] - self.nmo


def grid_factor_tiles(X, D, eps, nocc, mu=None, comm=None, numerics=None):
    """`TiledFactors` of the MO collocation X and the Coulomb factor D, the
    layout of every distributed sweep of the polarizability.

    X may be `SlicedFactors` over `comm`, D then None (their own D), the
    same factors, or a whole D on their grid in another gauge (a reaction
    field's bare factor), of which each rank cuts its own rows
    (`SlicedFactors.coulomb_rows`): each rank hands in its own rows and
    keeps its tiles, so X is whole on no rank. Whole arrays are cut into the
    tiles directly. The tile rows are set by the grid, the column count and
    `numerics.layout.tile_gb` (`grid_tile_rows`), never by the rank count,
    and a sweep's pass works within the same budget. comm: None is
    `current_comm()`."""
    comm = current_comm() if comm is None else comm
    size = 1 if comm is None else comm.Get_size()
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = particle_hole_window(eps, nocc)[2]
    if isinstance(X, SlicedFactors):
        X.require(comm)
        parts, npts, whole = [X.X_mo, X.coulomb_rows(D)], X.npts, False
    else:
        parts, npts, whole = [X, D], X.shape[0], True
    ncol = parts[0].shape[1] + parts[1].shape[1]
    gb = layout_tile_gb(numerics)
    tiles = column_tiles(parts, npts, grid_tile_rows(npts, ncol, numerics),
                         comm if size > 1 else None, whole=whole)
    tiles.pass_gb = gb
    return TiledFactors(tiles, len(occ), len(occ) + len(virt),
                        eps[occ] - mu, eps[virt] - mu)


def _proj_items(tiled, tau_points, items, on_done):
    """Every item (k, p) of `items` through the tile sweep: the grid rows of
    tile p of proj(tau_k) before its D_p^T projection,

        Y_p = sum_q (G^o(tau_k) * G^v(tau_k))[p, q] D[q],   q ascending,

    and on_done(k, p, D_p, Y_p) in the items' order. Each (p, q) block is
    the same three GEMMs at every rank count, on the same tile rows."""
    F, no, nmo = tiled.tiles, tiled.nocc, tiled.nmo
    naux = tiled.naux

    def start(k, p, rows):
        tau = tau_points[k]
        return (rows[:, :no] * np.exp(tiled.e_o * tau),
                rows[:, no:nmo] * np.exp(-tiled.e_v * tau),
                np.zeros((rows.shape[0], naux)))

    def step(state, q, rows):
        xo, xv, Y = state
        G = xo @ rows[:, :no].T
        G *= xv @ rows[:, no:nmo].T
        Y += G @ rows[:, nmo:]

    def finish(k, p, rows, state):
        on_done(k, p, rows[:, nmo:], state[2])

    F.sweep(items, start, step, finish,
            lambda p: (F.bounds[p][1] - F.bounds[p][0]) * 8
            * (nmo + naux + F.rows.shape[1]))


def polarizability_tile_rows(tiled, tau_points, comm=None, ordered=False):
    """(k, this rank's auxiliary rows of proj(tau_k)) for every tau point in
    order, (r1 - r0, naux), the rows its `contiguous_block` of naux owns,
    from the grid tiles alone (`grid_factor_tiles`): no rank holds X or D
    whole, only its tiles, the p tiles of its current pass and one streamed
    q tile.

    The (tau point, p tile) items go over the ranks in `sweep_waves` windows
    (`wave_items`), a rank's items a contiguous run of the window's
    tau-major list. Each item is the same calls at every rank count
    (`_proj_items`), and proj(tau) = -2 sum_p D_p^T Y_p over the p tiles:

      ordered=False: each rank adds its own items' terms, p ascending, onto
        zeros, and each point's partials are reduce-scattered to rows (a rank
        without items of it hands in zeros): a sum re-associated over the
        ranks' items, as `polarizability_rows_sweep`'s.
      ordered=True: the terms are added in p order onto zeros at every rank
        count, the running sum of a point handed from one rank to the next
        along its items (a lockstep round per hand-over, every point at
        once); a rank that starts a point's sum runs it first and a rank
        that continues one runs it once the sum has arrived. The sum is then
        the one-rank sum, bitwise, and is sent to the row owners verbatim.
        Serially both are that sum.

    The row buffer is reused: read it before asking for the next point."""
    comm = current_comm() if comm is None else comm
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    if size == 1:
        comm = None
    F, naux = tiled.tiles, tiled.naux
    tau_points = np.asarray(tau_points, float)
    ntau, nt = len(tau_points), len(F.bounds)
    bounds = [contiguous_block(naux, r, size) for r in range(size)]
    r0, r1 = bounds[rank]
    rows = np.empty((r1 - r0, naux))
    for k0, k1 in sweep_waves(ntau, size):
        everyone = [wave_items(k0, k1, nt, r, size) for r in range(size)]
        if not ordered:
            partial = {}

            def add(k, p, Dp, Y):
                if k not in partial:
                    partial[k] = np.zeros((naux, naux))
                partial[k] += Dp.T @ Y

            _proj_items(tiled, tau_points, everyone[rank], add)
            zero = None
            for k in range(k0, k1):
                send = partial.pop(k, None)
                if send is None:
                    if zero is None:
                        zero = np.zeros((naux, naux))
                    send = zero
                else:
                    send *= -2.0
                reduce_scatter_rows(send, comm, out=rows)
                del send
                yield k, rows
            continue
        # each point's rows popped as they are handed out, so the window's
        # receive buffer is gone before the next window starts
        window = _ordered_window(tiled, tau_points, everyone, rank, comm,
                                 bounds)
        for k in sorted(window):
            rows[...] = window.pop(k)
            yield k, rows
        del window


def _ordered_window(tiled, tau_points, everyone, rank, comm, bounds):
    """{k: this rank's rows of proj(tau_k)} for one window of
    `polarizability_tile_rows(ordered=True)`: `everyone[r]` is rank r's run
    of the window's (k, p) items. Each point's chain is the ranks holding
    its items, in rank order; round 0 runs every rank's items of the points
    whose chain it starts, round h hands each running sum from member h - 1
    to member h, which then runs its items onto it. A finished sum is
    scaled by -2 and its rows go to their owners after the round."""
    naux = tiled.naux
    size = len(everyone)
    pieces = [{} for _ in range(size)]
    for r, items in enumerate(everyone):
        for k, p in items:
            pieces[r].setdefault(k, []).append(p)
    chains = {}
    for r in range(size):
        for k in pieces[r]:
            chains.setdefault(k, []).append(r)
    totals, out = {}, {}

    def add(k, p, Dp, Y):
        totals[k] += Dp.T @ Y

    for h in range(max(len(c) for c in chains.values())):
        if h:
            totals.update(_hand_over_round(chains, h, totals, rank, comm, naux))
        mine = [k for k, c in sorted(chains.items())
                if len(c) > h and c[h] == rank]
        for k in mine:
            if h == 0:
                totals[k] = np.zeros((naux, naux))
        _proj_items(tiled, tau_points,
                    [(k, p) for k in mine for p in pieces[rank][k]], add)
        done = {}
        for k in mine:
            if chains[k][-1] == rank:
                done[k] = totals.pop(k)
                done[k] *= -2.0
        finished = [k for k, c in sorted(chains.items()) if len(c) == h + 1]
        out.update(_rows_to_owners(done, finished, chains, rank, comm,
                                   bounds, naux))
    return out


def _hand_over_round(chains, h, totals, rank, comm, naux):
    """Every running sum moved from member h - 1 of its chain to member h,
    verbatim, in one exchange: {k: the sum this rank receives}. A rank is
    one member of a chain at a time, so it sends and receives at most one."""
    size = 1 if comm is None else comm.Get_size()
    moves = [(k, c[h - 1], c[h]) for k, c in sorted(chains.items())
             if len(c) > h]
    send_k = [k for k, a, _ in moves if a == rank]
    recv_k = [k for k, _, b in moves if b == rank]
    if len(send_k) > 1 or len(recv_k) > 1:
        raise RuntimeError(f'rank {rank} hands over {send_k} and receives '
                           f'{recv_k} in one round; a chain runs along '
                           'consecutive ranks')
    if comm is None:
        return {}
    send = totals.pop(send_k[0]) if send_k else np.empty((0, naux))
    to = [b for k, a, b in moves if a == rank]
    frm = [a for k, a, b in moves if b == rank]
    got = np.empty((naux, naux) if recv_k else (0, naux))
    exchange_rows(send, [(0, naux) if to and r == to[0] else (0, 0)
                         for r in range(size)],
                  got, [(0, naux) if frm and r == frm[0] else (0, 0)
                        for r in range(size)], comm)
    return {recv_k[0]: got} if recv_k else {}


def _rows_to_owners(done, finished, chains, rank, comm, bounds, naux):
    """{k: this rank's rows of each finished sum}: `done` holds the sums this
    rank finished, `finished` every point finished this round (its last
    chain member the sender), each sent to the row owners verbatim, one
    point a sender per round."""
    size = 1 if comm is None else comm.Get_size()
    r0, r1 = bounds[rank]
    if comm is None:
        return {k: a[r0:r1] for k, a in done.items()}
    by_sender = {}
    for k in finished:
        by_sender.setdefault(chains[k][-1], []).append(k)
    out = {}
    for j in range(max((len(v) for v in by_sender.values()), default=0)):
        sends = {s: ks[j] for s, ks in by_sender.items() if j < len(ks)}
        order = sorted(sends)
        k_own = sends.get(rank)
        send = done.pop(k_own) if k_own is not None else np.empty((0, naux))
        got = np.empty((len(order) * (r1 - r0), naux))
        recv = [(0, 0)] * size
        for i, s in enumerate(order):
            recv[s] = (i * (r1 - r0), (i + 1) * (r1 - r0))
        exchange_rows(send, list(bounds) if k_own is not None
                      else [(0, 0)] * size, got, recv, comm)
        for i, s in enumerate(order):
            out[sends[s]] = got[i * (r1 - r0):(i + 1) * (r1 - r0)]
    return out


def frequency_blocks(nfreq, naux, tile_gb, live):
    """(k0, k1) ranges over the frequency axis such that `live` (naux, naux)
    arrays per frequency fit in tile_gb. The one knob for both the M-row tiles
    and the frequency blocks: a working-set budget, not a count."""
    nb = fits(tile_gb, live * naux * naux * 8, cap=nfreq)
    return [(k0, min(k0 + nb, nfreq)) for k0 in range(0, nfreq, nb)]


def frequency_rounds(nfreq, size):
    """The frequency axis in rounds of `size`: round t is t size, ...,
    (t + 1) size - 1, its j-th frequency owned by rank j -- the round-robin
    owner, k % size."""
    return [list(range(k0, min(k0 + size, nfreq)))
            for k0 in range(0, nfreq, size)]


def round_ranges(n, nrows, size):
    """Per rank, the rows of a round's buffer of n frequencies, nrows rows
    each, that belong to the frequency rank j owns: the j-th, none past n."""
    return [(j * nrows, (j + 1) * nrows) if j < n else (0, 0)
            for j in range(size)]


def transform_rows(c, rows):
    """sum_tau c[..., tau] rows[tau], (c.shape[:-1], nrows, naux), one
    auxiliary row at a time.

    Each call is (c, (ntau, naux)) whatever the number of rows, so a row of
    the transform is the same bits whether it sits among all naux rows or a
    rank's few: the shape a BLAS picks its kernel from never sees the rank
    count. On a BLAS that is row-stable in the whole call's shape (MKL, even
    naux) it is also that call's rows."""
    out = np.empty(c.shape[:-1] + rows.shape[1:])
    for i in range(rows.shape[1]):
        out[..., i, :] = np.tensordot(c, rows[:, i, :], axes=(c.ndim - 1, 0))
    return out


def fold_frequency_rows(out, c, blk):
    """out += c^T blk over a serial block of frequencies, (ntau, nrows, naux)
    from c (nb, ntau) and blk (nb, nrows, naux), one auxiliary row at a time:
    the adjoint of `transform_rows` in its row-fixed call shape, the whole
    serial block on every rank (`ProjRows.fold`), so the temporary is
    (ntau, naux) per row rather than (ntau, naux, naux)."""
    for i in range(blk.shape[1]):
        out[:, i, :] += np.tensordot(c, blk[:, i, :], axes=(0, 0))


def spin_summed(kernel, X_mos, D, spectra, noccs, *args, **kwargs):
    """`kernel`'s chi0 -- or proj(tau) -- of an unrestricted reference,
    chi0_alpha + chi0_beta, from the kernel's restricted form.

    Every kernel here carries the closed-shell spin factor 2 of
    `polarizability_projected_tau`, so one spin's own contribution is half the
    kernel called on that spin's collocation, spectrum and occupation. Each
    spin sits at its own mid-gap, which cancels from the pair product
    e^{(eps_i - mu) tau} e^{-(eps_a - mu) tau} and only keeps the two factors
    bounded. `kernel` is `chi0_imaginary_frequency`, `chi0_frequency_rows`
    (the sum then in this rank's rows) or `polarizability_projected_sweep`;
    args and kwargs follow its `nocc`.
    """
    out = None
    for X, eps, nocc in zip(X_mos, spectra, noccs):
        part = kernel(X, D, eps, nocc, *args,
                      mu=particle_hole_window(eps, nocc)[2], **kwargs)
        if out is None:
            out = part
        else:
            _held(out)[...] += _held(part)
    _held(out)[...] *= 0.5
    return out


def _held(a):
    """What this rank holds of `a`: a `ProjRows`' rows, an array itself."""
    return a.rows if isinstance(a, ProjRows) else a


def three_index_slice(X, D, p, tile_gb=ISDF_TILE_GB):
    """B_p[P, q] = sum_k D[k,P] X[k,p] X[k,q]: one bra state's pair density in
    the auxiliary basis, (naux, norb), in O(M naux norb). Tiled over grid rows
    so the (M, norb) product X[:,p] * X never exists whole either."""
    M, norb = X.shape
    Bp = np.zeros((D.shape[1], norb))
    rows = tile_rows(M, tile_gb, norb * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Bp += D[p0:p1].T @ (X[p0:p1, p, None] * X[p0:p1])
    return Bp


def b_block(X, D, p_idx, q_idx, Y=None):
    """B[P, p, q] = sum_k X[k,p] Y[k,q] D[k,P] for one index block, Y = X by default.

    One GEMM per bra function rather than a single einsum: numpy's einsum
    falls off BLAS for a three-operand contraction with a batch index (~2
    GFlop/s, against 60-85 for the same work as matrix products), and looping
    the outer index keeps the working set at (M, n_q) instead of the
    (M, n_p n_q) an einsum path materializes.
    """
    Xp, Xq = X[:, p_idx], (X if Y is None else Y)[:, q_idx]
    out = np.empty((D.shape[1], Xp.shape[1], Xq.shape[1]))
    for a in range(Xp.shape[1]):
        out[:, a, :] = D.T @ (Xp[:, a, None] * Xq)
    return out


def three_index_ov(X, D, eps, nocc, tile_gb=ISDF_TILE_GB):
    """C_ov[P, (i,a)] = sum_k D[k,P] X[k,i] X[k,a], (naux, nocc*nvirt).

    The particle-hole block of the three-index tensor -- the O(N^4) object a
    contour deformation's residue term needs, because W at a real frequency has
    no imaginary-time form and is built from it explicitly. Frontier states
    sweep no residues and never need it. Tiled so the (rows, nocc*nvirt) pair
    block stays inside tile_gb.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    M = X.shape[0]
    n_ov = len(occ) * len(virt)
    C = np.zeros((D.shape[1], n_ov))
    rows = tile_rows(M, tile_gb, n_ov * 8)
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
        Xo, Xv = X[p0:p1][:, occ], X[p0:p1][:, virt]
        C += D[p0:p1].T @ (Xo[:, :, None] * Xv[:, None, :]).reshape(p1 - p0, n_ov)
    return C


def owned_frequency_blocks(proj_tau, cosft_wt, tile_gb, freq_indices=None,
                           live=3):
    """(ks, chi0 rows) over the frequency axis, block by block.

    ks are the frequencies of the block this caller owns -- all of them when
    `freq_indices` is None -- and the rows are sum_tau cosft_wt[ks, tau]
    proj_tau for those; a block with no owned frequency is skipped. The
    contour deformation's frequency axis has a naux^3 factorization per point
    and per-point results, which makes it the reduction-free axis to split.

    Every row is the serial row, on any BLAS. A GEMM row is not a function of
    that row alone: OpenBLAS picks its tail kernel and its threading from the
    call's shape, so rows a rank transforms on their own differ from the
    serial block's in the last bits: on water/cc-pVDZ at 3 of 24 frequencies a
    rank, OpenBLAS moves 30% of the contour-deformation wc by 1 to 224 ulp
    where MKL moves none, enough to break bitwise quasiparticle roots. So every
    serial block holding one of the caller's frequencies is transformed with
    the serial call and the owned rows are compacted to its front, in place;
    one rank is that call alone. An output partition over frequencies therefore
    gathers to the serial result bitwise, while a sum over frequencies still
    re-associates.

    Each row a rank transforms and does not own costs 2 ntau naux^2 flops
    against the (2/3) naux^3 factorization it pays per frequency it owns, a
    fraction 3 ntau / naux per row, and the peak stays the serial block.

    proj_tau may be `ProjRows`: then every block is yielded on every rank --
    each holds rows of every frequency, so each takes part in every gather --
    with its owned frequencies' chi0 gathered whole (`ProjRows.blocks`), ks
    empty where it owns none.
    """
    if isinstance(proj_tau, ProjRows):
        for fb in proj_tau.blocks(cosft_wt, tile_gb, freq_indices, live=live):
            yield fb.ks, fb.chi0
        return
    nfreq, naux = cosft_wt.shape[0], proj_tau.shape[-1]
    owned = (None if freq_indices is None
             else set(int(k) for k in np.atleast_1d(freq_indices)))
    for k0, k1 in frequency_blocks(nfreq, naux, tile_gb, live=live):
        ks = list(range(k0, k1))
        rows = (range(len(ks)) if owned is None
                else [m for m, k in enumerate(ks) if k in owned])
        if not rows:
            continue
        blk = np.tensordot(cosft_wt[ks], proj_tau, axes=(1, 0))
        if len(rows) < len(ks):
            # rows ascend, rows[j] >= j: no row is overwritten before it is read
            for j, m in enumerate(rows):
                if m != j:
                    blk[j] = blk[m]
            ks, blk = [ks[m] for m in rows], blk[:len(rows)]
        yield ks, blk


def laplace_representation_error(grid, eps, nocc, freq, pair_energies=None):
    """max over y = d +- freq of |y sum_k w_k e^{-y tau_k} - 1|: how well the
    grid's bare quadrature carries this real frequency, on the pair energies
    that occur. inf when freq reaches the gap (a real pole).

    pair_energies: d itself, in place of the pairs (eps, nocc) spans -- both
    spins of an unrestricted reference.

    The gate on the imaginary-time form of W at a real frequency, which is what
    a contour-deformation residue below the particle-hole gap asks for:
    -2d/(d^2 - w^2) = -int 2 cosh(w tau) e^{-d tau} dtau holds only where the
    grid still represents e^{-y tau} on every y = d -/+ w.
    """
    if pair_energies is None:
        occ, virt = get_occ_virt_indices(eps, nocc)
        d = (np.asarray(eps)[virt][None, :]
             - np.asarray(eps)[occ][:, None]).ravel()
    else:
        d = np.asarray(pair_energies, float)
    y = np.concatenate([d - freq, d + freq])
    if y.min() <= 0.0:
        return np.inf
    q = np.exp(-np.outer(y, grid.tau_points)) @ grid.tau_weights
    return float(np.abs(q * y - 1.0).max())


def chi0_aux(blocks, spin_factor=1.0, scratch=None):
    """chi0 = spin_factor sum_s (C_s f_s) C_s^T in the auxiliary basis: the
    independent-particle response of the particle-hole blocks
    `blocks` = [(C_ov, f)], C_ov (naux, n_ov) and f its pairs' frequency
    factor (`frequency_factor`). spin_factor 2 is the restricted singlet's
    spin sum; an unrestricted chi0 sums its two spins' blocks at 1. scratch:
    a buffer of one block's shape, filled in place of a fresh C_ov f."""
    out = None
    for C, f in blocks:
        # C f stays a temporary (no name), so no second (naux, n_ov) array
        # is held next to C_ov
        if scratch is None:
            term = (spin_factor * (C * f) if spin_factor != 1.0
                    else C * f) @ C.T
        else:
            np.multiply(C, f, out=scratch)
            term = (spin_factor * scratch if spin_factor != 1.0
                    else scratch) @ C.T
        out = term if out is None else out + term
    return out


def dyson_screening(chi0, transform=None, correlation=False):
    """W = [I - T^T chi0 T]^-1 at one frequency: the Dyson step of every
    imaginary-axis screening route.

    transform: the bare-gauge T of a reaction field (`bare_gauge_transform`),
    None for the gauge chi0 is in. correlation: return W - I.
    """
    if transform is not None:
        chi0 = transform.T @ chi0 @ transform
    W = np.linalg.inv(np.eye(chi0.shape[-1]) - chi0)
    if correlation:
        W[np.diag_indices_from(W)] -= 1.0
    return W


def minimax_points_for_bse(eps, nocc, tau_target=DEFAULT_TAU_TARGET):
    """(ntau, residual): smallest minimax time grid for the static-W build.

    The BSE counterpart of `minimax_points_for_gw`: only chi0(i.tau) is
    transformed here, so the chi0 transition range alone binds -- none of the
    self-energy ranges that widen the GW choice apply. Public so a harness that
    records the grid used can call the resolver `isdf_bse_factors` uses,
    instead of re-deriving a number that could silently drift from it.
    """
    gap, span, _ = particle_hole_window(eps, nocc)
    return minimax_points_for_accuracy(gap, span, target=tau_target)


def static_screening_grid(eps, nocc, ntau=None, tau_target=DEFAULT_TAU_TARGET):
    """The one-point (omega = 0) minimax grid a static screening is built on.

    One frequency, omega = 0: the BSE kernel and the reaction field's
    self-polarization both want the static screened interaction, so only the
    tau -> omega direction is evaluated. with_inverse=False skips the
    omega -> tau matrices, which a single input frequency cannot constrain --
    they would report an error of order one and warn on every call.

    ntau: None or 'auto' sizes the grid from the chi0 transition range through
    `minimax_points_for_bse`.
    """
    e_min, e_max, _ = particle_hole_window(eps, nocc)
    if ntau is None or (isinstance(ntau, str) and ntau.lower() == 'auto'):
        ntau, _ = minimax_points_for_bse(eps, nocc, tau_target=tau_target)
    return TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, [0.0], [1.0],
                                           with_sine=False, with_inverse=False)


def static_screening_matrix(X, D, eps, nocc, grid=None, mu=None, comm=None,
                            tile_memory_gb=ISDF_TILE_GB, tiles=None):
    """W = [1 - chi0(i.omega = 0)]^-1 in the auxiliary basis, from the
    separable factors in imaginary time.

    The one build of the static screened interaction: the BSE kernel's W_aux
    (`isdf_bse_factors`), the reaction field of Duchemin, Guido, Jacquemin and
    Blase, Chem. Sci. 9, 4430 (2018) Eq. (18)
    (`gradients.gwbse.reaction_field_adjoint.static_screening`), the space-time GW
    route's W(0) (its BSE extras and its Eq. (18) shift) and the dense ISDF
    BSE are this matrix on `static_screening_grid`, in the auxiliary gauge of
    the factors handed in.

    comm: an MPI communicator (or `simulated_world` rank), by default the
    current `distributed` region's, whose ranks all call this in lockstep, on
    X, D, eps and a grid that already agree bit for bit across them --
    `mpi_grid.replicate` at the entry point, which `isdf_bse_factors` does
    before it sizes the grid. The sweep runs on the fixed grid tiles
    (`grid_factor_tiles`, or `tiles` already built from these factors), so a
    rank holds its tiles of X and D and never either whole, and every
    proj(tau) is summed over the tiles in tile order whatever the rank count
    (`polarizability_tile_rows(ordered=True)`); chi0 is folded on auxiliary
    rows with the elementwise update in tau order, so it is the one-rank
    bits at every rank count. The rows are gathered and the inversion
    afterwards is replicated.

    X may be `SlicedFactors` over `comm`, D then None or the same factors, or
    a whole D on their grid in another gauge (`grid_factor_tiles`).
    tile_memory_gb: the working-set budget of the grid tiles and of a pass
    of the sweep, read where `tiles` are not handed in.
    """
    comm = current_comm() if comm is None else comm
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if tiles is None:
        tiles = grid_factor_tiles(X, D, eps, nocc, mu, comm,
                                  layout_numerics(tile_memory_gb))
    naux = tiles.naux
    r0, r1 = contiguous_block(naux, rank, nranks)
    chi0 = np.zeros((grid.nfreq, r1 - r0, naux))
    for k, rows in polarizability_tile_rows(tiles, grid.tau_points, comm,
                                            ordered=True):
        chi0 += grid.cosft_wt[:, k, None, None] * rows
    whole = np.empty((naux, naux))
    whole[r0:r1] = chi0[0]
    del chi0
    if nranks > 1:
        allgather_rows(whole, comm)
    return dyson_screening(whole)


def static_screening_matrix_unrestricted(X_mos, D, spectra, noccs,
                                         tau_target=DEFAULT_TAU_TARGET):
    """`static_screening_matrix` of an unrestricted reference: one W from
    chi0_alpha + chi0_beta (`spin_summed`), each spin in its own collocation
    X_mos[s] = X_ao C_s, on a grid spanning the transitions of both spins and
    as many time points as the harder of the two needs."""
    e_min, e_max = transition_range(spectra, noccs)
    ntau = max(minimax_points_for_bse(e, n, tau_target=tau_target)[0]
               for e, n in zip(spectra, noccs))
    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, [0.0], [1.0],
                                           with_sine=False, with_inverse=False)
    chi0 = spin_summed(chi0_imaginary_frequency, X_mos, D, spectra, noccs,
                       grid)[0]
    return dyson_screening(chi0)


class NonPositiveDielectric(RuntimeWarning):
    """det(1 - chi0) is not positive at an imaginary frequency."""


def dielectric_log_determinant(one_minus_c, where, flags=None):
    """log|det(1 - c)| of the dRPA energy, from the matrix `one_minus_c`
    (real, or complex on a k-point mesh) at `where`: the frequency index, or
    (transfer q, frequency index) on a mesh.

    On the imaginary axis chi0(i.w) is negative semidefinite in any real gauge,
    so 1 - chi0 >= 1 and the logarithm exists. A determinant that is not
    positive is a numerical outcome (a broken chi0, or a convention error),
    not an input to refuse: log|det| is returned with a
    `NonPositiveDielectric` warning and the named flag 'nonpositive_det'
    (`numerical_flags.flag`), and `where` is appended to
    flags['nonpositive_det'] when a dict is passed. Either way the energy
    built on it is not the dRPA correlation energy.
    """
    sign, logdet = np.linalg.slogdet(one_minus_c)
    if not sign.real > 0:
        flag('nonpositive_det',
             f'det(1 - chi0) is not positive at {where} (sign {sign}): '
             f'the RPA dielectric has been driven singular, which on a '
             f'gapped system means a convention error rather than '
             f'physics; log|det| is used.',
             NonPositiveDielectric, stacklevel=3)
        if flags is not None:
            flags.setdefault('nonpositive_det', []).append(where)
    return logdet


def rpa_correlation_energy_space_time(X, D, eps, nocc, grid, mu=None,
                                      tile_gb=ISDF_TILE_GB, screening=None,
                                      counter_term=None, comm=None,
                                      want_tape=False, flags=None):
    """dRPA correlation energy E_c = (1/2pi) int dw Tr{log(1 - chi0) + chi0}.

    The quadrature of `rpa_energy.rpa_correlation_energy_imaginary_axis`, with
    chi0 from the imaginary-time route. The frequencies come out of proj(tau)
    one block at a time (`owned_frequency_blocks`), so the axis is never whole
    and the peak is proj(tau) plus one block.

    screening = (N, g): the interaction that builds the logarithm is rescaled
    to v + g(w) vtilde while D stays in the dressed gauge. In that gauge the
    rescaling is a similarity on chi0 alone -- with N = V_d^(-1/2) vtilde
    V_d^(-1/2), V_d = V + vtilde, and S_w = I - (1 - g_w) N,

        log det(I - S_w c_w) = log det(I - (v + g_w vtilde) P_1(iw)) ,

    so g == 1 is the dressed interaction (the default) and g == 0 the bare one.
    g is one scalar per frequency, N one naux x naux matrix.
    counter_term = C: the linear term becomes Tr(C c_w) in place of Tr(c_w).
    C = I - N puts the bare interaction there, which is what an exact block
    fold of the log-determinant over a non-overlapping solvent keeps, and what
    leaves the leading solute-solvent dispersion term in the energy instead of
    cancelling it (`solvated_rpa_energy.fold_terms` chooses the pair).

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep. The tau sweep is split over tau and reduced once,
    ntau x naux^2 over disjoint slots; the frequency loop is split over
    frequencies, whose partial E_c is one scalar reduction. Each frequency's
    chi0 is the serial one bitwise (`owned_frequency_blocks`), so only that
    sum re-associates. None is `current_comm()`.
    The inputs are identical by construction and are not broadcast; an audited
    run compares their digests and those of E_c and proj(tau)
    (`mpi_grid.agreement`).

    want_tape: also return the `RPAEnergyTape` a reverse pass reads.
    flags: a dict, or None: receives under 'nonpositive_det' the frequencies,
    on every rank, whose det(1 - chi0) was not positive
    (`dielectric_log_determinant`).
    """
    n_mat, g = (None, None) if screening is None else screening
    comm = current_comm() if comm is None else comm
    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    if nranks > 1:
        agreement((X, D, eps, screening, counter_term), comm, audit_only=True,
                  label='rpa_correlation_energy_space_time inputs')
    tau_mine = partition(grid.ntau, rank, nranks) if nranks > 1 else None
    nu_mine = partition(grid.nfreq, rank, nranks) if nranks > 1 else None
    proj_tau = polarizability_projected_sweep(X, D, eps, nocc, grid.tau_points,
                                              mu=mu, tau_indices=tau_mine,
                                              tile_memory_gb=tile_gb)
    if nranks > 1:
        reduce_sum(proj_tau, comm)            # the others' slots are zero
    terms, mine = [], {}
    for ks, blk in owned_frequency_blocks(proj_tau, grid.cosft_wt, tile_gb,
                                          nu_mine):
        for m, k in enumerate(ks):
            c0 = blk[m]
            linear = (np.trace(c0) if counter_term is None
                      else float(np.einsum('pq,qp->', counter_term, c0)))
            # S_w c_w, the argument of the logarithm. `one_minus_g` is 0 for the
            # dressed interaction, so S = I and c is untouched there.
            one_minus_g = 0.0 if n_mat is None else 1.0 - g[k]
            c = c0 if n_mat is None else c0 - one_minus_g * (n_mat @ c0)
            logdet = dielectric_log_determinant(np.eye(c.shape[-1]) - c, k, mine)
            terms.append((k, np.array(grid.omega_weights[k]
                                      * (logdet + linear))))
    if flags is not None:
        bad = mine.get('nonpositive_det', [])
        bad = sorted(int(k) for part in (comm.allgather(bad) if nranks > 1 else [bad])
                     for k in part)
        if bad:
            flags.setdefault('nonpositive_det', []).extend(bad)
    # every frequency's term added in frequency order: the one-rank bits
    e_c = float(ordered_sum(terms, comm, onto=np.zeros(())))
    e_c /= 2.0 * np.pi
    if nranks > 1:
        agreement((e_c, proj_tau), comm, audit_only=True,
                  label='rpa_correlation_energy_space_time outputs')
    if not want_tape:
        return e_c
    return e_c, RPAEnergyTape(proj_tau, tau_mine, nu_mine)
