"""The separable factors held as grid-point slices, one row block per rank.

Replicated, every rank holds X_mo (M, nmo), X_ao (M, nao) and D (M, naux)
whole at every stage. `SlicedFactors` keeps this rank's `contiguous_block` of
the grid rows of all three and the block of Zt the BSE action already owns.

The M^2 sweeps of the quasiparticle solve -- chi0 (GW window, static W) and
the self-energy -- contract the grid index on both sides of every block, so
they read the factors through fixed grid tiles (`GridTileRows`): the rows cut
into tiles set by the grid and the budget alone (`grid_factor_tiles`), each
held by one rank, and a sweep's (p, q) blocks run with the p tiles of the
current pass fetched and every q tile streamed past from its owner in tile
order (`GridTileRows.sweep`). No rank holds any factor whole, and what a rank
holds falls with the rank count. Each tile is the same rows at every rank
count and each block the same calls on them, so a sum the sweep adds in tile
order is the one-rank sum, bitwise (W(0), the self-energy); a sum it joins
over the ranks' partials re-associates (the GW chi0's proj(tau)). The block
action reads X_v and D by rows once Zt = (D_rows W) D^T is built, D's rows
reaching that product one owner's block at a time, and X_o whole in
(Zt * P) X_o, gathered once (`gather`, an output partition: every row travels
verbatim, so the gathered array is the replicated one bit for bit). The
static exchange reads none of the three (the mean field's own K). The
replicated fit forms X, X_mo, D and M whole on every rank before the cut;
`separable_factors(fit='rows')` builds the rows directly
(`separable_ri.fit_rows`, held through `from_rows`), so no rank holds any of
them whole during the fit. The gradient chains hold the same rows
(`FrozenFactorization(sliced=True)`); their adjoint kernels gather what they
read whole once per sweep through `whole_factor`.

The slices are cut from the whole products because a row block of a GEMM is
in general not the same bits as those rows of the whole GEMM ((D C)[r0:r1]
against D[r0:r1] C). `separable_factors(sliced=True)` forms and locksteps X_mo
and D as the replicated fit does and only then keeps this rank's rows, so
every row is the replicated array's: on one mean field, and on a pyscf that
repeats its bits, every distributed output is bitwise the replicated run's at
the same rank count. pyscf's OpenMP GEMM adds its partial sums in
thread-arrival order, so a gradient chain's own K builds and mean-field force
differ between two runs, which are compared on an anchored bar. The
row-distributed fit computes every array one fixed tile at a time, whoever
owns the tile, so a rank's rows are bitwise the same rows of a one-rank run
at any rank count: a different realization from the replicated fit, not its
bits.
"""
import contextlib
import threading

import numpy as np

from src.Base.numerics.constants import BYTES_PER_GB, ISDF_TILE_GB
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, allgather_rows,
                                              broadcast_rows, contiguous_block,
                                              current_comm, exchange_blocks,
                                              exchange_rows, partition)


class _Ledger(threading.local):
    """The open `tile_ledger` of this thread (one simulated rank), or None."""
    current = None


_LEDGER = _Ledger()


@contextlib.contextmanager
def tile_ledger():
    """{'tiles': bytes, 'pass': bytes}: the most this rank held at once in
    grid tiles (`GridTileRows` built while the ledger is open, the largest
    one) and in one pass of a tile sweep (its fetched p tiles, the items'
    arrays and the streamed tile), read off the arrays. Per thread, so each
    simulated rank keeps its own."""
    outer = _LEDGER.current
    _LEDGER.current = {'tiles': 0, 'pass': 0}
    try:
        yield _LEDGER.current
    finally:
        _LEDGER.current = outer


def _record(kind, nbytes):
    """Note `nbytes` held at once under `kind` in this thread's ledger."""
    led = _LEDGER.current
    if led is not None:
        led[kind] = max(led[kind], int(nbytes))


def _array_bytes(state):
    """Bytes of the arrays an item's state holds."""
    if isinstance(state, np.ndarray):
        return state.nbytes
    if isinstance(state, (list, tuple)):
        return sum(_array_bytes(x) for x in state)
    return 0


class SlicedFactors:
    """This rank's grid rows of (X_mo, D, X_ao), the grid points whole.

    X_mo, D, X_ao: this rank's `contiguous_block` of the grid rows of each,
    C-contiguous; coords: every grid point, (M, 3); comm: the communicator
    the rows were cut for, the only one they can be gathered over.
    `gathers` counts the whole-array gathers by name.

    Not a tuple: unpacking it raises, because a consumer that reads the
    factors as whole arrays would otherwise compute on a slice as if it were
    the grid.
    """

    def __init__(self, X_mo, D, X_ao, coords, comm):
        if comm is None or comm.Get_size() < 2:
            raise ValueError('sliced factors need a communicator of more than '
                             'one rank; serially the factors are whole')
        self.comm = comm
        self.coords = coords
        self.npts = int(len(coords))
        self.rows = contiguous_block(self.npts, comm.Get_rank(), comm.Get_size())
        nrows = self.rows[1] - self.rows[0]
        for name, a in (('X_mo', X_mo), ('D', D), ('X_ao', X_ao)):
            if a.ndim != 2 or a.shape[0] != nrows:
                raise ValueError(f'{name} holds {a.shape[0]} rows where rank '
                                 f'{comm.Get_rank()} owns {nrows} of {self.npts}')
        self.X_mo = np.ascontiguousarray(X_mo)
        self.D = np.ascontiguousarray(D)
        self.X_ao = np.ascontiguousarray(X_ao)
        self.gathers = {}

    @classmethod
    def from_whole(cls, factors, comm):
        """Copies of this rank's rows of whole (X_mo, D, X_ao, coords)."""
        X_mo, D, X_ao, coords = factors
        r0, r1 = contiguous_block(len(coords), comm.Get_rank(), comm.Get_size())
        return cls(X_mo[r0:r1].copy(), D[r0:r1].copy(), X_ao[r0:r1].copy(),
                   coords, comm)

    @classmethod
    def from_rows(cls, X_mo, D, X_ao, coords, comm, fit_held=None):
        """Factors a fit produced as this rank's rows
        (`separable_ri.fit_rows`), with no whole array on any rank to cut
        them from.

        fit_held: {array: bytes}, the most of each array of the fit this rank
        held at once, kept as `fit_held` for memory accounting; not an array,
        so `held_bytes` still reads the factors alone.
        """
        out = cls(X_mo, D, X_ao, coords, comm)
        out.fit_held = dict(fit_held or {})
        return out

    def __iter__(self):
        raise TypeError(
            f'SlicedFactors holds rows {self.rows} of {self.npts} grid points '
            'on this rank; a consumer that needs a factor whole calls '
            '`gather` once, and one that unpacks them as a tuple would read '
            'a slice as the grid')

    @property
    def nmo(self):
        return self.X_mo.shape[1]

    @property
    def nao(self):
        return self.X_ao.shape[1]

    @property
    def naux(self):
        return self.D.shape[1]

    def coulomb_rows(self, D=None):
        """This rank's rows of the Coulomb factor a sweep reads beside these
        collocations: the factors' own D (D None or these factors), or this
        rank's rows of a whole D on the same grid -- another gauge of it, as
        a reaction field's bare factor is. Anything else is refused."""
        if D is None or D is self:
            return self.D
        if isinstance(D, SlicedFactors):
            raise ValueError('a Coulomb factor beside sliced factors is their '
                             'own or a whole array on their grid, not other '
                             'sliced factors')
        D = np.asarray(D)
        if D.ndim != 2 or D.shape[0] != self.npts:
            raise ValueError(f'a whole D beside sliced factors has '
                             f'{self.npts} grid rows; got shape {D.shape}')
        return D[self.rows[0]:self.rows[1]]

    def require(self, comm):
        """Refuse a consumer running over any other rank layout than the one
        the rows were cut for: serially, or over a different rank count."""
        if (comm is None or comm.Get_size() != self.comm.Get_size()
                or comm.Get_rank() != self.comm.Get_rank()):
            have = 'no communicator' if comm is None else (
                f'rank {comm.Get_rank()} of {comm.Get_size()}')
            raise ValueError(
                f'sliced factors were cut for rank {self.comm.Get_rank()} of '
                f'{self.comm.Get_size()} and reached a consumer with {have}; '
                'a serial reference needs the whole factors')
        return self

    def gather(self, name):
        """The whole 'X_mo', 'D' or 'X_ao', (M, ncol): one collective."""
        return self._gathered(name, getattr(self, name))

    def gather_columns(self, name, columns, label):
        """`columns` of the whole `name`, C-contiguous, counted as `label`:
        one collective, and the other columns never cross."""
        return self._gathered(label, getattr(self, name)[:, columns])

    def branches(self, occ, virt):
        """(X_o, X_v): X_mo's occupied and virtual columns whole --
        `split_branches`'s pair, never the whole X_mo."""
        return (self.gather_columns('X_mo', occ, 'X_o'),
                self.gather_columns('X_mo', virt, 'X_v'))

    def held_bytes(self):
        """{attribute: bytes} of every array this object holds on this rank,
        read off the object rather than off the formula it is meant to meet."""
        return {name: int(a.nbytes) for name, a in vars(self).items()
                if isinstance(a, np.ndarray)}

    def _gathered(self, name, rows):
        """Every rank's `rows` stacked into the whole array, counted."""
        r0, r1 = self.rows
        whole = np.empty((self.npts,) + rows.shape[1:])
        whole[r0:r1] = rows
        allgather_rows(whole, self.comm)
        self.gathers[name] = self.gathers.get(name, 0) + 1
        return whole


def whole_factor(a, name):
    """`a` as an array for one sweep: a `SlicedFactors` gathered whole once
    as its `name` ('X_mo', 'D' or 'X_ao'), an array handed back as it is.
    Slices reaching a sweep on other ranks than they were cut for (a serial
    reference among them) are refused by `SlicedFactors.require`."""
    if not isinstance(a, SlicedFactors):
        return a
    return a.require(current_comm()).gather(name)


class GridTileRows:
    """This rank's fixed tiles of an (npts, ncol) grid array, an adjoint's
    output partition: tile t is rows [t*block, (t+1)*block), owned by rank
    t % size (the row fit's tiles and owners, `separable_ri.fit_rows`),
    and this rank's tiles are stacked in tile order in one C-contiguous
    `rows` array. Serially `rows` is the whole array.

    A kernel whose writes land on a row's own tile computes that tile the
    same way at every rank count, so the rows are the same bits whoever owns
    them. Every operation here moves rows verbatim or adds elementwise: `+=`
    (another of the same layout, or a whole array's rows), `from_whole`,
    `from_blocks` (`SlicedFactors`' contiguous rows), `gather` and
    `broadcast_tile`. A whole-array read raises, as `ProjRows` does.
    """

    def __init__(self, rows, npts, block, comm=None):
        size = 1 if comm is None else comm.Get_size()
        self.rank = 0 if comm is None else comm.Get_rank()
        self.size = size
        self.comm = comm if size > 1 else None
        self.npts, self.block = int(npts), int(block)
        ntiles = -(-self.npts // self.block)
        self.bounds = [(t * self.block, min((t + 1) * self.block, self.npts))
                       for t in range(ntiles)]
        self.mine = self.owned(self.rank)
        self.offsets, at = {}, 0
        for t in self.mine:
            self.offsets[t] = at
            at += self.bounds[t][1] - self.bounds[t][0]
        if rows.ndim != 2 or rows.shape[0] != at:
            raise ValueError(f'rows {rows.shape} are not the {at} rows of rank '
                             f'{self.rank} of {size} in tiles of {self.block}')
        self.rows = np.ascontiguousarray(rows)
        self.pass_gb = ISDF_TILE_GB
        _record('tiles', self.rows.nbytes)

    def owned(self, rank):
        """The tiles `rank` owns, in order."""
        return [int(t) for t in partition(len(self.bounds), rank, self.size)]

    @classmethod
    def zeros(cls, npts, ncol, block, comm=None):
        """Zeroed rows for this rank's tiles."""
        size = 1 if comm is None else comm.Get_size()
        rank = 0 if comm is None else comm.Get_rank()
        npts, block = int(npts), int(block)
        ntiles = -(-npts // block)
        n = sum(min((t + 1) * block, npts) - t * block
                for t in partition(ntiles, rank, size))
        return cls(np.zeros((n, ncol)), npts, block, comm)

    @classmethod
    def from_whole(cls, a, block, comm=None):
        """Copies of this rank's tiles of the whole array `a`."""
        out = cls.zeros(a.shape[0], a.shape[1], block, comm)
        for t in out.mine:
            out.tile(t)[...] = a[slice(*out.bounds[t])]
        return out

    @classmethod
    def from_blocks(cls, rows, npts, block, comm=None):
        """This rank's tiles from every rank's `contiguous_block` rows of the
        array, the layout `SlicedFactors` and the BSE grid adjoint hand out:
        one exchange, every row verbatim."""
        out = cls.zeros(npts, rows.shape[1], block, comm)
        if out.comm is None:
            out.rows[...] = rows
            return out
        blocks = [contiguous_block(npts, r, out.size) for r in range(out.size)]
        r0, r1 = blocks[out.rank]
        send, shapes = [], []
        for peer in range(out.size):
            pieces = [rows[lo - r0:hi - r0]
                      for lo, hi in _overlap(out, out.owned(peer), (r0, r1))]
            send.append(np.concatenate(pieces) if pieces
                        else np.empty((0, rows.shape[1])))
            n = sum(hi - lo for lo, hi in _overlap(out, out.mine, blocks[peer]))
            shapes.append((n, rows.shape[1]))
        got = exchange_blocks(send, shapes, out.comm)
        for peer in range(out.size):
            at = 0
            for t in out.mine:
                lo = max(out.bounds[t][0], blocks[peer][0])
                hi = min(out.bounds[t][1], blocks[peer][1])
                if hi > lo:
                    t0 = out.bounds[t][0]
                    out.tile(t)[lo - t0:hi - t0] = got[peer][at:at + hi - lo]
                    at += hi - lo
        return out

    @property
    def shape(self):
        """The whole array's shape, (npts, ncol)."""
        return (self.npts, self.rows.shape[1])

    @property
    def nbytes(self):
        """What this rank holds."""
        return self.rows.nbytes

    def __array__(self, *args, **kwargs):
        raise TypeError('the adjoint is held in grid tiles over the ranks; a '
                        'whole-array read has to `gather` it')

    def same_layout(self, other):
        """Whether `other` holds the same tiles of an array of the same shape."""
        return (isinstance(other, GridTileRows) and other.shape == self.shape
                and other.block == self.block and other.size == self.size
                and other.rank == self.rank)

    def tile(self, t):
        """Tile t's rows, a view; t must be this rank's."""
        at = self.offsets[t]
        return self.rows[at:at + self.bounds[t][1] - self.bounds[t][0]]

    def tiles(self):
        """{t: rows} for this rank's tiles, views."""
        return {t: self.tile(t) for t in self.mine}

    def __iadd__(self, other):
        if isinstance(other, GridTileRows):
            if not self.same_layout(other):
                raise ValueError('adding grid tiles of another layout')
            self.rows += other.rows
            return self
        other = np.asarray(other)
        if other.shape != self.shape:
            raise ValueError(f'adding a {other.shape} array to grid tiles of '
                             f'{self.shape}')
        for t in self.mine:
            self.tile(t)[...] += other[slice(*self.bounds[t])]
        return self

    def gather(self):
        """The whole array on every rank, every row moved verbatim."""
        whole = np.empty(self.shape)
        for t in self.mine:
            whole[slice(*self.bounds[t])] = self.tile(t)
        if self.comm is not None:
            allgather_ranges(whole, [[self.bounds[t] for t in self.owned(r)]
                                     for r in range(self.size)], self.comm)
        return whole

    def broadcast_tile(self, t):
        """Tile t's rows on every rank, sent by its owner, verbatim."""
        t0, t1 = self.bounds[t]
        buf = (np.array(self.tile(t)) if t in self.offsets
               else np.empty((t1 - t0, self.rows.shape[1])))
        if self.comm is not None:
            broadcast_rows(buf, t % self.size, self.comm)
        return buf

    @classmethod
    def from_rows(cls, rows, npts, block, comm=None, order=None):
        """This rank's tiles from every rank's `contiguous_block` rows of an
        (npts, ncol) array, the tiles cut from the grid taken in `order`: a
        permutation, position i of the tiled array holding grid row
        order[i]; None is the grid's own order. One windowed exchange
        (`exchange_rows`), every row verbatim."""
        out = cls.zeros(npts, rows.shape[1], block, comm)
        if order is not None:
            order = np.asarray(order, dtype=np.intp)
        if out.comm is None:
            out.rows[...] = rows if order is None else rows[order]
            return out
        size = out.size
        starts = [contiguous_block(out.npts, r, size)[0] for r in range(size)]
        r0, r1 = contiguous_block(out.npts, out.rank, size)
        mine = np.arange(r0, r1)
        if order is None:
            pos = mine
        else:
            where = np.empty(out.npts, dtype=np.intp)
            where[order] = np.arange(out.npts)
            pos = where[mine]
        owner = (pos // out.block) % size
        sort = np.lexsort((pos, owner))
        send = np.ascontiguousarray(rows[sort])
        counts = np.bincount(owner, minlength=size)
        edges = np.concatenate([[0], np.cumsum(counts)])
        held = np.concatenate([np.arange(*out.bounds[t]) for t in out.mine]
                              + [np.empty(0, dtype=np.intp)])
        source = held if order is None else order[held]
        peer = np.searchsorted(starts, source, side='right') - 1
        land = np.concatenate([np.flatnonzero(peer == r) for r in range(size)])
        got = np.empty_like(out.rows)
        lands = np.concatenate([[0], np.cumsum(np.bincount(peer,
                                                            minlength=size))])
        exchange_rows(send, [(int(edges[r]), int(edges[r + 1]))
                             for r in range(size)],
                      got, [(int(lands[r]), int(lands[r + 1]))
                            for r in range(size)], out.comm)
        out.rows[land] = got
        return out

    @classmethod
    def from_columns(cls, parts, block, comm=None, order=None):
        """This rank's tiles of [parts[0] | parts[1] | ...], whole arrays of
        one row count, the rows taken in `order` as for `from_rows`: each
        tile stacked from the parts' own rows, so no whole stack is formed."""
        npts = parts[0].shape[0]
        out = cls.zeros(npts, sum(p.shape[1] for p in parts), block, comm)
        for t in out.mine:
            t0, t1 = out.bounds[t]
            rows = slice(t0, t1) if order is None else np.asarray(order)[t0:t1]
            out.tile(t)[...] = np.hstack([p[rows] for p in parts])
        return out

    def map_tiles(self, fn, ncol):
        """A new layout of `ncol` columns whose tile t is fn(tile t's rows),
        computed on its owner one fixed tile at a time, so each tile is the
        same bits whoever owns it."""
        tiles = [np.asarray(fn(self.tile(t))) for t in self.mine]
        rows = np.concatenate(tiles) if tiles else np.empty((0, int(ncol)))
        out = GridTileRows(rows, self.npts, self.block, self.comm)
        out.pass_gb = self.pass_gb
        return out

    def fetch(self, wanted):
        """{t: tile t's rows} for every tile of `wanted`, this rank's list:
        each a fresh C-contiguous copy, its owner's rows verbatim, the owner's
        own included, so every rank computes on the same arrays. A collective:
        the lists travel first (one small allgather), then one
        `exchange_rows` per round hands each rank its next tile."""
        wanted = [int(t) for t in wanted]
        ncol = self.rows.shape[1]
        if self.comm is None:
            return {t: np.array(self.tile(t)) for t in wanted}
        lists = self.comm.allgather(wanted)
        out = {}
        for j in range(max(len(own) for own in lists)):
            asks = [own[j] if j < len(own) else None for own in lists]
            send = []
            for t in asks:
                if t is not None and t in self.offsets:
                    at = self.offsets[t]
                    send.append((at, at + self.bounds[t][1] - self.bounds[t][0]))
                else:
                    send.append((0, 0))
            t = asks[self.rank]
            if t is None:
                buf, recv = np.empty((0, ncol)), [(0, 0)] * self.size
            else:
                n = self.bounds[t][1] - self.bounds[t][0]
                buf = np.empty((n, ncol))
                recv = [(0, n) if r == t % self.size else (0, 0)
                        for r in range(self.size)]
            exchange_rows(self.rows, send, buf, recv, self.comm)
            if t is not None:
                out[t] = buf
        return out

    def stream(self):
        """(t, tile t's rows) for every tile in order, on every rank: each
        sent by its owner into one buffer that is reused, so read a tile
        before asking for the next. Serially the tile is copied into the
        same buffer, so a kernel reads the same arrays at every rank count.
        A collective: every rank runs the whole stream."""
        buf = np.empty((self.block, self.rows.shape[1]))
        for t, (t0, t1) in enumerate(self.bounds):
            view = buf[:t1 - t0]
            if t in self.offsets:
                view[...] = self.tile(t)
            if self.comm is not None:
                broadcast_rows(view, t % self.size, self.comm)
            yield t, view

    def sweep(self, items, start, step, finish, item_bytes):
        """Run this rank's `items`, [(key, p)] with p a tile, against every
        tile q of the grid, q ascending: the (p, q) block sweep of an M^2
        kernel whose p rows a rank holds while the q rows stream past.

        The items go in passes, each as many of them, in order, as
        `item_bytes(p)` fits into `pass_gb` (the layout's budget, set by the
        builder of the tiles); a pass fetches its p tiles
        (`fetch`) and streams the grid once (`stream`). For each item,
        state = start(key, p, rows of p), then step(state, q, rows of q) for
        every q in order, then finish(key, p, rows of p, state), items in
        their order. Every tile is the same rows at every rank count and each
        item runs the same calls on them, so an item's result does not depend
        on the rank count or on the pass it fell into. A collective: every
        rank runs the most passes any rank needs, with an empty pass where it
        has none left."""
        passes, cur, used = [], [], 0.0
        budget = self.pass_gb * BYTES_PER_GB
        for key, p in items:
            need = float(item_bytes(p))
            if cur and used + need > budget:
                passes.append(cur)
                cur, used = [], 0.0
            cur.append((key, p))
            used += need
        if cur:
            passes.append(cur)
        npass = (len(passes) if self.comm is None
                 else max(self.comm.allgather(len(passes))))
        for j in range(npass):
            work = passes[j] if j < len(passes) else []
            rows = self.fetch(sorted({p for _, p in work}))
            states = [start(key, p, rows[p]) for key, p in work]
            _record('pass', sum(a.nbytes for a in rows.values())
                    + _array_bytes(states)
                    + self.block * self.rows.shape[1] * 8)
            for q, rows_q in self.stream():
                for state in states:
                    step(state, q, rows_q)
            for (key, p), state in zip(work, states):
                finish(key, p, rows[p], state)
            del rows, states


def column_tiles(parts, npts, block, comm=None, order=None, whole=None):
    """`GridTileRows` of [parts[0] | parts[1] | ...] in tiles of `block`
    rows, the grid taken in `order` (`GridTileRows.from_rows`).

    The parts are this rank's `contiguous_block` rows of each array
    (`SlicedFactors`' layout) under a communicator of more than one rank,
    and whole arrays serially or where `whole` says so (factors replicated
    on every rank): either way a rank keeps its tiles alone, and the tiles
    are the same rows."""
    size = 1 if comm is None else comm.Get_size()
    if whole is None:
        whole = size == 1
    if whole:
        return GridTileRows.from_columns(parts, block, comm, order)
    return GridTileRows.from_rows(np.hstack(parts), npts, block, comm, order)

def _overlap(layout, tiles, span):
    """[lo, hi) of each of `tiles` inside the row range `span`, in order,
    the empty ones left out."""
    out = []
    for t in tiles:
        lo, hi = max(layout.bounds[t][0], span[0]), min(layout.bounds[t][1],
                                                        span[1])
        if hi > lo:
            out.append((lo, hi))
    return out


def unpack_factors(factors):
    """(X_mo, D, X_ao, coords) of a separable factorization's tuple; the last
    two are None for a shorter tuple."""
    return tuple(factors) + (None,) * (4 - len(factors))
