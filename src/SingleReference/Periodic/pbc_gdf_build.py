"""
Sources of the Gaussian density-fitting factors B^{k,k+q} (naux, nao, nao)
of the k-point routes, one interface for both: called on every rank with the
transfers it owns, a source returns {q: (nk, naux_q, nao, nao)} for them.

FileGDF reads them from the cderi file of a pyscf GDF object, which one
process builds (`gdf_pair_factors`).

DistributedGDF builds them over the ranks and never writes them whole. The
3-centre integrals (mu nu | P) and their fit B = L^-1 (mu nu | P) act on the
auxiliary index alone, so pyscf's range-separated builder run on AO rows
[i0, i1) gives exactly those rows of B: each rank builds the rows of one
slice, for the transfers of one group (when a node has more ranks than
slices), into a file of its own on node-local storage, and the rows travel
to the owners of the transfers over MPI. The ranks of one node build exactly
the transfers they own (`build_plan`), so the rows never cross the
interconnect; only the metric and the auxiliary sizes do. The fit's metric (P|Q)^q is
computed once, on rank 0, so every slice of a transfer is fitted in the same
auxiliary gauge. Two corrections to pyscf's builder for a slice that does not
start at AO 0: the q = 0 neutralisation subtracts the overlap of these rows,
not of rows 0..i1-i0, and the shell slice is given in the six-index form
pyscf's first pass expects.

The two sources agree to the precision of the lattice sums (cell.precision):
the subset build decomposes the metric of each transfer on its own, where a
full build takes the conjugate of its partner's, and stores s1, not s2.
"""
import os
import shutil
import tempfile

import h5py
import numpy as np

from pyscf.pbc.df.df import make_modrho_basis
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder

from src.Base.utils.parallel.mpi_grid import broadcast, current_comm, exchange_blocks
from src.SingleReference.Periodic.pbc_integrals import gdf_pair_factors


class FileGDF:
    """The factors from the cderi file of the GDF object `mydf`, built there
    (one process) if it has none."""

    def __init__(self, mydf):
        self.mydf = mydf

    def __call__(self, kpts, kplus, owned):
        mydf = self.mydf
        if getattr(mydf, '_cderi', None) is None or getattr(mydf, '_j_only', False):
            mydf._cderi = None              # build(j_only=False) is a no-op otherwise
            mydf.build(j_only=False)
        nk = len(kpts)
        out = {}
        pairs = [(k, kplus[q, k]) for q in owned for k in range(nk)]
        for n, (k, _, b) in enumerate(gdf_pair_factors(mydf, kpts, pairs)):
            q = owned[n // nk]
            if q not in out:
                out[q] = np.empty((nk,) + b.shape, dtype=complex)
            out[q][k] = b
        return out


class DistributedGDF:
    """The factors of the GDF object `mydf` (its auxiliary basis, mesh and
    linear-dependence threshold) built over the ranks by AO-row slices;
    `workdir` holds each rank's slice file while it is built (default
    $TMPDIR, node-local on the clusters); `ranks_per_node` consecutive ranks
    share a node (default SLURM_NTASKS_PER_NODE, else all ranks one node)."""

    def __init__(self, mydf, workdir=None, ranks_per_node=None):
        self.mydf = mydf
        self.workdir = workdir if workdir is not None else os.environ.get('TMPDIR')
        self.ranks_per_node = ranks_per_node

    def __call__(self, kpts, kplus, owned):
        comm = current_comm()
        rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
        cell, mydf, nk = self.mydf.cell, self.mydf, len(kpts)
        wanted = [list(owned)] if comm is None else comm.allgather([int(q) for q in owned])
        needed = sorted(set(q for w in wanted for q in w))
        per_node = (self.ranks_per_node or int(os.environ.get('SLURM_NTASKS_PER_NODE', 0))
                    or size)
        tasks, builders = build_plan(cell, wanted, per_node)
        node = rank // per_node

        auxcell = make_modrho_basis(cell, mydf.auxbasis, mydf.exp_to_discard)
        metric = broadcast(_metric_table(cell, auxcell, kpts, needed, mydf)
                           if rank == 0 else None, comm)

        mine = {}
        if tasks[rank] is not None:
            rows, transfers = tasks[rank]
            mine = _build_rows(cell, auxcell, kpts, kplus, transfers, rows, metric,
                               mydf, self.workdir, subset=len(transfers) < nk)
        naux = {}
        for found in ([{q: b.shape[1] for q, b in mine.items()}] if comm is None
                      else comm.allgather({q: b.shape[1] for q, b in mine.items()})):
            naux.update(found)

        nao = cell.nao_nr()
        out = {}
        for j in range(max(len(w) for w in wanted)):
            target = [w[j] if j < len(w) else None for w in wanted]
            send = [_as_real(mine[q]) if q in mine and t // per_node == node else np.zeros(0)
                    for t, q in enumerate(target)]
            q = target[rank]
            senders = dict(builders[node, q]) if q is not None else {}
            shapes = [(nk, naux[q], rows[1] - rows[0], 2 * nao) if s in senders else (0,)
                      for s, rows in ((s, senders.get(s)) for s in range(size))]
            got = exchange_blocks(send, shapes, comm)
            if q is not None:
                pieces = [got[s] for s, _ in sorted(senders.items(), key=lambda x: x[1][0])]
                out[q] = np.ascontiguousarray(
                    np.concatenate(pieces, axis=2)).view(np.complex128)
        return out


class _RowSliceBuilder(_RSGDFBuilder):
    """pyscf's range-separated GDF builder for the AO rows from `row0` on,
    with the metric (P|Q)^q taken from `metric` (keyed by `transfer_key`)."""

    def __init__(self, cell, auxcell, kpts, row0, metric):
        super().__init__(cell, auxcell, kpts)
        self._offset = row0 * cell.nao_nr()           # first s1 pair index of the slice
        self._metric = metric

    def get_2c2e(self, uniq_kpts):
        return [self._metric[transfer_key(self.cell, q)] for q in uniq_kpts]

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym):
        # pyscf reads the q = 0 neutralisation overlap from pair index 0: give
        # it the slice's own pair indices, reading the slice's swap rows
        load = super().gen_j3c_loader(_ShiftedGroup(h5group, self._offset), kpt,
                                      kpt_ij_idx, aosym)
        return lambda col0, col1: load(col0 + self._offset, col1 + self._offset)


class _ShiftedGroup:
    """An h5 group whose datasets are read at row indices shifted by `offset`."""

    def __init__(self, group, offset):
        self.group, self.offset = group, offset

    def __contains__(self, key):
        return key in self.group

    def __getitem__(self, key):
        return _ShiftedRows(self.group[key], self.offset)


class _ShiftedRows:
    def __init__(self, dset, offset):
        self.dset, self.offset = dset, offset

    def __getitem__(self, s):
        return self.dset[s.start - self.offset:s.stop - self.offset]


def transfer_key(cell, q):
    """A momentum transfer up to a reciprocal lattice vector, as a dict key."""
    s = cell.get_scaled_kpts(np.reshape(q, (1, 3)))[0]
    return tuple(float(x) + 0.0 for x in np.round(s - np.floor(s + 1e-9), 8) % 1.0)


def row_slices(cell, n):
    """At most n contiguous (first AO, last AO + 1, first shell, last shell + 1)
    slices of the AO rows, nearly equal in AO count, none empty."""
    ao_loc = cell.ao_loc_nr()
    nbas = cell.nbas
    cuts = np.searchsorted(ao_loc, np.linspace(0, ao_loc[-1], min(n, nbas) + 1)[1:-1])
    bounds = np.unique(np.concatenate([[0], np.clip(cuts, 1, nbas - 1), [nbas]]))
    return [(int(ao_loc[a]), int(ao_loc[b]), int(a), int(b))
            for a, b in zip(bounds[:-1], bounds[1:])]


def build_plan(cell, wanted, per_node):
    """
    (tasks, builders) for ranks owning the transfers `wanted[rank]`, taken in
    node blocks of `per_node` consecutive ranks. tasks[rank] is (row slice,
    transfers) or None; builders[b, q] lists (rank, row slice) of every rank
    of block b (rank // per_node) building rows of q. A block builds exactly
    the transfers its own ranks own, so a transfer owned on two nodes (every
    transfer, for PBCDFIntegrals.from_scf) is built on both: its ranks split
    the AO rows (`row_slices`) and, when there are more ranks than slices, the
    transfers into groups.
    """
    size = len(wanted)
    tasks, builders = [None] * size, {}
    for first in range(0, size, per_node):
        block = list(range(first, min(first + per_node, size)))
        needed = sorted({q for r in block for q in wanted[r]})
        if not needed:
            continue
        slices = row_slices(cell, len(block))
        ngroup = max(1, min(len(block) // len(slices), len(needed)))
        groups = [needed[g::ngroup] for g in range(ngroup)]
        for i, r in enumerate(block[:len(slices) * ngroup]):
            tasks[r] = (slices[i % len(slices)], groups[i // len(slices)])
        for g, group in enumerate(groups):
            for q in group:
                builders[first // per_node, q] = [(block[s + g * len(slices)], slices[s])
                                                  for s in range(len(slices))]
    return tasks, builders


def _configure(builder, mydf):
    """The settings pyscf's GDF gives its builder."""
    builder.mesh = mydf.mesh
    builder.linear_dep_threshold = mydf.linear_dep_threshold
    return builder


def _metric_table(cell, auxcell, kpts, needed, mydf):
    """(P|Q)^q of every needed transfer and its conjugate, keyed by transfer_key."""
    qs = np.concatenate([kpts[needed] - kpts[0], kpts[0] - kpts[needed]])
    keys = [transfer_key(cell, q) for q in qs]
    first = sorted({k: i for i, k in reversed(list(enumerate(keys)))}.values())
    builder = _configure(_RSGDFBuilder(cell, auxcell, kpts), mydf).build()
    return dict(zip([keys[i] for i in first], builder.get_2c2e(qs[first])))


def _build_rows(cell, auxcell, kpts, kplus, transfers, rows, metric, mydf,
                workdir, subset):
    """{q: (nk, naux, i1 - i0, nao)} of the transfers `transfers` on the AO rows
    of `rows`, through a slice file in `workdir` that is removed after."""
    i0, i1, s0, s1 = rows
    nk, nao = len(kpts), cell.nao_nr()
    pairs = [(k, kplus[q, k]) for q in transfers for k in range(nk)]
    tmp = tempfile.mkdtemp(dir=workdir, prefix='gdf_rows_')
    try:
        path = os.path.join(tmp, 'j3c.h5')
        builder = _configure(_RowSliceBuilder(cell, auxcell, kpts, i0, metric), mydf)
        builder.make_j3c(path, aosym='s1', j_only=False,
                         shls_slice=(s0, s1, 0, cell.nbas, 0, auxcell.nbas),
                         kptij_lst=(np.array([[kpts[a], kpts[b]] for a, b in pairs])
                                    if subset else None))
        out = {}
        with h5py.File(path, 'r') as f:
            for n, (k, kq) in enumerate(pairs):
                grp = f[f'j3c/{k * nk + kq}']
                block = np.hstack([grp[str(i)][()] for i in range(len(grp))])
                q = transfers[n // nk]
                if q not in out:
                    out[q] = np.empty((nk, block.shape[0], i1 - i0, nao), dtype=complex)
                out[q][k] = block.reshape(-1, i1 - i0, nao)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _as_real(a):
    """A complex array as float64 with its last axis doubled (exchange_blocks)."""
    return np.ascontiguousarray(a).view(np.float64)
