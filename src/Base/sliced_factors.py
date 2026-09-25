"""The separable factors held as grid-point slices, one row block per rank.

Replicated, every rank holds X_mo (M, nmo), X_ao (M, nao) and D (M, naux)
whole at every stage: 36.5 + 36.5 + 93.9 MB at pentacene/cc-pVTZ (M 5328,
nao 856, naux 2202), 10.3 + 10.3 + 26.6 GB at the chlorophyllide hexamer
/cc-pVTZ (M 117762, nao 10980, naux 28236, nocc 972). `SlicedFactors` keeps
this rank's `contiguous_block` of the grid rows of all three, the block the
BSE action already owns of Zt, and a stage that reads an array whole gathers
it once (`gather`, an output partition: every row travels verbatim, so the
gathered array is the replicated one bit for bit) and drops it when it
returns.

WHO READS WHAT, and the factor bytes a rank holds at 8 / 16 ranks,
replicated -> sliced (the replicated side counts the contiguous X_o and X_v
copies a stage makes of the whole X_mo, 3.1 + 33.4 MB and 0.92 + 9.43 GB):

  stage              whole         pentacene, MB         hexamer, GB
  between stages     -             167 -> 20.9 / 10.4    47.3 -> 5.9 / 3.0
  chi0 sweep         X_o X_v D     203 -> 151 / 141      57.6 -> 42.9 / 39.9
  self-energy sweep  X_ao D        167 -> 151 / 141      47.3 -> 42.9 / 39.9
  BSE action setup   X_o D         203 -> 122 / 110      57.6 -> 34.7 / 31.1
  BSE trial vector   X_o           203 -> 28.1 / 15.6    57.6 -> 8.0 / 4.5

The chi0 sweep (GW window, BSE diagonal, static W) and the self-energy sweep
contract the grid index on BOTH sides of every M^2 GEMM (P x Q tiles over the
whole grid), so a rank holding rows alone would need every other rank's rows
per tau point; they gather once per solve. Dividing them as well is a
two-dimensional (row x column) decomposition with the column blocks passed
between ranks, which re-associates their sums. The block action reads X_v
and D by rows once Zt = (D_rows W) D^T is built, and X_o whole in
(Zt * P) X_o. The static exchange reads none of the three (the mean field's
own K). The replicated fit's tail forms X, X_mo, D and M whole on every rank
before the cut, below its own nk^2 Gram matrix (227 MB, 111 GB);
`separable_factors(fit='rows')` builds the rows themselves instead
(`separable_ri.fit_rows`, held through `from_rows`), so no rank holds any
of them whole at any stage of the fit. The gradient chains hold the same
rows (`FrozenFactorization(sliced=True)`); their adjoint kernels gather what
they read whole once per sweep through `whole_factor` below.

WHY THE SLICES ARE CUT FROM THE WHOLE PRODUCTS. A row block of a GEMM is not
the same bits as those rows of the whole GEMM: (D C)[r0:r1] against
D[r0:r1] C differed in 14 of 18 row blocks on MKL 2021 at M 1333, naux 311.
So `separable_factors(sliced=True)` forms X_mo and D exactly as the
replicated fit does, locksteps them as it does, and only then keeps this
rank's rows; every gathered array is therefore the replicated array, and the
layout adds no difference of its own: on one mean field, and on a pyscf that
repeats its bits, every distributed output is bitwise the replicated run's.
pyscf's OpenMP GEMM does not repeat them -- it adds its partial sums in
thread-arrival order -- so a gradient chain's own K builds and mean-field
force differ between two runs, and two runs are compared on an anchored bar.
Rows built by the row-distributed fit are the other answer to the same
fact: every array of that fit is computed one fixed tile at a time, whoever
owns the tile, so a rank's rows are bitwise the same rows of a one-rank run of it at any rank
count -- another realization than the replicated fit's, not its bits.
"""
import numpy as np

from src.Base.utils.mpi_grid import (allgather_rows, contiguous_block,
                                     current_comm)


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
        held at once, kept as `fit_held` for a memory ledger; not an array,
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
    """`a` as an array for one sweep: `SlicedFactors` gathered whole ONCE as
    its `name`, 'X_mo', 'D' or 'X_ao', an array handed back as it is. Slices
    reaching a sweep on other ranks than they were cut for -- a serial
    reference among them -- are refused by `SlicedFactors.require`."""
    if not isinstance(a, SlicedFactors):
        return a
    return a.require(current_comm()).gather(name)
