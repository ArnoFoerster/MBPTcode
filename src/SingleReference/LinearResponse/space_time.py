"""Space-time RPA screening: the polarizability in imaginary time, then one
cosine transform to the imaginary-frequency axis.

With a separable (ISDF) ERI,

    (ia|jb) = sum_PQ X_o[P,i] X_v[P,a] Z[P,Q] X_o[Q,j] X_v[Q,b]

the particle-hole bubble separates into an occupied and a virtual half that each
carry only ONE orbital index,

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

Sign convention follows `imaginary_frequency._f_rpa`: f(i.w) = -2 d/(d^2 + w^2)
and chi0 = 2 (C_ov f) C_ov^T; the cosine transform maps e^{-d tau} to
2d/(d^2 + w^2), hence the leading -2 below.

ONE KERNEL FOR BOTH SIDES OF THE CODE. `polarizability_projected_tau` is the
N^3 sweep, and it is the only implementation of it: the energy routes stream
it into chi0 here, and the gradient chain
(`gradients.space_time_adjoint.polarizability_tau`) runs it over every tau
point through `polarizability_projected_sweep` and keeps the result whole,
because every contour-deformation frequency is a fixed linear combination of
proj(tau) and the adjoint lands back on an array of that shape. The reverse
pass mirrors this kernel's tiling block for block. A change to the arithmetic,
the tiling or the tau partition therefore reaches both, and
tests/test_space_time_shared_kernel.py pins the two callers to each other.
`polarizability_projected_rows` is the same sweep held by AUXILIARY ROWS over
the ranks (`ProjRows`), which is how the quasiparticle solves keep it: each
rank its rows of every tau slice, every transform over tau made one auxiliary
row per call so that a row is the same bits at every rank count.

`three_index_slice`, `b_block` and `three_index_ov` sit here for the same
reason: one bra state's pair density, an arbitrary index block and the
particle-hole block are the forward halves of the three-index chain, read by
the contour-deformation energy route and the gradient chain alike, while their
adjoints stay in `gradients.space_time_adjoint` and `gradients.bse_isdf`
beside the rest of the reverse pass.

`rpa_correlation_energy_space_time` is the dRPA energy's own forward sweep, and
`want_tape=True` hands back what a reverse pass through it consumes --
proj(tau) and the two partitions it was built on, so a reverse pass over ranks
splits exactly as the forward sweep did. The gradient's
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

from src.Base.constants import ISDF_TILE_GB
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (agreement, allgather_rows,
                                     contiguous_block, current_comm,
                                     exchange_rows, partition, reduce_sum)
from src.SingleReference.base import get_occ_virt_indices


@dataclass(frozen=True)
class RPAEnergyTape:
    """The forward sweep of E_c^dRPA, as its reverse pass reads it.

    proj_tau is the one whole object of the N^3 route, (ntau, naux, naux), from
    which every frequency is a fixed linear combination; the two index sets are
    the tau points and the frequencies THIS rank owns, so the reverse pass
    splits exactly as the forward did and its reductions carry the same
    disjoint slots. Both are None in a serial run.
    """

    proj_tau: np.ndarray
    tau_indices: np.ndarray
    freq_indices: np.ndarray


class FrequencyBlock(NamedTuple):
    """One serial block of the frequency axis as `ProjRows.blocks` yields it.

    block:  every frequency of the block, the serial block's own list
    owners: the rank that factorizes each of them, or None where every rank
            takes every frequency whole
    ks:     the frequencies this rank takes, in block order
    chi0:   their chi0(i.nu), whole, (len(ks), naux, naux); a reverse pass
            overwrites them with their adjoints before `ProjRows.fold`
    """

    block: list
    owners: list
    ks: list
    chi0: np.ndarray


class ProjRows:
    """proj(tau) held by AUXILIARY ROWS: this rank's rows [r0, r1) of every tau
    slice, (ntau, r1 - r0, naux), the `contiguous_block` of naux it owns.

    Every consumer of proj(tau) is linear in it and elementwise in (P, Q) --
    chi0(i.nu) = sum_tau cosft[nu, tau] proj(tau), W at a real frequency
    below the gap likewise -- so a rank transforms its own rows, and what
    needs a whole matrix gets one frequency or one tau point at a time: the
    rank that factorizes a frequency gathers that chi0, the rank that sweeps a
    tau point gathers that slice. The adjoints come back the same way, so
    projbar lives in the same rows (`zeros_like`, `fold`). At the
    chlorophyllide hexamer the whole array is 153 GB and a rank's rows 19.1 at
    eight ranks.

    THE ROWS ARE THE SAME BITS AT EVERY RANK COUNT. A GEMM's rows depend on
    the call's shape (`owned_frequency_blocks`), and a rank's row block is a
    shape that changes with the rank count, so every transform over tau here
    is made one auxiliary row at a time (`transform_rows`,
    `fold_frequency_rows`): the same (c, (ntau, naux)) call for a row whether
    one rank holds all of them or eight share them. A serial ProjRows is the
    whole sweep, its rows every row, and takes the same calls.

    Exactly one whole-matrix read is answered: np.tensordot(c, proj,
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

    def _size_rank(self):
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
        """`FrequencyBlock` for every serial block of the frequency axis, on
        every rank and in the same order, since every rank holds rows of every
        frequency.

        freq_indices: this rank's frequencies, the round-robin `partition`;
        each is gathered whole to it alone. None gives every rank every
        frequency, gathered to all. Every rank transforms its rows of the
        whole serial block, one auxiliary row at a time, so a frequency's rows
        are the same bits whoever owns it.
        """
        size, rank = self._size_rank()
        nfreq = cosft_wt.shape[0]
        mine = None
        if freq_indices is not None and size > 1:
            mine = sorted(int(k) for k in np.atleast_1d(freq_indices))
            if mine != list(partition(nfreq, rank, size)):
                raise ValueError('proj(tau) by rows routes each frequency to '
                                 'its round-robin owner; this rank was '
                                 f'handed {mine}')
        nrows, empty = self.r1 - self.r0, np.empty((0, self.naux))
        for k0, k1 in frequency_blocks(nfreq, self.naux, tile_gb, live=live):
            block = list(range(k0, k1))
            rows = transform_rows(cosft_wt[block], self.rows)
            if size == 1:
                yield FrequencyBlock(block, None, block, rows)
                continue
            owners = None if mine is None else [k % size for k in block]
            ks = (block if owners is None
                  else [k for k, o in zip(block, owners) if o == rank])
            chi0 = np.empty((len(ks), self.naux, self.naux))
            for m, k in enumerate(block):
                owner = None if owners is None else owners[m]
                here = owner is None or owner == rank
                exchange_rows(
                    rows[m], [(0, nrows) if owner in (None, s) else (0, 0)
                              for s in range(size)],
                    chi0[ks.index(k)] if here else empty,
                    [b if here else (0, 0) for b in self.bounds], self.comm)
            rows = None
            yield FrequencyBlock(block, owners, ks, chi0)

    def fold(self, cosft_wt, fb):
        """self += sum_{k in fb.block} cosft_wt[k, :] fb.chi0(k), on this
        rank's rows: the adjoint of `blocks`' transform, fb.chi0 holding each
        owned frequency's adjoint whole. The owners hand every rank its rows,
        and each rank adds the whole serial block into its rows, one row at a
        time -- the serial association, at every rank count."""
        size, rank = self._size_rank()
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
        size, rank = self._size_rank()
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


def polarizability_imaginary_time(X_o, X_v, eps_o, eps_v, tau_points,
                                  out=None, beta=None):
    """Pi_PQ(i.tau) on the interpolation grid, shape (ntau, M, M).

    X_o, X_v :     (M, n_occ) and (M, n_vir) collocation, occupied and virtual.
    eps_o, eps_v : orbital energies SHIFTED so every eps_v - eps_o > 0; any
                   chemical potential inside the gap does this.
    beta :         inverse temperature, giving the bosonic periodic object
                   Pi(tau) + Pi(beta - tau) that a Matsubara/IR grid needs.
                   Omit for the T = 0 half-line function of the minimax grids.

    The mirror term is not a small correction: e^{i nu_n beta} = 1 for bosonic
    frequencies, so tau -> beta - tau maps the integral onto itself and the
    mirror contributes exactly as much as the direct term. Dropping it is a
    factor of two at every beta.
    """
    M = X_o.shape[0]
    ntau = len(tau_points)
    if out is None:
        out = np.empty((ntau, M, M))
    for k, tau in enumerate(tau_points):
        Go = (X_o * np.exp(eps_o * tau)) @ X_o.T
        Gv = (X_v * np.exp(-eps_v * tau)) @ X_v.T
        np.multiply(Go, Gv, out=out[k])
        if beta is not None:
            tb = beta - tau
            out[k] += (((X_o * np.exp(eps_o * tb)) @ X_o.T)
                       * ((X_v * np.exp(-eps_v * tb)) @ X_v.T))
    return out


def tile_rows(M, tile_memory_gb, per_row_bytes):
    """Grid rows per tile so that `per_row_bytes` x rows fits in
    tile_memory_gb: at least one row, at most all M of them."""
    return max(1, min(M, int(tile_memory_gb * 1e9 / max(per_row_bytes, 1))))


def polarizability_work(M, tile_memory_gb=ISDF_TILE_GB):
    """The two (rows, M) Green's-function tiles `polarizability_projected_tau`
    works in, sized for this M and budget, to be reused across tau points."""
    rows = tile_rows(M, tile_memory_gb, 3 * M * 8)
    return np.empty((rows, M)), np.empty((rows, M))


def split_branches(X, eps, nocc, mu=None):
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
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    if isinstance(X, SlicedFactors):
        X_o, X_v = X.branches(occ, virt)
    else:
        X_o, X_v = np.ascontiguousarray(X[:, occ]), np.ascontiguousarray(X[:, virt])
    return X_o, X_v, eps[occ] - mu, eps[virt] - mu, mu, occ, virt


def polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                 tile_memory_gb=ISDF_TILE_GB, out=None,
                                 work=None):
    """chi0 at ONE imaginary time, already projected to the auxiliary basis:

        proj_ab(tau) = -2 sum_PQ D[P,a] (Go_PQ Gv_PQ) D[Q,b]

    Tiled over grid rows, which is exact because the expression is a sum over
    them, so the M x M object never exists. Contract the Pi block with D first:
    the other order builds an (naux, M) intermediate instead.

    THE ONE KERNEL OF THE N^3 SWEEP, for the energy routes and the gradient
    chain alike -- see the module docstring. The reverse pass in
    `gradients.space_time_adjoint.polarizability_backward` mirrors this tiling,
    so the row budget here, 3 * M * 8 bytes per row, is a convention and not a
    private detail.

    out:  (naux, naux) buffer the result is written into, so a sweep over many
          tau points allocates nothing per point. Zeroed here; allocated when
          None.
    work: the two (rows, M) tiles from `polarizability_work`, reused across
          calls for the same reason. Their contents are never read.
    """
    M = X_o.shape[0]
    naux = D.shape[1]
    rows = tile_rows(M, tile_memory_gb, 3 * M * 8)
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
    for p0 in range(0, M, rows):
        p1 = min(p0 + rows, M)
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
    `Base.utils.mpi_grid` hands out, and an all-reduce of the result is exact.
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
    tau owner's verbatim, so every row is the one the zero-padded all-reduce
    delivered. Serially the result wraps the whole sweep.

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


def chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=None, stream=True,
                             tau_indices=None, tile_memory_gb=ISDF_TILE_GB):
    """chi0(i.omega) in the DF auxiliary basis, shape (nfreq, naux, naux).

    X :    (M, norb) collocation in the MO basis, or `SlicedFactors`, whose
           occupied and virtual columns `split_branches` gathers whole.
    D :    (M, naux) Coulomb factor, Z = D D^T; carries the result back to the
           auxiliary basis every DF consumer speaks.
    grid : TimeFrequencyGrid with an imaginary-time axis.

    Same chi0 as `solve_rpa_screening_df`, so W follows as [I - chi0]^-1.

    stream=True projects and accumulates each Pi(i.tau) immediately. Identical
    algebraically, since

        chi0(i.w) = -2 sum_tau cosft_wt[w,tau] (D^T Pi(tau) D),

    but the peak drops from (ntau, M, M) to one (M, M).
    """
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, nocc, mu)
    npts = X_o.shape[0]

    if not stream:
        Pi_tau = polarizability_imaginary_time(X_o, X_v, e_o, e_v,
                                               grid.tau_points)
        Pi_w = np.tensordot(grid.cosft_wt, Pi_tau, axes=(1, 0))
        return -2.0 * np.einsum('Pa,wPQ,Qb->wab', D, Pi_w, D, optimize=True)

    naux = D.shape[1]
    chi0 = np.zeros((grid.nfreq, naux, naux))
    # One projected point and the two Green's-function tiles, allocated once
    # for the whole sweep and handed to the kernel every time.
    proj = np.empty((naux, naux))
    work = polarizability_work(npts, tile_memory_gb)
    which = range(grid.ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        polarizability_projected_tau(X_o, X_v, e_o, e_v, D, grid.tau_points[k],
                                     tile_memory_gb=tile_memory_gb, out=proj,
                                     work=work)
        chi0 += grid.cosft_wt[:, k, None, None] * proj
    return chi0


def frequency_blocks(nfreq, naux, tile_gb, live):
    """(k0, k1) ranges over the frequency axis such that `live` (naux, naux)
    arrays per frequency fit in tile_gb. The one knob for both the M-row tiles
    and the frequency blocks: a working-set budget, not a count."""
    nb = max(1, min(nfreq, int(tile_gb * 1e9 / max(live * naux * naux * 8, 1))))
    return [(k0, min(k0 + nb, nfreq)) for k0 in range(0, nfreq, nb)]


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
    serial block on every rank (`ProjRows.fold`), with a (ntau, naux)
    temporary per row where the whole product was (ntau, naux, naux)."""
    for i in range(blk.shape[1]):
        out[:, i, :] += np.tensordot(c, blk[:, i, :], axes=(0, 0))


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

    Written as one GEMM per bra function rather than a single einsum. The flops
    are identical; the library path is not. numpy's einsum falls off BLAS for
    a three-operand contraction carrying a batch index and runs it at ~2
    GFlop/s, where the same work as matrix products reaches 60-85 -- measured,
    a factor of FORTY. Looping the outer index also keeps the working set at
    (M, n_q) instead of the (M, n_p n_q) an einsum path materializes, which at
    production sizes is the difference between fitting and not.
    """
    Xp, Xq = X[:, p_idx], (X if Y is None else Y)[:, q_idx]
    out = np.empty((D.shape[1], Xp.shape[1], Xq.shape[1]))
    for a in range(Xp.shape[1]):
        out[:, a, :] = D.T @ (Xp[:, a, None] * Xq)
    return out


def three_index_ov(X, D, eps, nocc, tile_gb=ISDF_TILE_GB):
    """C_ov[P, (i,a)] = sum_k D[k,P] X[k,i] X[k,a], (naux, nocc*nvirt).

    The particle-hole block of the three-index tensor -- the O(N^4) object a
    contour deformation's RESIDUE term needs, because W at a real frequency has
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
    proj_tau for exactly those; a block with no owned frequency is skipped
    outright. The frequency axis of the contour deformation is the one whose
    per-point work is a naux^3 factorization and whose results are per point,
    which is what makes it the reduction-free axis to split.

    EVERY ROW IS THE SERIAL ROW, on any BLAS. A GEMM row is not a function of
    that row alone: OpenBLAS picks its tail kernel and its threading from the
    call's shape, so rows a rank transforms on their own differ from the
    serial block's in the last bits. At 8 ranks on water/cc-pVDZ, 3 of 24
    frequencies a rank, OpenBLAS 0.3.18 moves 30% of the contour-deformation
    wc by 1 to 224 ulp where MKL 2021.4 moves none, and that broke the bitwise
    quasiparticle roots over 8 ranks. So every serial block holding one of the
    caller's frequencies is transformed with the serial call and the owned
    rows are compacted to its front, in place; one rank is that call alone.
    An output partition over frequencies therefore gathers to the serial
    result bitwise, while a sum over frequencies still re-associates.

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


def laplace_representation_error(grid, eps, nocc, freq):
    """max over y = d +- freq of |y sum_k w_k e^{-y tau_k} - 1|: how well the
    grid's bare quadrature carries this real frequency, on the pair energies
    that actually occur. inf when freq reaches the gap (a real pole).

    The gate on the imaginary-time form of W at a REAL frequency, which is what
    a contour-deformation residue below the particle-hole gap asks for:
    -2d/(d^2 - w^2) = -int 2 cosh(w tau) e^{-d tau} dtau holds only where the
    grid still represents e^{-y tau} on every y = d -/+ w.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    d = (np.asarray(eps)[virt][None, :] - np.asarray(eps)[occ][:, None]).ravel()
    y = np.concatenate([d - freq, d + freq])
    if y.min() <= 0.0:
        return np.inf
    q = np.exp(-np.outer(y, grid.tau_points)) @ grid.tau_weights
    return float(np.abs(q * y - 1.0).max())


def screening_space_time(X, D, eps, nocc, grid, mu=None):
    """W(i.omega) = [I - chi0(i.omega)]^-1, via the imaginary-time route."""
    chi0 = chi0_imaginary_frequency(X, D, eps, nocc, grid, mu=mu)
    eye = np.eye(chi0.shape[-1])
    return np.array([np.linalg.inv(eye - c) for c in chi0])


def rpa_correlation_energy_space_time(X, D, eps, nocc, grid, mu=None,
                                      tile_gb=ISDF_TILE_GB, screening=None,
                                      counter_term=None, comm=None,
                                      want_tape=False):
    """dRPA correlation energy E_c = (1/2pi) int dw Tr{log(1 - chi0) + chi0}.

    The quadrature of `rpa_energy.rpa_correlation_energy_imaginary_axis`, with
    chi0 from the imaginary-time route. The frequencies come out of proj(tau)
    one block at a time (`owned_frequency_blocks`), so the axis is never whole
    and the peak is proj(tau) plus one block.

    screening = (N, g): the interaction that builds the LOGARITHM is rescaled
    to v + g(w) vtilde while D stays in the dressed gauge. In that gauge the
    rescaling is a similarity on chi0 alone -- with N = V_d^(-1/2) vtilde
    V_d^(-1/2), V_d = V + vtilde, and S_w = I - (1 - g_w) N,

        log det(I - S_w c_w) = log det(I - (v + g_w vtilde) P_1(iw)) ,

    so g == 1 is the dressed interaction (the default) and g == 0 the bare one.
    g is one scalar per frequency, N one naux x naux matrix.
    counter_term = C: the LINEAR term becomes Tr(C c_w) in place of Tr(c_w).
    C = I - N puts the BARE interaction there, which is what an exact block
    fold of the log-determinant over a non-overlapping solvent keeps, and what
    leaves the leading solute-solvent dispersion term in the energy instead of
    cancelling it (`solvated_rpa_energy.fold_terms` chooses the pair).

    comm: an MPI communicator (or `simulated_world` rank) whose ranks all call
    this in lockstep. The tau sweep is split over tau and reduced once,
    ntau x naux^2 over disjoint slots; the frequency loop is split over
    frequencies, whose partial E_c is one scalar reduction. Each frequency's
    chi0 is the serial one bitwise (`owned_frequency_blocks`), so only that
    sum re-associates; serial is bitwise unchanged. None is `current_comm()`.
    The inputs are identical by construction and are not broadcast; an audited
    run compares their digests and those of E_c and proj(tau)
    (`mpi_grid.agreement`).

    want_tape: also return the `RPAEnergyTape` a reverse pass reads.
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
    eye = np.eye(proj_tau.shape[-1])
    e_c = 0.0
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
            _, logdet = np.linalg.slogdet(eye - c)
            e_c += grid.omega_weights[k] * (logdet + linear)
    if nranks > 1:
        e_c = float(reduce_sum(np.array([e_c]), comm)[0])
    e_c /= 2.0 * np.pi
    if nranks > 1:
        agreement((e_c, proj_tau), comm, audit_only=True,
                  label='rpa_correlation_energy_space_time outputs')
    if not want_tape:
        return e_c
    return e_c, RPAEnergyTape(proj_tau, tau_mine, nu_mine)
