"""The Casida A/B block actions of the matrix-free Davidson, on the DF and the ISDF route.

`df_block_action` contracts pyscf's three-index factor directly;
`isdf_block_action` does Fock-like builds on the interpolation grid and divides
the rows of the screened kernel over MPI ranks. Both return `(apply_AB, diag_d)`
and carry the clocks and `screened_diagonal` the Davidson reads off them.
`_block_action` dispatches on the polarizability and the factorization.
"""
import time

import numpy as np

from src.Base.constants import DAVIDSON_DIAGONAL_TILE, ISDF_TILE_GB, KAPPA
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (allgather_blocks, contiguous_block,
                                     current_comm, lockstep,
                                     ordered_chain_sum, reduce_scatter_rows,
                                     reduce_sum)
from src.Base.utils.threads import openmp_threads, row_map
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.space_time import _unpack_factors
from src.SingleReference.LinearResponse.isdf_bse_adjoint import GridTiles


#: The collectives of a distributed block action, each timed apart as
#: `davidson_comm_<kind>`: the checked lockstep of the trial vectors (a digest,
#: and a broadcast only where one differs) and of the result; the all-gather of
#: z X_v^T, M x n_occ per vector; the reduce-scatter of X_o^T (Zt * P), n_occ x M
#: in and each rank's own columns out, per vector; and the all-reduces of p D,
#: naux per vector, and of the batch's (n_occ, n_vir) output slabs.
COMM_KINDS = ('lockstep', 'gather', 'reduce_grid', 'reduce_pairs')


#: The local work of an ISDF block action, each timed apart as
#: `davidson_action_<piece>`, serially too: the GEMMs of the head z X_v^T, of
#: the exchange tiles and of the tails; the Hadamard product with Zt; the
#: Hartree term's own contractions; the owner-order write of X_o^T (Zt * P);
#: and the batch's element-wise terms, d z and the reduced slabs added in.
#: With the collectives of COMM_KINDS they are the whole action but its
#: Python overhead.
ACTION_PIECES = ('gemm', 'hadamard', 'hartree', 'owner', 'batch')


class _CommClock:
    """Wall seconds one block action spends inside its collectives, read by
    the Davidson as `davidson_comm`, and the same seconds by collective in
    `by_kind` (COMM_KINDS); nothing is timed without a rank count."""

    __slots__ = ('seconds', 'by_kind')

    def __init__(self):
        self.seconds = 0.0
        self.by_kind = dict.fromkeys(COMM_KINDS, 0.0)


class _PieceClock:
    """Wall seconds one ISDF block action spends in each piece of its local
    work (ACTION_PIECES), read by the Davidson as `davidson_action_<piece>`;
    timed serially and under a comm alike, and never around a collective."""

    __slots__ = ('by_piece',)

    def __init__(self):
        self.by_piece = dict.fromkeys(ACTION_PIECES, 0.0)

    def lap(self, piece, t0):
        """The time now, with the seconds since t0 credited to `piece`."""
        t1 = time.time()
        self.by_piece[piece] += t1 - t0
        return t1


def _block_action(lr_solver, nocc, polarizability, W_aux, isdf_factors,
                  spin='singlet', comm=None, tda=False):
    """Mode dispatch shared by the solver and the (A-B) instability probe;
    `tda` returns the A half alone (Bz None)."""
    comm = current_comm() if comm is None else comm
    if spin not in KAPPA:
        raise ValueError(f"spin {spin!r}: one of {', '.join(sorted(KAPPA))}")
    mode = polarizability.upper()
    if mode == 'RPA':
        lBSE, w = False, None
    elif mode == 'TDHF':
        lBSE, w = True, None
    elif mode == 'BSE':
        if W_aux is None:
            raise ValueError("polarizability='BSE' requires W_aux (see LinearResponseSolver.solve_rpa_screening).")
        lBSE, w = True, W_aux
    else:
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose 'RPA', 'TDHF', or 'BSE'.")

    if isdf_factors is None:
        return df_block_action(lr_solver, nocc, lBSE, w, spin=spin, comm=comm,
                               tda=tda)
    return isdf_block_action(lr_solver, nocc, lBSE, w, isdf_factors, spin=spin,
                             comm=comm, tda=tda)


def bse_pair_diagonal(eps, nocc):
    """eps_a - eps_i on every particle-hole pair, (n_occ, n_vir).

    The one place a BSE puts a spectrum on its diagonal. A BSE@GW carries the
    quasiparticle energies here while its kernel screens at whatever spectrum
    built W (the G0W0-BSE split), so the two spectra enter the problem in
    two different places and each route has to take them from the same rule.
    The matrix-free action reads this straight off `lr_solver.eps`; the dense
    route builds A at the screening spectrum and moves the diagonal onto the
    quasiparticle one by the difference of two calls here. Both are this
    expression, and the pair layout -- occupied slow, virtual fast -- is what
    makes a vector of one route a vector of the other.

    The split is by index (`get_occ_virt_indices`), so a quasiparticle array
    whose entries do not ascend still pairs occupied i with virtual a.
    """
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    return eps[virt][None, :] - eps[occ][:, None]


def df_block_action(lr_solver, nocc, lBSE, W_aux,
                    tile_memory_gb=ISDF_TILE_GB, spin='singlet', comm=None,
                    tda=False):
    """(apply_AB, diag_d): the Casida blocks as an action on a batch of trial
    vectors, contracted straight out of pyscf's three-index factor.

    tda: the A half alone, for the Tamm-Dancoff problem: Bz is None and the
    swapped exchange term, B's alone, is never contracted.

    Returned rather than solved with, so the same action can be timed against
    `isdf_block_action` or driven by an eigensolver other than Davidson's.
    The exchange contractions go through (nvec, naux, nocc, nvirt)
    intermediates and, for the swapped term, (nvec, naux, nvirt, nvirt) ones
    where one vector's fits `tile_memory_gb` and (nvec, naux, nocc, nocc) ones
    where it does not, chunked over trial vectors to `tile_memory_gb`;
    Davidson batches ~2-3x the requested roots. What cannot be chunked away is
    W C_vv, naux*nvirt^2 held for the whole solve (formed a tile at a time
    serially, with no unscreened copy beside it), and is what
    `isdf_block_action` exists to remove.

    comm: by default the current `distributed` region's. Nothing here is
    divided over ranks: every rank applies the whole serial action, so the
    output is the same on every rank because the inputs are -- the factor
    slices, W_aux and the pair diagonal are lockstepped once here, in one
    call, and the trial vectors at entry, one `lockstep` per batch, in place.
    Both are checked (`mpi_grid.lockstep(check=True)`): only an array a
    digest shows apart from rank 0's is broadcast.
    No other collective runs, so the arithmetic is the serial one and the
    roots are bitwise the serial ones. The returned action carries the
    `comm_clock` its entry lockstep is timed on, and `screened_diagonal()`,
    d - (ii|W|aa) from the factor's own diagonals, formed only when a
    Davidson preconditions with it (`_preconditioner_diagonal`).
    """
    comm = current_comm() if comm is None else comm
    nranks = comm.Get_size() if comm is not None else 1
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)

    C_ov = lr_solver.df_coeff[:, occ[:, None], virt]
    C_oo = lr_solver.df_coeff[:, occ[:, None], occ]
    # (P|aa): all of C_vv the screened preconditioner reads
    C_vv_diag = lr_solver.df_coeff[:, virt, virt]
    C_vv = None
    diag_d = bse_pair_diagonal(lr_solver.eps, nocc)
    if nranks > 1:
        # checked: C_vv alone is naux n_vir^2
        C_vv = lr_solver.df_coeff[:, virt[:, None], virt]
        diag_d, C_ov, C_oo, C_vv, W_aux = lockstep(
            (diag_d, C_ov, C_oo, C_vv, W_aux), comm, check=True)
        C_vv_diag = np.einsum('Paa->Pa', C_vv).copy()
    comm_clock = _CommClock()
    factor = KAPPA[spin]

    def apply_V(z):
        t = np.einsum('Pjb,njb->nP', C_ov, z, optimize=True)
        return np.einsum('nP,Pia->nia', t, C_ov, optimize=True)

    # Precompute the W_aux contraction once; redoing it per apply_exchange_* call
    # would make each Davidson iteration ~naux times more expensive.
    if W_aux is None:
        WC_vv = (lr_solver.df_coeff[:, virt[:, None], virt] if C_vv is None
                 else C_vv)
        WC_ov = C_ov
    else:
        WC_vv = (_screened_vv(lr_solver.df_coeff, virt, W_aux, tile_memory_gb)
                 if C_vv is None
                 else np.einsum('PQ,Qab->Pab', W_aux, C_vv, optimize=True))
        WC_ov = np.einsum('PQ,Qjb->Pjb', W_aux, C_ov, optimize=True)
    del C_vv

    naux = C_ov.shape[0]
    no, nv = len(occ), len(virt)
    per_vec = 8 * naux * nv * max(no, nv)
    chunk = max(1, int(tile_memory_gb * 1e9 / max(per_vec, 1)))

    def apply_exchange_direct(z):
        # Explicit 2-step contraction (a single 3-operand einsum silently costs
        # O(naux*nocc^2*nvirt^2) instead of O(naux*nocc*nvirt*max(nocc,nvirt))).
        # optimize=True matters too: ~13x slower without it for these shapes.
        out = np.empty_like(z)
        for c0 in range(0, len(z), chunk):
            tmp = np.einsum('Pab,njb->nPja', WC_vv, z[c0:c0 + chunk], optimize=True)
            out[c0:c0 + chunk] = np.einsum('Pij,nPja->nia', C_oo, tmp, optimize=True)
        return out

    # (ib|W|ja) z_jb: contracting z into WC_ov first costs naux n_occ n_vir^2
    # and an (naux, n_vir, n_vir) temporary per vector; through the
    # (naux, n_occ, n_occ) overlap of C_ov with z it is naux n_occ^2 n_vir.
    # The first association is kept wherever its temporary fits a tile, so
    # the solves that fit keep their bits
    swap_through_oo = 8 * naux * nv * nv > tile_memory_gb * 1e9

    def apply_exchange_swap(z):
        out = np.empty_like(z)
        for c0 in range(0, len(z), chunk):
            if swap_through_oo:
                tmp = np.einsum('Pib,njb->nPij', C_ov, z[c0:c0 + chunk], optimize=True)
                out[c0:c0 + chunk] = np.einsum('nPij,Pja->nia', tmp, WC_ov, optimize=True)
            else:
                tmp = np.einsum('Pja,njb->nPab', WC_ov, z[c0:c0 + chunk], optimize=True)
                out[c0:c0 + chunk] = np.einsum('Pib,nPab->nia', C_ov, tmp, optimize=True)
        return out

    def apply_AB(z):
        if nranks > 1:
            z = _timed(comm_clock, 'lockstep', lockstep, z, comm,
                       check=True)                     # rank 0's
        # The Hartree term enters A and B identically, so it is contracted once
        # per trial vector rather than once per block. A triplet has kappa = 0
        # and drops it entirely -- not merely scaled to zero, skipped, since it
        # is two of the step's contractions.
        if factor:
            v = factor * apply_V(z)
            Az = diag_d[None, :, :] * z + v
            Bz = v
        else:
            Az = diag_d[None, :, :] * z
            Bz = np.zeros_like(z)
        if lBSE:
            Az = Az - apply_exchange_direct(z)
            if not tda:
                Bz = Bz - apply_exchange_swap(z)
        return Az, None if tda else Bz

    def screened_diagonal():
        """d - (ii|W|aa) on every pair, (n_occ, n_vir), from the factor's own
        diagonals C_oo[:, i, i] and C_vv[:, a, a]; d alone for RPA."""
        if not lBSE:
            return diag_d
        dressed = diag_d - screened_direct_diagonal(
            np.einsum('Pii->Pi', C_oo), C_vv_diag, W_aux)
        if nranks > 1:
            dressed = lockstep(dressed, comm, check=True)
        return dressed

    apply_AB.comm_clock = comm_clock
    apply_AB.screened_diagonal = screened_diagonal
    return apply_AB, diag_d


def _screened_vv(coeff, virt, W_aux, tile_memory_gb=ISDF_TILE_GB):
    """W C_vv, (naux, n_vir, n_vir), by blocks of the first virtual index read
    straight out of the full factor `coeff`: no unscreened naux n_vir^2 copy
    is held beside the screened one."""
    naux, nv = coeff.shape[0], len(virt)
    rows = max(1, int(tile_memory_gb * 1e9 / (8 * naux * nv)))
    if rows >= nv:
        # one tile: the product exactly as it was formed whole
        return np.einsum('PQ,Qab->Pab', W_aux, coeff[:, virt[:, None], virt],
                         optimize=True)
    out = np.empty((naux, nv, nv))
    for a0 in range(0, nv, rows):
        out[:, a0:a0 + rows] = np.einsum(
            'PQ,Qab->Pab', W_aux, coeff[:, virt[a0:a0 + rows, None], virt],
            optimize=True)
    return out


def _screened_rows(D_mine, D, W_aux, comm=None):
    """This rank's rows of the screened kernel Zt = D W D^T, the bare Z = D D^T
    where W_aux is None (TDHF).

    Under a rank count the product is reassociated: `D_mine (W D^T)` forms the
    (naux, M) inner product whole on every rank, naux^2 M multiply-adds and
    naux M doubles that no rank count divides, while `(D_mine W) D^T` gives
    the same matrix through an (nmine, naux) intermediate, both falling with
    the rank count.

    The two associations are not bitwise equal, so the serial path keeps
    `D_mine (W D^T)` and the reassociation is taken only where there is a
    rank count to divide. Every rank reads the same size and takes the same
    branch, which the partials being summed across ranks requires.
    """
    if W_aux is None:
        return D_mine @ D.T
    if comm is not None and comm.Get_size() > 1:
        return (D_mine @ W_aux) @ D.T
    return D_mine @ (W_aux @ D.T)


def self_pair_densities(D_rows, X_o_rows, X_v_rows, tiles, first=0):
    """(naux, n_occ + n_vir): the fitted densities of the orbitals' own
    squares, [D^T (X_o o X_o) | D^T (X_v o X_v)], summed over the grid-row
    `tiles`, each tile's addend added in tile order onto zeros; the arrays
    hold the grid's rows from `first` on."""
    B = np.zeros((D_rows.shape[1], X_o_rows.shape[1] + X_v_rows.shape[1]))
    addend = np.empty_like(B)
    for p0, p1 in tiles:
        a, b = p0 - first, p1 - first
        squares = np.hstack([X_o_rows[a:b], X_v_rows[a:b]])
        np.multiply(squares, squares, out=squares)
        np.matmul(D_rows[a:b].T, squares, out=addend)
        B += addend
    return B


def screened_direct_diagonal(B_o, B_v, W_aux):
    """(ii|W|aa) on every pair, (n_occ, n_vir), from the fitted densities of
    the occupied and the virtual squares: (B_o^T W) B_v, the bare Coulomb's
    B_o^T B_v where W_aux is None (TDHF). The screened direct term's
    diagonal, the same for a singlet and a triplet."""
    if W_aux is None:
        return B_o.T @ B_v
    return (B_o.T @ W_aux) @ B_v


def _timed(clock, kind, fn, *args, **kwargs):
    """fn(*args, **kwargs), its wall seconds added to `clock`, and to its
    `kind` of collective (COMM_KINDS), when there is one."""
    if clock is None:
        return fn(*args, **kwargs)
    t0 = time.time()
    out = fn(*args, **kwargs)
    dt = time.time() - t0
    clock.seconds += dt
    clock.by_kind[kind] += dt
    return out


def _hadamard_rows(blk, Zt_rows, threads):
    """blk *= Zt_rows, element by element, rows on `threads` threads."""
    def product_rows(a, b):
        np.multiply(blk[a:b], Zt_rows[a:b], out=blk[a:b])

    row_map(product_rows, len(blk), threads)


def _owner_write(T_owned, owners, Tb, first, threads):
    """Tb's columns [s0, s1) into owner s's block of T, n_occ rows on
    `threads` threads: copied for a row block's first tile and added in for
    every later one.

    The copy is the accumulation into a zeroed T, 0 + x = x, but for the
    sign of an exact zero: (+0) + (-0) is +0 where the copy keeps -0. T only
    enters sums -- the reduce-scatter, then the GEMM with X_v -- and x + (-0)
    = x + (+0) = x for every x != 0, so the sign can reach no output that is
    not itself an exact zero.
    """
    def write_rows(i0, i1):
        for (s0, s1), Ts in zip(owners, T_owned):
            if first:
                np.copyto(Ts[i0:i1], Tb[i0:i1, s0:s1])
            else:
                np.add(Ts[i0:i1], Tb[i0:i1, s0:s1], out=Ts[i0:i1])

    row_map(write_rows, len(Tb), threads)


def isdf_block_action(lr_solver, nocc, lBSE, W_aux, isdf_factors,
                      tile_memory_gb=ISDF_TILE_GB, spin='singlet', comm=None,
                      tda=False):
    """The same A/B action from a separable RI: Fock-like builds on the grid.

    tda: the A half alone, for the Tamm-Dancoff problem: Bz is None, and
    neither (Zt * P) X_o nor the B exchange slab is formed, one of the three
    M^2 n_occ GEMMs of a trial vector.

    Nothing here carries a (naux, nvirt, nvirt) array, and the only object that
    scales as M^2 is the screened kernel Zt, built once. Everything else is
    tiled over grid rows, because every contraction is a sum over them:

        [K z]_ia = sum_k X_o[k,i] sum_k' (Zt * P)[k,k'] X_v[k',a]

    so a row block contributes X_o[blk]^T (Zt[blk] * P[blk]) X_v and the M x M
    Hadamard product is never formed: same flop count and GEMM shapes, one
    (M, M) array where the whole product would hold three. The same row
    decomposition distributes Zt across ranks.

    comm: with an MPI communicator, by default the current `distributed`
    region's, this rank builds and holds only its contiguous block of rows of
    Zt (`contiguous_block`), M^2 / nranks, and the row loop runs over that
    block alone; the product that builds those rows is reassociated so its
    intermediate follows them (`_screened_rows`). Both exchange terms are sums
    over the row index (X_o[blk]^T (...) for A and, through U, for B), so
    each rank produces a partial (n_occ, n_vir) per trial vector and one
    all-reduce of the batch, nvec x n_occ x n_vir doubles per term, completes
    them. The Hartree term's last contraction, X_o^T (u X_v), is a sum over
    the same rows and rides the same reduction as one more slab.

    The three pieces that read or write the whole grid index each take the
    split that fits their shape:

      z X_v^T is an output partition: a rank builds only the grid rows it
        owns, M n_occ n_vir / nranks flops, and one all-gather of M x n_occ
        doubles (`allgather_blocks`) gives everyone the whole, which the
        exchange needs. Every row is computed once, by its owner, so the
        ranks hold identical bits.
      p and p D are a reduction: p is a per-grid-point quantity, so a rank
        forms its own rows of it and contracts them with its own rows of D;
        only the naux-long p D crosses.
      X_o^T (Zt * P) X_v is a partial over the rows in its first index and
        whole in its second, and a rank contracts only its own columns of it
        with its own rows of X_v. So the (n_occ, M) partial is
        reduce-scattered (`reduce_scatter_rows`): each rank receives the
        summed (n_occ, its columns) block alone, half the traffic of an
        all-reduce of the whole. The partial is laid out in owner order, rank
        s's columns filling rows [s0, s1) of an (M, n_occ) buffer, and a
        block's first tile is copied in rather than added to zeros
        (`_owner_write`, identical but for the sign of an exact zero), so no
        memset precedes it. Contracting X_v first would need no reduce but
        costs M n_vir per row in place of n_occ M, more since the virtual
        space is the wide one.

    The partials sum in rank order rather than in one GEMM, so a distributed
    run agrees with the serial one to the last bits, not in them.

    The output is the same on every rank by construction, which lets the
    Davidson and the probe run replicated: the trial vectors are lockstepped
    at entry (rank 0's written into every rank's in place, one checked
    `lockstep` per batch, so only a drifted batch is broadcast) and the pair
    diagonal once here, since d z is the one term no collective touches;
    every other term is gathered or all-reduced. Serially (comm None or one
    rank) the row block is the whole grid.

    The element-wise passes (the Hadamard product with Zt, the owner-order
    write, and the batch's d z and slab updates) run on the OpenMP pool's
    threads through `row_map`, cut over rows; every element is still the one
    operation the unsplit call applies, so the output is the same bits at
    every thread count. These passes are memory-bound, so a second thread
    shortens them by less than half.

    `SlicedFactors` in `isdf_factors`: once Zt is built the action reads only
    this rank's rows of X_v and D; D is gathered whole once for the Zt
    product and dropped, and X_o, which (Zt * P) X_o reads whole, is gathered
    once. The gathered arrays are the replicated ones, so the output is
    bitwise the replicated-factor action's.

    The returned action carries `comm_clock` (a `_CommClock`: seconds in the
    lockstep, the gather and the reductions, reported as `davidson_comm`),
    `piece_clock` (a `_PieceClock`: seconds per piece of local work,
    ACTION_PIECES, reported as `davidson_action_<piece>`), and
    `screened_diagonal()`, the screened direct term's diagonal taken off d in
    the fit's gauge,

        (ii|W|aa) = [D^T (X_o o X_o)]^T W [D^T (X_v o X_v)]

    formed only when a Davidson preconditions with it
    (`_preconditioner_diagonal`): each rank adds its rows' tiles
    (DAVIDSON_DIAGONAL_TILE) and one reduction of (naux, n_occ + n_vir)
    completes them; the remaining contractions are replicated.
    """
    if lr_solver.spin_mode == 'unrestricted':
        raise NotImplementedError("The ISDF Davidson route is restricted-spin only.")
    comm = current_comm() if comm is None else comm
    sliced = isinstance(isdf_factors, SlicedFactors)
    if sliced:
        isdf_factors.require(comm)
        nmo, naux = isdf_factors.nmo, isdf_factors.naux
    else:
        X_mo, D = _unpack_factors(isdf_factors)[:2]
        nmo, naux = X_mo.shape[1], D.shape[1]
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    if nmo != len(lr_solver.eps):
        raise ValueError(f"isdf_factors X has {nmo} orbitals, "
                         f"lr_solver.eps has {len(lr_solver.eps)}.")
    if W_aux is not None and W_aux.shape[0] != naux:
        raise ValueError(f"W_aux is ({W_aux.shape[0]}, ...) but D has "
                         f"{naux} auxiliary functions; they are not from "
                         "the same fit (see isdf_bse_factors).")

    rank, nranks = ((comm.Get_rank(), comm.Get_size()) if comm is not None
                    else (0, 1))
    diag_d = bse_pair_diagonal(lr_solver.eps, nocc)
    if nranks > 1:
        diag_d = lockstep(diag_d, comm)
    comm_clock = _CommClock()
    piece_clock = _PieceClock()
    threads = openmp_threads()          # the element-wise passes' threads
    factor = KAPPA[spin]
    if sliced:
        # This rank's rows are all the action reads of X_v and D once Zt is
        # built; X_o is read whole by (Zt * P) X_o, and only by the exchange.
        npts, no = isdf_factors.npts, len(occ)
        r0, r1 = isdf_factors.rows
        X_o = (isdf_factors.gather_columns('X_mo', occ, 'X_o') if lBSE
               else None)
        X_o_mine = (X_o[r0:r1] if lBSE
                    else np.ascontiguousarray(isdf_factors.X_mo[:, occ]))
        X_v_mine = np.ascontiguousarray(isdf_factors.X_mo[:, virt])
        X_v = None
        D_mine = isdf_factors.D
    else:
        X_o = np.ascontiguousarray(X_mo[:, occ])
        X_v = np.ascontiguousarray(X_mo[:, virt])
        npts, no = X_o.shape
        r0, r1 = contiguous_block(npts, rank, nranks)  # this rank's rows
        X_o_mine, X_v_mine = X_o[r0:r1], X_v[r0:r1]    # contiguous row blocks
        D_mine = D[r0:r1]
    nmine = r1 - r0
    # Slabs of the reduced batch: the exchange terms (A's alone under tda),
    # then the Hartree one where a rank holds only part of it.
    i_hartree = (1 if tda else 2) if lBSE else 0  # the slot after the exchange
    n_reduced = i_hartree + (1 if factor and nranks > 1 else 0)
    # z X_v^T with the grid index first, which is the index the gather splits.
    # Its transpose is the (n_occ, M) operand every contraction below reads.
    zXv_rows = np.empty((npts, no)) if nranks > 1 else None

    # Zt = D W_aux D^T is `screened_interaction_imaginary_time`'s per-tau step,
    # wanted once and statically here. W_aux = None is TDHF, where the exchange
    # kernel is the bare Coulomb, i.e. Zt = Z = D D^T. Only the rows this rank
    # owns are ever formed, and under a rank count so is its inner product
    # (`_screened_rows`).
    if lBSE:
        if W_aux is not None:
            asym = np.abs(W_aux - W_aux.T).max()
            if asym > 1e-10 * np.abs(W_aux).max():
                raise ValueError(f"W_aux is not symmetric (max asymmetry "
                                 f"{asym:.2e}); the B block below relies on it.")
        # Sliced, D is whole only for this product and gone after it.
        Zt = _screened_rows(D_mine, isdf_factors.gather('D') if sliced else D,
                            W_aux, comm)                # (nmine, M)
        rows = max(1, min(max(nmine, 1),
                          int(tile_memory_gb * 1e9 / max(npts * 8, 1))))
        S = np.empty((rows, npts))            # one row block of Zt * P
        # X_o^T (Zt * P), accumulated; under a rank count in owner order, rank
        # s's (n_occ, its columns) block in rows [s0, s1) of an (M, n_occ) one
        T = np.empty((npts, no) if nranks > 1 else (no, npts))
        owners = ([contiguous_block(npts, s, nranks) for s in range(nranks)]
                  if nranks > 1 else [(0, npts)])
        T_owned = ([T[s0:s1].reshape(no, s1 - s0) for s0, s1 in owners]
                   if nranks > 1 else [T])        # serially one owner, T
        if nranks > 1:
            T_mine = np.empty((nmine, no))    # this rank's rows of the sum...
            T_cols = T_mine.reshape(no, nmine)  # ...its columns of T
        Tb = np.empty((no, npts))
        # (Zt * P) X_o, this rank's rows; B's alone, so not under tda
        U = None if tda else np.empty((nmine, no))

    def add_slabs(Az, Bz, ex):
        """Az and Bz with the reduced slabs in, n_occ rows on `threads`
        threads: plus the Hartree slab where it was reduced, then less the
        exchange, each element the sequence the whole-array updates make."""
        def slab_rows(i0, i1):
            A = Az[:, i0:i1]
            B = None if tda else Bz[:, i0:i1]
            if nranks > 1 and factor:
                np.add(A, ex[i_hartree, :, i0:i1], out=A)
                if not tda:
                    np.add(B, ex[i_hartree, :, i0:i1], out=B)
            if lBSE:
                np.subtract(A, ex[0, :, i0:i1], out=A)
                if not tda:
                    np.subtract(B, ex[1, :, i0:i1], out=B)

        row_map(slab_rows, no, threads)

    def apply_AB(z):
        if nranks > 1:
            z = _timed(comm_clock, 'lockstep', lockstep, z, comm, check=True)
        t = time.time()
        if nranks > 1:
            # Rank 0's batch, computed on as a C-ordered array: MKL's GEMM
            # kernels follow operand layout, and a view into the driver's
            # buffer moves water's roots 2e-13 Ha off the root-driven solve's.
            z = np.require(z, requirements='C')
        Az = np.empty(z.shape)
        Bz = None if tda else np.zeros(z.shape)
        row_map(lambda i0, i1: np.multiply(diag_d[i0:i1], z[:, i0:i1],
                                           out=Az[:, i0:i1]), no, threads)
        if n_reduced:
            # the terms that are partial over this rank's rows, reduced once
            # for the whole batch after the vector loop
            ex = np.zeros((n_reduced,) + z.shape)
        t = piece_clock.lap('batch', t)
        for n, zn in enumerate(z):
            if nranks > 1:
                # This rank's grid rows of z X_v^T, then everyone's: an output
                # partition, so no row is formed twice and the gathered array
                # is the same bits on every rank.
                np.matmul(X_v_mine, zn.T, out=zXv_rows[r0:r1])
                piece_clock.lap('gemm', t)
                _timed(comm_clock, 'gather', allgather_blocks, zXv_rows, comm)
                t = time.time()
                zXv = zXv_rows.T                          # (n_occ, M)
            else:
                zXv = zn @ X_v.T                          # (n_occ, M)
                t = piece_clock.lap('gemm', t)
            if factor:
                # Hartree needs only P's diagonal, and the bare Z = D D^T is
                # never formed either: two (M, naux) products instead of an
                # (M, M) one. A triplet has kappa = 0 and skips the whole term.
                # p is a per-grid-point quantity, so a rank builds its own rows
                # of it and only p D, naux long, crosses between them.
                p = (np.einsum('kj,kj->k', X_o_mine, zXv_rows[r0:r1],
                               optimize=True) if nranks > 1 else
                     np.einsum('kj,jk->k', X_o, zXv, optimize=True))
                pD = p @ D_mine
                if nranks > 1:
                    piece_clock.lap('hartree', t)
                    _timed(comm_clock, 'reduce_pairs', reduce_sum, pD, comm)
                    t = time.time()
                u = D_mine @ pD
                hartree = factor * (X_o_mine.T @ (u[:, None] * X_v_mine))
                if nranks > 1:
                    ex[i_hartree, n] = hartree
                else:
                    Az[n] += hartree
                    if not tda:
                        Bz[n] = hartree
                t = piece_clock.lap('hartree', t)
            if not lBSE:
                continue
            if r1 == r0:
                T.fill(0.0)                   # no row, no tile: a zero partial
                t = piece_clock.lap('owner', t)
            for p0 in range(r0, r1, rows):
                p1 = min(p0 + rows, r1)
                blk = S[:p1 - p0]
                np.matmul(X_o[p0:p1], zXv, out=blk)       # P's rows
                t = piece_clock.lap('gemm', t)
                _hadamard_rows(blk, Zt[p0 - r0:p1 - r0], threads)
                t = piece_clock.lap('hadamard', t)
                np.matmul(X_o[p0:p1].T, blk, out=Tb)
                t = piece_clock.lap('gemm', t)
                _owner_write(T_owned, owners, Tb, p0 == r0, threads)
                t = piece_clock.lap('owner', t)
                if not tda:
                    np.matmul(blk, X_o, out=U[p0 - r0:p1 - r0])
                    t = piece_clock.lap('gemm', t)
            # T is a partial over this rank's rows and whole in the column
            # index, and this rank contracts only its own columns of the sum:
            # those are all it receives.
            if nranks > 1:
                _timed(comm_clock, 'reduce_grid', reduce_scatter_rows, T, comm,
                       out=T_mine)
                t = time.time()
                ex[0, n] = T_cols @ X_v_mine
            else:
                ex[0, n] = T[:, r0:r1] @ X_v_mine
            # The B block wants Zt * P^T. Formed directly, that Hadamard product
            # reads Zt against a transposed operand and costs as much as the
            # rest of the step; with Zt symmetric (checked above) it is
            # (Zt * P)^T, and (Zt * P)^T X_o is the U already built. Its
            # contraction with X_v runs over the row index, so this rank's rows
            # of U meet this rank's rows of X_v and the sum over ranks is exact.
            if not tda:
                ex[1, n] = U.T @ X_v_mine
                t = piece_clock.lap('gemm', t)
        if n_reduced:
            if nranks > 1:
                _timed(comm_clock, 'reduce_pairs', reduce_sum, ex, comm)
                t = time.time()
            add_slabs(Az, Bz, ex)
            piece_clock.lap('batch', t)
        return Az, Bz

    def screened_diagonal():
        """d - (ii|W|aa) on every pair, (n_occ, n_vir), the one-rank bits on
        every rank; d alone for RPA. The fitted squares are a sum over the
        grid's fixed tiles (`self_pair_densities` per tile), each tile's
        addend formed whole by one rank and added in tile order
        (`ordered_chain_sum`); the kernel's contraction is replicated. The
        addends are (naux, n_occ + n_vir) each, and a rank moves in up to a
        tile's rows of D, X_o and X_v."""
        if not lBSE:
            return diag_d
        # whole fixed tiles, each on the rank holding its first row (the rows
        # past the block moved in once), added in tile order on the chain
        tiles = GridTiles(npts, DAVIDSON_DIAGONAL_TILE, comm)
        parts = [tiles.tile_rows(np.ascontiguousarray(a))
                 for a in (D_mine, X_o_mine, X_v_mine)]
        terms = {t: self_pair_densities(parts[0][t], parts[1][t],
                                        parts[2][t], [(0, len(parts[0][t]))])
                 for t in tiles.mine}
        parts = None
        width = no + X_v_mine.shape[1]
        B = ordered_chain_sum(terms, tiles.owner, comm,
                              np.zeros((D_mine.shape[1], width)))
        dressed = diag_d - screened_direct_diagonal(B[:, :no], B[:, no:],
                                                    W_aux)
        if nranks > 1:
            dressed = lockstep(dressed, comm, check=True)
        return dressed

    apply_AB.comm_clock = comm_clock
    apply_AB.piece_clock = piece_clock
    apply_AB.screened_diagonal = screened_diagonal
    # What the action allocates and keeps: this rank's rows of Zt, its tiles
    # and grid buffers, and the factor slices it copied or gathered.
    held = [a for a in (Zt, S, T, Tb, U) if a is not None] if lBSE else []
    if nranks > 1:
        held += [zXv_rows] + ([T_mine] if lBSE else [])
    held += ([X_v_mine] + ([X_o] if lBSE else [X_o_mine]) if sliced
             else [X_o, X_v])
    apply_AB.held_bytes = int(sum(a.nbytes for a in held))
    # What one application allocates per trial vector: A z and B z (A z
    # alone under tda), the reduced slabs, and under a rank count the
    # C-ordered copy of the batch.
    nv = len(virt)
    apply_AB.vector_bytes = int(8 * no * nv * ((1 if tda else 2) + n_reduced
                                               + (nranks > 1)))
    # And once, for the vector in hand: two (n_occ, n_vir) products, the
    # Hartree term's (rows, n_vir) one, and serially z X_v^T with the
    # Hartree contraction's operand of its (n_occ, M) size.
    apply_AB.scratch_bytes = int(8 * (2 * no * nv
                                      + (nmine * nv if factor else 0)
                                      + (0 if nranks > 1 else 2 * no * npts)))
    return apply_AB, diag_d
