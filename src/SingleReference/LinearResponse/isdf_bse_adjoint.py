"""The Hellmann-Feynman adjoint of an ISDF BSE root, closed over the grid.

A root's force is Hellmann-Feynman on its Casida vectors,

    dOmega = sum C^A dA + sum C^B dB,
    C^A = x_m x_n^T + y_m y_n^T,   C^B = x_m y_n^T + y_m x_n^T,

(m = n for dOmega_n, m != n for the interstate element), and this is the
reverse of the block action that solved for them
(`davidson.isdf_block_action`). Substituting B[P,p,q] = sum_k D[k,P] X[k,p]
X[k,q] into every kernel term closes each pair sum into a grid object,

    P_T = X_o T X_v^T  (M x M),        T in {x_m, x_n, y_m, y_n}
    Zt  = D W D^T      (M x M),        rho_u = diag(P_u),  u = x + y

so the energy the adjoint differentiates is

    (i)   sum_ia (eps_a - eps_i) (x_m x_n + y_m y_n)_ia
    (ii)  kappa (D^T rho_um) . (D^T rho_un)
    (iii) -sum_kk' Zt (P_xm o P_xn + P_ym o P_yn)
    (iv)  -sum_kk' Zt (P_xm^T o P_yn + P_ym^T o P_xn)

and every screened term shares ONE Hadamard weight H and one Zt. With
S = -(H + H^T) and b = D^T rho_um, bn = D^T rho_un, the adjoints are

    W_bar   = 1/2 D^T S D
    D_bar   = S D W + kappa (rho_um bn^T + rho_un b^T)
    X_o_bar = sum_T Pbar_T (X_v T^T) + kappa [(D bn) o (X_v u_m^T) + ...]
    X_v_bar = sum_T (Pbar_T^T X_o) T  + kappa [(D bn) o (X_o u_m) + ...]
    Pbar_xm = -Zt o (P_xn + P_yn^T)   and its three permutations,

each scaled by omega_bar. Only the symmetric part of W_bar is formed: W is
[1 - chi0(0)]^-1 with chi0 symmetric, the block action already relies on
Zt = Zt^T for its B block, and the chi0 adjoint symmetrizes what reaches it,
so one S D product serves both W_bar and D_bar.

NO THREE-INDEX BLOCK EXISTS, and no (M x M) object beyond one (tile, tile)
block: the grid index is cut in FIXED tiles of BSE_ADJOINT_TILE_ROWS points
on both sides of every M^2 product, each block of P_T, P_T^T, Zt and S one
(tile, tile) GEMM, for M^2 (4 naux + 32 nocc) + 6 M naux^2 + 8 M nocc nvir
flops instead of the explicit route's (`gradients.bse_isdf.bse_backward`)
about 8 M naux nocc nvir and its ten (naux, nocc, nvir) blocks.

DISTRIBUTED OVER THE CURRENT COMMUNICATOR, THE SAME BITS AT EVERY RANK COUNT.
The rows of X_bar and D_bar are output partitions: row k is a sum over the
column index alone, so the tile holding k is computed whole by one rank, the
one whose `contiguous_block` of the factors holds the tile's first row, from
its own rows (the few past its block moved in once, `exchange_blocks`). The
column index is STREAMED: every column tile's D, X_o and X_v T^T (the only
form in which the column side reads X_v) are broadcast once per pass by
their owner, and each rank sums its tiles' products over them in column-tile
order. The two sums over the grid row index move data instead of reducing:
b and bn are gathered as per-tile partials and added in tile order on every
rank, and W_bar is built in a second pass by fixed slabs of
BSE_ADJOINT_AUX_ROWS auxiliary rows, slab s on rank s % nranks, from each
grid tile's D and S D broadcast by its owner in tile order, the slabs then
gathered verbatim. Every GEMM has a shape fixed by the tiles, so no result
depends on how many ranks share the grid: one rank runs the same tiles on
whole arrays. This is another blocking of the sums than an M-long GEMM, so
not the bits of one. The rows go back to their `contiguous_block` owners,
and `isdf_bse_backward` gathers them whole once (`adjoints_at_the_boundary`)
for the consumers that read X_bar and D_bar whole.

AT THE CHLOROPHYLLIDE HEXAMER / cc-pVTZ (M 117762, nmo 10980, nocc 972,
nvir 10008, naux 28236; 461 tiles, rank 0 computing 58 of them over 14721
rows at 8 ranks and 29 over 7361 at 16), per rank, GB, replicated (every
rank the whole kernel on gathered factors) against distributed:

                                          replicated        distributed
                                          8 rk   16 rk      8 rk   16 rk
    X_mo, D gathered whole               36.94   36.94       0      0
    X_o, X_v rows (column copies)            -       -       1.29   0.65
    (D W), (S D) of the owned tiles          -       -       6.71   3.35
    X_v T^T and the Pbar products         3.66    3.66       1.04   0.52
    b, bn tile partials (before pass 1)      -       -       0.21   0.21
    streamed tiles, blocks, halo          3.01    3.01       0.47   0.47
    W_aux, W, W_bar (two at once here)   19.13   19.13      12.76  12.76
    X_bar, D_bar (rows here)             36.94   36.94       4.62   2.31
    peak inside this adjoint              99.7    99.7       23.3   18.1
    X_bar, D_bar whole at the boundary       -       -      36.94  36.94
    peak with the boundary gather         99.7    99.7       54.3   52.0

Whole on every rank: the (naux, naux) arrays -- W_aux, the input; W, its
symmetric part, which every tile's D W and S D W read; W_bar, filled by
slabs and gathered because chi0's adjoint reads it whole -- and at the
boundary X_bar and D_bar, until their consumers read rows. Received per
rank: 27.2 / 29.2 GB in the column pass (D, X_o, X_v T^T), 46.5 / 49.8 in
the W_bar pass (D, S D). Flops per rank 3.2e14 / 1.6e14 against 2.6e15.
"""
import numpy as np

from src.Base.constants import (BSE_ADJOINT_AUX_ROWS, BSE_ADJOINT_TILE_ROWS,
                                KAPPA)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (allgather_ranges, allgather_rows,
                                     broadcast_rows, contiguous_block,
                                     current_comm, exchange_blocks, lockstep)
from src.SingleReference.base import get_occ_virt_indices

#: S = -(H + H^T) as products of blocks of P_T ('P') and of P_T^T ('Q'),
#: Casida blocks 0 x_m, 1 x_n, 2 y_m, 3 y_n; Tamm-Dancoff keeps x alone.
S_TERMS = {True: (('P', 0, 'P', 1), ('Q', 0, 'Q', 1)),
           False: (('P', 0, 'P', 1), ('P', 2, 'P', 3), ('Q', 0, 'P', 3),
                   ('Q', 2, 'P', 1), ('Q', 0, 'Q', 1), ('Q', 2, 'Q', 3),
                   ('P', 0, 'Q', 3), ('P', 2, 'Q', 1))}
#: Pbar_T = -Zt o (sum of these blocks) per T, contracted into X_o_bar ...
PBAR = {True: ((('P', 1),), (('P', 0),)),
        False: ((('P', 1), ('Q', 3)), (('P', 0), ('Q', 2)),
                (('P', 3), ('Q', 1)), (('P', 2), ('Q', 0)))}
#: ... and its transpose's blocks, contracted into X_v_bar
PBAR_T = {True: ((('Q', 1),), (('Q', 0),)),
          False: ((('Q', 1), ('P', 3)), (('Q', 0), ('P', 2)),
                  (('Q', 3), ('P', 1)), (('Q', 2), ('P', 0)))}


class GridTiles:
    """Fixed tiles of the grid index and the rank that computes each: the one
    whose `contiguous_block` holds the tile's first row.

    bounds: [start, stop) of every tile; owner: its rank; mine: this rank's
    tiles, consecutive; rows: this rank's block of the grid.
    """

    def __init__(self, npts, edge, comm):
        self.comm = comm
        self.size = 1 if comm is None else comm.Get_size()
        self.rank = 0 if comm is None else comm.Get_rank()
        self.npts = int(npts)
        self.bounds = [(t0, min(t0 + int(edge), self.npts))
                       for t0 in range(0, self.npts, int(edge))]
        self.blocks = [contiguous_block(self.npts, r, self.size)
                       for r in range(self.size)]
        self.owner = [next(r for r, (b0, b1) in enumerate(self.blocks)
                           if b0 <= t0 < b1) for t0, _ in self.bounds]
        self.mine = [t for t, r in enumerate(self.owner) if r == self.rank]
        self.rows = self.blocks[self.rank]

    def owned(self, rank):
        """[start, stop) of every tile `rank` computes, in order."""
        return [self.bounds[t] for t, r in enumerate(self.owner) if r == rank]

    def tile_rows(self, rows):
        """{tile: its rows} for this rank's tiles out of this rank's block
        `rows`: a view where the tile lies inside the block, else a new array
        completed with the rows other ranks hold."""
        r0, r1 = self.rows
        out = {}
        for t in self.mine:
            t0, t1 = self.bounds[t]
            if t1 <= r1:
                out[t] = rows[t0 - r0:t1 - r0]
            else:
                out[t] = np.empty((t1 - t0,) + rows.shape[1:])
                out[t][:r1 - t0] = rows[t0 - r0:]
        self._move([rows], [[b] for b in self.blocks],
                   [out[t] for t in self.mine],
                   [self.owned(r) for r in range(self.size)], rows.shape[1:])
        return out

    def block_rows(self, tiles, block):
        """Fill this rank's `block` rows from the tiles every rank computed:
        a tile inside its owner's block is already a view of it, the rest of
        a tile's rows travel to the ranks whose blocks hold them."""
        r0, r1 = self.rows
        for t in self.mine:
            t0, t1 = self.bounds[t]
            if t1 > r1:
                block[t0 - r0:] = tiles[t][:r1 - t0]
        self._move([tiles[t] for t in self.mine],
                   [self.owned(r) for r in range(self.size)], [block],
                   [[b] for b in self.blocks], block.shape[1:])

    def _move(self, src, have, dst, want, tail):
        """The rows of `dst` held in another rank's `src`, verbatim, in one
        `exchange_blocks`: src[i] holds rows have[rank][i], dst[j] wants
        want[rank][j]; the lists are the same on every rank."""
        if self.size == 1:
            return
        rank = self.rank
        send, shapes = [], []
        for peer in range(self.size):
            pieces = [] if peer == rank else [
                src[i][lo - have[rank][i][0]:hi - have[rank][i][0]]
                for i, _, lo, hi in _overlaps(have[rank], want[peer])]
            send.append(np.concatenate(pieces) if pieces
                        else np.empty((0,) + tuple(tail)))
            n = 0 if peer == rank else sum(
                hi - lo for _, _, lo, hi in _overlaps(have[peer], want[rank]))
            shapes.append((n,) + tuple(tail))
        got = exchange_blocks(send, shapes, self.comm)
        for peer in range(self.size):
            if peer == rank:
                continue
            at = 0
            for _, j, lo, hi in _overlaps(have[peer], want[rank]):
                w0 = want[rank][j][0]
                dst[j][lo - w0:hi - w0] = got[peer][at:at + hi - lo]
                at += hi - lo


def isdf_bse_backward(n, X_mo, D, eps_qp, W_aux, nocc, Xn, Yn,
                      spin='singlet', bse_tda=False, omega_bar=1.0, bra=None,
                      tile_rows=BSE_ADJOINT_TILE_ROWS,
                      aux_rows=BSE_ADJOINT_AUX_ROWS, stats=None, comm=None):
    """(eps_qp_bar, X_bar, D_bar, W_aux_bar) of omega_bar * Omega_n, or of
    the one-sided element <bra| dH |n>, from the grid, every array whole and
    the same bits on every rank.

    X_mo, D: whole arrays, or one `SlicedFactors` for both over the current
    communicator. Xn, Yn: (n_ov, nroots) Casida vectors; `bse_tda` drops the
    swap term and the y blocks. W_aux_bar is the symmetric part of the
    adjoint. stats: a dict filled with the held-bytes ledger ('held'), the
    tile broadcasts per pass ('streams'), the tiles received from other
    ranks ('received') and the row exchanges ('exchanges').
    """
    comm = current_comm() if comm is None else comm
    return adjoints_at_the_boundary(
        isdf_bse_backward_rows(n, X_mo, D, eps_qp, W_aux, nocc, Xn, Yn,
                               spin=spin, bse_tda=bse_tda, omega_bar=omega_bar,
                               bra=bra, tile_rows=tile_rows,
                               aux_rows=aux_rows, stats=stats, comm=comm),
        _grid_size(X_mo), stats=stats, comm=comm)


def isdf_interstate_backward(m, n, X_mo, D, eps_qp, W_aux, nocc, Xn, Yn,
                             spin='singlet', bse_tda=False, omega_bar=1.0,
                             tile_rows=BSE_ADJOINT_TILE_ROWS,
                             aux_rows=BSE_ADJOINT_AUX_ROWS, stats=None,
                             comm=None):
    """Adjoints of the symmetrized interstate element <m| dH |n>, m != n.

    One pass, not an average over both orderings: with W symmetric the grid
    energy is symmetric under m <-> n term by term -- the bare product is,
    and S = -(H + H^T) only permutes its addends -- so the one-sided element
    IS the symmetrized one.
    """
    if m == n:
        raise ValueError('an interstate element needs two different roots; '
                         'use isdf_bse_backward for dOmega_n')
    return isdf_bse_backward(n, X_mo, D, eps_qp, W_aux, nocc, Xn, Yn,
                             spin=spin, bse_tda=bse_tda, omega_bar=omega_bar,
                             bra=m, tile_rows=tile_rows, aux_rows=aux_rows,
                             stats=stats, comm=comm)


def adjoints_at_the_boundary(adjoints, npts, stats=None, comm=None):
    """(eps_qp_bar, X_bar, D_bar, W_aux_bar) with X_bar and D_bar whole on
    every rank, from the rows `isdf_bse_backward_rows` hands back: the one
    gather of the adjoint, rows moved verbatim (`allgather_rows`), for the
    consumers that read it whole."""
    comm = current_comm() if comm is None else comm
    eps_qp_bar, x_rows, d_rows, w_bar = adjoints
    x_bar = _gathered(x_rows, npts, comm)
    d_bar = _gathered(d_rows, npts, comm)
    if stats is not None:
        stats['held']['X_bar_whole'] = int(x_bar.nbytes)
        stats['held']['D_bar_whole'] = int(d_bar.nbytes)
    return eps_qp_bar, x_bar, d_bar, w_bar


def isdf_bse_backward_rows(n, X_mo, D, eps_qp, W_aux, nocc, Xn, Yn,
                           spin='singlet', bse_tda=False, omega_bar=1.0,
                           bra=None, tile_rows=BSE_ADJOINT_TILE_ROWS,
                           aux_rows=BSE_ADJOINT_AUX_ROWS, stats=None,
                           comm=None):
    """(eps_qp_bar, X_bar rows, D_bar rows, W_aux_bar): this rank's
    `contiguous_block` rows of X_bar and D_bar (every row on one rank),
    eps_qp_bar and W_aux_bar whole; see `isdf_bse_backward`."""
    comm = current_comm() if comm is None else comm
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    held, streams, received = {}, {'columns': 0, 'w_bar': 0}, {}

    def hold(name, *arrays):
        held[name] = max(held.get(name, 0),
                         sum(int(a.nbytes) for a in arrays))

    eps_qp = np.asarray(eps_qp, float)
    if size > 1:
        eps_qp, W_aux, Xn, Yn = lockstep((eps_qp, W_aux, Xn, Yn), comm,
                                         check=True)
    npts = _grid_size(X_mo)
    x_rows, d_rows = _own_rows(X_mo, 'X_mo', comm), _own_rows(D, 'D', comm)
    occ, virt = get_occ_virt_indices(eps_qp, nocc)
    no, nv = len(occ), len(virt)
    nmo, naux = x_rows.shape[1], d_rows.shape[1]
    # occ and virt are contiguous ranges
    so, sv = slice(occ[0], occ[-1] + 1), slice(virt[0], virt[-1] + 1)
    grid = GridTiles(npts, tile_rows, comm)
    r0, r1 = grid.rows
    D_rows = np.ascontiguousarray(d_rows)
    Xo_rows = np.ascontiguousarray(x_rows[:, so])
    Xv_rows = np.ascontiguousarray(x_rows[:, sv])
    hold('X_o_rows', Xo_rows)
    hold('X_v_rows', Xv_rows)
    m = n if bra is None else bra
    Xm, Xk = Xn[:, m].reshape(no, nv), Xn[:, n].reshape(no, nv)
    Ym, Yk = Yn[:, m].reshape(no, nv), Yn[:, n].reshape(no, nv)
    w = omega_bar
    W = 0.5 * (W_aux + W_aux.T)
    hold('W', W)

    eps_qp_bar = np.zeros(len(eps_qp))
    # (i) quasiparticle diagonal: the diagonal of C^A alone
    diag = w * (Xm * Xk + Ym * Yk)
    eps_qp_bar[sv] += diag.sum(axis=0)
    eps_qp_bar[so] -= diag.sum(axis=1)

    # this rank's tiles, with the rows past its block moved in once
    D_t, Xo_t, Xv_t = (grid.tile_rows(a) for a in (D_rows, Xo_rows, Xv_rows))
    hold('halo', *[a for tiles in (D_t, Xo_t, Xv_t) for a in tiles.values()
                   if a.base is None])
    tiles = {t: grid.bounds[t][1] - grid.bounds[t][0] for t in grid.mine}

    # (ii) bare kernel through the transition densities: C^A + C^B = u_m u_n^T
    c = KAPPA[spin] * w
    um, un = Xm + Ym, Xk + Yk
    rho = {}
    if c:
        # b and bn are sums over the grid rows: per-tile partials, gathered
        # and added in tile order, the same order at every rank count
        parts = np.zeros((len(grid.bounds), 2, naux))
        hold('b_partials', parts)
        for t in grid.mine:
            rho[t] = [np.einsum('ka,ka->k', np.matmul(Xo_t[t], u), Xv_t[t])
                      for u in (um, un)]
            for i in range(2):
                np.matmul(D_t[t].T, rho[t][i], out=parts[t, i])
        hold('rho', *[r for pair in rho.values() for r in pair])
        allgather_ranges(parts, [_index_ranges(grid, r)
                                 for r in range(size)], comm)
        b, bn = parts[0, 0].copy(), parts[0, 1].copy()
        for t in range(1, len(grid.bounds)):
            b += parts[t, 0]
            bn += parts[t, 1]
        del parts

    # (iii) + (iv): per tile t, X_v[t] T^T for every Casida block T, the
    # operand of P_T^T's rows (row role) and of P_T's columns (column role)
    Ts = tuple(np.ascontiguousarray(T) for T in
               ((Xm, Xk) if bse_tda else (Xm, Xk, Ym, Yk)))
    nT = len(Ts)
    Tcat = np.concatenate(Ts)                         # (nT n_occ, n_vir)
    s_terms, pbar, pbar_t = S_TERMS[bse_tda], PBAR[bse_tda], PBAR_T[bse_tda]
    DW = {t: np.matmul(D_t[t], W) for t in grid.mine}   # rows of D W
    AT = {t: np.matmul(Xv_t[t], Tcat.T) for t in grid.mine}
    SD = {t: np.zeros((r, naux)) for t, r in tiles.items()}   # rows of S D
    Go = {t: np.zeros((r, no)) for t, r in tiles.items()}  # Pbar_T X_v T^T
    Gv = {t: np.zeros((nT, r, no)) for t, r in tiles.items()}  # Pbar_T^T X_o
    hold('DW', *DW.values())
    hold('AT', *AT.values())
    hold('SD', *SD.values())
    hold('accumulators', *Go.values(), *Gv.values())
    edge = min(int(tile_rows), npts)
    # (tile, tile) blocks, C-ordered at every shape: P_T and P_T^T per T,
    # Zt, S, one product and one Pbar
    flat = np.empty((2 * nT + 4, edge * edge))
    prod = np.empty(edge * max(naux, no, nv))
    hold('blocks', flat)
    hold('block_products', prod)

    def block(i, r, q):
        return flat[i, :r * q].reshape(r, q)

    def product(r, ncol):
        return prod[:r * ncol].reshape(r, ncol)

    # pass 1: the column tiles, streamed once each in tile order
    for ci, (c0, c1) in enumerate(grid.bounds):
        q, owner = c1 - c0, grid.owner[ci]
        if owner == rank:
            Dc, Xoc, ATc = D_t[ci], Xo_t[ci], AT[ci]
        else:
            Dc, Xoc, ATc = (np.empty((q, naux)), np.empty((q, no)),
                            np.empty((q, nT * no)))
            received['columns'] = received.get('columns', 0) + 1
            hold('column_tile', Dc, Xoc, ATc)
        for a in (Dc, Xoc, ATc):
            broadcast_rows(a, owner, comm)
        streams['columns'] += 1
        for t in grid.mine:
            r = tiles[t]
            P = [block(i, r, q) for i in range(nT)]
            Q = [block(nT + i, r, q) for i in range(nT)]
            zt, s, tmp, pair = (block(2 * nT + i, r, q) for i in range(4))
            np.matmul(DW[t], Dc.T, out=zt)                  # Zt block
            for i in range(nT):
                cols = slice(i * no, (i + 1) * no)
                np.matmul(Xo_t[t], ATc[:, cols].T, out=P[i])    # P_T block
                np.matmul(AT[t][:, cols], Xoc.T, out=Q[i])      # P_T^T block

            def part(kind, i):
                return P[i] if kind == 'P' else Q[i]

            first, *rest = s_terms
            np.multiply(part(*first[:2]), part(*first[2:]), out=s)
            for term in rest:
                np.multiply(part(*term[:2]), part(*term[2:]), out=tmp)
                s += tmp
            s *= -w
            sd = product(r, naux)
            np.matmul(s, Dc, out=sd)
            SD[t] += sd
            for i in range(nT):
                for sources, operand, acc in (
                        (pbar[i], ATc[:, i * no:(i + 1) * no], Go[t]),
                        (pbar_t[i], Xoc, Gv[t][i])):
                    np.copyto(pair, part(*sources[0]))
                    for source in sources[1:]:
                        pair += part(*source)
                    pair *= zt                          # a block of Pbar_T
                    g = product(r, no)
                    np.matmul(pair, operand, out=g)
                    acc += g
    del DW

    # this rank's rows of the adjoints; a tile inside the block writes them
    X_bar = np.zeros((r1 - r0, nmo))
    D_bar = np.zeros((r1 - r0, naux))
    Xb_t, Db_t = {}, {}
    for t, r in tiles.items():
        t0, t1 = grid.bounds[t]
        inside = t1 <= r1
        Xb_t[t] = X_bar[t0 - r0:t1 - r0] if inside else np.zeros((r, nmo))
        Db_t[t] = D_bar[t0 - r0:t1 - r0] if inside else np.empty((r, naux))
    hold('X_bar_rows', X_bar)
    hold('D_bar_rows', D_bar)
    hold('hand_back', *[a for a in (*Xb_t.values(), *Db_t.values())
                        if a.base is None])
    for t, r in tiles.items():
        np.matmul(SD[t], W, out=Db_t[t])
        xo_bar, xv_bar = Xb_t[t][:, so], Xb_t[t][:, sv]
        np.multiply(Go[t], -w, out=xo_bar)
        for i, T in enumerate(Ts):
            g = product(r, nv)
            np.matmul(Gv[t][i], T, out=g)
            g *= w
            xv_bar -= g
        if c:
            # the bare kernel's adjoint on the same rows
            rm, rn = rho[t]
            Db_t[t] += c * (rm[:, None] * bn[None, :]
                            + rn[:, None] * b[None, :])
            gm, gn = c * np.matmul(D_t[t], bn), c * np.matmul(D_t[t], b)
            xo_bar += (gm[:, None] * np.matmul(Xv_t[t], um.T)
                       + gn[:, None] * np.matmul(Xv_t[t], un.T))
            xv_bar += (gm[:, None] * np.matmul(Xo_t[t], um)
                       + gn[:, None] * np.matmul(Xo_t[t], un))
    del W, AT, Go, Gv, rho, Xo_t, Xv_t, Xo_rows, Xv_rows
    grid.block_rows(Xb_t, X_bar)
    grid.block_rows(Db_t, D_bar)
    del Xb_t, Db_t

    # pass 2: W_bar by fixed auxiliary slabs, each grid tile's D and S D
    # broadcast once by its owner in tile order
    slabs = [(p0, min(p0 + int(aux_rows), naux))
             for p0 in range(0, naux, int(aux_rows))]
    mine = [slab for s, slab in enumerate(slabs) if s % size == rank]
    W_bar = np.zeros((naux, naux))
    wt = np.empty(min(int(aux_rows), naux) * naux)
    hold('W_bar', W_bar)
    hold('w_bar_slab', wt)
    for t, (t0, t1) in enumerate(grid.bounds):
        owner = grid.owner[t]
        if owner == rank:
            Dt, SDt = D_t[t], SD[t]
        else:
            Dt, SDt = np.empty((t1 - t0, naux)), np.empty((t1 - t0, naux))
            received['w_bar'] = received.get('w_bar', 0) + 1
            hold('w_bar_tile', Dt, SDt)
        broadcast_rows(Dt, owner, comm)
        broadcast_rows(SDt, owner, comm)
        streams['w_bar'] += 1
        for p0, p1 in mine:
            g = wt[:(p1 - p0) * naux].reshape(p1 - p0, naux)
            np.matmul(Dt[:, p0:p1].T, SDt, out=g)
            W_bar[p0:p1] += g
    allgather_ranges(W_bar, [[slab for s, slab in enumerate(slabs)
                              if s % size == r] for r in range(size)], comm)
    W_bar *= 0.5
    if stats is not None:
        stats.update(held=held, streams=streams, received=received,
                     exchanges={'halo': 3 if size > 1 else 0,
                                'hand_back': 2 if size > 1 else 0},
                     tiles=list(grid.mine), rows=(r0, r1))
    return eps_qp_bar, X_bar, D_bar, W_bar


def _grid_size(factor):
    """The number of grid points of whole or sliced factors."""
    return factor.npts if isinstance(factor, SlicedFactors) else len(factor)


def _own_rows(factor, name, comm):
    """This rank's `contiguous_block` rows of the factor `name`: a
    `SlicedFactors`' own, refused over any other rank layout, or cut from a
    whole array."""
    if isinstance(factor, SlicedFactors):
        return getattr(factor.require(comm), name)
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    r0, r1 = contiguous_block(len(factor), rank, size)
    return factor[r0:r1]


def _gathered(rows, npts, comm):
    """The whole array on every rank from each rank's `contiguous_block`
    rows; the rows themselves on one rank."""
    if comm is None or comm.Get_size() == 1:
        return rows
    r0, r1 = contiguous_block(npts, comm.Get_rank(), comm.Get_size())
    whole = np.empty((npts,) + rows.shape[1:])
    whole[r0:r1] = rows
    return allgather_rows(whole, comm)


def _index_ranges(grid, rank):
    """The tile indices `rank` computes as [start, stop) ranges."""
    own = [t for t, r in enumerate(grid.owner) if r == rank]
    return [(own[0], own[-1] + 1)] if own else []


def _overlaps(have, want):
    """(i, j, lo, hi) for every pair of ranges have[i], want[j] sharing rows
    [lo, hi), in the order of `have`, then of `want`."""
    for i, (h0, h1) in enumerate(have):
        for j, (w0, w1) in enumerate(want):
            lo, hi = max(h0, w0), min(h1, w1)
            if hi > lo:
                yield i, j, lo, hi
