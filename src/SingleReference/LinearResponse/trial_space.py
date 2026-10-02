"""The Casida Davidson's trial space cut by pair rows over ranks: pyscf's
`real_eig`, owned, with every pair-space-long array partitioned.

Replicated, every rank holds the four trial-space holders V, W,
U1 = A V + B W and U2 = A W + B V whole -- 32 bytes per pair per trial pair --
and repeats the whole Rayleigh-Ritz on them: projections, Ritz vectors,
residuals, preconditioner and Gram-Schmidt, about 32 passes over an (N, m)
holder a cycle, which no rank count divides. What must be the same on every
rank is only the m x m projected eigenproblem and the k x k Gram eigh's of the
Gram-Schmidt, functions of small matrices.

THE ROWS. The pair index is cut into fixed tiles of DAVIDSON_PAIR_TILE rows
(`pair_tiles`), and a rank owns the contiguous run of tiles
`contiguous_block(n_tiles, rank, size)` gives it (`PairRows`). Every N-long
array of the loop is held as one block per owned tile. A block's arithmetic is
the same call on the same shape at every rank count, so the row products --
Ritz vectors, residuals, the preconditioner step, the Gram-Schmidt updates --
are an OUTPUT PARTITION: the same bits in every row at any rank count for the
same small coefficients.

THE SUMS. A sum over the pair index -- the projected blocks a, b, sigma and
pi, the residual norms, the correction norms, the Gram-Schmidt overlaps -- is
formed per tile, the tiles' addends added in tile order onto the first, and
the ranks' partials joined by ONE `reduce_sum` of the small matrices. The
joins re-associate with the rank count, and that is the one place the rank
count enters the iteration's arithmetic.

WHAT CROSSES, per cycle: the new batch, gathered whole for the block action
(`allgather_ranges`, k x 2N doubles); one reduction of the four (k, m1)
projected blocks, one of the residual norms, one or two of the correction
norms, and two of the Gram-Schmidt overlaps, (2, m1, k') and (2, k', k'); and
a checked `lockstep` of the subspace solution and of the Gram-Schmidt
coefficients, whose digests prove every rank builds its rows from rank 0's.
The converged X and Y are gathered once at the end. Every decision -- which
roots converged, which corrections are sized, how many directions survive the
linear-dependence tests, when the space restarts -- is read off reduced or
lockstepped small matrices, so it is the same on every rank and the
collectives pair.

RESTARTS take the nroots Ritz vectors as real_eig does, and their products as
the same combinations of the stored products,
A (V x + W y) + B (W x + V y) = U1 x + U2 y, instead of applying the action
to them again.

ONE TILE. On one rank with one tile every expression is the one real_eig
evaluates: a tile's addend is real_eig's own sum, a sum over one tile is that
addend, and the square root of a squared norm is the norm again
(sqrt(fl(x^2)) = |x| in binary floating point), so a solve without restarts
returns real_eig's bits.
"""
import time

import numpy as np
from pyscf.lib import logger
from pyscf.lib.linalg_helper import _sort_elast
from pyscf.tdscf._lr_eig import (TDDFT_subspace_eigen_solver, _asym_dot,
                                 _sym_dot)

from src.Base.constants import DAVIDSON_PAIR_TILE
from src.Base.utils.mpi_grid import (allgather_ranges, contiguous_block,
                                     lockstep, reduce_sum)


#: The owned loop's own time outside the block action, each piece timed
#: apart: the per-tile arithmetic, the replicated small solves, the batch and
#: result gathers, the reductions of the small matrices, and the checked
#: locksteps of the subspace solution and the Gram-Schmidt coefficients.
SUBSPACE_PIECES = ('rows', 'small', 'gather', 'reduce', 'lockstep')


class PairRows:
    """This rank's tiles of the pair index, and the collectives that join the
    ranks' rows: the one place the distributed trial space communicates."""

    def __init__(self, n_pair, comm=None, tile=None):
        self.comm = comm
        self.rank, self.size = ((comm.Get_rank(), comm.Get_size())
                                if comm is not None else (0, 1))
        self.n_pair = int(n_pair)
        self.all_tiles = pair_tiles(self.n_pair, tile)
        blocks = [contiguous_block(len(self.all_tiles), r, self.size)
                  for r in range(self.size)]
        # every rank's rows, [start, stop), in the form allgather_ranges takes
        self.ranges = [[(self.all_tiles[t0][0], self.all_tiles[t1 - 1][1])]
                       if t1 > t0 else [] for t0, t1 in blocks]
        t0, t1 = blocks[self.rank]
        self.tiles = self.all_tiles[t0:t1]
        self.rows = (tuple(self.ranges[self.rank][0]) if self.ranges[self.rank]
                     else (0, 0))
        self.clock = dict.fromkeys(SUBSPACE_PIECES, 0.0)
        self.held_bytes = 0

    def holders(self, width):
        """The four holders V, W, U1, U2 of `width` trial pairs, one
        Fortran-ordered (tile rows, width) block per tile of this rank, as
        real_eig lays its whole ones out; their bytes are `held_bytes`."""
        out = [[np.empty((p1 - p0, width), order='F') for p0, p1 in self.tiles]
               for _ in range(4)]
        self.held_bytes = int(sum(b.nbytes for holder in out for b in holder))
        return out

    def lap(self, piece, t0):
        """The time now, with the seconds since t0 credited to `piece`."""
        t1 = time.time()
        self.clock[piece] += t1 - t0
        return t1

    def split(self, a):
        """This rank's tiles of `a` (..., rows of this rank), one view each."""
        r0 = self.rows[0]
        return [a[..., p0 - r0:p1 - r0] for p0, p1 in self.tiles]

    def join(self, addends, shape):
        """The sum over the whole pair index of per-tile addends: this rank's
        added in tile order onto the first (zeros where it owns no tile), then
        the ranks' partials all-reduced."""
        t = time.time()
        if addends:
            partial = np.array(addends[0], dtype=float, order='C', copy=True)
            for a in addends[1:]:
                partial += a
        else:
            partial = np.zeros(shape)
        t = self.lap('rows', t)
        reduce_sum(partial, self.comm)
        self.lap('reduce', t)
        return partial

    def norms(self, blocks, count):
        """2-norms of `count` vectors held as rows of this rank's tile blocks,
        (count, cols) each: squared per tile, summed, square-rooted."""
        t = time.time()
        addends = [np.linalg.norm(b, axis=1) ** 2 for b in blocks]
        self.lap('rows', t)
        return self.join(addends, (count,)) ** .5

    def agree(self, x):
        """Rank 0's copy of the replicated small result `x` on every rank,
        proven by digest and moved only where a rank's differs."""
        t = time.time()
        out = lockstep(x, self.comm, check=True)
        self.lap('lockstep', t)
        return out

    def gather(self, V, W, count):
        """The whole (count, 2N) batch [V, W] from every rank's tiles, V and W
        this rank's (count, tile rows) blocks: an output partition, moved
        verbatim. Rank r's rows travel as one (count, 2, rows) block, so
        nothing is transposed on either side."""
        t = time.time()
        n, k = self.n_pair, count
        buf = np.empty(2 * k * n)
        blocks = [[(2 * k * s0, 2 * k * s1) for s0, s1 in own]
                  for own in self.ranges]
        if self.tiles:
            r0, r1 = self.rows
            mine = buf[2 * k * r0:2 * k * r1].reshape(k, 2, r1 - r0)
            for (p0, p1), v, w in zip(self.tiles, V, W):
                mine[:, 0, p0 - r0:p1 - r0] = v
                mine[:, 1, p0 - r0:p1 - r0] = w
        allgather_ranges(buf, blocks, self.comm)
        out = np.empty((k, 2 * n))
        for own in self.ranges:
            for s0, s1 in own:
                block = buf[2 * k * s0:2 * k * s1].reshape(k, 2, s1 - s0)
                out[:, s0:s1] = block[:, 0]
                out[:, n + s0:n + s1] = block[:, 1]
        self.lap('gather', t)
        return out


def pair_tiles(n_pair, tile=None):
    """[start, stop) of every tile of the pair index, `tile` rows each
    (DAVIDSON_PAIR_TILE by default) but the last."""
    tile = DAVIDSON_PAIR_TILE if tile is None else int(tile)
    if tile < 1:
        raise ValueError(f'a pair tile holds at least one row, not {tile}')
    return [(p0, min(p0 + tile, n_pair)) for p0 in range(0, n_pair, tile)]


def projection_addend(V, W, U1, U2, m0, m1):
    """One tile's (4, m1 - m0, m1) share of the projected blocks a, b, sigma
    and pi: real_eig's `_sym_dot`/`_asym_dot` sums on the tile's holders."""
    a = _sym_dot(V, U1, m0, m1)
    a += _sym_dot(W, U2, m0, m1)
    b = _sym_dot(V, U2, m0, m1)
    b += _sym_dot(W, U1, m0, m1)
    sigma = _sym_dot(V, V, m0, m1)
    sigma -= _sym_dot(W, W, m0, m1)
    pi = _asym_dot(V, W, m0, m1)
    pi -= _asym_dot(W, V, m0, m1)
    return np.stack([a, b, sigma, pi])


def ritz_rows(V, W, U1, U2, m1, x, y, w):
    """One tile's rows of the Ritz vectors X, Y and residuals R_x, R_y for the
    subspace coefficients (x, y) and Ritz values w, and its addend of the
    squared residual norms."""
    X = V[:, :m1].dot(x)
    X += W[:, :m1].dot(y)
    Y = W[:, :m1].dot(x)
    Y += V[:, :m1].dot(y)
    R_x = U1[:, :m1].dot(x)
    R_x += U2[:, :m1].dot(y)
    R_x -= X * w
    R_y = U2[:, :m1].dot(x)
    R_y += U1[:, :m1].dot(y)
    R_y += Y * w
    r2 = np.linalg.norm(R_x, axis=0) ** 2
    r2 += np.linalg.norm(R_y, axis=0) ** 2
    return X, Y, R_x, R_y, r2


def overlap_addend(V, W, X_new, Y_new):
    """One tile's (2, m1, k) share of the new directions' overlaps with the
    held space and its partner space."""
    ox = V.T.dot(X_new)
    ox += W.T.dot(Y_new)
    oy = V.T.dot(Y_new)
    oy += W.T.dot(X_new)
    return np.stack([ox, oy])


def project_rows(V, W, X_new, Y_new, ox, oy):
    """One tile's rows of the new directions with the held space and its
    partner space projected off, in place, for the reduced overlaps."""
    X_new -= V.dot(ox)
    X_new -= W.dot(oy)
    Y_new -= W.dot(ox)
    Y_new -= V.dot(oy)


def combine_rows(X_new, Y_new, ca, cb):
    """One tile's (kept, rows) blocks of the orthonormalized directions and
    their partners' first halves, x_orth = X c_a + Y c_b, y_orth = Y c_a +
    X c_b."""
    x_orth = X_new.dot(ca)
    x_orth += Y_new.dot(cb)
    y_orth = Y_new.dot(ca)
    y_orth += X_new.dot(cb)
    return x_orth.T, y_orth.T


def gram_addend(X_new, Y_new):
    """One tile's (2, k, k) share of the projected directions' Gram matrix
    s11 and their overlap s21 with the partner directions."""
    s11 = X_new.T.dot(X_new)
    s11 += Y_new.T.dot(Y_new)
    s21 = X_new.T.dot(Y_new)
    s21 += Y_new.T.dot(X_new)
    return np.stack([s11, s21])


def symmetric_coefficients(s11, s21, lindep):
    """(c_a, c_b), the combinations x_orth = X c_a + Y c_b, y_orth = Y c_a +
    X c_b that orthonormalize k directions [X, Y] and their partners [Y, X]
    together, from their Gram matrices; None where none survives `lindep`.
    The k x k part of pyscf's `VW_Gram_Schmidt_fill_holder`, verbatim."""
    e, c = np.linalg.eigh(s11)
    mask = e > lindep
    e = e[mask]
    if e.size == 0:
        return None
    c = c[:, mask] * e**-.5
    csc = c.T.dot(s21).dot(c)
    n = csc.shape[0]
    lindep_sqrt = lindep**.5
    for i in range(n):
        w, u = np.linalg.eigh(csc[i:, i:])
        mask = 1 - abs(w) > lindep_sqrt
        if np.any(mask):
            c = c[:, i:]
            break
    else:
        return None
    w = w[mask]
    u = u[:, mask]
    c_orth = c.dot(u)
    if e[0] < lindep_sqrt or any(abs(w) > 1 - 1e-3):
        # re-orthogonalized once where the first pass was ill-conditioned
        e, c = np.linalg.eigh(c_orth.T.dot(s11).dot(c_orth))
        c *= e**-.5
        c_orth = c_orth.dot(c)
        csc = c_orth.T.dot(s21).dot(c_orth)
        w, u = np.linalg.eigh(csc)
        mask = 1 - abs(w) > lindep_sqrt
        w = w[mask]
        u = u[:, mask]
        c_orth = c_orth.dot(u)
    # [1 w; w 1] diagonalized by [a b; b a]
    a1 = (1 + w)**-.5
    a2 = (1 - w)**-.5
    a = (a1 + a2) / 2
    b = (a1 - a2) / 2
    return c_orth * a, c_orth * b


def gram_schmidt_rows(rows, V, W, X_new, Y_new, count, m1, lindep):
    """(V_new, W_new, kept): `count` new directions, given as this rank's
    (tile rows, count) blocks, projected off the held space and its partner
    space and orthonormalized with their partners, as pyscf's
    `VW_Gram_Schmidt_fill_holder` does whole; the kept directions come back
    as (kept, tile rows) blocks. X_new and Y_new are projected in place."""
    t = time.time()
    addends = [overlap_addend(v[:, :m1], w[:, :m1], x, y)
               for v, w, x, y in zip(V, W, X_new, Y_new)]
    t = rows.lap('rows', t)
    ox, oy = rows.join(addends, (2, m1, count))
    t = time.time()
    for v, w, x, y in zip(V, W, X_new, Y_new):
        project_rows(v[:, :m1], w[:, :m1], x, y, ox, oy)
    addends = [gram_addend(x, y) for x, y in zip(X_new, Y_new)]
    t = rows.lap('rows', t)
    s11, s21 = rows.join(addends, (2, count, count))
    t = time.time()
    coefficients = symmetric_coefficients(s11, s21, lindep)
    t = rows.lap('small', t)
    if coefficients is None:
        # the decision is the reduced Gram matrices', the same on every rank
        return ([np.zeros((0, x.shape[0])) for x in X_new],
                [np.zeros((0, y.shape[0])) for y in Y_new], 0)
    ca, cb = rows.agree(coefficients)
    t = time.time()
    V_new, W_new = [], []
    for x, y in zip(X_new, Y_new):
        v, w = combine_rows(x, y, ca, cb)
        V_new.append(v)
        W_new.append(w)
    rows.lap('rows', t)
    return V_new, W_new, ca.shape[1]


def real_eig_rows(aop, x0, precond, rows, tol_residual=1e-5, nroots=1,
                  x0sym=None, max_cycle=50, space_inc=None, max_space=None,
                  lindep=1e-12, verbose=logger.WARN, trace=None):
    """pyscf's `real_eig` on this rank's pair rows: (conv, e, xy) as it
    returns them, xy the nroots Ritz vectors [X, Y] gathered whole.

    aop: the whole (k, 2N) batch [V, W] in, this rank's rows of
        U1 = A V + B W and U2 = A W + B V out, (k, rows) each.
    precond: this rank's (k, 2 tile rows) blocks of the residuals [R_x, R_y]
        and the k Ritz values in, the corrections as blocks of that shape out.
    rows: the `PairRows` of this rank.
    space_inc, max_space: real_eig's corrections per cycle and the trial
        pairs held before it restarts.
    trace: optional list, one dict per cycle appended: the decisions the
        cycle took, every one the same on every rank.
    """
    log = logger.new_logger(verbose=verbose)
    n = x0.shape[1] // 2
    if x0sym is not None:
        x0_ir = np.asarray(x0sym)
    tiles = rows.tiles
    Vh, Wh, U1h, U2h = rows.holders(max_space)
    a = np.empty((max_space * 2, max_space * 2))
    b = np.empty_like(a)
    sigma = np.empty_like(a)
    pi = np.empty_like(a)
    e = None
    vlast = None
    conv_last = conv = np.zeros(nroots, dtype=bool)
    ritz = None                      # the last cycle's Ritz rows, per tile
    kept = None                      # its (x, y) for the restart products
    V = W = None
    count = len(x0)
    fresh_start = True
    for icyc in range(max_cycle):
        t = time.time()
        if fresh_start:
            m0 = m1 = 0
            if x0sym is not None:
                xs_ir = xt_ir = x0_ir
        if fresh_start and icyc == 0:
            V = [x0[:, p0:p1] for p0, p1 in tiles]
            W = [x0[:, n + p0:n + p1] for p0, p1 in tiles]
            U1, U2 = aop(np.hstack([x0[:, :n], x0[:, n:]]))
            count = len(x0)
            t = time.time()
            U1, U2 = rows.split(U1), rows.split(U2)
        elif fresh_start:
            # the restart products, combinations of the held ones
            xk, yk, m_old = kept
            V = [X[:, :nroots].T for X, _, _, _ in ritz]
            W = [Y[:, :nroots].T for _, Y, _, _ in ritz]
            U1, U2 = [], []
            for u1, u2 in zip(U1h, U2h):
                p = u1[:, :m_old].dot(xk)
                p += u2[:, :m_old].dot(yk)
                q = u2[:, :m_old].dot(xk)
                q += u1[:, :m_old].dot(yk)
                U1.append(p.T)
                U2.append(q.T)
            count = xk.shape[1]
        else:
            t = rows.lap('rows', t)
            batch = rows.gather(V, W, count)
            U1, U2 = aop(batch)
            del batch
            t = time.time()
            U1, U2 = rows.split(U1), rows.split(U2)
        m0, m1 = m1, m1 + count
        for i in range(len(tiles)):
            Vh[i][:, m0:m1] = V[i].T
            Wh[i][:, m0:m1] = W[i].T
            U1h[i][:, m0:m1] = U1[i].T
            U2h[i][:, m0:m1] = U2[i].T
        V = W = U1 = U2 = None
        addends = [projection_addend(Vh[i], Wh[i], U1h[i], U2h[i], m0, m1)
                   for i in range(len(tiles))]
        rows.lap('rows', t)
        a_block, b_block, sigma_block, pi_block = rows.join(
            addends, (4, m1 - m0, m1))
        t = time.time()
        a[:m1, m0:m1] = a_block.T
        a[m0:m1, :m1] = a_block
        b[:m1, m0:m1] = b_block.T
        b[m0:m1, :m1] = b_block
        sigma[:m1, m0:m1] = sigma_block.T
        sigma[m0:m1, :m1] = sigma_block
        pi[:m1, m0:m1] = -pi_block.T
        pi[m0:m1, :m1] = pi_block

        if x0sym is None:
            omega, x, y = TDDFT_subspace_eigen_solver(
                a[:m1, :m1], b[:m1, :m1], sigma[:m1, :m1], pi[:m1, :m1],
                space_inc)
            v_ir = None
        else:
            # each symmetry sector diagonalized apart
            omega = np.empty(m1)
            x = np.zeros((m1, m1))
            y = np.zeros_like(x)
            v_ir = []
            i1 = 0
            for ir in set(xs_ir):
                idx = np.nonzero(xs_ir[:m1] == ir)[0]
                _w, _x, _y = TDDFT_subspace_eigen_solver(
                    a[idx[:, None], idx], b[idx[:, None], idx],
                    sigma[idx[:, None], idx], pi[idx[:, None], idx], idx.size)
                i0, i1 = i1, i1 + idx.size
                omega[i0:i1] = _w
                x[idx, i0:i1] = _x
                y[idx, i0:i1] = _y
                v_ir.append([ir] * _w.size)
            idx = np.argsort(omega)
            omega = omega[idx]
            v_ir = np.hstack(v_ir)[idx]
            x = x[:, idx]
            y = y[:, idx]
        rows.lap('small', t)
        omega, x, y, v_ir = rows.agree((omega, x, y, v_ir))
        t = time.time()

        w, e, elast = omega[:space_inc], omega[:nroots], e
        v_sub = x[:, :space_inc]
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv, vlast,
                                           v_sub[:, :nroots], log)
        vlast = v_sub[:, :nroots]
        if elast is None or elast.size != e.size:
            de = e
        else:
            de = e - elast
        x = x[:, :space_inc]
        y = y[:, :space_inc]
        t = rows.lap('small', t)
        ritz, r2 = [], []
        for i in range(len(tiles)):
            X, Y, R_x, R_y, r2_i = ritz_rows(Vh[i], Wh[i], U1h[i], U2h[i],
                                             m1, x, y, w)
            ritz.append((X, Y, R_x, R_y))
            r2.append(r2_i)
        rows.lap('rows', t)
        r_norms = rows.join(r2, (x.shape[1],)) ** .5
        t = time.time()
        kept = (x[:, :nroots], y[:, :nroots], m1)
        if x0sym is not None:
            xt_ir = v_ir[:space_inc]
            x0_ir = v_ir[:nroots]

        max_r_norm = max(r_norms[:nroots])
        conv = r_norms[:nroots] <= tol_residual
        step = {'cycle': icyc, 'fresh': bool(fresh_start), 'new': int(count),
                'applied': int(count) if icyc == 0 or not fresh_start else 0,
                'm1': int(m1), 'r_norms': r_norms.copy(),
                'conv': tuple(bool(c) for c in conv), 'kept': None}
        if trace is not None:
            trace.append(step)
        for k, ek in enumerate(e[:nroots]):
            log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                       k, r_norms[k], ek, de[k])
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g', icyc, len(conv),
                      max_r_norm)
            rows.lap('small', t)
            break

        r_index = r_norms > tol_residual
        t = rows.lap('small', t)
        dx = [np.vstack([R_x[:, r_index], R_y[:, r_index]]).T
              for _, _, R_x, R_y in ritz]
        t = rows.lap('rows', t)
        timed = sum(rows.clock.values())
        XY_new = precond(dx, w[r_index])
        # the preconditioner's own tile work, its norms' pieces aside
        rows.clock['rows'] += (time.time() - t
                               - (sum(rows.clock.values()) - timed))
        t = time.time()
        X_new = [xy[:, :p1 - p0].T for xy, (p0, p1) in zip(XY_new, tiles)]
        Y_new = [xy[:, p1 - p0:].T for xy, (p0, p1) in zip(XY_new, tiles)]
        rows.lap('rows', t)
        n_new = int(np.count_nonzero(r_index))
        if x0sym is None:
            V, W, count = gram_schmidt_rows(rows, Vh, Wh, X_new, Y_new, n_new,
                                            m1, lindep)
        else:
            xt_ir = xt_ir[r_index]
            xt_orth_ir = []
            parts = []
            for ir in set(xt_ir):
                idx = np.nonzero(xt_ir == ir)[0]
                parts.append(gram_schmidt_rows(
                    rows, Vh, Wh, [X[:, idx] for X in X_new],
                    [Y[:, idx] for Y in Y_new], idx.size, m1, lindep))
                xt_orth_ir.append([ir] * parts[-1][2])
            count = sum(part[2] for part in parts)
            if len(parts) > 0:
                V = [np.vstack([part[0][i] for part in parts])
                     for i in range(len(tiles))]
                W = [np.vstack([part[1][i] for part in parts])
                     for i in range(len(tiles))]
                xt_ir = np.hstack(xt_orth_ir)
                xs_ir = np.hstack([xs_ir, xt_ir])
        X_new = Y_new = XY_new = dx = None
        step['kept'] = int(count)

        if count == 0:
            log.debug('Linear dependency in trial subspace. |r| for each '
                      'state %s', r_norms)
            break
        log.debug('real_lr_eig %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                  icyc, m1, max_r_norm, e, de[np.argmax(abs(de))])
        fresh_start = m1 + count > max_space

    if len(e) < min(n, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(e)}, '
                 f'nroots={nroots})')
    xy = rows.gather([X[:, :nroots].T for X, _, _, _ in ritz],
                     [Y[:, :nroots].T for _, Y, _, _ in ritz], len(e))
    return conv[:nroots], e[:nroots], xy
