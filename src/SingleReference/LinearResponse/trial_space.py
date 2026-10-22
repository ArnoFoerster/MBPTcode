"""The Casida Davidson's own loop (`casida_rows`), its trial space cut by
pair rows over ranks: pyscf's `real_eig` with every pair-space-long array
partitioned and Ritz pairs tracked above the requested roots. Serially it runs
on one tile as long as the pair space.

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
formed per tile, and every tile's addend reaches every rank to be added in
tile order onto the first (`ordered_sum`): the one-rank sum at any rank
count, so the rank count enters the iteration's arithmetic nowhere. The cost
is the tiles' small matrices gathered: the number of pair tiles times the
size of one projected block per join.

WHAT CROSSES, per cycle: the new batch, gathered whole for the block action
(`allgather_ranges`, k x 2N doubles); one ordered sum of the four (k, m1)
projected blocks, one of the residual norms, one or two of the correction
norms, and two of the Gram-Schmidt overlaps, (2, m1, k') and (2, k', k'); and
a checked `lockstep` of the subspace solution and of the Gram-Schmidt
coefficients, whose digests prove every rank builds its rows from rank 0's.
The converged X and Y are gathered once at the end. Every decision -- which
roots converged, which corrections are sized, how many directions survive the
linear-dependence tests, when the space restarts -- is read off reduced or
lockstepped small matrices, so it is the same on every rank and the
collectives pair.

RESTARTS take the lowest max(2 nroots, tracked) Ritz vectors (real_eig takes
the nroots requested ones), and their products as the same combinations of the
stored products, A (V x + W y) + B (W x + V y) = U1 x + U2 y, instead of
applying the action to them again.

ONE TILE. On one rank with one tile every expression is the one real_eig
evaluates: a tile's addend is real_eig's own sum, a sum over one tile is that
addend, and the square root of a squared norm is the norm again
(sqrt(fl(x^2)) = |x| in binary floating point), so a solve without restarts
whose tracked pairs are no more than real_eig corrects, and settled where its
requested roots converge, takes real_eig's decisions on real_eig's sums.
"""
import time

import numpy as np
from pyscf.lib import logger
from pyscf.lib.linalg_helper import _sort_elast
from pyscf.tdscf._lr_eig import (TDDFT_subspace_eigen_solver, _asym_dot,
                                 _sym_dot)

from src.Base.constants import (DAVIDSON_ACTION_SYMMETRY_TOL, DAVIDSON_LINDEP,
                                DAVIDSON_PAIR_TILE)
from src.Base.utils.mpi_grid import (allgather_ranges, contiguous_block,
                                     lockstep, ordered_sum)


#: The owned loop's own time outside the block action, each piece timed
#: apart: the per-tile arithmetic, the replicated small solves, the batch and
#: result gathers, the reductions of the small matrices, and the checked
#: locksteps of the subspace solution and the Gram-Schmidt coefficients.
SUBSPACE_PIECES = ('rows', 'small', 'gather', 'reduce', 'lockstep')


class ActionNotSymmetric(RuntimeError):
    """A Casida block action whose projected A or B is not symmetric beyond
    round-off: the action is wrong, and so is every root solved on it.
    `asymmetry` is the measured skew part relative to max |a|
    (`projected_asymmetry`)."""

    def __init__(self, message, asymmetry):
        super().__init__(message)
        self.asymmetry = float(asymmetry)


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
        self.first_tile = t0
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
        """The sum over the whole pair index of per-tile addends, every
        tile's added in tile order onto the first (`ordered_sum`): the
        one-rank bits at every rank count."""
        t = self.lap('rows', time.time())
        total = ordered_sum([(self.first_tile + i, a)
                             for i, a in enumerate(addends)], self.comm,
                            shape=shape)
        self.lap('reduce', t)
        return total

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


def projected_asymmetry(a, a_skew, b_skew):
    """max(|a_skew|, |b_skew|) / max |a|: the parts of the projected A and B
    that a symmetric action leaves to round-off, relative to the block."""
    scale = np.abs(a).max()
    if scale == 0.0:
        return 0.0
    return float(max(np.abs(a_skew).max(), np.abs(b_skew).max()) / scale)


def _sym_skew_dot(V, U1, m0, m1):
    """(`_sym_dot`, its skew part): real_eig's (V^T U1 + U1^T V)[m0:m1, :m1]
    / 2 to the bit, and V^T U1 - U1^T V there, round-off for a symmetric
    operator however wrong its value."""
    p = V[:, m0:m1].T.dot(U1[:, :m1])
    q = U1[:, m0:m1].T.dot(V[:, :m1])
    skew = p - q
    p += q
    p *= .5
    return p, skew


def projection_addend(V, W, U1, U2, m0, m1):
    """One tile's (6, m1 - m0, m1) share of the projected blocks a, b, sigma
    and pi -- real_eig's `_sym_dot`/`_asym_dot` sums on the tile's holders
    -- and of the skew parts of a and b, which real_eig's symmetrization
    hides: an action that is not symmetric is iterated as its symmetric
    part."""
    a, a_skew = _sym_skew_dot(V, U1, m0, m1)
    a2, skew = _sym_skew_dot(W, U2, m0, m1)
    a += a2
    a_skew += skew
    b, b_skew = _sym_skew_dot(V, U2, m0, m1)
    b2, skew = _sym_skew_dot(W, U1, m0, m1)
    b += b2
    b_skew += skew
    sigma = _sym_dot(V, V, m0, m1)
    sigma -= _sym_dot(W, W, m0, m1)
    pi = _asym_dot(V, W, m0, m1)
    pi -= _asym_dot(W, V, m0, m1)
    return np.stack([a, b, sigma, pi, a_skew, b_skew])


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
    `VW_Gram_Schmidt_fill_holder` does whole but projected twice; the kept
    directions come back as (kept, tile rows) blocks. X_new and Y_new are
    projected in place.

    Twice, as classical Gram-Schmidt needs to hold orthogonality (and as
    `warm_trial_space` does): a correction that lies almost wholly inside the
    held space keeps, after one projection, the round-off of the part
    removed, and the combination that normalizes it multiplies that
    round-off by up to lindep^-1/2. Projected once, formaldehyde/G1 BSE in
    toluene (240 pairs, 4 roots at 1e-9, the Eq. (18) after-solve QP
    energies) drifted from orthonormality by 2.5e-4 at 112 held pairs and
    0.4 at 232, until V - W lost rank and the projected (A-B) block broke
    down with min eig(A-B) = 0.15 Ha; projected twice the drift stays at
    6e-8 up to the whole pair space."""
    t = time.time()
    for _ in range(2):
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


def warm_trial_space(X, Y, x0, lindep):
    """The first trial batch of a warm-started Davidson: (k', 2N) rows
    [V, W], the guess vectors [X, Y] -- (N, k) each, a previous geometry's
    roots carried into this one's orbitals -- and then the unit-vector rows
    of the cold guess `x0`, each block projected off everything before it
    and its partner space and orthonormalized with its own partners
    (`symmetric_coefficients`), as real_eig holds a trial space: the rows
    and their x/y-swapped partners one orthonormal set of 2k' vectors.

    The guess comes first, so a cold direction it already spans drops at
    `lindep` instead of crowding it; the cold rows stay because they span
    what the previous roots do not -- a root that moves into the lowest few
    from above has nothing else to be found from. Whole arrays on every
    rank: the batch is a function of replicated inputs alone and the same at
    any rank count, and `casida_rows` cuts it into its tiles as it does the
    cold one.
    """
    n = x0.shape[1] // 2
    V, W = np.zeros((n, 0)), np.zeros((n, 0))
    for x_new, y_new in ((X, Y), (x0[:, :n].T, x0[:, n:].T)):
        x_new = np.array(x_new, dtype=float, order='C')
        y_new = np.array(y_new, dtype=float, order='C')
        # twice, as classical Gram-Schmidt needs to hold orthogonality
        for _ in range(2):
            ox, oy = overlap_addend(V, W, x_new, y_new)
            project_rows(V, W, x_new, y_new, ox, oy)
        coefficients = symmetric_coefficients(*gram_addend(x_new, y_new),
                                              lindep)
        if coefficients is None:
            continue
        v, w = combine_rows(x_new, y_new, *coefficients)
        V, W = np.hstack([V, v.T]), np.hstack([W, w.T])
    return np.hstack([V.T, W.T])


def _orthonormal_corrections(rows, Vh, Wh, XY_new, count, m1, lindep,
                             xt_ir=None):
    """(V, W, kept, irreps): the corrections `XY_new`, this rank's (count,
    2 tile rows) blocks, projected off the held space and orthonormalized
    with their partners (`gram_schmidt_rows`), each irrep of `xt_ir` apart
    where one is given; `irreps` labels the kept directions (None without
    `xt_ir`). XY_new is projected in place."""
    tiles = rows.tiles
    X_new = [xy[:, :p1 - p0].T for xy, (p0, p1) in zip(XY_new, tiles)]
    Y_new = [xy[:, p1 - p0:].T for xy, (p0, p1) in zip(XY_new, tiles)]
    if xt_ir is None:
        V, W, kept = gram_schmidt_rows(rows, Vh, Wh, X_new, Y_new, count, m1,
                                       lindep)
        return V, W, kept, None
    xt_orth_ir = []
    parts = []
    for ir in set(xt_ir):
        idx = np.nonzero(xt_ir == ir)[0]
        parts.append(gram_schmidt_rows(
            rows, Vh, Wh, [X[:, idx] for X in X_new],
            [Y[:, idx] for Y in Y_new], idx.size, m1, lindep))
        xt_orth_ir.append([ir] * parts[-1][2])
    kept = sum(part[2] for part in parts)
    if kept == 0:
        return None, None, 0, None
    V = [np.vstack([part[0][i] for part in parts]) for i in range(len(tiles))]
    W = [np.vstack([part[1][i] for part in parts]) for i in range(len(tiles))]
    return V, W, kept, np.hstack(xt_orth_ir)


def casida_rows(aop, x0, precond, rows, tol_residual, nroots, max_cycle,
                ntrack=None, x0sym=None, space_inc=None,
                max_space=None, verbose=logger.WARN,
                trace=None, resize=None,
                symmetry_tol=DAVIDSON_ACTION_SYMMETRY_TOL):
    """The Casida Davidson on this rank's pair rows, tracking Ritz pairs above
    the requested ones: (conv, e, xy, lower).

    conv: the `nroots` requested roots' |r| <= tol_residual.
    e, xy: the Ritz values and vectors [X, Y], gathered whole, of the
        requested roots and the tracked extra pairs above them.
    lower: the lowest bound theta - r_S of a tracked extra pair left
        unsettled where the solve stopped (inf where every one settled);
        a requested root above it may not be the lowest.

    aop: the whole (k, 2N) batch [V, W] in, this rank's rows of
        U1 = A V + B W and U2 = A W + B V out, (k, rows) each.
    precond: this rank's (k, 2 tile rows) blocks of the residuals [R_x, R_y]
        and the k Ritz values in, the corrections as blocks of that shape out.
    rows: the `PairRows` of this rank; one tile as long as the pair space is
        pyscf's `real_eig` arithmetic.
    ntrack: the Ritz pairs tracked, the requested ones included (nroots by
        default).
    space_inc, max_space: the Ritz pairs corrected per cycle while a
        requested root is unconverged (at least ntrack) and the trial pairs
        held before the space restarts on its lowest max(2 nroots, ntrack)
        Ritz vectors, at most space_inc.
    trace: optional list, one dict per cycle appended: the decisions the
        cycle took, every one the same on every rank; 'intervals' counts the
        Ritz pairs whose r_S it measured, one block action each.
    symmetry_tol: the largest skew part V^T A V - (A V)^T V (and B's) of
        the projected blocks on each cycle's new trial pairs, relative to
        max |a|, that a symmetric action leaves (`projected_asymmetry`);
        above it the action is refused (`ActionNotSymmetric`), whatever the
        residuals say.
    resize: optional, `precond`'s signature: the corrections with every one
        sized to the length the lindep test accepts, asked for only where a
        cycle kept no direction of `precond`'s (the cycle's 'rescued').

    THE ITERATION is real_eig's while a requested root is unconverged: every
    unconverged pair among the lowest space_inc is corrected. Where the
    requested roots have converged, a tracked extra pair is settled once it
    has converged or its residual interval clears the highest requested root,
    theta - r_S > omega_nroots, and the solve stops when every one is
    settled; until then the unsettled extras alone are corrected. A root
    whose Ritz value starts above the next root up is so reached rather than
    converged past.

    THE INTERVAL is in the metric S = [[A, B], [B, A]], in which H = J S,
    J = diag(1, -1), is self-adjoint: for any z and theta an eigenvalue of
    H lies within r_S = |H z - theta z|_S / |z|_S of theta. H z - theta z =
    J R for the Casida residual R = S z - theta J z, so
    r_S^2 = R^T (J S J) R / z^T S z, one block action on [R_x, -R_y] per
    unconverged extra pair; S not positive definite (an unstable reference)
    raises LinAlgError.
    """
    log = logger.new_logger(verbose=verbose)
    n = x0.shape[1] // 2
    ntrack = nroots if ntrack is None else max(int(ntrack), nroots)
    space_inc = ntrack if space_inc is None else max(int(space_inc), ntrack)
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
    w = np.zeros(0)
    extra = np.zeros(0, int)
    settled = np.zeros(0, bool)
    r_s = None

    def interval(idx):
        """r_S of the Ritz pairs `idx` of the last cycle: one block action
        on their [R_x, -R_y], two joined sums."""
        k = len(idx)
        U1, U2 = aop(rows.gather([R_x[:, idx].T for _, _, R_x, _ in ritz],
                                 [-R_y[:, idx].T for _, _, _, R_y in ritz],
                                 k))
        t = time.time()
        U1, U2 = rows.split(U1), rows.split(U2)
        addends = []
        for (X, Y, R_x, R_y), u1, u2 in zip(ritz, U1, U2):
            x, y, rx, ry = X[:, idx], Y[:, idx], R_x[:, idx], R_y[:, idx]
            # R^T (J S J) R and z^T S z = z . (R + theta J z)
            jsj = np.sum(rx * u1.T, axis=0) - np.sum(ry * u2.T, axis=0)
            szz = (np.sum(x * rx, axis=0) + np.sum(y * ry, axis=0)
                   + w[idx] * (np.sum(x * x, axis=0) - np.sum(y * y, axis=0)))
            addends.append(np.stack([jsj, szz]))
        rows.lap('rows', t)
        jsj, szz = rows.join(addends, (2, k))
        if np.any(jsj < 0) or np.any(szz <= 0):
            raise np.linalg.LinAlgError(
                'S = [[A, B], [B, A]] is not positive definite on a tracked '
                'Ritz pair: the Casida residual interval does not exist')
        return np.sqrt(jsj / szz)

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
            V = [X[:, :xk.shape[1]].T for X, _, _, _ in ritz]
            W = [Y[:, :xk.shape[1]].T for _, Y, _, _ in ritz]
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
        a_block, b_block, sigma_block, pi_block, a_skew, b_skew = rows.join(
            addends, (6, m1 - m0, m1))
        t = time.time()
        asymmetry = projected_asymmetry(a_block, a_skew, b_skew)
        if asymmetry > symmetry_tol:
            raise ActionNotSymmetric(
                f'Casida-form Davidson failed: the block action is not '
                f'symmetric, its projected A and B on the {m1 - m0} new trial '
                f'pairs of cycle {icyc} carry a skew part of '
                f'{asymmetry:.2e} of max |a| against {symmetry_tol:g}. The '
                'action is wrong (a grid or pair row block out of place, a '
                'kernel not symmetric), not merely unconverged.', asymmetry)
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
        nkeep = min(ntrack, len(w))
        # a restart keeps the Tamm-Dancoff loop's 2 nroots, or the tracked
        nrestart = min(max(2 * nroots, ntrack), len(w))
        kept = (x[:, :nrestart], y[:, :nrestart], m1)
        if x0sym is not None:
            xt_ir = v_ir[:space_inc]
            x0_ir = v_ir[:nrestart]

        max_r_norm = max(r_norms[:nroots])
        conv = r_norms[:nroots] <= tol_residual
        extra = np.arange(min(nroots, nkeep), nkeep)
        settled = r_norms[extra] <= tol_residual
        r_s = None                   # r_S of the extras, 0 where converged
        if all(conv) and not settled.all():
            rows.lap('small', t)
            r_s = np.zeros(len(extra))
            r_s[~settled] = interval(extra[~settled])
            t = time.time()
            settled |= w[extra] - r_s > e.max()
        step = {'cycle': icyc, 'fresh': bool(fresh_start), 'new': int(count),
                'applied': int(count) if icyc == 0 or not fresh_start else 0,
                'm1': int(m1), 'omega': w.copy(), 'r_norms': r_norms.copy(),
                'asymmetry': asymmetry,
                'conv': tuple(bool(c) for c in conv), 'kept': None,
                'settled': tuple(bool(s) for s in settled),
                'r_s': None if r_s is None else r_s.copy(),
                'intervals': 0 if r_s is None else int(np.count_nonzero(
                    r_norms[extra] > tol_residual))}
        if trace is not None:
            trace.append(step)
        for k, ek in enumerate(e[:nroots]):
            log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                       k, r_norms[k], ek, de[k])
        if all(conv) and settled.all():
            log.debug('converged %d %d  |r|= %4.3g', icyc, len(conv),
                      max_r_norm)
            rows.lap('small', t)
            break

        if all(conv):
            r_index = np.zeros(len(w), dtype=bool)
            r_index[extra[~settled]] = True
        else:
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
        n_new = int(np.count_nonzero(r_index))
        if x0sym is not None:
            xt_ir = xt_ir[r_index]
        V, W, count, xt_new = _orthonormal_corrections(
            rows, Vh, Wh, XY_new, n_new, m1, DAVIDSON_LINDEP,
            None if x0sym is None else xt_ir)
        # A CYCLE THAT KEEPS NOTHING IS RE-SIZED ONCE before it stops the
        # solve: a correction from |r| above the sizing threshold keeps its
        # raw length r / (d - omega), and one too short for DAVIDSON_LINDEP is
        # dropped however much of it lies outside the space (benzene/cc-pVDZ
        # BSE on the G1 grid, bare preconditioner: root 3 of 12 at
        # |r| = 1.5e-5, a correction 6.3e-6 long with 0.1 of it outside). Only
        # where the pair space has not run out, whose stop the dense
        # completion owns; a solve that keeps a direction every cycle never
        # comes here.
        rescued = (count == 0 and resize is not None and n_new > 0
                   and n_new <= n - m1)
        if rescued:
            V, W, count, xt_new = _orthonormal_corrections(
                rows, Vh, Wh, resize(dx, w[r_index]), n_new, m1,
                DAVIDSON_LINDEP,
                None if x0sym is None else xt_ir)
        if x0sym is not None and count > 0:
            xt_ir = xt_new
            xs_ir = np.hstack([xs_ir, xt_ir])
        XY_new = dx = None
        step['kept'] = int(count)
        step['rescued'] = bool(rescued)

        if count == 0:
            log.debug('Linear dependency in trial subspace. |r| for each '
                      'state %s', r_norms)
            break
        log.debug('casida_rows %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                  icyc, m1, max_r_norm, e, de[np.argmax(abs(de))])
        fresh_start = m1 + count > max_space

    if len(e) < min(n, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(e)}, '
                 f'nroots={nroots})')
    lower = np.inf
    if not settled.all():
        # stopped with an extra unsettled: its interval, measured here where
        # the last cycle did not measure it, bounds what may lie below
        if r_s is None:
            r_s = np.zeros(len(extra))
            r_s[~settled] = interval(extra[~settled])
            if trace is not None:
                trace[-1]['intervals'] = int(np.count_nonzero(~settled))
            settled |= w[extra] - r_s > e.max()
        if not settled.all():
            lower = float(np.min((w[extra] - r_s)[~settled]))
    nkeep = min(ntrack, len(w))
    xy = rows.gather([X[:, :nkeep].T for X, _, _, _ in ritz],
                     [Y[:, :nkeep].T for _, Y, _, _ in ritz], nkeep)
    return conv[:nroots], w[:nkeep], xy, lower
