"""The adjoint of the row-distributed ISDF fit (`separable_ri.fit_rows`):
an adjoint on D, X_mo, M^T, the collocation or the metric carried back to the
nuclei.

`fit_rows_adjoint` runs in the fit's own tiles and under the same rules: every
call's shape fixed by tile indices, and no per-rank partial summed across
ranks. A sum over the grid is accumulated by one rank in tile order from tiles
that travel to it: the Gram terms by the row tile's owner over the streamed
column tiles, a shell block's (mu nu|P) adjoint by the block's owner over the
streamed U, the metric's two (naux, naux) sums on rank 0; every (natm, 3)
partial is gathered and added in its fixed order. The force is therefore the
same bits at every rank count, and matches the whole adjoint
(`isdf_derivatives.dfactor_adjoint_gauges` on every product pair) to the fit's
conditioning.

The pass holds three row arrays beside the kept factor and the LU, which is
its peak; the metric adjoint holds four metric-sized arrays on rank 0 alone
(the eigenvectors, one accumulated sum and two products), whatever the rank
count. D_bar and X_bar arrive whole or in the fit's tiles, and a rank reads
its tiles of them; rank 0 receives each D_bar tile beside its M^T tile.

`fit_rows_adjoints` runs several targets in one pass: the Gram tiles and
their factor, the three-centre pass and its integrals, the re-solved M^T,
the collocations and the derivative integrals once, the seed-linear part
once per target, so each target is the bits of its own call; its per-target
row arrays (MT_bar, Q_bar, U, X_bar, B_bar, P_bar) add to the table above.
A fit already made is read rather than repeated: `FitTiles`, a stage's own
tiles of it, which `fit_rows(tiles=)` reads where they are its fit.
"""
import numpy as np
import scipy.linalg

from src.Base.numerics.constants import BYTES_PER_GB, FIT_CHOLESKY_BLOCK, THREE_CENTER_BLOCK_BYTES
from src.Base.factorization.sliced_factors import GridTileRows
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, broadcast, broadcast_rows,
                                              current_comm, exchange_blocks, lockstep, partition)
from src.Base.factorization.separable_ri import (ANGULAR_WEIGHTS, fit_settings, KeptIntegrals,
                                                 ao_l_labels, ao_blocks, gram_tiles,
                                                 block_coefficients, total_bytes, frozen_columns,
                                                 tile_collocation, trsm, root_slab, metric_root,
                                                 metric_spectrum)


class RowFitAdjoint:
    """The row fit's adjoint on this rank (`fit_rows_adjoint`), contracted to
    the few arrays a nuclear gradient reads, the same bits on every rank.

    fit_centre: (natm, 3), what the fit's integrals contribute through their
        centres: the AO and auxiliary collocations', the three-centre
        (mu nu|P) and the two-centre (P|Q) integrals'.
    fit_points: (nk, 3), dE/dr_g through the fit's collocations at every
        point, for the chain through the points and their frames.
    coll_centre, coll_points: the same two of the X_mo collocation
        adjoint X_bar C^T, None without an X_bar.
    held: {array: most bytes of it this rank held at once}.
    """

    def __init__(self, fit_centre, fit_points, coll_centre, coll_points,
                 held):
        self.fit_centre, self.fit_points = fit_centre, fit_points
        self.coll_centre, self.coll_points = coll_centre, coll_points
        self.held = held


class AdjointSeeds:
    """One target's seeds of the row fit's adjoint (`fit_rows_adjoints`).

    d_bar: (nk, naux) adjoint on D = M^T V^1/2, whole and the same bits on
        every rank, or `GridTileRows` in the fit's tiles.
    mt_bar: the adjoint on M^T itself, {tile: rows} of this rank's tiles,
        for a target reading M and not D (the interpolated exchange,
        Z = M^T V M); beside d_bar it adds to D_bar V^1/2.
    x_bar, mo_coeff: the adjoint on X_mo = X_ao C (whole or tiles) and C,
        whose collocation adjoint X_bar C^T rides the fit's derivative
        collocation.
    x_bar_ao: the adjoint on the AO collocation, {tile: rows}; beside x_bar
        it adds to X_bar C^T.
    metric_bar: an adjoint on the bare metric V from outside the fit (Z's
        own V), on rank 0 (None elsewhere), added before its derivative.
    gauges: `GaugeSeeds` of further factors D_g = M^T (V + vtilde_g)^1/2 on
        the same fit, each with its own dressed metric root.
    """

    def __init__(self, d_bar=None, x_bar=None, mo_coeff=None, mt_bar=None,
                 x_bar_ao=None, metric_bar=None, gauges=()):
        if d_bar is None and mt_bar is None and not gauges:
            raise ValueError('the fit adjoint takes an adjoint on D (d_bar), '
                             'on M^T (mt_bar), on a dressed gauge (gauges), '
                             'or several')
        if x_bar is not None and mo_coeff is None:
            raise ValueError('an adjoint on X_mo needs the orbitals C of '
                             'X_mo = X_ao C')
        self.d_bar, self.x_bar, self.mo_coeff = d_bar, x_bar, mo_coeff
        self.mt_bar, self.x_bar_ao = mt_bar, x_bar_ao
        self.metric_bar = metric_bar
        self.gauges = tuple(gauges)


class GaugeSeeds:
    """The adjoint on one DRESSED gauge of the fit, D_g = M^T R_g with
    R_g = (V + vtilde_g)^1/2 (`aux_metric_sqrt` with `environment`).

    d_bar: (nk, naux) adjoint on D_g, whole or `GridTileRows` in the fit's
        tiles.
    environment: what dresses the metric (`aux_kernel`, `aux_kernel_adjoint`).
    root_bar: an adjoint on R_g from a term reading the root without D,
        on rank 0 (None elsewhere, or none at all).
    kernel_bar: an adjoint on vtilde_g alone, on rank 0 (None elsewhere).
    """

    def __init__(self, d_bar, environment, root_bar=None, kernel_bar=None):
        if environment is None:
            raise ValueError('a dressed gauge needs the environment that '
                             'dresses its metric; the bare gauge is d_bar')
        self.d_bar, self.environment = d_bar, environment
        self.root_bar, self.kernel_bar = root_bar, kernel_bar


def fit_rows_adjoint(mol, auxmol, coords, d_bar, layout, x_bar=None,
                     mo_coeff=None, numerics=None, comm=None, block=None,
                     mt_bar=None,
                     x_bar_ao=None, metric_bar=None, gauges=()):
    """The nuclear adjoint of `fit_rows`' factor D = M^T V^1/2 on the frozen
    `layout`, in the forward fit's own tiles: a `RowFitAdjoint`, the one
    target of `fit_rows_adjoints` whose seeds are these (`AdjointSeeds`;
    gauges: `GaugeSeeds` of dressed factors on the same fit).

    d_bar: (nk, naux) adjoint on D, the same bits on every rank (a kernel's
    all-reduced output); x_bar, mo_coeff: the adjoint on X_mo = X_ao C and
    C, whose collocation adjoint rides on the same derivative collocation.
    mt_bar: the adjoint on M^T itself as {tile: rows}, this rank's tiles,
    for a target that reads M and not D (the interpolated exchange,
    Z = M^T V M); alone no metric root is formed, beside d_bar it adds to
    D_bar V^1/2. x_bar_ao: the adjoint on the AO collocation as {tile:
    rows}, this rank's tiles, alone or added to X_bar C^T. metric_bar: an
    adjoint on the bare metric V from outside the fit (Z's own V), on rank 0
    (None elsewhere), added before its derivative.
    """
    seeds = AdjointSeeds(d_bar=d_bar, x_bar=x_bar, mo_coeff=mo_coeff,
                         mt_bar=mt_bar, x_bar_ao=x_bar_ao,
                         metric_bar=metric_bar, gauges=gauges)
    return fit_rows_adjoints(mol, auxmol, coords, layout, [seeds],
                             numerics=numerics, comm=comm,
                             block=block)[0]


def fit_rows_adjoints(mol, auxmol, coords, layout, targets, numerics=None,
                      comm=None, block=None):
    """[RowFitAdjoint], one per target: the nuclear adjoint of `fit_rows`'
    factor D = M^T V^1/2 on the frozen `layout`, in the forward fit's own
    tiles, every target's in one pass. A target is an `AdjointSeeds`, or a
    list of them whose adjoints are only wanted summed (a pool).

    The estimator is `fit_rows`' product form, and its reverse pass never
    forms an nk x nk or a test-set-wide array (Q = (F D^T)^T, R = d Q,
    Z = G^-1 R, M^T = d Z, G = d S d + reg, S = (X X^T) o (B B^T) + P P^T):

        MT_bar = D_bar V^1/2 (+ MT_bar),  W = G^-1 d MT_bar,  Q_bar = d W,
        U = Q_bar V^-1,  H = Q_bar M + M^T Q_bar^T  (tile by tile)
        d_bar_k = MT_bar_k . Z_k + W_k . Q_k - (1/d_k) sum_l H_kl S_kl
        X_bar = -(H o B B^T) X,  B_bar = -(H o X X^T) B,
        P_bar = Q_bar - H P,  and the balancing's diagonal term
        (mu nu|P)_bar = w U^T D_pair,  V_bar = (V^1/2)'^T(M D_bar)
                        - (Q - P)^T U

    since G^-1 applied to the solve's adjoint is low rank, G_bar = -W Z^T,
    and the V^-1 inside F reaches the metric as a sum over the grid. Every
    step is a call of a shape fixed by the tile indices, and nothing is
    summed across ranks: a sum over the grid is accumulated on one rank in
    tile order from tiles that travel (the metric's two, on rank 0), the
    Gram terms by the row tile's owner over the column tiles streamed in
    order, each block's (mu nu|P) adjoint by the block's owner over U
    streamed in order, and every (natm, 3) partial (a tile's, a block's,
    a metric slab's) is gathered and added in its fixed order. So the
    result is the same bits at every rank count, one rank included.

    One pass serves several targets. What no seed enters is formed once: the
    Gram tiles and their Cholesky factor, the metric's LU and root, every
    shell block's fitting coefficients and the pass's Q, the re-solved M^T,
    the collocations, each (tile, tile) Gram block (X_i X_k^T) o (B_i B_k^T)
    + P_i P_k^T, and the derivative integrals of the kept pairs and of the
    metric. What is linear in the seeds (MT_bar's root and solves, the
    pass's pair adjoint, U, the H blocks, the balancing and the collocation
    adjoints) runs per target in the calls a pass with that target alone
    makes, its tiles travelling as they do there, so every target's
    adjoint is the same bits as its own `fit_rows_adjoint`. A pool is summed
    into one seed set first (`summed_seeds`) and is then a target like any:
    the same bits as `fit_rows_adjoint` of those summed seeds.

    Its arrays (tile t of `block` points owned by rank t % size), per
    target where marked:
      Gram S, then its factor L   this rank's lower row tiles, then its
                                  diagonal blocks and gathered panels
      X, B, P collocation         this rank's tiles (B, P dropped through
                                  the pass); column tiles streamed
      MT_bar, Q_bar, U, P_bar     per target, this rank's tiles, (rows,
                                  naux) each
      Q, then M^T                 this rank's tiles, (rows, naux)
      X_bar, B_bar                per target, this rank's tiles
      M D_bar, (Q - P)^T U        per target, rank 0, (naux, naux), from
                                  tiles sent to it one at a time
      V^1/2 and its adjoint       rank 0 (`metric_root`, an in-place
                                  eigendecomposition), slabs elsewhere
      F_b, (mu nu|P) of a block   its owner, the kept pairs alone, F_b
                                  broadcast as in the forward pass
      the (mu nu|P) adjoints      per target, the owner's blocks of one
                                  batch of at most `block_memory_gb` per
                                  rank over the targets, U streamed once
                                  per batch
      (P|Q)' of a slab            the slab's owner, `block` rows
    D_bar and X_bar arrive whole, or as `GridTileRows` in these tiles and
    owners; a rank reads its tiles of them, and rank 0 receives each D_bar
    tile beside its M^T tile for M D_bar. The same values reach the same
    calls either way, so the two are the same bits.

    DRESSED GAUGES (`AdjointSeeds.gauges`): a factor D_g = M^T R_g on the
    same fit with R_g = (V + vtilde_g)^1/2 adds D_bar_g R_g to MT_bar (R_g
    on rank 0, streamed in slabs, after the bare gauge), so everything below
    MT_bar runs once for the gauges summed. Its root adjoint M D_bar_g
    (+ root_bar) is summed on rank 0 like M D_bar; there its Frechet adjoint
    Y_g on V + vtilde_g joins V_bar, and vtilde_g's own derivative,
    `environment.aux_kernel_adjoint(Y_g + kernel_bar)`, joins fit_centre,
    sent from rank 0. A gauge adds one (rows, naux) slab stream, one tile
    pass to rank 0 and, on rank 0, the root and its adjoint
    (`_dressed_root_adjoint`, four metric-sized arrays), one gauge at a time.
    """
    l_max_second, regularization, block_memory_gb = fit_settings(
        numerics, 'l_max_second', 'regularization', 'block_memory_gb')
    comm = current_comm() if comm is None else comm
    rank, nranks = ((0, 1) if comm is None
                    else (comm.Get_rank(), comm.Get_size()))
    if nranks > 1:
        coords = lockstep(coords, comm)
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    nk = len(coords)
    ntiles = -(-nk // block)
    tiles = [(t * block, min((t + 1) * block, nk)) for t in range(ntiles)]
    owners = [[int(t) for t in partition(ntiles, r, nranks)]
              for r in range(nranks)]
    mine = owners[rank]
    targets = [summed_seeds(t) if isinstance(t, (list, tuple)) else t
               for t in targets]
    ntargets = len(targets)
    held = dict.fromkeys((
        'metric_lu', 'X_rows', 'B_rows', 'aux_rows', 'X_ext', 'stream_tile',
        'S_rows', 'panel', 'metric_root', 'metric_slab', 'MT_bar_rows',
        'Q_bar_rows', 'Q_rows', 'U_rows', 'MT_rows', 'X_bar_rows',
        'B_bar_rows', 'P_bar_rows', 'shell_block', 'F_block', 'root_gather',
        'column_tile', 'g_bar_batch', 'derivative_block', 'root_adjoint',
        'two_centre_slab'), 0)

    def hold(name, nbytes):
        held[name] = max(held[name], int(nbytes))

    def total(per_target):
        return sum(total_bytes(a.values()) for a in per_target)

    def rows_per_target(ncol):
        return [{t: np.zeros((size_of[t], ncol)) for t in mine}
                for _ in range(ntargets)]

    l_ao = ao_l_labels(mol)
    second = np.where(l_ao <= l_max_second)[0]
    w = np.array([ANGULAR_WEIGHTS.get(l_ao[j], 1.0) for j in second])
    n2 = len(second)
    lu = scipy.linalg.lu_factor(auxmol.intor('int2c2e', aosym='s1'))
    if nranks > 1:
        lu = lockstep(lu, comm, check=True)
    hold('metric_lu', lu[0].nbytes)
    frozen = frozen_columns(layout, second, w, nao)
    ao_loc = mol.ao_loc_nr()
    blocks = ao_blocks(mol, nk, n2, naux, block_memory_gb)
    width = [ao_loc[s1] - ao_loc[s0] for s0, s1 in blocks]
    order = sorted(range(len(blocks)), key=lambda ib: (-width[ib], ib))
    size_of = {t: tiles[t][1] - tiles[t][0] for t in range(ntiles)}

    # the forward's Gram tiles; B and P re-evaluated where next read, the
    # same calls, so they are not held through the pass
    S, d, X_own, B_own, P_own, _, _ = gram_tiles(
        mol, auxmol, coords, tiles, owners, rank, second, w, regularization,
        (0, 0), comm, hold)
    del B_own, P_own

    d_tiles = [_adjoint_tiles(s.d_bar, tiles, mine, block, nk)
               for s in targets]
    x_tiles = [_adjoint_tiles(s.x_bar, tiles, mine, block, nk)
               for s in targets]
    rooted = [g for g, s in enumerate(targets) if s.d_bar is not None]
    g_tiles = [[_adjoint_tiles(q.d_bar, tiles, mine, block, nk)
                for q in s.gauges] for s in targets]
    # an adjoint arriving on M^T alone needs no root, and nothing reaches it
    MTb = [{t: s.mt_bar[t] for t in mine} if s.d_bar is None and not s.gauges
           else {t: np.empty((size_of[t], naux)) for t in mine}
           if s.d_bar is not None
           else {t: np.zeros((size_of[t], naux)) for t in mine}
           for s in targets]
    if rooted:
        # MT_bar = D_bar V^1/2, the root on rank 0 and streamed in slabs
        def assign(c0, c1, slab):
            for g in rooted:
                for t in mine:
                    MTb[g][t][:, c0:c1] = d_tiles[g][t] @ slab.T

        _streamed_root(auxmol, None, block, rank, comm, hold, assign)
        for g in rooted:
            if targets[g].mt_bar is not None:
                for t in mine:
                    MTb[g][t] += targets[g].mt_bar[t]
    for g, s in enumerate(targets):
        if s.d_bar is None and s.gauges and s.mt_bar is not None:
            for t in mine:
                MTb[g][t] += s.mt_bar[t]
    # each dressed gauge adds D_bar_g R_g, R_g = (V + vtilde_g)^1/2, one root
    # per environment, after the bare gauge
    for env in _distinct(q.environment for s in targets for q in s.gauges):
        pairs = [(g, j) for g, s in enumerate(targets)
                 for j, q in enumerate(s.gauges) if q.environment is env]

        def add(c0, c1, slab, pairs=pairs):
            for g, j in pairs:
                for t in mine:
                    MTb[g][t][:, c0:c1] += g_tiles[g][j][t] @ slab.T

        _streamed_root(auxmol, env, block, rank, comm, hold, add)
    hold('MT_bar_rows', total(MTb))

    # W = G^-1 d MT_bar, the forward substitution in the factorization sweep
    Qb = [{t: mtb[t] * d[slice(*tiles[t]), None] for t in mine}
          for mtb in MTb]
    hold('Q_bar_rows', total(Qb))
    diagonal, panels = _cholesky_tiles(S, tiles, owners, rank, comm, Qb,
                                       regularization, hold)
    del S
    for qb in Qb:
        _backward_tiles(diagonal, panels, qb, tiles, owners, rank, comm)
        for t in mine:
            qb[t] *= d[slice(*tiles[t]), None]
    # the three-centre pass: Q as the forward accumulates it, and the
    # adjoint of its pair columns, Q_bar F_b, onto the AO collocation
    Q = {t: np.zeros((size_of[t], naux)) for t in mine}
    hold('Q_rows', total_bytes(Q.values()))
    aob = rows_per_target(nao)
    hold('X_bar_rows', total(aob))
    integrals = KeptIntegrals(mol, auxmol)
    for c0 in range(0, len(order), nranks):
        batch = order[c0:c0 + nranks]
        spans = [(ao_loc[blocks[ib][0]], ao_loc[blocks[ib][1]])
                 for ib in batch]
        keeps = [frozen[a0 * n2:a1 * n2] for a0, a1 in spans]
        F_own = None
        if rank < len(batch) and keeps[rank].any():
            F_own = block_coefficients(integrals, lu, blocks[batch[rank]],
                                        spans[rank][0], keeps[rank], second,
                                        w, hold)
        for pos, keep in enumerate(keeps):
            kept = np.flatnonzero(keep)
            if not len(kept):
                continue
            Ft = F_own if pos == rank else np.empty((len(kept), naux))
            hold('F_block', Ft.nbytes + (0 if F_own is None or pos == rank
                                         else F_own.nbytes))
            broadcast_rows(Ft, pos, comm)
            mu = spans[pos][0] + kept // n2
            nu = second[kept % n2]
            wk = w[kept % n2]
            for i in mine:
                D_t = X_own[i][:, mu] * X_own[i][:, nu]
                D_t *= wk[None, :]
                Q[i] += D_t @ Ft
                del D_t
                for qb, ob in zip(Qb, aob):
                    E = qb[i] @ Ft.T
                    E *= wk[None, :]
                    _scatter_pairs(ob[i], E, X_own[i], mu, nu)
                    del E
            del Ft
        F_own = None
    del integrals

    # U = Q_bar V^-1, the grid sum (Q - P)^T U on rank 0, then R = d Q
    U = [{t: np.ascontiguousarray(scipy.linalg.lu_solve(lu, qb[t].T).T)
          for t in mine} for qb in Qb]
    hold('U_rows', total(U))
    del lu
    VbarF = [np.zeros((naux, naux)) if rank == 0 else None
             for _ in range(ntargets)]
    for u, vbar in zip(U, VbarF):
        for t, got in tiles_at_root([Q, u], tiles, rank, nranks, comm, hold):
            if got is not None:
                vbar -= got[0].T @ got[1]
    P_own = {t: tile_collocation(auxmol, coords[slice(*tiles[t])])
             for t in mine}
    d6 = [{} for _ in range(ntargets)]
    for t in mine:
        Q[t] += P_own[t]                 # the auxiliary block, F = identity
        for qb, d6_g in zip(Qb, d6):
            # W . Q with W = Q_bar / d
            d6_g[t] = np.einsum('kb,kb->k', qb[t], Q[t]) / d[slice(*tiles[t])]
        Q[t] *= d[slice(*tiles[t]), None]

    # Z = G^-1 R on the kept factor, M^T = d Z: the forward's M^T
    _forward_tiles(diagonal, panels, Q, tiles, owners, rank, comm, hold)
    _backward_tiles(diagonal, panels, Q, tiles, owners, rank, comm,
                    keep=False)
    del diagonal, panels
    d8 = [{} for _ in range(ntargets)]
    for t in mine:
        for mtb, d8_g in zip(MTb, d8):
            d8_g[t] = np.einsum('kb,kb->k', mtb[t], Q[t])
        Q[t] *= d[slice(*tiles[t]), None]
    MT = Q
    del MTb, Q
    hold('MT_rows', total_bytes(MT.values()))
    Vhbar = [None] * ntargets
    for g in rooted:
        Vhbar[g] = np.zeros((naux, naux)) if rank == 0 else None
        d_bar = targets[g].d_bar
        rows_in = isinstance(d_bar, GridTileRows)
        for t, got in tiles_at_root([MT, d_tiles[g]] if rows_in else [MT],
                                     tiles, rank, nranks, comm, hold):
            if got is not None:
                Vhbar[g] += got[0].T @ (got[1] if rows_in
                                        else d_bar[slice(*tiles[t])])
    # each dressed gauge's root adjoint M D_bar_g, the same sum on rank 0
    Rbar = [[None] * len(s.gauges) for s in targets]
    for g, s in enumerate(targets):
        for j, q in enumerate(s.gauges):
            Rbar[g][j] = np.zeros((naux, naux)) if rank == 0 else None
            rows_in = isinstance(q.d_bar, GridTileRows)
            for t, got in tiles_at_root([MT, g_tiles[g][j]] if rows_in
                                         else [MT], tiles, rank, nranks,
                                         comm, hold):
                if got is not None:
                    Rbar[g][j] += got[0].T @ (got[1] if rows_in
                                              else q.d_bar[slice(*tiles[t])])

    # the (mu nu|P) adjoints, w U^T D_pair, by the block's owner
    three = _three_centre_rows_adjoint(
        mol, auxmol, coords, U, X_own, tiles, owners, rank, comm, blocks,
        order, frozen, second, w, block_memory_gb * BYTES_PER_GB, hold)
    del U

    # the Gram matrix's adjoint, H = Q_bar M + M^T Q_bar^T, by row tile
    B_own = {t: X_own[t][:, second] * w[None, :] for t in mine}
    Xb = rows_per_target(nao)
    Bb = rows_per_target(n2)
    Pb = rows_per_target(naux)
    dg = [{t: np.zeros(size_of[t]) for t in mine} for _ in range(ntargets)]
    hold('B_bar_rows', total(Bb))
    hold('P_bar_rows', total(Pb))
    for k, (k0, k1) in enumerate(tiles):
        owner = k % nranks
        # each target's [M^T | Q_bar] tile, as its own pass broadcasts it
        boths = []
        for qb in Qb:
            both = (np.hstack([MT[k], qb[k]]) if rank == owner
                    else np.empty((k1 - k0, 2 * naux)))
            broadcast_rows(both, owner, comm)
            hold('column_tile', both.nbytes)
            boths.append(both)
        if k in X_own:
            X_k, B_k, P_k = X_own[k], B_own[k], P_own[k]
        else:
            X_k = tile_collocation(mol, coords[k0:k1])
            B_k = X_k[:, second] * w[None, :]
            P_k = tile_collocation(auxmol, coords[k0:k1])
            hold('stream_tile', total_bytes((X_k, B_k, P_k)))
        for i in mine:
            XX = X_own[i] @ X_k.T
            BB = B_own[i] @ B_k.T
            S_ik = None
            for g, both in enumerate(boths):
                MT_k, Qb_k = both[:, :naux], both[:, naux:]
                H = Qb[g][i] @ MT_k.T
                H += MT[i] @ Qb_k.T
                if S_ik is None:
                    S_ik = XX * BB
                    S_ik += P_own[i] @ P_k.T
                dg[g][i] -= np.einsum('kl,kl->k', H, S_ik)
                Xb[g][i] -= (BB * H) @ X_k
                Bb[g][i] -= (XX * H) @ B_k
                Pb[g][i] -= H @ P_k
                del H
            del XX, BB, S_ik
        del boths, X_k, B_k, P_k
    # the balancing s = diag(S)^1/2, d = 1/s: its adjoint on diag(S)
    for g in range(ntargets):
        for t in mine:
            rows = slice(*tiles[t])
            dbar = d8[g][t] + d6[g][t] + dg[g][t] / d[rows]
            c = -dbar * d[rows] ** 3     # s_bar / s, s_bar = -d_bar d^2
            Xb[g][t] += (c * np.einsum('kj,kj->k', B_own[t], B_own[t]))[
                :, None] * X_own[t]
            Bb[g][t] += (c * np.einsum('km,km->k', X_own[t], X_own[t]))[
                :, None] * B_own[t]
            Pb[g][t] += c[:, None] * P_own[t]
            Pb[g][t] += Qb[g][t]         # Q's auxiliary block
            aob[g][t] += Xb[g][t]
            aob[g][t][:, second] += Bb[g][t] * w[None, :]
    del Xb, Bb, d6, d8, dg, MT, Qb, B_own

    # the collocations' centres and points, one tile at a time
    ao_slices = [(p0, p1) for _, _, p0, p1 in mol.aoslice_by_atom()]
    aux_slices = [(q0, q1) for _, _, q0, q1 in auxmol.aoslice_by_atom()]
    fit_tile = [np.zeros((ntiles, natm * 3)) for _ in range(ntargets)]
    fit_points = [np.zeros((nk, 3)) for _ in range(ntargets)]
    with_coll = [s.x_bar is not None or s.x_bar_ao is not None
                 for s in targets]
    coll_tile = [np.zeros((ntiles, natm * 3)) if c else None
                 for c in with_coll]
    coll_points = [np.zeros((nk, 3)) if c else None for c in with_coll]
    for t in mine:
        rows = slice(*tiles[t])
        g_ao = mol.eval_gto('GTOval_ip_sph', coords[rows])
        f = []
        for g, s in enumerate(targets):
            f.append(_centre_forces(g_ao, aob[g][t], ao_slices))
            fit_points[g][rows] = np.einsum('xgm,gm->gx', g_ao, aob[g][t])
            if with_coll[g]:
                xa = _collocation_rows(s, x_tiles[g], t)
                coll_tile[g][t] = _centre_forces(g_ao, xa, ao_slices).ravel()
                coll_points[g][rows] = np.einsum('xgm,gm->gx', g_ao, xa)
                del xa
        del g_ao
        g_aux = auxmol.eval_gto('GTOval_ip_sph', coords[rows])
        for g in range(ntargets):
            f[g] += _centre_forces(g_aux, Pb[g][t], aux_slices)
            fit_points[g][rows] += np.einsum('xgm,gm->gx', g_aux, Pb[g][t])
            fit_tile[g][t] = f[g].ravel()
        del g_aux, f
    del aob, Pb, X_own, P_own
    tile_ranges = [[(t, t + 1) for t in own] for own in owners]
    point_ranges = [[tiles[t] for t in own] for own in owners]
    for g in range(ntargets):
        allgather_ranges(fit_tile[g], tile_ranges, comm)
        allgather_ranges(fit_points[g], point_ranges, comm)
        if with_coll[g]:
            allgather_ranges(coll_tile[g], tile_ranges, comm)
            allgather_ranges(coll_points[g], point_ranges, comm)

    # the metric: (V^1/2)'s adjoint on rank 0, (P|Q)' by slab; each dressed
    # gauge's root adjoint and its cavity derivative on rank 0
    V_bar = [None] * ntargets
    cavity = [np.zeros(natm * 3) for _ in range(ntargets)]
    if rank == 0:
        for g, s in enumerate(targets):
            sums = {'root': Vhbar[g], 'fit': VbarF[g]}
            Vhbar[g] = VbarF[g] = None
            V_bar[g] = (sums.pop('fit') if sums['root'] is None
                        else _metric_root_adjoint(auxmol, sums, hold))
            for j, q in enumerate(s.gauges):
                # Y_g on V + vtilde_g reaches V and vtilde_g alike
                rbar, Rbar[g][j] = Rbar[g][j], None
                if q.root_bar is not None:
                    rbar += q.root_bar
                Y = _dressed_root_adjoint(auxmol, q.environment, rbar, hold)
                del rbar
                V_bar[g] += Y
                if q.kernel_bar is not None:
                    Y += q.kernel_bar
                cavity[g] += np.asarray(q.environment.aux_kernel_adjoint(
                    auxmol, Y)).ravel()
                del Y
            if s.metric_bar is not None:
                V_bar[g] += s.metric_bar
    del Vhbar, VbarF, Rbar
    two = _two_centre_rows_adjoints(auxmol, V_bar, block, rank, nranks, comm,
                                    hold)
    del V_bar
    if any(s.gauges for s in targets):
        cavity = broadcast(cavity, comm)

    out = []
    for g in range(ntargets):
        fit_centre = np.zeros(natm * 3)
        for t in range(ntiles):
            fit_centre += fit_tile[g][t]
        fit_centre += three[g]
        fit_centre += two[g]
        if targets[g].gauges:
            fit_centre += cavity[g]
        coll_centre = None
        if with_coll[g]:
            coll_centre = np.zeros(natm * 3)
            for t in range(ntiles):
                coll_centre += coll_tile[g][t]
            coll_centre = coll_centre.reshape(natm, 3)
        out.append(RowFitAdjoint(fit_centre.reshape(natm, 3), fit_points[g],
                                 coll_centre, coll_points[g], dict(held)))
    return out


def summed_seeds(seeds):
    """One `AdjointSeeds` holding the sum of `seeds` kind by kind (D_bar,
    X_bar, MT_bar, the AO collocation's and the metric's adjoints), in the
    order given: the first present copied, each next one added onto it;
    dressed gauges, each on its own root, are listed side by side.
    The fit adjoint is linear in every seed, so its adjoint is theirs summed;
    an adjoint on X_mo joins only others on the same orbitals. Tiles are this
    rank's, metric_bar rank 0's (None elsewhere)."""
    seeds = list(seeds)
    if len(seeds) == 1:
        return seeds[0]                  # nothing to add, nothing copied
    orbitals = [s.mo_coeff for s in seeds if s.x_bar is not None]
    if any(not np.array_equal(c, orbitals[0]) for c in orbitals[1:]):
        raise ValueError('adjoints on X_mo of different orbitals do not add '
                         'into one X_bar C^T')

    def added(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        first = values[0]
        if isinstance(first, dict):
            out = {t: a.copy() for t, a in first.items()}
            for v in values[1:]:
                for t in out:
                    out[t] += v[t]
            return out
        if isinstance(first, GridTileRows):
            out = GridTileRows(first.rows.copy(), first.npts, first.block,
                               first.comm)
        else:
            out = np.array(first, copy=True)
        for v in values[1:]:
            out += v
        return out

    return AdjointSeeds(
        d_bar=added([s.d_bar for s in seeds]),
        x_bar=added([s.x_bar for s in seeds]),
        mo_coeff=orbitals[0] if orbitals else None,
        mt_bar=added([s.mt_bar for s in seeds]),
        x_bar_ao=added([s.x_bar_ao for s in seeds]),
        metric_bar=added([s.metric_bar for s in seeds]),
        gauges=[q for s in seeds for q in s.gauges])


def _collocation_rows(seed, x_tiles, t):
    """Tile t of the AO collocation adjoint of `seed`: X_bar C^T (+ its
    x_bar_ao), or its x_bar_ao alone."""
    if seed.x_bar is None:
        return seed.x_bar_ao[t]
    xa = x_tiles[t] @ seed.mo_coeff.T
    if seed.x_bar_ao is not None:
        xa += seed.x_bar_ao[t]
    return xa


def _adjoint_tiles(bar, tiles, mine, block, nk):
    """{t: rows} of this rank's tiles of an adjoint handed in whole or as
    `GridTileRows` of the same tiles (None stays None), views either way."""
    if bar is None:
        return None
    if isinstance(bar, GridTileRows):
        if bar.block != block or bar.npts != nk or bar.mine != list(mine):
            raise ValueError(f'the adjoint is held in tiles of {bar.block} of '
                             f'{bar.npts} points over {bar.size} ranks, the '
                             f'fit in tiles of {block} of {nk} over another '
                             f'layout')
        return bar.tiles()
    return {t: bar[slice(*tiles[t])] for t in mine}


def _cholesky_tiles(S, tiles, owners, rank, comm, rhs_sets, regularization,
                    hold):
    """(diagonal, panels): `fit_rows`' right-looking Cholesky of the Gram
    tiles S (consumed), the forward substitution of each right-hand side of
    `rhs_sets` riding in the sweep as it does there, and the factor kept for
    later solves: the diagonal block and the gathered panel below it of
    every tile this rank owns, the same arrays and the same calls as that
    sweep's."""
    nk, nranks = tiles[-1][1], len(owners)
    mine = owners[rank]
    ncols = [broadcast(next(iter(rhs.values())).shape[1] if rhs else 0, comm)
             for rhs in rhs_sets]
    diagonal, panels = {}, {}
    for j, (j0, j1) in enumerate(tiles):
        owner = j % nranks
        LT = np.empty((j1 - j0, j1 - j0))
        info = 0
        if rank == owner:
            L, info = scipy.linalg.lapack.dpotrf(S.pop(j), lower=1, clean=1)
            LT = L.T
        info = broadcast(info, comm, root=owner)
        if info != 0:
            raise np.linalg.LinAlgError(
                f'Cholesky of the balanced Gram matrix failed at leading '
                f'minor {j0 + info}: it is not positive definite, which for a '
                f'Gram matrix plus a {regularization:g} shift means the grid '
                'is degenerate.')
        broadcast_rows(LT, owner, comm)
        L = LT.T
        ys = []
        for rhs, ncol in zip(rhs_sets, ncols):
            y = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
            if rank == owner:
                trsm(L, y.T, side=1, trans_a=1)
            broadcast_rows(y, owner, comm)
            ys.append(y)
        if rank == owner:
            diagonal[j] = L
        below = [i for i in mine if i > j]
        Lcol = {}
        for i in below:
            Lcol[i] = np.ascontiguousarray(S[i][:, :j1 - j0])
            trsm(L, Lcol[i].T, side=0, trans_a=0)
        for rhs, y in zip(rhs_sets, ys):
            for i in below:
                rhs[i] -= Lcol[i] @ y
        del ys
        panel = np.empty((nk - j1, j1 - j0))
        for i in below:
            panel[tiles[i][0] - j1:tiles[i][1] - j1] = Lcol[i]
        allgather_ranges(panel, [[(tiles[i][0] - j1, tiles[i][1] - j1)
                                  for i in own if i > j] for own in owners],
                         comm)
        for i in below:
            U = Lcol.pop(i) @ panel[:tiles[i][1] - j1].T
            S[i] = S[i][:, j1 - j0:] - U
            del U
        if rank == owner:
            panels[j] = panel
        hold('panel', panel.nbytes)
        hold('S_rows', total_bytes(S.values()) + total_bytes(panels.values())
             + total_bytes(diagonal.values()))
        del panel
    return diagonal, panels


def _forward_tiles(diagonal, panels, rhs, tiles, owners, rank, comm, hold):
    """rhs <- L^-1 rhs on the kept factor: the owner of tile j solves its
    block and broadcasts it with its panel, and every rank updates its tiles
    below with the call `_cholesky_tiles` makes there."""
    nk, nranks = tiles[-1][1], len(owners)
    mine = owners[rank]
    ncol = next(iter(rhs.values())).shape[1] if rhs else 0
    ncol = broadcast(ncol, comm)
    for j, (j0, j1) in enumerate(tiles):
        owner = j % nranks
        y = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
        if rank == owner:
            trsm(diagonal[j], y.T, side=1, trans_a=1)
        broadcast_rows(y, owner, comm)
        panel = panels[j] if rank == owner else np.empty((nk - j1, j1 - j0))
        broadcast_rows(panel, owner, comm)
        hold('panel', panel.nbytes)
        for i in mine:
            if i > j:
                rhs[i] -= panel[tiles[i][0] - j1:tiles[i][1] - j1] @ y
        del y, panel


def _backward_tiles(diagonal, panels, rhs, tiles, owners, rank, comm,
                    keep=True):
    """rhs <- L^-T rhs on the kept factor, `fit_rows`' backward sweep: each
    solved block broadcast by its owner and pushed into the rows that own a
    panel. keep=False drops the factor as it goes, as that sweep does."""
    nranks = len(owners)
    mine = owners[rank]
    ncol = next(iter(rhs.values())).shape[1] if rhs else 0
    ncol = broadcast(ncol, comm)
    for j in reversed(range(len(tiles))):
        j0, j1 = tiles[j]
        owner = j % nranks
        z = rhs[j] if rank == owner else np.empty((j1 - j0, ncol))
        if rank == owner:
            trsm(diagonal[j] if keep else diagonal.pop(j), z.T, side=1,
                  trans_a=0)
            if not keep:
                panels.pop(j, None)
        broadcast_rows(z, owner, comm)
        for i in mine:
            if i < j:
                i1 = tiles[i][1]
                rhs[i] -= panels[i][j0 - i1:j1 - i1].T @ z
        del z


def tiles_at_root(arrays, tiles, rank, nranks, comm, hold):
    """Yield (t, [a[t] for a in arrays]) for every tile in order, on rank 0
    (None elsewhere): the owner sends its tile's rows to rank 0 alone, one
    `exchange_blocks` per tile, verbatim, so rank 0 holds one tile at a
    time."""
    ncol = [a[next(iter(a))].shape[1] if a else 0 for a in arrays]
    ncol = broadcast(ncol, comm)
    width = sum(ncol)
    for t, (t0, t1) in enumerate(tiles):
        owner = t % nranks
        send = [np.empty((0, width)) for _ in range(nranks)]
        if rank == owner:
            send[0] = np.hstack([a[t] for a in arrays])
        shapes = [(0, width)] * nranks
        if rank == 0:
            shapes[owner] = (t1 - t0, width)
        got = exchange_blocks(send, shapes, comm)[owner if rank == 0 else 0]
        hold('root_gather', got.nbytes if rank == 0 else 0)
        del send
        if rank != 0:
            yield t, None
            continue
        pieces, at = [], 0
        for n in ncol:
            pieces.append(got[:, at:at + n])
            at += n
        yield t, pieces


def _scatter_pairs(out, E, X, mu, nu):
    """out[:, m] += sum over the pairs (m, n) of E X[:, n], and
    out[:, n] += E X[:, m]: an adjoint E on the pair columns
    X[:, mu] X[:, nu] onto the AO collocation, one mu's run at a time."""
    starts = np.flatnonzero(np.r_[True, mu[1:] != mu[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(mu)]):
        m, nus = mu[a], nu[a:b]
        out[:, m] += np.einsum('kp,kp->k', E[:, a:b], X[:, nus])
        out[:, nus] += E[:, a:b] * X[:, m][:, None]


def _centre_forces(g, bar, slices):
    """(natm, 3) of sum_{k mu} bar[k, mu] chi_mu(r_k) through the centres:
    g = GTOval_ip = +d chi/dr, whose nuclear derivative is its negative."""
    f = np.zeros((len(slices), 3))
    for ia, (p0, p1) in enumerate(slices):
        f[ia] = -np.einsum('xgm,gm->x', g[:, :, p0:p1], bar[:, p0:p1])
    return f


def _three_centre_rows_adjoint(mol, auxmol, coords, U, X_own, tiles, owners,
                               rank, comm, blocks, order, frozen, second, w,
                               budget, hold):
    """[(natm * 3,)] of sum_p sum_P g[p, P] d(mu_p nu_p|P)/dR per target,
    with g_p = w_p U^T D[:, p] the adjoint on the kept pairs' integrals and
    U a list of each target's tiles of U.

    Block order[c] belongs to rank c % size, as in the forward pass. Its
    owner accumulates every target's g over that target's U streamed tile
    by tile in grid order, in batches of consecutive rounds whose
    accumulators stay within `budget` bytes on every rank (the batch does
    not enter a block's sum, which runs over every tile in order whatever
    the batch), then evaluates the kept pairs' derivative integrals
    (`KeptIntegrals` of int3c2e_ip1 and _ip2) once and contracts them with
    each target's g. The nu centre's derivative is -(the mu and P
    centres'), by translational invariance of (mu nu|P), and the integrals
    are evaluated in chunks of pairs holding `THREE_CENTER_BLOCK_BYTES` of
    each. Each block's (natm, 3) is gathered and the blocks are added in
    `order`."""
    nranks, n2 = len(owners), len(second)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    ntargets = len(U)
    ao_loc = mol.ao_loc_nr()
    ao_atom = np.empty(nao, dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[p0:p1] = ia
    aux_atom = np.empty(naux, dtype=int)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        aux_atom[q0:q1] = ia
    pairs = {}
    for ib, (s0, s1) in enumerate(blocks):
        a0, a1 = ao_loc[s0], ao_loc[s1]
        kept = np.flatnonzero(frozen[a0 * n2:a1 * n2])
        pairs[ib] = (a0 + kept // n2, second[kept % n2], w[kept % n2])
    batches, current, load = [], [], np.zeros(nranks)
    for c0 in range(0, len(order), nranks):
        add = np.zeros(nranks)
        for pos, ib in enumerate(order[c0:c0 + nranks]):
            add[pos] = len(pairs[ib][0]) * naux * 8 * ntargets
        if current and np.any(load + add > budget):
            batches.append(current)
            current, load = [], np.zeros(nranks)
        current.extend(order[c0:c0 + nranks])
        load += add
    if current:
        batches.append(current)
    owner_of = {ib: c % nranks for c, ib in enumerate(order)}
    # pairs per derivative evaluation: three components of (kept, naux)
    chunk = max(1, THREE_CENTER_BLOCK_BYTES // (3 * naux * 8))
    first = KeptIntegrals(mol, auxmol, 'int3c2e_ip1', comp=3)
    other = KeptIntegrals(mol, auxmol, 'int3c2e_ip2', comp=3)
    forces = [np.zeros((len(blocks), natm * 3)) for _ in U]
    for batch in batches:
        own = [ib for ib in batch
               if owner_of[ib] == rank and len(pairs[ib][0])]
        acc = [{ib: np.zeros((len(pairs[ib][0]), naux)) for ib in own}
               for _ in U]
        hold('g_bar_batch', sum(total_bytes(a.values()) for a in acc))
        for t, (t0, t1) in enumerate(tiles):
            owner = t % nranks
            U_t = []
            for u in U:
                u_t = u[t] if rank == owner else np.empty((t1 - t0, naux))
                broadcast_rows(u_t, owner, comm)
                U_t.append(u_t)
            if not own:
                continue
            if t in X_own:
                X_t = X_own[t]
            else:
                X_t = tile_collocation(mol, coords[t0:t1])
                hold('stream_tile', X_t.nbytes)
            for ib in own:
                mu, nu, wk = pairs[ib]
                D_t = X_t[:, mu] * X_t[:, nu]
                D_t *= wk[None, :]
                for a, u_t in zip(acc, U_t):
                    a[ib] += D_t.T @ u_t
                del D_t
            del X_t, U_t
        for ib in own:
            mu, nu, wk = pairs[ib]
            gs = []
            for a in acc:
                g = a.pop(ib)
                g *= wk[:, None]
                gs.append(g)
            fs = [np.zeros((natm, 3)) for _ in U]
            for p0 in range(0, len(mu), chunk):
                c = slice(p0, min(p0 + chunk, len(mu)))
                I, call = first(blocks[ib], mu[c], nu[c])
                hold('derivative_block', I.nbytes + call)
                t_mu = [np.einsum('xpP,pP->xp', I, g[c])    # -d/dR_mu
                        for g in gs]
                del I
                I, call = other(blocks[ib], mu[c], nu[c])
                hold('derivative_block', I.nbytes + call)
                for g, f, t_m in zip(gs, fs, t_mu):
                    t_aux = np.einsum('xpP,pP->xP', I, g[c])   # -d/dR_P
                    t_nu = t_m + np.einsum('xpP,pP->xp', I, g[c])
                    np.add.at(f, ao_atom[mu[c]], -t_m.T)
                    np.add.at(f, ao_atom[nu[c]], t_nu.T)
                    np.add.at(f, aux_atom, -t_aux.T)
                del I, t_mu
            del gs
            for out, f in zip(forces, fs):
                out[ib] = f.ravel()
    totals = []
    for out in forces:
        allgather_ranges(out, [[(ib, ib + 1) for ib in order
                                if owner_of[ib] == r] for r in range(nranks)],
                         comm)
        total = np.zeros(natm * 3)
        for ib in order:
            total += out[ib]
        totals.append(total)
    return totals


def two_centre_rows_adjoint(auxmol, V_bar, block, rank, nranks, comm,
                             hold, omega=0.0):
    """(natm * 3,) of sum_PQ V_bar[P, Q] d(P|Q)/dR, V_bar held on rank 0
    alone (None elsewhere): `_two_centre_rows_adjoints` of one."""
    return _two_centre_rows_adjoints(auxmol, [V_bar], block, rank, nranks,
                                     comm, hold, omega=omega)[0]


def _two_centre_rows_adjoints(auxmol, V_bars, block, rank, nranks, comm,
                              hold, omega=0.0):
    """[(natm * 3,)] of sum_PQ V_bar[P, Q] d(P|Q)/dR for each of `V_bars`,
    held on rank 0 alone (None elsewhere). Each symmetric part travels in
    slabs of aux shells of at most `block` functions, slab s to rank
    s % size, which evaluates that slab's (grad P|Q) once and contracts it
    with every target's slab; the slabs' (natm, 3) are gathered and added
    in slab order. omega != 0 takes the erf-attenuated operator's
    integrals."""
    naux, natm = auxmol.nao_nr(), auxmol.natm
    aux_loc = auxmol.ao_loc_nr()
    aux_atom = np.empty(naux, dtype=int)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        aux_atom[q0:q1] = ia
    slabs, sh0 = [], 0
    while sh0 < auxmol.nbas:
        sh1 = sh0 + 1
        while (sh1 < auxmol.nbas
               and int(aux_loc[sh1 + 1] - aux_loc[sh0]) <= block):
            sh1 += 1
        slabs.append((sh0, sh1))
        sh0 = sh1
    forces = [np.zeros((len(slabs), natm * 3)) for _ in V_bars]
    for s, (sh0, sh1) in enumerate(slabs):
        p0, p1 = aux_loc[sh0], aux_loc[sh1]
        parts = []
        for V_bar in V_bars:
            slab = (np.ascontiguousarray(0.5 * (V_bar[p0:p1]
                                                + V_bar[:, p0:p1].T))
                    if rank == 0 else np.empty((p1 - p0, naux)))
            broadcast_rows(slab, 0, comm)
            parts.append(slab)
        if rank == s % nranks:
            with auxmol.with_range_coulomb(omega):
                v1 = auxmol.intor('int2c2e_ip1', comp=3,
                                  shls_slice=(sh0, sh1, 0, auxmol.nbas))
            for out, slab in zip(forces, parts):
                hold('two_centre_slab', slab.nbytes + v1.nbytes)
                # +(grad P|Q), so the nuclear derivative carries the minus
                t_P = -2.0 * np.einsum('xPQ,PQ->xP', v1, slab)
                f = np.zeros((natm, 3))
                np.add.at(f, aux_atom[p0:p1], t_P.T)
                out[s] = f.ravel()
            del v1
        del parts
    totals = []
    for out in forces:
        allgather_ranges(out, [[(s, s + 1) for s in range(len(slabs))
                                if s % nranks == r] for r in range(nranks)],
                         comm)
        total = np.zeros(natm * 3)
        for s in range(len(slabs)):
            total += out[s]
        totals.append(total)
    return totals


def _metric_root_adjoint(auxmol, sums, hold):
    """V_bar = (V^1/2)'s adjoint of sums['root'] plus sums['fit'], both
    (naux, naux) and taken out of `sums` so that each is dropped as soon as
    it is read: at most four metric-sized arrays at once.

    The Frechet derivative of the root, `metric_root`'s, is the Hadamard
    multiplier 1 / (w_i^1/2 + w_j^1/2) in V's eigenbasis, self-adjoint,
    over the spectrum `metric_spectrum` keeps.
    """
    V = np.asfortranarray(auxmol.intor('int2c2e', aosym='s1'))
    w, Z = scipy.linalg.eigh(V, lower=True, overwrite_a=True,
                             check_finite=False, driver='evr')
    hold('root_adjoint', 4 * Z.nbytes)
    del V
    keep = metric_spectrum(w, False)
    root = np.where(keep, np.sqrt(np.where(keep, w, 0.0)), np.inf)
    E = Z.T @ sums.pop('root')
    E = E @ Z
    E += E.T
    E *= 0.5
    E /= root[:, None] + root[None, :]    # a dropped direction carries none
    Y = Z @ E
    del E
    V_bar = Y @ Z.T
    del Y, Z
    V_bar += sums.pop('fit')
    return V_bar


def _dressed_root_adjoint(auxmol, environment, root_bar, hold):
    """Y = ((V + vtilde)^1/2)'s adjoint of `root_bar`, (naux, naux), the
    root `metric_root(auxmol, environment)` forms: the Hadamard multiplier
    1 / (w_i^1/2 + w_j^1/2) in the dressed metric's eigenbasis, over the
    spectrum `metric_spectrum` keeps. Y is the adjoint on V and on vtilde
    alike."""
    V = np.asfortranarray(auxmol.intor('int2c2e', aosym='s1'))
    V += environment.aux_kernel(auxmol)
    w, Z = scipy.linalg.eigh(V, lower=True, overwrite_a=True,
                             check_finite=False, driver='evr')
    hold('root_adjoint', 4 * Z.nbytes)
    del V
    keep = metric_spectrum(w, True)
    root = np.where(keep, np.sqrt(np.where(keep, w, 0.0)), np.inf)
    E = Z.T @ root_bar
    E = E @ Z
    E += E.T
    E *= 0.5
    E /= root[:, None] + root[None, :]    # a dropped direction carries none
    Y = Z @ E
    del E
    Y = Y @ Z.T
    return Y


def _streamed_root(auxmol, environment, block, rank, comm, hold, apply):
    """`apply(c0, c1, slab)` for every slab of `block` rows of the metric
    root (V + vtilde)^1/2 of `environment` (V^1/2 for None), the root
    formed on rank 0 alone (`metric_root`) and each slab sent from it."""
    naux = auxmol.nao_nr()
    root, failure = None, None
    if rank == 0:
        try:
            root, nbytes = metric_root(auxmol, environment)
            hold('metric_root', nbytes)
        except (RuntimeError, np.linalg.LinAlgError) as err:
            failure = str(err)
    failure = broadcast(failure, comm)
    if failure is not None:
        raise RuntimeError(failure)
    for c0 in range(0, naux, block):
        c1 = min(c0 + block, naux)
        slab = root_slab(root, slice(c0, c1), slice(0, naux), comm)
        hold('metric_slab', slab.nbytes)
        apply(c0, c1, slab)
    del root


def _distinct(items):
    """`items` with repeats of one object dropped, in first-seen order."""
    out = []
    for item in items:
        if not any(item is seen for seen in out):
            out.append(item)
    return out
