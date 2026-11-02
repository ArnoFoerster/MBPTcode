"""The nuclear adjoint of the AO-product Coulomb-metric ISDF fit
(`ao_product_fit`) on frozen points: the LS-THC fit of Parrish, Hohenstein,
Martinez and Sherrill (J. Chem. Phys. 137, 224106 (2012)) on points from
Matthews' pivoted Cholesky selection (J. Chem. Theory Comput. 16, 1382
(2020)), the selection held fixed so the force is the derivative of the
energy along a walk.

The forward fit, in the notation of `ao_product_fit`:

    A  = X X^T,  s = diag(A),  d = 1/s
    G  = (A o A) o (d d^T) + reg I
    E  = sum_mn (P|mn) X_gm X_gn,   F = V^-1 E
    B  = (F diag(d)) G^-1,          M = B diag(d)
    D_g = M^T R_g,  R_g = (V + vtilde_g)^1/2 for every gauge g

read backwards from adjoints D_bar_g on the factors:

    M_bar = sum_g R_g D_bar_g^T,  R_bar_g = M D_bar_g  (-> V, vtilde_g)
    B_bar = M_bar d,              H_bar = B_bar G^-1     (H = F d)
    G_bar = -B^T H_bar            low rank, the two solves made once
    F_bar = H_bar d,              E_bar = V^-1 F_bar,  V_bar -= E_bar F^T
    d_bar = sum_P (M_bar o B + H_bar o F) + ((G_bar + G_bar^T) o A o A) d
    X_bar = 2 ((G_bar + G_bar^T) o A o d d^T) X - 2 (d_bar d^2) o X
            + 2 sum_nP E_bar[P, g] (P|mn) X_gn
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn

Nothing of the test set's nao^2 width is formed: the Gram terms are (nk, nk)
products, formed by every rank, and E and its adjoint are carried by one pass
over the three-centre integrals and their derivatives, blocked over the
shells of the first AO, each rank on its own share of the blocks
(`ao_product_fit.own_runs`). (mn|P)_bar is symmetric in (m, n), as (mn|P) is, so the derivative
of the second AO's centre equals the first's and only int3c2e_ip1 and
int3c2e_ip2 of the block are needed. X_bar closes on the AO centres and,
through the frozen points carried with their atoms, on the point chain
(`isdf_derivatives.point_chain`).

The forward's arithmetic is repeated here rather than read from the fit:
the force is the derivative of the energy the fit gives, to the rounding of
the balanced Gram matrix.

THE ROW FIT. `ao_product_fit_rows_adjoints` is the same reverse pass for the
row fit (`ao_product_fit.ao_product_fit_rows`), in its tiles and under the
rules of the Duchemin-Blase row fit's adjoint (`isdf_fit_adjoint`): every
call's shape fixed by tile and run indices, nothing summed across ranks.
With W = G^-1 d MT_bar (the factor's sweep and a backward solve), the rows
E_bar^T = d W V^-1 and Z = G^-1 d F^T (M^T = d Z):

    d_bar  = MT_bar . Z + W . F^T + ((G_bar + G_bar^T) o A o A) d
    G_bar + G_bar^T = -(Z W^T + W Z^T)       by row tile, column tiles streamed
    X_bar  = 2 ((G_bar + G_bar^T) o A o d d^T) X - 2 (d_bar d^2) o X
             + 2 sum_nP (P|mn) X_gn E_bar[P, g]       riding the forward pass
    V_bar  = (V^1/2)'^T (M D_bar) - E_bar F^T         on rank 0, tile by tile
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn           by each run's owner,
                                                       E_bar streamed by tile

The seeds are `isdf_fit_adjoint.AdjointSeeds` and the result its
`RowFitAdjoint`, so a force pools the exchange skeletons' seeds into one call
as on the other row fit (`isdf_derivatives.FitKey`). Same bits at every rank
count, one rank included; the replicated pass's to the fit's rounding.
"""
import numpy as np
import scipy.linalg

from src.Base.factorization.ao_product_fit import (ao_product_projection,
                                                   balanced_gram,
                                                   half_transform_operand,
                                                   half_transformed,
                                                   own_runs, point_tiles,
                                                   product_gram_tiles,
                                                   projection_rows,
                                                   shell_runs,
                                                   three_centre_block)
from src.Base.factorization.separable_ri import (ao_values, aux_metric_sqrt,
                                                 backward_tiles,
                                                 cholesky_tiles, fit_settings,
                                                 forward_tiles,
                                                 tile_collocation,
                                                 total_bytes)
from src.Base.numerics.constants import BYTES_PER_GB, FIT_CHOLESKY_BLOCK
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, broadcast_rows,
                                              current_comm, lockstep,
                                              ordered_sum, partition,
                                              reduce_sum)
from src.gradients.isdf_fit_adjoint import (centre_forces, collocation_rows,
                                            adjoint_inputs, closed_adjoints,
                                            root_sums, summed_seeds,
                                            tiles_at_root)
from src.environments.environment import dresses_interaction
from src.Base.pyscf.pyscf_interface import auxmol_of
from src.gradients.isdf_derivatives import (GaugeAdjoint, PointChain,
                                            basis_centre_forces,
                                            isdf_exchange_adjoints,
                                            mean_field_point_chain,
                                            point_layout, sqrtm_adjoint,
                                            two_centre_adjoint)


def ao_product_fit_adjoint_gauges(mol, auxmol, coords, gauges, pts_local,
                                  atom_of_point, frames=None,
                                  with_frames=False, body=None,
                                  numerics=None):
    """(natm, 3) gradient of sum over `gauges` of sum_{gP} D_bar[g,P] D[g,P],
    D = M^T (V + vtilde)^1/2 with M the AO-product fit on `coords`.

    gauges: `GaugeAdjoint`s, or (D_bar, environment) pairs, as in
    `isdf_derivatives.dfactor_adjoint_gauges`: one fit, every gauge's metric
    root. The points are r_g = p_g F_i + R_i over the clouds `pts_local` and
    their owners `atom_of_point`; frozen frames turn with the body frame
    `body` (with_frames False), the chains' convention.
    """
    def seeds(X, M, V):
        M_bar = np.zeros_like(M)
        V_bar = np.zeros_like(V)
        env_grad = np.zeros((mol.natm, 3))
        for gauge in gauges:
            gauge = (gauge if isinstance(gauge, GaugeAdjoint)
                     else GaugeAdjoint(*gauge))
            environment = gauge.environment
            dressed = dresses_interaction(environment, auxmol)
            if not dressed and gauge.kernel_bar is not None:
                raise ValueError(
                    f'{environment!r} dresses nothing, so this gauge has no '
                    'vtilde for a kernel adjoint to land on; it belongs to '
                    'the gauge that carries the environment')
            V_gauge = V + environment.aux_kernel(auxmol) if dressed else V
            root = aux_metric_sqrt(auxmol, environment, V=V)
            M_bar += root @ gauge.d_bar.T
            root_bar = M @ gauge.d_bar
            if gauge.root_bar is not None:
                root_bar = root_bar + gauge.root_bar
            Y = sqrtm_adjoint(V_gauge, root_bar)
            V_bar += Y
            if dressed:
                kernel_bar = (Y if gauge.kernel_bar is None
                              else Y + gauge.kernel_bar)
                env_grad += environment.aux_kernel_adjoint(auxmol, kernel_bar)
        return M_bar, V_bar, None, env_grad

    chain = PointChain(pts_local, atom_of_point, frames, with_frames,
                       body=body)
    return ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds,
                                  numerics=numerics)


def ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds, numerics=None):
    """(natm, 3) gradient of a target that reads the AO-product fit on
    `coords` through its outputs (X, M, V): the fit's reverse pass, its
    integrals and collocation, and the points closed by `chain` (a
    `PointChain`, r_g = p_g F_i + R_i).

    seeds(X, M, V) -> (M_bar, V_bar, X_bar, grad): the target's adjoints on
    M, on the bare metric V (beside the fit's own V^-1) and on the
    collocation X (None for none), and whatever of its gradient it closed
    itself ((natm, 3) or 0) -- the factor gauges'
    (`ao_product_fit_adjoint_gauges`), an interpolated exchange's
    (`exchange_skeleton`). The forward is formed here once and handed to it.
    """
    # the ranks of the enclosing distribution, if any (`distributed`)
    comm = current_comm()
    X = ao_values(mol, coords)
    G, d = balanced_gram(X, numerics)
    V = auxmol.intor('int2c2e', aosym='s1')
    E = ao_product_projection(mol, auxmol, X, numerics, comm)
    F = np.linalg.solve(V, E)
    del E
    cho = scipy.linalg.cho_factor(G, lower=True, overwrite_a=True)
    del G
    B = scipy.linalg.cho_solve(cho, (F * d[None, :]).T).T
    M = B * d[None, :]

    M_bar, V_bar, X_seed, extra = seeds(X, M, V)
    V_bar = np.array(V_bar, dtype=float)
    del M

    # --- the fit's solve: B = H G^-1 with H = F d
    B_bar = M_bar * d[None, :]
    d_bar = np.einsum('Pg,Pg->g', M_bar, B)
    del M_bar
    H_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
    del cho, B_bar
    S = -(B.T @ H_bar)                 # G_bar, low rank
    S += S.T                           # G_bar + G_bar^T
    del B
    F_bar = H_bar * d[None, :]
    d_bar += np.einsum('Pg,Pg->g', H_bar, F)
    del H_bar

    # --- the balanced Gram matrix G = (A o A) o d d^T + reg I
    A = X @ X.T
    S *= A                             # (G_bar + G_bar^T) o A
    d_bar += (S * A) @ d
    del A
    S *= d[:, None]
    S *= d[None, :]
    s_bar = -d_bar * d * d             # d = 1/s, s_g = A_gg = |X_g|^2
    X_bar = 2.0 * (S @ X) + 2.0 * s_bar[:, None] * X
    del S
    if X_seed is not None:
        X_bar += X_seed

    # --- F = V^-1 E: the metric and the projection's adjoint
    E_bar = np.linalg.solve(V, F_bar)
    V_bar -= E_bar @ F.T
    del F, F_bar

    # --- E = sum_mn (P|mn) X_gm X_gn: the collocation and the integrals
    grad = _projection_adjoint(mol, auxmol, X, E_bar, X_bar, numerics)
    del E_bar
    grad += two_centre_adjoint(auxmol, V_bar)
    grad += extra
    centre, P = basis_centre_forces(mol, coords, X_bar)
    grad += centre
    grad += chain(mol, P)
    return grad


def _projection_adjoint(mol, auxmol, X, E_bar, X_bar, numerics):
    """(natm, 3) of the three-centre integrals' centres under
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn, and the collocation's share
    2 sum_nP E_bar[P, g] (P|mn) X_gn added into X_bar in place.

    Each rank evaluates its own share of the shell runs (`own_runs`) and
    their derivative integrals, over every point tile. A run's X_bar columns
    and AO-centre terms are its own AOs', disjoint from every other run's, so
    they join exactly; the auxiliary-centre terms are added in run order
    (`ordered_sum`): the force's integral branch is the same bits at every
    rank count.

    pyscf's int3c2e_ip1 and int3c2e_ip2 are +(grad .), so each nuclear
    derivative carries a minus; the second AO's centre gives what the
    first's does, (mn|P)_bar and (mn|P) both being symmetric in (m, n).
    """
    comm = current_comm()
    nao, naux, nk = mol.nao_nr(), auxmol.nao_nr(), X.shape[0]
    Et = np.ascontiguousarray(E_bar.T)                     # (nk, naux)
    x_bar = np.zeros_like(X_bar)
    t_ao = np.zeros((3, nao))
    aux_terms = []
    runs = shell_runs(mol, auxmol, numerics, comp=3)
    mine = own_runs(runs, comm)
    first = runs.index(mine[0]) if mine else 0
    for k, run in enumerate(mine, start=first):
        a0, a1 = run[2], run[3]
        nb = a1 - a0
        operand = half_transform_operand(three_centre_block(mol, auxmol, run))
        # the derivative blocks AO-major, (nb, 3, nao naux) and
        # (nb, 3, nao, naux), so each AO's slab is contiguous
        d1 = np.ascontiguousarray(three_centre_block(
            mol, auxmol, run, 'int3c2e_ip1', 3).transpose(1, 0, 2, 3)
        ).reshape(nb, 3, -1)
        d2 = np.ascontiguousarray(three_centre_block(
            mol, auxmol, run, 'int3c2e_ip2', 3).transpose(1, 0, 2, 3))
        t_aux = np.zeros((3, naux))
        for t0, t1 in point_tiles(nk):
            T = half_transformed(operand, X, t0, t1, nb)
            x_bar[t0:t1, a0:a1] += 2.0 * np.einsum('mPg,gP->gm', T, Et[t0:t1])
            del T
            Xt, Ett = X[t0:t1], Et[t0:t1]
            for m in range(a0, a1):
                # (mn|P)_bar of this tile for the AO m, (nao, naux)
                G3 = (Xt * Xt[:, m:m + 1]).T @ Ett
                t_ao[:, m] -= 2.0 * (d1[m - a0] @ G3.reshape(-1))
                t_aux -= np.einsum('xnP,nP->xP', d2[m - a0], G3)
        aux_terms.append((k, t_aux))
        del operand, d1, d2
    # disjoint columns: a sum of one term and zeros, exact in any order
    reduce_sum(x_bar, comm)
    reduce_sum(t_ao, comm)
    X_bar += x_bar
    aux_total = ordered_sum(aux_terms, comm, shape=(3, naux))
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] += t_ao[:, p0:p1].sum(axis=1)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += aux_total[:, q0:q1].sum(axis=1)
    return grad


def ao_product_fit_rows_adjoints(mol, auxmol, coords, targets, numerics=None,
                                 comm=None, block=None):
    """[RowFitAdjoint], one per target: the nuclear adjoint of the AO-product
    row fit's factor D = M^T V^1/2 on `coords` (the module docstring's row
    fit), every target's in one pass. A target is an
    `isdf_fit_adjoint.AdjointSeeds`, or a list of them wanted summed (a
    pool, `summed_seeds`); `isdf_fit_adjoint.fit_rows_adjoints` is the same
    call on the Duchemin-Blase row fit, its seeds and output the same.

    What no seed enters runs once: the Gram tiles and their factor, the
    metric's LU, the forward pass and its integrals, Z and M^T, the
    collocations and the derivative integrals. What is linear in the seeds
    (MT_bar's roots and solves, E_bar, the Gram, balancing and projection
    adjoints, the metric's grid sums) runs per target in the calls a pass of
    that target alone makes, so each target's adjoint is the bits of its own
    call. Per rank it holds its tiles of the Gram rows and then the factor,
    and per target of MT_bar, W, E_bar and X_bar, (rows, naux) or
    (rows, nao) each, beside the forward's F^T, Z and M^T; one run's
    integrals and the one it receives; and the metric's sums on rank 0.
    """
    regularization, block_memory_gb = fit_settings(
        numerics, 'regularization', 'block_memory_gb')
    comm = current_comm() if comm is None else comm
    rank, nranks = (0, 1) if comm is None else (comm.Get_rank(),
                                                comm.Get_size())
    if nranks > 1:
        coords = lockstep(coords, comm)
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    nk = len(coords)
    tiles = [(t0, min(t0 + block, nk)) for t0 in range(0, nk, block)]
    ntiles = len(tiles)
    owners = [[int(t) for t in partition(ntiles, r, nranks)]
              for r in range(nranks)]
    mine = owners[rank]
    targets = [summed_seeds(t) if isinstance(t, (list, tuple)) else t
               for t in targets]
    ntargets = len(targets)
    # every array the pass may hold, zero until it does: the same keys on
    # every rank
    held = dict.fromkeys((
        'metric_lu', 'X_rows', 'X_ext', 'stream_tile', 'S_rows', 'panel',
        'metric_root', 'metric_slab', 'MT_bar_rows', 'W_rows', 'E_bar_rows',
        'X_bar_rows', 'E_rows', 'run_block', 'root_gather', 'MT_rows',
        'G3_batch', 'derivative_block', 'column_tile', 'root_adjoint',
        'two_centre_slab'), 0)

    def hold(name, nbytes):
        held[name] = max(held[name], int(nbytes))

    def total(per_target):
        return sum(total_bytes(a.values()) for a in per_target)

    def rows(t):
        return slice(*tiles[t])

    lu = scipy.linalg.lu_factor(auxmol.intor('int2c2e', aosym='s1'))
    if nranks > 1:
        lu = lockstep(lu, comm, check=True)
    hold('metric_lu', lu[0].nbytes)
    S, d, X_own, _ = product_gram_tiles(mol, coords, tiles, owners, rank,
                                        numerics, (0, 0), hold)
    d_tiles, x_tiles, g_tiles, rooted, MTb = adjoint_inputs(
        targets, tiles, mine, block, nk, naux, auxmol, rank, comm, hold)

    # W = G^-1 d MT_bar: the forward substitution in the factorization sweep
    W = [{t: mtb[t] * d[rows(t), None] for t in mine} for mtb in MTb]
    hold('W_rows', total(W))
    diagonal, panels = cholesky_tiles(S, tiles, owners, rank, comm, W,
                                      regularization, hold)
    del S
    for w in W:
        backward_tiles(diagonal, panels, w, tiles, owners, rank, comm)
    # E_bar^T = F_bar^T V^-1, F_bar^T = d W
    Ub = [{t: np.ascontiguousarray(scipy.linalg.lu_solve(
        lu, (w[t] * d[rows(t), None]).T).T) for t in mine} for w in W]
    hold('E_bar_rows', total(Ub))

    # the forward pass: E^T, and the projection's collocation adjoint
    # 2 sum_nP (P|mn) X_gn E_bar[P, g] from the same half-transformed runs
    aob = [{t: np.zeros((len(X_own[t]), nao)) for t in mine}
           for _ in range(ntargets)]
    hold('X_bar_rows', total(aob))

    def also(t, T, a0, a1):
        for ub, ob in zip(Ub, aob):
            ob[t][:, a0:a1] += 2.0 * np.einsum('mPg,gP->gm', T, ub[t])

    FT = projection_rows(mol, auxmol, X_own, numerics, comm, hold, also=also)
    for t in mine:
        FT[t] = np.ascontiguousarray(scipy.linalg.lu_solve(lu, FT[t].T).T)
    del lu
    # V_bar's -E_bar F^T, the grid sum on rank 0; d_bar's W . F^T
    VbarF = [np.zeros((naux, naux)) if rank == 0 else None
             for _ in range(ntargets)]
    for ub, vbar in zip(Ub, VbarF):
        for t, got in tiles_at_root([ub, FT], tiles, rank, nranks, comm,
                                    hold):
            if got is not None:
                vbar -= got[0].T @ got[1]
    dbar = [{t: np.einsum('kb,kb->k', w[t], FT[t]) for t in mine} for w in W]

    # Z = G^-1 d F^T on the kept factor, M^T = d Z
    Z = FT
    for t in mine:
        Z[t] *= d[rows(t), None]
    forward_tiles(diagonal, panels, Z, tiles, owners, rank, comm, hold)
    backward_tiles(diagonal, panels, Z, tiles, owners, rank, comm, keep=False)
    del diagonal, panels, FT
    for g in range(ntargets):
        for t in mine:
            dbar[g][t] += np.einsum('kb,kb->k', MTb[g][t], Z[t])
    del MTb
    MT = {t: Z[t] * d[rows(t), None] for t in mine}
    hold('MT_rows', total_bytes(MT.values()))
    Vhbar, Rbar = root_sums(targets, MT, d_tiles, g_tiles, rooted, tiles,
                            rank, nranks, comm, naux, hold)
    del MT

    # the (mn|P) adjoints, by each run's owner over E_bar streamed by tile
    three = _projection_rows_adjoint(mol, auxmol, coords, Ub, X_own, tiles,
                                     owners, rank, comm, numerics,
                                     block_memory_gb * BYTES_PER_GB, hold)
    del Ub

    # the Gram matrix's adjoint by row tile, [Z_k | W_k] streamed in order
    for k, (k0, k1) in enumerate(tiles):
        owner = k % nranks
        both = (np.hstack([Z[k]] + [w[k] for w in W]) if rank == owner
                else np.empty((k1 - k0, (1 + ntargets) * naux)))
        broadcast_rows(both, owner, comm)
        hold('column_tile', both.nbytes)
        Z_k = both[:, :naux]
        if k in X_own:
            X_k = X_own[k]
        else:
            X_k = tile_collocation(mol, coords[k0:k1])
            hold('stream_tile', X_k.nbytes)
        d_k = d[k0:k1]
        for i in mine:
            A = X_own[i] @ X_k.T
            for g in range(ntargets):
                W_k = both[:, (1 + g) * naux:(2 + g) * naux]
                H = Z[i] @ W_k.T
                H += W[g][i] @ Z_k.T
                H *= -A                  # (G_bar + G_bar^T) o A
                dbar[g][i] += (H * A) @ d_k
                H *= d[rows(i), None]
                H *= d_k[None, :]
                aob[g][i] += 2.0 * (H @ X_k)
                del H
            del A
        del both, X_k
    del Z, W
    # the balancing d = 1/s, s_g = |X_g|^2
    for g in range(ntargets):
        for t in mine:
            s_bar = -dbar[g][t] * d[rows(t)] * d[rows(t)]
            aob[g][t] += 2.0 * s_bar[:, None] * X_own[t]
    del dbar

    # the collocations' centres and points, one tile at a time
    ao_slices = [(p0, p1) for _, _, p0, p1 in mol.aoslice_by_atom()]
    fit_tile = [np.zeros((ntiles, natm * 3)) for _ in range(ntargets)]
    fit_points = [np.zeros((nk, 3)) for _ in range(ntargets)]
    with_coll = [s.x_bar is not None or s.x_bar_ao is not None
                 for s in targets]
    coll_tile = [np.zeros((ntiles, natm * 3)) if c else None
                 for c in with_coll]
    coll_points = [np.zeros((nk, 3)) if c else None for c in with_coll]
    for t in mine:
        g_ao = mol.eval_gto('GTOval_ip_sph', coords[rows(t)])
        for g, s in enumerate(targets):
            fit_tile[g][t] = centre_forces(g_ao, aob[g][t],
                                            ao_slices).ravel()
            fit_points[g][rows(t)] = np.einsum('xgm,gm->gx', g_ao, aob[g][t])
            if with_coll[g]:
                xa = collocation_rows(s, x_tiles[g], t)
                coll_tile[g][t] = centre_forces(g_ao, xa, ao_slices).ravel()
                coll_points[g][rows(t)] = np.einsum('xgm,gm->gx', g_ao, xa)
                del xa
        del g_ao
    del aob, X_own
    tile_ranges = [[(t, t + 1) for t in own] for own in owners]
    point_ranges = [[tiles[t] for t in own] for own in owners]
    for g in range(ntargets):
        allgather_ranges(fit_tile[g], tile_ranges, comm)
        allgather_ranges(fit_points[g], point_ranges, comm)
        if with_coll[g]:
            allgather_ranges(coll_tile[g], tile_ranges, comm)
            allgather_ranges(coll_points[g], point_ranges, comm)
    return closed_adjoints(targets, Vhbar, VbarF, Rbar, auxmol, block, rank,
                           nranks, comm, hold, held, fit_tile, fit_points,
                           coll_tile, coll_points, three, with_coll)


def _projection_rows_adjoint(mol, auxmol, coords, Ub, X_own, tiles, owners,
                             rank, comm, numerics, budget, hold):
    """[(natm * 3,)] per target of sum_{mnP} (mn|P)_bar d(mn|P)/dR with
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn, Ub each target's tiles of
    E_bar^T.

    The shell runs (`shell_runs` sized for the three derivative components)
    go largest first, run order[c] to rank c % size. Its owner accumulates
    every target's (mn|P)_bar over E_bar streamed tile by tile in grid
    order, in batches of consecutive rounds whose accumulators stay within
    `budget` bytes on every rank, then evaluates the run's int3c2e_ip1 and
    _ip2 once and contracts them with each target's sum: the first AO's
    centre counted twice ((mn|P)_bar and (mn|P) are both symmetric in m, n)
    and the auxiliary centre once. Each run's (natm, 3) is gathered and the
    runs are added in their order."""
    nranks = len(owners)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    ntargets = len(Ub)
    runs = shell_runs(mol, auxmol, numerics, comp=3)
    width = [a1 - a0 for _, _, a0, a1 in runs]
    order = sorted(range(len(runs)), key=lambda ib: (-width[ib], ib))
    owner_of = {ib: c % nranks for c, ib in enumerate(order)}
    ao_atom = np.empty(nao, dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[p0:p1] = ia
    aux_atom = np.empty(naux, dtype=int)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        aux_atom[q0:q1] = ia
    batches, current, load = [], [], np.zeros(nranks)
    for c0 in range(0, len(order), nranks):
        add = np.zeros(nranks)
        for pos, ib in enumerate(order[c0:c0 + nranks]):
            add[pos] = width[ib] * nao * naux * 8 * ntargets
        if current and np.any(load + add > budget):
            batches.append(current)
            current, load = [], np.zeros(nranks)
        current.extend(order[c0:c0 + nranks])
        load += add
    if current:
        batches.append(current)
    forces = [np.zeros((len(runs), natm * 3)) for _ in range(ntargets)]
    for batch in batches:
        own = [ib for ib in batch if owner_of[ib] == rank]
        acc = [{ib: np.zeros((width[ib], nao, naux)) for ib in own}
               for _ in range(ntargets)]
        hold('G3_batch', sum(total_bytes(a.values()) for a in acc))
        for t, (t0, t1) in enumerate(tiles):
            owner = t % nranks
            U_t = []
            for ub in Ub:
                u_t = ub[t] if rank == owner else np.empty((t1 - t0, naux))
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
                a0 = runs[ib][2]
                for m in range(a0, runs[ib][3]):
                    pair = (X_t * X_t[:, m:m + 1]).T          # (nao, rows)
                    for a, u_t in zip(acc, U_t):
                        a[ib][m - a0] += pair @ u_t
                    del pair
            del X_t, U_t
        for ib in own:
            a0, a1 = runs[ib][2], runs[ib][3]
            # AO-major, so each AO's slab is contiguous
            d1 = np.ascontiguousarray(three_centre_block(
                mol, auxmol, runs[ib], 'int3c2e_ip1', 3).transpose(1, 0, 2, 3))
            d2 = np.ascontiguousarray(three_centre_block(
                mol, auxmol, runs[ib], 'int3c2e_ip2', 3).transpose(1, 0, 2, 3))
            hold('derivative_block', d1.nbytes + d2.nbytes)
            for g in range(ntargets):
                G3 = acc[g].pop(ib)
                t_ao = np.zeros((3, a1 - a0))
                t_aux = np.zeros((3, naux))
                for m in range(a0, a1):
                    # +(grad .) integrals: each nuclear derivative a minus
                    t_ao[:, m - a0] = -2.0 * np.einsum('xnP,nP->x',
                                                       d1[m - a0], G3[m - a0])
                    t_aux -= np.einsum('xnP,nP->xP', d2[m - a0], G3[m - a0])
                del G3
                f = np.zeros((natm, 3))
                np.add.at(f, ao_atom[a0:a1], t_ao.T)
                np.add.at(f, aux_atom, t_aux.T)
                forces[g][ib] = f.ravel()
            del d1, d2
        del acc
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


def exchange_skeleton(mf, dm, dm_other, prefactor, channels):
    """(natm, 3) of d/dR E_K^ISDF at fixed density, for an ISDF-K mean field
    on the AO-product fit: `isdf_derivatives.isdf_exchange_skeleton`'s
    terms, with the fit's reverse pass `ao_product_fit_adjoint`. Rank 0's on
    every rank.

    Per exchange channel (omega, weight), E = -(w/4) sum Z W1 W2 with
    Z = M^T V_w M: the collocation and Z take their adjoints
    (`isdf_exchange_adjoints`), Z's reaches M on both sides and V_w between;
    the bare metric joins the fit's own V, an attenuated one is
    differentiated apart. The points are the mean field's own: its pivoted
    selection, else its grid's clouds, closed on its frames
    (`mean_field_point_chain`).
    """
    with_df = mf.with_df
    if not any(weight != 0.0 for _, weight in channels):
        return np.zeros((mf.mol.natm, 3))
    mol = with_df.mol
    auxmol = auxmol_of(mf)
    coords = with_df.coords
    if coords is None:
        coords = lockstep(with_df.place_grid()[0])
    if with_df.selection is not None:
        pts_local, owner = with_df.selection
    else:
        pts_local, owner = point_layout(mol, with_df.grid_radii,
                                        with_df.grid_origins)
    chain = mean_field_point_chain(with_df, mol, pts_local, owner)

    def seeds(X, M, V):
        X_bar = np.zeros_like(X)
        M_bar = np.zeros_like(M)
        V_bar = np.zeros_like(V)
        extra = np.zeros((mol.natm, 3))
        for omega, weight in channels:
            if weight == 0.0:
                continue
            if omega == 0.0:
                V_w = V
            else:
                with mol.with_range_coulomb(omega), \
                        auxmol.with_range_coulomb(omega):
                    V_w = auxmol.intor('int2c2e', aosym='s1')
            xb, Z_bar = isdf_exchange_adjoints(X, M.T @ (V_w @ M), dm,
                                               prefactor * weight, dm_other)
            X_bar += xb
            M_bar += 2.0 * (V_w @ M) @ Z_bar
            if omega == 0.0:
                V_bar += M @ Z_bar @ M.T
            else:
                extra += two_centre_adjoint(auxmol, M @ Z_bar @ M.T,
                                            omega=omega)
        return M_bar, V_bar, X_bar, extra

    return lockstep(ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds,
                                           numerics=with_df.numerics))
