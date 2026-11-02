"""The ISDF fit trained on every AO product in the Coulomb metric, and the
interpolation points it selects for itself.

This is least-squares tensor hypercontraction (LS-THC; Parrish, Hohenstein,
Martinez and Sherrill, J. Chem. Phys. 137, 224106 (2012)) in the Coulomb
metric, with a per-point row scaling, on points chosen by Matthews' pivoted
Cholesky selection over a larger parent grid (J. Chem. Theory Comput. 16,
1382 (2020)), used here as the ISDF fit of the GW-BSE factors.

THE FIT. With X[g, m] = chi_m(r_g) on the points, the factors interpolate
every AO product, chi_m chi_n ~ sum_g X_gm X_gn zeta_g, and M[P, g] holds the
auxiliary coefficients of zeta_g, so that D = M^T V^1/2 is the factor every
consumer reads (Z = D D^T). M minimizes the Coulomb-metric residual summed
over all AO products,

    sum_mn || V^-1/2 [ (P|mn) - sum_g (V M)[P, g] X_gm X_gn ] ||^2,

with the rows of the test set balanced to unit length and the Tikhonov floor
of the Duchemin-Blase estimator (`separable_ri.fit_M`) on their Gram matrix:

    A  = X X^T,  s_g = A_gg,  d = 1/s
    G  = (A o A) o (d d^T) + reg I        the balanced Gram matrix
    E  = sum_mn (P|mn) X_gm X_gn          (naux, nk)
    M  = V^-1 E diag(d) G^-1 diag(d)

It is `separable_ri.fit_M` with every AO product chi_m chi_n as the test set
and no auxiliary functions, in product form: the test set over nao^2 columns
is never formed, the Gram matrix is a Hadamard square of an (nk, nk) product
and the projection E one pass over the three-centre integrals, blocked over
the shells of the first AO.

Why this test set. The Duchemin-Blase set (every AO times the low-l AOs, plus
the auxiliary functions, residual on the RI coefficients) misses the
densities of the near-null AO combinations an augmented basis carries, and a
BSE on such factors binds a spurious root: CBP, LRC-wPBEh/aug-cc-pVDZ in the
full basis, S1 2.6 eV below DF-RI on the tabulated G3 grid (27528 points),
the same on points pivoted for the AO products. This fit on those pivoted
points puts the lowest three singlets and triplets within 0.45 meV of DF-RI
at 9000 points and within 0.17 meV at 11000 (PXZ-TRZ: 0.57 meV at 10000,
0.11 at 15000, where G3 is 3.6 meV off at 26640).

THE POINTS (Matthews' selection). The AO-product test set is what a named
grid means where no test set is named, its points always pivoted:
`pivot_points` of the numerics, else `pivot_count`. `pivoted_points` takes
the tabulated atomic grid of the level with every sub-shell's radii
resampled ISDF_PIVOT_PARENT_FACTOR times as densely (the parent), places it
on the covariant atomic frames, and keeps the first `count` pivots of the
exact pivoted Cholesky factorization of the parent's balanced Gram matrix G
above, its columns formed on demand (`column_pivots`): the points whose AO
products are least represented by the points already taken. Each kept point
stays attached to its atom, in its atom's frame, so a selection frozen at a
reference geometry is carried with the molecule exactly as a tabulated grid
is (`FrozenFactorization`, `body_frame.BodyFrame`). Re-pivoting at every
geometry is not smooth: a pivot changes discretely under a small step.

RANKS. The pivot works on the parent in fixed tiles of FIT_CHOLESKY_BLOCK
points, owned block-cyclically (`point_tiles`): each rank keeps its tiles'
collocation and factor columns, count x n_parent / ranks float64 (about 16 GB
over the ranks at 20000 of 100000 parent points), never the parent's
n_parent^2 Gram matrix, and the gathered rows of the candidates of one
block (`column_pivots`); a tile's arithmetic does not depend on who owns it,
so the pivots are the same at every rank count. The projection E and its
adjoint divide the shell runs of the three-centre integrals over the ranks
(`own_runs`), integrals and contraction alike, 2 nao^2 naux nk over the
ranks; E's partials are added in rank order, the same bits on every rank and
at another rank count the same to rounding. Every rank forms the (nk, nk)
Gram matrix of the selected points and its Cholesky factor.

THE ROW FIT. `ao_product_fit_rows` solves the same estimator with the points
in the fixed tiles of `separable_ri.fit_rows` (FIT_CHOLESKY_BLOCK points,
tile t rank t % size's), so that no rank holds an (nk, nk) array, or any
nk-indexed array, whole:

  Gram     G_ik = (X_i X_k^T)^2 o d_i d_k^T + reg I, i >= k, i owned, the
           column tiles' collocation streamed (every rank evaluates each
           once), d_k = 1/s_k from the tile's own rows;
  pass     every node evaluates every shell run, largest first, in rounds
           of one run per rank of the node: the run's owner evaluates its
           (m n|P) for n up to the run's end and broadcasts it inside the
           node, and every rank adds the run's pairs, each pair of two runs
           once for both orders, into its own tiles of E^T, one matrix
           product a tile, in the fixed run order; then E^T_i V^-1 per tile.
           The integral blocks, about nao^2 naux / 2 doubles in all, never
           cross a node;
  Cholesky `separable_ri.cholesky_tiles` and `backward_tiles`, the row
           fit's right-looking sweep, M^T_i = d_i (G^-1 d F^T)_i.

Every call has a shape fixed by the tile and run indices, and no partial is
summed across ranks, so the rows are the same bits at every rank count, one
rank included: another realization of the replicated fit's estimator, within
its rounding (the balanced Gram matrix's conditioning), not its bits.
"""
import threading
import time
from collections import OrderedDict

import numpy as np
import scipy.linalg
from pyscf import df as pyscf_df

from src.Base.numerical_flags import flag
from src.Base.geometry.geometry import grid_key
from src.Base.numerics.constants import (AO_PRODUCT_BLOCK_SHARE, BYTES_PER_GB,
                                         FIT_CHOLESKY_BLOCK,
                                         ISDF_PIVOT_CANDIDATES,
                                         ISDF_PIVOT_FRACTION,
                                         ISDF_PIVOT_MAX_GB,
                                         ISDF_PIVOT_PARENT_FACTOR,
                                         ISDF_PIVOT_POINTS_PER_ATOM)
from src.Base.factorization.separable_ri import (RowFit, ao_values,
                                                 atomic_points, auxmol_key,
                                                 backward_tiles,
                                                 cholesky_tiles, content_key,
                                                 fit_settings,
                                                 tile_collocation,
                                                 total_bytes)
from src.Base.factorization.shell_runs import shell_ranges
from src.Base.utils.parallel.mpi_grid import (allgather_blocks,
                                              allgather_ranges, broadcast_rows,
                                              contiguous_block,
                                              current_comm, lockstep,
                                              node_comm, ordered_chain_sum,
                                              partition)


#: Pivots and fits already made, by content: the ISDF-K mean field and the
#: post-SCF factors of one geometry ask for the same ones where they stand on
#: the same parent, count and settings, and pay for them once. Keyed on the
#: rank's place in its communicator; whether a lookup hits is agreed over the
#: ranks (`_made`), since one rank's entry can be dropped while another's is
#: kept (simulated ranks share this table).
_MADE = OrderedDict()
_MADE_LOCK = threading.Lock()
#: Entries kept: a pivot and a fit for each of two geometries.
_MADE_SIZE = 4


def _made(key, build, comm=None):
    """`build()`'s value under `key`, made once while it stays among the
    last `_MADE_SIZE` asked for. Under ranks (`comm`) it is read only where
    every rank holds its own entry, else every rank builds again, so none
    skips a collective of `build` the others enter."""
    with _MADE_LOCK:
        value = _MADE.get(key)
        if value is not None:
            _MADE.move_to_end(key)
    hit = value is not None
    if comm is not None and comm.Get_size() > 1:
        hit = all(comm.allgather(hit))
    if hit:
        return value
    value = build()
    with _MADE_LOCK:
        _MADE[key] = value
        while len(_MADE) > _MADE_SIZE:
            _MADE.popitem(last=False)
    return value


def _array_key(a):
    return (np.shape(a), np.ascontiguousarray(a, dtype=float).tobytes())


def _rank_key(comm):
    """The rank's place in its distribution, and the communicator itself
    while it lives (simulated ranks share one process)."""
    if comm is None:
        return None
    return (comm.Get_rank(), comm.Get_size(), id(comm))


class ISDFPivotShort(RuntimeWarning):
    """The parent grid's Gram matrix has a numerical rank below the asked
    point count: the pivot keeps the points up to that rank, which already
    span every AO product the parent can."""


# ---------------------------------------------------------------- the parent

def densified_radii(shell_radii, factor=ISDF_PIVOT_PARENT_FACTOR):
    """{sub-shell: radii} with each sub-shell's sorted radii (`shell_radii`,
    {sub-shell: radii} of one element's tabulated grid) resampled at
    `factor` times as many, by linear interpolation of log r over the index
    (the smallest and largest radius kept). A sub-shell with one radius is
    spread over a factor e^(+-1/2) about it."""
    out = {}
    for shell, rs in shell_radii.items():
        r = np.sort(np.atleast_1d(np.asarray(rs, float)))
        if r.size == 0:
            continue
        lr = np.log(r) if r.size > 1 else np.log(r[0]) + np.array([-0.5, 0.5])
        t = np.linspace(0.0, lr.size - 1.0, int(factor) * r.size)
        out[shell] = np.exp(np.interp(t, np.arange(lr.size), lr))
    return out


def parent_layout(mol, element_radii, origins=None,
                  factor=ISDF_PIVOT_PARENT_FACTOR):
    """(pts_local, owner): the parent's atom-local clouds, per atom, and the
    atom of every point, in the order the atoms come (the order a selection
    is read in). element_radii: {element: {sub-shell: radii}} of the
    tabulated grid; origins: {element: whether its grid samples the
    nucleus}."""
    dense = {el: densified_radii(r, factor) for el, r in element_radii.items()}
    pts_local, owner = [], []
    for ia in range(mol.natm):
        el = grid_key(mol, ia)
        p = atomic_points(dense[el], centre=(0.0, 0.0, 0.0),
                          origin=bool((origins or {}).get(el, False)))
        pts_local.append(p)
        owner.append(np.full(len(p), ia, dtype=int))
    return pts_local, np.concatenate(owner)


def placed(mol, pts_local, frames):
    """r_g = p_g F_i + R_i: the clouds on the frames (natm, 3, 3), stacked
    in atom order, as `FrozenFactorization.coords` places them."""
    return np.vstack([pts_local[ia] @ frames[ia] + mol.atom_coord(ia)
                      for ia in range(mol.natm)])


# ------------------------------------------------------------- the tiles

def point_tiles(n, block=FIT_CHOLESKY_BLOCK):
    """[(t0, t1)]: the points in fixed tiles of `block`, the unit every
    distributed step of this module works in, owned block-cyclically
    (`tile_owner`). A tile's arithmetic is the same calls on the same shapes
    whoever owns it, so every result is the same bits at every rank count."""
    return [(t0, min(t0 + block, n)) for t0 in range(0, n, block)]


def tile_owner(t, comm):
    """The rank that owns tile `t`."""
    return 0 if comm is None else t % comm.Get_size()


def own_tiles(n, comm):
    """[(t, t0, t1)]: this rank's tiles of `n` points."""
    rank = 0 if comm is None else comm.Get_rank()
    return [(t, t0, t1) for t, (t0, t1) in enumerate(point_tiles(n))
            if tile_owner(t, comm) == rank]


# ------------------------------------------------------------- the equations

def inverse_norms(s):
    """d = 1/s for the row norms s of the AO-product test set, a vanishing
    norm taken as 1."""
    s[s == 0.0] = 1.0
    return 1.0 / s


def balanced_gram(X, numerics=None):
    """(G, d): G = (A o A) o (d d^T) + reg I with A = X X^T and d = 1/s
    (`inverse_norms`), s_g = A_gg the norm of point g's row of the
    AO-product test set, reg the fit's regularization (`fit_settings`). G is
    the Gram matrix of the balanced test rows, formed in place."""
    regularization, = fit_settings(numerics, 'regularization')
    G = X @ X.T
    d = inverse_norms(np.diag(G).copy())
    G *= G
    G *= d[:, None]
    G *= d[None, :]
    G[np.diag_indices_from(G)] += regularization
    return G, d


def fit_shell_runs(mol, auxmol, numerics=None, comp=1):
    """[(sh0, sh1, a0, a1)]: runs of consecutive shells of `mol`
    (`shell_ranges`) whose (comp, a1 - a0, nao, naux) block of three-centre
    integrals takes at most the share AO_PRODUCT_BLOCK_SHARE of the fit's
    working set (`fit_settings` 'block_memory_gb'; one shell where a single
    one takes more)."""
    block_memory_gb, = fit_settings(numerics, 'block_memory_gb')
    per_ao = comp * mol.nao_nr() * auxmol.nao_nr() * 8
    max_aos = max(1, int(AO_PRODUCT_BLOCK_SHARE * block_memory_gb
                         * BYTES_PER_GB // per_ao))
    return shell_ranges(mol.ao_loc_nr(), max_aos)


def three_centre_block(mol, auxmol, run, intor='int3c2e', comp=1):
    """(mu nu|P) for the AOs mu of one shell run (`fit_shell_runs`), shape
    (a1 - a0, nao, naux), or (comp, a1 - a0, nao, naux) for a derivative
    integral."""
    sh0, sh1, a0, a1 = run
    block = pyscf_df.incore.aux_e2(
        mol, auxmol, intor=intor, aosym='s1', comp=comp,
        shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas))
    shape = (a1 - a0, mol.nao_nr(), auxmol.nao_nr())
    return block.reshape(shape if comp == 1 else (comp,) + shape)


def half_transform_operand(block):
    """The block (m n|P) as the (nb naux, nao) matrix `half_transformed`
    multiplies, formed once per block."""
    nb, nao, naux = block.shape
    return np.ascontiguousarray(block.transpose(0, 2, 1)).reshape(nb * naux, nao)


def half_transformed(operand, X, t0, t1, nb):
    """T[m, P, g] = sum_n (m n|P) X[g, n] for one block's operand
    (`half_transform_operand`) and the points t0:t1, (nb, naux, t1 - t0)."""
    return (operand @ X[t0:t1].T).reshape(nb, -1, t1 - t0)


def own_runs(runs, comm):
    """This rank's consecutive share of the shell runs (`contiguous_block`)."""
    if comm is None:
        return runs
    lo, hi = contiguous_block(len(runs), comm.Get_rank(), comm.Get_size())
    return runs[lo:hi]


def ao_product_projection(mol, auxmol, X, numerics=None, comm=None):
    """E[P, g] = sum_mn (P|mn) X_gm X_gn, (naux, nk), the Coulomb projection
    of every AO product onto the points, one pass over the three-centre
    integrals blocked over the shells of the first AO.

    Under ranks each rank evaluates and contracts its own consecutive share
    of the shell runs (`own_runs`), over every point tile, and the ranks'
    partials are added in rank order (`ordered_chain_sum`): the same bits on
    every rank. At another rank count the partials group the runs otherwise,
    so E moves by rounding.
    """
    comm = current_comm() if comm is None else comm
    nk, naux = X.shape[0], auxmol.nao_nr()
    Et = np.zeros((nk, naux))
    for run in own_runs(fit_shell_runs(mol, auxmol, numerics), comm):
        a0, a1 = run[2], run[3]
        operand = half_transform_operand(three_centre_block(mol, auxmol, run))
        for t0, t1 in point_tiles(nk):
            T = half_transformed(operand, X, t0, t1, a1 - a0)
            Et[t0:t1] += np.einsum('mPg,gm->gP', T, X[t0:t1, a0:a1])
            del T
        del operand
    if comm is not None and comm.Get_size() > 1:
        size = comm.Get_size()
        Et = ordered_chain_sum({comm.Get_rank(): Et}, list(range(size)), comm,
                               np.zeros((nk, naux)))
    return np.ascontiguousarray(Et.T)


def ao_product_fit(mol, auxmol, coords, numerics=None, comm=None):
    """(X_ao, M, V) of the AO-product Coulomb-metric fit on `coords`: the
    collocation, the auxiliary coefficients M (naux, nk) of the
    interpolation functions (D = M^T V^1/2) and the metric V = (P|Q).

    The projection E is divided over the ranks by shell runs
    (`ao_product_projection`); the (nk, nk) Gram matrix and its solve are
    formed by every rank, the same bits. A fit already made on the same
    molecule, auxiliary basis, points and settings is returned again, the
    same arrays (the ISDF-K mean field's and the post-SCF factors' at one
    geometry): callers do not write into them. The fit's choices are `numerics`'
    (`fit_settings`: regularization and the working set of the blocked
    three-centre pass); l_max_second and pair_tol belong to the
    Duchemin-Blase test set and are not read.
    """
    regularization, block_gb = fit_settings(numerics, 'regularization',
                                            'block_memory_gb')
    comm = current_comm() if comm is None else comm

    def build():
        X = ao_values(mol, coords)
        G, d = balanced_gram(X, numerics)
        E = ao_product_projection(mol, auxmol, X, numerics, comm)
        V = auxmol.intor('int2c2e', aosym='s1')
        F = np.linalg.solve(V, E)
        del E
        cho = scipy.linalg.cho_factor(G, lower=True, overwrite_a=True)
        M = scipy.linalg.cho_solve(cho, (F * d[None, :]).T).T * d[None, :]
        return X, M, V
    return _made(('fit', _rank_key(comm), content_key(mol), auxmol_key(auxmol),
                  _array_key(coords), regularization, block_gb), build, comm)


# ---------------------------------------------------------- the row fit

def inverse_row_norms(X):
    """d = 1/s with s_g = |X_g|^2, the norm of point g's row of the
    AO-product test set (`inverse_norms`), from these rows alone."""
    return inverse_norms(np.einsum('gm,gm->g', X, X))


def product_gram_tiles(mol, coords, tiles, owners, rank, numerics, ext,
                       hold):
    """(S, d, X, X_ext) of the row fit: the balanced, regularized Gram tiles
    S_ik = (X_i X_k^T)^2 o d_i d_k^T (i >= k) of the row tiles `rank` owns,
    the fit's regularization (`fit_settings`) on the diagonal, as
    `separable_ri.cholesky_tiles` reads them (row tile i holds columns
    [0, end of tile i)); d over every point; the
    collocation of this rank's tiles; and the rows [ext) of X. The column
    tiles are streamed, each evaluated once by every rank, the tile's one
    call (`tile_collocation`)."""
    regularization, = fit_settings(numerics, 'regularization')
    nao, nk = mol.nao_nr(), len(coords)
    mine = owners[rank]
    e0, e1 = ext
    X_own = {t: tile_collocation(mol, coords[slice(*tiles[t])]) for t in mine}
    hold('X_rows', total_bytes(X_own.values()))
    X_ext = np.empty((e1 - e0, nao))
    hold('X_ext', X_ext.nbytes)
    d = np.empty(nk)
    S = {i: np.empty((tiles[i][1] - tiles[i][0], tiles[i][1])) for i in mine}
    for k, (k0, k1) in enumerate(tiles):
        if k in X_own:
            X_k = X_own[k]
        else:
            X_k = tile_collocation(mol, coords[k0:k1])
            hold('stream_tile', X_k.nbytes)
        d[k0:k1] = inverse_row_norms(X_k)
        if k0 < e1 and k1 > e0:
            X_ext[k0 - e0:k1 - e0] = X_k
        for i in mine:
            if i >= k:
                A = X_own[i] @ X_k.T
                A *= A
                S[i][:, k0:k1] = A
        del X_k
    for i in mine:
        i0, i1 = tiles[i]
        S[i] *= d[i0:i1, None]
        S[i] *= d[None, :i1]
        S[i][:, i0:i1][np.diag_indices(i1 - i0)] += regularization
    hold('S_rows', total_bytes(S.values()))
    return S, d, X_own, X_ext


def lower_block(mol, auxmol, run):
    """(mu nu|P) for the AOs mu of one shell run (`fit_shell_runs`) and every nu
    up to the run's end, (a1 - a0) a1 rows of naux: the pairs the
    symmetric projection reads (`projection_rows`)."""
    sh0, sh1, a0, a1 = run
    block = pyscf_df.incore.aux_e2(
        mol, auxmol, intor='int3c2e', aosym='s1', comp=1,
        shls_slice=(sh0, sh1, 0, sh1, 0, auxmol.nbas))
    return block.reshape((a1 - a0) * a1, auxmol.nao_nr())


def pair_products(X, a0, a1):
    """K[g, (m, n)] = w_n X_gm X_gn for m in a0:a1 and n < a1, w_n = 2 below
    a0 and 1 inside the run: with `lower_block`, sum over the runs of
    K B = sum_mn (P|mn) X_gm X_gn, each pair m != n of two runs taken once
    for both orders (exactly: the weight is a power of two)."""
    W = X[:, :a1].copy()
    W[:, :a0] *= 2.0
    return (X[:, a0:a1, None] * W[:, None, :]).reshape(len(X), -1)


def projection_rows(mol, auxmol, X_own, numerics=None, comm=None, hold=None,
                    also=None, node=None):
    """{t: E^T_t}, E[P, g] = sum_mn (P|mn) X_gm X_gn on this rank's tiles
    (X_own, {tile: collocation}), one pass over the shell runs
    (`fit_shell_runs`), largest first.

    Every node makes the whole pass itself: the runs go in rounds of one per
    rank of the node, each rank evaluates its run's integrals and every run
    is broadcast by its owner in turn inside the node, so no integral block
    crosses a node and each tile adds the runs in the one fixed order
    whoever evaluated them, the same calls on the same shapes: the same bits
    at every rank and node count. node: the ranks that share the block
    (`node_comm` of `comm` when None; a test hands smaller groups).

    Without `also` the pass reads the pairs once, by the symmetry (P|mn) =
    (P|nm): a run of AOs m takes nu up to its end (`lower_block`) and adds
    E^T += K B, one matrix product over the pairs (`pair_products`), half
    the integrals and half the products of the full pass. also(t, T, a0,
    a1), where given, is called with each tile's half-transformed run
    T[m, P, g] = sum_n (m n|P) X[g, n] (`half_transformed`, every nu) and its
    AO span (the adjoint reads the pass through it).
    """
    comm = current_comm() if comm is None else comm
    split = node is None
    if split:
        node = node_comm(comm)
    rank, nranks = (0, 1) if node is None else (node.Get_rank(),
                                                node.Get_size())
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    runs = fit_shell_runs(mol, auxmol, numerics)
    lower = also is None
    cost = [(a1 - a0) * (a1 if lower else nao) for _, _, a0, a1 in runs]
    order = sorted(range(len(runs)), key=lambda ib: (-cost[ib], ib))
    Et = {t: np.zeros((len(x), naux)) for t, x in X_own.items()}
    if hold is not None:
        hold('E_rows', total_bytes(Et.values()))
    try:
        for c0 in range(0, len(order), nranks):
            batch = order[c0:c0 + nranks]
            own = None
            if rank < len(batch):
                run = runs[batch[rank]]
                own = (np.ascontiguousarray(lower_block(mol, auxmol, run))
                       if lower else half_transform_operand(
                           three_centre_block(mol, auxmol, run)))
            for pos, ib in enumerate(batch):
                a0, a1 = runs[ib][2], runs[ib][3]
                nb = a1 - a0
                shape = (nb * a1, naux) if lower else (nb * naux, nao)
                operand = own if pos == rank else np.empty(shape)
                broadcast_rows(operand, pos, node)
                if hold is not None:
                    hold('run_block', operand.nbytes
                         + (0 if own is None or pos == rank else own.nbytes))
                for t, X_t in X_own.items():
                    if lower:
                        K = pair_products(X_t, a0, a1)
                        if hold is not None:
                            hold('pair_products', K.nbytes)
                        Et[t] += K @ operand
                        del K
                        continue
                    T = half_transformed(operand, X_t, 0, len(X_t), nb)
                    Et[t] += np.einsum('mPg,gm->gP', T, X_t[:, a0:a1])
                    also(t, T, a0, a1)
                    del T
                del operand
            own = None
    finally:
        if split and node is not None and node is not comm:
            node.Free()
    return Et


def ao_product_fit_rows(mol, auxmol, coords, numerics=None, comm=None,
                        block=None, tiles=None, timings=None):
    """The AO-product fit's M^T as this rank's block-cyclic tiles of `block`
    points (FIT_CHOLESKY_BLOCK when None), a `separable_ri.RowFit`: the
    row fit of the module docstring, the same bits at every rank count.

    tiles: a `separable_ri.FitTiles` of this fit another stage holds (the
    distributed ISDF-K SCF's), read in place of fitting where it is this
    fit -- the AO-product test set, molecule, auxiliary basis, points, tile
    edge, settings and ranks -- and ignored otherwise.
    timings: filled with 'fit_gram', 'fit_integrals', 'fit_cholesky'
    (factorization and both substitutions); read from `tiles`,
    'fit_collocation' and 'fit_reused_tiles'.
    """
    settings = fit_settings(numerics, 'l_max_second', 'pair_tol',
                            'regularization', 'block_memory_gb')
    regularization = settings[2]
    comm = current_comm() if comm is None else comm
    rank, nranks = (0, 1) if comm is None else (comm.Get_rank(),
                                                comm.Get_size())
    if nranks > 1:
        coords = lockstep(coords, comm)
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    if block < 1:
        raise ValueError(f'block={block}: a tile holds at least one point')
    if tiles is not None:
        _t = time.time()
        handed = tiles.row_fit(mol, auxmol, coords, block, comm, settings,
                               test_set='ao_products')
        if handed is not None:
            if timings is not None:
                timings['fit_collocation'] = time.time() - _t
                timings['fit_reused_tiles'] = len(handed.mt)
            return handed
    nk = len(coords)
    grid = [(t0, min(t0 + block, nk)) for t0 in range(0, nk, block)]
    owners = [[int(t) for t in partition(len(grid), r, nranks)]
              for r in range(nranks)]
    mine = owners[rank]
    r0, r1 = contiguous_block(nk, rank, nranks)
    e0 = e1 = r0             # the whole tiles over this rank's contiguous rows
    if r1 > r0:
        e0, e1 = (r0 // block) * block, min(-(-r1 // block) * block, nk)
    # every array the fit may hold, zero until it does: the same keys on
    # every rank
    held = dict.fromkeys(('metric_lu', 'X_rows', 'X_ext', 'stream_tile',
                          'S_rows', 'E_rows', 'run_block',
                          'pair_products', 'panel', 'MT_rows'), 0)

    def hold(name, nbytes):
        held[name] = max(held[name], int(nbytes))

    _t = time.time()
    lu = scipy.linalg.lu_factor(auxmol.intor('int2c2e', aosym='s1'))
    if nranks > 1:
        # one metric, rank 0's, for every tile's solve
        lu = lockstep(lu, comm, check=True)
    hold('metric_lu', lu[0].nbytes)
    S, d, X_own, X_ext = product_gram_tiles(mol, coords, grid, owners, rank,
                                            numerics, (e0, e1), hold)
    if timings is not None:
        timings['fit_gram'] = time.time() - _t
    _t = time.time()
    R = projection_rows(mol, auxmol, X_own, numerics, comm, hold)
    del X_own
    for i in mine:
        # F^T = E^T V^-1 (V symmetric), balanced
        R[i] = np.ascontiguousarray(scipy.linalg.lu_solve(lu, R[i].T).T)
        R[i] *= d[slice(*grid[i]), None]
    del lu
    if timings is not None:
        timings['fit_integrals'] = time.time() - _t
    _t = time.time()
    diagonal, panels = cholesky_tiles(S, grid, owners, rank, comm, [R],
                                      regularization, hold)
    backward_tiles(diagonal, panels, R, grid, owners, rank, comm, keep=False)
    for i in mine:
        R[i] *= d[slice(*grid[i]), None]
    if timings is not None:
        timings['fit_cholesky'] = time.time() - _t
    hold('MT_rows', total_bytes(R.values()))
    return RowFit(nk, block, comm, R, X_ext, (e0, e1), held)


# ---------------------------------------------------------------- the pivot

def pivot_gb(n_parent, count, comm=None):
    """GB one rank holds for the pivot: its tiles' collocation is counted by
    the caller's basis, its factor columns (count, rows) here, at the most
    tiles any rank owns, and the factor columns of a block's candidates
    (count, ISDF_PIVOT_CANDIDATES)."""
    ntiles = len(point_tiles(n_parent))
    size = 1 if comm is None else comm.Get_size()
    rows = -(-ntiles // size) * FIT_CHOLESKY_BLOCK
    rows = min(rows, n_parent) + min(ISDF_PIVOT_CANDIDATES, n_parent)
    return 8.0 * count * rows / BYTES_PER_GB


def column_pivots(mol, coords, count, numerics=None, comm=None,
                  candidates=None, timings=None):
    """The first `count` pivots of the pivoted Cholesky factorization of the
    balanced AO-product Gram matrix G over `coords` (the largest remaining
    diagonal first, ties to the lowest index), stopping early where the
    largest remaining diagonal falls to n eps max(diag G) (LAPACK dpstrf's
    rule; warned, `ISDFPivotShort`).

    G is never formed: column p, G[:, p] = (X x_p)^2 d d_p + reg e_p, is
    made from the pivot's collocation row, and only the factor's columns are
    kept, (count, rows) per rank. The pivots are taken in blocks, so that the
    factor is read once a block by a matrix product rather than once a pivot:

      candidates  the `candidates` points of largest residual diagonal
                  (ISDF_PIVOT_CANDIDATES when None), C, and the largest
                  residual outside them, tau, over every rank;
      block       the residual Gram matrix on C, G_CC - L_C^T L_C, pivoted
                  greedily for as long as its largest residual beats tau
                  (larger, or equal at a lower index): no point outside C can
                  then be the largest, as a residual only falls, so these are
                  the pivots the one-at-a-time selection takes;
      update      the block's factor columns on every point, (G[:, P] -
                  L^T L_P) R^-1 with R the block's triangular factor on its
                  pivots, and the residual diagonal.

    Every rank holds its own point tiles (their collocation, residual
    diagonal and factor columns) and the candidates' rows, gathered; the
    calls have shapes fixed by the tiles and the candidate count, so the
    pivots are the same at every rank count. The Tikhonov floor is the
    fit's regularization (`fit_settings`). timings: filled with the seconds
    of each part ('pivot_candidates', 'pivot_gather', 'pivot_block_gram',
    'pivot_block', 'pivot_update') and the block count ('pivot_blocks').
    """
    regularization, = fit_settings(numerics, 'regularization')
    comm = current_comm() if comm is None else comm
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    ncand = ISDF_PIVOT_CANDIDATES if candidates is None else int(candidates)
    if ncand < 1:
        raise ValueError(f'candidates={ncand}: a block needs at least one')
    clock = dict.fromkeys(('pivot_candidates', 'pivot_gather',
                           'pivot_block_gram', 'pivot_block',
                           'pivot_update'), 0.0)
    n = len(coords)
    mine = own_tiles(n, comm)
    X, d, r, L = {}, {}, {}, {}
    for t, t0, t1 in mine:
        X[t] = np.ascontiguousarray(ao_values(mol, coords[t0:t1]))
        d[t] = inverse_row_norms(X[t])
        # (a a) d d is 1 up to rounding; its exact value, so the first pivot,
        # a tie of every point, goes to the lowest index whatever the
        # rounding
        r[t] = np.full(t1 - t0, 1.0 + regularization)
        L[t] = np.zeros((count, t1 - t0))
    nao = mol.nao_nr()
    index = (np.concatenate([np.arange(t0, t1) for _, t0, t1 in mine])
             if mine else np.empty(0, dtype=int))
    local = max((float(v.max()) for v in r.values()), default=-np.inf)
    largest = local if comm is None else max(comm.allgather(local))
    tol = n * np.finfo(float).eps * largest
    order, short, blocks = [], False, 0
    while len(order) < count and not short:
        j0 = len(order)
        _t = time.time()
        # this rank's best ncand + 1 (value, point), as floats (a point
        # index is exact in a double), padded; every rank's gathered
        want = ncand + 1
        cands = np.full((size * want, 2), [-np.inf, -1.0])
        resid = (np.concatenate([r[t] for t, _, _ in mine]) if mine
                 else np.empty(0))
        if resid.size:
            top = np.lexsort((index, -resid))[:want]
            cands[rank * want:rank * want + len(top), 0] = resid[top]
            cands[rank * want:rank * want + len(top), 1] = index[top]
        if comm is not None:
            allgather_blocks(cands, comm)
        cands = cands[cands[:, 1] >= 0]
        cands = cands[np.lexsort((cands[:, 1], -cands[:, 0]))]
        # tau: the best point outside the candidates (none: below all)
        tau = (cands[ncand, 0], cands[ncand, 1]) if len(cands) > ncand \
            else (-np.inf, np.inf)
        cands = cands[:ncand]
        cands = cands[cands[:, 0] > -np.inf]
        pts = cands[:, 1].astype(int)
        owner = np.array([tile_owner(p // FIT_CHOLESKY_BLOCK, comm)
                          for p in pts], dtype=int)
        # the candidates' rows grouped by owner, by point within an owner
        place = np.lexsort((pts, owner))
        pts, owner, value = pts[place], owner[place], cands[place, 0]
        clock['pivot_candidates'] += time.time() - _t
        _t = time.time()
        nc = len(pts)
        rows = np.empty((nc, nao + 1 + j0))
        for i in np.flatnonzero(owner == rank):
            tp = pts[i] // FIT_CHOLESKY_BLOCK
            kp = pts[i] - tp * FIT_CHOLESKY_BLOCK
            rows[i, :nao] = X[tp][kp]
            rows[i, nao] = d[tp][kp]
            rows[i, nao + 1:] = L[tp][:j0, kp]
        if comm is not None and size > 1:
            bounds = np.searchsorted(owner, np.arange(size + 1))
            allgather_ranges(rows, [[(int(bounds[q]), int(bounds[q + 1]))]
                                    if bounds[q + 1] > bounds[q] else []
                                    for q in range(size)], comm)
        x_c, d_c, l_c = rows[:, :nao], rows[:, nao], rows[:, nao + 1:]
        clock['pivot_gather'] += time.time() - _t
        _t = time.time()
        # the residual Gram matrix on the candidates, its diagonal the
        # residuals the selection reads
        R = x_c @ x_c.T
        R *= R
        R *= d_c[:, None]
        R *= d_c[None, :]
        if j0:
            R -= l_c @ l_c.T
        R[np.diag_indices(nc)] = value
        clock['pivot_block_gram'] += time.time() - _t
        _t = time.time()
        res = value.copy()
        room = min(nc, count - j0)
        Lb = np.zeros((room, nc))
        chosen, roots = [], []
        for s in range(room):
            top = float(res.max())
            k = int(pts[res == top].min())
            k = int(np.flatnonzero(pts == k)[0])
            if s and not (top > tau[0] or (top == tau[0] and pts[k] < tau[1])):
                break
            if top <= tol:
                flag('isdf_pivot_rank_short',
                     f'the parent grid spans the AO products to numerical '
                     f'rank {j0 + s}, below the {count} points asked for: '
                     f'{j0 + s} points are kept', ISDFPivotShort)
                short = True
                break
            col = R[:, k] - Lb[:s].T @ Lb[:s, k] if s else R[:, k].copy()
            roots.append(np.sqrt(top))
            Lb[s] = col / roots[-1]
            res -= Lb[s] * Lb[s]
            res[k] = -np.inf
            chosen.append(k)
        chosen = np.array(chosen, dtype=int)
        b = len(chosen)
        clock['pivot_block'] += time.time() - _t
        if not b:
            break
        _t = time.time()
        # Y U = G[:, P] - L^T L_P, U upper triangular: the block's factor
        # entries on its own pivots, the residual roots on the diagonal
        U = np.triu(Lb[:b][:, chosen], 1)
        U[np.diag_indices(b)] = roots
        x_p = np.ascontiguousarray(x_c[chosen])
        d_p = d_c[chosen]
        l_p = np.ascontiguousarray(l_c[chosen]) if j0 else None
        p_pts = pts[chosen]
        for t, t0, t1 in mine:
            g = X[t] @ x_p.T
            g *= g
            g *= d[t][:, None]
            g *= d_p[None, :]
            inside = np.flatnonzero((p_pts >= t0) & (p_pts < t1))
            g[p_pts[inside] - t0, inside] += regularization
            if j0:
                g -= L[t][:j0].T @ l_p.T
            L[t][j0:j0 + b] = scipy.linalg.solve_triangular(
                U, g.T, trans='T', lower=False, check_finite=False)
            for s in range(b):
                r[t] -= L[t][j0 + s] * L[t][j0 + s]
            r[t][p_pts[inside] - t0] = -np.inf
        order.extend(int(p) for p in p_pts)
        blocks += 1
        clock['pivot_update'] += time.time() - _t
    if timings is not None:
        timings.update(clock)
        timings['pivot_blocks'] = blocks
    return np.array(order, dtype=int)


def pivot_count(natm, n_parent):
    """The point count pivoted where none is asked for:
    max(ISDF_PIVOT_FRACTION x n_parent, ISDF_PIVOT_POINTS_PER_ATOM x natm),
    at most the parent (constants.py has the measurements)."""
    want = max(int(np.ceil(ISDF_PIVOT_FRACTION * n_parent)),
               ISDF_PIVOT_POINTS_PER_ATOM * int(natm))
    return min(int(n_parent), want)


def pivot_key(mol, coords, count, numerics, comm):
    """The key a pivot is kept under (`_made`): this rank's place, the
    molecule, the placed parent, the count and the fit's regularization
    (`fit_settings` of `numerics`)."""
    regularization, = fit_settings(numerics, 'regularization')
    return ('pivot', _rank_key(comm), content_key(mol), _array_key(coords),
            int(count), float(regularization))


def pivoted_points(mol, element_radii, origins, frames, count, numerics=None,
                   comm=None):
    """(pts_local, owner, parent_size): the `count` points pivoted from the
    parent grid (`parent_layout` on `element_radii` and `origins`, placed on
    `frames`) for the AO-product test set (`column_pivots`), each kept with
    its atom, in the parent's order; the clouds as
    `FrozenFactorization.coords` reads them. count None is `pivot_count`'s.

    A count above the parent's size, and a pivot whose factor columns would
    exceed ISDF_PIVOT_MAX_GB on one rank, are refused before anything is
    formed. Under ranks (`comm`, default `current_comm()`) every rank places
    the parent on the `frames` it is handed (a caller locksteps them).
    """
    comm = current_comm() if comm is None else comm
    pts_local, owner = parent_layout(mol, element_radii, origins)
    n_parent = len(owner)
    if count is None:
        count = pivot_count(mol.natm, n_parent)
    if count > n_parent:
        raise ValueError(
            f'pivot_points={count} exceeds the {n_parent} points of the '
            'parent grid; ask for fewer points or a larger level')
    gb = pivot_gb(n_parent, count, comm)
    if gb > ISDF_PIVOT_MAX_GB:
        raise MemoryError(
            f'the pivot holds {count} factor columns over the parent rows '
            f'one rank owns ({n_parent} parent points), {gb:.3g} GB, above '
            f'ISDF_PIVOT_MAX_GB = {ISDF_PIVOT_MAX_GB:g} GB; run it over more '
            'ranks or ask for fewer points')
    coords = placed(mol, pts_local, frames)
    keep = np.sort(_made(pivot_key(mol, coords, count, numerics, comm),
                         lambda: column_pivots(mol, coords, int(count),
                                               numerics, comm), comm))
    starts = np.concatenate([[0], np.cumsum([len(p) for p in pts_local])])
    local = []
    for ia in range(mol.natm):
        mine = keep[(keep >= starts[ia]) & (keep < starts[ia + 1])] - starts[ia]
        local.append(pts_local[ia][mine])
    return local, owner[keep], n_parent
