"""The Casida Davidson's trial space cut by pair rows over ranks.

Over more than one rank `_davidson_root` runs `trial_space.real_eig_rows`,
pyscf's real_eig owned: V, W, U1 and U2 are held as fixed tiles of pair rows,
each sum over the pair index is formed per tile, added in tile order and
reduced once, the row products are computed on a rank's own tiles, and the
batch and the result are gathered whole. Gated here:

  * ONE TILE is real_eig's arithmetic: on one rank with a tile as long as the
    pair space the loop returns real_eig's roots and vectors bitwise, on a
    Casida model, on a symmetry-blocked one and on water BSE@HF;
  * ONE RANK, MANY TILES against real_eig: benzene/cc-pVDZ BSE@HF, 12 roots at
    1e-5, on a one-rank communicator at the default tile and at BENZENE_TILE,
    roots within ONE_RANK_ROOT_TOL and vectors within ONE_RANK_VECTOR_TOL once
    each root's sign is fixed;
  * the OUTPUT PARTITIONS -- Ritz vectors, residuals, the preconditioner step,
    the Gram-Schmidt projection and combination, the gathered batch -- are
    the same bits at 1 to 64 simulated ranks for the same coefficients;
  * every REDUCED SUM -- the projected blocks, the residual norms, the
    correction norms, the Gram-Schmidt overlaps and Gram matrices -- passes
    `reduced_sum_verdict` on every rank, and a partial added out of tile
    order, or one missing a tile, fails it;
  * every rank's DECISIONS -- batch sizes, directions kept, converged flags,
    residual norms, restarts, the cycle count -- are rank 0's, gathered from
    every rank, at rank counts where ranks own no tile;
  * the ROOTS at 1 to 64 simulated ranks lie within the Davidson's resolution
    (`roots_resolution`) of the serial ones, every root converged: the model,
    water and benzene on the ISDF action;
  * the HOLDERS of a rank take 32 bytes per pair row it owns per trial pair,
    and the ranks' together what one rank held;
  * the SERIAL path, and a one-rank communicator, stay pyscf's real_eig;
  * RESTARTS take the Ritz vectors' products from the stored ones and apply
    no action, and still converge to the dense roots;
  * one rank's subspace solution moved one ulp is repaired by the loop's
    checked lockstep, every rank returning the unperturbed solve;
  * the subspace time is split into its pieces under a comm.
"""
import os
import sys
import copy
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import DAVIDSON_PAIR_TILE
from src.Base.utils.mpi_grid import (contiguous_block, current_comm,
                                     distributed, lockstep_stats,
                                     run_simulated, simulated_world)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson, trial_space
from src.SingleReference.LinearResponse.davidson import (bse_pair_diagonal,
                                                         isdf_bse_factors,
                                                         isdf_block_action)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.trial_space import (
    SUBSPACE_PIECES, PairRows, combine_rows, gram_addend, overlap_addend,
    project_rows, projection_addend, ritz_rows)
from tests.reduction_bounds import reduced_sum_verdict
from tests.test_distributed_fit_mpi import relative, roots_resolution

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; '
           'H 1.240 2.147 0; H -1.240 2.147 0; H -2.479 0 0; '
           'H -1.240 -2.147 0; H 1.240 -2.147 0')
#: The Casida model of the RPA shape, A = D + 2K, B = 2K, K = V^T V.
NOCC, NVIR, NAUX, NROOTS, CONV_TOL = 8, 150, 40, 12, 1e-6
BSE_CONV_TOL = 1e-5
#: Ha. The owned loop on one rank against real_eig, many tiles.
ONE_RANK_ROOT_TOL = 1e-10
#: The vectors to the same, each root's sign fixed.
ONE_RANK_VECTOR_TOL = 1e-6
#: Pair rows per tile where a gate splits a small pair space: the model's
#: 1200 pairs into 75 tiles, water's 95 into 6, benzene's 1953 into 31.
MODEL_TILE, WATER_TILE, BENZENE_TILE = 16, 16, 64
ROOT_SIZES = [2, 3, 5, 8, 16, 32, 64]
#: Rank counts at which some ranks own no tile of the split pair space.
EMPTY_SIZES = [32, 64]
#: The fixed arrays of the partition and reduction gates.
N_GATE, M_GATE, K_GATE, TILE_GATE = 1000, 30, 6, 32
GATE_SIZES = [1, 2, 3, 5, 8, 16, 64]
#: The trial pairs a forced bound holds: the model collapses every few cycles.
FORCED_SPACE = 60
FORCED_MAX_CYCLE = 200
#: The subspace solve the drift is planted in, and the rank it is planted on.
PLANT_AT = 3
#: s. The pieces are read with time.time() around the loop's phases.
TIMER_SLACK = 1e-3


def model_problem(seed=11):
    """(apply_AB, diag, eps, dense roots) of a model Casida problem."""
    rng = np.random.default_rng(seed)
    eps = np.concatenate([-np.sort(rng.uniform(0.3, 2.0, NOCC))[::-1],
                          np.sort(rng.uniform(0.05, 3.0, NVIR))])
    diag = bse_pair_diagonal(eps, NOCC)
    n_ov = diag.size
    V = rng.standard_normal((NAUX, n_ov)) / np.sqrt(n_ov) * 0.5

    def apply_AB(z):
        f = z.reshape(len(z), -1)
        v = (2 * (f @ V.T) @ V).reshape(z.shape)
        return diag[None] * z + v, v

    # RPA's screened diagonal is d itself, as the production actions return
    # it; the default preconditioner is now the screened diagonal, which a
    # block action without one refuses.
    apply_AB.screened_diagonal = lambda: diag

    sqrt_d = np.sqrt(diag.ravel())
    m = sqrt_d[:, None] * (np.diag(diag.ravel()) + 4 * V.T @ V) * sqrt_d[None, :]
    return apply_AB, diag, eps, np.sqrt(np.linalg.eigvalsh(m))


def symmetric_problem(seed=5):
    """The model with its pairs labelled by parity and K block diagonal over
    the two labels: (apply_AB, diag, labels)."""
    _, diag, _, _ = model_problem()
    n_ov = diag.size
    labels = np.arange(n_ov) % 2
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((NAUX, n_ov)) / np.sqrt(n_ov) * 0.5
    V[:NAUX // 2, labels == 1] = 0.0
    V[NAUX // 2:, labels == 0] = 0.0

    def apply_AB(z):
        f = z.reshape(len(z), -1)
        v = (2 * (f @ V.T) @ V).reshape(z.shape)
        return diag[None] * z + v, v
    apply_AB.screened_diagonal = lambda: diag        # RPA's, as above
    return apply_AB, diag, labels


def root(apply_AB, diag, comm=None, nroots=NROOTS, conv_tol=CONV_TOL,
         x_sym=None, max_cycle=100, **kw):
    """(omega, X, Y, converged, cycles, counters) of one `_davidson_root`."""
    counters = davidson._davidson_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = davidson._davidson_root(apply_AB, diag, nroots, conv_tol,
                                      max_cycle, x_sym, counters=counters,
                                      comm=comm, **kw)
    return out[:5] + (counters,)


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def sign_fixed(X, Y):
    """(X, Y) with each root's sign taken from the largest component of X:
    the Davidson fixes no phase."""
    sign = np.sign(X[np.abs(X).argmax(axis=0), np.arange(X.shape[1])])
    return X * sign, Y * sign


def decisions(counters):
    """The per-cycle decisions of the pair-row loop, residual norms included."""
    return [(s['cycle'], s['fresh'], s['new'], s['applied'], s['m1'],
             s['conv'], s['kept'], s['r_norms'].tobytes())
            for s in counters['trace']]


def _system(geometry):
    mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(nocc=nocc, factors=factors, W_aux=W_aux,
                eps=np.asarray(mf.mo_energy, float))


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    return _system(WATER)


@pytest.fixture(scope='module')
def benzene():
    warnings.simplefilter('ignore')
    return _system(BENZENE)


def bse_root(s, nroots, comm=None, **kw):
    """BSE@HF by `_davidson_root` on the ISDF action, this rank's own copies."""
    lr = LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')
    act, diag = isdf_block_action(lr, s['nocc'], True, s['W_aux'].copy(),
                                  tuple(np.array(a) for a in s['factors']),
                                  comm=comm)
    return root(act, diag, comm, nroots=nroots, conv_tol=BSE_CONV_TOL, **kw)


@pytest.fixture(scope='module')
def model():
    return model_problem()


@pytest.fixture(scope='module')
def model_serial(model):
    return root(*model[:2])


@pytest.fixture(scope='module')
def benzene_serial(benzene):
    return bse_root(benzene, 12)


def test_one_tile_is_real_eigs_bits(model, model_serial, water):
    """One rank, one tile: every expression the loop evaluates is real_eig's."""
    om_s, X_s, Y_s = model_serial[:3]
    om, X, Y = root(*model[:2], pair_rows=True, pair_tile=model[1].size)[:3]
    assert bitwise(om, om_s) and bitwise(X, X_s) and bitwise(Y, Y_s)
    act, diag, labels = symmetric_problem()
    ref = root(act, diag, x_sym=labels)
    got = root(act, diag, x_sym=labels, pair_rows=True, pair_tile=diag.size)
    for a, b in zip(ref[:3], got[:3]):
        assert bitwise(a, b)
    ref = bse_root(water, 3)
    got = bse_root(water, 3, pair_rows=True, pair_tile=DAVIDSON_PAIR_TILE)
    assert water['nocc'] * (len(water['eps']) - water['nocc']) <= DAVIDSON_PAIR_TILE
    for a, b in zip(ref[:3], got[:3]):
        assert bitwise(a, b)


@pytest.mark.parametrize('tile', [None, BENZENE_TILE])
def test_one_rank_against_real_eig(benzene, benzene_serial, tile):
    """The owned loop on a one-rank communicator, the pair space in several
    tiles: real_eig's roots and vectors to the tiles' re-association."""
    om_s, X_s, Y_s = benzene_serial[:3]
    comm = simulated_world(1)[0]
    with distributed(comm):
        om, X, Y, conv, cycles, counters = bse_root(benzene, 12, comm,
                                                    pair_rows=True,
                                                    pair_tile=tile)
    assert len(counters['trace']) == cycles > 0
    assert len(PairRows(X.shape[0], comm, tile).tiles) > 1
    assert conv.all()
    assert np.abs(om - om_s).max() <= ONE_RANK_ROOT_TOL
    Xf, Yf = sign_fixed(X, Y)
    Xs, Ys = sign_fixed(X_s, Y_s)
    assert np.abs(Xf - Xs).max() <= ONE_RANK_VECTOR_TOL
    assert np.abs(Yf - Ys).max() <= ONE_RANK_VECTOR_TOL


def gate_arrays(seed=3):
    """Fixed pair-space arrays and small coefficients for the partition and
    reduction gates."""
    rng = np.random.default_rng(seed)
    n, m, k = N_GATE, M_GATE, K_GATE
    out = {name: rng.standard_normal((n, m)) for name in ('V', 'W', 'U1', 'U2')}
    out.update(x=rng.standard_normal((m, k)), y=0.1 * rng.standard_normal((m, k)),
               w=np.linspace(0.2, 0.9, k), ox=rng.standard_normal((m, k)),
               oy=0.1 * rng.standard_normal((m, k)),
               ca=rng.standard_normal((k, k - 1)),
               cb=0.1 * rng.standard_normal((k, k - 1)),
               X_new=rng.standard_normal((n, k)),
               Y_new=0.1 * rng.standard_normal((n, k)),
               d=np.linspace(0.3, 5.0, n))
    return out


def rank_products(comm, g, tile):
    """This rank's tiles of every output partition for the fixed arrays, and
    the batch gathered from them."""
    rows = PairRows(N_GATE, comm, tile)
    holders = rows.holders(M_GATE)
    for name, blocks in zip(('V', 'W', 'U1', 'U2'), holders):
        for (p0, p1), b in zip(rows.tiles, blocks):
            b[...] = g[name][p0:p1]
    Vh, Wh, U1h, U2h = holders
    out, V_new, W_new = {}, [], []
    for i, (p0, p1) in enumerate(rows.tiles):
        X, Y, R_x, R_y, r2 = ritz_rows(Vh[i], Wh[i], U1h[i], U2h[i], M_GATE,
                                       g['x'], g['y'], g['w'])
        dx = np.vstack([R_x, R_y]).T
        t = davidson._correction(dx, np.hstack([g['d'][p0:p1],
                                                -g['d'][p0:p1]]), g['w'])
        x_new, y_new = g['X_new'][p0:p1].copy(), g['Y_new'][p0:p1].copy()
        project_rows(Vh[i], Wh[i], x_new, y_new, g['ox'], g['oy'])
        v, w = combine_rows(x_new, y_new, g['ca'], g['cb'])
        V_new.append(v)
        W_new.append(w)
        out[(p0, p1)] = (X, Y, R_x, R_y, r2, t, x_new, y_new, v, w)
    return out, rows.gather(V_new, W_new, g['ca'].shape[1])


def test_output_partitions_are_bitwise_across_ranks():
    """Every row a rank computes on its own tiles is the one-rank row, bit
    for bit, at 1 to 64 ranks, and the gathered batch is one rank's."""
    g = gate_arrays()
    ref, batch_1 = rank_products(None, g, TILE_GATE)
    assert len(ref) == -(-N_GATE // TILE_GATE)
    for size in GATE_SIZES:
        out = run_simulated(rank_products, size, g, TILE_GATE)
        seen = {}
        for mine, batch in out:
            assert bitwise(batch, batch_1)
            seen.update(mine)
        assert sorted(seen) == sorted(ref)
        for key, arrays in ref.items():
            for a, b in zip(arrays, seen[key]):
                assert bitwise(a, b), (size, key)


def reduced_sums(comm, g, tile):
    """Every reduced sum of the loop, through `PairRows.join`, on the fixed
    arrays: (name, this rank's addends by tile, the reduced value)."""
    rows = PairRows(N_GATE, comm, tile)
    tiles = rows.tiles
    V, W = ([g[name][p0:p1] for p0, p1 in tiles] for name in ('V', 'W'))
    U1, U2 = ([g[name][p0:p1] for p0, p1 in tiles] for name in ('U1', 'U2'))
    Xn = [g['X_new'][p0:p1] for p0, p1 in tiles]
    Yn = [g['Y_new'][p0:p1] for p0, p1 in tiles]
    m0, k = M_GATE - K_GATE, K_GATE
    ritz = [ritz_rows(V[i], W[i], U1[i], U2[i], M_GATE, g['x'], g['y'], g['w'])
            for i in range(len(tiles))]
    dx = [np.vstack([r[2], r[3]]).T for r in ritz]
    addends = {
        'projection': [projection_addend(V[i], W[i], U1[i], U2[i], m0, M_GATE)
                       for i in range(len(tiles))],
        'residual norms': [r[4] for r in ritz],
        'correction norms': [np.linalg.norm(b, axis=1) ** 2 for b in dx],
        'overlaps': [overlap_addend(V[i], W[i], Xn[i], Yn[i])
                     for i in range(len(tiles))],
        'gram': [gram_addend(Xn[i], Yn[i]) for i in range(len(tiles))]}
    shapes = {'projection': (4, k, M_GATE), 'residual norms': (k,),
              'correction norms': (k,), 'overlaps': (2, M_GATE, k),
              'gram': (2, k, k)}
    return {name: (a, rows.join(a, shapes[name])) for name, a in addends.items()}


def recorded_partials(monkeypatch):
    """Patch the module's reduce_sum to record what each rank hands it."""
    seen = {}
    real = trial_space.reduce_sum

    def recording(a, comm):
        rank = comm.Get_rank() if comm is not None else 0
        seen.setdefault(rank, []).append(np.array(a, copy=True))
        return real(a, comm)

    monkeypatch.setattr(trial_space, 'reduce_sum', recording)
    return seen


def verdicts(size, g, seen, one_rank):
    """Every rank's `reduced_sum_verdict` of every reduced sum at `size`."""
    seen.clear()
    out = run_simulated(reduced_sums, size, g, TILE_GATE)
    n_tiles = -(-N_GATE // TILE_GATE)
    owners = [list(range(*contiguous_block(n_tiles, r, size)))
              for r in range(size)]
    result = []
    for rank, sums in enumerate(out):
        for i, name in enumerate(one_rank):
            addends, whole = one_rank[name]
            result.append((name, rank, reduced_sum_verdict(
                whole, addends, owners, rank, seen[rank][i], sums[name][1])))
    return result


def test_every_reduced_sum_is_within_its_join_bound(monkeypatch):
    """Each rank hands the reduction its own tiles' addends in tile order,
    and the reduced value lies within the join bound of the partials' exact
    sum, for every sum the loop reduces, at 2 to 64 ranks."""
    g = gate_arrays()
    one_rank = reduced_sums(None, g, TILE_GATE)
    seen = recorded_partials(monkeypatch)
    worst = {}
    for size in GATE_SIZES[1:]:
        for name, rank, v in verdicts(size, g, seen, one_rank):
            assert v.serial and v.partial, (size, name, rank)
            assert v.ratio <= 1.0, (size, name, rank, v.ratio)
            worst[name] = max(worst.get(name, 0.0), v.ratio)
    assert set(worst) == set(one_rank)


def test_a_misordered_or_short_partial_fails_the_verdict(monkeypatch):
    """The verdict is a gate: a rank adding its tiles in reverse fails the
    partial part, and a rank leaving out its last tile fails the bound."""
    g = gate_arrays()
    one_rank = reduced_sums(None, g, TILE_GATE)
    seen = recorded_partials(monkeypatch)
    real = PairRows.join

    def reversed_join(self, addends, shape):
        return real(self, addends[::-1], shape)

    def short_join(self, addends, shape):
        return real(self, addends[:-1] if self.rank == 1 else addends, shape)

    for plant, part in ((reversed_join, 'partial'), (short_join, 'ratio')):
        monkeypatch.setattr(PairRows, 'join', plant)
        failed = set()
        for name, rank, v in verdicts(3, g, seen, one_rank):
            if part == 'partial' and not v.partial:
                failed.add(name)
            if part == 'ratio' and v.ratio > 1.0:
                failed.add(name)
        assert failed == set(one_rank), (part, failed)
    monkeypatch.setattr(PairRows, 'join', real)


@pytest.mark.parametrize('size', EMPTY_SIZES)
def test_every_rank_takes_rank_zeros_decisions(model, size):
    """Gathered from every rank: the same cycles, batch sizes, kept
    directions, converged flags and residual norms, where ranks own no tile."""
    tile = 64
    n_tiles = -(-model[1].size // tile)
    assert size > n_tiles
    out = run_simulated(lambda comm: root(*model[:2], comm, pair_tile=tile),
                        size)
    ref = decisions(out[0][5])
    assert ref and out[0][3].all()
    for om, X, Y, conv, cycles, counters in out:
        assert decisions(counters) == ref
        assert cycles == out[0][4] == len(ref)
        assert bitwise(om, out[0][0]) and bitwise(X, out[0][1])


def test_water_decisions_where_ranks_own_no_tile(water):
    """The ISDF BSE on water at 16 ranks over 6 tiles: 10 ranks own no pair
    row and every rank takes rank 0's decisions."""
    size = 16
    out = run_simulated(lambda comm: bse_root(water, 3, comm,
                                              pair_tile=WATER_TILE), size)
    ref = decisions(out[0][5])
    n_pair = water['nocc'] * (len(water['eps']) - water['nocc'])
    assert len(PairRows(n_pair, None, WATER_TILE).tiles) < size
    for om, X, Y, conv, cycles, counters in out:
        assert decisions(counters) == ref and conv.all()
        assert bitwise(om, out[0][0]) and bitwise(Y, out[0][2])


@pytest.mark.parametrize('size', ROOT_SIZES)
def test_model_roots_at_every_rank_count(model, model_serial, size):
    """Within the Davidson's resolution of the serial roots, every root
    converged, from the serial number of cycles."""
    om_s = model_serial[0]
    out = run_simulated(lambda comm: root(*model[:2], comm,
                                          pair_tile=MODEL_TILE), size)
    om, X, Y, conv, cycles, counters = out[0]
    assert conv.all()
    assert relative(om, om_s) <= roots_resolution(model[2], NOCC, om_s)
    assert np.abs(om - model[3][:NROOTS]).max() <= CONV_TOL
    assert cycles == model_serial[4]


@pytest.mark.parametrize('size', [2, 3, 8, 16])
def test_water_roots_at_every_rank_count(water, size):
    """The ISDF BSE on water, 6 tiles over 2 to 16 ranks."""
    om_s, _, _, _, cyc_s, _ = bse_root(water, 3)
    out = run_simulated(lambda comm: bse_root(water, 3, comm,
                                              pair_tile=WATER_TILE), size)
    om, _, _, conv, cycles, _ = out[0]
    assert conv.all() and cycles == cyc_s
    assert relative(om, om_s) <= roots_resolution(water['eps'], water['nocc'],
                                                  om_s)


@pytest.mark.parametrize('size', [2, 3, 8])
def test_benzene_roots_at_every_rank_count(benzene, benzene_serial, size):
    """The ISDF BSE on benzene, 12 roots, 31 tiles over 2 to 8 ranks."""
    om_s, cyc_s = benzene_serial[0], benzene_serial[4]
    out = run_simulated(lambda comm: bse_root(benzene, 12, comm,
                                              pair_tile=BENZENE_TILE), size)
    om, _, _, conv, cycles, _ = out[0]
    assert conv.all() and cycles == cyc_s
    assert relative(om, om_s) <= roots_resolution(benzene['eps'],
                                                  benzene['nocc'], om_s)


@pytest.mark.parametrize('size', [1, 2, 3, 8, 64])
def test_holders_take_their_rows_alone(model, size):
    """32 bytes per owned pair row per trial pair on each rank, and one
    rank's worth over all of them."""
    tile = 64
    n_pair = model[1].size
    bound = davidson._real_eig_space(
        davidson._trial_space_memory(NROOTS, n_pair), NROOTS, n_pair)[1]

    def one_rank(comm):
        t = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            davidson._run_davidson(*model[:2], NROOTS, CONV_TOL, 100, None,
                                   comm=comm, timings=t, pair_tile=tile,
                                   pair_rows=True)
        rows = PairRows(n_pair, comm, tile)
        return t, rows.rows, max(p1 - p0 for p0, p1 in rows.tiles or [(0, 0)])

    out = (run_simulated(one_rank, size) if size > 1 else [one_rank(None)])
    total = 0
    for t, (p0, p1), longest in out:
        held = t['davidson_trial_space_mb'] * 1e6
        assert held == 32 * (p1 - p0) * bound
        assert longest <= tile
        total += held
    assert total == 32 * n_pair * bound
    t_serial = {}
    davidson._run_davidson(*model[:2], NROOTS, CONV_TOL, 100, None,
                           timings=t_serial)
    assert t_serial['davidson_trial_space_mb'] * 1e6 == 32 * n_pair * bound


def test_the_serial_path_is_pyscfs_real_eig(model, monkeypatch):
    """No comm, and a one-rank comm, run pyscf's real_eig and never the
    pair-row loop; two ranks run the loop and never real_eig."""
    calls = {'real_eig': 0, 'rows': 0}
    real, rows = davidson.real_eig, davidson.real_eig_rows

    def counted(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(davidson, 'real_eig', counted('real_eig', real))
    monkeypatch.setattr(davidson, 'real_eig_rows', counted('rows', rows))
    root(*model[:2])
    comm = simulated_world(1)[0]
    with distributed(comm):
        root(*model[:2], comm)
    assert calls == {'real_eig': 2, 'rows': 0}
    run_simulated(lambda comm: root(*model[:2], comm), 2)
    assert calls == {'real_eig': 2, 'rows': 2}


@pytest.mark.parametrize('size', [1, 3])
def test_restarts_apply_no_action(model, monkeypatch, size):
    """A bound forced small: every restart takes the Ritz vectors' products
    from the stored ones, so no action is applied on a restart cycle, and the
    roots are the dense ones."""
    n_pair = model[1].size
    monkeypatch.setattr(davidson.param, 'MAX_MEMORY', 1)
    monkeypatch.setattr(davidson, 'DAVIDSON_SPACE_GB',
                        (FORCED_SPACE + 0.5) * 32 * n_pair / 1e9)

    def one_rank(comm):
        return root(*model[:2], comm, max_cycle=FORCED_MAX_CYCLE,
                    pair_rows=True, pair_tile=100)

    out = run_simulated(one_rank, size) if size > 1 else [one_rank(None)]
    om, X, Y, conv, cycles, counters = out[0]
    trace = counters['trace']
    restarts = [s for s in trace[1:] if s['fresh']]
    assert conv.all() and restarts
    assert counters['collapses'] == len(restarts)
    assert all(s['applied'] == 0 and s['new'] == NROOTS for s in restarts)
    assert counters['vectors'] == 2 * sum(s['applied'] for s in trace)
    assert max(s['m1'] for s in trace) <= FORCED_SPACE
    assert np.abs(om - model[3][:NROOTS]).max() <= CONV_TOL


@pytest.mark.parametrize('size', [2, 3])
def test_a_drifted_subspace_solution_is_rank_zeros(model, monkeypatch, size):
    """The last rank's subspace solution moved one ulp in one cycle: the
    loop's checked lockstep hands it rank 0's, so every rank returns the
    unperturbed solve bitwise, and the audit counts that one repair there."""
    clean = run_simulated(lambda comm: root(*model[:2], comm,
                                            pair_tile=MODEL_TILE), size)
    real = trial_space.TDDFT_subspace_eigen_solver
    target = size - 1
    calls = {}

    def drifting(*args):
        omega, x, y = real(*args)
        rank = current_comm().Get_rank()
        calls[rank] = calls.get(rank, 0) + 1
        if rank == target and calls[rank] == PLANT_AT:
            i = np.unravel_index(np.abs(x).argmax(), x.shape)
            x[i] = np.nextafter(x[i], np.inf)
        return omega, x, y

    monkeypatch.setattr(trial_space, 'TDDFT_subspace_eigen_solver', drifting)

    def one_rank(comm):
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = root(*model[:2], comm, pair_tile=MODEL_TILE)
            return out, lockstep_stats()

    moved = run_simulated(one_rank, size)
    for r, (out, stats) in enumerate(moved):
        for a, b in zip(out[:3], clean[0][:3]):
            assert bitwise(a, b)
        assert stats['mismatched_calls'] == (1 if r == target else 0)


def test_the_subspace_time_is_split_into_its_pieces(model):
    """Under a comm the pieces are the subspace work's; serially zero."""
    def one_rank(comm):
        t = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            davidson._run_davidson(*model[:2], NROOTS, CONV_TOL, 100, None,
                                   comm=comm, timings=t, pair_tile=MODEL_TILE)
        return t

    for t in run_simulated(one_rank, 2):
        pieces = [t[f'davidson_subspace_{piece}'] for piece in SUBSPACE_PIECES]
        assert min(pieces) >= 0.0 and t['davidson_subspace_rows'] > 0.0
        assert t['davidson_subspace_reduce'] > 0.0
        assert t['davidson_subspace_gather'] > 0.0
        assert sum(pieces) <= t['davidson_subspace'] + TIMER_SLACK
    t = one_rank(None)
    assert all(t[f'davidson_subspace_{piece}'] == 0.0
               for piece in SUBSPACE_PIECES)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
