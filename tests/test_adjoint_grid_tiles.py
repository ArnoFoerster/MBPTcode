"""The GW adjoint sweeps over (tau, grid-row tile), the factors' adjoints held
in the grid's fixed tiles end to end (`GridTileRows`).

The static-W adjoint (`chi0_backward_rows`) and the quasiparticle set's
reverse sweep (`polarizability_backward_rows`) are tile-major: each rank owns
the tiles t % size of `block` points, the row fit's tiles and owners, and
sweeps every tau point for them, so X_bar and D_bar are output partitions and
no rank holds a pair whole. The BSE grid adjoint's rows are moved into the
same tiles, and the row fit's adjoint and X_mo^T X_bar read them there.

Gated:
  * `GridTileRows` moves rows verbatim: from whole arrays, from contiguous
    blocks, gathered, broadcast by tile, added (1/2/3/5/8 ranks, ranks with
    no tile included);
  * the tile-major sweep is the same bits at 1/2/3/5/8 simulated ranks, on
    whole and on sliced factors, projbar whole or `ProjRows`;
  * against the tau-split sweep it lies within the worst-case rounding of
    the two evaluation orders, (M + 2 naux + nmo + 8) ulp of the same sweep
    on the absolute values, with no safety factor; projbar moved by 1e-10
    relative fails it;
  * `fit_rows_adjoint` and `orbital_rotation_rows` read tiles bitwise the
    whole adjoint's result at 2/3/8 ranks;
  * the chain gates (water): the quasiparticle tape read back bitwise and a
    forward on it gathering nothing; the ranked force against one rank and
    a 4-point difference of the ranked energy; the census of what a rank
    holds through the reverse pass; tracemalloc totals.
"""
import hashlib
import os
import sys
import tracemalloc
import types

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (ISDF_GRADIENT_FLOOR, NUCLEAR_FD_STEP,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.separable_ri import fit_rows_adjoint
from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.mpi_grid import (contiguous_block, distributed,
                                     run_simulated)
from src.SingleReference.LinearResponse.space_time import ProjRows
from src.gradients.factor_chain import FrozenFactorization
from src.gradients.isdf_derivatives import orbital_rotation_rows
from src.gradients.qp_space_time import qp_set_gradient
from src.gradients.space_time_adjoint import (chi0_backward,
                                              chi0_backward_rows,
                                              polarizability_backward,
                                              polarizability_backward_rows)
from src.properties.excitations import SurfaceSpec, surface_of
from tests.test_chain_row_fit import TILE, molecule
from tests.test_chain_sliced_factors import H2O

SIZES = (1, 2, 3, 5, 8)
ULP = np.finfo(float).eps
#: Random factors: 300 points in 5 tiles of 64, so 8 ranks leave 3 empty.
M, NMO, NOCC, NAUX, NTAU, BLOCK = 300, 20, 6, 40, 4, 64


def random_case(seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, NMO)) * 0.3
    D = rng.standard_normal((M, NAUX)) * 0.3
    eps = np.sort(rng.uniform(-1.0, 1.0, NMO))
    eps[:NOCC] -= 1.0
    grid = types.SimpleNamespace(
        tau_points=np.linspace(0.1, 2.0, NTAU), ntau=NTAU,
        cosft_wt=rng.standard_normal((1, NTAU)))
    chi0_bar = rng.standard_normal((NAUX, NAUX))
    proj_bar = rng.standard_normal((NTAU, NAUX, NAUX))
    return X, D, eps, grid, chi0_bar + chi0_bar.T, proj_bar


def same(a, b):
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes()
               and np.shape(x) == np.shape(y) for x, y in zip(a, b))


# ----------------------------------------------------------------- layouts
@pytest.mark.parametrize('size', SIZES)
def test_grid_tile_rows_move_rows_verbatim(size):
    whole = np.random.default_rng(2).standard_normal((M, 7))

    def rank(comm):
        c = comm if size > 1 else None
        tiles = GridTileRows.from_whole(whole, BLOCK, c)
        r0, r1 = contiguous_block(M, comm.Get_rank(), comm.Get_size())
        moved = GridTileRows.from_blocks(whole[r0:r1], M, BLOCK, c)
        tiles += whole
        broadcast = [tiles.broadcast_tile(t) for t in range(len(tiles.bounds))]
        return moved.gather(), tiles.gather(), broadcast, moved.nbytes

    for moved, doubled, broadcast, nbytes in run_simulated(rank, size):
        assert same([moved], [whole])
        assert same([doubled], [whole + whole])
        assert same(broadcast, [2.0 * whole[t0:t0 + BLOCK]
                                for t0 in range(0, M, BLOCK)])
        assert nbytes <= -(-5 // size) * BLOCK * 7 * 8


def test_grid_tile_rows_refuse_a_whole_read():
    tiles = GridTileRows.zeros(M, 3, BLOCK)
    with pytest.raises(TypeError):
        np.asarray(tiles)


# -------------------------------------------------- the tile-major sweep
def sweeps(comm, case, sliced, proj_rows):
    X, D, eps, grid, chi0_bar, proj_bar = case
    if sliced and comm.Get_size() > 1:
        f = SlicedFactors.from_whole((X, D, X, np.zeros((M, 3))), comm)
        Xs, Ds = f, f
    else:
        Xs, Ds = X, D
    c = comm if comm.Get_size() > 1 else None
    e1, x1, d1 = chi0_backward_rows(chi0_bar[None], Xs, Ds, eps, NOCC, grid,
                                    block=BLOCK, comm=c)
    pb = proj_bar
    if proj_rows:
        r0, r1 = contiguous_block(NAUX, comm.Get_rank(), comm.Get_size())
        pb = ProjRows(np.ascontiguousarray(proj_bar[:, r0:r1]), NAUX, c)
    e2, x2, d2 = polarizability_backward_rows(pb, Xs, Ds, eps, NOCC, grid,
                                              block=BLOCK, comm=c)
    return e1, x1.gather(), d1.gather(), e2, x2.gather(), d2.gather()


@pytest.mark.parametrize('sliced', [False, True], ids=['whole', 'sliced'])
@pytest.mark.parametrize('proj_rows', [False, True], ids=['array', 'rows'])
def test_tile_major_sweep_is_the_same_bits_at_every_rank_count(sliced,
                                                               proj_rows):
    case = random_case()
    one = run_simulated(lambda comm: sweeps(comm, case, sliced, proj_rows),
                        1)[0]
    for size in SIZES[1:]:
        out = run_simulated(lambda comm: sweeps(comm, case, sliced,
                                                proj_rows), size)
        for r, got in enumerate(out):
            assert same(got, one), f'rank {r} of {size}'


def absolute_bound(fn, X, D, eps, grid, bar):
    """(M + 2 naux + nmo + 8) ulp of the tau-split sweep on the absolute
    values: every product of either evaluation order sums at most that many
    terms along any path, each bounded by the absolute sweep's term."""
    ref = fn(np.abs(bar), np.abs(X), np.abs(D), eps, NOCC, grid)
    return [(M + 2 * NAUX + NMO + 8) * ULP * np.abs(r) for r in ref]


def within(got, ref, bound):
    return max(float((np.abs(g - r) / np.maximum(b, 1e-300)).max())
               for g, r, b in zip(got, ref, bound))


def test_tile_major_sweep_lies_within_its_rounding_of_the_tau_split_sweep():
    X, D, eps, grid, chi0_bar, proj_bar = random_case()
    ratios = {}
    for name, old, new, bar in (
            ('chi0', chi0_backward, chi0_backward_rows, chi0_bar[None]),
            ('proj', polarizability_backward, polarizability_backward_rows,
             proj_bar)):
        ref = old(bar, X, D, eps, NOCC, grid)
        e, x, d = new(bar, X, D, eps, NOCC, grid, block=BLOCK, comm=None)
        got = (e, x.gather(), d.gather())
        bound = absolute_bound(old, X, D, eps, grid, bar)
        ratios[name] = within(got, ref, bound)
        assert ratios[name] <= 1.0, (name, ratios)
        # the bound can fail: projbar moved by 1e-10 relative
        e, x, d = new(bar * (1 + 1e-10), X, D, eps, NOCC, grid, block=BLOCK,
                      comm=None)
        assert within((e, x.gather(), d.gather()), ref, bound) > 1.0, name
    print('\nworst |new - tau split| / bound:', ratios)


# ------------------------------------------------- the fit adjoint reads tiles
def test_fit_adjoint_and_y_read_tiles_bitwise():
    """`fit_rows_adjoint` on D_bar and X_bar tiles and `orbital_rotation_rows`
    on X_bar tiles give the whole adjoints' results bitwise at 2/3/8 ranks."""
    mol = molecule(H2O)
    with distributed(None):
        fac = FrozenFactorization(mol)
    aux, crd = fac.auxmol(mol), fac.coords(mol)
    npts, naux, nao = len(crd), aux.nao_nr(), mol.nao
    rng = np.random.default_rng(3)
    d_bar = rng.normal(size=(npts, naux))
    x_bar = rng.normal(size=(npts, nao))
    C = rng.normal(size=(nao, nao))
    x_mo = mol.eval_gto('GTOval_sph', crd) @ C

    def rank(comm, tiled):
        d = GridTileRows.from_whole(d_bar, TILE, comm) if tiled else d_bar
        x = GridTileRows.from_whole(x_bar, TILE, comm) if tiled else x_bar
        adj = fit_rows_adjoint(mol, aux, crd, d, fac.layout, x_bar=x,
                               mo_coeff=C, block=TILE)
        rows = SlicedFactors.from_whole((x_mo, d_bar, x_mo, crd), comm)
        y = orbital_rotation_rows(rows, x, TILE)
        return (adj.fit_centre, adj.fit_points, adj.coll_centre,
                adj.coll_points, y)

    for size in (2, 3, 8):
        ref = run_simulated(lambda comm: rank(comm, False), size)
        got = run_simulated(lambda comm: rank(comm, True), size)
        for r in range(size):
            assert same(got[r], ref[r]), f'rank {r} of {size}'


# ------------------------------------------------------------ the chain
#: The chain gates' tile edge and working-set budget: water's grid in tiles
#: of 64, the forward's tiles below M, so a whole (M, M) array is a finding.
CHAIN_BLOCK = 64
CHAIN_TILE_GB = 3e-4
#: The components the 4-point difference of the ranked energy checks.
FD_COMPONENTS = ((0, 2), (1, 1))


def water():
    return gto.M(atom=H2O, basis='cc-pvdz', verbose=0)


def hf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def ladder_chain(mol):
    """The SOP-scissor S1 surface on Hartree-Fock water, the row fit in
    tiles of `CHAIN_BLOCK`."""
    spec = SurfaceSpec(GroundState('dft', 'hf'), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics={'sliced': True, 'fit': 'rows',
                                 'fit_block': CHAIN_BLOCK,
                                 'tile_gb': CHAIN_TILE_GB,
                                 'bse_adjoint': 'grid'})
    return surface_of(spec, Excitation('singlet', root=1, kernel='bse'), mol,
                      hf_factory)


def tau_split_path(chain):
    """The chain on the tau-split ranked path: whole adjoints, the tau-split
    sweeps, no tape."""
    chain._adjoint_rows = lambda x_mo: None
    forward = chain._forward

    def untaped(mol, mf):
        om, pieces = forward(mol, mf)
        return om, pieces[:16] + (None,)

    chain._forward = untaped
    return chain


def at(mol, atom, axis, step):
    crd = mol.atom_coords().copy()
    crd[atom, axis] += step
    return mol.set_geom_(crd, unit='Bohr', inplace=False)


def test_the_tape_is_read_back_bitwise():
    """At 1 and 2 ranks: the reverse solve on the forward's tape gives the
    untaped reverse bitwise, whole and in tiles; a forward on a tape that
    applies gathers no factor."""

    def rank(comm):
        mol = water()
        chain = ladder_chain(mol)
        mf = chain.mf0
        om, pieces = chain._forward(mol, mf)
        x_mo, d, eps, mu, tape = pieces[4], pieces[5], pieces[6], pieces[12], \
            pieces[16]
        xc = chain._xc_correction(mf, chain.qp_set, None)
        weights = np.linspace(0.3, 1.0, len(chain.qp_set))
        kw = chain._qp_kw()
        args = (x_mo, d, eps, chain.nocc, chain.gw_grid, chain.nu, chain.wt,
                chain.qp_set)
        out = {}
        for block in (None, chain._adjoint_rows(x_mo)):
            if comm.Get_size() == 1 and block is not None:
                block = CHAIN_BLOCK
            runs = []
            for t in (tape, None):
                got = qp_set_gradient(*args, weights, mu=mu, xc_correction=xc,
                                      tape=t, rows_block=block, **kw)
                runs.append([got[0], got[1]] + [
                    a.gather() if isinstance(a, GridTileRows) else np.array(a)
                    for a in got[2:]])
            out[block] = runs
        before = dict(getattr(x_mo, 'gathers', {}))
        qp_set_gradient(*args, np.zeros(len(chain.qp_set)), mu=mu,
                        xc_correction=xc, tape=tape, **kw)
        after = dict(getattr(x_mo, 'gathers', {}))
        return out, before, after, tape is not None

    for size in (1, 2):
        for out, before, after, taped in run_simulated(rank, size):
            assert taped
            for block, (with_tape, without) in out.items():
                assert same(with_tape, without), (size, block)
            assert before == after, (before, after)


def ranked_forces(size, tau_split=False):
    def rank(comm):
        mol = water()
        chain = ladder_chain(mol)
        if tau_split:
            tau_split_path(chain)
        grad, _ = chain.excitation_gradient()
        return np.array(grad)
    return run_simulated(rank, size)


def test_the_ranked_force_is_the_tau_split_and_one_ranks():
    """At 2 and 3 ranks: every rank's force is rank 0's, within
    ISDF_GRADIENT_FLOOR of the tau-split ranked path and of the one-rank
    (serial) path."""
    one = ranked_forces(1)[0]
    lines = []
    for size in (2, 3):
        new, old = ranked_forces(size), ranked_forces(size, tau_split=True)
        for g in new[1:]:
            assert np.array_equal(g, new[0])
        d_old = float(np.abs(new[0] - old[0]).max())
        d_one = float(np.abs(new[0] - one).max())
        lines.append(f'{size} ranks: |F - tau split| {d_old:.1e}, '
                     f'|F - one rank| {d_one:.1e}')
        assert max(d_old, d_one) < ISDF_GRADIENT_FLOOR, lines
    assert np.abs(one).max() > 1e-3
    print('\n' + '; '.join(lines))


def test_the_ranked_force_follows_its_energy():
    """At 2 ranks: the force against a 4-point difference of the chain's
    own energy, evaluated over the same ranks."""
    def rank(comm):
        mol = water()
        chain = ladder_chain(mol)
        grad, _ = chain.excitation_gradient()
        out = []
        for atom, axis in FD_COMPONENTS:
            om = {k: chain.excitation(
                      at(mol, atom, axis, k * NUCLEAR_FD_STEP))
                  for k in (2, 1, -1, -2)}
            fd = ((8.0 * (om[1] - om[-1]) - (om[2] - om[-2]))
                  / (12.0 * NUCLEAR_FD_STEP))
            out.append(float(grad[atom, axis] - fd))
        return np.array(out)

    diff = run_simulated(rank, 2)[0]
    print(f'\n2 ranks: analytic - FD {np.array2string(diff, precision=2)}')
    assert np.abs(diff).max() < ISDF_GRADIENT_FLOOR, diff


class AdjointCensus:
    """This thread's census, at every line of every frame under src/ between
    `root` and the running line, of the arrays bound to a name: shapes whole
    along the grid twice ('grid square'), and (M, ncol) arrays of a factor's
    width whose bytes, once the traced call returns, are no factor's ('whole
    adjoint': a gather's buffer is judged filled, an accumulator summed)."""

    def __init__(self, root, M, widths, factors):
        self.root, self.M, self.widths = root, M, set(widths)
        self.allowed = {self.digest(a) for a in factors}
        self.flags, self.seen = {}, {}

    @staticmethod
    def digest(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()

    def kind(self, a):
        if sum(n == self.M for n in a.shape) >= 2:
            return 'grid square'
        if a.ndim == 2 and a.shape[0] == self.M and a.shape[1] in self.widths:
            return 'wide'
        return None

    def verdict(self):
        """{kind: {(shape, frame)}} of what the census found."""
        flags = {k: set(v) for k, v in self.flags.items()}
        for a, where in self.seen.values():
            if self.digest(a) not in self.allowed:
                flags.setdefault('whole adjoint', set()).update(where)
        return flags

    def take(self, frame):
        chain = []
        while frame is not None and frame.f_code is not self.root:
            chain.append(frame)
            frame = frame.f_back
        if frame is None:
            return
        for f in chain + [frame]:
            for v in list(f.f_locals.values()):
                for a in self.arrays(v):
                    kind = self.kind(a)
                    where = (a.shape, f.f_code.co_name)
                    if kind == 'wide':
                        # held, so its id stays its own until the verdict
                        self.seen.setdefault(id(a), (a, set()))[1].add(where)
                    elif kind is not None:
                        self.flags.setdefault(kind, set()).add(where)

    def arrays(self, v, depth=0):
        if isinstance(v, np.ndarray):
            base = v
            while isinstance(base.base, np.ndarray):
                base = base.base
            return [base]
        if depth < 2 and isinstance(v, (list, tuple)) and len(v) < 64:
            return [a for x in v for a in self.arrays(x, depth + 1)]
        if depth < 2 and isinstance(v, dict):
            return [a for x in v.values() for a in self.arrays(x, depth + 1)]
        if depth < 2 and isinstance(v, GridTileRows):
            rows = getattr(v, 'rows', None)      # None while it is built
            return [] if rows is None else [rows]
        return []

    def trace(self, fn, *args):
        src = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'src')

        def local(frame, event, arg):
            if event == 'line':
                self.take(frame)
            return local

        def tracer(frame, event, arg):
            return (local if frame.f_code.co_filename.startswith(src)
                    else None)

        sys.settrace(tracer)
        try:
            return fn(*args)
        finally:
            sys.settrace(None)


def reverse(chain, pieces):
    """The excitation force's reverse pass: the BSE seeds, then everything
    `_fold_to_nuclei` folds them through; (force, rows the seed holds)."""
    seeds = chain._casida_seeds(pieces, chain.state)
    held = (seeds[1].rows if isinstance(seeds[1], GridTileRows)
            else seeds[1]).shape[0]
    return chain._fold_to_nuclei(pieces, *seeds)[0], held


@pytest.mark.parametrize('size', [2, 3, 8])
def test_no_rank_holds_a_whole_adjoint_or_a_grid_square(size):
    """Through the reverse pass of the excitation force -- the BSE seeds, the
    quasiparticle reverse, the static-W adjoint and the nuclear assembly --
    no rank binds an (M, M) array or an (M, nmo)/(M, naux) array that is not
    one of the factors gathered whole."""
    def rank(comm):
        mol = water()
        chain = ladder_chain(mol)
        mf = chain.mf0
        om, pieces = chain._forward(mol, mf)
        x_mo = pieces[4]
        eps = np.asarray(mf.mo_energy)
        nocc = chain.nocc
        X = x_mo.gather('X_mo')
        factors = [X, x_mo.gather('D'), x_mo.gather('X_ao'),
                   np.ascontiguousarray(X[:, :nocc]),
                   np.ascontiguousarray(X[:, nocc:])]
        census = AdjointCensus(reverse.__code__, x_mo.npts,
                               (len(eps), x_mo.naux, nocc, len(eps) - nocc),
                               factors)
        _, held = census.trace(reverse, chain, pieces)
        return census.verdict(), held, x_mo.npts

    for r, (flags, held, npts) in enumerate(run_simulated(rank, size)):
        assert not flags, (r, flags)
        assert held < npts, (r, held, npts)


def test_the_reverse_pass_holds_less_than_the_tau_split_path():
    """tracemalloc over 3 ranks sharing one process: the excitation force's
    peak on the tiles is below the tau-split ranked path's, which holds the
    pair whole on every rank."""

    def peak(tau_split):
        def rank(comm):
            mol = water()
            chain = ladder_chain(mol)
            if tau_split:
                tau_split_path(chain)
            om, pieces = chain._forward(mol, chain.mf0)
            comm.allgather(None)
            if comm.Get_rank() == 0:
                tracemalloc.reset_peak()
            comm.allgather(None)
            reverse(chain, pieces)
            return pieces[4].npts, pieces[4].nmo, pieces[4].naux
        tracemalloc.start()
        try:
            shape = run_simulated(rank, 3)[0]
            return tracemalloc.get_traced_memory()[1], shape
        finally:
            tracemalloc.stop()

    new, (npts, nmo, naux) = peak(False)
    old, _ = peak(True)
    pair = npts * (nmo + naux) * 8
    print(f'\npeak over 3 ranks: tiles {new / 1e6:.1f} MB, tau split '
          f'{old / 1e6:.1f} MB, one pair {pair / 1e6:.2f} MB')
    assert new < old - pair, (new, old, pair)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
