"""The grid BSE adjoint (`LinearResponse.isdf_bse_adjoint`) distributed over
simulated ranks (`run_simulated`), RHF/cc-pVDZ at 148 points per atom, on the
chain's own roots and vectors, the factors cut by `SlicedFactors.from_whole`:

  (a) eps_bar, X_bar, D_bar and W_bar of dOmega_0 and of the interstate
      element at 1, 2, 3 and 8 ranks are the one-rank run's bits on every
      rank (singlet and triplet, full and TDA, water and ethylene, 256- and
      64-point tiles); the composed force and interstate element of a sliced
      chain with `bse_adjoint='grid'` are bitwise those of the same chain with
      the grid adjoint run serially on the gathered factors;
  (b) a line-by-line trace of every frame under src/ finds no whole-grid
      array outside the boundary gather (`adjoints_at_the_boundary`), no
      whole-factor gather, and the kernel's held-bytes ledger equal to this
      rank's rows and tiles; the same scan finds the whole factors of the
      serial-replicated realization, so it can fail;
  (c) with the kernel's GEMMs scaled by a function of their call shape
      (`ShapeSensitiveNumpy`), every rank is still bitwise the one-rank run;
  (d) each tile is computed by one rank and broadcast once per pass (column
      and W_bar), with three halo and two hand-back row exchanges.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import gto, lib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import BSE_ADJOINT_TILE_ROWS
from src.Base.sliced_factors import GridTileRows, SlicedFactors, whole_factor
from src.Base.utils.mpi_grid import (contiguous_block, distributed,
                                     run_simulated)
from src.SingleReference.LinearResponse import isdf_bse_adjoint
from src.SingleReference.LinearResponse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_interstate_backward)
from src.gradients import excited_state
from src.gradients.excited_state import ExcitedStateChain
from tests.test_bse_grid_adjoint import ETHYLENE, grid_bound_ratios
from tests.test_chain_sliced_factors import (BASIS, H2O, H2O_DISPLACED,
                                             bitwise, chain_scf)

SIZES = [2, 3, 8]
#: The default tile and one that cuts water's 444 points into 7 tiles.
TILES = [256, 64]
MOLECULES = {'water': H2O, 'ethylene': ETHYLENE}
CASES = [
    # (molecule, spin, tda, solver)
    ('water', 'singlet', False, 'davidson'),
    ('water', 'triplet', True, 'dense'),
    ('ethylene', 'singlet', False, 'davidson'),
]
#: The shape-sensitive stand-in's relative change per unit of a GEMM's rows,
#: columns and inner dimension (tens of units move a result by ~1e-7).
SHAPE_SKEW = 2.0 ** -30
#: How far a tile-shape change moves the result under the stand-in at the
#: least; a last-bit reassociation moves it ~1e-15.
TILE_SHAPE_MOVES = 1e-10
#: The frames the scan traces, and the one gather it allows.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'src') + os.sep
BOUNDARY = 'adjoints_at_the_boundary'


class ShapeSensitiveNumpy:
    """numpy, save a matmul whose result depends on the call's shape: scaled
    by 1 + (rows + 2 columns + 3 inner) * SHAPE_SKEW. A kernel whose GEMM
    shapes follow a rank's share of the grid gives different ranks different
    results under it; one whose shapes are fixed by its tiles does not."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def matmul(a, b, out=None):
        res = np.matmul(a, b, out=out)
        rows = a.shape[0] if np.ndim(a) > 1 else 1
        cols = b.shape[-1] if np.ndim(b) > 1 else 1
        res *= 1.0 + (rows + 2 * cols + 3 * a.shape[-1]) * SHAPE_SKEW
        return res


class GridScan:
    """Every array with the whole grid on an axis that any frame under src/
    holds at any line of a call -- its locals and the containers they hold --
    outside the named boundary gather and what it calls."""

    def __init__(self, npts):
        self.npts = int(npts)
        self.found = {}
        self.boundary_calls = 0

    def visit(self, frame, value, depth=0):
        if isinstance(value, np.ndarray):
            if value.ndim and self.npts in value.shape:
                key = (frame.f_code.co_name,
                       os.path.basename(frame.f_code.co_filename), value.shape)
                self.found.setdefault(key, frame.f_lineno)
        elif depth < 3 and isinstance(value, dict):
            for v in value.values():
                self.visit(frame, v, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for v in value:
                self.visit(frame, v, depth + 1)

    def local(self, frame, event, arg):
        if event in ('line', 'return'):
            for value in frame.f_locals.values():
                self.visit(frame, value)
        return self.local

    def enter(self, frame, event, arg):
        if not frame.f_code.co_filename.startswith(SRC):
            return None
        if frame.f_code.co_name == BOUNDARY:
            self.boundary_calls += 1
        f = frame
        while f is not None:
            if f.f_code.co_name == BOUNDARY:
                return None
            f = f.f_back
        return self.local

    def run(self, fn, *args, **kwargs):
        held = sys.gettrace()
        sys.settrace(self.enter)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.settrace(held)


def own_molecule(name):
    return gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def inputs():
    """{case: (X_mo, D, eps_qp, W_aux, nocc, Xn, Yn, spin, tda)} of the
    chain's own forward pass, serially."""
    out = {}
    with distributed(None):
        for name, spin, tda, solver in CASES:
            mol = own_molecule(name)
            chain = ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol),
                                      spin=spin, bse_tda=tda, solver=solver)
            _, pieces = chain._forward(chain.mol0, chain.mf0)
            x, d, eq, w, no, _, xn, yn = chain._casida_args(pieces)
            out[(name, spin, tda)] = (x, d, eq, w, no, xn, yn, spin, tda)
    return out


def sliced(x, d, comm):
    """This rank's rows of whole X_mo and D (X_ao and the points unread)."""
    return SlicedFactors.from_whole(
        (x, d, np.zeros((len(x), 1)), np.zeros((len(x), 3))), comm)


def adjoint(case, comm=None, bra=None, tile=256, stats=None):
    """The kernel on this rank's rows (whole arrays without a comm) of copies
    of the case's inputs: dOmega_0, or <bra| dH |0> symmetrized."""
    x, d, eq, w, no, xn, yn, spin, tda = case
    f = (x, d) if comm is None else (sliced(x, d, comm),) * 2
    args = (eq.copy(), w.copy(), no, xn.copy(), yn.copy())
    kw = dict(spin=spin, bse_tda=tda, tile_rows=tile, stats=stats)
    if bra is None:
        return isdf_bse_backward(0, *f, *args, **kw)
    return isdf_interstate_backward(bra, 0, *f, *args, **kw)


def rel(a, b):
    """max |a - b| relative to max |b|."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()
                 / max(np.abs(np.asarray(b)).max(), 1e-300))


# ------------------------------------------------ (a) the same bits
@pytest.mark.parametrize('case', [c[:3] for c in CASES],
                         ids=lambda c: '-'.join(map(str, c)))
@pytest.mark.parametrize('tile', TILES)
def test_every_rank_count_gives_the_one_rank_bits(inputs, case, tile):
    """eps_bar, X_bar, D_bar and W_bar of dOmega_0 and of <1| dH |0> at 2, 3
    and 8 ranks: on every rank the one-rank run's bits."""
    for bra in (None, 1):
        with distributed(None):
            one = adjoint(inputs[case], bra=bra, tile=tile)
        for size in SIZES:
            got = run_simulated(
                lambda comm: adjoint(inputs[case], comm, bra=bra, tile=tile),
                size)
            for r, out in enumerate(got):
                assert bitwise(out, one), (case, tile, bra, size, r)
    print(f'[info] {case} tile {tile}: 1/2/3/8 ranks bitwise, dOmega and '
          'the interstate element')


def serial_replicated(kernel, first_factor, **override):
    """`kernel` run whole by every rank, serially, on the gathered factors:
    the serial-replicated realization of the grid adjoint. `calls` counts
    its calls; `override` replaces keyword arguments (a tile)."""

    def run(*args, **kwargs):
        run.calls.append(1)
        args = list(args)
        args[first_factor] = whole_factor(args[first_factor], 'X_mo')
        args[first_factor + 1] = whole_factor(args[first_factor + 1], 'D')
        with distributed(None):
            return kernel(*args, **dict(kwargs, **override))

    run.calls = []
    return run


def serial_replicated_rows(**override):
    """`isdf_bse_backward_rows` (the chain's grid adjoint over more than one
    rank, dOmega_n and, with `bra`, the interstate element) realized by
    `serial_replicated`: the whole adjoint on every rank, cut to its
    `contiguous_block` rows."""
    whole = serial_replicated(isdf_bse_backward, 1, **override)

    def run(*args, comm=None, **kwargs):
        e_bar, x_bar, d_bar, w_bar = whole(*args, **kwargs)
        r0, r1 = contiguous_block(len(x_bar), comm.Get_rank(),
                                  comm.Get_size())
        return e_bar, x_bar[r0:r1], d_bar[r0:r1], w_bar

    run.calls = whole.calls
    return run


@pytest.fixture
def pyscf_one_thread():
    """pyscf's OpenMP GEMM on one thread, so its K partials are summed in one
    order and two runs of a chain repeat their bits; a no-op without OpenMP."""
    threads = lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(1)
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(threads)


def mean_fields(size):
    """factory(rank): `chain_scf` converging once per geometry on each rank
    and handing that mean field to every chain there, so two runs share it
    and differ only in the adjoint."""
    converged = [{} for _ in range(size)]

    def factory_of(rank):
        def factory(mol):
            key = np.asarray(mol.atom_coords()).tobytes()
            if key not in converged[rank]:
                converged[rank][key] = chain_scf(mol)
            return converged[rank][key]
        return factory

    return factory_of


def grid_forces(comm, factory_of):
    """(excitation force at the displaced water, Omega, interstate element)
    of a sliced chain with the grid adjoint, on this rank's own Mole."""
    factory = factory_of(comm.Get_rank())
    mol = own_molecule('water')
    chain = ExcitedStateChain(mol, factory, mf=factory(mol),
                              solver='davidson', sliced=True,
                              bse_adjoint='grid')
    here = gto.M(atom=H2O_DISPLACED, basis=BASIS, verbose=0)
    g_ex, d_ex = chain.excitation_gradient(here)
    g_st, _ = chain.interstate_gradient(0, 1)
    return g_ex, d_ex['omega'], g_st


@pytest.mark.parametrize('size', [1] + SIZES)
def test_the_composed_force_is_the_replicated_adjoints(size, monkeypatch,
                                                       pyscf_one_thread):
    """The sliced chain's excitation force and interstate element with the
    distributed grid adjoint are the bits of the same chain with the grid
    adjoint run serially by every rank on the gathered factors."""
    factory_of = mean_fields(size)
    dist = run_simulated(grid_forces, size, factory_of)
    omega = serial_replicated(isdf_bse_backward, 1)
    element = serial_replicated(isdf_interstate_backward, 2)
    monkeypatch.setattr(excited_state, 'isdf_bse_backward', omega)
    monkeypatch.setattr(excited_state, 'isdf_interstate_backward', element)
    rows = serial_replicated_rows()
    monkeypatch.setattr(excited_state, 'isdf_bse_backward_rows', rows)
    rep = run_simulated(grid_forces, size, factory_of)
    # serially the whole kernels, over ranks the rows kernel, twice a rank
    assert len(omega.calls) + len(element.calls) + len(rows.calls) == 2 * size
    for r in range(size):
        assert bitwise(dist[r], rep[r]), f'rank {r} of {size}: {dist[r]}'
        assert bitwise(dist[r], dist[0]), f'rank {r} of {size} != rank 0'
    print(f'[info] {size} ranks: composed force and interstate element '
          f'bitwise the serial-replicated adjoint on every rank; '
          f'|g| {np.abs(dist[0][0]).max():.6e}')


def traced_grid_forces(comm, factory_of):
    """`grid_forces`, recording at every reverse call this chain's seeds
    beside the default-tile kernel's run serially on the gathered factors and
    folded on the same pieces: (the forces, the records)."""
    factory = factory_of(comm.Get_rank())
    mol = own_molecule('water')
    chain = ExcitedStateChain(mol, factory, mf=factory(mol),
                              solver='davidson', sliced=True,
                              bse_adjoint='grid')
    calls = []
    seeds_of, fold = chain._casida_seeds, chain._fold_to_nuclei

    def seeds(pieces, n, m=None):
        out = seeds_of(pieces, n, m)
        x, d, eq, w, no, _, xn, yn = chain._casida_args(pieces)
        X, D = whole_factor(x, 'X_mo'), whole_factor(d, 'D')
        kw = dict(spin=chain.spin, bse_tda=chain.bse_tda)
        with distributed(None):
            ref = (isdf_bse_backward(n, X, D, eq, w, no, xn, yn, **kw)
                   if m is None else
                   isdf_interstate_backward(m, n, X, D, eq, w, no, xn, yn,
                                            **kw))
        calls.append(dict(n=n, m=m, ref=ref, inputs=(X, D, eq, w, no, xn, yn),
                          spin=chain.spin, tda=chain.bse_tda,
                          seeds=tuple(a.gather() if isinstance(a, GridTileRows)
                                      else np.array(a, copy=True)
                                      for a in out)))
        return out

    def folded(pieces, *own, **fold_kw):
        # the refold first, keeping the tape the probed fold may release
        refold = np.asarray(fold(pieces, *[
            np.array(a, copy=True) for a in calls[-1]['ref']],
            release_tape=False)[0])
        result = fold(pieces, *own, **fold_kw)
        calls[-1]['refold'] = refold
        return result

    chain._casida_seeds, chain._fold_to_nuclei = seeds, folded
    here = gto.M(atom=H2O_DISPLACED, basis=BASIS, verbose=0)
    g_ex, d_ex = chain.excitation_gradient(here)
    g_st, _ = chain.interstate_gradient(0, 1)
    return (g_ex, d_ex['omega'], g_st), calls


def test_the_force_carries_the_adjoints_bits(monkeypatch, pyscf_one_thread):
    """At 2 ranks the replicated adjoint in 128-point tiles moves the composed
    force off the distributed bits (so the bitwise gate can fail), by its
    seeds alone: the root is unchanged, the fold of the default-tile seeds on
    the reblocked run's pieces is the distributed force bitwise, and the
    reblocked seeds lie within the derived rounding bound
    (`grid_bound_ratios`). The force difference is only printed: the fit
    adjoint carries last-bit input changes to ~1e-8 Ha/Bohr."""
    factory_of = mean_fields(2)
    dist = run_simulated(grid_forces, 2, factory_of)
    monkeypatch.setattr(excited_state, 'isdf_bse_backward',
                        serial_replicated(isdf_bse_backward, 1,
                                          tile_rows=128))
    monkeypatch.setattr(excited_state, 'isdf_interstate_backward',
                        serial_replicated(isdf_interstate_backward, 2,
                                          tile_rows=128))
    monkeypatch.setattr(excited_state, 'isdf_bse_backward_rows',
                        serial_replicated_rows(tile_rows=128))
    moved = run_simulated(traced_grid_forces, 2, factory_of)
    for r, (forces, calls) in enumerate(moved):
        d = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                for a, b in zip(forces, dist[r]))
        assert not bitwise(forces, dist[r]), f'rank {r}: nothing moved'
        assert forces[1] == dist[r][1], f'rank {r}: the root moved'
        assert len(calls) == 2, len(calls)
        for label, call, g in zip(('excitation', 'interstate'), calls,
                                  (dist[r][0], dist[r][2])):
            ratios = grid_bound_ratios(call['ref'], call['seeds'],
                                       call['inputs'], call['spin'],
                                       call['tda'], call['n'], bra=call['m'],
                                       tiles=(BSE_ADJOINT_TILE_ROWS, 128))
            same = np.array_equal(call['refold'], g)
            print(f'[info] rank {r} {label}: 128-point tiles move the force '
                  f'{d:.2e} Ha/Bohr, the seeds X / D / W at {ratios[0]:.3f} / '
                  f'{ratios[1]:.3f} / {ratios[2]:.3f} of the derived bound; '
                  f'the fold of the default-tile seeds bitwise the '
                  f'distributed force: {same}')
            assert same, (r, label)
            assert np.array_equal(call['seeds'][0], call['ref'][0]), label
            assert max(ratios) <= 1, (r, label, ratios)


# ------------------------------------------------ (b) the memory scan
def expected_ledger(npts, tile, size, rank, naux, nmo, no, nv, nT, bare):
    """(bytes by name, this rank's tiles, the tile count) of the kernel's
    arrays from the layout alone: tiles of `tile` points, each computed by
    the rank whose contiguous block holds its first row."""
    blocks = [contiguous_block(npts, r, size) for r in range(size)]
    bounds = [(t0, min(t0 + tile, npts)) for t0 in range(0, npts, tile)]
    owner = [next(r for r, (b0, b1) in enumerate(blocks) if b0 <= t0 < b1)
             for t0, _ in bounds]
    r0, r1 = blocks[rank]
    n = r1 - r0
    mine = [t for t, o in enumerate(owner) if o == rank]
    rows = sum(bounds[t][1] - bounds[t][0] for t in mine)
    past = sum(bounds[t][1] - bounds[t][0] for t in mine if bounds[t][1] > r1)
    others = [b1 - b0 for (b0, b1), o in zip(bounds, owner) if o != rank]
    want = {'X_o_rows': n * no, 'X_v_rows': n * nv, 'DW': rows * naux,
            'SD': rows * naux, 'AT': rows * nT * no,
            'accumulators': rows * (1 + nT) * no, 'X_bar_rows': n * nmo,
            'D_bar_rows': n * naux, 'halo': past * (naux + no + nv),
            'hand_back': past * (nmo + naux), 'W': naux * naux,
            'W_bar': naux * naux, 'X_bar_whole': npts * nmo,
            'D_bar_whole': npts * naux,
            'column_tile': max(others, default=0) * (naux + no + nT * no),
            'w_bar_tile': max(others, default=0) * 2 * naux,
            'b_partials': len(bounds) * 2 * naux if bare else 0}
    return {k: 8 * v for k, v in want.items()}, mine, len(bounds)


def grid_widths(naux, nmo, no, nv, nT):
    """The width beside the grid of every grid-indexed ledger entry."""
    return {'X_o_rows': no, 'X_v_rows': nv, 'DW': naux, 'SD': naux,
            'AT': nT * no, 'accumulators': (1 + nT) * no, 'X_bar_rows': nmo,
            'D_bar_rows': naux, 'halo': naux + no + nv,
            'hand_back': nmo + naux, 'column_tile': naux + no + nT * no,
            'w_bar_tile': 2 * naux}


def scanned(case, tile):
    """Per rank: (scan, stats, gathers) of one traced dOmega_0 on rows."""

    def rank(comm):
        x, d, eq, w, no, xn, yn, spin, tda = case
        f = sliced(x, d, comm)
        scan, stats = GridScan(len(x)), {}
        scan.run(isdf_bse_backward, 0, f, f, eq.copy(), w.copy(), no,
                 xn.copy(), yn.copy(), spin=spin, bse_tda=tda,
                 tile_rows=tile, stats=stats)
        return scan, stats, dict(f.gathers)

    return rank


@pytest.mark.parametrize('case', [c[:3] for c in CASES],
                         ids=lambda c: '-'.join(map(str, c)))
@pytest.mark.parametrize('tile', TILES)
@pytest.mark.parametrize('size', SIZES)
def test_no_rank_holds_a_whole_grid_array(inputs, case, tile, size):
    """(b) and (d): no whole-grid array at any line of the kernel outside
    the boundary gather, no whole-factor gather, the held-bytes ledger this
    rank's rows and tiles, and each tile streamed once per pass."""
    x, d, eq, w, no, xn, yn, spin, tda = inputs[case]
    npts, nmo, naux = x.shape[0], x.shape[1], d.shape[1]
    nv, nT = nmo - no, 2 if tda else 4
    widths = grid_widths(naux, nmo, no, nv, nT)
    owned = []
    for r, (scan, stats, gathers) in enumerate(
            run_simulated(scanned(inputs[case], tile), size)):
        assert scan.found == {}, (f'rank {r} of {size}: whole-grid arrays '
                                  f'{sorted(scan.found.items())}')
        assert scan.boundary_calls == 1, scan.boundary_calls
        assert gathers == {}, f'rank {r} gathered {gathers}'
        want, mine, nt = expected_ledger(npts, tile, size, r, naux, nmo, no,
                                         nv, nT, spin == 'singlet')
        held = stats['held']
        faults = [f'{k} {held.get(k, 0)} != {v}' for k, v in want.items()
                  if held.get(k, 0) != v]
        faults += [f'{k} {held.get(k, 0)} holds the whole grid'
                   for k, width in widths.items()
                   if held.get(k, 0) >= npts * width * 8]
        assert not faults, f'rank {r} of {size}: {faults}'
        assert stats['tiles'] == mine
        assert stats['streams'] == {'columns': nt, 'w_bar': nt}
        others = nt - len(mine)
        assert (stats['received'].get('columns', 0),
                stats['received'].get('w_bar', 0)) == (others, others)
        assert stats['exchanges'] == {'halo': 3, 'hand_back': 2}
        owned += mine
    assert sorted(owned) == list(range(nt)), owned
    print(f'[info] {case} tile {tile}, {size} ranks: no whole-grid array, '
          f'no gather, the ledger the tiles; {nt} tiles streamed once per '
          'pass')


def test_the_scan_sees_the_replicated_realization(inputs):
    """The same scan through the serial-replicated realization at 2 ranks
    finds X_mo and D gathered whole and the kernel's whole-grid arrays."""
    x, d, eq, w, no, xn, yn, spin, tda = inputs[('water', 'singlet', False)]
    run = serial_replicated(isdf_bse_backward, 1)

    def rank(comm):
        f = sliced(x, d, comm)
        scan = GridScan(len(x))
        scan.run(run, 0, f, f, eq.copy(), w.copy(), no, xn.copy(), yn.copy())
        return scan

    for r, scan in enumerate(run_simulated(rank, 2)):
        shapes = {shape for _, _, shape in scan.found}
        print(f'[info] rank {r}: the replicated realization holds {shapes}')
        assert {x.shape, d.shape} <= shapes, shapes


# ------------------------------------------------ (c) the shape-sensitive BLAS
@pytest.mark.parametrize('case', [c[:3] for c in CASES[:2]],
                         ids=lambda c: '-'.join(map(str, c)))
def test_a_shape_sensitive_blas_moves_no_rank(inputs, case, monkeypatch):
    """Under the stand-in every rank at 2, 3 and 8 ranks is the one-rank
    stand-in run's bits; the stand-in moves the result, and a 128-point tile
    moves it by more than TILE_SHAPE_MOVES."""
    with distributed(None):
        real = adjoint(inputs[case], bra=1)
    monkeypatch.setattr(isdf_bse_adjoint, 'np', ShapeSensitiveNumpy())
    with distributed(None):
        one = adjoint(inputs[case], bra=1)
        half = adjoint(inputs[case], bra=1, tile=128)
    moved = max(rel(a, b) for a, b in zip(one[1:], real[1:]))
    by_tile = max(rel(a, b) for a, b in zip(half[1:], one[1:]))
    print(f'[info] {case}: the stand-in moves the adjoints {moved:.1e}, a '
          f'128-point tile under it {by_tile:.1e}')
    assert moved > TILE_SHAPE_MOVES and by_tile > TILE_SHAPE_MOVES
    for size in SIZES:
        for r, out in enumerate(run_simulated(
                lambda comm: adjoint(inputs[case], comm, bra=1), size)):
            assert bitwise(out, one), (case, size, r)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
