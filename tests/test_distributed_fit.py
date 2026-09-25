"""The row-distributed ISDF fit, `separable_factors(fit='rows')`: no rank holds
an nk-indexed array whole, and the rows are bitwise identical at every rank
count.

Gated over 1, 2, 3 and 8 simulated ranks (`run_simulated`, threads of this
process with the real collectives), on water and ethylene/cc-pVDZ Hartree-Fock
at the default 148 points per atom, with tiles of `TILE` points so that the
444- and 888-point grids are cut into 7 and 14 tiles and every rank count
owns a different set of them:

  * BITWISE ACROSS RANK COUNTS: every rank's rows of X_mo, D and X_ao are the
    same rows of the one-rank run, bit for bit, and the quasiparticle window
    and the whole ISDF BSE on them are the same distributed solves on the
    one-rank fit's whole arrays;
  * ROWS-ONLY MEMORY, read off the objects: the factors a rank holds are its
    `contiguous_block` of rows, and every grid-indexed array the fit held --
    the Gram tiles, F D^T, the collocation, the gathered panel, D before and
    after it moves -- is the size of the tiles the rank owns;
  * THE ANCHORED GATE against the replicated fit: D, the quasiparticle
    energies, W(0) and the BSE roots of the row fit sit within
    `FIT_REASSOCIATION_K` times the distance the replicated fit itself moves
    when its three-centre blocks are cut per shell and summed in reverse --
    a bar measured on this run, never a fixed number;
  * the new collectives the fit rests on (`reduce_max`, `broadcast_rows`,
    `allgather_ranges`, `cyclic_tiles_to_blocks`) against their serial
    meaning;
  * THE KEPT PAIRS' INTEGRALS ALONE (`KeptIntegrals`): every kept (mu nu|P)
    is the bits of the whole shell block's call over every nu shell, block
    by block and shell by shell on ethylene/cc-pVTZ (f shells break the
    runs) for the screened pairs and a sparse subset of them, the rows of
    the fit are bitwise those of the whole-block evaluation at 1 and 3
    ranks, and the ledger's `shell_block` holds between one and two times
    the largest kept block, below the whole block and its l <= 2 copy, when
    a loose screen drops pairs;
  * THE METRIC ROOT ON ONE RANK (`metric_root`, `RowFit.metric_root_rows`):
    the root is `aux_metric_sqrt`'s to 1e-12, bare and dressed, held in two
    metric-sized arrays; D on it is bitwise at 1/2/3/8 ranks with the root
    on rank 0 alone and a slab elsewhere, within the anchored bar of the
    replicated fit, and an indefinite dressed metric is refused on every
    rank;
  * A FROZEN LAYOUT in place of the screen (`layout=`, what a walk passes to
    keep the reference geometry's pairs): the geometry's own layout gives the
    screened rows bitwise at 1 and 3 ranks, another layout another D, and
    the replicated fit refuses one.

SHOWN TO FAIL, the kept-pair evaluation: one kept nu shell's run dropped
(its rows left unevaluated) fails the integral and the fit-level gates;
restored and byte-compared.

SHOWN TO FAIL: a tile edge that follows the rank's share (nk / 2 size points
in place of the fixed edge) fails the cross-rank gate at all six multi-rank
points, the one-rank points passing as they must; restored and
byte-compared. A rank's rows batched into one GEMM (the trailing update, the
three-centre contraction) passed here: this laptop's MKL gives those rows
the same bits at every call shape, which OpenBLAS does not, so that class of
defect shows under tests/test_distributed_fit_mpi.py on the cluster.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df, gto, scf

from src.Base import separable_ri
from src.Base.constants import (FIT_CHOLESKY_BLOCK, FIT_REASSOCIATION_K,
                                HARTREE_TO_EV)
from src.Base.polarizable_sites import PolarizableSites
from src.Base.separable_ri import (KeptIntegrals, aux_metric_sqrt,
                                   fit_M_streaming, metric_root)
from src.Base.solvent_screening import SolventScreening
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (allgather_ranges, broadcast_rows,
                                     contiguous_block, cyclic_tiles_to_blocks,
                                     distributed, partition, reduce_max,
                                     run_simulated)
from src.SingleReference.GW.space_time import (DEFAULT_COUNTS,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         solve_bse_isdf)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
#: Small enough that water's 444 points make 7 tiles and ethylene's 888 make
#: 14, so 2, 3 and 8 ranks each own a different mixture of them.
TILE = 64
SIZES = [1, 2, 3, 8]
NROOTS = 3


def mean_field(atom):
    """(mol, mf, nocc) for one molecule, DF Hartree-Fock."""
    mol = gto.M(atom=atom, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.kernel()
    return mol, mf, mol.nelectron // 2


@pytest.fixture(scope='module')
def water():
    return mean_field(WATER)


@pytest.fixture(scope='module')
def ethylene():
    return mean_field(ETHYLENE)


def row_factors(mol, mf, block=TILE):
    """The row fit in whatever region the caller is in."""
    return separable_factors(mf, mol, auxbasis=AUXBASIS, fit='rows',
                             fit_block=block)


def one_rank(mol, mf, block=TILE):
    """The row fit on one rank: the whole tuple every rank count is held to."""
    with distributed(None):
        return row_factors(mol, mf, block)


def rows_of(factors):
    """((r0, r1), X_mo, D, X_ao) of sliced or whole factors."""
    if isinstance(factors, SlicedFactors):
        return factors.rows, factors.X_mo, factors.D, factors.X_ao
    return (0, len(factors[3])), factors[0], factors[1], factors[2]


def bitwise(a, b):
    """Every array of `a` holds the bits of the same array of `b`."""
    return len(a) == len(b) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        for x, y in zip(a, b))


def relative(a, b):
    """||a - b|| / ||b||."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def tile_layout(npts, block, rank, size):
    """[(start, stop)] of the tiles `rank` owns."""
    return [(t * block, min((t + 1) * block, npts))
            for t in partition(-(-npts // block), rank, size)]


def observables(mf, mol, nocc, factors):
    """D, the quasiparticle window, W(0) and the BSE roots of whole
    factors, serially."""
    with distributed(None):
        window = np.array([nocc - 1, nocc])
        qp = solve_qp_energy_space_time(mf, mol, nocc, window, factors=factors)
        W = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
        om = solve_bse_isdf(mf, mol, nocc, nroots=NROOTS, factors=factors,
                            progress=False)[0]
    return {'D': factors[1], 'qp': qp, 'W0': W, 'bse': om}


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('case', ['water', 'ethylene'])
def test_rows_are_bitwise_across_rank_counts(case, size, request):
    mol, mf, _ = request.getfixturevalue(case)
    whole = one_rank(mol, mf)

    def rank(comm):
        return rows_of(row_factors(mol, mf))

    out = run_simulated(rank, size)
    npts = len(whole[3])
    for r, ((r0, r1), X_mo, D, X_ao) in enumerate(out):
        assert (r0, r1) == contiguous_block(npts, r, size)
        assert bitwise((X_mo, D, X_ao),
                       (whole[0][r0:r1], whole[1][r0:r1], whole[2][r0:r1])), (
            f'{case}: rank {r} of {size} rows differ from the one-rank fit')


@pytest.mark.parametrize('size', [2, 3, 8])
def test_each_rank_holds_rows_only(water, size):
    """THE MEMORY ASSERTION: the factors and every grid-indexed array the fit
    held are the rank's own tiles or rows, read off the arrays."""
    mol, mf, _ = water
    nao = mol.nao_nr()
    naux = df.addons.make_auxmol(mol, auxbasis=AUXBASIS).nao_nr()
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())

    def rank(comm):
        part = row_factors(mol, mf)
        return part.held_bytes(), part.fit_held, part.nmo

    out = run_simulated(rank, size)
    npts = out[0][0]['coords'] // (3 * 8)
    whole_S = npts * npts * 8
    for r, (held, fit, nmo) in enumerate(out):
        r0, r1 = contiguous_block(npts, r, size)
        assert held == {'X_mo': (r1 - r0) * nmo * 8, 'D': (r1 - r0) * naux * 8,
                        'X_ao': (r1 - r0) * nao * 8, 'coords': npts * 3 * 8}
        own = tile_layout(npts, TILE, r, size)
        rows = sum(b - a for a, b in own)
        gram = sum((b - a) * b * 8 for a, b in own)
        panels = sum((npts - b) * (b - a) * 8 for a, b in own)
        diagonal = sum((b - a) ** 2 * 8 for a, b in own)
        e0 = (r0 // TILE) * TILE
        e1 = min(-(-r1 // TILE) * TILE, npts)
        assert fit['X_rows'] == rows * nao * 8
        assert fit['B_rows'] == rows * n2 * 8
        assert fit['aux_rows'] == rows * naux * 8
        assert fit['FD_rows'] == rows * naux * 8
        assert fit['D_tiles'] == rows * naux * 8
        assert fit['D_rows'] == (r1 - r0) * naux * 8
        assert fit['X_ext'] == (e1 - e0) * nao * 8
        assert gram <= fit['S_rows'] <= gram + panels + diagonal, (r, fit)
        assert fit['S_rows'] < whole_S, (r, fit['S_rows'], whole_S)
        assert fit['panel'] <= (npts - TILE) * TILE * 8
        assert fit['stream_tile'] <= TILE * (nao + n2 + naux) * 8
        assert fit['metric_lu'] == naux * naux * 8     # not grid-indexed
        # water is one shell block, computed by rank 0, every pair kept: its
        # integrals and the one call over every nu shell
        assert fit['shell_block'] == (2 * nao * n2 * naux * 8 if r == 0
                                      else 0), (r, fit['shell_block'])
        for name in ('X_rows', 'aux_rows', 'FD_rows', 'D_tiles', 'D_rows',
                     'X_ext'):
            ncol = nao if name.startswith('X') else naux
            assert fit[name] < npts * ncol * 8, (r, name)


@pytest.mark.parametrize('block', [TILE, FIT_CHOLESKY_BLOCK])
@pytest.mark.parametrize('case', ['water', 'ethylene'])
def test_row_fit_within_the_reassociation_bar(case, block, request,
                                              monkeypatch):
    """The row fit against the replicated fit, anchored on the replicated
    fit's own response to one reordering of its three-centre sum."""
    mol, mf, nocc = request.getfixturevalue(case)
    with distributed(None):
        old = separable_factors(mf, mol, auxbasis=AUXBASIS)
    with monkeypatch.context() as patch:
        per_shell_reversed = [(s, s + 1) for s in reversed(range(mol.nbas))]
        patch.setattr(separable_ri, 'ao_blocks',
                      lambda *args, **kwargs: per_shell_reversed)
        with distributed(None):
            reassociated = separable_factors(mf, mol, auxbasis=AUXBASIS)
    new = one_rank(mol, mf, block)
    ref = observables(mf, mol, nocc, old)
    bar = observables(mf, mol, nocc, reassociated)
    got = observables(mf, mol, nocc, new)
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    ref['D_root'], bar['D_root'] = ref['D'], bar['D']
    with distributed(None):
        got['D_root'] = fit_M_streaming(mol, auxmol, new[3], fit='rows',
                                        block=block).metric_root_rows(auxmol)
    lines = []
    for key in ref:
        anchor = relative(bar[key], ref[key])
        dist = relative(got[key], ref[key])
        # A quantity the reordering leaves bitwise must stay bitwise: its
        # bar is zero, and so must the row fit's distance be.
        lines.append(f'{key}: reassociation {anchor:.2e}, row fit {dist:.2e}'
                     + (f' ({dist / anchor:.2f} x)' if anchor else ''))
        assert dist <= FIT_REASSOCIATION_K * anchor, lines[-1]
    worst = {k: float(np.abs(np.asarray(got[k]) - np.asarray(ref[k])).max())
             * HARTREE_TO_EV for k in ('qp', 'bse')}
    print(f'\n{case}, tile {block}: ' + '; '.join(lines)
          + f"; max |dE| qp {worst['qp']:.2e} eV, bse {worst['bse']:.2e} eV")


@pytest.mark.parametrize('size', [2, 3])
def test_consumers_on_row_factors_are_the_one_rank_fits(water, size):
    """The window and the whole ISDF BSE on each rank's rows are, bit for
    bit, the same distributed solves on the one-rank row fit's whole arrays,
    and every rank holds rank 0's answer."""
    mol, mf, nocc = water
    whole = one_rank(mol, mf)
    window = np.array([nocc - 1, nocc])

    def rank(comm):
        part = row_factors(mol, mf)
        out = []
        for fac in (part, whole):
            qp = solve_qp_energy_space_time(mf, mol, nocc, window, factors=fac)
            om = solve_bse_isdf(mf, mol, nocc, nroots=NROOTS, factors=fac,
                                progress=False)[0]
            out.append((qp, om))
        return out

    res = run_simulated(rank, size)
    for r, (on_rows, on_whole) in enumerate(res):
        assert bitwise(on_rows, on_whole), f'rank {r}: rows != whole'
        assert bitwise(on_rows, res[0][0]), f'rank {r} != rank 0'


def test_unknown_fit_is_refused(water):
    mol, mf, _ = water
    with distributed(None):
        with pytest.raises(ValueError, match='rows'):
            separable_factors(mf, mol, auxbasis=AUXBASIS, fit='cyclic')


@pytest.mark.parametrize('size', [1, 2, 3, 8])
def test_new_collectives(size):
    rng = np.random.default_rng(size)
    npts, block, ncol = 37, 5, 3
    whole = rng.normal(size=(npts, ncol))
    peaks = rng.normal(size=(size, 11))

    def rank(comm):
        r = comm.Get_rank()
        top = peaks[r].copy()
        reduce_max(top, comm)
        sent = whole[:4].copy() if r == size - 1 else np.zeros((4, ncol))
        broadcast_rows(sent, size - 1, comm)
        owned = [tile_layout(npts, block, p, size) for p in range(size)]
        gathered = np.zeros_like(whole)
        for a, b in owned[r]:
            gathered[a:b] = whole[a:b]
        allgather_ranges(gathered, owned, comm)
        tiles = {a // block: whole[a:b].copy() for a, b in owned[r]}
        return top, sent, gathered, cyclic_tiles_to_blocks(tiles, npts, block,
                                                           ncol, comm)

    for r, (top, sent, gathered, rows) in enumerate(run_simulated(rank, size)):
        r0, r1 = contiguous_block(npts, r, size)
        assert np.array_equal(top, peaks.max(axis=0))
        assert np.array_equal(sent, whole[:4])
        assert np.array_equal(gathered, whole)
        assert np.array_equal(rows, whole[r0:r1])


def whole_block(mol, auxmol, shells, mu, nu):
    """The kept pairs' (mu nu|P) the way the pass took them before it
    evaluated them alone: one `aux_e2` call over the whole shell block and
    every nu shell, building its own optimizer, the kept pairs picked."""
    sh0, sh1 = shells
    e3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                           shls_slice=(sh0, sh1, 0, mol.nbas, 0, auxmol.nbas))
    return np.ascontiguousarray(e3c[mu - mol.ao_loc_nr()[sh0], nu])


@pytest.mark.parametrize('atom, basis', [(ETHYLENE, 'cc-pvtz'),
                                         (WATER, BASIS)],
                         ids=['ethylene-tz', 'water-dz'])
def test_kept_integrals_are_the_whole_block_bits(atom, basis):
    """Every kept pair's (mu nu|P), evaluated over its runs of kept nu shells
    alone, is the bits of the whole shell block's call, for the fit's own
    blocks and for single shells, on the screened pairs and on a sparse
    subset of them that leaves gaps inside every run; no call holds more
    pairs than are kept."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis=basis + '-ri')
    rows = {el: separable_ri.atomic_grid(el, basis, basis + '-ri',
                                         DEFAULT_COUNTS)
            for el in {mol.atom_pure_symbol(i) for i in range(mol.natm)}}
    grid = separable_ri.molecular_points_covariant(
        mol, {el: row[0] for el, row in rows.items()},
        origin_by_element={el: row[1] for el, row in rows.items()})
    mu_all, nu_all, _ = separable_ri.test_set_layout(mol, grid)
    ao_loc = mol.ao_loc_nr()
    integrals = KeptIntegrals(mol, auxmol)
    naux = auxmol.nao_nr()
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())
    blocks = (separable_ri.ao_blocks(mol, len(grid), n2, naux, 4.0)
              + [(s, s + 1) for s in range(mol.nbas)])
    rng = np.random.default_rng(7)
    checked = 0
    for sh0, sh1 in blocks:
        sel = (mu_all >= ao_loc[sh0]) & (mu_all < ao_loc[sh1])
        for pick in (np.ones(sel.sum(), bool),
                     rng.random(sel.sum()) < 0.3):
            mu, nu = mu_all[sel][pick], nu_all[sel][pick]
            if not len(mu):
                continue
            got, call = integrals((sh0, sh1), mu, nu)
            want = whole_block(mol, auxmol, (sh0, sh1), mu, nu)
            assert got.flags.c_contiguous, (sh0, sh1)
            assert got.tobytes() == want.tobytes(), (
                f'shells {sh0}:{sh1}, {len(mu)} pairs')
            widest = max(ao_loc[j + 1] - ao_loc[j] for j in range(mol.nbas))
            assert call <= max(len(mu), (ao_loc[sh1] - ao_loc[sh0]) * widest
                               ) * naux * 8
            checked += 1
    assert checked > 2 * mol.nbas


@pytest.mark.parametrize('size', [1, 3])
def test_row_fit_on_kept_pairs_is_the_whole_block_fit(ethylene, size,
                                                      monkeypatch):
    """The rows of the fit whose pass evaluates the kept pairs alone are, bit
    for bit, the rows of the same fit taking them from whole-block calls."""
    mol, mf, _ = ethylene
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)

    def rank(comm):
        return rows_of(row_factors(mol, mf))

    kept = run_simulated(rank, size)
    monkeypatch.setattr(KeptIntegrals, '__call__',
                        lambda self, shells, mu, nu: (
                            whole_block(mol, auxmol, shells, mu, nu), 0))
    whole = run_simulated(rank, size)
    for r, (a, b) in enumerate(zip(kept, whole)):
        assert a[0] == b[0] and bitwise(a[1:], b[1:]), f'rank {r} of {size}'


@pytest.mark.parametrize('size', [1, 3])
def test_shell_block_holds_the_kept_pairs(ethylene, size):
    """`shell_block`, read off the arrays: the rank that computed the block's
    coefficients held its kept pairs' integrals and one call no larger than
    they, or than one nu shell where the kept pairs are fewer -- below the
    whole block over every nu and its l <= 2 copy, at the default screen and
    at one loose enough to drop most pairs."""
    mol, mf, _ = ethylene
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    naux, nao = auxmol.nao_nr(), mol.nao_nr()
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())
    widest = max(mol.ao_loc_nr()[s + 1] - mol.ao_loc_nr()[s]
                 for s in range(mol.nbas) if mol.bas_angular(s) <= 2)
    for pair_tol in (separable_ri.DEFAULT_PAIR_TOL, 1e-2):
        def rank(comm):
            fit = fit_M_streaming(mol, auxmol, coords, fit='rows',
                                  block=TILE, pair_tol=pair_tol)
            return fit.held['shell_block']

        coords = one_rank(mol, mf)[3]
        mu, _, _ = separable_ri.test_set_layout(mol, coords,
                                                pair_tol=pair_tol)
        held = run_simulated(rank, size)
        # ethylene is one shell block of every AO, and rank 0 computes it; a
        # call carries at least one nu shell
        kept = len(mu) * naux * 8
        one_shell = nao * widest * naux * 8
        assert kept <= held[0] <= kept + max(kept, one_shell), (
            pair_tol, held[0], kept)
        assert all(h == 0 for h in held[1:]), held
        assert held[0] < nao * (nao + n2) * naux * 8, (held[0], kept)


def test_metric_root_is_aux_metric_sqrt_to_rounding(water):
    """The root formed in two metric-sized arrays is `aux_metric_sqrt`'s,
    bare and dressed by a continuum, to 1e-12, and exactly symmetric."""
    mol, _, _ = water
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    naux = auxmol.nao_nr()
    for env in (None, SolventScreening(mol, solvent='water')):
        root, held = metric_root(auxmol, env)
        ref = aux_metric_sqrt(auxmol, env)
        assert relative(root, ref) < 1e-12, relative(root, ref)
        assert np.array_equal(root, root.T)
        assert held == 2 * naux * naux * 8


@pytest.mark.parametrize('size', SIZES)
def test_metric_root_rows_are_bitwise_across_rank_counts(water, size):
    """D = M^T V^1/2 with the root on rank 0 alone and a slab of it
    elsewhere: every rank's rows bitwise the one-rank rows, the root held on
    rank 0 only, and D the whole-metric product to rounding."""
    mol, mf, _ = water
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    naux = auxmol.nao_nr()
    coords = one_rank(mol, mf)[3]
    with distributed(None):
        fit = fit_M_streaming(mol, auxmol, coords, fit='rows', block=TILE)
        whole = fit.metric_root_rows(auxmol)
        assert relative(whole, fit.metric_rows(aux_metric_sqrt(auxmol))) \
            < 1e-12

    def rank(comm):
        part = fit_M_streaming(mol, auxmol, coords, fit='rows', block=TILE)
        return part.rows, part.metric_root_rows(auxmol), dict(part.held)

    for r, ((r0, r1), D, held) in enumerate(run_simulated(rank, size)):
        assert D.tobytes() == whole[r0:r1].tobytes(), f'rank {r} of {size}'
        assert held['metric_root'] == (2 * naux * naux * 8 if r == 0 else 0)
        assert held['metric_slab'] == min(TILE, naux) * naux * 8
        assert held['D_rows'] == (r1 - r0) * naux * 8


def test_an_indefinite_dressed_root_is_refused_on_every_rank(water):
    """A reaction field that over-screens the bare interaction is refused by
    the rank that forms the root, and every other rank raises with it
    rather than waiting for a slab."""
    mol, mf, _ = water
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    coords = one_rank(mol, mf)[3]
    runaway = PolarizableSites(np.array([[0.0, 0.0, 7.0]]),
                               np.array([1000.0]), unit='Bohr')

    def rank(comm):
        fit = fit_M_streaming(mol, auxmol, coords, fit='rows', block=TILE)
        try:
            fit.metric_root_rows(auxmol, runaway)
        except RuntimeError as err:
            return str(err)
        return None

    for r, msg in enumerate(run_simulated(rank, 2)):
        assert msg is not None and 'indefinite' in msg, (r, msg)


@pytest.mark.parametrize('size', [1, 3])
def test_a_frozen_layout_replaces_the_screen(ethylene, size):
    """A frozen `test_set_layout` in place of the screen: the geometry's own
    layout gives the screened fit's rows bitwise at every rank, and a layout
    holding other pairs gives another D. The replicated fit refuses one."""
    mol, mf, _ = ethylene
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    coords = one_rank(mol, mf)[3]
    own = separable_ri.test_set_layout(mol, coords)
    other = tuple(a[::2] for a in own)
    root = aux_metric_sqrt(auxmol)

    def fitted(layout):
        def rank(comm):
            fit = fit_M_streaming(mol, auxmol, coords, fit='rows',
                                  block=TILE, layout=layout)
            return fit.rows, fit.metric_rows(root)
        return run_simulated(rank, size)

    screened, frozen, moved = fitted(None), fitted(own), fitted(other)
    for r, (a, b, c) in enumerate(zip(screened, frozen, moved)):
        assert a[0] == b[0] and a[1].tobytes() == b[1].tobytes(), r
        assert relative(c[1], a[1]) > 1e-3, (r, relative(c[1], a[1]))
    with distributed(None), pytest.raises(ValueError, match="fit='rows'"):
        fit_M_streaming(mol, auxmol, coords, layout=own)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
