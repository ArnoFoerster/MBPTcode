"""The gradient chains on the row-distributed fit: `FrozenFactorization(
sliced=True, fit='rows')` builds each rank's grid rows of X_mo, D and X_ao
with `separable_ri.fit_rows` on the frozen points, so no rank forms the fit
whole at any geometry, and the rows are bitwise the same at every rank count.

Gated over 1, 2, 3 and 8 simulated ranks (`run_simulated`, threads of this
process with the real collectives) on water/cc-pVDZ Hartree-Fock at 148
points per atom, in tiles of `TILE` points so that the 444-point grid is cut
into 7 tiles and every rank count owns a different set of them, each rank on
its own Mole:

  (a) the rows every rank holds, at the reference and at a displaced
      geometry, are bitwise the same rows of a one-rank run of the chain;
  (b) THE ANCHORED GATE: against the whole-fit sliced chain at the SAME rank
      count, the composed state-pair force, energy and root at a displaced
      geometry, the BSE roots at the reference, the dRPA force and a
      one-cycle optimizer step sit within `FIT_REASSOCIATION_K` times the
      distance the whole-fit chain itself moves when the sums of its fit
      over the test set are cut per shell and accumulated in reverse -- a bar
      measured on the run;
  (c) THE MEMORY SCAN: at every evaluation the walk asks for, before it and
      right after each factor build inside it, no array reachable from the
      surface has the whole grid on one axis and a factor's or the fit's
      width on the other (X_mo, X_ao, D, M, F D^T, the Gram matrix), and
      every rows object is the row fit's own, its ledger naming every
      grid-indexed array of the fit at this rank's tiles; and at EVERY
      FORCE, the nuclear assembly traced line by line in every frame under
      src/ (`traced`): no array it names holds the grid by a factor's, the
      fit's or the test set's width (the Gram matrix, the test set's
      collocation, M, F D^T, a whole adjoint) beyond the X_bar and D_bar
      handed in, nor the dense (mu nu|P) where the fit's shell blocks cut
      the AO index, and the adjoint's own ledger (`adjoint_held`) names
      every grid-indexed array of the fit adjoint at this rank's tiles;
  (d) the whole-array gathers of one composed force are the whole-fit
      sliced chain's, {X_mo 7, D 8, X_o 3, X_v 2}, less the X_mo of its two
      nuclear assemblies, which stream X_mo by tiles on the row fit.

WATER CANNOT TELL THE TWO ESTIMATORS APART, which is why ethylene is here.
The row fit's Gram matrix is the unscreened product (A A^T) o (B B^T) and its
F D^T keeps the screened pairs; `FrozenFactorization._fit` screens both. At
water/cc-pVDZ the screen keeps every pair (576 of 576) and the two are one
estimator; at ethylene/cc-pVDZ it drops 72 of 2304 and they are 2.0e-4 of D
apart. So on ethylene the row-fit chain is gated against its own estimator
formed whole (every product pair in D, one Cholesky of the whole Gram
matrix) on that fit's own reassociation bar; its dRPA force against a
five-point finite difference of its own energy is gated in
tests/test_chain_row_fit_adjoint.py, beside the adjoint's own gates.

SHOWN TO FAIL, then restored and byte-compared (`cmp`): `row_fit_factors`
falling back to the whole fit -- `FrozenFactorization._fit` at the geometry,
D = M^T V^(1/2) formed whole and the rows cut from it
(`SlicedFactors.from_whole`) -- failed (c) at 2, 3 and 8 ranks on every scan
(rows without the fit's ledger), while (a), (b) and (d) stayed green: the
same rows and the same numbers, only formed whole.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto

from src.Base.constants import FIT_REASSOCIATION_K, HARTREE_TO_EV
from src.Base.declaration import Excitation, GroundState
# the module, not its test_* helpers, which pytest would collect as tests
from src.Base import separable_ri
from src.Base.separable_ri import (DEFAULT_REGULARIZATION, aux_metric_sqrt,
                                   fit_M_stable)
from src.Base.sliced_factors import SlicedFactors, whole_factor
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import (contiguous_block, distributed, lockstep,
                                     partition, run_simulated)
from src.gradients import factor_chain
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            dfactor_adjoint_gauges,
                                            pair_positions, product_pairs)
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import optimize
from src.properties.surface import evaluate
from src.properties.surfaces import potential_energy_surface
from tests.test_chain_sliced_factors import (COMPOSED_GATHERS, H2O,
                                             H2O_DISPLACED, chain_scf,
                                             own_water, reachable_arrays,
                                             reachable_rows)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#: The package the memory scan traces: every frame whose code lives here.
SRC = os.path.join(REPO, 'src') + os.sep
SIZES = [2, 3, 8]
BASIS = 'cc-pvdz'
#: Water's 444 points in 7 tiles, ethylene's 888 in 14.
TILE = 64
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')
#: One hydrogen 0.03 A along y.
ETHYLENE_DISPLACED = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.953 1.238; '
                      'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; '
                      'H 0.0 -0.923 -1.238')
#: The finite-difference step of the dRPA gate, Bohr: the five-point stencil
#: of tests/test_rpa_ground_state.py.
FD_STEP = 1e-4
#: The Cartesian components of ethylene the finite difference takes: an H
#: in-plane, a C along the bond, another H along the bond.
FD_COMPONENTS = ((2, 1), (0, 2), (3, 2))
#: The Davidson residual of the anchored gates, below the fit's response. At
#: the default `BSE_DAVIDSON_CONV_TOL` the solver's own convergence moved
#: ethylene's second root 7.0e-10 Ha between the row fit and its estimator
#: formed whole, where one reassociation of the fit moves it 5.6e-13; at this
#: residual the two are 2.1e-13 and 3.6e-13.
GATE_BSE_CONV_TOL = 1e-9
#: The whole-array gathers of one composed force on the row fit: the
#: whole-fit chain's, less the X_mo of its two nuclear assemblies, which
#: stream X_mo by tiles instead (`orbital_rotation_rows`).
ROW_FIT_GATHERS = dict(COMPOSED_GATHERS, X_mo=COMPOSED_GATHERS['X_mo'] - 2)


def molecule(atom):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def row_kw(block=TILE):
    return dict(sliced=True, fit='rows', fit_block=block)


def state_pair(atom=H2O, **kw):
    """The composed singlet surface on this rank's OWN Mole, the Casida step
    matrix-free so the block action runs on rows."""
    mol = molecule(atom)
    return RPABSESurface(mol, chain_scf, spin='singlet', mf=chain_scf(mol),
                         solver='davidson', **kw)


def rows_of(factors):
    """((r0, r1), X_mo, D, X_ao) of sliced or whole factors_at output."""
    x_mo, d, _, _, crd, x_ao = factors
    if isinstance(x_mo, SlicedFactors):
        return x_mo.rows, x_mo.X_mo, x_mo.D, x_mo.X_ao
    return (0, len(crd)), x_mo, d, x_ao


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


def reassociated_fit(mol, layout):
    """`fit_M_stable` with its three sums over the test set -- the row norms,
    the Gram matrix and F Dt^T -- cut per mu shell and accumulated in
    reverse: one reordering of the whole fit's own sums, the gate's anchor."""
    mu = np.asarray(layout[0])
    ao_loc = mol.ao_loc_nr()
    shells = [np.flatnonzero((mu >= ao_loc[s]) & (mu < ao_loc[s + 1]))
              for s in reversed(range(mol.nbas))]
    shells = [c for c in shells if len(c)]

    def fit(D, F, regularization=DEFAULT_REGULARIZATION):
        blocks = shells + [np.arange(len(mu), D.shape[1])]
        s2 = np.zeros(D.shape[0])
        for c in blocks:
            s2 += np.einsum('kr,kr->k', D[:, c], D[:, c])
        s = np.sqrt(s2)
        d = 1.0 / np.where(s == 0.0, 1.0, s)
        Dt = D * d[:, None]
        G = np.zeros((D.shape[0], D.shape[0]))
        FD = np.zeros((F.shape[0], D.shape[0]))
        for c in blocks:
            G += Dt[:, c] @ Dt[:, c].T
            FD += F[:, c] @ Dt[:, c].T
        G[np.diag_indices_from(G)] += regularization
        cho = scipy.linalg.cho_factor(G, lower=True)
        return scipy.linalg.cho_solve(cho, FD.T).T * d[None, :]

    return fit


def widths_of(surface):
    """(npts, the widths a whole factor or fit array carries beside it)."""
    chain = surface.ground
    npts = chain.M
    return npts, {chain.mf0.mo_coeff.shape[1], chain.mol0.nao, chain.naux,
                  npts}


def whole_fit_arrays(surface):
    """Shapes of the reachable float arrays with the whole grid on one axis
    and a factor's or the fit's width on the other: X_mo, X_ao, D, M,
    F D^T or the Gram matrix, whole."""
    npts, widths = widths_of(surface)
    return sorted({a.shape for a in reachable_arrays(surface)
                   if a.ndim == 2 and a.dtype.kind == 'f'
                   and ((a.shape[0] == npts and a.shape[1] in widths)
                        or (a.shape[1] == npts and a.shape[0] in widths))})


def ledger_faults(surface, rank, size, block=TILE):
    """What is wrong with the rows reachable from `surface`: each must be
    the row fit's, holding this rank's rows and carrying the fit's ledger
    with every grid-indexed array at this rank's tiles. [] when all is so."""
    faults = []
    npts, _ = widths_of(surface)
    r0, r1 = contiguous_block(npts, rank, size)
    own = [(t * block, min((t + 1) * block, npts))
           for t in partition(-(-npts // block), rank, size)]
    rows = sum(b - a for a, b in own)
    found = reachable_rows(surface)
    if not found:
        faults.append('no rows reachable')
    for part in found:
        held = getattr(part, 'fit_held', None)
        if part.rows != (r0, r1):
            faults.append(f'rows {part.rows}, block {(r0, r1)}')
        if not held:
            faults.append('rows without the fit ledger: cut from a whole fit')
            continue
        naux, nao = part.naux, part.nao
        want = {'FD_rows': rows * naux * 8, 'D_tiles': rows * naux * 8,
                'D_rows': (r1 - r0) * naux * 8, 'X_rows': rows * nao * 8,
                'aux_rows': rows * naux * 8}
        for name, nbytes in want.items():
            if held.get(name) != nbytes:
                faults.append(f'{name} {held.get(name)} != {nbytes}')
        if not held.get('S_rows', 0) < npts * npts * 8:
            faults.append(f"S_rows {held.get('S_rows')} is the whole Gram")
    return faults


def adjoint_widths(chain):
    """(npts, widths, dims): the grid, every width a whole factor, fit or
    test-set array carries beside it, and (nao, naux) of the dense
    (mu nu|P) where the fit's shell blocks cut the AO index -- None where
    one block holds every AO, whose kept pairs' integrals are then the
    dense tensor by the fit's own blocking, at any rank count."""
    mol = chain.mol0
    nao, naux = mol.nao, chain.naux
    ngram = len(product_pairs(mol)[0])
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())
    budget = inspect.signature(
        separable_ri.fit_rows).parameters['block_memory_gb'].default
    cut = len(separable_ri.ao_blocks(mol, chain.M, n2, naux, budget)) > 1
    return (chain.M, {chain.mf0.mo_coeff.shape[1], nao, naux, chain.M, n2,
                      len(chain.layout[0]), ngram, ngram + naux},
            (nao, naux) if cut else None)


def whole_shapes(values, npts, widths, dims, inputs):
    """Shapes of the float arrays among `values` and one level of their
    containers that hold the whole grid by a width, or (dims not None) the
    dense (mu nu|P) over two AO indices, sharing no memory with the whole
    `inputs`."""
    items = []
    for v in values:
        if isinstance(v, dict):
            items.extend(v.values())
        elif isinstance(v, (list, tuple)):
            items.extend(v)
        else:
            items.append(v)
    found = set()
    for a in items:
        if (not isinstance(a, np.ndarray) or a.dtype.kind != 'f'
                or a.ndim < 2
                or any(np.may_share_memory(a, e) for e in inputs)):
            continue
        shape = a.shape
        at = [i for i, n in enumerate(shape) if n == npts]
        grid = any(shape[j] in widths for i in at
                   for j in range(len(shape)) if j != i)
        dense = (dims is not None and a.ndim >= 3
                 and shape[-3:] == (dims[0], dims[0], dims[1]))
        if grid or dense:
            found.add(shape)
    return found


def traced(call, found, npts, widths, dims, inputs):
    """call() with every line of every frame under src/ scanned for
    `whole_shapes` into `found`: the arrays a force forms, named, however
    briefly, not only what survives it. Below `eps_chain_gradient` the
    dense check is off: the orbital response's skeleton derivative is the
    mean field's own density fitting, blocked by `THREE_CENTER_BLOCK_BYTES`,
    which holds all of a small molecule's (mu nu|P) in one block."""
    def tracer(frame, event, arg):
        if not frame.f_code.co_filename.startswith(SRC):
            return None
        up = frame
        while up is not None and up.f_code.co_name != 'eps_chain_gradient':
            up = up.f_back
        here = dims if up is None else None

        def local(frame, event, arg):
            if event in ('line', 'return'):
                values = list(frame.f_locals.values())
                if event == 'return':
                    values.append(arg)
                found.update(whole_shapes(values, npts, widths, here, inputs))
            return local
        return local

    sys.settrace(tracer)
    try:
        return call()
    finally:
        sys.settrace(None)


def adjoint_ledger_faults(held, chain, rank, size, block=TILE):
    """What is wrong with a force's `adjoint_held`: every grid-indexed array
    of the adjoint at this rank's tiles, the metric root and its adjoint on
    rank 0 alone, the streamed pieces one tile or slab. [] when all is so."""
    if not held:
        return ['a force without the adjoint ledger: the whole adjoint']
    mol = chain.mol0
    npts, nao, naux = chain.M, mol.nao, chain.naux
    n2 = int((separable_ri._ao_l_labels(mol) <= 2).sum())
    own = [(t * block, min((t + 1) * block, npts))
           for t in partition(-(-npts // block), rank, size)]
    rows = sum(b - a for a, b in own)
    want = {'X_rows': rows * nao, 'B_rows': rows * n2, 'aux_rows': rows * naux,
            'MT_bar_rows': rows * naux, 'Q_bar_rows': rows * naux,
            'Q_rows': rows * naux, 'U_rows': rows * naux,
            'MT_rows': rows * naux, 'P_bar_rows': rows * naux,
            'X_bar_rows': rows * nao, 'B_bar_rows': rows * n2}
    faults = [f'{name} {held.get(name)} != {8 * n}'
              for name, n in want.items() if held.get(name) != 8 * n]
    if not held.get('S_rows', 0) < npts * npts * 8:
        faults.append(f"S_rows {held.get('S_rows')} is the whole Gram")
    if held['metric_root'] != (2 * naux * naux * 8 if rank == 0 else 0):
        faults.append(f"metric_root {held['metric_root']} on rank {rank}")
    if (held['root_adjoint'] > 0) != (rank == 0):
        faults.append(f"root_adjoint {held['root_adjoint']} on rank {rank}")
    for name, cap in (('column_tile', block * 2 * naux),
                      ('root_gather', block * 2 * naux),
                      ('metric_slab', block * naux),
                      ('two_centre_slab', 4 * block * naux),
                      ('panel', (npts - block) * block)):
        if held.get(name, 0) > 8 * cap:
            faults.append(f'{name} {held.get(name)} above {8 * cap}')
    return faults


def watch(surface, log, rank, size):
    """Scan `surface` at the start of every evaluation (what the previous
    step left) and right after every factor build inside one, logging
    (whole arrays, ledger faults); and every force's nuclear assembly as it
    runs, logging (whole arrays it formed, adjoint ledger faults)."""
    def scan():
        log.append((whole_fit_arrays(surface),
                    ledger_faults(surface, rank, size)))

    for half in (surface.ground, surface.excited):
        assemble, branches = half.nuclear_gradient, half.row_fit_branches
        ledgers = []

        def own_ledger(*args, branches=branches, ledgers=ledgers, **kwargs):
            out = branches(*args, **kwargs)
            ledgers.append(out[2])      # this rank's; the diagnostics carry 0's
            return out

        def scanned_force(*args, assemble=assemble, half=half,
                          ledgers=ledgers, **kwargs):
            bound = inspect.signature(type(half).nuclear_gradient).bind(
                half, *args, **kwargs)
            inputs = [bound.arguments['x_bar'], bound.arguments['d_bar']]
            found = set()
            ledgers.clear()
            out = traced(lambda: assemble(*args, **kwargs), found,
                         *adjoint_widths(half), inputs)
            log.append((sorted(found), adjoint_ledger_faults(
                ledgers[0] if ledgers else None, half, rank, size)))
            half.forces_scanned = getattr(half, 'forces_scanned', 0) + 1
            return out

        half.row_fit_branches = own_ledger
        half.nuclear_gradient = scanned_force

    evaluate_now = surface.total_gradient

    def scanned(mol=None, mf=None):
        scan()
        return evaluate_now(mol, mf)

    surface.total_gradient = scanned
    for half in (surface.ground, surface.excited):
        build = half.factors_at

        def built(mol, mf, build=build):
            out = build(mol, mf)
            scan()
            return out

        half.factors_at = built


def observables(surface, log=None, rank=0, size=1):
    """Every gated quantity of one composed surface, the walk last; with a
    `log`, every evaluation and factor build scanned into it."""
    if log is not None:
        watch(surface, log, rank, size)
    ex = surface.excited
    roots = ex._forward(ex.mol0, ex.mf0)[0]
    force, energy, diags = evaluate(surface, own_water(H2O_DISPLACED))
    drpa = surface.ground.total_gradient(own_water(H2O_DISPLACED))[0]
    mol_opt, info = optimize(surface, max_cycle=1, verbose=False)
    if log is not None:
        log.append((whole_fit_arrays(surface),
                    ledger_faults(surface, rank, size)))
    return {'force': force, 'energy': energy, 'root': diags['omega'],
            'bse_roots': roots, 'drpa_force': drpa,
            'step': mol_opt.atom_coords() - surface.mol0.atom_coords(),
            'walk_energy': info['energy'],
            'gathers': dict(diags['factor_gathers']),
            'forces_scanned': [getattr(half, 'forces_scanned', 0) for half
                               in (surface.ground, surface.excited)]}


def whole_assembly(gram=True):
    """`FactorChain.row_fit_branches` formed WHOLE, the nuclear assembly
    before the row fit had an adjoint of its own: the collocation adjoint
    and `dfactor_adjoint_gauges` on every rank, over every product pair
    (`gram`, the row fit's estimator) or over the frozen test set alone
    (`_fit`'s), with no ledger."""
    def branches(self, mol, mf, auxmol, crd, x_bar, d_bar, **extra):
        g_coll = collocation_adjoint(mol, crd, x_bar @ mf.mo_coeff.T,
                                     self.pts_local, self.owner,
                                     frames=self.frames,
                                     with_frames=self.with_frames)
        g_fit = dfactor_adjoint_gauges(
            mol, auxmol, crd, [(d_bar, None)], self.layout, self.pts_local,
            self.owner, frames=self.frames, with_frames=self.with_frames,
            gram_layout=product_pairs(mol) if gram else None)
        return g_coll, g_fit, None
    return branches


def gathered_rotation(x_mo, x_bar, block=None):
    """Y = X_mo^T X_bar on X_mo gathered whole, the product before the row
    fit streamed it."""
    return whole_factor(x_mo, 'X_mo').T @ x_bar


def patch_whole_assembly(patch, gram=True):
    """The row-fit chain's nuclear assembly replaced by the whole one."""
    patch.setattr(FactorChain, 'row_fit_branches', whole_assembly(gram))
    patch.setattr(factor_chain, 'orbital_rotation_rows', gathered_rotation)


def anchored(ref, bar, got, keys):
    """(lines, {key: ratio}): each key's distance from `ref` in units of the
    bar's."""
    lines, ratios = [], {}
    for key in keys:
        anchor = relative(bar[key], ref[key])
        dist = relative(got[key], ref[key])
        ratios[key] = (dist / anchor if anchor
                       else (0.0 if dist == 0 else np.inf))
        lines.append(f'{key}: bar {anchor:.2e}, row fit {dist:.2e} '
                     f'({ratios[key]:.2f} x)')
    return lines, ratios


GATED = ('force', 'energy', 'root', 'bse_roots', 'drpa_force', 'step',
         'walk_energy')


# ------------------------------------------------------------------- (a)
@pytest.mark.parametrize('size', [1] + SIZES)
def test_rows_through_the_chain_are_bitwise_across_rank_counts(size):
    """Every rank's rows, at the reference and at a displaced geometry, are
    the same rows of a one-rank run of the chain, bit for bit."""

    def factors(comm):
        mol = own_water()
        chain = RPAGroundStateChain(mol, chain_scf, mf=chain_scf(mol),
                                    **row_kw())
        here, mf = chain.mean_field(own_water(H2O_DISPLACED))
        return (rows_of(chain.factors_at(chain.mol0, chain.mf0)),
                rows_of(chain.factors_at(here, mf)))

    with distributed(None):
        whole = factors(None)
    out = run_simulated(factors, size)
    for r, per_rank in enumerate(out):
        for (rows, *arrays), (_, *full) in zip(per_rank, whole):
            r0, r1 = rows
            assert rows == contiguous_block(len(full[0]), r, size)
            assert bitwise(arrays, [a[r0:r1] for a in full]), (
                f'rank {r} of {size}: rows differ from the one-rank chain')


# ------------------------------------------------------ (b), (c) and (d)
@pytest.mark.parametrize('size', SIZES)
def test_state_pair_on_the_row_fit(size, monkeypatch):
    """Force, energy, root, BSE roots, dRPA force and a one-cycle step
    within the anchored bar of the whole-fit sliced chain at the same rank
    count; no whole factor or fit array at any evaluation; the gathers of
    the whole-fit chain."""
    with distributed(None):
        reference = FrozenFactorization(own_water())
    anchor = reassociated_fit(own_water(), reference.layout)

    def run(tag):
        def rank(comm):
            log = [] if tag == 'rows' else None
            kw = row_kw() if tag == 'rows' else dict(sliced=True)
            out = observables(state_pair(bse_conv_tol=GATE_BSE_CONV_TOL,
                                         **kw), log, comm.Get_rank(), size)
            out['scans'] = log
            return out
        return run_simulated(rank, size)

    whole, rows = run('whole'), run('rows')
    with monkeypatch.context() as patch:
        patch.setattr(factor_chain, 'fit_M_stable', anchor)
        bar = run('whole')
    for r in range(size):
        lines, ratios = anchored(whole[r], bar[r], rows[r], GATED)
        if r == 0:
            print(f'\n{size} ranks: ' + '; '.join(lines))
        assert max(ratios.values()) <= FIT_REASSOCIATION_K, (r, lines)
        for key in GATED:
            assert bitwise([rows[r][key]], [rows[0][key]]), (
                f'rank {r} of {size}: {key} != rank 0')
        assert rows[r]['gathers'] == ROW_FIT_GATHERS
        assert whole[r]['gathers'] == COMPOSED_GATHERS
        scans = rows[r]['scans']
        assert min(rows[r]['forces_scanned']) >= 2, rows[r]['forces_scanned']
        assert len(scans) >= 10 and all(s == ([], []) for s in scans), (
            f'rank {r} of {size}: {[s for s in scans if s != ([], [])]}')


def test_the_scan_sees_a_whole_fit():
    """The scan can fail: on the replicated fit, unsliced, it finds the
    cached fit, and the ledger check finds rows cut from a whole fit."""

    def rank(comm):
        out = []
        for kw in ({}, dict(sliced=True)):
            surface = state_pair(**kw)
            evaluate(surface, own_water(H2O_DISPLACED))
            out.append((whole_fit_arrays(surface),
                        ledger_faults(surface, comm.Get_rank(), 2)))
        return out

    for (unsliced, sliced) in run_simulated(rank, 2):
        assert unsliced[0], 'the whole layout caches its fit'
        assert sliced[0] == [] and any('ledger' in f for f in sliced[1])


# ------------------------------------------------------------- ethylene
def product_fit_factors(fit):
    """`FactorChain.row_fit_factors` replaced by the row fit's ESTIMATOR
    formed whole -- every product pair in the Gram matrix, F on the frozen
    test set, solved by `fit` -- and the rows cut from it: the reference
    realization of the ethylene gate, and with a reassociated `fit` its bar."""
    def factors(chain, mol, mf, auxmol, crd):
        gram = product_pairs(mol)
        mu, nu, wc = chain.layout
        V = auxmol.intor('int2c2e', aosym='s1')
        e3 = factor_chain.test_set_three_center(mol, auxmol, mu, nu)
        F = np.zeros((auxmol.nao_nr(), len(gram[0]) + auxmol.nao_nr()))
        F[:, pair_positions(chain.layout, gram, mol.nao_nr())] = (
            np.linalg.solve(V, e3.T) * wc[None, :])
        F[:, len(gram[0]):] = np.eye(auxmol.nao_nr())
        M = lockstep(fit(separable_ri.test_set_D(mol, auxmol, crd, gram), F))
        x_ao = mol.eval_gto('GTOval_sph', crd)
        whole = (x_ao @ mf.mo_coeff, M.T @ aux_metric_sqrt(auxmol, None, V=V),
                 x_ao, crd)
        comm = chain.factorization.slice_comm()
        return (whole[:3] if comm is None
                else SlicedFactors.from_whole(whole, comm))
    return factors


def test_ethylene_row_fit_is_its_own_estimator(monkeypatch):
    """Where the pair screen drops pairs, the row-fit chain sits within the
    anchored bar of its own estimator formed whole, and `_fit`'s estimator,
    which it does not realize, outside that bar in every quantity."""
    mol = molecule(ETHYLENE)

    def run(**kw):
        def rank(comm):
            surface = state_pair(ETHYLENE, bse_conv_tol=GATE_BSE_CONV_TOL,
                                 **kw)
            roots = surface.excited._forward(surface.excited.mol0,
                                             surface.excited.mf0)[0]
            force, energy, diags = evaluate(surface,
                                            molecule(ETHYLENE_DISPLACED))
            drpa = surface.ground.total_gradient(
                molecule(ETHYLENE_DISPLACED))[0]
            return {'force': force, 'energy': energy, 'root': diags['omega'],
                    'bse_roots': roots, 'drpa_force': drpa}
        return run_simulated(rank, 2)[0]

    keys = ('force', 'energy', 'root', 'bse_roots', 'drpa_force')
    rows, replicated = run(**row_kw()), run(sliced=True)
    with monkeypatch.context() as patch:
        patch.setattr(FactorChain, 'row_fit_factors',
                      product_fit_factors(fit_M_stable))
        whole = run(**row_kw())
        patch.setattr(FactorChain, 'row_fit_factors', product_fit_factors(
            reassociated_fit(mol, product_pairs(mol))))
        bar = run(**row_kw())
    own, near = anchored(whole, bar, rows, keys)
    other, apart = anchored(whole, bar, replicated, keys)
    print('\nethylene, 2 ranks, against its own estimator: ' + '; '.join(own)
          + '\n  the replicated fit against the same: ' + '; '.join(other)
          + f"\n  max |d force| row fit - replicated "
            f"{np.abs(rows['force'] - replicated['force']).max():.2e} "
            f"Ha/Bohr, |d root| "
            f"{abs(rows['root'] - replicated['root']) * HARTREE_TO_EV:.2e} eV")
    assert max(near.values()) <= FIT_REASSOCIATION_K, own
    assert min(apart.values()) > FIT_REASSOCIATION_K, other


# ------------------------------------------------------ plumbing, refusals
def test_refreeze_carries_settings():
    """The fit realization and its tile edge survive a refreeze, of the
    composed surface and of a chain on its own."""

    def rank(comm):
        surface = state_pair(**row_kw())
        refrozen = surface.refreeze(own_water(H2O_DISPLACED))
        fac = refrozen.ground.factorization
        assert fac is refrozen.excited.factorization
        chains = [RPAGroundStateChain(own_water(), chain_scf, **row_kw()),
                  ExcitedStateChain(own_water(), chain_scf, **row_kw())]
        alone = [c.refreeze(own_water(H2O_DISPLACED)).factorization
                 for c in chains]
        return [(f.sliced, f.fit, f.fit_block) for f in [fac] + alone]

    for settings in run_simulated(rank, 2):
        assert settings == [(True, 'rows', TILE)] * 3


def test_the_dispatcher_carries_the_fit():
    """`potential_energy_surface` reaches the factorization with `fit` and
    `fit_block`, the record of a force carries the fit's ledger, and a dense
    row refuses the keyword."""
    ground = GroundState('rpa', 'hf')

    def rank(comm):
        mol = own_water()
        surface = potential_energy_surface(mol, chain_scf, ground_state=ground,
                                           excitation=Excitation('singlet'),
                                           **row_kw())
        fac = surface.ground.factorization
        assert fac is surface.excited.factorization
        assert (fac.fit, fac.fit_block) == ('rows', TILE)
        return evaluate(surface, own_water(H2O_DISPLACED))[2]

    for diags in run_simulated(rank, 2):
        assert set(diags['factor_gathers']) == {'X_mo', 'D', 'X_o', 'X_v'}
        assert diags['fit_held']['S_rows'] > 0
    with pytest.raises(TypeError, match='fit'):
        potential_energy_surface(own_water(), chain_scf, ground_state=ground,
                                 excitation=Excitation('singlet'),
                                 chi0='dense-qb', factorization='four-index',
                                 fit='rows')


def test_what_the_row_fit_cannot_serve_is_refused():
    mol = own_water()
    solvated = SolventScreening(mol, eps=1.78)
    built = []

    def counted(m):
        built.append(m)
        return chain_scf(m)

    with pytest.raises(ValueError, match='sliced=True'):
        FrozenFactorization(mol, fit='rows')
    with pytest.raises(ValueError, match='rows'):
        FrozenFactorization(mol, sliced=True, fit='cyclic')
    with pytest.raises(ValueError, match='tile'):
        FrozenFactorization(mol, sliced=True, fit_block=TILE)
    with pytest.raises(ValueError, match='sliced=False'):
        FactorChain(mol, counted, **row_kw())
    with pytest.raises(ValueError, match='fit='):
        ExcitedStateChain(mol, counted, fit='rows',
                          factorization=FrozenFactorization(mol, sliced=True))
    with pytest.raises(ValueError, match='fit_block'):
        ExcitedStateChain(mol, counted, fit_block=32,
                          factorization=FrozenFactorization(mol, **row_kw()))
    with pytest.raises(ValueError, match='row fit'):
        RPAGroundStateChain(mol, counted, environment=solvated, **row_kw())
    with pytest.raises(ValueError, match='reaction field'):
        ExcitedStateChain(mol, counted, environment=solvated, **row_kw())
    assert built == [], 'a refusal came only after an SCF'
    with pytest.raises(ValueError, match='never forms the whole fit'):
        chain = RPAGroundStateChain(mol, chain_scf, **row_kw())
        chain.factors(mol, chain.auxmol(mol), chain.coords(mol))
    # one rank is the row fit too, and the whole grid
    x_mo, d, *_ = chain.factors_at(chain.mol0, chain.mf0)
    assert isinstance(x_mo, np.ndarray) and len(x_mo) == chain.M


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
