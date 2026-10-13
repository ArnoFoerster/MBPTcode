"""The row fit's own adjoint in the nuclear assembly: `FrozenFactorization(
sliced=True, fit='rows')` differentiates `separable_ri.fit_rows`' estimator
with `separable_ri.fit_rows_adjoint`, in the fit's tiles, so no rank forms
the Gram matrix, the test set's collocation, (mu nu|P), M, F D^T or an
adjoint of their size whole at any force, and the X_mo collocation adjoint
and the orbital-rotation product X_mo^T X_bar run in the same tiles.

Gated over 1, 2, 3 and 8 simulated ranks (`run_simulated`) on water/cc-pVDZ
Hartree-Fock at 148 points per atom in tiles of `TILE` points (7 tiles):

  (a) On fixed adjoints, water and ethylene: `fit_rows_adjoint` (its centre
      terms and point adjoints, fit and collocation) and
      `orbital_rotation_rows` give every rank the one-rank run's bits.
      Against the whole adjoint (`dfactor_adjoint_gauges` over every product
      pair, `collocation_adjoint`, X_mo^T X_bar in one product), at stated
      tolerances: the fit branch, two realizations of the fit on random
      seeds, within FIT_REALIZATION_REL_TOL relative; the collocation and the
      product, re-associated sums over the grid, within
      REASSOCIATED_SUM_REL_TOL.
  (b) The force: the composed state-pair force at a displaced geometry and
      the dRPA force on the row fit, the tiled assembly against the whole one
      at the same rank count. Bitwise between the two: the energy, the root
      and every adjoint the kernels hand the assembly (eps_bar, X_bar,
      D_bar). The orbital, collocation and fit branches and the forces
      within FIT_REALIZATION_FORCE_TOL, the whole assembly refitting the
      estimator with `fit_M_stable`. Every rank holds rank 0's force.
  (c) The memory scan at every force (`test_chain_row_fit.watch`): every line
      of every frame under src/ inside the nuclear assembly is traced, and no
      array it names holds the grid by a factor's, the fit's or the test
      set's width, or the dense (mu nu|P), beyond the X_bar and D_bar handed
      in; the adjoint's ledger holds every grid-indexed array at this rank's
      tiles, the metric root and its adjoint on rank 0 alone. The scan fails
      the whole assembly.
  (d) The dRPA force of ethylene on the row fit against a five-point
      difference of its own energy, beside the whole assembly of the same
      estimator and the screened-Gram estimator (the Gram matrix over the
      screened pairs alone), which misses it. One carbon sits
      `CROSSING_SHIFT` along the bond, 3e-5 Bohr short of a geometry where
      the screen drops two pairs, so the stencil along that bond straddles
      the crossing: a fit screened again at each geometry steps the energy
      there (6.7e-6 Ha/Bohr in the difference), the frozen layout the
      adjoint differentiates does not.
  (e) The mean field's own force is rank 0's on every rank. pyscf blocks a
      density-fitted gradient's auxiliary index by the process's free memory,
      so separate processes reassociate it differently; here each simulated
      rank reports its own free memory to `pyscf.df.grad.rhf`
      (`RankMemoryLib`), and the composed `total_gradient`, whole-fit and
      row-fit, is the same bits on every rank at 3 and 8 ranks.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.df.grad import rhf as pyscf_df_grad

from src.Base.constants import (FIT_REALIZATION_REL_TOL,
                                FIT_REALIZATION_FORCE_TOL, ISDF_GRADIENT_FLOOR,
                                REASSOCIATED_SUM_REL_TOL)
from src.Base import separable_ri
from src.Base.separable_ri import fit_rows_adjoint
from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.mpi_grid import current_comm, distributed, run_simulated
from src.gradients.factor_chain import FrozenFactorization
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            dfactor_adjoint_gauges,
                                            orbital_rotation_rows,
                                            point_chain, product_pairs)
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.surface import evaluate
from tests.test_chain_row_fit import (ETHYLENE, FD_COMPONENTS, FD_STEP,
                                      GATE_BSE_CONV_TOL, TILE, bitwise,
                                      molecule, patch_whole_assembly,
                                      relative, row_kw,
                                      state_pair, watch)
from tests.test_chain_sliced_factors import (H2O, H2O_DISPLACED, chain_scf,
                                             own_water)

SIZES = [1, 2, 3, 8]
#: Bohr, carbon 0 along the C=C bond: the screen of ethylene/cc-pVDZ at 148
#: points per atom drops two pairs 3.1e-5 Bohr further on.
CROSSING_SHIFT = 0.15587


# ------------------------------------------------------------------- (a)
@pytest.mark.parametrize('atom', [H2O, ETHYLENE], ids=['water', 'ethylene'])
def test_the_adjoint_on_fixed_adjoints(atom):
    """fit_rows_adjoint and orbital_rotation_rows bitwise at 1/2/3/8 ranks
    on fixed adjoints, and the whole branches within the stated
    tolerances."""
    mol = molecule(atom)
    with distributed(None):
        fac = FrozenFactorization(mol)
    aux, crd = fac.auxmol(mol), fac.coords(mol)
    npts, naux, nao = len(crd), aux.nao_nr(), mol.nao
    rng = np.random.default_rng(3)
    d_bar = rng.normal(size=(npts, naux))
    x_bar = rng.normal(size=(npts, nao))
    C = rng.normal(size=(nao, nao))
    x_mo = mol.eval_gto('GTOval_sph', crd) @ C
    chain = dict(pts_local=fac.pts_local, atom_of_point=fac.owner,
                 with_frames=False)

    def rank(comm):
        adj = fit_rows_adjoint(mol, aux, crd, d_bar, fac.layout,
                               x_bar=x_bar, mo_coeff=C, block=TILE)
        rows = (SlicedFactors.from_whole((x_mo, d_bar, x_mo, crd), comm)
                if comm.Get_size() > 1 else x_mo)
        y = orbital_rotation_rows(rows, x_bar, TILE)
        return (adj.fit_centre + point_chain(mol, adj.fit_points, **chain),
                adj.coll_centre + point_chain(mol, adj.coll_points, **chain),
                y)

    runs = {size: run_simulated(rank, size) for size in SIZES}
    one = runs[1][0]
    for size, out in runs.items():
        for r, got in enumerate(out):
            assert bitwise(got, one), f'rank {r} of {size} != one rank'

    with distributed(None):
        def whole_fit():
            return dfactor_adjoint_gauges(
                mol, aux, crd, [(d_bar, None)], fac.layout, fac.pts_local,
                fac.owner, with_frames=False, gram_layout=product_pairs(mol))

        g_fit = whole_fit()
        g_coll = collocation_adjoint(mol, crd, x_bar @ C.T, **chain)
        y = x_mo.T @ x_bar
    lines = []
    for name, got, ref, tol in (
            ('fit', one[0], g_fit, FIT_REALIZATION_REL_TOL),
            ('collocation', one[1], g_coll, REASSOCIATED_SUM_REL_TOL),
            ('X_mo^T X_bar', one[2], y, REASSOCIATED_SUM_REL_TOL)):
        dist = relative(got, ref)
        lines.append(f'{name}: tiles {dist:.2e} relative, {dist / tol:.3f} '
                     f'of {tol:.0e}')
        assert dist <= tol, lines[-1]
    print('\n' + '; '.join(lines))


# ------------------------------------------------------------ (b) and (c)
def logged(a):
    """A copy of the adjoint `a` for the log, whole: grid tiles gathered with
    the force's scan paused, since the log's copy is not the force's."""
    if not isinstance(a, GridTileRows):
        return np.array(a)
    tracer = sys.gettrace()
    sys.settrace(None)
    try:
        return a.gather()
    finally:
        sys.settrace(tracer)


def record(surface, branches, inputs):
    """Wrap both halves of `surface` so that every nuclear assembly logs the
    adjoints the kernels hand it into `inputs` and its three branches into
    `branches`: per instance, since the ranks are threads of one process."""
    for half in (surface.ground, surface.excited):
        assemble, assembled = half.nuclear_gradient, half._assembled

        def logged_inputs(*args, assemble=assemble, half=half, **kwargs):
            got = inspect.signature(type(half).nuclear_gradient).bind(
                half, *args, **kwargs).arguments
            inputs.append(tuple(logged(got[k])
                                for k in ('eps_bar', 'x_bar', 'd_bar')))
            return assemble(*args, **kwargs)

        def logged_branches(g_orb, g_coll, g_fit, *args, assembled=assembled,
                            **kwargs):
            branches.append((np.array(g_orb), np.array(g_coll),
                             np.array(g_fit)))
            return assembled(g_orb, g_coll, g_fit, *args, **kwargs)

        half.nuclear_gradient = logged_inputs
        half._assembled = logged_branches


def per_shell_blocks(mol, nk, n2, naux, block_memory_gb):
    """Every shell its own block of the fit's pass, the AO index cut as a
    large molecule's is, so the scan can tell a block's kept pairs from the
    dense (mu nu|P)."""
    return [(s, s + 1) for s in range(mol.nbas)]


@pytest.mark.parametrize('size', SIZES)
def test_the_force_on_the_row_fit_adjoint(size, monkeypatch):
    """The composed and the dRPA force of the row fit, the tiled assembly
    against the whole one at the same rank count, every force scanned."""
    monkeypatch.setattr(separable_ri, 'ao_blocks', per_shell_blocks)

    def run(tag):
        def rank(comm):
            branches, inputs, log = [], [], []
            surface = state_pair(bse_conv_tol=GATE_BSE_CONV_TOL, **row_kw())
            ex = surface.excited
            ex._forward(ex.mol0, ex.mf0)       # the reference's rows
            record(surface, branches, inputs)
            if tag == 'rows' and comm.Get_size() > 1:
                watch(surface, log, comm.Get_rank(), comm.Get_size())
            force, energy, diags = evaluate(surface, own_water(H2O_DISPLACED))
            drpa = surface.ground.total_gradient(own_water(H2O_DISPLACED))[0]
            return {'force': force, 'energy': energy, 'root': diags['omega'],
                    'drpa_force': drpa, 'branches': branches,
                    'inputs': inputs, 'scans': log,
                    'forces_scanned': [getattr(h, 'forces_scanned', 0) for h
                                       in (surface.ground, surface.excited)]}
        return run_simulated(rank, size)

    rows = run('rows')
    with monkeypatch.context() as patch:
        patch_whole_assembly(patch)
        whole = run('whole')
    lines = []
    tol = FIT_REALIZATION_FORCE_TOL
    for r in range(size):
        new, old = rows[r], whole[r]
        # the same run up to the assembly
        assert new['energy'] == old['energy'] and new['root'] == old['root']
        assert len(new['inputs']) == len(old['inputs']) == 3
        for a, b in zip(new['inputs'], old['inputs']):
            assert bitwise(a, b), f'rank {r}: the kernels moved'
        for key in ('force', 'drpa_force'):
            dist = float(np.abs(np.asarray(new[key])
                                - np.asarray(old[key])).max())
            if r == 0:
                lines.append(f'{key}: {dist:.2e} = {dist / tol:.3f} of '
                             f'{tol:.0e} Ha/Bohr')
            assert dist <= tol, (r, key, dist)
            assert bitwise([new[key]], [rows[0][key]]), f'rank {r} != 0'
        # each branch of each assembly, the same estimator's two fits
        for i, (a, b) in enumerate(zip(new['branches'], old['branches'])):
            for j, name in enumerate(('orbital', 'collocation', 'fit')):
                dist = float(np.abs(a[j] - b[j]).max())
                if r == 0:
                    lines.append(f'assembly {i} {name} {dist:.2e}')
                assert dist <= tol, (r, i, name, dist)
        if size > 1:
            assert new['forces_scanned'] == [2, 1], new['forces_scanned']
            assert all(s == ([], []) for s in new['scans']), (
                r, [s for s in new['scans'] if s != ([], [])])
    print(f'\n{size} ranks: ' + '; '.join(lines))


def test_the_scan_fails_the_whole_assembly(monkeypatch):
    """The traced scan finds what the whole assembly forms (the Gram
    matrix, the test set's collocation, the dense (mu nu|P)), and the
    ledger check finds no ledger."""
    def rank(comm):
        log = []
        surface = state_pair(bse_conv_tol=GATE_BSE_CONV_TOL, **row_kw())
        watch(surface, log, comm.Get_rank(), comm.Get_size())
        surface.ground.total_gradient(own_water(H2O_DISPLACED))
        return log

    monkeypatch.setattr(separable_ri, 'ao_blocks', per_shell_blocks)
    with monkeypatch.context() as patch:
        patch_whole_assembly(patch)
        out = run_simulated(rank, 2)
    npts = FrozenFactorization(own_water()).M
    for r, log in enumerate(out):
        found, faults = log[-1]
        assert (npts, npts) in found, (r, found)
        assert any(len(s) == 3 for s in found), (r, found)
        assert any('ledger' in f for f in faults), (r, faults)


# ------------------------------------------------------------------- (d)
def test_ethylene_force_is_the_derivative_of_its_energy(monkeypatch):
    """The dRPA force of the row fit against a five-point difference of its
    own energy, beside the whole assembly of the same estimator and that of
    the screened-Gram estimator, which misses it."""
    mol = molecule(ETHYLENE)
    near = mol.atom_coords()
    near[0, 2] += CROSSING_SHIFT
    mol.set_geom_(near, unit='Bohr')
    mol.build(False, False)
    with distributed(None):
        chain = RPAGroundStateChain(mol, chain_scf, **row_kw())
        grad = chain.total_gradient()[0]
        fd, crossed = [], []
        for ia, x in FD_COMPONENTS:
            values = []
            for k in (-2, -1, 1, 2):
                m = mol.copy()
                shift = np.zeros((mol.natm, 3))
                shift[ia, x] = k * FD_STEP
                m.set_geom_(mol.atom_coords() + shift, unit='Bohr')
                m.build(False, False)
                # the screen here, which the row fit must not follow
                columns = separable_ri.test_set_layout(m, chain.coords(m))
                if not bitwise(columns, chain.layout):
                    crossed.append((ia, x, k))
                values.append(chain.energy(m)[0])
            fd.append((values[0] - 8 * values[1] + 8 * values[2] - values[3])
                      / (12 * FD_STEP))
        # the case is only a test if the stencil crosses a change of screen
        assert crossed == [(0, 2, 1), (0, 2, 2)], crossed
        others = {}
        for name, gram in (('whole', True), ('screened', False)):
            with monkeypatch.context() as patch:
                patch_whole_assembly(patch, gram=gram)
                others[name] = chain.total_gradient()[0]
    pick = lambda g: np.array([g[ia, x] for ia, x in FD_COMPONENTS])
    err = np.abs(pick(grad) - fd).max()
    whole = np.abs(pick(others['whole']) - fd).max()
    wrong = np.abs(pick(others['screened']) - fd).max()
    print(f'\nethylene dRPA on the row fit: |analytic - fd| {err:.2e}, the '
          f'whole assembly {whole:.2e}, the screened-Gram estimator {wrong:.2e} '
          'Ha/Bohr')
    assert err < ISDF_GRADIENT_FLOOR
    assert whole < ISDF_GRADIENT_FLOOR
    assert wrong > 100 * ISDF_GRADIENT_FLOOR


# ------------------------------------------------ the mean field's own force
class RankMemoryLib:
    """pyscf.lib as `pyscf.df.grad.rhf` reads it, save `current_memory`,
    which reports a resident size that depends on the rank: each rank then
    blocks the density-fitted gradient's auxiliary index differently, as
    separate processes whose resident sizes differ do."""

    def __init__(self, lib, free_mb):
        self._lib, self._free = lib, free_mb

    def __getattr__(self, name):
        return getattr(self._lib, name)

    def current_memory(self):
        comm = current_comm()
        rank = 0 if comm is None else comm.Get_rank()
        return (self._lib.param.MAX_MEMORY - self._free(rank), 0)


@pytest.mark.parametrize('sliced', [dict(sliced=True), row_kw()],
                         ids=['whole-fit', 'row-fit'])
def test_the_force_is_rank_0s_whatever_each_rank_blocks(sliced, monkeypatch):
    """The composed force at a displaced geometry is the same bits on every
    rank when each rank's density-fitted mean-field gradient blocks its
    auxiliary index by its own free memory."""
    monkeypatch.setattr(pyscf_df_grad, 'lib', RankMemoryLib(
        pyscf_df_grad.lib, lambda rank: 0.6 + 0.2 * rank))

    def rank(comm):
        surface = state_pair(bse_conv_tol=GATE_BSE_CONV_TOL, **sliced)
        here = own_water(H2O_DISPLACED)
        force = surface.total_gradient(here)[0]     # not `evaluate`'s lockstep
        mf = surface.ground.mean_field(here)[1]
        return force, np.asarray(mf.Gradients().kernel())

    for size in (3, 8):
        out = run_simulated(rank, size)
        own = [o[1] for o in out]
        assert len({g.tobytes() for g in own}) > 1, 'the emulation blocks alike'
        for r, (force, _) in enumerate(out):
            assert bitwise([force], [out[0][0]]), f'rank {r} of {size} != 0'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
