"""The row fit's own adjoint in the nuclear assembly: `FrozenFactorization(
sliced=True, fit='rows')` differentiates `separable_ri.fit_rows`' estimator
with `separable_ri.fit_rows_adjoint`, in the fit's tiles, so no rank forms
the Gram matrix, the test set's collocation, (mu nu|P), M, F D^T or an
adjoint of their size whole at any force, and the X_mo collocation adjoint
and the orbital-rotation product X_mo^T X_bar run in the same tiles.

Gated over 1, 2, 3 and 8 simulated ranks (`run_simulated`) on water/cc-pVDZ
Hartree-Fock at 148 points per atom in tiles of `TILE` points (7 tiles):

  (a) BITWISE ACROSS RANK COUNTS on fixed adjoints, water and ethylene:
      `fit_rows_adjoint` (its centre terms and point adjoints, fit and
      collocation) and `orbital_rotation_rows` give every rank the bits of
      the one-rank run. THE ANCHORED BAR against the whole adjoint
      (`dfactor_adjoint_gauges` over every product pair,
      `collocation_adjoint`, X_mo^T X_bar in one product): each branch sits
      within `FIT_REASSOCIATION_K` times what the whole branch moves when its
      own sums are reordered -- the fit's sums over the test set cut per
      shell and accumulated in reverse, the collocation's and the product's
      sums over the grid cut per tile and accumulated in reverse.
  (b) THE FORCE: the composed state-pair force at a displaced geometry and
      the dRPA force on the row fit, the new assembly against the whole one
      at the same rank count. Bitwise between the two: the energy, the root
      and every adjoint the kernels hand the assembly (eps_bar, X_bar,
      D_bar) -- the same run up to the assembly. Anchored: the orbital,
      collocation and fit branches and the force, within
      `FIT_REASSOCIATION_K` times what the whole assembly's force moves when
      its fit's sums are reordered. Every rank holds rank 0's force.
  (c) THE MEMORY SCAN at every force (`test_chain_row_fit.watch`): every
      line of every frame under src/ inside the nuclear assembly is traced,
      and no array it names holds the grid by a factor's, the fit's or the
      test set's width, or the dense (mu nu|P), beyond the X_bar and D_bar
      handed in; the adjoint's ledger holds every grid-indexed array at this
      rank's tiles, the metric root and its adjoint on rank 0 alone. The
      scan fails the whole assembly.
  (d) the dRPA force of ethylene on the row fit (72 of 2304 pairs screened)
      against a five-point difference of its own energy, beside the whole
      assembly of the same estimator and the one of `_fit`'s.
  (e) THE MEAN FIELD'S OWN FORCE is rank 0's on every rank. pyscf blocks a
      density-fitted gradient's auxiliary index by the process's free
      memory, so separate processes re-associate it differently; here each
      simulated rank reports its own free memory to `pyscf.df.grad.rhf`
      (`RankMemoryLib`) and the composed `total_gradient`, whole-fit and
      row-fit, must still be the same bits on every rank at 3 and 8 ranks.
      Without the lockstep in `FactorChain.mean_field_gradient` it failed
      at 3 ranks for both layouts, rank 1 != rank 0, as the 8-rank cluster
      run of tests/test_mpi_routes.py did.

SHOWN TO FAIL, then restored and byte-compared (`cmp`): the adjoint's Gram
stream silently falling back to the whole Gram matrix -- S = (X X^T) o
(B B^T) + P P^T formed over the whole grid on every rank and read by tile
for d_bar -- left (a), the force and branch gates of (b) and the ledger
green, and failed the traced scan of (c) at 2, 3 and 8 ranks on all three
forces, naming the whole Gram matrix (444, 444) and the whole collocations
it was formed from, (444, 24) and (444, 84).
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg
from pyscf.df.grad import rhf as pyscf_df_grad

from src.Base.constants import FIT_REASSOCIATION_K, ISDF_GRADIENT_FLOOR
from src.Base import separable_ri
from src.Base.separable_ri import DEFAULT_REGULARIZATION, fit_rows_adjoint
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import current_comm, distributed, run_simulated
from src.gradients import isdf_derivatives
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
                                      reassociated_fit, relative, row_kw,
                                      state_pair, watch)
from tests.test_chain_sliced_factors import (H2O, H2O_DISPLACED, chain_scf,
                                             own_water)

SIZES = [1, 2, 3, 8]


def reassociated_fit_adjoint(mol, gram):
    """`isdf_derivatives.fit_adjoint` with its sums over the test set -- the
    row norms, the Gram matrix, F Dt^T and the balancing's row sums -- cut
    per mu shell and accumulated in reverse: the whole adjoint's anchor."""
    mu = np.asarray(gram[0])
    ao_loc = mol.ao_loc_nr()
    shells = [np.flatnonzero((mu >= ao_loc[s]) & (mu < ao_loc[s + 1]))
              for s in reversed(range(mol.nbas))]
    shells = [c for c in shells if len(c)]

    def adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
        blocks = shells + [np.arange(len(mu), D.shape[1])]

        def rowsum(a, b):
            out = np.zeros(a.shape[0])
            for c in blocks:
                out += np.einsum('kr,kr->k', a[:, c], b[:, c])
            return out

        s = np.sqrt(rowsum(D, D))
        s = np.where(s == 0.0, 1.0, s)
        d = 1.0 / s
        Dt = D * d[:, None]
        G = np.zeros((D.shape[0], D.shape[0]))
        A = np.zeros((F.shape[0], D.shape[0]))
        for c in blocks:
            G += Dt[:, c] @ Dt[:, c].T
            A += F[:, c] @ Dt[:, c].T
        G[np.diag_indices_from(G)] += regularization
        cho = scipy.linalg.cho_factor(G, lower=True)
        B = scipy.linalg.cho_solve(cho, A.T).T
        B_bar = M_bar * d[None, :]
        d_bar = np.einsum('bk,bk->k', M_bar, B)
        A_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
        Y = scipy.linalg.cho_solve(cho, A.T @ B_bar)
        G_bar = -scipy.linalg.cho_solve(cho, Y.T).T
        F_bar = A_bar @ Dt
        Dt_bar = A_bar.T @ F + (G_bar + G_bar.T) @ Dt
        D_bar = d[:, None] * Dt_bar
        d_bar = d_bar + rowsum(Dt_bar, D)
        s_bar = -d_bar * d ** 2
        D_bar += (s_bar / s)[:, None] * D
        return D_bar, F_bar

    return adjoint


def patch_reassociated_whole(patch, mol):
    """The whole assembly's fit with its sums over the test set reordered."""
    gram = product_pairs(mol)
    patch_whole_assembly(patch)
    patch.setattr(isdf_derivatives, 'fit_M_stable', reassociated_fit(mol, gram))
    patch.setattr(isdf_derivatives, 'fit_adjoint',
                  reassociated_fit_adjoint(mol, gram))


def tile_reversed(npts, block=TILE):
    """The grid's tiles, last first."""
    return [slice(t0, min(t0 + block, npts))
            for t0 in reversed(range(0, npts, block))]


def ratio(got, ref, bar):
    """|got - ref| in units of |bar - ref|; 0 when both are bitwise."""
    dist, anchor = relative(got, ref), relative(bar, ref)
    return dist / anchor if anchor else (0.0 if dist == 0 else np.inf), \
        dist, anchor


# ------------------------------------------------------------------- (a)
@pytest.mark.parametrize('atom', [H2O, ETHYLENE], ids=['water', 'ethylene'])
def test_the_adjoint_on_fixed_adjoints(atom):
    """fit_rows_adjoint and orbital_rotation_rows bitwise at 1/2/3/8 ranks
    on fixed adjoints, and within the anchored bar of the whole branches."""
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
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isdf_derivatives, 'fit_M_stable',
                          reassociated_fit(mol, product_pairs(mol)))
            patch.setattr(isdf_derivatives, 'fit_adjoint',
                          reassociated_fit_adjoint(mol, product_pairs(mol)))
            g_fit_bar = whole_fit()
        xa = x_bar @ C.T
        g_coll = collocation_adjoint(mol, crd, xa, **chain)
        g_coll_bar = np.zeros((mol.natm, 3))
        P = np.zeros((npts, 3))
        for rows in tile_reversed(npts):
            c, P[rows] = isdf_derivatives.basis_centre_forces(mol, crd[rows],
                                                              xa[rows])
            g_coll_bar += c
        g_coll_bar += point_chain(mol, P, **chain)
        y = x_mo.T @ x_bar
        y_bar = np.zeros_like(y)
        for rows in tile_reversed(npts):
            y_bar += x_mo[rows].T @ x_bar[rows]
    lines = []
    for name, got, ref, bar in (('fit', one[0], g_fit, g_fit_bar),
                                ('collocation', one[1], g_coll, g_coll_bar),
                                ('X_mo^T X_bar', one[2], y, y_bar)):
        k, dist, anchor = ratio(got, ref, bar)
        lines.append(f'{name}: bar {anchor:.2e}, tiles {dist:.2e} ({k:.2f} x)')
        assert k <= FIT_REASSOCIATION_K, lines[-1]
    print('\n' + '; '.join(lines))


# ------------------------------------------------------------ (b) and (c)
def record(surface, branches, inputs):
    """Wrap both halves of `surface` so that every nuclear assembly logs the
    adjoints the kernels hand it into `inputs` and its three branches into
    `branches`: per instance, since the ranks are threads of one process."""
    for half in (surface.ground, surface.excited):
        assemble, assembled = half.nuclear_gradient, half._assembled

        def logged_inputs(*args, assemble=assemble, half=half, **kwargs):
            got = inspect.signature(type(half).nuclear_gradient).bind(
                half, *args, **kwargs).arguments
            inputs.append(tuple(np.array(got[k])
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
    """The composed and the dRPA force of the row fit, the new assembly
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
    with monkeypatch.context() as patch:
        patch_reassociated_whole(patch, own_water())
        bar = run('bar')
    lines = []
    for r in range(size):
        new, old, ref = rows[r], whole[r], bar[r]
        # the same run up to the assembly
        assert new['energy'] == old['energy'] and new['root'] == old['root']
        assert len(new['inputs']) == len(old['inputs']) == 3
        for a, b in zip(new['inputs'], old['inputs']):
            assert bitwise(a, b), f'rank {r}: the kernels moved'
        for key in ('force', 'drpa_force'):
            k, dist, anchor = ratio(new[key], old[key], ref[key])
            if r == 0:
                lines.append(f'{key}: bar {anchor:.2e}, new {dist:.2e} '
                             f'({k:.2f} x)')
            assert k <= FIT_REASSOCIATION_K, (r, key, k, dist, anchor)
            assert bitwise([new[key]], [rows[0][key]]), f'rank {r} != 0'
        # each branch against what the reordered fit moves the fit branch
        for i, (a, b, c) in enumerate(zip(new['branches'], old['branches'],
                                          ref['branches'])):
            anchor = np.linalg.norm(c[2] - b[2])
            for j, name in enumerate(('orbital', 'collocation', 'fit')):
                dist = np.linalg.norm(a[j] - b[j])
                if r == 0:
                    lines.append(f'assembly {i} {name} {dist:.2e} '
                                 f'({dist / anchor:.2f} x {anchor:.2e})')
                assert dist <= FIT_REASSOCIATION_K * anchor, (r, i, name)
        if size > 1:
            assert new['forces_scanned'] == [2, 1], new['forces_scanned']
            assert all(s == ([], []) for s in new['scans']), (
                r, [s for s in new['scans'] if s != ([], [])])
    print(f'\n{size} ranks: ' + '; '.join(lines))


def test_the_scan_fails_the_whole_assembly(monkeypatch):
    """The traced scan finds what the whole assembly forms -- the Gram
    matrix, the test set's collocation, the dense (mu nu|P) -- and the
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
    own energy, beside the whole assembly of the same estimator and the one
    of `_fit`'s, which misses it."""
    mol = molecule(ETHYLENE)
    with distributed(None):
        chain = RPAGroundStateChain(mol, chain_scf, **row_kw())
        grad = chain.total_gradient()[0]
        fd = []
        for ia, x in FD_COMPONENTS:
            values = []
            for k in (-2, -1, 1, 2):
                m = mol.copy()
                shift = np.zeros((mol.natm, 3))
                shift[ia, x] = k * FD_STEP
                m.set_geom_(mol.atom_coords() + shift, unit='Bohr')
                m.build(False, False)
                # the row fit screens here; the adjoint reads the frozen set
                columns = separable_ri.test_set_layout(m, chain.coords(m))
                assert bitwise(columns, chain.layout)
                values.append(chain.energy(m)[0])
            fd.append((values[0] - 8 * values[1] + 8 * values[2] - values[3])
                      / (12 * FD_STEP))
        others = {}
        for name, gram in (('whole', True), ("_fit's", False)):
            with monkeypatch.context() as patch:
                patch_whole_assembly(patch, gram=gram)
                others[name] = chain.total_gradient()[0]
    pick = lambda g: np.array([g[ia, x] for ia, x in FD_COMPONENTS])
    err = np.abs(pick(grad) - fd).max()
    whole = np.abs(pick(others['whole']) - fd).max()
    wrong = np.abs(pick(others["_fit's"]) - fd).max()
    print(f'\nethylene dRPA on the row fit: |analytic - fd| {err:.2e}, the '
          f"whole assembly {whole:.2e}, _fit's estimator {wrong:.2e} Ha/Bohr")
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
