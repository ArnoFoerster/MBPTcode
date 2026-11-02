"""The AO-product ISDF fit in grid-row tiles (`ao_product_fit.ao_product_fit_rows`)
and its nuclear adjoint in the same tiles
(`ao_product_fit_adjoint.ao_product_fit_rows_adjoints`).

Gated on C1 water/cc-pVDZ, 150 points pivoted from the G1 parent, tiles of
40 points and several shell runs:
  * the rows are the replicated fit's M^T to the rounding of its balanced
    Gram matrix (1e-8 relative on the recording BLAS, measured 4e-9; 5e-8 on
    any other, where the Gram matrix's conditioning turns other last bits of
    the projection, 6e-16 relative, into up to 1.1e-8 of the fit), and three
    simulated ranks give the one-rank tiles bit for bit;
  * the adjoint of every kind of seed -- on D in the bare gauge, on a gauge
    a continuum dresses, on X_mo, and the exchange's seeds on M^T, the AO
    collocation and the bare metric -- is the replicated pass's force
    (`ao_product_fit_adjoint`, `collocation_adjoint`) to 1e-8 of its scale,
    the rounding of the balanced Gram matrix's solve (measured 3e-10 on D,
    1.6e-9 on random M^T seeds, which meet G^-1 first), and three ranks
    give the one-rank force bit for bit;
  * the three-centre pass over two nodes of two ranks: every node evaluates
    every shell run once and no block leaves its node, and the projection's
    rows are the one-rank rows bit for bit; the pass reads each AO pair once
    (nu up to the end of mu's run) and gives the explicit sum over every AO
    product to rounding;
  * `separable_factors(fit='rows')` on the AO-product test set gives the
    replicated factors to the same rounding;
  * the excited-state chain on the row fit (sliced) is the chain on the
    replicated fit: energy within 1e-9 Ha (measured 3e-14) and force within
    ISDF_GRADIENT_FLOOR (measured 1.5e-12).
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.factorization import ao_product_fit as apf
from src.Base.geometry.body_frame import BodyFrame
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import partition
from src.Base.factorization.separable_ri import tile_collocation
from src.Base.utils.parallel.simulated_comm import (run_simulated,
                                                    simulated_world)
from src.SingleReference.GW.space_time import separable_factors
from src.environments.solvent_screening import SolventScreening
from src.gradients.ao_product_fit_adjoint import (ao_product_fit_adjoint,
                                                  ao_product_fit_adjoint_gauges,
                                                  ao_product_fit_rows_adjoints)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.isdf_derivatives import (GaugeAdjoint, PointChain,
                                            collocation_adjoint)
from src.gradients.isdf_fit_adjoint import AdjointSeeds, GaugeSeeds
from tests.test_ao_product_fit import H2O_C1, pivoted_water
from tests.test_body_fixed_frames import rhf, water as body_water
from tests.pins import RECORDING_BLAS, blas_identity, isdf_gradient_bar

#: Rounding of the fit through its balanced Gram matrix, relative, under the
#: recording BLAS.
FIT_ROUNDING = 1e-8
#: The same under any other BLAS: the projection's last bits differ, and the
#: Gram matrix's conditioning carries them into the fit.
CROSS_BLAS_FIT_ROUNDING = 5e-8
#: The row adjoint against the replicated one, relative to the force's scale.
ADJOINT_ROUNDING = 1e-8
TILE = 40
#: Small enough a working set that the three-centre pass runs in many runs.
RUNS = Numerics().derive('several shell runs', layout={'fit_gb': 1e-3})
#: More shell runs than one round of a node's ranks.
MANY_RUNS = Numerics().derive('many shell runs', layout={'fit_gb': 2e-4})


@pytest.fixture(scope='module')
def water():
    mol, auxmol, pts, owner, frames = pivoted_water()
    chain = PointChain(pts, owner, frames, False,
                       body=BodyFrame(mol.atom_coords()))
    return mol, auxmol, apf.placed(mol, pts, frames), chain


def fit_rounding():
    """The fit's relative rounding bar under this process's BLAS."""
    if RECORDING_BLAS == blas_identity():
        return FIT_ROUNDING
    return CROSS_BLAS_FIT_ROUNDING


def tiles_of(nk):
    return [(t0, min(t0 + TILE, nk)) for t0 in range(0, nk, TILE)]


def mine(comm, ntiles):
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    return rank, [int(t) for t in partition(ntiles, rank, size)]


def test_the_rows_are_the_replicated_fit_on_any_rank_count(water):
    mol, auxmol, coords, _ = water
    _, M, _ = apf.ao_product_fit(mol, auxmol, coords)

    def rows(comm):
        return apf.ao_product_fit_rows(mol, auxmol, coords, numerics=RUNS,
                                       block=TILE).mt
    serial = rows(None)
    tiles = tiles_of(len(coords))
    assert sorted(serial) == list(range(len(tiles)))
    MT = np.vstack([serial[t] for t in range(len(tiles))])
    assert np.abs(MT - M.T).max() < fit_rounding() * np.abs(M).max()
    held = {}
    for part in run_simulated(rows, 3):
        assert not set(part) & set(held)
        held.update(part)
    assert sorted(held) == sorted(serial)
    assert all(np.array_equal(held[t], serial[t]) for t in serial)


def test_every_node_makes_the_whole_three_centre_pass(water, monkeypatch):
    mol, auxmol, coords, _ = water
    tiles = tiles_of(len(coords))
    runs = apf.fit_shell_runs(mol, auxmol, MANY_RUNS)
    assert len(runs) > 4
    evaluated = []
    block = apf.lower_block

    def counted(mol, auxmol, run, *args, **kwargs):
        evaluated.append(run)
        return block(mol, auxmol, run, *args, **kwargs)
    monkeypatch.setattr(apf, 'lower_block', counted)

    def projection(comm, node=None):
        _, own = mine(comm, len(tiles))
        X_own = {t: tile_collocation(mol, coords[slice(*tiles[t])])
                 for t in own}
        return apf.projection_rows(mol, auxmol, X_own, MANY_RUNS, comm, node=node)
    serial = projection(None)
    assert len(evaluated) == len(runs)
    evaluated.clear()
    # ranks 0, 1 on one node and 2, 3 on the other
    nodes = [simulated_world(2), simulated_world(2)]
    held = {}
    for part in run_simulated(
            lambda comm: projection(comm, nodes[comm.Get_rank() // 2]
                                    [comm.Get_rank() % 2]), 4):
        held.update(part)
    assert sorted(evaluated) == sorted(runs + runs)
    assert sorted(held) == sorted(serial)
    assert all(np.array_equal(held[t], serial[t]) for t in serial)


def test_the_pass_reads_each_ao_pair_once(water, monkeypatch):
    """Over several shell runs the projection evaluates (mu nu|P) for nu up
    to the end of mu's run alone, nao (nao + 1) / 2 pairs plus the runs'
    upper triangles, and its rows are the explicit sum over every AO product
    to rounding (1e-13 of their scale)."""
    mol, auxmol, coords, _ = water
    runs = apf.fit_shell_runs(mol, auxmol, MANY_RUNS)
    assert len(runs) > 4
    rows = []
    block = apf.lower_block

    def counted(mol, auxmol, run):
        out = block(mol, auxmol, run)
        rows.append(out.shape[0])
        return out
    monkeypatch.setattr(apf, 'lower_block', counted)
    X = tile_collocation(mol, coords)
    got = apf.projection_rows(mol, auxmol, {0: X}, MANY_RUNS, None)[0]
    nao = mol.nao_nr()
    assert sum(rows) == sum((a1 - a0) * a1 for _, _, a0, a1 in runs)
    assert sum(rows) < 0.6 * nao * nao
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    want = np.einsum('mnP,gm,gn->gP', e3c, X, X)
    assert np.abs(got - want).max() < 1e-13 * np.abs(want).max()


def seeds_and_reference(water, kind):
    """(seeds(comm), the replicated force) of one kind of seed."""
    mol, auxmol, coords, chain = water
    rng = np.random.default_rng(3)
    nk, nao, naux = len(coords), mol.nao_nr(), auxmol.nao_nr()
    tiles = tiles_of(nk)
    kw = dict(frames=chain.frames, with_frames=False, body=chain.body)
    pts, owner = chain.pts_local, chain.owner
    if kind == 'bare':
        d_bar = rng.normal(size=(nk, naux))
        return (lambda comm: AdjointSeeds(d_bar=d_bar),
                ao_product_fit_adjoint_gauges(
                    mol, auxmol, coords, [GaugeAdjoint(d_bar, None)], pts,
                    owner, **kw))
    if kind == 'dressed':
        d_bar = rng.normal(size=(nk, naux))
        env = SolventScreening(mol, solvent='water')
        return (lambda comm: AdjointSeeds(gauges=[GaugeSeeds(d_bar, env)]),
                ao_product_fit_adjoint_gauges(
                    mol, auxmol, coords, [GaugeAdjoint(d_bar, env)], pts,
                    owner, **kw))
    if kind == 'orbitals':
        d_bar = np.zeros((nk, naux))
        C = rng.normal(size=(nao, 6))
        x_bar = rng.normal(size=(nk, 6))
        return (lambda comm: AdjointSeeds(d_bar=d_bar, x_bar=x_bar,
                                          mo_coeff=C),
                collocation_adjoint(mol, coords, x_bar @ C.T, pts, owner,
                                    **kw))
    # the exchange's seeds: M^T and the AO collocation by tile, V on rank 0
    mt_bar = rng.normal(size=(nk, naux))
    x_ao = rng.normal(size=(nk, nao))
    v_bar = rng.normal(size=(naux, naux))
    v_bar += v_bar.T

    def seeds(comm):
        rank, own = mine(comm, len(tiles))
        return AdjointSeeds(mt_bar={t: mt_bar[slice(*tiles[t])] for t in own},
                            x_bar_ao={t: x_ao[slice(*tiles[t])] for t in own},
                            metric_bar=v_bar if rank == 0 else None)

    def dense_seeds(X, M, V):
        return mt_bar.T.copy(), v_bar.copy(), None, 0.0
    return seeds, (ao_product_fit_adjoint(mol, auxmol, coords, chain,
                                          dense_seeds)
                   + collocation_adjoint(mol, coords, x_ao, pts, owner, **kw))


@pytest.mark.parametrize('kind', ['bare', 'dressed', 'orbitals', 'exchange'])
def test_the_row_adjoint_is_the_replicated_adjoint(water, kind):
    mol, auxmol, coords, chain = water
    seeds, want = seeds_and_reference(water, kind)

    def force(comm):
        out = ao_product_fit_rows_adjoints(mol, auxmol, coords, [seeds(comm)],
                                           numerics=RUNS, block=TILE)[0]
        g = out.fit_centre + chain(mol, out.fit_points)
        if out.coll_centre is not None:
            g = g + out.coll_centre + chain(mol, out.coll_points)
        return g
    g = force(None)
    assert np.abs(g - want).max() < ADJOINT_ROUNDING * np.abs(want).max(), \
        np.abs(g - want).max() / np.abs(want).max()
    for other in run_simulated(force, 3):
        assert np.array_equal(other, g)


def test_the_factors_on_the_row_fit():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri').run()
    numerics = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                              'pivot_points': 150},
                        layout={'fit_block': TILE})
    whole = separable_factors(mf, mol, numerics=numerics)
    rows = separable_factors(mf, mol, fit='rows', numerics=numerics)
    for a, b in zip(rows[:3], whole[:3]):
        assert np.abs(a - b).max() < fit_rounding() * np.abs(b).max()


def test_the_chain_on_the_row_fit_is_the_chain_on_the_whole_fit():
    warnings.simplefilter('ignore')
    mol = body_water()
    isdf = {'level': 'G1', 'test_set': 'ao_products', 'pivot_points': 200}
    whole = ExcitedStateChain(mol, rhf, spin='singlet',
                              numerics=Numerics(isdf=isdf))
    rows = ExcitedStateChain(mol, rhf, spin='singlet', numerics=Numerics(
        isdf=dict(isdf, fit='rows'),
        layout={'sliced': True, 'fit_block': 64}))
    assert rows.factorization.fit == 'rows'
    assert abs(rows.energy(mol)[0] - whole.energy(mol)[0]) < 1e-9
    g_rows, g_whole = rows.total_gradient(mol)[0], whole.total_gradient(mol)[0]
    assert np.abs(g_rows - g_whole).max() < isdf_gradient_bar()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
