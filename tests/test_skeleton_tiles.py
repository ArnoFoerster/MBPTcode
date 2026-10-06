"""The mean-field skeleton derivatives in fixed tiles over the ranks
(`src/Base/skeleton_tiles.py`), on water, ethylene and formaldehyde/cc-pVDZ
(C1-distorted, so no force component cancels by symmetry), PBE0 and
LRC-wPBEh, on density-fitted (cc-pVDZ-JKFIT) and ISDF-K (cc-pVDZ-RI)
references, at 1, 2, 3 and 8 simulated ranks:

  * the fitted Fock skeleton (`fock_partial_skeleton_df`) is the
    whole-tensor contraction to `ISDF_GRADIENT_FLOOR`: the tiles contract
    (mn|P) in another order and the exchange applies an explicit Cholesky
    inverse of the metric where the whole form solves against it;
  * over the ranks every rank holds the serial skeleton bitwise: every
    tile's (natm, 3) addend added in tile order (`ordered_sum`);
  * planted: a rank dropping one of its tiles, a rank counting one twice --
    both break the bits and move the skeleton past the floor;
  * the interpolated exchange skeleton on the row fit is the same bits at
    every rank count, one rank included, for the mean-field force, the
    folded Fock partial and the Sigma_x - v_xc channels; planted, a rank
    skipping one streamed column tile moves it;
  * on water, where the pair screen keeps every pair, the row realization
    is the replicated one to the floor; on ethylene, where it drops 70, both
    are the derivative of the ISDF-K SCF energy, since both realize the
    SCF's estimator (the Gram matrix over every product pair): the rows to
    `FD_BAR` of a Richardson central difference, the whole form within
    FIT_REALIZATION_FORCE_TOL of the rows and of the difference beyond the
    rows' bar, two realizations of the one fit (measured 3.5e-12 apart, both
    9.0e-10 and 4.2e-10 from the difference; a 1e-6 relative defect of the
    whole form's D_bar puts them 3.4e-7 apart);
  * on a distributed ISDF-K SCF the skeleton reuses the SCF's own tiles
    (bitwise the refit at the same tiles) and never builds the whole ISDFJK;
  * no rank binds a (nao, nao, naux) array or one whole along the grid twice
    in any frame of either kernel (the `HeldCensus` of
    tests/test_mpi_routes.py), and the census finds both in the whole forms;
  * on a shape-sensitive BLAS (tests/test_force_serial_shaped.py's
    stand-in, a subprocess) the row kernel stays bitwise across rank
    counts and every fitted partial stays its own addends' bits.
"""
import importlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.df import incore

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.Base import skeleton_tiles  # noqa: E402
from src.Base import separable_ri  # noqa: E402
from src.Base.constants import (FIT_REALIZATION_FORCE_TOL,  # noqa: E402
                                ISDF_GRADIENT_FLOOR)
from src.Base.distributed_df import distributed_mean_field  # noqa: E402
from src.Base.distributed_isdf_jk import distributed_isdf_jk  # noqa: E402
from src.Base.isdf_jk import isdf_grid, isdf_jk  # noqa: E402
from src.Base.separable_ri import (RowFit, RowFitAdjoint,  # noqa: E402
                                   screened_layout)
from src.Base.skeleton_tiles import (FittedFockSkeleton,  # noqa: E402
                                     fitted_fock_skeleton,
                                     isdf_exchange_rows)
from src.Base.utils.mpi_grid import (current_comm, distributed,  # noqa: E402
                                     run_simulated)
from src.gradients.isdf_derivatives import (  # noqa: E402
    _auxmol_of, exchange_channel_skeleton, fock_partial_skeleton_df,
    isdf_exchange_skeleton, isdf_scf_handle, three_centre_adjoint,
    two_centre_adjoint)
from src.gradients.isdf_mean_field import isdf_mean_field_gradient  # noqa
from src.SingleReference.LinearResponse.rpa_energy import (  # noqa: E402
    exchange_channels)
from tests import test_force_serial_shaped as shaped  # noqa: E402
from tests.test_mpi_routes import HeldCensus  # noqa: E402

GEOMS = {
    'water': 'O 0 0 0.1173; H 0 0.7872 -0.4692; H 0 -0.7572 -0.4492',
    'ethylene': ('C 0.02 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9589 1.2321; H 0.03 0.9289 -1.2321; '
                 'H 0 -0.9289 -1.2021'),
    'formaldehyde': ('C 0 0 -0.5296; O 0 0.02 0.6763; H 0 0.9357 -1.1172; '
                     'H 0.02 -0.9557 -1.1172'),
}
XCS = ('pbe0', 'lrc-wpbeh')
SIZES = (2, 3, 8)
DF_AUX = 'cc-pvdz-jkfit'
ISDF_AUX = 'cc-pvdz-ri'
#: The row tile edge of the kernel gates: water's 444 points in 7 tiles.
ROW_BLOCK = 64
#: Central-difference steps (Bohr) of the Richardson gate.
FD_STEPS = (1e-3, 2e-3)
#: What the row realization may miss a Richardson difference of the ISDF-K
#: SCF energy by (Ha/Bohr).
FD_BAR = 5e-9
#: The row fit's working-set cap in the census (GB): one AO shell per block
#: of its three-centre pass, so a block is a small part of the AO range as on
#: a large system.
CENSUS_BLOCK_GB = 1e-4
#: The mean fields, converged once per case: {key: (mo_coeff, mo_occ, e)}.
SCF_ARRAYS = {}


def molecule(name, coords=None):
    """`name`/cc-pVDZ, at `coords` (Bohr) where given."""
    mol = gto.M(atom=GEOMS[name], basis='cc-pvdz', verbose=0, max_memory=8000)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    return mol


def unrun(name, xc, route, coords=None):
    """A fresh, unconverged mean field: its own molecule and fit objects."""
    mol = molecule(name, coords)
    base = dft.RKS(mol, xc=xc) if xc != 'hf' else scf.RHF(mol)
    mf = (base.density_fit(auxbasis=DF_AUX) if route == 'df'
          else isdf_jk(base, auxbasis=ISDF_AUX))
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    return mf


def mean_field(name, xc, route):
    """A fresh mean field object holding the case's one converged SCF."""
    key = (name, xc, route)
    if key not in SCF_ARRAYS:
        mf = unrun(name, xc, route)
        mf.kernel()
        SCF_ARRAYS[key] = (mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot)
    mf = unrun(name, xc, route)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot = SCF_ARRAYS[key]
    mf.converged = True
    return mf


def partial_of(mf, seed=7):
    """A symmetric MO partial, the stand-in of a relaxed density."""
    n = mf.mo_coeff.shape[1]
    r = np.random.default_rng(seed).standard_normal((n, n))
    return 0.5 * (r + r.T) / n


def channels_of(mf):
    """(the reference's channels, the Sigma_x - v_xc ones)."""
    chan = exchange_channels(mf)
    return chan, [(0.0, 1.0)] + [(o, -w) for o, w in chan]


def whole_fitted_skeleton(mf, auxmol, gamma, channels, coulomb=True):
    """The fitted skeleton on the whole (nao, nao, naux) tensors: the
    untiled reference contraction."""
    mol, C = mf.mol, mf.mo_coeff
    g = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    J = incore.aux_e2(mol, auxmol, intor='int3c2e',
                      aosym='s1').reshape(nao, nao, naux)
    V = auxmol.intor('int2c2e', aosym='s1')
    grad = np.zeros((mol.natm, 3))
    if coulomb:
        Va = np.linalg.solve(V, np.einsum('ab,abP->P', g, J))
        Vb = np.linalg.solve(V, np.einsum('ab,abP->P', D, J))
        grad += three_centre_adjoint(mol, auxmol, g[:, :, None] * Vb
                                     + D[:, :, None] * Va)
        grad += two_centre_adjoint(auxmol, -np.outer(Va, Vb))
    for omega, weight in channels:
        if weight == 0.0:
            continue
        if omega != 0.0:
            grad += exchange_channel_skeleton(mf, g, D, omega, weight)
            continue
        K = np.linalg.solve(V, J.reshape(-1, naux).T).reshape(naux, nao, nao)
        GK = np.einsum('ab,Pbc,cd->Pad', g, K, D, optimize=True)
        T = np.einsum('ab,bcP,cd,daQ->PQ', g, J, D, J, optimize=True)
        VTV = np.linalg.solve(V, np.linalg.solve(V, T).T).T
        grad += weight * (three_centre_adjoint(mol, auxmol,
                                               -GK.transpose(1, 2, 0))
                          + two_centre_adjoint(auxmol, 0.5 * VTV))
    return grad


def fitted_addends(mf, gamma, channels, coulomb=True):
    """(one-rank skeleton, its tiles' (natm, 3) addends, the tile count) of
    `fitted_fock_skeleton` for the Fock partial `gamma`."""
    C, occ = mf.mo_coeff, mf.mo_occ
    g = C @ (0.5 * (gamma + gamma.T)) @ C.T
    full = sum(w for o, w in channels if o == 0.0)
    Cd = C[:, occ > 0] * np.sqrt(occ[occ > 0])
    args = (mf.mol, _auxmol_of(mf), g, mf.make_rdm1())
    kw = dict(occ=Cd if full else None, coulomb=coulomb, exchange=full)
    sk = FittedFockSkeleton(*args, **kw)
    sk.prepare(range(len(sk.tiles)), None)
    addends = [sk.addend(t) for t in range(len(sk.tiles))]
    return fitted_fock_skeleton(*args, comm=None, **kw), addends, len(addends)


def fitted_over_ranks(case, size, plant=None):
    """Every rank's skeleton of the Fock partial of `case`, and the one
    rank's."""
    name, xc = case

    def rank(comm):
        mf = mean_field(name, xc, 'df')
        chan, _ = channels_of(mf)
        return fock_partial_skeleton_df(mf, _auxmol_of(mf), partial_of(mf),
                                        0, channels=chan)

    if plant is None:
        return run_simulated(rank, size)
    original = FittedFockSkeleton.addend

    def planted(self, t):
        # rank 1's first tile dropped from its partial, or counted twice
        out = original(self, t)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1 and t == 1:
            return 0.0 * out if plant == 'dropped' else 2.0 * out
        return out

    FittedFockSkeleton.addend = planted
    try:
        return run_simulated(rank, size)
    finally:
        FittedFockSkeleton.addend = original


def fitted_distances(case, size, plant=None):
    """(worst |distributed - one rank|, every rank the serial skeleton's
    bits)."""
    name, xc = case
    mf = mean_field(name, xc, 'df')
    chan, _ = channels_of(mf)
    whole, _, _ = fitted_addends(mf, partial_of(mf), chan)
    with distributed(None):
        serial = fock_partial_skeleton_df(mf, _auxmol_of(mf), partial_of(mf),
                                          0, channels=chan)
    res = fitted_over_ranks(case, size, plant)
    # the four-centre long-range channel rides beside the reduction
    four = sum((exchange_channel_skeleton(
        mf, mf.mo_coeff @ partial_of(mf) @ mf.mo_coeff.T, mf.make_rdm1(),
        o, w) for o, w in chan if o != 0.0), np.zeros_like(whole))
    one = whole + four
    worst = max(float(np.abs(g - one).max()) for g in res)
    same = all(np.array_equal(g, serial) for g in res)
    return worst, same


CASES = [(name, xc) for name in GEOMS for xc in XCS]


@pytest.mark.parametrize('case', CASES, ids=['-'.join(c) for c in CASES])
def test_fitted_skeleton_is_the_whole_contraction(case):
    name, xc = case
    mf = mean_field(name, xc, 'df')
    auxmol, gamma = _auxmol_of(mf), partial_of(mf)
    chan, delta = channels_of(mf)
    for label, kw in (('fock', dict(channels=chan)),
                      ('xc', dict(channels=delta, coulomb=False))):
        got = fock_partial_skeleton_df(mf, auxmol, gamma, 0, **kw)
        ref = whole_fitted_skeleton(mf, auxmol, gamma, **kw)
        d = float(np.abs(got - ref).max())
        print(f'{name} {xc} {label}: |tiled - whole| {d:.2e} of '
              f'{np.abs(ref).max():.2e}')
        assert d <= ISDF_GRADIENT_FLOOR, (label, d)


@pytest.mark.parametrize('size', SIZES)
def test_fitted_skeleton_over_ranks(size):
    for case in CASES:
        worst, same = fitted_distances(case, size)
        print(f'{size} ranks {case}: |d| {worst:.2e}, every rank the serial '
              f'skeleton bitwise: {same}')
        assert same, f'{case}: a rank holds another skeleton than the serial'
        assert worst <= ISDF_GRADIENT_FLOOR, (case, worst)


@pytest.mark.parametrize('plant', ('dropped', 'doubled'))
def test_fitted_skeleton_planted_tiles_fail(plant):
    worst, same = fitted_distances(('ethylene', 'pbe0'), 3, plant)
    print(f'{plant}: |d| {worst:.2e}, bitwise {same}')
    assert not same
    assert worst > ISDF_GRADIENT_FLOOR


def row_densities(mf):
    """The three exchange skeletons of an ISDF-K chain, as (dm, dm_other,
    prefactor, channels): the mean-field force, the folded Fock partial and
    the Sigma_x - v_xc correction."""
    C, D = mf.mo_coeff, mf.make_rdm1()
    chan, delta = channels_of(mf)
    g = C @ partial_of(mf) @ C.T
    w = np.diag(np.linspace(-1.0, 1.0, C.shape[1]))
    return {'mean field': (D, None, 1.0, chan),
            'fock partial': (g, D, 2.0, chan),
            'sigma_x - v_xc': (C @ w @ C.T, D, 2.0, delta)}


def rows_kernel(mf, dm, dm_other, prefactor, channels, block=ROW_BLOCK,
                block_memory_gb=4.0):
    """`isdf_exchange_rows` on the mean field's grid at `block`, flattened."""
    wd = mf.with_df
    if not wd._built:
        wd.build()
    layout = screened_layout(mf.mol, wd.coords, block=block)
    centre, points, _ = isdf_exchange_rows(
        mf.mol, _auxmol_of(mf), wd.coords, layout, dm, dm_other=dm_other,
        prefactor=prefactor, channels=channels, block=block,
        block_memory_gb=block_memory_gb)
    return np.concatenate([centre.ravel(), points.ravel()])


@pytest.mark.parametrize('size', SIZES)
def test_row_exchange_skeleton_is_bitwise_over_ranks(size):
    for name in GEOMS:
        for xc in XCS:
            mf = mean_field(name, xc, 'isdf')
            one = {k: rows_kernel(mf, *v) for k, v in row_densities(mf).items()}

            def rank(comm):
                mine = mean_field(name, xc, 'isdf')
                return {k: rows_kernel(mine, *v)
                        for k, v in row_densities(mine).items()}

            res = run_simulated(rank, size)
            for k in one:
                ok = all(np.array_equal(r[k], one[k]) for r in res)
                assert ok, f'{size} ranks {name} {xc} {k}: not the one-rank bits'


def test_row_exchange_skeleton_planted_tile_fails():
    mf = mean_field('water', 'pbe0', 'isdf')
    dm, other, p, chan = row_densities(mf)['fock partial']
    one = rows_kernel(mf, dm, other, p, chan)
    original = skeleton_tiles.broadcast_rows
    seen = {}

    def planted(a, root, comm):
        out = original(a, root, comm)
        rank = 0 if comm is None else comm.Get_rank()
        count = seen[rank] = seen.get(rank, 0) + 1
        if rank == 1 and count == 3:
            out[...] = 0.0
        return out

    def rank(comm):
        return rows_kernel(mean_field('water', 'pbe0', 'isdf'), dm, other, p,
                           chan)

    skeleton_tiles.broadcast_rows = planted
    try:
        res = run_simulated(rank, 3)
    finally:
        skeleton_tiles.broadcast_rows = original
    assert not all(np.array_equal(r, one) for r in res)


def test_rows_is_replicated_where_every_pair_is_kept():
    mf = mean_field('water', 'pbe0', 'isdf')
    wd = mf.with_df
    wd.build()
    layout = separable_ri.test_set_layout(mf.mol, wd.coords)
    assert len(layout[0]) == mf.mol.nao_nr() ** 2
    rows = isdf_mean_field_gradient(mf, fit='rows')
    whole = isdf_mean_field_gradient(mf, fit='replicated')
    d = float(np.abs(rows - whole).max())
    print(f'water PBE0 mean-field force, rows - replicated {d:.2e}')
    assert d <= ISDF_GRADIENT_FLOOR


def test_the_distributed_screen_is_the_whole_screen():
    for name in GEOMS:
        mf = mean_field(name, 'pbe0', 'isdf')
        mf.with_df.build()
        crd = mf.with_df.coords
        whole = separable_ri.test_set_layout(mf.mol, crd)
        for size in (1, 3):
            got = run_simulated(
                lambda comm: screened_layout(mf.mol, crd, block=ROW_BLOCK),
                size)
            assert all(all(np.array_equal(a, b) for a, b in zip(g, whole))
                       for g in got), (name, size)


def test_both_skeletons_are_the_scf_energys_derivative():
    """Richardson central difference of the ISDF-K SCF energy of ethylene
    PBE0 against both realizations' mean-field force, at a carbon z and a
    hydrogen y: the row realization to `FD_BAR`, the whole form within
    FIT_REALIZATION_FORCE_TOL of the rows and of that bar."""
    mol0 = molecule('ethylene')
    R0 = mol0.atom_coords()
    mf = unrun('ethylene', 'pbe0', 'isdf')
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    grad = {fit: isdf_mean_field_gradient(mf, fit=fit)
            for fit in ('rows', 'replicated')}
    apart = float(np.abs(grad['rows'] - grad['replicated']).max())
    print(f'rows - replicated {apart:.2e} = '
          f'{apart / FIT_REALIZATION_FORCE_TOL:.3f} of '
          f'FIT_REALIZATION_FORCE_TOL {FIT_REALIZATION_FORCE_TOL:.0e}')
    assert apart <= FIT_REALIZATION_FORCE_TOL
    for atom, x in ((0, 2), (2, 1)):
        diff = {}
        for h in FD_STEPS:
            e = []
            for s in (1, -1):
                R = R0.copy()
                R[atom, x] += s * h
                m = unrun('ethylene', 'pbe0', 'isdf', coords=R)
                m.conv_tol, m.conv_tol_grad = 1e-13, 1e-10
                e.append(m.kernel())
            diff[h] = (e[0] - e[1]) / (2 * h)
        fd = (4 * diff[FD_STEPS[0]] - diff[FD_STEPS[1]]) / 3
        miss = {fit: float(g[atom, x] - fd) for fit, g in grad.items()}
        print(f'atom {atom} x{x}: rows - fd {miss["rows"]:+.2e}, '
              f'replicated - fd {miss["replicated"]:+.2e}')
        assert abs(miss['rows']) <= FD_BAR
        assert abs(miss['replicated']) <= FD_BAR + FIT_REALIZATION_FORCE_TOL


@pytest.mark.parametrize('size', (2, 3))
def test_the_distributed_scf_tiles_are_reused(size):
    """On a distributed ISDF-K SCF (tiles of ROW_BLOCK points) the mean
    field's exchange skeleton takes the SCF's own M^T and collocation tiles,
    bitwise the row fit rebuilt at those tiles, and the whole ISDFJK is never
    built."""
    def rank(comm):
        mf = unrun('water', 'lrc-wpbeh', 'isdf')
        distributed_isdf_jk(mf, comm, tile=ROW_BLOCK)
        distributed_mean_field(mf)
        handle = isdf_scf_handle(mf)
        dm, _, _, chan = row_densities(mf)['mean field']
        force = isdf_exchange_skeleton(mf, dm=dm, channels=chan)
        wd = mf.with_df
        crd = isdf_grid(mf.mol, radii=wd.radii, auxbasis=wd.auxbasis,
                        n_start=wd.n_start)
        layout = screened_layout(mf.mol, crd, block=ROW_BLOCK)
        runs = []
        for mt, X in ((handle.MT, handle.X), (None, None)):
            centre, points, _ = isdf_exchange_rows(
                mf.mol, _auxmol_of(mf), crd, layout, dm, channels=chan,
                mt=mt, X=X, block=ROW_BLOCK)
            runs.append(np.concatenate([centre.ravel(), points.ravel()]))
        return (force, wd._built, np.array_equal(handle.coords, crd),
                np.array_equal(*runs))

    res = run_simulated(rank, size)
    assert all(np.array_equal(r[0], res[0][0]) for r in res)
    assert not any(r[1] for r in res), 'the whole ISDFJK was built'
    assert all(r[2] for r in res), "the SCF's points are not the grid's"
    assert all(r[3] for r in res), "the SCF's tiles are not the refit's bits"


class SkeletonCensus(HeldCensus):
    """`HeldCensus` flagging the skeletons' whole shapes: any array whole
    along the grid twice, and any whose axes hold (nao, nao, naux)."""

    def visit(self, obj, found, depth=0):
        if depth < 3 and isinstance(obj, (FittedFockSkeleton, RowFit,
                                          RowFitAdjoint)):
            return super().visit(vars(obj), found, depth + 1)
        return super().visit(obj, found, depth)

    def kind(self, a):
        shape = a.shape
        if self.M and sum(n == self.M for n in shape) >= 2:
            return 'grid square'
        three = sorted((self.nao, self.nao, self.naux))
        if a.ndim >= 3 and sorted(shape[-3:]) == three:
            return 'three-index'
        if a.ndim == 2 and sorted(shape) == sorted((self.nao ** 2, self.naux)):
            return 'three-index'
        return None


def census_of(fn, root, M, naux, nao):
    """The census of fn() traced from `root`'s frame down."""
    census = SkeletonCensus(root, M, naux, nao)
    census.trace(fn)
    return census


@pytest.mark.parametrize('size', (1, 3))
def test_no_rank_holds_a_whole_skeleton_array(size):
    def rank(comm):
        out = {}
        mf = mean_field('water', 'pbe0', 'df')
        aux = _auxmol_of(mf)
        chan, _ = channels_of(mf)
        dims = (0, aux.nao_nr(), mf.mol.nao_nr())
        out['fitted'] = census_of(
            lambda: fock_partial_skeleton_df(mf, aux, partial_of(mf), 0,
                                             channels=chan),
            fitted_fock_skeleton.__code__, *dims)
        out['fitted, whole'] = census_of(
            lambda: whole_fitted_skeleton(mf, aux, partial_of(mf), chan),
            whole_fitted_skeleton.__code__, *dims)
        mi = mean_field('water', 'pbe0', 'isdf')
        mi.with_df.build()
        dims = (len(mi.with_df.coords), _auxmol_of(mi).nao_nr(),
                mi.mol.nao_nr())
        dm, other, p, chan = row_densities(mi)['fock partial']
        # one shell per block of the fit, so water's AO range is not one block
        out['rows'] = census_of(
            lambda: rows_kernel(mi, dm, other, p, chan,
                                block_memory_gb=CENSUS_BLOCK_GB),
            isdf_exchange_rows.__code__, *dims)
        out['rows, whole'] = census_of(
            lambda: isdf_exchange_skeleton(mi, dm=dm, dm_other=other,
                                           prefactor=p, channels=chan,
                                           fit='replicated'),
            isdf_exchange_skeleton.__code__, *dims)
        return {k: (sorted(c.flags), c.bytes) for k, c in out.items()}

    res = run_simulated(rank, size)
    for r, got in enumerate(res):
        print(f'rank {r}: ' + ', '.join(f'{k} {f} {b / 1e6:.1f} MB'
                                        for k, (f, b) in got.items()))
        assert got['fitted'][0] == [] and got['rows'][0] == [], (r, got)
        assert 'three-index' in got['fitted, whole'][0]
        assert 'grid square' in got['rows, whole'][0]
        assert 'three-index' in got['rows, whole'][0]


def rank_shaped_fit(fit_rows):
    """`fit_rows` whose M^T tiles pass through one product over all of this
    rank's tiles stacked, on the shape-sensitive BLAS: a GEMM whose shape is
    the rank's share, which the row kernel must not contain."""
    def fit(*args, **kwargs):
        out = fit_rows(*args, **kwargs)
        if out.mt:
            order = sorted(out.mt)
            stacked = shaped.shape_mm(np.vstack([out.mt[t] for t in order]),
                                      np.eye(out.mt[order[0]].shape[1]))
            at = 0
            for t in order:
                n = len(out.mt[t])
                out.mt[t] = np.ascontiguousarray(stacked[at:at + n])
                at += n
        return out
    return fit


def shape_main(size, plant=False):
    """In a fresh interpreter: src imported on the shape-sensitive BLAS, the
    row kernel and the fitted skeleton's partials at one and `size` ranks;
    `plant` routes the row fit's M^T through a rank-shaped product."""
    shaped.import_src_on_the_shape_sensitive_blas()
    grid = importlib.import_module('src.Base.utils.mpi_grid')
    tiles = importlib.import_module('src.Base.skeleton_tiles')
    sep = importlib.import_module('src.Base.separable_ri')
    if plant:
        tiles.fit_rows = rank_shaped_fit(tiles.fit_rows)
    ref = mean_field('ethylene', 'lrc-wpbeh', 'isdf')
    ref.with_df.build()
    crd, aux = ref.with_df.coords, _auxmol_of(ref)
    dm, other, p, chan = row_densities(ref)['fock partial']
    dmf = mean_field('ethylene', 'pbe0', 'df')
    daux = _auxmol_of(dmf)
    C, occ = dmf.mo_coeff, dmf.mo_occ
    g = C @ partial_of(dmf) @ C.T
    Cd = C[:, occ > 0] * np.sqrt(occ[occ > 0])
    full = sum(w for o, w in channels_of(dmf)[0] if o == 0.0)

    def rank(comm):
        layout = sep.screened_layout(ref.mol, crd, block=ROW_BLOCK)
        centre, points, _ = tiles.isdf_exchange_rows(
            ref.mol, aux, crd, layout, dm, dm_other=other, prefactor=p,
            channels=chan, block=ROW_BLOCK)
        size_, r = (1, 0) if comm is None else (comm.Get_size(),
                                                comm.Get_rank())
        sk = tiles.FittedFockSkeleton(dmf.mol, daux, g, dmf.make_rdm1(),
                                      occ=Cd, exchange=full)
        mine = [int(t) for t in tiles.partition(len(sk.tiles), r, size_)]
        sk.prepare(mine, comm)
        partial = np.zeros((dmf.mol.natm, 3))
        for t in mine:
            partial += sk.addend(t)
        return np.concatenate([centre.ravel(), points.ravel()]), partial, mine

    one = grid.run_simulated(rank, 1)[0]
    sk = tiles.FittedFockSkeleton(dmf.mol, daux, g, dmf.make_rdm1(), occ=Cd,
                                  exchange=full)
    sk.prepare(range(len(sk.tiles)), None)
    addends = [sk.addend(t) for t in range(len(sk.tiles))]
    res = grid.run_simulated(rank, size)
    partial_ok = []
    for got, partial, mine in res:
        want = np.zeros_like(partial)
        for t in mine:
            want += addends[t]
        partial_ok.append(bool(np.array_equal(partial, want)))
    return {'perturbed': shaped.SHAPE_STATE['perturbed'],
            'rows bitwise': all(np.array_equal(r[0], one[0]) for r in res),
            'partials': partial_ok}


def shape_run(size, plant=False):
    """`shape_main(size, plant)` in a fresh interpreter, its verdict."""
    env = dict(os.environ, MPI4PY_RC_INITIALIZE='0',
               PYTHONPATH=os.pathsep.join(
                   [str(REPO)] + [p for p in [os.environ.get('PYTHONPATH')]
                                  if p]))
    out = subprocess.run([sys.executable, os.path.abspath(__file__),
                          str(size)] + (['plant'] if plant else []), env=env,
                         cwd=REPO, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith('RESULT '))
    return json.loads(line[len('RESULT '):])


@pytest.mark.parametrize('size', (3, 8))
def test_on_the_shape_sensitive_blas(size):
    got = shape_run(size)
    print(f'{size} ranks: {got}')
    assert got['perturbed'] > 0, 'the shape-sensitive BLAS scaled nothing'
    assert got['rows bitwise'], got
    assert all(got['partials']), got


def test_the_shape_sensitive_blas_sees_a_rank_shaped_product():
    got = shape_run(3, plant=True)
    print(f'planted, 3 ranks: {got}')
    assert not got['rows bitwise'], got


if __name__ == '__main__':
    if len(sys.argv) > 1:              # the child of `shape_run`
        print('RESULT ' + json.dumps(shape_main(int(sys.argv[1]),
                                                plant='plant' in sys.argv[2:])),
              flush=True)
    else:
        sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
