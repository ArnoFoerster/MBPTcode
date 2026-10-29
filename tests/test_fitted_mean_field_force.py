"""The force of a density-fitted restricted mean field, every term in fixed
tiles over the ranks (`isdf_mean_field.fitted_mean_field_gradient`).

pyscf's `mf.Gradients()` holds the whole derivative three-centre blocks, the
fitted exchange's mixed-basis density and the PCM's cavity integrals on every
rank whatever the rank count. The tiled force takes the one-electron terms
by atoms, the Coulomb and every exchange channel by auxiliary tiles (each
operator on its own metric, solved as pyscf's gradient solves it), the xc
energy by grid tiles and the PCM's integrals by cavity blocks, and adds every
tile's (natm, 3) addend in tile order.

Gates, formaldehyde/aug-cc-pVDZ, aug-cc-pVDZ-JKFIT, XC grid 4:
  * Hartree-Fock, gas and IEF-PCM toluene: pyscf's force within 1e-12;
  * PBE0 with the density pruning off: pyscf's within 1e-12 (pyscf's grid
    response regenerates the unpruned atomic grids, so on the SCF's pruned
    grid the two differ by the pruning, 2e-10 here);
  * LRC-wPBEh, unpruned: the attenuated metric is singular (131 of 236
    eigenvalues below pyscf's 1e-7), so both solve it on the eigenvalue
    pseudo-inverse, whose 1/lambda magnifies rounding: pyscf's long-range
    term sums over the atoms to 1.2e-9 where every exact form sums to zero.
    The tiled one within 1e-8 of pyscf's and its net force below 1e-12;
  * LRC-wPBEh in toluene on the pruned grid: the five-point difference of the
    SCF energy along two random directions within 5e-9 (the difference's own
    floor at conv_tol 1e-12 and step 1e-3 Bohr is about 1e-9);
  * 2 and 3 simulated ranks: every rank the one-rank force, bit for bit;
  * at 3 simulated ranks through `mean_field_skeleton_force`, the entry every
    chain takes: the derivative three-centre integrals of each auxiliary tile
    and the cavity integrals of each PCM point are asked of libcint by one
    rank only, together every tile and every point once (pyscf's force asks
    each rank for all of them);
  * on the distributed DF SCF (`distributed_mean_field`, each rank its slice
    of the fitted tensor): every rank the same bits, within
    CONVERGED_SCF_FORCE_TOL of the serial SCF's force (two SCFs, 6e-10
    here).
"""
import copy
import threading

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.df import incore

from src.Base.factorization.distributed_df import distributed_mean_field
from src.Base.numerics.constants import CONVERGED_SCF_FORCE_TOL
from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.isdf_mean_field import (fitted_force_supported,
                                           fitted_mean_field_gradient,
                                           mean_field_skeleton_force)
from src.gradients.skeleton_tiles import FittedFockSkeleton

FORMALDEHYDE = 'C 0 0 -0.6; O 0 0 0.6; H 0 0.94 -1.18; H 0 -0.94 -1.18'
TOLUENE_EPS = 2.3741
_CACHE = {}


def mean_field(xc, solvent, pruned=True, atom=FORMALDEHYDE, unit='Angstrom',
               run=True):
    """A converged DF mean field (unconverged with run=False); xc None is
    Hartree-Fock."""
    mol = gto.M(atom=atom, basis='aug-cc-pvdz', unit=unit, verbose=0)
    mf = (scf.RHF(mol) if xc is None else dft.RKS(mol, xc=xc)).density_fit(
        auxbasis='aug-cc-pvdz-jkfit')
    if xc is not None:
        mf.grids.level = 4
        if not pruned:
            mf.small_rho_cutoff = 0
    if solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = TOLUENE_EPS
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    if run:
        mf.kernel()
    return mf


def cached(xc, solvent, pruned=True):
    key = (xc, solvent, pruned)
    if key not in _CACHE:
        _CACHE[key] = mean_field(xc, solvent, pruned)
    return _CACHE[key]


def pyscf_force(mf):
    g = mf.Gradients()
    if hasattr(mf, 'xc'):
        g.grid_response = True
    return g.kernel()


def tiled(mf):
    with distributed(None):
        return fitted_mean_field_gradient(mf)


@pytest.mark.parametrize('solvent', [False, True], ids=['gas', 'toluene'])
def test_hartree_fock_force_is_pyscfs(solvent):
    mf = cached(None, solvent)
    assert fitted_force_supported(mf)
    assert np.abs(tiled(mf) - pyscf_force(mf)).max() <= 1e-12


def test_hybrid_force_is_pyscfs_on_an_unpruned_grid():
    mf = cached('pbe0', False, pruned=False)
    assert np.abs(tiled(mf) - pyscf_force(mf)).max() <= 1e-12


def test_long_range_channel_on_the_singular_metric():
    mf = cached('lrc-wpbeh', False, pruned=False)
    got, ref = tiled(mf), pyscf_force(mf)
    print(f'\nLRC-wPBEh: max |tiled - pyscf| {np.abs(got - ref).max():.2e}, '
          f'net force tiled {np.abs(got.sum(0)).max():.1e}, '
          f'pyscf {np.abs(ref.sum(0)).max():.1e} Ha/Bohr')
    assert np.abs(got - ref).max() <= 1e-8
    assert np.abs(got.sum(axis=0)).max() <= 1e-12


def test_force_is_the_energys_difference():
    mf = cached('lrc-wpbeh', True)
    got = tiled(mf)
    x0, step = mf.mol.atom_coords(), 1e-3
    rng = np.random.default_rng(3)
    for _ in range(2):
        d = rng.normal(size=x0.shape)
        d /= np.linalg.norm(d)
        e = [mean_field('lrc-wpbeh', True, atom=list(zip(
                 mf.mol.elements, x0 + k * step * d)), unit='Bohr').e_tot
             for k in (-2, -1, 1, 2)]
        fd = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * step)
        assert abs(float((got * d).sum()) - fd) <= 5e-9


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('xc,solvent', [(None, True), ('lrc-wpbeh', True)],
                         ids=['hf-toluene', 'lrc-wpbeh-toluene'])
def test_every_rank_count_gives_the_same_bits(xc, solvent, size):
    mf = cached(xc, solvent)
    serial = tiled(copy.deepcopy(mf))
    copies = [copy.deepcopy(mf) for _ in range(size)]
    out = run_simulated(lambda comm: fitted_mean_field_gradient(
        copies[comm.Get_rank()]), size)
    for grad in out:
        np.testing.assert_array_equal(grad, serial)


@pytest.mark.parametrize('size', [2, 3])
def test_on_a_distributed_scf(size):
    serial = tiled(cached('lrc-wpbeh', True))

    def rank(comm):
        mf = mean_field('lrc-wpbeh', True, run=False)
        distributed_mean_field(mf)
        return fitted_mean_field_gradient(mf)

    out = run_simulated(rank, size)
    for grad in out:
        np.testing.assert_array_equal(grad, out[0])
    apart = float(np.abs(out[0] - serial).max())
    print(f'\n{size} ranks: distributed SCF - serial {apart:.1e} Ha/Bohr')
    assert apart <= CONVERGED_SCF_FORCE_TOL


def test_each_rank_asks_only_for_its_own_integrals(monkeypatch):
    mf = cached('lrc-wpbeh', True)
    copies = [copy.deepcopy(mf) for _ in range(3)]
    asked, lock = {}, threading.Lock()
    own = incore.aux_e2

    def counted(mol, auxmol, intor='int3c2e', *args, **kwargs):
        if intor.startswith('int3c2e_ip1'):
            sl = kwargs.get('shls_slice')
            # a cavity block (fake charges) or one auxiliary tile's slab
            key = (('cavity', auxmol.natm) if sl is None
                   else ('aux', mol.omega, sl[4], sl[5]))
            with lock:
                asked.setdefault(threading.current_thread().name, []).append(key)
        return own(mol, auxmol, intor, *args, **kwargs)

    monkeypatch.setattr(incore, 'aux_e2', counted)
    run_simulated(lambda comm: mean_field_skeleton_force(
        copies[comm.Get_rank()]), 3)
    tiles = FittedFockSkeleton(mf.mol, mf.with_df.auxmol, np.eye(1), np.eye(1)).tiles
    for omega in (0.0, 0.2):
        per_rank = [{k[2:] for k in keys if k[0] == 'aux' and k[1] == omega}
                    for keys in asked.values()]
        assert sum(len(t) for t in per_rank) == len(tiles)
        assert set().union(*per_rank) == {(int(a), int(b)) for a, b in tiles}
    points = sum(k[1] for keys in asked.values() for k in keys if k[0] == 'cavity')
    assert points == mf.with_solvent.surface['grid_coords'].shape[0]
    assert len(asked) == 3
