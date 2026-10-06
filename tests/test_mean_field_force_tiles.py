"""The ISDF-K mean field's own force in fixed tiles over the ranks
(`isdf_mean_field_gradient`: `skeleton_tiles`' one-electron, fitted Coulomb
and xc energy terms beside the row exchange skeleton), on the C1-distorted
water, ethylene and formaldehyde/cc-pVDZ of tests/test_skeleton_tiles.py,
Hartree-Fock, PBE0, LRC-wPBEh and TPSS (a meta-GGA):

  * it is pyscf's gradient of the same terms (`exchange_free_reference`, the
    functional with its exact exchange zeroed, grid response on) plus the row
    exchange skeleton, to `PYSCF_BAR`, on the unpruned grid pyscf's full
    response differentiates (`grids_response_cc` rebuilds the grid without
    the SCF's density pruning; on the pruned grid the two differ by the
    pruned points' own terms, 1.3e-9 on water PBE0);
  * the xc energy term alone follows a 4-point difference of E_xc at fixed
    density on the grid rebuilt at each geometry;
  * the force follows a Richardson difference of the ISDF-K SCF energy on
    the default grid, LRC-wPBEh and TPSS (Hartree-Fock and PBE0 are
    tests/test_one_fit_forces_follow_their_energy.py's);
  * every term and the whole force are the same bits at 1, 2, 3 and 8
    simulated ranks, ranks owning no atom, auxiliary or grid tile included,
    and each tile's addend is its owner's;
  * on a distributed ISDF-K SCF every rank holds one force, within
    `ISDF_GRADIENT_FLOOR` of the serial SCF's;
  * the translation residual stays at rounding.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.grad import rhf as rhf_grad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import ISDF_GRADIENT_FLOOR
from src.Base.dispersion import dispersion_gradient
from src.Base.distributed_df import distributed_mean_field
from src.Base.distributed_isdf_jk import distributed_isdf_jk
from src.Base.isdf_jk import exchange_free_reference, isdf_jk
from src.Base.skeleton_tiles import (fitted_coulomb_energy_skeleton,
                                     one_electron_energy_skeleton, tile_sum,
                                     xc_energy_grid_skeleton)
from src.Base.utils.mpi_grid import distributed, run_simulated
from src.gradients.isdf_derivatives import _auxmol_of
from src.gradients.isdf_mean_field import (isdf_exchange_skeleton,
                                           isdf_mean_field_gradient)
from src.SingleReference.LinearResponse.rpa_energy import xc_hybrid_coeff
from tests.test_skeleton_tiles import GEOMS

AUX = 'cc-pvdz-ri'
XCS = ('hf', 'pbe0', 'lrc-wpbeh', 'tpss')
#: What the tiled force may miss pyscf's by on one grid: the fitted Coulomb
#: skeleton's other summation order and the tiled xc terms (3-5e-14).
PYSCF_BAR = 1e-10
SIZES = (1, 2, 3, 8)
#: A grid tile wide enough that water's grid is 4 tiles: ranks 4-7 own none.
WIDE_TILE = 8192
#: The 4-point step of the xc energy's difference (Bohr) and its bar.
XC_STEP, XC_BAR = 1e-4, 1e-8
#: The Richardson steps of the ISDF-K SCF energy's difference (Bohr), the
#: components it takes and the SCF it converges.
SCF_STEPS = (2.5e-4, 5e-4)
COMPONENTS = ((0, 2), (1, 1))
CONV_TOL, CONV_TOL_GRAD = 1e-13, 1e-10
#: The row fit's tile edge of the distributed SCF.
ROW_TILE = 64
#: {(name, xc, cutoff): (mo_coeff, mo_occ, mo_energy, e_tot)}
SCF_ARRAYS = {}


def unrun(name, xc, cutoff=None, coords=None):
    """A fresh ISDF-K mean field of `name` (at `coords`, Bohr); `cutoff`
    the small-rho pruning, pyscf's default when None."""
    mol = gto.M(atom=GEOMS[name], basis='cc-pvdz', verbose=0, max_memory=8000)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    base = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
    if cutoff is not None and xc != 'hf':
        base.small_rho_cutoff = cutoff
    mf = isdf_jk(base, auxbasis=AUX)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = CONV_TOL, CONV_TOL_GRAD, 200
    return mf


def converged(name, xc, cutoff=None):
    """A fresh mean field object holding the case's one converged SCF."""
    key = (name, xc, cutoff)
    mf = unrun(name, xc, cutoff)
    if key not in SCF_ARRAYS:
        mf.kernel()
        assert mf.converged
        SCF_ARRAYS[key] = (mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot,
                           None if xc == 'hf' else mf.grids)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot, grids = SCF_ARRAYS[key]
    if grids is not None:
        mf.grids = grids
    mf.converged = True
    return mf


def pyscf_force(mf):
    """pyscf's gradient of the exchange-free reference, grid response on,
    plus dispersion and the row exchange skeleton: the untiled assembly."""
    g0 = exchange_free_reference(mf).Gradients()
    g0.grid_response = True
    out = np.asarray(g0.kernel()) + dispersion_gradient(mf)
    if xc_hybrid_coeff(mf)[1] != 0.0:
        out = out + isdf_exchange_skeleton(mf)
    return out


CASES = [(n, x) for n in GEOMS for x in XCS]


@pytest.mark.parametrize('case', CASES, ids=['-'.join(c) for c in CASES])
def test_the_tiled_force_is_pyscfs_on_one_grid(case):
    """The tiled terms against pyscf's whole ones on the unpruned grid."""
    mf = converged(*case, cutoff=0.0)
    force = isdf_mean_field_gradient(mf)
    apart = float(np.abs(force - pyscf_force(mf)).max())
    print(f'\n{case}: tiled - pyscf {apart:.1e}, translation '
          f'{np.abs(force.sum(axis=0)).max():.1e}')
    assert apart < PYSCF_BAR
    assert np.abs(force.sum(axis=0)).max() < PYSCF_BAR


def test_the_pruned_grid_moves_only_its_own_points():
    """On the SCF's density-pruned grid the tiled force differentiates the
    points that energy sums and pyscf's the whole grid: apart by more than
    rounding and by less than the pruned points could carry."""
    mf = converged('water', 'pbe0')
    apart = float(np.abs(isdf_mean_field_gradient(mf)
                         - pyscf_force(mf)).max())
    print(f'\npruned grid: tiled - pyscf {apart:.1e}')
    assert PYSCF_BAR < apart < ISDF_GRADIENT_FLOOR


def xc_energy(mol, mf, dm):
    """E_xc^DFT[dm] on the mean field's grid rebuilt on `mol`, unpruned."""
    ref = mf.grids
    grids = dft.gen_grid.Grids(mol)
    grids.level, grids.prune = ref.level, ref.prune
    grids.atom_grid, grids.radi_method = ref.atom_grid, ref.radi_method
    return mf._numint.nr_rks(mol, grids.build(), mf.xc, dm)[1]


@pytest.mark.parametrize('xc', ('pbe0', 'lrc-wpbeh', 'tpss'))
def test_the_xc_energy_term_follows_the_moving_grid(xc):
    """xc_energy_grid_skeleton against a 4-point difference of E_xc at fixed
    density, every atom and axis of water."""
    mf = converged('water', xc, cutoff=0.0)
    mol, dm = mf.mol, mf.make_rdm1()
    term = xc_energy_grid_skeleton(mol, mf.grids, mf._numint, mf.xc, dm)
    fd = np.zeros_like(term)
    for atom in range(mol.natm):
        for axis in range(3):
            e = {}
            for k in (2, 1, -1, -2):
                crd = mol.atom_coords().copy()
                crd[atom, axis] += k * XC_STEP
                e[k] = xc_energy(mol.set_geom_(crd, unit='Bohr',
                                               inplace=False), mf, dm)
            fd[atom, axis] = (8.0 * (e[1] - e[-1])
                              - (e[2] - e[-2])) / (12.0 * XC_STEP)
    miss = float(np.abs(term - fd).max())
    print(f'\n{xc}: xc energy term - FD {miss:.1e}')
    assert miss < XC_BAR


@pytest.mark.parametrize('case', [(n, 'lrc-wpbeh') for n in GEOMS]
                         + [('water', 'tpss')],
                         ids=lambda c: '-'.join(c))
def test_the_force_follows_the_isdf_k_scf_energy(case):
    """Richardson central difference of the ISDF-K SCF energy on the default
    grid against the force, a heavy atom and a hydrogen."""
    mf = converged(*case)
    force = isdf_mean_field_gradient(mf)
    R0 = mf.mol.atom_coords()
    misses = []
    for atom, axis in COMPONENTS:
        d = {}
        for h in SCF_STEPS:
            e = []
            for s in (1, -1):
                R = R0.copy()
                R[atom, axis] += s * h
                m = unrun(*case, coords=R)
                e.append(m.kernel())
                assert m.converged
            d[h] = (e[0] - e[1]) / (2 * h)
        fd = (4 * d[SCF_STEPS[0]] - d[SCF_STEPS[1]]) / 3
        misses.append(float(force[atom, axis] - fd))
    print(f'\n{case}: force - FD {np.array2string(np.array(misses), precision=2)}')
    assert np.abs(misses).max() < ISDF_GRADIENT_FLOOR


def terms(mf, xc_tile=None):
    """The three tiled terms and the whole force of `mf`, one array."""
    mol = mf.mol
    dm = mf.make_rdm1()
    dme = rhf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    hcore = rhf_grad.Gradients(mf).hcore_generator(mol.copy())
    return np.stack([
        one_electron_energy_skeleton(mol, hcore, dm, dme),
        fitted_coulomb_energy_skeleton(mol, _auxmol_of(mf), dm),
        xc_energy_grid_skeleton(mol, mf.grids, mf._numint, mf.xc, dm,
                                tile=xc_tile),
        isdf_mean_field_gradient(mf)])


@pytest.mark.parametrize('xc', ('lrc-wpbeh', 'tpss'))
def test_every_term_is_the_same_bits_at_every_rank_count(xc):
    """Water: 3 atoms, 3 auxiliary tiles and (on the wide tile) 4 grid tiles
    of 30640 points, so at 8 ranks five own no atom or auxiliary tile and
    four no grid tile."""
    mf = converged('water', xc)
    with distributed(None):
        serial = terms(mf, WIDE_TILE)
    for size in SIZES:
        out = run_simulated(lambda comm: terms(mf, WIDE_TILE), size)
        for r, got in enumerate(out):
            assert np.array_equal(got, serial), (size, r)


@pytest.mark.parametrize('size', (2, 3, 8))
def test_each_tile_is_its_owners(size):
    """An addend that names the rank computing it: every rank's sum takes
    tile t from rank t % size, so a rank that computed or kept another's
    tile would show."""
    def rank(comm):
        me = comm.Get_rank()
        return tile_sum(lambda t: np.full((2, 3), 10.0 ** t * (me + 1)), 5, 2)

    want = sum(10.0 ** t * (t % size + 1) for t in range(5))
    for got in run_simulated(rank, size):
        assert np.array_equal(got, np.full((2, 3), want))


@pytest.mark.parametrize('size', (2, 3))
def test_on_a_distributed_scf(size):
    """On the distributed ISDF-K SCF every rank's force is rank 0's, within
    the floor of the serial SCF's force."""
    serial = isdf_mean_field_gradient(converged('water', 'lrc-wpbeh'))

    def rank(comm):
        mf = unrun('water', 'lrc-wpbeh')
        distributed_isdf_jk(mf, comm, tile=ROW_TILE)
        distributed_mean_field(mf)
        return isdf_mean_field_gradient(mf)

    out = run_simulated(rank, size)
    assert all(np.array_equal(f, out[0]) for f in out)
    apart = float(np.abs(out[0] - serial).max())
    print(f'\n{size} ranks: distributed SCF - serial {apart:.1e}')
    assert apart < ISDF_GRADIENT_FLOOR


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
