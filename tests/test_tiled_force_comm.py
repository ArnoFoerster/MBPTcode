"""The tiled MP2 force keeps to its factors' communicator and its declared
tile budget (`gradients.df_correlation_force_tiled`).

  * the fitted mean-field addend runs over the factors' ranks, the
    communicator its Z-vector and correlation force run over, even when the
    caller's region is another: it ran over the current region, so a force
    called with comm= from inside a serial region took that addend whole on
    every rank;
  * an MP2State on the tiled route tiles its loop density and its force alike,
    on the Numerics layout's tile_gb: the density took mp2_pieces' default;
  * an MP2State on an interpolated (ISDF-K) mean field is refused when it is
    built: its force's Z-vector kernel and JK skeleton are the density fit's;
  * a replicated piece run on rank 0 alone (`on_root`) runs in a serial
    region, so a collective inside it does not wait for the other ranks;
  * the metric factor's choice between Cholesky and the eigen-replacement is
    rank 0's on every rank, in the correlation fit's whitening and in the
    fitted Fock skeleton: planted, a rank whose Cholesky fails falls back
    alone and the ranks no longer agree;
  * every rank holds rank 0's factor in rank 0's memory order: the others
    received it in C order, so their triangular solves took another BLAS path
    and the skeleton changed bits with the rank count.

Water, cc-pVDZ, frozen core, DF-HF (tests/adc_common.py), over 2 simulated
ranks.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import scf

from src.Base.factorization.distributed_df import metric_factor, rank0_metric_factor
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients import df_factors
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.gradients.skeleton_tiles import fitted_fock_skeleton
from src.gradients.state_models import HFState, MP2State
from tests.isdf_test_grid import g1
from tests.adc_common import ATOMS, AUX, NF, make_scf_loop, molecule
from tests.test_correlated_mean_field_force import TILED, water

#: GB: one occupied orbital per tile
TILE_GB = 1e-9


def test_the_mean_field_addend_runs_over_the_factors_ranks(monkeypatch):
    mol, mf = water()
    seen, own = [], tcf.mean_field_skeleton_force

    def recorded(m):
        comm = current_comm()
        seen.append(1 if comm is None else comm.Get_size())
        return own(m)
    monkeypatch.setattr(tcf, 'mean_field_skeleton_force', recorded)

    def run(comm):
        with distributed(None):
            return tcf.mp2_force_tiled(mf, mol, df_factors.mp2_factors, NF, AUX,
                                       comm=comm, **TILED, numerics=Numerics())[0]
    out = run_simulated(run, 2)
    assert seen == [2, 2], seen
    np.testing.assert_array_equal(out[0], out[1])


def test_the_loop_density_and_the_force_tile_alike(monkeypatch):
    tiles, own = [], tcf.mp2_pieces

    def recorded(factors, tile_gb=None, scaling=None):
        tiles.append(tile_gb)
        return own(factors, tile_gb, scaling)
    monkeypatch.setattr(tcf, 'mp2_pieces', recorded)
    mol = molecule(ATOMS)
    model = MP2State(HFState(mol, make_scf_loop), NF, AUX, force_route='tiled',
                     numerics=Numerics(layout={'tile_gb': TILE_GB}))
    h = np.zeros((mol.nao, mol.nao))
    model(h, 0.0)
    model.force(h, 0.0)
    assert len(tiles) == 2 and set(tiles) == {TILE_GB}, tiles


def test_an_mp2_state_on_an_interpolated_mean_field_is_refused():
    def make_isdf(mol):
        mf = isdf_jk_with_force(scf.RHF(mol), auxbasis=AUX, counts=g1(scf.RHF(mol)))
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        return mf
    mol = molecule(ATOMS, basis='cc-pvdz')
    with pytest.raises(ValueError, match='ISDF-K'):
        MP2State(HFState(mol, make_isdf), NF, AUX, numerics=Numerics())


def test_a_piece_on_rank_0_runs_in_a_serial_region():
    def piece():
        assert current_comm() is None, 'on_root ran fn over the ranks'
        return (np.ones(3),)

    def run(comm):
        return tcf.on_root(piece, [(3,)], comm)[0]
    for got in run_simulated(run, 2):
        np.testing.assert_array_equal(got, np.ones(3))


def rank_1_cannot_factor(monkeypatch):
    """Planted: scipy's Cholesky fails on rank 1's thread alone, as a
    rank whose threaded factorization meets a near-singular metric would;
    the returned dict maps a rank thread to its rank (run_simulated)."""
    ranks, own = {}, scipy.linalg.cholesky

    def cholesky(a, *args, **kwargs):
        if ranks.get(threading.get_ident()) == 1:
            raise scipy.linalg.LinAlgError('planted: not positive definite')
        return own(a, *args, **kwargs)
    monkeypatch.setattr(scipy.linalg, 'cholesky', cholesky)
    return ranks


def test_every_rank_takes_rank_0s_metric_factor(monkeypatch):
    mol, mf = water()
    auxmol = pyscf_df.addons.make_auxmol(mol, AUX)
    auxjk = pyscf_df.addons.make_auxmol(mol, mf.with_df.auxbasis)
    C = np.asarray(mf.mo_coeff)
    Co = C[:, :mol.nelectron // 2]
    g = C @ np.diag(np.linspace(-1.0, 1.0, C.shape[1])) @ C.T
    with distributed(None):
        X0 = dcf.whitening(auxmol)[1]
        f0 = fitted_fock_skeleton(mol, auxjk, g, 2.0 * Co @ Co.T,
                                  occ=np.sqrt(2.0) * Co, exchange=1.0)
    ranks = rank_1_cannot_factor(monkeypatch)

    def run(comm):
        ranks[threading.get_ident()] = comm.Get_rank()
        X = dcf.whitening(auxmol)[1]
        f = fitted_fock_skeleton(mol, auxjk, g, 2.0 * Co @ Co.T,
                                 occ=np.sqrt(2.0) * Co, exchange=1.0, comm=comm)
        return X, f
    for X, f in run_simulated(run, 2):
        np.testing.assert_array_equal(X, X0)
        np.testing.assert_allclose(f, f0, rtol=0, atol=1e-12)


def test_every_rank_holds_rank_0s_factor_in_its_memory_order():
    mol, mf = water()
    auxmol = pyscf_df.addons.make_auxmol(mol, AUX)
    L0, mode0 = metric_factor(mol, auxmol)
    for low, mode in run_simulated(
            lambda comm: rank0_metric_factor(mol, auxmol, comm), 3):
        assert mode == mode0
        assert low.flags.f_contiguous == L0.flags.f_contiguous
        np.testing.assert_array_equal(low, L0)
