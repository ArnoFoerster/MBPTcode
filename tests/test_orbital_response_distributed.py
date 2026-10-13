"""The orbital response under ranks runs on the distributed SCF's handle.

Under ranks every Fock piece of the orbital response (`check_scf_quality`,
`fock_mo`, `response_kernel`, the multiplier solve's matvecs, the
Sigma_x - v_xc correction) is answered inside `distributed_fock(mf,
build=False)` by the ISDF-K SCF's tiles and each rank's slice of the grid,
not by the mean field's serial `ISDFJK`, which would build the (M, M) Gram
matrix, its M^3 Cholesky and X, M and L whole on every rank.

Gated here:
  * `lgmres_root_driven` serially is scipy's `lgmres` call, bitwise; under
    2 and 3 simulated ranks, with a matvec that is a collective and a rank
    whose own arithmetic differs, every rank applies the operator the same
    number of times and ends on rank 0's solution, which is the serial one;
    rank 0 raising raises on every rank instead of leaving them in a
    reduction;
  * `ISDFJK.build` refuses under ranks once a `DistributedISDFJK` divides it,
    and builds serially;
  * `DistributedNumInt.nr_rks_fxc` on the kernel `cache_xc_kernel` returned
    is pyscf's whole-grid f_xc response within round-off at 2, 3 and 8 ranks,
    and at 3 with the grid left whole on rank 0 (ranks owning no point), the
    same bits on every rank.
The whole state-pair force, `_built` False on every rank, is gated in
`test_state_pair_force_distributed_ks.py`.
"""
import os
import sys
import copy
import threading

import numpy as np
import pytest
import scipy.sparse.linalg
from pyscf import dft, gto
from pyscf.dft import numint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.distributed_df import distributed_fock
from src.Base.distributed_isdf_jk import DistributedISDFJK
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.utils.krylov import lgmres_root_driven
from src.Base.utils.mpi_grid import distributed, reduce_sum, run_simulated

WATER = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
BASIS = 'cc-pvdz'
N = 40
ATOL = 1e-12
MAX_ITER = 200
FXC_ROUNDOFF = 1e-12
#: small enough that a replicated lgmres takes rank 0's iteration count
RHS_DRIFT = 1e-13
_LOCK = threading.Lock()


def system():
    """A nonsymmetric, diagonally dominant operator, its diagonal
    preconditioner and a right-hand side."""
    rng = np.random.default_rng(7)
    a = np.diag(np.linspace(1.0, 5.0, N)) + 0.05 * rng.standard_normal((N, N))
    return a, (lambda v: v / np.diag(a)), rng.standard_normal(N)


def test_root_driven_lgmres_is_scipys_serially():
    a, precond, rhs = system()
    op = scipy.sparse.linalg.LinearOperator((N, N), matvec=lambda v: a @ v)
    pre = scipy.sparse.linalg.LinearOperator((N, N), matvec=precond)
    x_ref, info_ref = scipy.sparse.linalg.lgmres(op, rhs, M=pre, rtol=0.0,
                                                 atol=ATOL, maxiter=MAX_ITER)
    with distributed(None):
        x, info = lgmres_root_driven(lambda v: a @ v, rhs, precond, ATOL,
                                     MAX_ITER)
    assert info == info_ref == 0
    assert np.array_equal(x, x_ref)


@pytest.mark.parametrize('size', (2, 3))
def test_every_rank_ends_on_rank_zeros_solution(size):
    """The matvec reduces over the ranks and a rank other than 0 holds its
    right-hand side `RHS_DRIFT` apart, as a replicated input from another
    rank's arithmetic may: every rank still ends on rank 0's solution, where
    a replicated lgmres would end on each rank's own (or, drifted further, on
    another iteration count, pairing mismatched reductions)."""
    a, precond, rhs = system()
    calls = {}

    def solve(comm):
        rank = comm.Get_rank()
        rows = np.array_split(np.arange(N), comm.Get_size())[rank]

        def matvec(v):
            with _LOCK:
                calls[rank] = calls.get(rank, 0) + 1
            out = np.zeros(N)
            out[rows] = a[rows] @ v
            reduce_sum(out, comm)
            return out
        mine = rhs + RHS_DRIFT * rank
        return lgmres_root_driven(matvec, mine, precond, ATOL, MAX_ITER)

    out = run_simulated(solve, size)
    with distributed(None):
        x_ref, _ = lgmres_root_driven(lambda v: a @ v, rhs, precond, ATOL,
                                      MAX_ITER)
    assert len(set(calls.values())) == 1, calls
    for x, info in out:
        assert info == 0
        assert np.array_equal(x, out[0][0])
    assert np.abs(out[0][0] - x_ref).max() < 1e-10
    assert np.abs(a @ out[0][0] - rhs).max() < 1e-10


def test_a_raise_on_rank_zero_reaches_every_rank():
    a, precond, rhs = system()
    seen = []

    def solve(comm):
        count = [0]

        def failing(v):
            count[0] += 1
            if comm.Get_rank() == 0 and count[0] == 3:
                raise FloatingPointError('the preconditioner diverged')
            return precond(v)
        try:
            return lgmres_root_driven(lambda v: a @ v, rhs, failing, ATOL,
                                      MAX_ITER)
        except Exception as exc:
            with _LOCK:
                seen.append((comm.Get_rank(), type(exc).__name__, str(exc)))
            return None

    assert run_simulated(solve, 3) == [None] * 3
    assert sorted(r for r, _, _ in seen) == [0, 1, 2]
    for rank, kind, msg in seen:
        assert 'the preconditioner diverged' in msg, (rank, kind, msg)


def isdf_mean_field(mol, xc='pbe0'):
    mf = dft.RKS(mol, xc=xc)
    counts, n_start = resolve_isdf_grid('G1', BASIS, ['H', 'O'],
                                        auxbasis=BASIS + '-ri')
    return isdf_jk(mf, auxbasis=BASIS + '-ri', counts=counts, n_start=n_start)


def test_a_divided_isdfjk_refuses_to_build_whole_under_ranks():
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)

    def attempt(comm):
        mf = isdf_mean_field(mol)
        DistributedISDFJK(mf.with_df, comm=comm)
        with pytest.raises(RuntimeError, match='distributed_fock'):
            mf.with_df.build()
        return mf.with_df._built

    assert run_simulated(attempt, 2) == [False, False]
    mf = isdf_mean_field(mol)
    DistributedISDFJK(mf.with_df, comm=None)
    with distributed(None):
        assert mf.with_df.build()._built


@pytest.fixture(scope='module')
def converged():
    """Water PBE0, converged serially, and pyscf's whole-grid f_xc response
    to one symmetric trial density."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-ri')
    mf.kernel()
    rng = np.random.default_rng(3)
    dm1 = rng.standard_normal((mol.nao, mol.nao))
    dm1 = dm1 + dm1.T
    ni = numint.NumInt()
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mf.mo_coeff,
                                        mf.mo_occ, 0)
    ref = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dm1, 0, 1, rho0, vxc,
                        fxc)
    return mol, mf, dm1, ref


@pytest.mark.parametrize('size, split', ((2, True), (3, True), (8, True),
                                         (3, False)))
def test_the_ranks_f_xc_response_is_the_whole_grids(converged, size, split):
    """`split=False` leaves every point on rank 0: ranks 1 and 2 own none."""
    mol, mf0, dm1, ref = converged

    def response(comm):
        mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-ri')
        mf.grids = copy.copy(mf0.grids)
        mf.mo_coeff, mf.mo_occ = mf0.mo_coeff.copy(), mf0.mo_occ.copy()
        with distributed_fock(mf, comm, split_grid=split) as handles:
            assert handles is not None
            ni = mf._numint
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mf.mo_coeff, mf.mo_occ, 0)
            points = ni._grids.weights.size
            out = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dm1.copy(), 0,
                                1, rho0, vxc, fxc)
        return points, out

    out = run_simulated(response, size)
    assert sum(p for p, _ in out) == mf0.grids.weights.size
    for _, v in out:
        assert np.array_equal(v, out[0][1])
    assert np.abs(out[0][1] - ref).max() < FXC_ROUNDOFF


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
