"""<p| Sigma_x - v_xc |q> built over the ranks that converged the mean field.

The static exchange is one K and one exchange-correlation potential on the
DFT grid, and it carries no state index: a whole quasiparticle window and the
whole BSE diagonal are built from one of them. That is what made it the
largest replicated stage left once the SCF divided -- 7.4 s at pentacene,
whatever the rank count -- and it divides through exactly the handles the SCF
left behind: every rank makes the same three pyscf calls it makes serially,
each a collective that locksteps the density and adds every rank's rows of
the fitted tensor and block of the grid, and the finished matrix is locked to
rank 0's.

WHAT THE GATES SAY. Against rank 0's own serial build of the same mean field
the reduced matrix differs by 3.7e-15 (water/cc-pVDZ) to 1.2e-13
(ethylene/cc-pVTZ) on a matrix of order 1 Ha -- the re-associated sums over
the auxiliary index and over the grid points, nothing else. It is identical
bit for bit on every rank, which the serial build is NOT: two ranks running
pyscf's own threaded K and quadrature on the same orbitals disagree in their
last bits, and the lockstep is what removes that from the quasiparticle
equation.

A WORKER'S OWN SERIAL BUILD IS A DIFFERENT QUANTITY, which is why the
comparison is rank 0's. A worker's `initialize_grids` builds nothing while the
quadrature handle is installed, so its `mf.grids` is unbuilt after the SCF
and a serial build there prunes a fresh grid against the CONVERGED density
where rank 0's was pruned against the initial guess: measured 5.3e-11 apart,
the grid's difference and not the reduction's.

A MEAN FIELD CONVERGED ON EACH RANK has no slices and takes the replicated
build, and the matrix is rank 0's there too: the orbitals, occupations and
correction density are locked at entry and the product once more at exit, so
a rank whose SCF stopped on other last bits, or whose own K and quadrature
round differently, still hands the quasiparticle equation rank 0's matrix.

EVERY GATE HERE WAS SHOWN TO FAIL: scaling one rank's exchange partial by
1 + 1e-6 moves the matrix 3.9e-08 away from the serial build, against the
1e-12 the reduction costs; without the exit lockstep a rank whose own build
is one ulp off returns it, and without the entry lockstep a rank keeps the
orbitals one ulp off that it arrived with.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base import distributed_df
from src.Base.distributed_df import (DistributedDF, distributed_handles,
                                     distributed_mean_field,
                                     release_distributed)
from src.Base.utils.mpi_grid import (current_comm, distributed,
                                     lockstep_stats, run_simulated)
from src.SingleReference.GW import qp_solve
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             static_exchange_mean_field_matrix)
from tests.test_distributed_df import (ETHYLENE, SIZES, WATER, mean_field)

#: Hartree, on a matrix of order 1. The reduce re-associates the sum over the
#: auxiliary index and the sum over the grid points; measured 3.7e-15 at
#: water/cc-pVDZ and 1.2e-13 at ethylene/cc-pVTZ, at two and three ranks.
MATRIX_TOL = 1e-12
#: Relative scaling of one rank's exchange partial, the size of a
#: rounding-sized defect.
PERTURBATION = 1e-6
CASES = [('water', WATER, 'cc-pvdz'), ('ethylene', ETHYLENE, 'cc-pvtz')]
STATES = [3, 4, 5]
#: The rank that arrives with different last bits.
ODD_RANK = 1


def converged_on_ranks(fn, size, atom, basis):
    """`fn(comm, mf)` on every rank, each with its own mean field converged by
    the ranks together, so the handles are the SCF's own."""
    mfs = [mean_field(atom, basis, 'pbe0') for _ in range(size)]

    def one_rank(comm):
        mf = mfs[comm.Get_rank()]
        distributed_mean_field(mf, comm)
        return fn(comm, mf)

    return run_simulated(one_rank, size)


def built_both_ways(comm, mf):
    """The distributed matrix, and rank 0's serial build of the same thing --
    inside `distributed(None)`, since rank 0 alone makes it and a build that
    found the ranks' communicator would wait for the others."""
    out = dict(distributed=static_exchange_mean_field_matrix(mf, mf.mol,
                                                             comm=comm),
               with_df=type(mf.with_df).__name__,
               numint=type(mf._numint).__name__,
               handles=distributed_handles(mf, comm) is not None)
    if comm.Get_rank() == 0:
        with distributed(None):
            out['serial'] = static_exchange_mean_field_matrix(mf, mf.mol)
    return out


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name,atom,basis', CASES)
def test_distributed_matrix_matches_the_serial_build(size, name, atom, basis):
    """The reduced matrix is the serial one to the reduction's last bits, and
    the same bits on every rank."""
    out = converged_on_ranks(built_both_ways, size, atom, basis)
    assert np.abs(out[0]['distributed'] - out[0]['serial']).max() <= MATRIX_TOL
    for r in out:
        assert np.array_equal(r['distributed'], out[0]['distributed'])
        # The handles stay reachable and stay OFF the mean field: a
        # downstream loop() or response kernel must meet pyscf's own objects.
        assert r['handles']
        assert (r['with_df'], r['numint']) == ('DF', 'NumInt')


@pytest.mark.parametrize('size', SIZES)
def test_the_context_is_the_communicator(size):
    """Without `comm=`, inside the ranks' `distributed` block, the build finds
    the communicator and is the same collective bit for bit -- which is how a
    quasiparticle route that never names a communicator reaches the slices."""
    def both(comm, mf):
        return (static_exchange_mean_field_matrix(mf, mf.mol, comm=comm),
                static_exchange_mean_field_matrix(mf, mf.mol))

    out = converged_on_ranks(both, size, WATER, 'cc-pvdz')
    for explicit, implicit in out:
        assert np.array_equal(explicit, implicit)
        assert np.array_equal(implicit, out[0][1])


@pytest.mark.parametrize('size', SIZES)
def test_the_diagonal_is_the_matrix_indexed(size):
    """`static_exchange_diagonal(comm=)` is the distributed matrix's diagonal,
    bit for bit: indexing is all that is left once the build is done."""
    def both(comm, mf):
        matrix = static_exchange_mean_field_matrix(mf, mf.mol, comm=comm)
        return (static_exchange_diagonal(mf, mf.mol, STATES, comm=comm),
                np.diag(matrix)[STATES])

    for diagonal, indexed in converged_on_ranks(both, size, WATER, 'cc-pvdz'):
        assert np.array_equal(diagonal, indexed)


def test_a_mean_field_without_handles_falls_back():
    """A mean field converged one rank at a time has no slices to answer
    with, and the build stays the replicated one, bitwise.

    Not a fallback to be silent about: making the slices here costs the whole
    build again, which is more than this stage takes below about eight ranks.
    """
    mf = mean_field(WATER, 'cc-pvdz', 'pbe0')
    mf.kernel()
    serial = static_exchange_mean_field_matrix(mf, mf.mol)
    out = run_simulated(
        lambda comm: static_exchange_mean_field_matrix(mf, mf.mol, comm=comm),
        2)
    assert all(np.array_equal(m, serial) for m in out)


def one_ulp(a):
    """`a` moved one ulp per element, the sign drawn once and fixed."""
    sign = np.random.default_rng(0).choice([-1.0, 1.0], np.shape(a))
    return sign * np.spacing(np.abs(a))


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('where', ['orbitals', 'build'])
def test_the_replicated_build_is_rank_zeros(size, where, monkeypatch):
    """A mean field converged on each rank carries no slices, and the matrix
    it builds is still rank 0's on every rank, bit for bit.

    'orbitals': rank 1's mo_coeff one ulp off per element, what a node whose
    SCF stopped on other last bits hands in; the entry lockstep writes rank
    0's into it. 'build': rank 1's own K and quadrature one ulp off on the
    same inputs, what a node whose threaded arithmetic rounds differently
    produces; the exit lockstep replaces it. The audit counts the one repair
    on rank 1 and none anywhere else.
    """
    mfs = [mean_field(WATER, 'cc-pvdz', 'pbe0') for _ in range(size)]
    for mf in mfs:
        mf.kernel()
    reference = static_exchange_mean_field_matrix(mfs[0], mfs[0].mol)
    if where == 'orbitals':
        mfs[ODD_RANK].mo_coeff += one_ulp(mfs[ODD_RANK].mo_coeff)
    else:
        local = qp_solve._local_static_exchange

        def rounds_differently(mf, mol, dm_correction, exchange):
            out = local(mf, mol, dm_correction, exchange)
            if current_comm().Get_rank() == ODD_RANK:
                out = out + one_ulp(out)
            return out

        monkeypatch.setattr(qp_solve, '_local_static_exchange',
                            rounds_differently)

    def one_rank(comm):
        mf = mfs[comm.Get_rank()]
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            matrix = static_exchange_mean_field_matrix(mf, mf.mol)
            return matrix, lockstep_stats(), np.asarray(mf.mo_coeff).copy()

    out = run_simulated(one_rank, size)
    for matrix, _, orbitals in out:
        assert np.array_equal(matrix, reference)
        assert np.array_equal(orbitals, out[0][2])
    assert ([stats['mismatched_calls'] for _, stats, _ in out]
            == [int(r == ODD_RANK) for r in range(size)])


def test_comm_none_is_the_serial_path():
    """No communicator, and nothing about the build changes."""
    mf = mean_field(WATER, 'cc-pvdz', 'pbe0')
    mf.kernel()
    assert np.array_equal(static_exchange_mean_field_matrix(mf, mf.mol),
                          static_exchange_mean_field_matrix(mf, mf.mol,
                                                            comm=None))


@pytest.mark.parametrize('size', SIZES)
def test_the_scf_record_survives_and_the_slice_can_go(size):
    """A later stage on the SCF's handles leaves the SCF's timings alone, and
    `release_distributed` gives the slice back."""
    def use_and_release(comm, mf):
        before = dict(mf._distributed_timings)
        static_exchange_mean_field_matrix(mf, mf.mol, comm=comm)
        after = dict(mf._distributed_timings)
        handle = distributed_handles(mf, comm)[0]
        release_distributed(mf)
        return before == after, handle._cderi, distributed_handles(mf, comm)

    for untouched, cderi, handles in converged_on_ranks(use_and_release, size,
                                                        WATER, 'cc-pvdz'):
        assert untouched
        assert cderi is None
        assert handles is None


def test_the_ranks_must_agree_on_having_the_handles():
    """A worker whose handles were released while rank 0's were not says so,
    instead of building a local matrix while rank 0 waits for its partials.

    The decision is rank 0's and it is broadcast, so the failure is one rank
    raising a named error rather than the whole job hanging in a collective
    that only some of it entered.
    """
    def release_on_one(comm, mf):
        if comm.Get_rank() == 1:
            release_distributed(mf)
        return static_exchange_mean_field_matrix(mf, mf.mol, comm=comm)

    with pytest.raises(RuntimeError, match='released together'):
        converged_on_ranks(release_on_one, 2, WATER, 'cc-pvdz')


def test_a_perturbed_partial_moves_the_matrix(monkeypatch):
    """One rank's exchange partial scaled by 1 + 1e-6 fails the gate, so the
    workers' rows are reaching the matrix rank 0 returns."""
    partial_jk = DistributedDF.partial_jk

    def perturbed(self, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
        vj, vk = partial_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        if self.comm.Get_rank() == 1 and vk is not None:
            vk = vk * (1 + PERTURBATION)
        return vj, vk

    out = converged_on_ranks(built_both_ways, 2, WATER, 'cc-pvdz')
    monkeypatch.setattr(distributed_df.DistributedDF, 'partial_jk', perturbed)
    moved = converged_on_ranks(built_both_ways, 2, WATER, 'cc-pvdz')
    assert np.abs(moved[0]['distributed'] - out[0]['serial']).max() > MATRIX_TOL


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
