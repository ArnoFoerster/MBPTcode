"""GDF factors from one open file, and built over the ranks (`pbc_gdf_build`).

On silicon (gth-szv, 3x1x1) at integral precision 1e-12, where the s1 and s2
builds agree to 3e-12:

* The factors read from one open cderi file are pyscf's per-pair sr_loop bit
  for bit: the Gamma pair (real), the k = k' pairs (packed triangle) and the
  rest alike.
* Built on one rank, the factors are those of the cderi file, and so are the
  MO factors of PBCDFIntegrals.
* Over three ranks, each building one AO-row slice, the rows assembled on the
  owners are the one-rank factors (`assert_same`: a few ulp under the
  recording BLAS, the vector class bar under another, where a slice's lattice
  sums and fit run in another call shape: 3.2e-13 on the cluster's MKL, 5e-13 of the
  largest element): the q = 0 neutralisation of a
  slice not starting at AO 0 takes its own overlap rows (pyscf's loader would
  take rows 0..i1-i0 and put the q = 0 blocks O(10) off), and every slice is
  fitted with the one metric from rank 0.
* With more ranks than slices the transfers are split into groups as well,
  each built as a subset, and with symmetry only the irreducible transfers
  are wanted. A subset fixes its own auxiliary gauge, so the comparison is on
  the pair Coulomb blocks sum_P B_P,mn conj(B_P,sl), which are the file's.
* The ranks of one node build exactly the transfers they own, so no row
  crosses the interconnect; a transfer owned on both nodes is built on both.
  On two nodes of three ranks, a node that owns every transfer builds the
  one-rank factors, as above; nodes that each own some build subsets,
  compared on the pair Coulomb blocks.
* The slice files are removed.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto, scf

from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.Periodic.pbc_gdf_build import (DistributedGDF, FileGDF,
                                                        build_plan, row_slices)
from src.SingleReference.Periodic.pbc_integrals import (PBCDFIntegrals,
                                                        gdf_pair_factors,
                                                        get_momentum_transfer_map)
from src.SingleReference.Periodic.pbc_symmetry import KMeshSymmetry
from tests.pins import assert_same

SI_LATTICE = np.array([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715],
                       [2.715, 2.715, 0.0]])


@pytest.fixture(scope='module')
def si(tmp_path_factory):
    cell = gto.M(a=SI_LATTICE, atom='Si 0 0 0; Si 1.3575 1.3575 1.3575',
                 basis='gth-szv', pseudo='gth-pade', verbose=0, precision=1e-12)
    mf = scf.KRHF(cell, cell.make_kpts([3, 1, 1]), exxdiv='ewald').density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    kpts = np.asarray(mf.kpts)
    kplus = get_momentum_transfer_map(cell, kpts)
    every = np.arange(len(kpts))
    work = str(tmp_path_factory.mktemp('slices'))
    return mf, kpts, kplus, FileGDF(mf.with_df)(kpts, kplus, every), work


def _distributed(mf, kpts, kplus, owned, work):
    return DistributedGDF(mf.with_df, work)(kpts, kplus, owned)


def _same_factors(got, want, q):
    """Two builds of one transfer's factors, as `assert_same` reads them:
    the real and imaginary parts as one float array."""
    assert_same(np.ascontiguousarray(got).view(np.float64),
                np.ascontiguousarray(want).view(np.float64), unit='1',
                quantity='vector', label=('transfer', q))


def _pair_coulomb(B):
    """sum_P B_P,mn conj(B_P,sl) of every pair of a transfer: (nk, n, n, n, n)."""
    return np.einsum('kPmn,kPsl->kmnls', B, B.conj(), optimize=True)


def test_one_open_file_is_sr_loop(si):
    mf, kpts, _, _, _ = si
    mydf = mf.with_df
    pairs = [(ki, kj) for ki in range(len(kpts)) for kj in range(len(kpts))]
    for ki, kj, B in gdf_pair_factors(mydf, kpts, pairs):
        ref = np.vstack([r + 1j * i for r, i, _ in
                         mydf.sr_loop([kpts[ki], kpts[kj]], compact=False)])
        assert np.array_equal(B, ref.reshape(B.shape))


def test_row_slices_cover_the_rows_once(si):
    cell = si[0].cell
    for n in (1, 2, 3, 8):
        slices = row_slices(cell, n)
        assert 1 <= len(slices) <= min(n, cell.nbas)
        assert slices[0][0] == 0 and slices[-1][1] == cell.nao_nr()
        assert all(a[1] == b[0] and a[3] == b[2] for a, b in zip(slices[:-1], slices[1:]))
        assert all(s[1] > s[0] for s in slices)


def test_one_rank_builds_the_file_factors(si):
    mf, kpts, kplus, ref, work = si
    one = _distributed(mf, kpts, kplus, np.arange(len(kpts)), work)
    assert max(np.abs(one[q] - ref[q]).max() for q in ref) < 1e-11
    dfints = PBCDFIntegrals.from_scf(mf.cell, mf)
    built = PBCDFIntegrals.from_scf(mf.cell, mf, factors=DistributedGDF(mf.with_df, work))
    assert max(np.abs(a - b).max() for a, b in zip(dfints.L, built.L)) < 1e-11
    assert os.listdir(work) == []


def test_row_slices_over_ranks_are_the_one_rank_factors(si):
    mf, kpts, kplus, _, work = si
    every = np.arange(len(kpts))
    one = _distributed(mf, kpts, kplus, every, work)
    assert len(row_slices(mf.cell, 3)) == 3
    seen = []
    for held in run_simulated(lambda comm: _distributed(mf, kpts, kplus,
                                                        every[comm.Get_rank()::3], work), 3):
        seen += list(held)
        for q, B in held.items():
            _same_factors(B, one[q], q)
    assert sorted(seen) == list(every)


def test_transfer_groups_and_irreducible_transfers_keep_the_pair_coulomb(si):
    mf, kpts, kplus, ref, work = si
    every = np.arange(len(kpts))
    for held in run_simulated(lambda comm: _distributed(mf, kpts, kplus,
                                                        every[comm.Get_rank()::8], work), 8):
        for q, B in held.items():
            assert np.abs(_pair_coulomb(B) - _pair_coulomb(ref[q])).max() < 1e-13
    irreducible = KMeshSymmetry(mf.cell, kpts).irreducible
    assert len(irreducible) < len(kpts)
    for held in run_simulated(lambda comm: _distributed(mf, kpts, kplus,
                                                        irreducible[comm.Get_rank()::2], work), 2):
        for q, B in held.items():
            assert np.abs(_pair_coulomb(B) - _pair_coulomb(ref[q])).max() < 1e-13
    assert os.listdir(work) == []


def test_rows_stay_on_the_node_of_their_owner(si):
    cell = si[0].cell
    spread = [list(range(r, 9, 6)) for r in range(6)]          # 9 transfers on 6 ranks
    everywhere = [list(range(9))] * 6                           # PBCDFIntegrals.from_scf
    for wanted, per_node in itertools.product((spread, everywhere), (1, 2, 3, 6)):
        tasks, builders = build_plan(cell, wanted, per_node)
        for owner, qs in enumerate(wanted):
            for q in qs:
                built = builders[owner // per_node, q]
                rows = sorted(rs for _, rs in built)
                assert all(r // per_node == owner // per_node for r, _ in built)
                assert rows[0][0] == 0 and rows[-1][1] == cell.nao_nr()
        for r, task in enumerate(tasks):
            if task is not None:
                block = range(r - r % per_node, r - r % per_node + per_node)
                assert set(task[1]) <= {q for b in block if b < 6 for q in wanted[b]}
                assert all((r, task[0]) in builders[r // per_node, q] for q in task[1])


def test_two_nodes_build_the_one_rank_factors(si):
    mf, kpts, kplus, _, work = si
    every = np.arange(len(kpts))
    one = _distributed(mf, kpts, kplus, every, work)

    def held(owned):
        return run_simulated(lambda comm: DistributedGDF(mf.with_df, work, ranks_per_node=3)(
            kpts, kplus, owned(comm.Get_rank())), 6)

    # round-robin, as the solvers own them: the first node owns every transfer;
    # and every transfer on every rank, as PBCDFIntegrals.from_scf asks
    for owned in (lambda r: every[r::6], lambda r: every):
        for B in held(owned):
            for q, b in B.items():
                _same_factors(b, one[q], q)
    # spread over both nodes, each building a subset in its own gauge
    seen = []
    for B in held(lambda r: every[every * 2 == r]):
        seen += list(B)
        for q, b in B.items():
            assert np.abs(_pair_coulomb(b) - _pair_coulomb(one[q])).max() < 1e-13
    assert sorted(seen) == list(every)
    assert os.listdir(work) == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
