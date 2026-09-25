"""The ISDF fit distributed over ranks returns the fit the serial pass returns.

`build_separable_ri` spends nearly all of its time in one loop: per shell-pair
block, one `aux_e2` call, one LU solve against the auxiliary metric, one
contraction into the (naux, nk) accumulator F D^T. The blocks are independent
and the loop ends in a sum, so a `comm` stripes them over ranks and reduces
that accumulator once. Everything else -- the Gram matrix, the Cholesky solve
-- is replicated, which is the whole contract of the keyword: a rank that
passes a comm gets the same factorization back as one that does not, and no
consumer of X, Z or M has to know which happened.

That contract has two halves and this file gates both, because either half
failing is silent. A factorization is a fit: it returns a plausible M whatever
it was built from, and a rank that dropped its blocks or added someone else's
twice produces a slightly worse fit, not an error. Every quasiparticle energy
and BSE root downstream then moves by an amount nobody can distinguish from
the grid.

  * RANKS AGREE WITH EACH OTHER, bitwise. The reduction is rank-ordered
    (`SimulatedComm.allreduce_sum`, and MPI_Allreduce with a fixed
    communicator), so this is exact, not approximate, and anything less means
    the ranks have silently forked into different functionals.
  * RANKS AGREE WITH SERIAL. Not bitwise, and it cannot be: the serial sum
    adds block 0, 1, 2, ... in order while r ranks add r partial sums whose
    terms were gathered in strides. Same terms, reordered additions.

WHAT THE REORDERING COSTS. On the accumulator itself it is rounding and
nothing more -- F D^T caught at the reduction differs from the serial one by
1.2e-16 relative on water/cc-pVDZ and at most 3.3e-16 on ethylene/cc-pVTZ at
2 and 3 ranks, against the 1e-13 the reduction was budgeted. The Cholesky
solve that follows amplifies it, because the balanced Gram matrix is
conditioned around 2e8, and that is what the gates below measure:

    water/cc-pVDZ     M  1.1e-09    Z  6.1e-09    D  2.9e-08
    ethylene/cc-pVTZ  M  1.9e-12    Z  8.9e-12    D  6.1e-11

Put that beside the discrepancy the repo already lives with between its own
two realizations of the same estimator, `fit_M_stable` (Cholesky) against
`fit_M_streaming`: on water, 1.2e-08 on M and 2.6e-08 on D, on ethylene
1.8e-06 on M. The rank split moves the fit by the same amount that choosing
between the two existing fits already moves it on water, and a million times
less than that on ethylene. So `FIT_TOL` is set an order above the worst of
these, and the gate is not a converged number to re-baseline but a ceiling: a
real defect in the partition or the reduction misses by whole digits, as the
perturbations below show.

The numbers are reproducible run to run, not a spread -- the stripe and the
rank order of the reduction are both fixed -- so a change in them is a change
in the arithmetic.

SHOWN TO FAIL. Each gate was broken once, on a copy of `separable_ri.py`
restored and `cmp`-verified afterwards:

  skip the reduction (`reduce_sum(FD, comm)` deleted) -- 10 of 14 fail. Every
      rank keeps only its own stripe, so the ranks disagree with each other
      and miss serial by M 4.8e+01 / 6.7e+01 on water at 2 / 3 ranks and
      1.7e+00 on ethylene. The 1-rank gate still passes, correctly: a size-1
      reduction is a no-op, so deleting it cannot show there.
  off-by-one on the comm path, rank 0 doing block 0 a second time -- 8 of 14
      fail, including the 1-rank gate, which is the one only this can reach:
      serial is untouched, so a duplicated block has to be caught against it.
      M misses by 2.7e+01 / 4.1e+01 on water and 4.1e-01 / 6.2e-01 on
      ethylene, D by 1.9e+02 on water. The ranks still AGREE with each other
      here -- they all duplicate the same block -- which is why agreeing
      across ranks and agreeing with serial are two gates and not one.
  off-by-one on both paths, block 0 done twice everywhere -- serial itself
      moves off the record taken before any of this was written: M by 1.4e+01
      on water and 2.1e-01 on ethylene, Z by 2.6e+03 and 1.0e+00. X stays
      bitwise under all three, being collocation the fit never touches.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df, gto, scf

from src.Base.separable_ri import (_ao_l_labels, ao_blocks, atomic_grid,
                                   build_separable_ri,
                                   molecular_points_covariant)
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import DEFAULT_COUNTS, separable_factors

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')

#: (geometry, basis, block_memory_gb). The memory budgets are far below what
#: either molecule needs, which is the point: they cut the AO index one shell
#: per block, so the pass has many more blocks than ranks and the stripe is
#: ragged (s against d and f shells). A budget that fits the whole index in
#: one block would test the distribution on a single block, i.e. not at all.
CASES = {'water': (WATER, 'cc-pvdz', 2e-4),
         'ethylene': (ETHYLENE, 'cc-pvtz', 1e-3)}

#: Ranks to simulate. 1 is the bitwise gate; 2 and 3 straddle the even split,
#: so 3 leaves the stripe uneven over both block counts.
RANKS = (2, 3)

#: Ceiling on the fit's disagreement with serial: one order above the worst
#: measured, and below the discrepancy between the repo's two fit
#: realizations on ethylene. Not a measurement to re-baseline.
FIT_TOL = 1e-7


def reldiff(a, b):
    """max |a - b| relative to the scale of a."""
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


def interpolation_points(mol, auxbasis):
    """A tabulated grid at `DEFAULT_COUNTS` (`atomic_grid`): the fit is gated
    on a fixed set of points, whatever `separable_factors` would choose."""
    radii, origins = {}, {}
    for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
        radii[el], origins[el] = atomic_grid(el, mol.basis, auxbasis,
                                             DEFAULT_COUNTS)
    return molecular_points_covariant(mol, radii, origin_by_element=origins)


@pytest.fixture(scope='module', params=sorted(CASES))
def case(request):
    """(name, mol, auxmol, coords, block_memory_gb) for one molecule."""
    atom, basis, block_memory_gb = CASES[request.param]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    auxbasis = basis + '-ri'
    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    return dict(name=request.param, mol=mol, auxbasis=auxbasis, auxmol=auxmol,
                coords=interpolation_points(mol, auxbasis),
                block_memory_gb=block_memory_gb)


@pytest.fixture(scope='module')
def serial(case):
    """(X, Z, M) with no comm: the reference every rank count is held to."""
    return build_separable_ri(case['mol'], case['coords'],
                              auxmol=case['auxmol'],
                              block_memory_gb=case['block_memory_gb'])


def distributed(case, size):
    """(X, Z, M) per rank of a `size`-rank simulated world."""
    def build(comm):
        return build_separable_ri(case['mol'], case['coords'],
                                  auxmol=case['auxmol'],
                                  block_memory_gb=case['block_memory_gb'],
                                  comm=comm)
    return run_simulated(build, size)


def test_the_pass_is_many_blocks(case):
    """The stripe is only a stripe if there are blocks to spread."""
    mol, auxmol = case['mol'], case['auxmol']
    n2 = int((_ao_l_labels(mol) <= 2).sum())
    blocks = ao_blocks(mol, len(case['coords']), n2, auxmol.nao_nr(),
                       case['block_memory_gb'])
    widths = [mol.ao_loc_nr()[s1] - mol.ao_loc_nr()[s0] for s0, s1 in blocks]
    print(f"\n{case['name']}: {len(blocks)} blocks of {min(widths)}-{max(widths)} "
          f"AOs at block_memory_gb={case['block_memory_gb']:g} "
          f"(nbas={mol.nbas}, nk={len(case['coords'])}, naux={auxmol.nao_nr()})")
    assert len(blocks) >= 8
    assert len(blocks) > max(RANKS)


def test_one_rank_is_the_serial_factorization(case, serial):
    """A comm of size 1 owns every block and reduces nothing, so the arithmetic
    is the serial arithmetic -- no tolerance belongs here."""
    X0, Z0, M0 = serial
    (X, Z, M), = distributed(case, 1)
    assert np.array_equal(X, X0)
    assert np.array_equal(Z, Z0)
    assert np.array_equal(M, M0)


@pytest.mark.parametrize('size', RANKS)
def test_every_rank_returns_the_same_arrays(case, size):
    """Bitwise across ranks: the reduction is rank-ordered, so there is no
    reason for two ranks to hold different factors, and every reason to catch
    it if they do -- they would be running different functionals in lockstep."""
    per_rank = distributed(case, size)
    for r, (X, Z, M) in enumerate(per_rank[1:], start=1):
        assert np.array_equal(X, per_rank[0][0]), f'X differs on rank {r}'
        assert np.array_equal(Z, per_rank[0][1]), f'Z differs on rank {r}'
        assert np.array_equal(M, per_rank[0][2]), f'M differs on rank {r}'


@pytest.mark.parametrize('size', RANKS)
def test_ranks_reproduce_the_serial_fit(case, size, serial):
    """X is collocation, untouched by the split, so it stays bitwise. Z and M
    come through the reduced accumulator and the Cholesky solve."""
    X0, Z0, M0 = serial
    X, Z, M = distributed(case, size)[0]
    dz, dm = reldiff(Z0, Z), reldiff(M0, M)
    print(f"\n{case['name']}, {size} ranks: M {dm:.2e}, Z {dz:.2e} relative")
    assert np.array_equal(X, X0)
    assert dm < FIT_TOL
    assert dz < FIT_TOL


def test_separable_factors_passes_the_comm_through(case):
    """The production entry point. It hands the comm to the fit and nothing
    else of it changes, so X_mo, X_ao and the grid stay bitwise and only D,
    which carries M, moves by the reordering."""
    mol, auxbasis = case['mol'], case['auxbasis']
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    kwargs = dict(auxbasis=auxbasis, block_memory_gb=case['block_memory_gb'])
    ser = separable_factors(mf, mol, **kwargs)
    per_rank = run_simulated(lambda comm: separable_factors(mf, mol, comm=comm,
                                                            **kwargs), 3)
    for j, name in enumerate(('X_mo', 'D', 'X_ao', 'coords')):
        for r in range(1, 3):
            assert np.array_equal(per_rank[0][j], per_rank[r][j]), \
                f'{name} differs between ranks 0 and {r}'
    for j, name in ((0, 'X_mo'), (2, 'X_ao'), (3, 'coords')):
        assert np.array_equal(ser[j], per_rank[0][j]), f'{name} moved'
    dd = reldiff(ser[1], per_rank[0][1])
    print(f"\n{case['name']}, separable_factors on 3 ranks: D {dd:.2e} relative")
    assert dd < FIT_TOL


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
