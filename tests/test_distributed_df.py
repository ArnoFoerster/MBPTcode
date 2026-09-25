"""The distributed Fock build reproduces the serial mean field, and divides
the fitted tensor and the quadrature grid.

`mpi_grid.simulated_world` runs each rank in a thread of this process and
broadcasts and reduces through shared memory, so the whole sequence runs here
-- pyscf's SCF on every rank, the density and its orbitals locked at every J/K
build and every quadrature, the reductions of (J, K) and of (nelec, E_xc,
V_xc), the loop's exit settings and the converged mean field locked -- where
MPI itself cannot start. What it does not test is the wire protocol.

Two and three ranks: 116 and 278 auxiliary functions divide evenly by neither,
so both gates carry a rank whose block is one row short, and the slices have to
tile their index rather than merely cover it.

WHAT IS BITWISE AND WHAT IS NOT. The rows a rank builds are, against the same
column blocks walked by one rank: the metric factor is rank 0's, replicated,
every AO-pair column is evaluated by exactly one rank and travels verbatim, so
the slices reassemble with no difference at all -- that is
`test_slices_tile_the_auxiliary_index`, and it is what says the split is of
ONE tensor. Against pyscf's DEFAULT blocking they differ by the last bit of
the smallest elements, which is a property of how many columns go into one
`solve_triangular` and not of the split: pyscf's own builder differs from
itself by the same 2.5e-17 between two `max_memory` settings. The grid points
are bitwise outright: they are rank 0's own, broadcast and sliced, so the
weights a rank integrates concatenate back to rank 0's grid bit for bit. What
the ranks REDUCE is not: the sums over the auxiliary index and
over the grid points are re-associated at the rank boundaries, so the Fock
matrix moves in its last bits and the SCF stops at a mean field that differs by
the convergence tolerance. Every rank takes every decision -- DIIS, the
occupation, when it has converged -- from rank 0's density and orbitals,
locked into its own arrays at every Fock piece, and the grid is rank 0's, so
the ranks cannot diverge further than that, and after the final lockstep they
hold the same orbitals bit for bit. Without a communicator, or on a world of
one, nothing here runs at all and the mean field is pyscf's own, bit for bit
(`test_one_rank_world_is_pyscf`).

WHAT SPANS TWO SCF RUNS. A distributed mean field against the serial one is
two SCF runs, and a threaded pyscf, whose OpenMP GEMM adds its K-split
partials in thread-arrival order, does not repeat one bit for bit. Those
comparisons are gated on an anchored bar (`scf_bars`): `COMPOSED_GRAD_K`
times what running the serial SCF again moves the energy, the density and the
orbital energies, one ulp at the least, floored at `E_TOL`, `DM_TOL` and
`MO_TOL`; where two distributed runs of one layout are compared, within
`COMPOSED_GRAD_K` times the repeat alone, which is bitwise here, where pyscf
repeats its bits. The checks that the serial path IS pyscf's own SCF are `==`
on two runs: laptop gates by design, met where pyscf repeats its bits.

EVERY GATE HERE WAS SHOWN TO FAIL:

  what is broken                         what then fails
  one rank's partial K scaled by         the energy gate, 2.2e-8 Ha out
  1 + 1e-6                               against a tolerance of 1e-10
  one rank's partial V_xc scaled by      the DENSITY gate, 2.2e-7 out, and the
  1 + 1e-6                               orbital energies, 1.9e-7 Ha -- not the
                                         energy, which is stationary in the
                                         density and reads E_xc, not V_xc
  one rank's partial E_xc scaled by      the energy gate, 3.0e-6 Ha out
  1 + 1e-6
  the reduction of J and K made a        every gate: each rank iterates on its
  no-op (the collective still called)    own rows' J and K, the ranks stop at
                                         different cycles and the next
                                         lockstep refuses on every rank
  the reduction of the quadrature made   every gate: the ranks fall out of step
  a no-op                                the same way and the run raises
  the row split made a no-op, every      the tiling gate on the blocks and on
  rank keeping the whole tensor          the bytes, and the energy by 28 Ha
  the point split made a no-op, every    the tiling gate on the point counts,
  rank integrating the whole grid        and the PBE0 energy by 7.2 Ha
  one AO-pair column block skipped by    the reassembly gate, and the energy
  the rank that owns it                  by 47.3 Ha with the SCF still
                                         reporting itself converged
  the density's lockstep taken out of    the repair gate: rank 1 keeps the
  `get_jk`, rank 1's density moved one   density it was handed, its DIIS
  ulp before one J/K build               extrapolates another Fock matrix from
                                         the next cycle on, and the audit
                                         counts no repair
  rank 1 given conv_tol 1e-2 and         nothing -- which is the point:
  max_cycle 1                            `test_worker_takes_no_decision` says
                                         the exit is rank 0's; with the exit
                                         settings' lockstep taken out, rank 1
                                         stops early and the next lockstep
                                         refuses on every rank
  the refusal of a with_df that is not   the interpolated-exchange gate: both
  pyscf's DF taken out                   ranks converge an ISDFJK mean field
                                         on plain density-fitted rows and
                                         report nothing

THE TIMERS read the clock at stage boundaries and are gated the way every
other instrumented route in this repo is: every key present on every rank,
the stages inside the total, the cycle count pyscf's own, and the dict
JSON-clean for a caller that records it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.df import addons
from pyscf.dft.gen_grid import ALIGNMENT_UNIT

from src.Base import distributed_df
from src.Base.constants import COMPOSED_GRAD_K
from src.Base.distributed_df import (TIMING_KEYS, DistributedDF,
                                     DistributedNumInt, distributed_df_jk,
                                     distributed_df_storage,
                                     distributed_handles,
                                     distributed_mean_field,
                                     distributed_numint)
from src.Base.isdf_jk import isdf_jk
from src.Base.utils.mpi_grid import (contiguous_block, current_comm,
                                     distributed, lockstep_stats,
                                     run_simulated)

SIZES = [2, 3]
#: The widest world the gates run: more ranks than water's density fit has
#: rows for evenly (116 = 8 x 14 + 4), and eight threads of one process.
WIDE = 8
#: Hartree, the least of the bars against the serial SCF (`scf_bars`). The
#: re-associated auxiliary sum moves the Fock matrix in its last bits and the
#: SCF stops one tolerance away from where the serial one did; measured 3e-13
#: at worst over these four cases.
E_TOL = 1e-10
DM_TOL = 1e-8
MO_TOL = 1e-8
#: Tight enough that the SCF stops on the Fock matrix rather than on its own
#: cycle count, which is what makes the comparison a comparison of solutions.
CONV_TOL = 1e-12
#: pyscf's default is sqrt(conv_tol) = 1e-6, which stops the iteration with a
#: density good to ~1e-8 -- the density gate itself. Two runs whose fitted
#: tensors differ in the last bit then stop a cycle apart and differ by that
#: floor rather than by anything the split did: ethylene/cc-pVTZ measures
#: 1.7e-8 at the default and 2.6e-12 here, on the same solution.
CONV_TOL_GRAD = 1e-9
#: Relative scaling of one rank's partial. Small enough to be a rounding-sized
#: defect and large enough that the energy gate catches it.
PERTURBATION = 1e-6
#: Which of rank 1's J/K builds is handed a density one ulp off rank 0's: far
#: enough into the iteration that DIIS holds several vectors, so a rank that
#: kept its own density would extrapolate another Fock matrix from there on.
PERTURBED_BUILD = 4
#: How far the fitted tensor may sit from pyscf's own default-blocked one.
#: Not the split's error: `incore.cholesky_eri` differs from ITSELF by 2.5e-17
#: on ethylene/cc-pVTZ between max_memory 4000 and 20, because the number of
#: columns in one `solve_triangular` decides how LAPACK blocks the solve.
CDERI_BLOCKING_TOL = 1e-15
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
CASES = [('water', WATER, 'cc-pvdz', None),
         ('water', WATER, 'cc-pvdz', 'pbe0'),
         ('ethylene', ETHYLENE, 'cc-pvtz', None),
         ('ethylene', ETHYLENE, 'cc-pvtz', 'pbe0')]

#: The stage keys that must sum to no more than `scf_total`.
STAGE_KEYS = ('scf_build_slices', 'scf_guess', 'scf_grid', 'scf_fock_jk',
              'scf_fock_xc', 'scf_reduce', 'scf_lockstep', 'scf_driver')
#: s. The clock is read at stage boundaries only, so the stages fall short of
#: the whole by the reads between them rather than exceeding it; the slack is
#: for the other direction, where a stage ends after the total was started.
TIMER_SLACK = 1e-3


def mean_field(atom, basis, xc, charge=0, spin=0):
    """An unconverged density-fitted mean field, the same on every rank.

    No auxbasis: pyscf's own choice from the orbital basis and the functional,
    which is the JK-fitting set for both a Hartree-Fock and a hybrid Fock
    build -- an -ri set is fitted for the correlation kernel and is the wrong
    basis for J and K.
    """
    mol = gto.M(atom=atom, basis=basis, verbose=0, charge=charge, spin=spin)
    if xc is None:
        mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc) if spin == 0 else dft.UKS(mol, xc=xc)
    mf = mf.density_fit()
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    return mf


def scf_bars(mf, again):
    """The bars a comparison with the converged `mf` is gated at when it spans
    two SCF runs: `COMPOSED_GRAD_K` times what `again`, the same SCF run a
    second time, moved the energy, the density and the orbital energies (one
    ulp at the least), floored at `E_TOL`, `DM_TOL` and `MO_TOL`; `repeat` is
    what the repeat moved the energy."""
    def bar(floor, got, ref):
        got, ref = np.asarray(got), np.asarray(ref)
        return max(floor, COMPOSED_GRAD_K * max(
            np.abs(got - ref).max(), np.spacing(np.abs(ref).max())))

    return dict(e=bar(E_TOL, again.e_tot, mf.e_tot),
                dm=bar(DM_TOL, again.make_rdm1(), mf.make_rdm1()),
                eps=bar(MO_TOL, again.mo_energy, mf.mo_energy),
                repeat=abs(again.e_tot - mf.e_tot))


@pytest.fixture(scope='module')
def serial():
    """The serial mean field of every case, and the bars a comparison with it
    stands at, read off the same SCF converged a second time."""
    out = {}
    for name, atom, basis, xc in CASES:
        mf = mean_field(atom, basis, xc)
        mf.kernel()
        again = mean_field(atom, basis, xc)
        again.kernel()
        out[(name, xc)] = dict(e=mf.e_tot, dm=mf.make_rdm1(),
                               eps=np.asarray(mf.mo_energy), mf=mf,
                               bars=scf_bars(mf, again))
    return out


def on_ranks(fn, size, atom, basis, xc, charge=0, spin=0):
    """`fn(comm, mf)` on every rank of a simulated world, each rank with its
    own mean field object -- built here rather than in the threads, because
    under MPI they are separate processes and the point is that they built the
    same thing, not that they built it at the same time."""
    mfs = [mean_field(atom, basis, xc, charge, spin) for _ in range(size)]
    return run_simulated(lambda comm: fn(comm, mfs[comm.Get_rank()]), size)


def converge(comm, mf, **kwargs):
    """One rank's share of a distributed SCF, and what it ended holding."""
    distributed_mean_field(mf, comm, **kwargs)
    return dict(e=mf.e_tot, dm=np.asarray(mf.make_rdm1()),
                eps=np.asarray(mf.mo_energy), mo=np.asarray(mf.mo_coeff),
                occ=np.asarray(mf.mo_occ), converged=mf.converged,
                with_df=type(mf.with_df).__name__,
                numint=type(getattr(mf, '_numint', None)).__name__,
                handles=distributed_handles(mf, comm))


def converge_with_grid(comm, mf):
    """A distributed SCF, plus the grid rank 0 ended up integrating over and
    whether this rank built a grid of its own."""
    out = converge(comm, mf)
    out['weights'] = (np.asarray(mf.grids.weights) if comm.Get_rank() == 0
                      else None)
    out['built_grid'] = mf.grids.coords is not None
    return out


def integrated(monkeypatch):
    """Record the points each rank actually ran the quadrature on.

    Read from inside `partial_xc`, on the grid object the quadrature is handed,
    because the slice is freed when the SCF that needed it ends -- so this is
    what a rank integrated, not what it says it integrated.
    """
    seen, partial_xc = {}, DistributedNumInt.partial_xc

    def spy(self, *args, **kwargs):
        seen.setdefault(self.comm.Get_rank(),
                        (self.point_slice, self._grids.weights.copy()))
        return partial_xc(self, *args, **kwargs)

    monkeypatch.setattr(DistributedNumInt, 'partial_xc', spy)
    return seen


def assert_one_mean_field(out, ref):
    """Rank 0's mean field is the serial one within the anchored bars of two
    SCF runs, and every rank holds rank 0's, bit for bit."""
    bars = ref['bars']
    assert all(r['converged'] for r in out)
    assert abs(out[0]['e'] - ref['e']) <= bars['e']
    assert np.abs(out[0]['dm'] - ref['dm']).max() <= bars['dm']
    assert np.abs(out[0]['eps'] - ref['eps']).max() <= bars['eps']
    for r in out:
        assert r['e'] == out[0]['e']
        assert np.array_equal(r['dm'], out[0]['dm'])
        assert np.array_equal(r['mo'], out[0]['mo'])
        assert np.array_equal(r['eps'], out[0]['eps'])
        assert np.array_equal(r['occ'], out[0]['occ'])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name,atom,basis,xc', CASES)
def test_distributed_scf_matches_serial(serial, size, name, atom, basis, xc):
    """The mean field is the serial one, and the ranks hold one copy of it."""
    ref = serial[(name, xc)]
    out = on_ranks(converge, size, atom, basis, xc)
    assert_one_mean_field(out, ref)
    for r in out:
        # Both slices are freed with the SCF that needed them: a caller that
        # iterates mf.with_df.loop() or asks mf._numint for a response kernel
        # afterwards must meet a whole tensor and a whole grid, not one rank's
        # block of either.
        assert r['with_df'] == 'DF'
        assert r['numint'] == ('NumInt' if xc else 'NoneType')
        # ...and reachable, holding this rank's slice for the next stage.
        df_handle, ni_handle, _ = r['handles']
        assert df_handle._cderi is not None
        assert (ni_handle is not None) == bool(xc)


@pytest.mark.parametrize('xc', [None, 'pbe0'])
def test_eight_ranks_hold_one_mean_field(serial, xc):
    """Eight ranks, each running pyscf's loop: still one mean field, and still
    the serial one to the reduction's last bits."""
    assert_one_mean_field(on_ranks(converge, WIDE, WATER, 'cc-pvdz', xc),
                          serial[('water', xc)])


def slice_record(comm, mf):
    """Build this rank's rows, report them, and take part in nothing else."""
    dist = distributed_df_jk(mf, comm)
    out = dict(rows=dist.row_slice, nbytes=dist._cderi.nbytes,
               cderi=dist._cderi.copy(), blocks=dist.column_blocks,
               all=dist._blocks)
    dist.release()
    # The reporter answers after the slice is gone, which is where the probe
    # reads it: what a rank held, not what the mean field's own DF would say.
    out['reported'] = distributed_df_storage(mf, comm)
    return out


def serial_column_walk(mf, blocks):
    """The whole cderi, one rank walking the column blocks the ranks divided.

    THE REFERENCE THE SPLIT IS EXACT AGAINST. pyscf's own serial tensor is not
    bit-stable in its own `max_memory`: the column count handed to one
    `solve_triangular` decides how LAPACK blocks the solve, and
    `incore.cholesky_eri` on ethylene/cc-pVTZ at max_memory 4000 against 20
    differs by 2.5e-17 on elements of order 1. So what says the split is exact
    is the same block list walked by one rank, and what says the blocking is
    immaterial is that its distance to pyscf's default is that same last bit.
    """
    auxmol = addons.make_auxmol(mf.mol, mf.with_df.auxbasis)
    low, mode = distributed_df._metric_factor(mf.mol, auxmol)
    env = distributed_df._int3c_env(mf.mol, auxmol)
    widest = max(width for _, _, width, _ in blocks)
    bufs = [np.empty((widest, low.shape[1])), np.empty((widest, low.shape[1]))]
    # A block aliases the integral buffer the next one overwrites, so it is
    # copied out here exactly as the distributed build consumes it in place.
    return np.hstack([np.array(distributed_df._solve_column_block(
        env, low, mode, b, bufs)) for b in blocks])


def test_slices_tile_the_auxiliary_index(serial):
    """Each rank holds ceil(naux/nranks) rows at most, the blocks tile the
    auxiliary index, and the resident bytes fall with the rank count.

    The reassembled slices are BITWISE the same block list walked serially:
    every column is evaluated by exactly one rank and travels verbatim, so
    anything less than exact would mean the ranks are holding rows of
    different tensors. That reference shares this module's own block kernel,
    so it is the exactness of the SPLIT it pins and not the correctness of
    the kernel; what pins the kernel is the second comparison, against
    pyscf's own `DF.build` tensor, which is a different implementation and
    differs only by the blocking's last bit -- see `serial_column_walk`.
    """
    for name, atom, basis, xc in (CASES[0], CASES[2]):
        ref = serial[(name, xc)]
        ref['mf'].with_df.build()
        whole = np.asarray(ref['mf'].with_df._cderi)
        naux, nao_pair = whole.shape
        largest = {}
        for size in SIZES:
            out = on_ranks(slice_record, size, atom, basis, xc)
            blocks = [r['rows'] for r in out]
            assert blocks == [contiguous_block(naux, r, size)
                              for r in range(size)]
            assert blocks[0][0] == 0 and blocks[-1][1] == naux
            assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
            rebuilt = np.concatenate([r['cderi'] for r in out], axis=0)
            assert np.array_equal(rebuilt,
                                  serial_column_walk(ref['mf'], out[0]['all']))
            assert np.abs(rebuilt - whole).max() <= CDERI_BLOCKING_TOL
            rows = -(-naux // size)             # ceil
            assert max(b[1] - b[0] for b in blocks) == rows
            assert [r['nbytes'] for r in out] == [(b[1] - b[0]) * nao_pair * 8
                                                  for b in blocks]
            assert sum(r['nbytes'] for r in out) == whole.nbytes
            largest[size] = max(r['nbytes'] for r in out)
            assert largest[size] == rows * nao_pair * 8
            for r in out:
                assert r['reported']['rows'] == r['rows'][1] - r['rows'][0]
                assert r['reported']['slice_gb'] == r['nbytes'] / 1e9
                assert r['reported']['whole_gb'] == whole.nbytes / 1e9
        assert whole.nbytes > largest[2] > largest[3]


def test_column_blocks_tile_and_divide(serial):
    """Every AO-pair column block is evaluated by exactly one rank, and how
    many each evaluates falls with the rank count.

    The integral pass is the half of the build that used to run whole on every
    rank; what divides it is this partition, and what keeps the tensor one
    tensor is that the parts tile pyscf's own block list.
    """
    for name, atom, basis, xc in (CASES[0], CASES[2]):
        counts = {}
        for size in SIZES:
            out = on_ranks(slice_record, size, atom, basis, xc)
            mine = [list(r['blocks']) for r in out]
            total = len(out[0]['all'])
            assert sorted(sum(mine, [])) == list(range(total))
            assert all(len(m) > 0 for m in mine)
            assert max(len(m) for m in mine) == -(-total // size)
            counts[size] = max(len(m) for m in mine)
        assert counts[2] > counts[3]


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name,atom,basis,xc',
                         [c for c in CASES if c[3] is not None])
def test_points_tile_rank_zero_grid(monkeypatch, size, name, atom, basis, xc):
    """The blocks a rank integrates tile rank 0's grid, bit for bit.

    The grid is rank 0's -- pyscf's own build, pyscf's own pruning against the
    first density -- and it is broadcast, not rebuilt, so the weights the ranks
    integrate concatenate back to exactly the array rank 0 holds. The blocks
    are whole multiples of ALIGNMENT_UNIT, which is what keeps every rank on
    the sparse AO kernels the serial run uses. No other rank builds a grid of
    its own: its `initialize_grids` builds nothing while the handle is in.
    """
    seen = integrated(monkeypatch)
    out = on_ranks(converge_with_grid, size, atom, basis, xc)
    assert all(r['converged'] for r in out)
    assert [r['built_grid'] for r in out] == [True] + [False] * (size - 1)
    whole = out[0]['weights']
    blocks = [seen[r][0] for r in range(size)]
    weights = [seen[r][1] for r in range(size)]
    assert blocks == [tuple(x * ALIGNMENT_UNIT for x in
                            contiguous_block(whole.size // ALIGNMENT_UNIT,
                                             r, size))
                      for r in range(size)]
    assert blocks[0][0] == 0 and blocks[-1][1] == whole.size
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
    assert sum(w.size for w in weights) == whole.size
    assert all(w.size % ALIGNMENT_UNIT == 0 for w in weights)
    assert np.array_equal(np.concatenate(weights), whole)


def perturb_grid_partial(monkeypatch, which):
    """Scale rank 1's share of one of the three reduced quadrature terms."""
    partial_xc = DistributedNumInt.partial_xc

    def perturbed(self, *args, **kwargs):
        out = list(partial_xc(self, *args, **kwargs))
        if self.comm.Get_rank() == 1:
            out[which] = out[which] * (1 + PERTURBATION)
        return tuple(out)

    monkeypatch.setattr(DistributedNumInt, 'partial_xc', perturbed)


def test_perturbed_grid_partial_moves_the_mean_field(serial, monkeypatch):
    """One rank's V_xc scaled by 1 + 1e-6 fails the density and orbital gates.

    THE ENERGY DOES NOT CATCH THIS ONE, and that is a property of the
    functional, not of the split: `rks.energy_elec` builds E from `vhf.exc`,
    the quadrature's ENERGY, and never from its potential, and E is stationary
    in the density -- so a first-order defect in V_xc costs second order in the
    energy, measured 1.8e-13 Ha against 2.2e-7 in the density and 1.9e-7 Ha in
    the orbital energies. The gate on the potential is therefore the density,
    and the energy gate below covers the other half of the same reduction.
    """
    ref = serial[('water', 'pbe0')]
    perturb_grid_partial(monkeypatch, 2)
    out = on_ranks(converge, 2, WATER, 'cc-pvdz', 'pbe0')
    assert np.abs(out[0]['dm'] - ref['dm']).max() > ref['bars']['dm']
    assert np.abs(out[0]['eps'] - ref['eps']).max() > ref['bars']['eps']


def test_perturbed_grid_energy_moves_the_energy(serial, monkeypatch):
    """One rank's E_xc scaled by 1 + 1e-6 fails the energy gate, 3.0e-6 Ha out.

    The half of the reduced triple that the total energy does read. Its
    partner `nelec` moves nothing at all -- pyscf only logs the integrated
    electron count -- so of the three terms every rank reduces, one is checked
    here, one in the test above, and the third is a diagnostic.
    """
    ref = serial[('water', 'pbe0')]
    perturb_grid_partial(monkeypatch, 1)
    out = on_ranks(converge, 2, WATER, 'cc-pvdz', 'pbe0')
    assert abs(out[0]['e'] - ref['e']) > ref['bars']['e']


def test_unrestricted_matches_serial():
    """A spin-polarized mean field takes the same two splits.

    `nr_uks` sums the same grid points and `df_jk` the same auxiliary rows,
    with one density matrix per spin, so both partials add exactly as the
    closed-shell ones do.
    """
    mf = mean_field(WATER, 'cc-pvdz', 'pbe0', charge=1, spin=1)
    mf.kernel()
    again = mean_field(WATER, 'cc-pvdz', 'pbe0', charge=1, spin=1)
    again.kernel()
    bars = scf_bars(mf, again)
    out = on_ranks(converge, 2, WATER, 'cc-pvdz', 'pbe0', charge=1, spin=1)
    assert all(r['converged'] for r in out)
    assert abs(out[0]['e'] - mf.e_tot) <= bars['e']
    assert np.abs(out[0]['dm'] - np.asarray(mf.make_rdm1())).max() <= bars['dm']
    assert np.abs(out[0]['eps'] - np.asarray(mf.mo_energy)).max() <= bars['eps']
    assert np.array_equal(out[1]['mo'], out[0]['mo'])


def test_split_grid_off_leaves_the_quadrature_whole(serial, monkeypatch):
    """`split_grid=False` divides the fit alone: rank 0 integrates its whole
    grid, the others no point, and rank 0's quadrature reaches them through
    the reduction as exact zeros added to it.

    Bitwise against the J/K-only handle installed by hand -- the same mean
    field through `distributed_df_jk` with pyscf's own numint, every rank
    integrating its own whole grid -- so rank 0's quadrature is pyscf's own
    call on pyscf's own grid, and the mean field keeps pyscf's own numint
    object, not a restored copy of it.
    """
    seen = integrated(monkeypatch)
    mfs = [mean_field(WATER, 'cc-pvdz', 'pbe0') for _ in range(2)]
    numints = [mf._numint for mf in mfs]
    flagged = run_simulated(
        lambda comm: converge(comm, mfs[comm.Get_rank()], split_grid=False), 2)
    whole = mfs[0].grids.weights.size
    assert [seen[r][0] for r in range(2)] == [(0, whole), (whole, whole)]
    assert np.array_equal(seen[0][1], mfs[0].grids.weights)
    assert seen[1][1].size == 0
    assert mfs[1].grids.coords is None
    assert all(mf._numint is ni for mf, ni in zip(mfs, numints))

    def jk_only(comm, mf):
        dist = distributed_df_jk(mf, comm)
        mf.kernel()
        dist.release()
        return mf.e_tot

    by_hand = on_ranks(jk_only, 2, WATER, 'cc-pvdz', 'pbe0')
    ref = serial[('water', 'pbe0')]
    # two runs of one layout: within the repeat alone, bitwise where it is
    assert (abs(flagged[0]['e'] - by_hand[0])
            <= COMPOSED_GRAD_K * ref['bars']['repeat'])
    assert abs(flagged[0]['e'] - ref['e']) <= ref['bars']['e']


def test_a_skipped_column_block_breaks_the_tensor(serial, monkeypatch):
    """A rank that does not evaluate one of its blocks leaves those columns
    unwritten, and nothing downstream would say so on its own.

    The check that the column partition is load-bearing: the exchange places
    what it is given, so a block nobody computed is a hole in the tensor, and
    the SCF still converges -- 47.3 Ha out, and calling itself converged.
    """
    ref = serial[('water', None)]
    first = on_ranks(slice_record, 2, WATER, 'cc-pvdz', None)[0]
    skipped = first['all'][1]              # rank 1's first block, at any size
    reference = serial_column_walk(ref['mf'], first['all'])
    solve = distributed_df._solve_column_block

    def hole(env, low, mode, block, bufs):
        out = solve(env, low, mode, block, bufs)
        return np.zeros_like(out) if block[:2] == skipped[:2] else out

    monkeypatch.setattr(distributed_df, '_solve_column_block', hole)
    out = on_ranks(slice_record, 2, WATER, 'cc-pvdz', None)
    rebuilt = np.concatenate([r['cderi'] for r in out], axis=0)
    assert not np.array_equal(rebuilt, reference)
    holed = on_ranks(converge, 2, WATER, 'cc-pvdz', None)
    assert abs(holed[0]['e'] - ref['e']) > ref['bars']['e']


def test_perturbed_partial_moves_the_energy(serial, monkeypatch):
    """One rank's exchange partial scaled by 1 + 1e-6 fails the energy gate.

    The check that says the reduction is load-bearing: the workers' rows reach
    the Fock matrix rank 0 iterates on, so a defect in one rank's partial is a
    defect in the answer rather than something the SCF absorbs.
    """
    ref = serial[('water', None)]
    partial_jk = DistributedDF.partial_jk

    def perturbed(self, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
        vj, vk = partial_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        if self.comm.Get_rank() == 1 and vk is not None:
            vk = vk * (1 + PERTURBATION)
        return vj, vk

    monkeypatch.setattr(distributed_df.DistributedDF, 'partial_jk', perturbed)
    out = on_ranks(converge, 2, WATER, 'cc-pvdz', None)
    assert abs(out[0]['e'] - ref['e']) > ref['bars']['e']


def test_worker_takes_no_decision(serial):
    """A worker handed a broken SCF driver changes nothing.

    Rank 1 gets conv_tol 1e-2 and one cycle -- settings that on its own would
    give a mean field nowhere near converged, and a loop that leaves while
    rank 0 is still in it -- and the answer is still rank 0's, because the
    loop's exit settings are locked to rank 0's before it starts and every
    other decision is taken from rank 0's density.
    """
    ref = serial[('water', None)]
    mfs = [mean_field(WATER, 'cc-pvdz', None) for _ in range(2)]
    mfs[1].conv_tol, mfs[1].max_cycle = 1e-2, 1
    out = run_simulated(lambda comm: converge(comm, mfs[comm.Get_rank()]), 2)
    assert out[1]['e'] == out[0]['e']
    assert abs(out[1]['e'] - ref['e']) <= ref['bars']['e']
    assert np.array_equal(out[1]['mo'], out[0]['mo'])
    assert (mfs[1].conv_tol, mfs[1].max_cycle) == (mfs[0].conv_tol,
                                                   mfs[0].max_cycle)


def test_a_perturbed_density_is_repaired_at_entry(serial, monkeypatch):
    """Rank 1's density moved one ulp per element before one J/K build is
    rank 0's again when the build starts.

    What stands in for a node whose eigensolver differed in a last bit. The
    lockstep writes rank 0's density into rank 1's own array, the one pyscf's
    loop keeps, so every Fock matrix pyscf then diagonalizes -- the DIIS
    extrapolation included, which reads the density through its error vector
    -- holds the same bits on every rank, and the audit counts exactly the
    one repair. Hartree-Fock, so the J/K build is the only lockstep in a cycle
    and nothing after it can repair what it missed.
    """
    get_jk, eig = DistributedDF.get_jk, scf.hf.SCF.eig
    builds, focks = {}, {}

    def perturbed(self, dm, *args, **kwargs):
        rank = self.comm.Get_rank()
        builds[rank] = builds.get(rank, 0) + 1
        if rank == 1 and builds[rank] == PERTURBED_BUILD:
            d = np.asarray(dm)
            sign = np.random.default_rng(0).choice([-1.0, 1.0], d.shape)
            d += sign * np.spacing(np.abs(d))
        return get_jk(self, dm, *args, **kwargs)

    def recorded(self, h, s):
        comm = current_comm()
        if comm is not None:
            focks.setdefault(comm.Get_rank(), []).append(np.array(h))
        return eig(self, h, s)

    def audited(comm, mf):
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            out = converge(comm, mf)
            out['stats'] = lockstep_stats()
        return out

    ref = serial[('water', None)]
    monkeypatch.setattr(DistributedDF, 'get_jk', perturbed)
    monkeypatch.setattr(scf.hf.SCF, 'eig', recorded)
    out = on_ranks(audited, 2, WATER, 'cc-pvdz', None)
    assert builds[1] > PERTURBED_BUILD
    assert len(focks[0]) == len(focks[1]) > PERTURBED_BUILD
    assert all(np.array_equal(a, b) for a, b in zip(focks[0], focks[1]))
    assert_one_mean_field(out, ref)
    assert out[0]['stats']['mismatched_calls'] == 0
    assert out[1]['stats']['mismatched_calls'] == 1
    assert 0.0 < out[1]['stats']['max_abs_diff'] < 1e-15
    assert all(r['stats']['audited_calls'] == r['stats']['calls']
               > PERTURBED_BUILD for r in out)


def test_the_context_is_the_communicator(serial):
    """Inside `distributed(comm)` the SCF finds its communicator unasked, and
    inside `distributed(None)` it is pyscf's own `kernel()`.

    The first is how a job script hands the ranks to every kernel at once;
    the second is how a stretch of it that every rank runs on its own --
    `mpi_map`'s items -- stays serial.
    """
    explicit = on_ranks(converge, 2, WATER, 'cc-pvdz', 'pbe0')

    def implicit(comm, mf):
        distributed_mean_field(mf)
        return dict(e=mf.e_tot, mo=np.asarray(mf.mo_coeff),
                    handles=distributed_handles(mf) is not None,
                    timed=hasattr(mf, '_distributed_timings'))

    def serial_region(comm, mf):
        with distributed(None):
            distributed_mean_field(mf)
        return dict(e=mf.e_tot, handles=hasattr(mf, '_distributed'),
                    timed=hasattr(mf, '_distributed_timings'))

    for mine, theirs in zip(on_ranks(implicit, 2, WATER, 'cc-pvdz', 'pbe0'),
                            explicit):
        assert mine['e'] == theirs['e']
        assert np.array_equal(mine['mo'], theirs['mo'])
        assert mine['handles'] and mine['timed']
    for r in on_ranks(serial_region, 2, WATER, 'cc-pvdz', 'pbe0'):
        assert r['e'] == serial[('water', 'pbe0')]['e']
        assert not r['handles'] and not r['timed']


def test_one_rank_world_is_pyscf(serial):
    """Without a communicator, and on a world of one, nothing here runs.

    Bitwise against pyscf's own SCF, and the mean field still carries pyscf's
    own DF object and numint: a rank count of one must not pay a reduction, a
    broadcast or a re-associated sum for a distribution that is not happening.
    """
    for xc in (None, 'pbe0'):
        ref = serial[('water', xc)]
        mf = mean_field(WATER, 'cc-pvdz', xc)
        numint_before = getattr(mf, '_numint', None)
        distributed_mean_field(mf, None)
        assert mf.e_tot == ref['e']
        assert type(mf.with_df) is type(ref['mf'].with_df)
        assert getattr(mf, '_numint', None) is numint_before
        assert distributed_df_jk(mf, None) is None
        assert distributed_numint(mf, None) is None

        alone = run_simulated(
            lambda comm: converge(comm, mean_field(WATER, 'cc-pvdz', xc)), 1)
        assert alone[0]['e'] == ref['e']
        assert alone[0]['with_df'] == 'DF'
        # No dict either: a mean field that carries one ran distributed.
        assert getattr(mf, '_distributed_timings', None) is None


def isdf_mean_field():
    """Water/cc-pVDZ Hartree-Fock whose J/K is ISDF's (`isdf_jk`), unrun."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = CONV_TOL, CONV_TOL_GRAD
    return mf


def test_an_interpolated_exchange_is_refused_over_ranks():
    """An ISDF `with_df` answers J/K from its own interpolated factors, which
    the slices would silently replace by plain density-fitted rows: over two
    ranks it is refused on every rank, by class name, before any slice is
    built or any cycle run; without a communicator the same mean field is its
    own serial kernel, bit for bit."""
    mfs = [isdf_mean_field() for _ in range(2)]

    def one_rank(comm):
        mf = mfs[comm.Get_rank()]
        try:
            distributed_mean_field(mf, comm)
        except NotImplementedError as exc:
            return str(exc), type(mf.with_df).__name__, mf.mo_coeff
        return None, type(mf.with_df).__name__, mf.mo_coeff

    for message, with_df, mo_coeff in run_simulated(one_rank, 2):
        assert message is not None, 'an ISDF mean field ran distributed'
        assert "ISDFJK is not pyscf's DF" in message
        assert 'density-fitted three-centre tensor' in message
        assert with_df == 'ISDFJK' and mo_coeff is None
    ref = isdf_mean_field()
    ref.kernel()
    mf = isdf_mean_field()
    distributed_mean_field(mf, comm=None)
    assert type(mf.with_df).__name__ == 'ISDFJK'
    assert mf.e_tot == ref.e_tot
    assert np.array_equal(mf.mo_coeff, ref.mo_coeff)
    assert getattr(mf, '_distributed_timings', None) is None


def timed(comm, mf, **kwargs):
    """A distributed SCF and the timings it left on the mean field."""
    out = converge(comm, mf, **kwargs)
    out['timings'] = mf._distributed_timings
    out['cycles'] = int(mf.cycles)
    return out


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name,atom,basis,xc', CASES)
def test_timings_account_for_the_wall(size, name, atom, basis, xc):
    """Every key on every rank, the stages inside the total, cycles pyscf's.

    The stages fall SHORT of the total by the installation of the handles,
    which belongs to no stage; nothing may exceed it. Every rank runs the
    guess and the loop and takes part in every lockstep, and every rank's
    locksteps cover the same payload, at least the density at every Fock
    piece -- moved, or skipped by the checked locksteps where every rank's
    digest was rank 0's.
    """
    out = on_ranks(timed, size, atom, basis, xc)
    nao = gto.M(atom=atom, basis=basis, verbose=0).nao_nr()
    counts = []
    for r in out:
        t = r['timings']
        assert sorted(t) == sorted(TIMING_KEYS)
        json.dumps(t)                  # a caller may record it as JSON
        assert all(isinstance(t[k], float)
                   for k in STAGE_KEYS + ('scf_lockstep_mb',
                                          'scf_lockstep_skipped_mb'))
        assert sum(t[k] for k in STAGE_KEYS) <= t['scf_total'] + TIMER_SLACK
        assert t['scf_cycles'] == r['cycles'] > 0
        assert t['scf_build_slices'] > 0
        # The build's two halves: the columns this rank evaluated and the
        # exchange that turned everyone's columns into everyone's rows.
        assert t['scf_build_integrals'] > 0
        assert t['scf_build_exchange'] > 0
        assert t['scf_build_blocks'] > 0
        assert (t['scf_build_integrals'] + t['scf_build_exchange']
                <= t['scf_build_slices'] + TIMER_SLACK)
        assert t['scf_fock_jk'] > 0
        assert (t['scf_fock_xc'] > 0) == bool(xc)
        assert (t['scf_grid'] > 0) == bool(xc)
        assert t['scf_guess'] > 0 and t['scf_driver'] > 0
        assert t['scf_lockstep'] > 0
        counts.append((t['scf_requests_jk'], t['scf_requests_xc'],
                       t['scf_lockstep_mb'], t['scf_lockstep_skipped_mb']))
    # Every rank takes part in every Fock piece and every lockstep: a rank
    # that counted a different number of them fell out of step.
    assert len(set(counts)) == 1
    assert counts[0][0] >= out[0]['cycles']
    assert counts[0][1] == (counts[0][0] if xc else 0)
    assert (counts[0][2] + counts[0][3]
            >= (counts[0][0] + counts[0][1]) * nao ** 2 * 8 / 1e6)


def test_timings_survive_split_grid_off():
    """With the quadrature whole on rank 0 every rank still takes part in
    every quadrature -- the others with no point, adding zeros -- and the
    grid it builds is rank 0's own time."""
    out = on_ranks(lambda comm, mf: timed(comm, mf, split_grid=False), 2,
                   WATER, 'cc-pvdz', 'pbe0')
    for r in out:
        t = r['timings']
        assert sorted(t) == sorted(TIMING_KEYS)
        assert t['scf_requests_xc'] == t['scf_requests_jk'] > 0
        assert sum(t[k] for k in STAGE_KEYS) <= t['scf_total'] + TIMER_SLACK
    assert out[0]['timings']['scf_grid'] > 0        # rank 0 builds it


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
