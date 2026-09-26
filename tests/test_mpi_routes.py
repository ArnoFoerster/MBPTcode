"""Every distributed route must reproduce the serial answer, under real ranks.

    python tests/test_mpi_routes.py              # serial references only
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_mpi_routes.py   # the check

Script, not a pytest module: it initializes MPI, which a sandboxed test runner
cannot, and it exits with a status; pytest collects nothing from it. With
MBPT_USE_MPI=0 the script runs serially even where mpi4py is installed.
`main(comm)` takes the communicator, so `run_simulated(main, n)` runs the same
checks over thread-ranks where MPI cannot start.

THE RUN IS ONE DISTRIBUTED REGION. `main` enters `with distributed(comm):`
once, as a job script does, and every rank runs every driver replicated -- the
SCF loop, the Davidson, the chains; MPI lives inside the kernels, which
lockstep at entry what can differ between ranks and all-reduce or gather their
partials. A kernel called without `comm=` inside the region takes the region's
communicator (the CONTEXT path, which is what a chain or a driver reaches); one
called with `comm=comm` takes it explicitly (the EXPLICIT path, the kernel's
own entry). Both are gated, each check says which it covers, and the serial
reference beside it is computed on every rank inside `distributed(None)`. An
exception on any rank aborts the job through the region's guard instead of
leaving the others waiting in a collective.

THE INPUTS ARE ONE CALCULATION'S. The factors are built inside the region, so
their points, the mean-field arrays they read and the fit's replicated tail are
rank 0's on every rank; every kernel below takes those inputs, identical by
construction, and a serial reference on rank r is of the same numbers as rank
0's. The chain gates compare against rank 0's serial force, since each rank
converges the chain's SCF on its own.

FOUR STANDARDS, each where the design puts it:
  bitwise   a lockstepped output, or an output partition (rows computed once
            and gathered verbatim): identical on every rank, and identical to
            serial where the only reductions add exact zeros
  reduced   a sum re-associated over the partition: a bar at the summation
            order (1e-12 or 1e-11 relative, 1e-10 Ha, 1e-9 eV), `SIGMA_REL`
            for the cancelling Wt(tau) sum
  bounded   a sum reduced over the ranks from the serial kernel's own
            addends: every rank's partial its addends in the kernel's order,
            bitwise, and the reduced sum within the rounding bound of the
            partials' exact sum, the most any reduction tree can move it
            (tests/reduction_bounds.py)
  anchored  a bar measured on this run: `COMPOSED_GRAD_K` times the one-thread
            repeat of the serial force, `SOP_ANCHOR_K` times the pole model's
            own ulp response. Also wherever two evaluations each ran pyscf's
            own threaded work: its OpenMP GEMM (`lib.ddot`) splits K over the
            threads and adds the partials in thread-arrival order, so at 16
            threads no pyscf result repeats bit for bit, and two SCF runs, or
            two chains' K builds on one mean field, are never bitwise

Checks (path; standard):
  * SCF, RHF and PBE0, `distributed_mean_field`: split_grid=True (context),
    split_grid=False (explicit): e_tot and density against serial (anchored:
    `COMPOSED_GRAD_K` times what running the serial SCF again moves them, one
    ulp at the least, and at least the reduced 1e-10 Ha and 1e-8, since two
    SCF runs are compared), mo_coeff across ranks (bitwise), the cderi slices
    tile naux (exact), PBE0 split_grid True against False (anchored, the same
    energy bar)
  * the fixture's factors, `separable_factors` in the region (context):
    identical on every rank (bitwise)
  * the space-time GW window's low-memory path: `freq_block` (context),
    `scratch_dir` (explicit): against in-core (reduced)
  * static W, `isdf_bse_factors` (context): against serial (reduced)
  * the ISDF BSE, `solve_bse_isdf` (context): roots, (A-B) probe and GW
    diagonal against serial (reduced); every rank holds rank 0's roots
    (bitwise) after the same Davidson -- `davidson_vind_calls` and
    `davidson_iterations` equal to rank 0's (exact)
  * the same BSE with rank != 0 handed a perturbed spectrum and perturbed
    factors (explicit): roots against serial (reduced), the perturbation
    overwritten with rank 0's arrays (bitwise), rank 0's roots and Davidson on
    every rank (bitwise, exact)
  * the BSE block action's grid reduction (explicit), recorded on the
    communicator it is handed: per trial vector one reduce-scatter of the
    (n_occ, M) partial X_o^T (Zt * P), each rank receiving its
    `contiguous_block` rows alone and the ranks' rows tiling the grid once,
    no all-reduce as long as that intermediate, and A and B the bits of the
    same action on the unrecorded communicator (exact, bitwise)
  * the factors held as grid-point slices, `separable_factors(sliced=True)`
    (context): each rank's rows of the fixture and nothing more (bitwise,
    exact); the GW window (explicit) and the BSE (context) on them against
    the fixture's factors (bitwise), rank 0's Davidson on every rank, and
    each whole-array gather once per solve or sweep (exact)
  * the in-core GW window split over grid rows (`_qp_grid_rows`), on sliced
    factors at tiles and blocks small enough that the ranks share a tau
    point's tiles and block pairs (explicit): traced line by line in every
    frame between the solve and the running line, no array whole along the
    grid twice (an (M, M) block), no whole stack of (naux, naux) slices and
    no (n, nao, nao) AO self-energy, and no more whole (naux, naux) slices
    at once than the rank's own frequencies and `ROW_SPARE_SLICES` (exact);
    the same trace with a whole Pi(tau) slice planted in the sweep finds it
    on every rank (exact, shown failing); the window against serial
    (anchored: `COMPOSED_GRAD_K` times what relabelling the grid points moves
    it serially, floored at the root finder's `QP_BISECTION_TOL`), the window
    and W(0) rank 0's on every rank (bitwise)
  * the route's two reduced sums on this run's ranks (explicit, recorded on
    the communicator): proj(tau) over the grid-row tiles and the branch sums
    of Sigma over the block pairs, each against the serial kernel's own
    addends -- in its order they are its result (bitwise), this rank's
    partial of every tau point is its own items' addends in that order
    (bitwise), and what it receives lies within `join_bound` of the exact
    sum of the ranks' partials, half an ulp per join, floored at one ulp of
    the sum's largest element (bounded: the most any reduction tree can
    move it, tests/reduction_bounds.py); chi0's rows the update of the
    received proj rows, W(0) the inversion of their chi0(0) and the route's
    W(0) that one (bitwise)
  * `qp_set_gradient` (explicit): roots against serial (bitwise), adjoints
    (reduced)
  * `rpa_energy_and_adjoint` (context): energy and adjoints (reduced)
  * `qp_gradient_space_time`: contour deformation (context) root, Z and
    adjoints (reduced); pole model (explicit) (anchored)
  * `static_screening` (explicit) (reduced); `screened_interaction_tau`,
    `selfenergy_diag` and its backward pass (context), `selfenergy_block` and
    its backward pass (explicit) (reduced, `SIGMA_REL`)
  * the BSE@GW chain, `ExcitedStateChain` in the region (context): ONE grid on
    every rank (bitwise hashes), the excitation force (anchored), the
    quasiparticle force (anchored, at least `ISDF_GRADIENT_FLOOR`)
  * the chain with a build-only factory (context: the chain converges its SCF
    through `distributed_mean_field` on the region's ranks): orbitals across
    ranks (bitwise), mf0 against the self-converged one (reduced), its force
    (anchored); at one rank, where that is `mf.kernel()`, ONE plain run of it
    on the mean field the factory built, no distributed handles left on it,
    and the chain holding that run's energy and orbitals (bitwise)
  * the state-pair surface on sliced factors, `RPABSESurface(sliced=True)`
    (context), both layouts handed ONE reference and ONE displaced mean
    field: its composed force, energy and root at a displaced geometry
    against the whole layout's on the same ranks (anchored: `COMPOSED_GRAD_K`
    times the largest difference between `SLICED_SAMPLES` evaluations of one
    layout on those mean fields, the force at the excitation force's gate at
    the least, since each layout's chains still run pyscf's K builds and the
    mean field's force for themselves), every rank rank 0's (bitwise); the
    factorization holding this rank's rows and no whole fit, and each
    whole-array gather once per sweep or solve (exact). The layout itself is
    bitwise on one mean field where pyscf repeats its bits, which
    tests/test_force_serial_shaped.py shows on a shape-sensitive BLAS
  * the same surface on the row-distributed fit, `fit='rows'` (context),
    every surface on the same reference and displaced mean field: its
    composed force against the whole-fit sliced surface on the same ranks
    (anchored: `FIT_REASSOCIATION_K` times what one reassociation of the
    whole fit's sums moves that force), every rank rank 0's (bitwise), and
    nothing reachable from it holding the grid whole beside a factor's or
    the fit's width, its rows the row fit's own at this rank's tiles
    (exact); the row fit's own adjoint in the nuclear assembly against the
    whole adjoint of the same estimator (anchored: `FIT_REASSOCIATION_K`
    times what reordering the whole adjoint's sums over the test set moves
    the force), and each assembly traced line by line in every frame under
    src/: no array of the grid by a factor's, the fit's or the test set's
    width beyond the X_bar and D_bar handed in, and the adjoint's ledger at
    this rank's tiles, the metric root on rank 0 alone (exact)
  * the state-pair surface on sliced factors with the grid BSE adjoint,
    `bse_adjoint='grid'` (context), both adjoints handed ONE reference and ONE
    displaced mean field: its composed force and energy against the default
    adjoint's on the same ranks (anchored: `COMPOSED_GRAD_K` times what a
    repeat of the sliced surface moves them in the section above, the force
    at the excitation force's gate and the energy at one ulp at the least,
    since each surface's chains run pyscf's K builds for themselves), every
    rank rank 0's (bitwise), and no
    three-index block reachable from the chain or its forward pass at the
    reverse call, where the default route's cache is found (exact); at that
    call the grid adjoint's four Casida-level seeds, computed over the ranks,
    the bits of the same kernel run serially on the gathered factors and
    every rank's rank 0's (bitwise), and the kernel traced line by line in
    every frame it enters: no array with the whole grid on an axis outside
    the one named boundary gather of X_bar and D_bar
    (`adjoints_at_the_boundary`), no whole-factor gather (exact)
  * the lockstep counters of the whole run: calls and bytes identical on every
    rank (exact; `lockstep` is a collective)
  * ONE audited chain forward, `distributed(comm, audit=True)`, on a mean field
    each rank converged alone: no `agreement` found the ranks apart, and the
    audit ran; `mismatched_calls` and `max_abs_diff` report the cross-rank
    drift the locksteps absorbed
"""
import copy
import hashlib
import inspect
import os
import sys
import tempfile
import threading
import warnings
import weakref

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import dft, gto, scf
try:
    from threadpoolctl import threadpool_limits
except ImportError:  # without it the excitation gate keeps its bare floor
    threadpool_limits = None

from src.Base.constants import (COMPOSED_GRAD_K, FIT_REASSOCIATION_K,
                                QP_BISECTION_TOL,
                                HARTREE_TO_EV, ISDF_GRADIENT_FLOOR)
from src.Base.distributed_df import distributed_df_storage, distributed_mean_field
from src.Base.utils.grids import (gauss_legendre_grid, minimax_frequency_grid,
                                  minimax_time_grid)
from src.Base.utils.mpi_grid import (SimulatedComm, broadcast,
                                     contiguous_block, distributed,
                                     grid_comm, lockstep_stats, partition)
from src.Base import separable_ri
from src.Base.separable_ri import DEFAULT_REGULARIZATION
from src.Base.sliced_factors import SlicedFactors, whole_factor
from src.Base.utils.time_frequency import (COSINE_WT, TimeFrequencyGrid,
                                           minimax_transform_weights)
from src.SingleReference.GW.imaginary_time import (SigmaPairs,
                                                   screened_interaction_rows,
                                                   self_energy_branch_sums,
                                                   self_energy_fit_ranges)
from src.SingleReference.GW.space_time import (DEFAULT_NPADE, _dyson_owned,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_interstate_backward)
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         isdf_block_action,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, chi0_frequency_rows, polarizability_projected_tau,
    polarizability_tiles, split_branches, wave_items)
from src.gradients import factor_chain, isdf_derivatives
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.isdf_derivatives import (collocation_adjoint,
                                            dfactor_adjoint_gauges,
                                            product_pairs)
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.qp_space_time import (qp_gradient_space_time,
                                         qp_set_gradient)
from src.gradients.reaction_field_adjoint import static_grid, static_screening
from src.gradients.space_time_adjoint import (polarizability_tau,
                                              rpa_energy_and_adjoint,
                                              screened_interaction_tau,
                                              selfenergy_block,
                                              selfenergy_block_backward,
                                              selfenergy_diag,
                                              selfenergy_diag_backward,
                                              sigma_transforms)
from tests.reduction_bounds import reduced_sum_verdict

warnings.simplefilter('ignore')

#: The LOWER BOUND of the excitation gradient's gate, in Ha/Bohr. The gate
#: itself is `COMPOSED_GRAD_K` times the re-association scatter measured on
#: the machine at hand (`one_thread_scatter`) wherever that is larger, because
#: the composed force scatters by more than this on a machine with more
#: threads than the laptop this number was read on: 6.0e-9 over two ranks
#: there, 1.57e-8 over one rank per node on two cluster nodes at 16 threads,
#: against a 1.8e-8 one-thread repeat of the SERIAL force on the laptop
#: (water/cc-pVDZ). Held alone, a number read on one machine gates the other
#: machine's BLAS rather than its distribution.
COMPOSED_GRAD_FLOOR = 1.5e-8
#: Wt(tau) = sum_w Ctw[t,w] (W_w - I) is a CANCELLING sum -- the minimax
#: weights alternate in sign and are large against Wt itself -- so a
#: partition over frequency re-associates it and moves the last bits by
#: that cancellation factor, 1e-12 here rather than 1e-16.
SIGMA_REL = 1e-10
#: How many times its own ulp response the pole model may sit from serial.
#: `fit_poles` reads its poles off a nonsymmetric eigenvalue problem built from
#: a least-squares solve, so a last-bit change of the screening arrives
#: magnified by a factor the fixture sets: the same code on three 8-rank runs
#: read sop adjoints 5.0e-12, 2.1e-10 and 4.8e-9 from serial, each run on a
#: freshly converged mean field. So the bar is measured, not fixed: the root,
#: Z and adjoints of the serial sop gradient with X moved one ulp per element
#: (`sop_anchor`). On the laptop fixture that anchor is 2.1e-13 Ha, 5.0e-12
#: and 5.8e-11, and the last-bit wc change a shape-dependent GEMM made before
#: the rows were cut from the serial block moved the answer by 0.63 of it at
#: the median and 2.5 at most over 41 draws; five covers that twice and fails
#: a deviation of ten anchors. A real defect in this chain shows 1e-3.
SOP_ANCHOR_K = 5
#: Fixed ulp draws the anchor takes the largest response over: one draw alone
#: spans 3.9e-12 to 5.9e-11 on the adjoints.
SOP_ANCHOR_SEEDS = 3
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: One rank thread at a time inside `one_thread`: simulated ranks share the
#: process's BLAS pool, and two overlapping `threadpool_limits` regions leave
#: it at one thread, since the later one restores what the earlier one set.
ONE_THREAD_LOCK = threading.Lock()
#: The package whose frames the assembly scan traces.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'src') + os.sep
#: One hydrogen 0.03 A along y, where a surface fits and differentiates anew.
WATER_DISPLACED = 'O 0 0 0.1173; H 0 0.7872 -0.4692; H 0 -0.7572 -0.4692'
#: The whole-array gathers of one composed force on sliced factors: the dRPA
#: sweep and assembly, then the static W, the quasiparticle solve, the
#: Davidson, the BSE cache and adjoint, the quasiparticle and chi0 adjoints
#: and the excited assembly, one each.
SLICED_CHAIN_GATHERS = {'X_mo': 7, 'D': 8, 'X_o': 3, 'X_v': 2}
#: How many times each layout of the sliced section is evaluated on the same
#: mean fields: the spread of a layout's own repeats is the bar the layouts
#: are compared at. A threaded pyscf moves a scalar like the root by a
#: roughly Gaussian amount per run; with three samples a layout the larger
#: same-layout difference sits under a third of the cross-layout one in 0.16%
#: of runs, with two in 3.7%.
SLICED_SAMPLES = 3
#: The row fit's tile edge here: water's 444 points in 7 tiles, so every rank
#: count owns a different set of them and the tiled Cholesky crosses ranks.
ROW_FIT_BLOCK = 64
#: The Davidson residual of the row-fit section, below the fit's response:
#: at the default `BSE_DAVIDSON_CONV_TOL` the solver's own convergence moves
#: a root by more than one reassociation of the fit does
#: (tests/test_chain_row_fit.py).
ROW_FIT_BSE_CONV_TOL = 1e-9
#: Trial vectors in the batch the recorded block action is applied to.
RECORDED_BATCH = 3
#: The grid-row section's working-set budget: water's 444 points in 16 chi0
#: tiles and 4 self-energy blocks, so ranks share a tau point's tiles and
#: block pairs, not whole points.
ROW_TILE_GB = 3e-4
#: Its tau points: 3 and 8 ranks leave a last window of the sweep that splits
#: a point's tiles and block pairs over the ranks.
ROW_NTAU = 14
#: Whole (naux, naux) slices a rank of the grid-row route may hold at once
#: beside its own frequencies' W - I: W(0), a gathered chi0 slice and the
#: identity it is inverted against, or the two Wt(tau) slices of a window
#: and W(0).
ROW_SPARE_SLICES = 3
#: Random grid orders the grid-row section's relabelling anchor takes the
#: largest response over.
ROW_PERMUTATIONS = 3


class Gate:
    """This rank's verdicts, gathered from every rank at the end: a gate that
    fails on a rank other than 0 would otherwise be invisible -- rank 0 prints
    its own pass, the other rank exits non-zero and the launcher reports only
    an exit code."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()
        self.size = 1 if comm is None else comm.Get_size()
        self.verdicts = []

    def say(self, text):
        """One line from rank 0."""
        if self.rank == 0:
            print(text, flush=True)

    def section(self, title):
        self.say(f'\n-- {title}, {self.size} rank(s)')

    def info(self, text):
        """A number printed beside the checks; it gates nothing."""
        self.say(f'  [info] {text}')

    def check(self, ok, label, detail=''):
        """Record one verdict of this rank; rank 0 prints its own."""
        ok = bool(ok)
        self.verdicts.append((ok, label, detail))
        self.say(f"  [{'ok' if ok else 'FAIL'}] {label}"
                 + (f'  ({detail})' if detail else ''))
        return ok

    def everyone(self, obj):
        """Every rank's `obj`, rank-ordered."""
        return self.comm.allgather(obj) if self.comm is not None else [obj]

    def distinct(self, *arrays):
        """How many different byte strings the ranks hold in `arrays`."""
        return len(set(self.everyone(digest(*arrays))))

    def finish(self):
        """0 when every check passed on every rank, 1 otherwise."""
        failures = [(r, label, detail)
                    for r, own in enumerate(self.everyone(self.verdicts))
                    for ok, label, detail in own if not ok]
        if self.rank == 0:
            for r, label, detail in failures:
                if r != 0:
                    print(f'  [FAIL on rank {r}] {label}'
                          + (f'  ({detail})' if detail else ''))
            print('\n' + ('All checks passed on every rank.' if not failures
                          else 'FAILURES above.'), flush=True)
        return 0 if not failures else 1


class HeldCensus:
    """This thread's census of the arrays bound to a name in every frame
    between a call of `root` and the running line: the shapes found whole
    along the grid twice ('grid square'), a whole stack of (naux, naux)
    slices ('aux stack') or of (nao, nao) ones ('ao stack'), and the most
    whole (naux, naux) slices and the most bytes alive at one line."""

    def __init__(self, root, M, naux, nao):
        self.root, self.M, self.naux, self.nao = root, M, naux, nao
        self.flags, self.slices, self.bytes = {}, 0, 0

    def kind(self, a):
        shape = a.shape
        if sum(n == self.M for n in shape) >= 2:
            return 'grid square'
        if a.ndim == 3 and shape[0] > 1 and shape[1:] == (self.naux,) * 2:
            return 'aux stack'
        if a.ndim == 3 and shape[0] > 1 and shape[1:] == (self.nao,) * 2:
            return 'ao stack'
        return None

    def visit(self, obj, found, depth=0):
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            found[id(base)] = base
        elif depth < 3 and isinstance(obj, dict):
            for v in obj.values():
                self.visit(v, found, depth + 1)
        elif depth < 3 and isinstance(obj, (list, tuple)) and len(obj) < 256:
            for v in obj:
                self.visit(v, found, depth + 1)
        elif depth < 3 and isinstance(obj, ProjRows):
            self.visit(getattr(obj, 'rows', None), found, depth + 1)
        elif depth < 3 and type(obj).__name__ in ('SigmaPairs',
                                                  'SlicedFactors'):
            self.visit(vars(obj), found, depth + 1)

    def take(self, frame):
        chain = []
        while frame is not None and frame.f_code is not self.root:
            chain.append(frame)
            frame = frame.f_back
        if frame is None:
            return
        found = {}
        for f in chain + [frame]:
            for v in list(f.f_locals.values()):
                self.visit(v, found)
        for a in found.values():
            kind = self.kind(a)
            if kind is not None:
                self.flags.setdefault(kind, set()).add(a.shape)
        self.slices = max(self.slices, sum(
            a.shape == (self.naux, self.naux) for a in found.values()))
        self.bytes = max(self.bytes, sum(a.nbytes for a in found.values()))

    def trace(self, fn, *args, **kwargs):
        """fn(*args, **kwargs) with every line of every frame under src/, or
        of this file, taken into the census."""
        here = os.path.abspath(__file__)

        def local(frame, event, arg):
            if event == 'line':
                self.take(frame)
            return local

        def tracer(frame, event, arg):
            name = frame.f_code.co_filename
            return (local if name.startswith(SRC)
                    or os.path.abspath(name) == here else None)

        sys.settrace(tracer)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.settrace(None)


class RecordedSimulatedComm(SimulatedComm):
    """A simulated rank whose sums are recorded in `log`: ('allreduce',
    elements, elements) and ('reduce_scatter', elements in, elements out);
    and in `sums`, each sum's (kind, what this rank handed in, what it
    received), copies."""

    def __init__(self, comm, log):
        super().__init__(comm._world, comm.Get_rank())
        self.log, self.sums = log, []

    def allreduce_sum(self, buf):
        self.log.append(('allreduce', buf.size, buf.size))
        sent = buf.copy()
        super().allreduce_sum(buf)
        self.sums.append(('allreduce', sent, buf.copy()))

    def reduce_scatter_sum(self, send, recv, counts):
        self.log.append(('reduce_scatter', send.size, recv.size))
        super().reduce_scatter_sum(send, recv, counts)
        self.sums.append(('reduce_scatter', send.copy(), recv.copy()))


class RecordedComm:
    """An mpi4py communicator whose sums are recorded in `log` and `sums`
    as `RecordedSimulatedComm` records them, the all-reduce being the
    in-place one `reduce_sum` makes; every other call passes through."""

    def __init__(self, comm, log):
        self._comm, self.log, self.sums = comm, log, []

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def Allreduce(self, sendbuf, recvbuf, op):
        n = np.asarray(recvbuf).size
        self.log.append(('allreduce', n, n))
        sent = np.array(recvbuf, copy=True)
        self._comm.Allreduce(sendbuf, recvbuf, op=op)
        self.sums.append(('allreduce', sent, np.array(recvbuf, copy=True)))

    def Reduce_scatter(self, sendbuf, recvbuf, recvcounts=None, op=None):
        self.log.append(('reduce_scatter', sendbuf[0].size, recvbuf[0].size))
        self._comm.Reduce_scatter(sendbuf, recvbuf, recvcounts=recvcounts,
                                  op=op)
        self.sums.append(('reduce_scatter', np.array(sendbuf[0], copy=True),
                          np.array(recvbuf[0], copy=True)))


def serial(fn, *args, **kwargs):
    """fn(*args, **kwargs) in a serial region: the reference every rank
    computes for itself, with no kernel inside picking up the region's comm."""
    with distributed(None):
        return fn(*args, **kwargs)


def worst_rel(ref, got):
    """The largest relative deviation over a tuple of arrays."""
    return max(np.abs(np.asarray(a) - np.asarray(b)).max()
               / max(np.abs(np.asarray(a)).max(), 1e-300)
               for a, b in zip(ref, got))


def digest(*arrays):
    """A short hash of the exact bytes: what a rank can send to be compared."""
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:12]


def sop_anchor(X, D, eps, nocc, grid, nu, wt, p, mu, ref):
    """(root Ha, Z, adjoint rel) the serial pole-model gradient moves by when
    X moves one ulp per element, the largest over `SOP_ANCHOR_SEEDS` draws."""
    worst = np.zeros(3)
    for seed in range(SOP_ANCHOR_SEEDS):
        sign = np.random.default_rng(seed).choice([-1.0, 1.0], X.shape)
        moved = serial(qp_gradient_space_time,
                       X + sign * np.spacing(np.abs(X)), D, eps, nocc, grid,
                       nu, wt, p, mu=mu, residue_route='sop')
        worst = np.maximum(worst, [abs(moved[0] - ref[0]), abs(moved[1] - ref[1]),
                                   worst_rel(ref[2:], moved[2:])])
    return worst


def grid_tag(chain, m):
    """(points, radii, layout) of the grid THIS rank's chain differentiates.

    The interpolation points, the radii they come from and the pair columns the
    fit keeps ARE the functional, so two ranks holding different ones are
    differentiating two surfaces. It shows up nowhere else: every route gate
    passes, because each kernel locksteps what it reads, and only the
    end-to-end force carries the difference -- 1.5e-3 Ha/Bohr over two nodes,
    which reads as a bad gradient rather than as a grid nobody agreed on.

    A difference of that size is a DISCRETE one. The same force moves 1.6e-2
    per unit relative change of the radii (water/cc-pVDZ), so the last bits of
    an `eigh` or of a dense solve reach 1e-17 here and cannot be it: a grid
    explanation needs the radii themselves to land elsewhere, or the pair
    screen to keep another column set. All three are hashed for that reason.
    """
    radii = repr([(el, [(s, np.asarray(chain.radii[el][s]).tolist())
                        for s in sorted(chain.radii[el])])
                  for el in sorted(chain.radii)])
    return (digest(chain.coords(m)),
            hashlib.sha1(radii.encode()).hexdigest()[:12],
            digest(*chain.layout))


def scf_mean_field(mol, xc):
    """RHF or PBE0, built and not run, at the file's own gradient tolerance so
    the gates test the distribution rather than where the iteration stopped."""
    out = scf.RHF(mol) if xc is None else dft.RKS(mol, xc=xc)
    out = out.density_fit(auxbasis='cc-pvdz-ri')
    out.conv_tol, out.conv_tol_grad, out.max_cycle = 1e-14, 1e-11, 200
    return out


def chain_scf(m):
    """A mean field converged for gradient work (conv_tol_grad 1e-11), which
    is what the excited-state Lagrangian assumes."""
    out = scf.RHF(m).density_fit(auxbasis='cc-pvdz-ri')
    out.conv_tol, out.conv_tol_grad, out.max_cycle = 1e-14, 1e-11, 200
    out.kernel()
    return out


def chain_scf_unrun(m):
    """The same mean field BUILT AND NOT RUN: the chain converges it.

    `chain_scf` runs its own SCF, which every rank then repeats -- the one
    stage of a chain that does not divide. Leaving the `kernel()` out hands it
    to `factor_chain.converged_factory`, which converges it through
    `distributed_mean_field` on the region's ranks: every rank runs pyscf's
    loop against the reduced J/K on its block of the auxiliary index. The
    tolerances are the object's, not the caller's, so they are the same either
    way.
    """
    out = scf.RHF(m).density_fit(auxbasis='cc-pvdz-ri')
    out.conv_tol, out.conv_tol_grad, out.max_cycle = 1e-14, 1e-11, 200
    return out


def recorded(unrun, runs):
    """The build-only factory `unrun` whose mean field's `kernel`, when the
    chain's factory runs it, appends to `runs` how it was called, the mean
    field and the energy, orbitals and orbital energies that one run
    converged to."""
    def build(m):
        out = unrun(m)
        run = out.kernel

        def kernel(*args, **kwargs):
            result = run(*args, **kwargs)
            runs.append(dict(args=args, kwargs=kwargs, mf=out, e_tot=out.e_tot,
                             mo_coeff=np.array(out.mo_coeff, copy=True),
                             mo_energy=np.array(out.mo_energy, copy=True)))
            return result

        out.kernel = kernel
        return out

    return build


def one_thread(evaluate):
    """`evaluate()` in a serial region with BLAS held to one thread: the same
    calculation with every threaded GEMM's sums re-associated, as a partition
    over ranks re-associates them. None without threadpoolctl, which leaves a
    gate anchored on it at its floor."""
    if threadpool_limits is None:
        return None
    with ONE_THREAD_LOCK, threadpool_limits(limits=1), distributed(None):
        return evaluate()


def one_thread_scatter(m, mf, g_ref):
    """|d| between a serial excitation force and the SAME force with BLAS held
    to one thread: what re-associating these sums costs on THIS machine.

    A thread count is a summation order. The reductions inside every threaded
    GEMM of the chain re-associate when it changes, which is what a partition
    over tau and over frequency does to the sweeps, and the orbital-response
    solve amplifies the last bits of either the same way -- so this is the
    size of difference the distribution is entitled to produce, measured where
    the test runs rather than carried in from another machine.

    It is the SOLVERS' tolerances that this does not measure, and they are not
    what moves the force: the Lagrangian's lgmres tightened a hundredfold
    (ORBITAL_MULTIPLIER_TOL 1e-11 -> 1e-13, its residual 5.9e-12 -> 8.8e-14)
    moves this gradient 4.8e-13, and the Casida step here is dense (95 pairs,
    well under BSE_DENSE_MAX_NOV) so BSE_DAVIDSON_CONV_TOL never enters. The
    scatter measured here is 1.8e-8 Ha/Bohr, four orders above either.

    Zero without threadpoolctl, which leaves the gate at COMPOSED_GRAD_FLOOR.
    """
    g = one_thread(lambda: ExcitedStateChain(m, chain_scf, mf=mf)
                   .excitation_gradient()[0])
    return 0.0 if g is None else float(np.abs(g - g_ref).max())


def region_factors(gate, mf, mol):
    """The fixture's factors, built inside the region: rank 0's on every rank.

    `separable_factors` locksteps the mean-field arrays it reads and the points
    it placed, reduces the three-centre pass and locksteps the fit's tail, so
    every kernel below reads one set of factors and no kernel broadcasts them
    again. It also leaves rank 0's orbitals in every rank's `mf`.
    """
    gate.section('the ISDF factors inside the region')
    F = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')              # context
    if gate.size == 1:
        gate.info(f'serial fit: M {F[1].shape[0]}, naux {F[1].shape[1]}')
        return F
    n = gate.distinct(*F)
    gate.check(n == 1, f'separable_factors [context] over {gate.size} ranks: '
               'X_mo, D, X_ao and coords bitwise identical on every rank',
               f'{n} distinct of {gate.size}')
    F_serial = serial(separable_factors, mf, mol, auxbasis='cc-pvdz-ri')
    gate.info(f'the fit over {gate.size} ranks against the serial fit: X rel '
              f'{worst_rel(F_serial[:1], F[:1]):.2e}, D rel '
              f'{worst_rel(F_serial[1:2], F[1:2]):.2e} (the reduced three-centre '
              'pass re-associates; tests/test_isdf_fit_ranks.py gates it)')
    return F


def scf_routes(gate, mol):
    """The distributed SCF: pyscf's loop on every rank against reduced J/K."""
    gate.section('the distributed SCF')
    for label, xc in (('RHF', None), ('PBE0', 'pbe0')):
        mf_ref = scf_mean_field(mol, xc)
        serial(mf_ref.kernel)
        dm_ref = np.asarray(mf_ref.make_rdm1())
        # Every comparison below spans two SCF runs, which a threaded pyscf
        # does not repeat bit for bit: the bars are anchored on a repeat.
        mf_rep = scf_mean_field(mol, xc)
        serial(mf_rep.kernel)
        rep_e = abs(mf_rep.e_tot - mf_ref.e_tot)
        rep_dm = np.abs(np.asarray(mf_rep.make_rdm1()) - dm_ref).max()
        bar_e = max(1e-10, COMPOSED_GRAD_K * max(
            rep_e, np.spacing(abs(mf_ref.e_tot))))
        bar_dm = max(1e-8, COMPOSED_GRAD_K * max(
            rep_dm, np.spacing(np.abs(dm_ref).max())))
        gate.info(f'{label} serial SCF repeat: dE = {rep_e:.2e} Ha, d(dm) = '
                  f'{rep_dm:.2e}; bars {bar_e:.2e} Ha, {bar_dm:.2e}')
        split_mfs = {}
        for split_grid in (True, False):
            mf_d = scf_mean_field(mol, xc)
            if split_grid:
                path = 'context'
                distributed_mean_field(mf_d, split_grid=True)
            else:
                path = 'explicit'
                distributed_mean_field(mf_d, gate.comm, split_grid=False)
            split_mfs[split_grid] = mf_d
            tag = f'{label} split_grid={split_grid} [{path}]'
            d_e = abs(mf_d.e_tot - mf_ref.e_tot)
            d_dm = np.abs(np.asarray(mf_d.make_rdm1()) - dm_ref).max()
            gate.info(f'{tag}: dE = {d_e:.2e} Ha, d(dm) = {d_dm:.2e}')
            gate.check(d_e < bar_e,
                       f'{tag} e_tot == serial, {gate.size} rank(s)',
                       f'|dE| = {d_e:.2e} of {bar_e:.2e} Ha')
            gate.check(d_dm < bar_dm, f'{tag} density == serial',
                       f'|d(dm)| = {d_dm:.2e} of {bar_dm:.2e}')
            n = gate.distinct(mf_d.mo_coeff)
            gate.check(n == 1, f'{tag} mo_coeff bitwise identical across ranks',
                       f'{n} distinct of {gate.size}')
            storage = distributed_df_storage(mf_d, gate.comm)
            naux = storage['naux']
            block = contiguous_block(naux, gate.rank, gate.size)
            blocks = gate.everyone(block)
            tiles = (storage['rows'] == block[1] - block[0]
                     and blocks[0][0] == 0 and blocks[-1][1] == naux
                     and all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
                     and sum(b[1] - b[0] for b in blocks) == naux)
            gate.check(tiles, f'{tag} distributed_df_storage rows tile naux, '
                       f'{gate.size} rank(s)',
                       f"rows {storage['rows']} of block {block}; "
                       f'blocks {blocks}, naux {naux}')
        if xc is not None:
            d_split = abs(split_mfs[True].e_tot - split_mfs[False].e_tot)
            gate.info(f'{label} split_grid True vs False: dE = {d_split:.2e} Ha')
            gate.check(d_split < bar_e,
                       f'{label} split_grid True and False agree',
                       f'|dE| = {d_split:.2e} of {bar_e:.2e} Ha')


def gw_low_memory(gate, mf, mol, F):
    """The space-time GW window with W folded a frequency block at a time."""
    gate.section('GW low-memory path (freq_block, scratch_dir)')
    nocc = mol.nelectron // 2
    homo = serial(solve_qp_energy_space_time, mf, mol, nocc, nocc - 1,
                  factors=F) * HARTREE_TO_EV
    blocked = serial(solve_qp_energy_space_time, mf, mol, nocc, nocc - 1,
                     factors=F, freq_block=3) * HARTREE_TO_EV
    gate.check(abs(blocked - homo) < 1e-9, 'serial freq_block == in-core',
               f'|d| = {abs(blocked - homo):.2e} eV')
    if gate.size == 1:
        gate.info(f'serial reference {homo:.9f} eV')
        return
    dist = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1, factors=F,
                                      freq_block=3) * HARTREE_TO_EV
    gate.check(abs(dist - homo) < 1e-9,
               f'freq_block [context] over {gate.size} ranks == in-core',
               f'|d| = {abs(dist - homo):.2e} eV')
    with tempfile.TemporaryDirectory(prefix=f'mbpt_mpi_r{gate.rank}_') as scratch:
        dist2 = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1, factors=F,
                                           freq_block=2, scratch_dir=scratch,
                                           comm=gate.comm) * HARTREE_TO_EV
    gate.check(abs(dist2 - homo) < 1e-9,
               f'scratch_dir [explicit] over {gate.size} ranks == in-core',
               f'|d| = {abs(dist2 - homo):.2e} eV')


def static_w(gate, mf, mol, F):
    """The BSE kernel's static screening over the tau partition."""
    gate.section('static W for the BSE kernel')
    nocc = mol.nelectron // 2
    W_serial = serial(isdf_bse_factors, mf, mol, nocc, factors=F)[2]
    if gate.size == 1:
        gate.info(f'serial reference |W| max {np.abs(W_serial).max():.6f}')
        return
    W_dist = isdf_bse_factors(mf, mol, nocc, factors=F)[2]
    d = np.abs(W_dist - W_serial).max()
    gate.check(d < 1e-12, f'W(0) [context] over {gate.size} ranks == serial',
               f'|d| = {d:.2e}')


def same_davidson(gate, label, om, info):
    """Every rank holds rank 0's roots and ran rank 0's iteration.

    The Davidson runs on every rank on the block action's lockstepped output
    and its roots are lockstepped at the end, so the roots are bitwise rank
    0's; the counts are what the lockstep cannot repair after the fact -- a
    rank that took another decision would have called the action a different
    number of times, and the last lockstep would hide that it iterated on
    alone.
    """
    seen = gate.everyone((digest(om), info['stats'].get('davidson_vind_calls'),
                          info['timings'].get('davidson_iterations')))
    n = len({s[0] for s in seen})
    gate.check(n == 1, f"{label}: every rank holds rank 0's roots, bitwise",
               f'{n} distinct of {gate.size}')
    calls, iterations = [s[1] for s in seen], [s[2] for s in seen]
    ran = (calls[0] is not None and calls[0] > 0 and iterations[0] is not None
           and all(c == calls[0] for c in calls)
           and all(i == iterations[0] for i in iterations))
    gate.check(ran, f"{label}: every rank ran rank 0's Davidson",
               f'davidson_vind_calls {calls}, davidson_iterations {iterations}')


def bse_routes(gate, mf, mol, F):
    """The whole ISDF BSE: GW diagonal, static W, probe and Davidson."""
    gate.section('ISDF BSE')
    nocc = mol.nelectron // 2
    om_s, _, _, info_s = serial(solve_bse_isdf, mf, mol, nocc, nroots=3,
                                probe=True, progress=False, factors=F)
    if gate.size == 1:
        gate.info('serial reference '
                  + ' '.join(f'{w * HARTREE_TO_EV:.6f}' for w in om_s) + ' eV')
        return
    om_d, _, _, info_d = solve_bse_isdf(mf, mol, nocc, nroots=3, probe=True,
                                        progress=False, factors=F)
    d = np.abs(om_d - om_s).max() * HARTREE_TO_EV
    gate.check(d < 1e-9, f'BSE roots [context] over {gate.size} ranks == serial',
               f'|d| = {d:.2e} eV; nranks reported {info_d["nranks"]}')
    d_amb = abs(info_d['min_eig_amb'] - info_s['min_eig_amb'])
    gate.check(d_amb < 1e-9, 'min eig(A-B) probe agrees', f'|d| = {d_amb:.2e}')
    d_eps = np.abs(info_d['eps'] - info_s['eps']).max() * HARTREE_TO_EV
    gate.check(d_eps < 1e-9, 'GW diagonal agrees', f'|d| = {d_eps:.2e} eV')
    same_davidson(gate, 'BSE [context]', om_d, info_d)

    # THE RANKS NEED NOT ARRIVE WITH THE SAME NUMBERS. Across nodes a rank's
    # own SCF and fit do not repeat bit for bit; here rank != 0 is handed a
    # spectrum shifted by 1e-3 Ha and factors scaled by 1 + 1e-6 -- far more
    # than the last-bit drift, so a run that took its data or its Davidson
    # decisions from such a rank could not land on the serial roots. It must:
    # the entry locksteps write rank 0's arrays into every rank's buffers, and
    # the Davidson then reads the same block-action output everywhere.
    mf_r = copy.copy(mf)
    mf_r.mo_energy = np.asarray(mf.mo_energy, float).copy()
    mf_r.mo_coeff = np.asarray(mf.mo_coeff, float).copy()
    F_r = tuple(np.array(a, copy=True) for a in F)
    if gate.rank != 0:
        mf_r.mo_energy[nocc:] += 1e-3
        F_r[0][...] *= 1.0 + 1e-6
    om_p, _, _, info_p = solve_bse_isdf(mf_r, mol, nocc, nroots=3, probe=True,
                                        progress=False, factors=F_r,
                                        comm=gate.comm)
    d = np.abs(om_p - om_s).max() * HARTREE_TO_EV
    gate.check(d < 1e-9, 'BSE roots [explicit] with rank != 0 perturbed == serial',
               f'|d| = {d:.2e} eV')
    # BITWISE, because the fixture's factors and spectrum are rank 0's on
    # every rank already: overwriting the perturbation with rank 0's copy must
    # leave exactly the fixture's bits behind, not something near them.
    overwritten = (all(np.array_equal(a, b) for a, b in zip(F_r, F))
                   and np.array_equal(mf_r.mo_energy, mf.mo_energy))
    gate.check(overwritten, "the perturbed factors and spectrum were overwritten "
               "with rank 0's, bitwise",
               f'max|X - X_0| = {np.abs(F_r[0] - F[0]).max():.2e}, '
               f'max|eps - eps_0| = {np.abs(mf_r.mo_energy - mf.mo_energy).max():.2e} '
               f'after a {1e-6 * np.abs(F[0]).max():.2e} / 1e-3 perturbation')
    same_davidson(gate, 'BSE [explicit] with rank != 0 perturbed', om_p, info_p)


def grid_reduction_rows(gate, mf, mol, F):
    """The BSE block action's grid-length reduction hands each rank its own
    rows of X_o^T (Zt * P) and nothing longer, read off the communicator."""
    gate.section('the BSE block action\'s grid reduction')
    if gate.size == 1:
        gate.info('one rank: nothing is reduced')
        return
    nocc = mol.nelectron // 2
    npts, nmo = F[0].shape
    W = isdf_bse_factors(mf, mol, nocc, factors=F)[2]
    lr = LinearResponseSolver(np.asarray(mf.mo_energy, float),
                              spin_mode='restricted')
    z = np.random.default_rng(3).normal(size=(RECORDED_BATCH, nocc,
                                              nmo - nocc))
    log = []
    recorded = (RecordedSimulatedComm(gate.comm, log)
                if isinstance(gate.comm, SimulatedComm)
                else RecordedComm(gate.comm, log))
    act = isdf_block_action(lr, nocc, True, W, F, comm=recorded)[0]
    plain = isdf_block_action(lr, nocc, True, W, F, comm=gate.comm)[0]
    del log[:]
    A, B = act(z.copy())
    A0, B0 = plain(z.copy())
    r0, r1 = contiguous_block(npts, gate.rank, gate.size)
    scattered = [e for e in log if e[0] == 'reduce_scatter']
    long_sums = [e for e in log if e[0] == 'allreduce' and e[1] >= npts * nocc]
    own = (len(scattered) == RECORDED_BATCH and not long_sums
           and all(e[1:] == (npts * nocc, (r1 - r0) * nocc) for e in scattered))
    tiled = sum(gate.everyone(sum(e[2] for e in scattered)))
    gate.check(own and tiled == RECORDED_BATCH * npts * nocc
               and np.array_equal(A, A0) and np.array_equal(B, B0),
               "BSE block action [explicit]: each rank receives its rows of "
               "the grid reduction alone, and the same A and B",
               f'{len(scattered)} reduce-scatters of {npts * nocc} into '
               f'{(r1 - r0) * nocc} on rank {gate.rank}, {len(long_sums)} '
               f'grid-length all-reduces, {tiled} elements received over the '
               f'ranks against {RECORDED_BATCH} x M x n_occ')


def sliced_factors_routes(gate, mf, mol, F):
    """The factors held as grid-point slices: the window and the BSE on them.

    `separable_factors(sliced=True)` cuts each rank's rows from the same
    lockstepped fit as the fixture's, and every stage that reads an array
    whole gathers it once through `mpi_grid.allgather_rows` -- the row-counted
    Allgatherv this is the real-MPI gate of -- so the energies are the
    fixture's bitwise.
    """
    gate.section('sliced factors (SlicedFactors)')
    nocc = mol.nelectron // 2
    if gate.size == 1:
        gate.info('one rank: sliced=True returns the whole tuple')
        return
    Fs = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', sliced=True)
    r0, r1 = Fs.rows
    rows_ok = (np.array_equal(Fs.X_mo, F[0][r0:r1])
               and np.array_equal(Fs.D, F[1][r0:r1])
               and np.array_equal(Fs.X_ao, F[2][r0:r1])
               and np.array_equal(Fs.coords, F[3]))
    gate.check(rows_ok, "sliced factors [context]: this rank's rows of the "
               'fixture, bitwise', f'rows {Fs.rows} of {Fs.npts}')
    held = Fs.held_bytes()
    want = {'X_mo': (r1 - r0) * F[0].shape[1] * 8,
            'D': (r1 - r0) * F[1].shape[1] * 8,
            'X_ao': (r1 - r0) * F[2].shape[1] * 8, 'coords': F[3].nbytes}
    gate.check(held == want, 'sliced factors hold the rank\'s rows alone',
               f'{held}')
    window = np.array([nocc - 1, nocc])
    qp_w = solve_qp_energy_space_time(mf, mol, nocc, window, factors=F)
    qp_s = solve_qp_energy_space_time(mf, mol, nocc, window, factors=Fs,
                                      comm=gate.comm)
    n = gate.distinct(qp_s)
    gate.check(np.array_equal(qp_s, qp_w) and n == 1,
               f'GW window [explicit] on sliced factors over {gate.size} '
               'ranks == whole factors, bitwise, on every rank',
               f'max|d| {np.abs(qp_s - qp_w).max():.2e} Ha, {n} distinct')
    om_w, _, _, _ = solve_bse_isdf(mf, mol, nocc, nroots=3, probe=True,
                                   progress=False, factors=F)
    om_s, _, _, info_s = solve_bse_isdf(mf, mol, nocc, nroots=3, probe=True,
                                        progress=False, factors=Fs)
    gate.check(np.array_equal(om_s, om_w),
               f'BSE roots [context] on sliced factors == whole factors, bitwise',
               f'max|d| {np.abs(om_s - om_w).max() * HARTREE_TO_EV:.2e} eV')
    same_davidson(gate, 'BSE [context] on sliced factors', om_s, info_s)
    gathers = gate.everyone(dict(Fs.gathers))
    once = {'D': 4, 'X_o': 4, 'X_v': 2, 'X_ao': 2}
    gate.check(all(g == once for g in gathers),
               'each whole-array gather once per solve or sweep, every rank',
               f'{gathers[0]} on rank 0')


def grid_row_axes(eps, nocc):
    """The in-core driver's axes at ROW_NTAU: the chi0 grid with its
    omega = 0 passenger, the self-energy's tau points, the Pade points, the
    omega -> tau weights of W - I, the chi0 frequencies and mu."""
    e_min, e_max = eps[nocc] - eps[nocc - 1], eps[-1] - eps[0]
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    fp, fw = minimax_frequency_grid(ROW_NTAU, e_min, e_max)
    grid = TimeFrequencyGrid.minimax_split(ROW_NTAU, e_min, e_max,
                                           np.append(fp, 0.0),
                                           np.append(fw, 0.0))
    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    tau = 0.5 * minimax_time_grid(ROW_NTAU, *rS)[0]
    Ctw = minimax_transform_weights(COSINE_WT, tau, fp, *rW)[0]
    pade = gauss_legendre_grid(DEFAULT_NPADE, w0=1.0)[0]
    return dict(grid=grid, tau=tau, Ctw=Ctw, pade=pade, fp=fp, mu=mu)


def sweep_owners(k, ntau, nitems, size):
    """Every rank's items of tau point k, rank-ordered, in the order it adds
    them, when a (tau point, item) sweep goes over `size` ranks in
    `sweep_waves` windows (`wave_items`): the grid-row tiles of proj(tau_k),
    the block pairs of its branch sums."""
    k0 = k - k % size
    k1 = min(k0 + size, ntau)
    return [[j for kk, j in wave_items(k0, k1, nitems, r, size) if kk == k]
            for r in range(size)]


def proj_tile_addends(X_o, X_v, e_o, e_v, D, tau):
    """proj(tau) of the serial kernel at ROW_TILE_GB and its addends, one per
    grid-row tile in tile order: the kernel on that tile alone, which adds
    it onto zeros."""
    kw = dict(tile_memory_gb=ROW_TILE_GB)
    whole = polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau, **kw)
    return whole, [polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                                tiles=[tile], **kw)
                   for tile in polarizability_tiles(X_o.shape[0],
                                                    ROW_TILE_GB)]


def branch_pair_addends(pairs, Wk, tau):
    """The serial branch sums (Sigma^<, Sigma^>)_pp(tau) of `pairs`
    (`SigmaPairs`) on Wt(tau) = Wk, (2, nstates), and their addends, one per
    block pair in pair order: `SigmaPairs.add` of that pair alone onto
    zeros."""
    shape = (2, pairs.X_s.shape[1])
    whole = np.zeros(shape)
    pairs.add(whole, Wk, tau, range(len(pairs.pairs)))
    each = []
    for j in range(len(pairs.pairs)):
        one = np.zeros(shape)
        pairs.add(one, Wk, tau, [j])
        each.append(one)
    return whole, each


def recording(comm):
    """`comm` with its sums recorded (`sums`), simulated or mpi4py."""
    if isinstance(comm, SimulatedComm):
        return RecordedSimulatedComm(comm, [])
    return RecordedComm(comm, [])


def bounded_sum_checks(gate, name, item, verdicts, recorded):
    """The three parts of a reduced sum over every tau point, from this
    rank's `reduced_sum_verdict`s; `recorded`: the recorded communicator saw
    the sums the kernel makes, one per point or one for all."""
    ratios = gate.everyone(max((v.ratio for v in verdicts), default=np.inf))
    bounds = gate.everyone(max((v.bound_ulp for v in verdicts),
                               default=np.inf))
    gate.check(recorded and all(v.serial for v in verdicts),
               f'{name}: the serial {item} addends in {item} order are the '
               'serial sum at every tau point, bitwise')
    gate.check(recorded and all(v.partial for v in verdicts),
               f"{name}: this rank's partial of every tau point is its own "
               f"{item}s' serial addends in order, bitwise, zeros where it "
               'owns none', f'{len(verdicts)} tau points')
    gate.check(recorded and all(v.ratio <= 1 for v in verdicts),
               f"{name}: what this rank receives lies within the join bound "
               "of the ranks' partials at every element",
               f'at most {max(ratios):.2f} x the bound over the ranks, which '
               f'is at most {max(bounds):.2f} ulp of the largest element')


def grid_row_sums(gate, mf, mol, F, w0_route):
    """The grid-row route's two reduced sums on their rounding bound, and
    W(0) on them bitwise.

    proj(tau) through `chi0_frequency_rows` and the branch sums of Sigma
    through `self_energy_branch_sums` on this run's communicator, recorded:
    the serial kernel's addends -- one per grid-row tile, one per block pair
    -- in its order are its result, bitwise; this rank's partial of every
    tau point is its own items' addends in that order, bitwise; and what it
    receives lies within `join_bound` of the exact sum of the ranks'
    partials at every element, floored at one ulp of the serial sum's
    largest element (tests/reduction_bounds.py). The branch sums are those
    of the Wt(tau) this chi0 screens. W(0) is the inversion of chi0(0)
    folded from the received proj rows, and the route's W(0), bitwise.
    """
    gate.section('the grid-row GW sums on their rounding bound')
    if gate.size == 1:
        gate.info('one rank: nothing is reduced')
        return
    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    ax = grid_row_axes(eps, nocc)
    grid, mu = ax['grid'], ax['mu']
    X_mo, D, X_ao = F[0], F[1], F[2]
    naux = D.shape[1]
    r0, r1 = contiguous_block(naux, gate.rank, gate.size)
    X_o, X_v, e_o, e_v = split_branches(X_mo, eps, nocc, mu)[:4]
    rec = recording(gate.comm)
    chi0 = chi0_frequency_rows(X_mo, D, eps, nocc, grid, mu=mu, comm=rec,
                               tile_memory_gb=ROW_TILE_GB)
    fold, verdicts = np.zeros_like(chi0.rows), []
    for k, (tau, (_, sent, got)) in enumerate(zip(grid.tau_points,
                                                   rec.sums)):
        whole, adds = proj_tile_addends(X_o, X_v, e_o, e_v, D, tau)
        rows = got.reshape(r1 - r0, naux)
        verdicts.append(reduced_sum_verdict(
            whole, adds, sweep_owners(k, grid.ntau, len(adds), gate.size),
            gate.rank, sent.reshape(naux, naux), rows, rows=(r0, r1)))
        fold += grid.cosft_wt[:, k, None, None] * rows
    bounded_sum_checks(gate, 'proj(tau) [explicit]', 'tile', verdicts,
                       [s[0] for s in rec.sums]
                       == ['reduce_scatter'] * grid.ntau)
    gate.check(np.array_equal(chi0.rows, fold), "chi0's rows the serial "
               'update of the received proj rows, bitwise')
    owned, w0 = _dyson_owned(chi0, grid.nfreq - 1)
    c0 = np.concatenate(gate.everyone(chi0.rows[-1]))
    gate.check(np.array_equal(w0, np.linalg.inv(np.eye(naux) - c0)),
               'W(0) the inversion of chi0(0) folded from them, bitwise')
    gate.check(np.array_equal(w0_route, w0), "the route's W(0) this one, "
               'bitwise', f'{np.abs(w0_route - w0).max():.2e} apart')

    ntau = len(ax['tau'])
    Wt = screened_interaction_rows(owned, ax['Ctw'], naux, gate.comm)
    pairs = SigmaPairs(X_ao, D, mf.mo_coeff, eps, nocc, np.arange(len(eps)),
                       mu, block_memory_gb=ROW_TILE_GB)
    slabs = [slab.copy() for _, slab in Wt.gather_slices(range(ntau))]
    rec = recording(gate.comm)
    self_energy_branch_sums(pairs, ProjRows(Wt.rows, naux, rec), ax['tau'])
    recorded = [s[0] for s in rec.sums] == ['allreduce']
    verdicts = []
    if recorded:
        shape = (2, ntau, len(eps))
        sent, got = (a.reshape(shape) for a in rec.sums[0][1:])
        for k, tau in enumerate(ax['tau']):
            whole, adds = branch_pair_addends(pairs, slabs[k], tau)
            verdicts.append(reduced_sum_verdict(
                whole, adds, sweep_owners(k, ntau, len(adds), gate.size),
                gate.rank, sent[:, k], got[:, k]))
    bounded_sum_checks(gate, 'the branch sums of Sigma [explicit]',
                       'block pair', verdicts, recorded)


def planted_whole_slice(kernel):
    """`kernel` with the whole Pi(tau) = Go * Gv of its tau point formed
    beside it, (M, M): what the grid-row sweep exists not to hold."""
    def whole_slice(X_o, X_v, e_o, e_v, D, tau, *args, **kwargs):
        Pi = ((X_o * np.exp(e_o * tau)) @ X_o.T) * ((X_v * np.exp(-e_v * tau))
                                                    @ X_v.T)
        out = kernel(X_o, X_v, e_o, e_v, D, tau, *args, **kwargs)
        del Pi
        return out
    return whole_slice


def gw_grid_rows(gate, mf, mol, F):
    """The in-core GW window split over grid rows: what a rank holds at
    every line, the window against serial, and the route's two reduced sums
    and W(0) on them (`grid_row_sums`)."""
    gate.section('the GW window split over grid rows (_qp_grid_rows)')
    nocc = mol.nelectron // 2
    window = np.array([nocc - 1, nocc])
    kw = dict(ntau=ROW_NTAU, tile_gb=ROW_TILE_GB)

    def window_and_w0(factors, comm=None):
        extras = {}
        qp = solve_qp_energy_space_time(mf, mol, nocc, window, factors=factors,
                                        comm=comm, extras=extras, **kw)
        return qp, extras['w_static']

    ref = serial(window_and_w0, F)
    if gate.size == 1:
        gate.info(f'serial reference {ref[0] * HARTREE_TO_EV} eV')
        return
    moved = np.zeros(2)
    for seed in range(ROW_PERMUTATIONS):
        perm = np.random.default_rng(seed).permutation(F[0].shape[0])
        got = serial(window_and_w0, tuple(a[perm] for a in F))
        moved = np.maximum(moved, [np.abs(a - b).max()
                                   for a, b in zip(got, ref)])
    Fs = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', sliced=True)
    M, naux, nao = F[1].shape[0], F[1].shape[1], F[2].shape[1]
    census = HeldCensus(solve_qp_energy_space_time.__code__, M, naux, nao)
    qp, w0 = census.trace(window_and_w0, Fs, gate.comm)
    owned = len(partition(ROW_NTAU + 1, gate.rank, gate.size))
    flags = gate.everyone(census.flags)
    gate.check(not any(flags), 'GW window [explicit] on sliced factors: no '
               'whole (M, M) block, stack of (naux, naux) slices or AO stack '
               'bound at any line, every rank', f'{flags}')
    slices = gate.everyone((census.slices, owned))
    gate.check(all(n <= own + ROW_SPARE_SLICES for n, own in slices),
               'at most the own frequencies and ROW_SPARE_SLICES whole '
               '(naux, naux) slices at once, every rank',
               f'(slices, own frequencies) per rank {slices}')
    gate.info(f'rank 0 held at most {census.bytes / 1e6:.3f} MB of named '
              f'arrays; the serial W(0) alone is {naux * naux * 8 / 1e6:.3f} '
              'MB a slice')
    bar_qp = COMPOSED_GRAD_K * max(moved[0], QP_BISECTION_TOL)
    d_qp, d_w0 = np.abs(qp - ref[0]).max(), np.abs(w0 - ref[1]).max()
    gate.check(d_qp <= bar_qp, 'the window within the anchored bar of serial',
               f'{d_qp:.2e} of {bar_qp:.2e} Ha (relabelling {moved[0]:.2e})')
    gate.info(f'W(0) {d_w0:.2e} from serial, where relabelling the grid moves '
              f'it {moved[1]:.2e}: gated on the sums it inverts, below')
    n = gate.distinct(qp, w0)
    gate.check(n == 1, "the window and W(0) rank 0's on every rank, bitwise",
               f'{n} distinct of {gate.size}')

    real = ls_space_time.polarizability_projected_tau
    gate.everyone(None)                        # every rank's run is done
    ls_space_time.polarizability_projected_tau = planted_whole_slice(real)
    gate.everyone(None)                        # ...and every rank is planted
    try:
        planted = HeldCensus(solve_qp_energy_space_time.__code__, M, naux, nao)
        planted.trace(window_and_w0, Fs, gate.comm)
    finally:
        gate.everyone(None)                    # every planted run is done
        ls_space_time.polarizability_projected_tau = real
    found = gate.everyone(sorted(planted.flags.get('grid square', ())))
    gate.check(all((M, M) in f for f in found), 'the census finds a planted '
               'whole Pi(tau) slice on every rank', f'{found}')
    grid_row_sums(gate, mf, mol, F, w0)


def qp_set_and_rpa(gate, X_mo, D, eps, nocc, grid, nu, wt, mu):
    """The quasiparticle-set gradient and the dRPA energy with its adjoint."""
    gate.section('quasiparticle-set gradient and dRPA adjoint')
    states = [nocc - 2, nocc - 1, nocc, nocc + 1]
    weights = np.array([0.3, -1.1, 0.8, 0.45])
    # the explicit residue backend only: it is the route every checkout has.
    # BITWISE ROOTS, BY CONSTRUCTION. The roots read proj(tau) and wc and
    # nothing else reduced, and both reductions add exact zeros: a rank writes
    # only its own tau slots and its own frequency rows. The rows themselves
    # are cut out of the serial block's transform (`owned_frequency_blocks`),
    # because a GEMM that transforms only a rank's rows gives other bits on
    # OpenBLAS -- which failed this gate on a cluster at 8 ranks, 3 rows a rank.
    # The adjoints are partial sums over the partition and re-associate.
    route = 'explicit'
    ref = serial(qp_set_gradient, X_mo, D, eps, nocc, grid, nu, wt, states,
                 weights, mu=mu, residue_route=route)
    if gate.size > 1:
        got = qp_set_gradient(X_mo, D, eps, nocc, grid, nu, wt, states, weights,
                              mu=mu, residue_route=route, comm=gate.comm)
        worst = worst_rel(ref[1:], got[1:])
        d_roots = np.abs(np.asarray(ref[0]) - np.asarray(got[0])).max()
        gate.check(np.array_equal(ref[0], got[0]) and worst < 1e-11,
                   f'qp_set_gradient[{route}] [explicit] over {gate.size} ranks '
                   '== serial',
                   f'roots bitwise {np.array_equal(ref[0], got[0])} '
                   f'(max |d| {d_roots:.2e} Ha), adjoints rel {worst:.2e}')
    else:
        gate.info(f'serial reference [{route}] '
                  + ' '.join(f'{w * HARTREE_TO_EV:.5f}' for w in ref[0]) + ' eV')
    ref = serial(rpa_energy_and_adjoint, X_mo, D, eps, nocc, grid, mu=mu)
    if gate.size > 1:
        got = rpa_energy_and_adjoint(X_mo, D, eps, nocc, grid, mu=mu)
        worst = worst_rel(ref[1:4], got[1:4])
        gate.check(abs(ref[0] - got[0]) < 1e-11 * abs(ref[0]) and worst < 1e-11,
                   f'rpa_energy_and_adjoint [context] over {gate.size} ranks '
                   '== serial',
                   f'E_c {ref[0]:.12f} vs {got[0]:.12f}, adjoints rel {worst:.2e}')
    else:
        gate.info(f'serial reference E_c = {ref[0]:.12f} Ha')


def single_state_qp(gate, X_mo, D, eps, nocc, grid, nu, wt, mu):
    """The single-state quasiparticle gradient on both residue routes."""
    gate.section('single-state quasiparticle gradient')
    # The ADJOINTS are not bitwise across rank counts: the reverse pass's
    # projbar and Bp_bar are partials over the frequencies a rank owns, and an
    # all-reduce adds them in whatever order its tree picks. The root and Z
    # read only wc, the same zero-padded rows cut from the serial blocks as in
    # the set route above, and are expected bitwise on both routes; they are
    # still gated on the number, with the bitwise flag printed, since the set
    # route holds the bitwise line. The pole model is gated at `SOP_ANCHOR_K`
    # of its own measured ulp response. Z is bounded by one, so a relative bar
    # read as an absolute one on it is at least as tight.
    for route, path in (('explicit', 'context'), ('sop', 'explicit')):
        ref = serial(qp_gradient_space_time, X_mo, D, eps, nocc, grid, nu, wt,
                     nocc - 1, mu=mu, residue_route=route)
        if gate.size == 1:
            gate.info(f'serial reference [{route}] {ref[0] * HARTREE_TO_EV:.6f} '
                      f'eV, Z = {ref[1]:.6f}')
            continue
        comm = gate.comm if path == 'explicit' else None
        got = qp_gradient_space_time(X_mo, D, eps, nocc, grid, nu, wt, nocc - 1,
                                     mu=mu, residue_route=route, comm=comm)
        d_w, d_z = abs(ref[0] - got[0]), abs(ref[1] - got[1])
        worst = worst_rel(ref[2:], got[2:])
        if route == 'sop':
            bars = SOP_ANCHOR_K * sop_anchor(X_mo, D, eps, nocc, grid, nu, wt,
                                             nocc - 1, mu, ref)
        else:
            bars = np.array([1e-11, SIGMA_REL, SIGMA_REL])
        gate.check(np.all(np.array([d_w, d_z, worst]) <= bars),
                   f'qp_gradient_space_time[{route}] [{path}] over {gate.size} '
                   'ranks == serial',
                   f'root {d_w:.2e} of {bars[0]:.1e} Ha, Z {d_z:.2e} of '
                   f'{bars[1]:.1e}, adjoints rel {worst:.2e} of {bars[2]:.1e}; '
                   f'bitwise {ref[0] == got[0] and ref[1] == got[1]}')


def screening_and_sigma(gate, X_mo, D, eps, nocc, grid, mu):
    """Static W and the self-energy matrix and diagonal with their adjoints."""
    gate.section('static W and the self-energy from the gradient package')
    w_grid = static_grid(eps, nocc)
    ref_ss = serial(static_screening, X_mo, D, eps, nocc, w_grid)
    transforms = sigma_transforms(eps, nocc, grid.tau_points, grid.omega_points,
                                  grid.omega_points, mu=mu)
    sigma_states = [nocc - 1, nocc]
    sig_args = (X_mo, D, eps, nocc, grid, sigma_states, transforms, mu)
    proj = serial(polarizability_tau, X_mo, D, eps, nocc, grid, mu=mu)
    ref_wt = serial(screened_interaction_tau, proj, grid, transforms[0])
    ref_blk, cache_blk = serial(selfenergy_block, *sig_args)
    ref_dg, cache_dg = serial(selfenergy_diag, *sig_args)
    rng = np.random.default_rng(11)
    shape_b = (len(grid.omega_points), len(sigma_states), len(sigma_states))
    b_re, b_im = rng.normal(size=shape_b), rng.normal(size=shape_b)
    shape_d = (len(sigma_states), len(grid.omega_points))
    d_re, d_im = rng.normal(size=shape_d), rng.normal(size=shape_d)
    ref_blk_bar = serial(selfenergy_block_backward, b_re, b_im, *sig_args,
                         cache_blk)
    ref_dg_bar = serial(selfenergy_diag_backward, d_re, d_im, *sig_args, cache_dg)
    if gate.size == 1:
        gate.info(f'serial reference |W(0)| max {np.abs(ref_ss[1]).max():.6f}, '
                  f'|Sigma_pp| max {np.abs(ref_dg).max():.3e} Ha')
        return
    n = gate.size
    d = worst_rel(ref_ss, static_screening(X_mo, D, eps, nocc, w_grid,
                                           comm=gate.comm))
    gate.check(d < 1e-12, f'static_screening [explicit] over {n} ranks == serial',
               f'rel = {d:.2e}')
    d = worst_rel([ref_wt], [screened_interaction_tau(proj, grid, transforms[0])])
    gate.check(d < SIGMA_REL,
               f'screened_interaction_tau [context] over {n} ranks == serial',
               f'rel = {d:.2e}')
    blk, cache = selfenergy_block(*sig_args, comm=gate.comm)
    d = worst_rel([ref_blk.real, ref_blk.imag], [blk.real, blk.imag])
    gate.check(d < SIGMA_REL,
               f'selfenergy_block [explicit] over {n} ranks == serial',
               f'rel = {d:.2e}')
    d = worst_rel(ref_blk_bar, selfenergy_block_backward(b_re, b_im, *sig_args,
                                                         cache, comm=gate.comm))
    gate.check(d < SIGMA_REL,
               f'selfenergy_block_backward [explicit] over {n} ranks == serial',
               f'rel = {d:.2e}')
    dg, cache = selfenergy_diag(*sig_args)
    d = worst_rel([ref_dg.real, ref_dg.imag], [dg.real, dg.imag])
    gate.check(d < SIGMA_REL,
               f'selfenergy_diag [context] over {n} ranks == serial',
               f'rel = {d:.2e}')
    d = worst_rel(ref_dg_bar, selfenergy_diag_backward(d_re, d_im, *sig_args,
                                                       cache))
    gate.check(d < SIGMA_REL,
               f'selfenergy_diag_backward [context] over {n} ranks == serial',
               f'rel = {d:.2e}')


def chain_end_to_end(gate, mol):
    """The BSE@GW surface, run replicated in the region with no comm of its
    own; returns the self-converged mean field, rank 0's serial excitation
    force and the anchored gate, which the build-only factory reuses."""
    gate.section('the BSE@GW surface end to end')
    with distributed(None):
        mf_grad = chain_scf(mol)
        g_ex = ExcitedStateChain(mol, chain_scf, mf=mf_grad).excitation_gradient()[0]
        g_qp = ExcitedStateChain(mol, chain_scf,
                                 mf=mf_grad).quasiparticle_gradient(0)[0]
    # WHAT THE FORCES ARE COMPARED AT, measured before they are compared:
    # each rank re-associates its own serial forces by differentiating them
    # again on one BLAS thread. The gates take RANK 0's, because both numbers
    # they compare are rank 0's -- the serial reference below and the
    # distributed gradient, whose every kernel input is locked to rank 0's --
    # while the others are printed, since a node that reproduces itself less
    # well than rank 0 is worth seeing even though it does not set the gate.
    # Every rank's references are done first: simulated ranks share one BLAS
    # pool, which a one-thread region holds at one thread for all of them.
    gate.everyone(None)
    again = one_thread(lambda: (
        ExcitedStateChain(mol, chain_scf, mf=mf_grad).excitation_gradient()[0],
        ExcitedStateChain(mol, chain_scf,
                          mf=mf_grad).quasiparticle_gradient(0)[0]))
    reps = gate.everyone((0.0, 0.0) if again is None else
                         (float(np.abs(again[0] - g_ex).max()),
                          float(np.abs(again[1] - g_qp).max())))
    gate_bar = max(COMPOSED_GRAD_FLOOR, COMPOSED_GRAD_K * reps[0][0])
    qp_bar = max(ISDF_GRADIENT_FLOOR, COMPOSED_GRAD_K * reps[0][1])
    if gate.size == 1:
        gate.info(f'serial reference |dOmega/dR| max {np.abs(g_ex).max():.9f}, '
                  f'|deps^QP/dR| max {np.abs(g_qp).max():.9f} Ha/Bohr; one-thread '
                  f'repeat {reps[0][0]:.2e} / {reps[0][1]:.2e} Ha/Bohr')
        return mf_grad, g_ex, gate_bar
    # WHAT EACH RANK IS DIFFERENTIATING, before the forces are compared. The
    # chain freezes its own radii, points, frames and pair layout -- a kernel
    # locksteps its inputs, a chain decides its conventions -- so this is the
    # one thing the route gates above cannot see. The mean field is printed
    # beside it because it is the other per-rank input, and the two say which
    # of them a failing force below belongs to.
    tag = grid_tag(ExcitedStateChain(mol, chain_scf, mf=mf_grad), mol)
    tags = gate.everyone(tag + (digest(mf_grad.mo_energy, mf_grad.mo_coeff),))
    for r, (crd_t, radii_t, layout_t, mft) in enumerate(tags):
        gate.info(f'rank {r} chain grid: points {crd_t}, radii {radii_t}, '
                  f'layout {layout_t}, mean field {mft}')
    n_grids = len({t[:3] for t in tags})
    gate.check(n_grids == 1, 'every rank differentiates ONE grid [context]',
               f'{n_grids} distinct of {gate.size}')

    # The distributed surface is rank 0's inputs' surface, so the reference
    # is rank 0's serial gradient on every rank: a rank on another node
    # holds its own SCF, and against it the comparison would carry the
    # node-to-node input noise on top of the reduction noise.
    g_ex = broadcast(g_ex, gate.comm)
    g_qp = broadcast(g_qp, gate.comm)
    g_ex_d = ExcitedStateChain(mol, chain_scf, mf=mf_grad).excitation_gradient()[0]
    d_ex = np.abs(g_ex_d - g_ex).max()
    print(f'  [info] rank {gate.rank} excitation_gradient: |d| = {d_ex:.2e}, '
          f'one-thread repeat {reps[gate.rank][0]:.2e} (rank 0 '
          f'{reps[0][0]:.2e}), gate {gate_bar:.2e} Ha/Bohr', flush=True)
    gate.check(d_ex < gate_bar,
               f'excitation_gradient [context] over {gate.size} ranks == serial',
               f'|d| = {d_ex:.2e} vs gate {gate_bar:.2e} = '
               f'max({COMPOSED_GRAD_FLOOR:.1e}, {COMPOSED_GRAD_K} x '
               f'{reps[0][0]:.2e}) Ha/Bohr')
    n_ex = gate.distinct(g_ex_d)
    gate.info(f'excitation_gradient holds {n_ex} distinct value(s) over '
              f'{gate.size} ranks')
    # The same standard, on the quasiparticle force's own one-thread repeat.
    # Its floor is the single-chain ISDF_GRADIENT_FLOOR: without the Casida
    # step and its adjoint the repeat moves it an order less than the
    # excitation force.
    d_qp = np.abs(ExcitedStateChain(mol, chain_scf, mf=mf_grad)
                  .quasiparticle_gradient(0)[0] - g_qp).max()
    gate.check(d_qp < qp_bar,
               f'quasiparticle_gradient [context] over {gate.size} ranks == serial',
               f'|d| = {d_qp:.2e} vs gate {qp_bar:.2e} = max('
               f'{ISDF_GRADIENT_FLOOR:.1e}, {COMPOSED_GRAD_K} x '
               f'{reps[0][1]:.2e}) Ha/Bohr')
    return mf_grad, g_ex, gate_bar


def build_only_factory(gate, mol, mf_grad, g_ex, gate_bar):
    """The chain whose factory builds a mean field and leaves it unrun."""
    gate.section('the chain with a build-only factory')
    # NO `mf=` HERE, WHICH IS THE POINT. A mean field handed to a chain is the
    # user's and is taken as it is; the SCF this exercises is the one the chain
    # runs ITSELF through its factory, and with `chain_scf_unrun` that SCF is
    # converged over the region's ranks instead of on each of them. Its
    # reference is the self-converging factory's mean field, `mf_grad`, and
    # rank 0's serial force off it, `g_ex` -- the same two numbers the block
    # above compares against.
    runs = []
    chain_unrun = ExcitedStateChain(mol, recorded(chain_scf_unrun, runs))
    d_scf = abs(chain_unrun.mf0.e_tot - mf_grad.e_tot)
    n_mo = gate.distinct(chain_unrun.mf0.mo_coeff)
    g_unrun = chain_unrun.excitation_gradient()[0]
    d_ex_unrun = np.abs(g_unrun - g_ex).max()
    gate.info(f'build-only factory: dE = {d_scf:.2e} Ha, excitation_gradient '
              f'|d| = {d_ex_unrun:.2e} Ha/Bohr')
    gate.check(n_mo == 1, 'build-only factory [context] mo_coeff bitwise '
               f'identical across ranks, {gate.size} rank(s)',
               f'{n_mo} distinct of {gate.size}')
    if gate.size > 1:
        # The reductions re-associate the sums over the auxiliary index and
        # over the grid points, so the SCF lands on the same fixed point to
        # its convergence tolerance and no closer; the force it carries is then
        # compared at the SAME anchored gate the excitation force above uses,
        # since it is the same quantity off a mean field that moved by that
        # much.
        gate.check(d_scf < 1e-10, 'build-only factory mf0 == the self-converged '
                   f'one, {gate.size} rank(s)', f'|dE| = {d_scf:.2e} Ha')
        gate.check(d_ex_unrun < gate_bar,
                   f'build-only factory excitation_gradient over {gate.size} '
                   'ranks == serial',
                   f'|d| = {d_ex_unrun:.2e} vs gate {gate_bar:.2e} Ha/Bohr')
        return
    # Without a communicator `distributed_mean_field` IS `mf.kernel()`, the
    # call the factory would have made itself on the object it built: ONE run
    # of it with no initial guess, pyscf's own path with no distributed
    # handles left behind, and the chain holding that run's bits -- not close
    # to them. The run is compared with itself, not with `mf_grad`, because
    # two SCF runs of a threaded pyscf do not repeat their bits.
    # The FORCE is compared with the one off `mf_grad`, a SECOND SCF run, so
    # it is gated at the anchored bar the block above measured, not bitwise.
    mf0 = chain_unrun.mf0
    run = runs[0] if len(runs) == 1 else None
    plain = (run is not None and run['mf'] is mf0 and not run['args']
             and set(run['kwargs']) <= {'dm0'}
             and run['kwargs'].get('dm0') is None
             and not hasattr(mf0, '_distributed')
             and not hasattr(mf0, '_distributed_timings'))
    gate.check(plain and mf0.e_tot == run['e_tot'],
               'build-only factory mf0 is ONE plain mf.kernel() on the mean '
               "field it built, its energy that run's, bitwise",
               f'{len(runs)} kernel run(s)' + ('' if run is None else
               f", called with {run['args']} {run['kwargs']}, |dE| "
               f"{abs(mf0.e_tot - run['e_tot']):.2e} Ha"))
    gate.check(run is not None
               and np.array_equal(np.asarray(mf0.mo_coeff), run['mo_coeff']),
               "build-only factory orbitals are bitwise that run's")
    gate.check(d_ex_unrun < gate_bar,
               'build-only factory excitation_gradient == serial',
               f'|d| = {d_ex_unrun:.2e} vs gate {gate_bar:.2e} Ha/Bohr')


def sliced_chain_routes(gate, mol, gate_bar):
    """The state-pair surface on sliced factors against the whole layout.

    `RPABSESurface(sliced=True)` hands both halves one set of grid rows per
    geometry, cut from the products the whole layout forms; each kernel
    gathers what it reads whole through `mpi_grid.allgather_rows`, once per
    sweep or solve, and the Davidson runs on the rows. Both layouts are handed
    ONE reference and ONE displaced mean field, so the layout is all that
    differs between them -- and pyscf's own work inside each chain, its K
    builds and the mean field's force, which a threaded pyscf does not repeat
    bit for bit. What that work moves is measured by evaluating each layout
    `SLICED_SAMPLES` times on the same mean fields, the largest difference
    between two evaluations of one layout, and the force, the energy and the
    root are gated at `COMPOSED_GRAD_K` times it (the force at `gate_bar` at
    the least); every rank's force is rank 0's bitwise. Returns that spread of
    the force and the energy, None on one rank.
    """
    gate.section('the state-pair surface on sliced factors')
    if gate.size == 1:
        gate.info('one rank: sliced=True lays nothing out')
        return
    # ONE MEAN FIELD FOR BOTH LAYOUTS. Two SCF runs of a threaded pyscf do not
    # repeat their bits, and the force carries that difference: 1.97e-8
    # Ha/Bohr between the layouts on eight cluster nodes at 16 threads when
    # each converged its own.
    mf_ref = chain_scf(mol)
    here = gto.M(atom=WATER_DISPLACED, basis='cc-pvdz', verbose=0)
    mf_here = chain_scf(here)
    out = {'whole': [], 'sliced': []}
    for _ in range(SLICED_SAMPLES):
        for tag, sliced in (('whole', None), ('sliced', True)):
            surface = RPABSESurface(mol, chain_scf, spin='singlet', mf=mf_ref,
                                    solver='davidson', sliced=sliced)
            g, e, diags = surface.total_gradient(here, mf_here)       # context
            # the reference's rows, which live as long as its mean field
            ex = surface.excited
            ex._forward(ex.mol0, ex.mf0)
            out[tag].append((np.asarray(g), e, diags, ex))
    g_w, e_w, d_w, _ = out['whole'][0]
    g_s, e_s, d_s, ex = out['sliced'][0]
    # WHAT THE CHAINS' OWN PYSCF WORK MOVES, read off each layout's repeats
    # (under an emulated 16-thread race: 6e-9 to 1.4e-8 Ha/Bohr in the
    # force, 36 to 83 ulp in the root, the energy not at all). A repeat that
    # lands on the same bits resolves nothing below one ulp of the number,
    # so that is the least the energy and the root are allowed.
    again = [(a, b) for runs in out.values()
             for i, a in enumerate(runs) for b in runs[i + 1:]]
    rep_g = max(np.abs(a[0] - b[0]).max() for a, b in again)
    rep_e = max(abs(a[1] - b[1]) for a, b in again)
    rep_om = max(abs(a[2]['omega'] - b[2]['omega']) for a, b in again)
    bar_g = max(gate_bar, COMPOSED_GRAD_K * rep_g)
    bar_e = COMPOSED_GRAD_K * max(rep_e, np.spacing(abs(e_w)))
    bar_om = COMPOSED_GRAD_K * max(rep_om, np.spacing(abs(d_w['omega'])))
    d = np.abs(g_s - g_w).max()
    d_e = abs(e_s - e_w)
    d_om = abs(d_s['omega'] - d_w['omega'])
    gate.info(f'sliced == whole bitwise on one mean field: '
              f'{np.array_equal(g_s, g_w) and e_s == e_w}; a repeat moved '
              f'the force {rep_g:.2e} Ha/Bohr, the energy {rep_e:.2e} and '
              f'the root {rep_om:.2e} Ha')
    gate.check(d <= bar_g and d_e <= bar_e and d_om <= bar_om,
               f'state-pair force [context] on sliced factors over '
               f'{gate.size} ranks == whole factors on one mean field, within '
               'the anchored bar of a repeat',
               f'max|d| {d:.2e} of {bar_g:.2e} Ha/Bohr, dE {d_e:.2e} of '
               f'{bar_e:.2e}, dOmega {d_om:.2e} of {bar_om:.2e} Ha')
    n = gate.distinct(g_s, np.atleast_1d(e_s))
    gate.check(n == 1, 'state-pair force on sliced factors: every rank holds '
               "rank 0's, bitwise", f'{n} distinct of {gate.size}')
    rows = ex.factorization.cached_rows(ex.mol0, ex.mf0, None)
    held = None if rows is None else rows.held_bytes()
    want = None
    if rows is not None:
        r0, r1 = contiguous_block(rows.npts, gate.rank, gate.size)
        want = {'X_mo': (r1 - r0) * rows.nmo * 8,
                'D': (r1 - r0) * rows.naux * 8,
                'X_ao': (r1 - r0) * rows.nao * 8,
                'coords': rows.npts * 3 * 8}
    gate.check(rows is not None and held == want
               and len(ex.factorization._fit_cache) == 0,
               "sliced state-pair surface holds the rank's rows and no whole "
               'fit between evaluations', f'{held}')
    gathers = gate.everyone(d_s['factor_gathers'])
    gate.check(all(g == SLICED_CHAIN_GATHERS for g in gathers),
               'each whole-array gather of the composed force once per sweep '
               'or solve, every rank', f'{gathers[0]} on rank 0')
    return {'force': rep_g, 'energy': rep_e}


def reassociated_fit(mol, layout):
    """`fit_M_stable` with its sums over the test set -- the row norms, the
    Gram matrix and F Dt^T -- cut per mu shell and accumulated in reverse:
    one reordering of the whole fit's own sums, the anchor of the row fit."""
    mu = np.asarray(layout[0])
    ao_loc = mol.ao_loc_nr()
    shells = [np.flatnonzero((mu >= ao_loc[s]) & (mu < ao_loc[s + 1]))
              for s in reversed(range(mol.nbas))]
    shells = [c for c in shells if len(c)]

    def fit(D, F, regularization=DEFAULT_REGULARIZATION):
        blocks = shells + [np.arange(len(mu), D.shape[1])]
        s2 = np.zeros(D.shape[0])
        for c in blocks:
            s2 += np.einsum('kr,kr->k', D[:, c], D[:, c])
        s = np.sqrt(s2)
        d = 1.0 / np.where(s == 0.0, 1.0, s)
        Dt = D * d[:, None]
        G = np.zeros((D.shape[0], D.shape[0]))
        FD = np.zeros((F.shape[0], D.shape[0]))
        for c in blocks:
            G += Dt[:, c] @ Dt[:, c].T
            FD += F[:, c] @ Dt[:, c].T
        G[np.diag_indices_from(G)] += regularization
        cho = scipy.linalg.cho_factor(G, lower=True)
        return scipy.linalg.cho_solve(cho, FD.T).T * d[None, :]

    return fit


def reassociated_fit_adjoint(mol, gram):
    """`isdf_derivatives.fit_adjoint` with its sums over the test set -- the
    row norms, the Gram matrix, F Dt^T and the balancing's row sums -- cut
    per mu shell and accumulated in reverse: the whole adjoint's anchor."""
    mu = np.asarray(gram[0])
    ao_loc = mol.ao_loc_nr()
    shells = [np.flatnonzero((mu >= ao_loc[s]) & (mu < ao_loc[s + 1]))
              for s in reversed(range(mol.nbas))]
    shells = [c for c in shells if len(c)]

    def adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
        blocks = shells + [np.arange(len(mu), D.shape[1])]

        def rowsum(a, b):
            out = np.zeros(a.shape[0])
            for c in blocks:
                out += np.einsum('kr,kr->k', a[:, c], b[:, c])
            return out

        s = np.sqrt(rowsum(D, D))
        s = np.where(s == 0.0, 1.0, s)
        d = 1.0 / s
        Dt = D * d[:, None]
        G = np.zeros((D.shape[0], D.shape[0]))
        A = np.zeros((F.shape[0], D.shape[0]))
        for c in blocks:
            G += Dt[:, c] @ Dt[:, c].T
            A += F[:, c] @ Dt[:, c].T
        G[np.diag_indices_from(G)] += regularization
        cho = scipy.linalg.cho_factor(G, lower=True)
        B = scipy.linalg.cho_solve(cho, A.T).T
        B_bar = M_bar * d[None, :]
        d_bar = np.einsum('bk,bk->k', M_bar, B)
        A_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
        Y = scipy.linalg.cho_solve(cho, A.T @ B_bar)
        G_bar = -scipy.linalg.cho_solve(cho, Y.T).T
        F_bar = A_bar @ Dt
        Dt_bar = A_bar.T @ F + (G_bar + G_bar.T) @ Dt
        D_bar = d[:, None] * Dt_bar
        d_bar = d_bar + rowsum(Dt_bar, D)
        s_bar = -d_bar * d ** 2
        D_bar += (s_bar / s)[:, None] * D
        return D_bar, F_bar

    return adjoint


def whole_row_fit_branches(self, mol, mf, auxmol, crd, x_bar, d_bar, **extra):
    """`FactorChain.row_fit_branches` formed whole on every rank: the
    collocation adjoint and `dfactor_adjoint_gauges` over every product
    pair, the row fit's estimator, with no ledger."""
    g_coll = collocation_adjoint(mol, crd, x_bar @ mf.mo_coeff.T,
                                 self.pts_local, self.owner,
                                 frames=self.frames,
                                 with_frames=self.with_frames)
    g_fit = dfactor_adjoint_gauges(
        mol, auxmol, crd, [(d_bar, None)], self.layout, self.pts_local,
        self.owner, frames=self.frames, with_frames=self.with_frames,
        gram_layout=product_pairs(mol))
    return g_coll, g_fit, None


def gathered_rotation(x_mo, x_bar, block=None):
    """X_mo^T X_bar on X_mo gathered whole."""
    return whole_factor(x_mo, 'X_mo').T @ x_bar


def swapped(gate, swaps, run):
    """run() with each (owner, name, value) of `swaps` in place. Over
    thread-ranks the modules are shared: every rank reads the originals
    before any rank swaps, and none restores before every rank is done."""
    originals = [(owner, name, getattr(owner, name))
                 for owner, name, _ in swaps]
    gate.everyone(None)
    for owner, name, value in swaps:
        setattr(owner, name, value)
    try:
        return run()
    finally:
        gate.everyone(None)
        for owner, name, value in originals:
            setattr(owner, name, value)


def scan_assembly(chain, found, ledgers):
    """Trace every nuclear assembly of `chain`: each line of each frame under
    src/ scanned for a float array holding the whole grid by a factor's, the
    fit's or the test set's width, not sharing memory with the X_bar and
    D_bar handed in, into `found`; this rank's adjoint ledger into
    `ledgers`."""
    mol = chain.mol0
    npts = chain.M
    ngram = len(product_pairs(mol)[0])
    widths = {chain.mf0.mo_coeff.shape[1], mol.nao, chain.naux, npts,
              int((separable_ri._ao_l_labels(mol) <= 2).sum()),
              len(chain.layout[0]), ngram, ngram + chain.naux}
    assemble, branches = chain.nuclear_gradient, chain.row_fit_branches

    def own_ledger(*args, **kwargs):
        out = branches(*args, **kwargs)
        ledgers.append(out[2])
        return out

    def traced_force(*args, **kwargs):
        got = inspect.signature(type(chain).nuclear_gradient).bind(
            chain, *args, **kwargs).arguments
        inputs = [got['x_bar'], got['d_bar']]

        def local(frame, event, arg):
            if event in ('line', 'return'):
                values = list(frame.f_locals.values()) + (
                    [arg] if event == 'return' else [])
                for v in values:
                    items = (list(v.values()) if isinstance(v, dict) else
                             list(v) if isinstance(v, (list, tuple)) else [v])
                    for a in items:
                        if (isinstance(a, np.ndarray) and a.dtype.kind == 'f'
                                and a.ndim >= 2 and npts in a.shape
                                and any(n in widths for i, n in
                                        enumerate(a.shape)
                                        if i != a.shape.index(npts))
                                and not any(np.may_share_memory(a, e)
                                            for e in inputs)):
                            found.add(a.shape)
            return local

        def tracer(frame, event, arg):
            return (local if frame.f_code.co_filename.startswith(SRC)
                    else None)

        sys.settrace(tracer)
        try:
            return assemble(*args, **kwargs)
        finally:
            sys.settrace(None)

    chain.row_fit_branches = own_ledger
    chain.nuclear_gradient = traced_force


def ledger_faults(held, chain, rank, size, block):
    """What is wrong with this rank's adjoint ledger: every grid-indexed row
    array at this rank's tiles, the Gram tiles below the whole Gram matrix,
    the metric root on rank 0 alone. [] when all is so."""
    if not held:
        return ['no adjoint ledger: the whole adjoint']
    npts, nao, naux = chain.M, chain.mol0.nao, chain.naux
    rows = sum(min((t + 1) * block, npts) - t * block
               for t in partition(-(-npts // block), rank, size))
    faults = [f'{name} {held.get(name)}' for name, width in (
        ('X_rows', nao), ('X_bar_rows', nao), ('aux_rows', naux),
        ('MT_bar_rows', naux), ('Q_bar_rows', naux), ('Q_rows', naux),
        ('U_rows', naux), ('MT_rows', naux), ('P_bar_rows', naux))
        if held.get(name) != rows * width * 8]
    if not held.get('S_rows', 0) < npts * npts * 8:
        faults.append(f"S_rows {held.get('S_rows')}")
    if held.get('metric_root') != (2 * naux * naux * 8 if rank == 0 else 0):
        faults.append(f"metric_root {held.get('metric_root')}")
    return faults

def reachable(root):
    """(every float ndarray, every `SlicedFactors`) reachable from `root`
    through instance attributes and containers; communicators not entered."""
    seen, arrays, rows, stack = set(), [], [], [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type(sys), type, str, bytes,
                                               int, float, complex)):
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == 'f':
                arrays.append(obj)
            continue
        if 'Comm' in type(obj).__name__ or callable(obj):
            continue
        if isinstance(obj, SlicedFactors):
            rows.append(obj)
        if isinstance(obj, (dict, weakref.WeakKeyDictionary,
                            weakref.WeakValueDictionary)):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(getattr(obj, '__dict__', None), dict):
            stack.extend(vars(obj).values())
    return arrays, rows


def row_fit_chain_routes(gate, mol):
    """The state-pair surface on the row-distributed fit against the
    whole-fit sliced surface on the same ranks.

    `fit='rows'` builds each rank's rows with `separable_ri.fit_rows`, so no
    rank forms the fit whole; it is another realization than the whole fit's
    (and on water, where the pair screen keeps every pair, the same
    estimator), so the force is gated at an anchored bar: what the whole-fit
    surface's force moves when its fit's sums are reassociated, measured on
    these ranks. One rank is the row fit too. Every surface is handed ONE
    reference and ONE displaced mean field, so the bars measure fits and
    their adjoints, not two SCF runs of a threaded pyscf.
    """
    gate.section('the state-pair surface on the row-distributed fit')
    mf_ref = chain_scf(mol)
    here = gto.M(atom=WATER_DISPLACED, basis='cc-pvdz', verbose=0)
    mf_here = chain_scf(here)

    def force(scan=None, **kw):
        surface = RPABSESurface(mol, chain_scf, spin='singlet', mf=mf_ref,
                                solver='davidson',
                                bse_conv_tol=ROW_FIT_BSE_CONV_TOL, **kw)
        if scan is not None:
            scan(surface)
        g, e, _ = surface.total_gradient(here, mf_here)               # context
        return surface, np.asarray(g), e

    _, g_whole, e_whole = force(sliced=True)
    anchor = reassociated_fit(mol, FrozenFactorization(mol).layout)
    fit_now = factor_chain.fit_M_stable
    # Over thread-ranks the module is shared: no rank swaps the fit before
    # every rank is done with the whole one, nor back before every rank has
    # used the reassociated one.
    gate.everyone(None)
    factor_chain.fit_M_stable = anchor
    try:
        _, g_bar, _ = force(sliced=True)
    finally:
        gate.everyone(None)
        factor_chain.fit_M_stable = fit_now
    found, ledgers = set(), []

    def scanned(surface):
        for half in (surface.ground, surface.excited):
            scan_assembly(half, found, ledgers)

    rows_kw = dict(sliced=True, fit='rows', fit_block=ROW_FIT_BLOCK)
    surface, g_rows, e_rows = force(scan=scanned, **rows_kw)
    bar = np.linalg.norm(g_bar - g_whole)
    dist = np.linalg.norm(g_rows - g_whole)
    gate.check(dist <= FIT_REASSOCIATION_K * bar,
               f'row-fit state-pair force [context] over {gate.size} ranks '
               'within the anchored bar of the whole-fit sliced force',
               f'|d| {dist:.2e} = {dist / bar:.2f} x the reassociation '
               f'{bar:.2e} Ha/Bohr (bar {FIT_REASSOCIATION_K} x), dE '
               f'{abs(e_rows - e_whole):.2e} Ha')
    n = gate.distinct(g_rows, np.atleast_1d(e_rows))
    gate.check(n == 1, 'row-fit state-pair force: every rank holds rank 0\'s, '
               'bitwise', f'{n} distinct of {gate.size}')
    # what the force left reachable, read before the next surfaces are built
    npts = surface.ground.M
    widths = {surface.ground.mf0.mo_coeff.shape[1], mol.nao,
              surface.ground.naux, npts}
    arrays, rows = reachable(surface)
    whole = sorted({a.shape for a in arrays if a.ndim == 2
                    and ((a.shape[0] == npts and a.shape[1] in widths)
                         or (a.shape[1] == npts and a.shape[0] in widths))})
    r0, r1 = contiguous_block(npts, gate.rank, gate.size)
    faults = [] if rows else ['no rows reachable']
    for part in rows:
        held = getattr(part, 'fit_held', None) or {}
        if (part.rows != (r0, r1) or not held
                or held['D_rows'] != (r1 - r0) * part.naux * 8
                or not held['S_rows'] < npts * npts * 8):
            faults.append(f'rows {part.rows} ledger {held}')
    nrows = len(rows)
    del arrays, rows
    # the nuclear assembly: the row fit's adjoint in its tiles against the
    # whole adjoint of the same estimator, anchored on what reordering the
    # whole adjoint's sums over the test set moves that force
    gram = product_pairs(mol)
    whole_asm = [(FactorChain, 'row_fit_branches', whole_row_fit_branches),
                 (factor_chain, 'orbital_rotation_rows', gathered_rotation)]
    g_asm = swapped(gate, whole_asm, lambda: force(**rows_kw)[1])
    g_asm_bar = swapped(gate, whole_asm + [
        (isdf_derivatives, 'fit_M_stable', reassociated_fit(mol, gram)),
        (isdf_derivatives, 'fit_adjoint', reassociated_fit_adjoint(mol, gram))],
        lambda: force(**rows_kw)[1])
    bar = np.linalg.norm(g_asm_bar - g_asm)
    dist = np.linalg.norm(g_rows - g_asm)
    gate.check(dist <= FIT_REASSOCIATION_K * bar,
               f'row-fit state-pair force [context] over {gate.size} ranks: '
               'the tiled adjoint within the anchored bar of the whole one',
               f'|d| {dist:.2e} = {dist / bar:.2f} x the reassociation '
               f'{bar:.2e} Ha/Bohr (bar {FIT_REASSOCIATION_K} x)')
    if gate.size == 1:
        gate.info('one rank: the row fit\'s rows are the whole grid')
        return
    gate.check(not whole and not faults,
               'row-fit state-pair surface holds no whole factor or fit '
               "array, only this rank's rows of the row fit",
               f'whole {whole}, {nrows} rows objects, faults {faults}')
    adjoint_faults = [f for held, chain in zip(
        ledgers, (surface.ground, surface.excited))
        for f in ledger_faults(held, chain, gate.rank, gate.size,
                               ROW_FIT_BLOCK)]
    gate.check(not found and len(ledgers) == 2 and not adjoint_faults,
               'row-fit nuclear assemblies form no array of the grid by a '
               'factor\'s, the fit\'s or the test set\'s width, their '
               "ledgers at this rank's tiles",
               f'named {sorted(found)}, {len(ledgers)} ledgers, faults '
               f'{adjoint_faults}')


def grid_adjoint_routes(gate, mol, gate_bar, repeat):
    """The state-pair surface on sliced factors with the grid BSE adjoint
    against the same surface with the default one, on the same ranks and on
    ONE reference and ONE displaced mean field; `repeat` is what a repeat of
    the sliced surface moved its force and energy (`sliced_chain_routes`)."""
    gate.section('the grid BSE adjoint on sliced factors')
    if gate.size == 1:
        gate.info('one rank: sliced=True lays nothing out')
        return
    # ONE MEAN FIELD FOR BOTH ADJOINTS: two SCF runs of a threaded pyscf do
    # not repeat their bits, and the force and energy would carry them.
    mf_ref = chain_scf(mol)
    here = gto.M(atom=WATER_DISPLACED, basis='cc-pvdz', verbose=0)
    mf_here = chain_scf(here)
    out = {}
    for adjoint in ('explicit', 'grid'):
        surface = RPABSESurface(mol, chain_scf, spin='singlet', mf=mf_ref,
                                solver='davidson', sliced=True,
                                bse_adjoint=adjoint)
        ex = surface.excited
        nocc, nmo = ex.nocc, ex.mf0.mo_coeff.shape[1]
        three = {tuple(sorted((ex.naux, nocc, nmo - nocc))),
                 tuple(sorted((ex.naux, nocc, nocc)))}
        found = set()
        fold = ex._fold_to_nuclei

        def probed(pieces, *seeds, fold=fold, ex=ex, three=three,
                   found=found):
            # what the reverse pass holds beside the seeds: chain and forward
            found.update(a.shape for a in reachable((ex, pieces))[0]
                         if tuple(sorted(a.shape)) in three
                         or (a.ndim == 2 and tuple(sorted(a.shape)) ==
                             tuple(sorted((ex.naux, nocc * (nmo - nocc))))))
            return fold(pieces, *seeds)

        ex._fold_to_nuclei = probed
        seeds = []
        if adjoint == 'grid':
            probe_grid_seeds(ex, seeds)
        g, e, _ = surface.total_gradient(here, mf_here)               # context
        out[adjoint] = (np.asarray(g), e, sorted(found), seeds)
    g_d, e_d, found_d, _ = out['explicit']
    g_g, e_g, found_g, seeds = out['grid']
    # the forward pass is one code for both adjoints, so what moves the energy
    # between them is the chains' own pyscf work, which the repeat measures
    bar_g = max(gate_bar, COMPOSED_GRAD_K * repeat['force'])
    bar_e = COMPOSED_GRAD_K * max(repeat['energy'], np.spacing(abs(e_d)))
    d = np.abs(g_g - g_d).max()
    d_e = abs(e_g - e_d)
    n = gate.distinct(g_g, np.atleast_1d(e_g))
    gate.check(d <= bar_g and d_e <= bar_e and n == 1,
               f'state-pair force [context] with the grid BSE adjoint on '
               f'sliced factors over {gate.size} ranks == the default '
               'adjoint on one mean field within the anchored bar, every '
               "rank rank 0's",
               f'|d| {d:.2e} = {d / bar_g:.2f} of {bar_g:.2e} Ha/Bohr, dE '
               f'{d_e:.2e} of {bar_e:.2e} Ha (bitwise {e_g == e_d}), '
               f'{n} distinct')
    gate.check(found_g == [] and found_d != [],
               'no three-index block reachable from the grid-adjoint chain '
               "at its reverse call, where the default route's cache is",
               f'grid {found_g}, default {found_d}')
    n = gate.distinct(*[a for call in seeds for a in call['seeds']])
    gate.check(seeds and all(call['same'] for call in seeds) and n == 1,
               f'grid BSE adjoint seeds [context] over {gate.size} ranks == '
               'the serial kernel on the gathered factors (bitwise), every '
               "rank rank 0's",
               f'{len(seeds)} reverse call(s), {n} distinct')
    gate.check(seeds and all(call['whole'] == [] and call['boundary'] == 1
                             and call['gathers'] == {} for call in seeds),
               'no whole-grid array at any line of the grid adjoint on this '
               'rank outside its one named boundary gather of X_bar and '
               'D_bar, no whole-factor gather inside it',
               '; '.join(f"whole {call['whole']}, boundary "
                         f"{call['boundary']}, gathers {call['gathers']}"
                         for call in seeds))


def probe_grid_seeds(ex, log):
    """Wrap the grid chain's Casida-level adjoint: trace the kernel's frames
    line by line for arrays with the whole grid on an axis, outside the
    boundary gather `adjoints_at_the_boundary`, count the factor gathers it
    makes, and compare its seeds with the same kernel run serially on the
    gathered factors; one record per call into `log`."""
    seeds_of = ex._casida_seeds
    kernel = os.path.join(SRC, 'SingleReference', 'LinearResponse',
                          'isdf_bse_adjoint.py')
    boundary = 'adjoints_at_the_boundary'

    def probed(pieces, n, m=None):
        rows = ex._casida_args(pieces)[0]
        before = dict(rows.gathers)
        whole, entered = set(), []

        def local(frame, event, arg):
            if event in ('line', 'return'):
                for v in frame.f_locals.values():
                    items = (list(v.values()) if isinstance(v, dict) else
                             list(v) if isinstance(v, (list, tuple)) else [v])
                    for a in items:
                        if (isinstance(a, np.ndarray) and a.ndim
                                and rows.npts in a.shape):
                            whole.add((frame.f_code.co_name, a.shape))
            return local

        def tracer(frame, event, arg):
            if frame.f_code.co_name == boundary:
                entered.append(1)
                return None
            f, inside = frame, False
            while f is not None:
                if f.f_code.co_name == boundary:
                    return None
                inside = inside or f.f_code.co_filename == kernel
                f = f.f_back
            return local if inside else None

        sys.settrace(tracer)
        try:
            out = seeds_of(pieces, n, m)
        finally:
            sys.settrace(None)
        grew = {k: c - before.get(k, 0) for k, c in rows.gathers.items()
                if c != before.get(k, 0)}
        x, d, eq, w, no, _, xn, yn = ex._casida_args(pieces)
        X, D = whole_factor(x, 'X_mo'), whole_factor(d, 'D')
        kw = dict(spin=ex.spin, bse_tda=ex.bse_tda)
        with distributed(None):
            ref = (isdf_bse_backward(n, X, D, eq, w, no, xn, yn, **kw)
                   if m is None else
                   isdf_interstate_backward(m, n, X, D, eq, w, no, xn, yn,
                                            **kw))
        log.append(dict(whole=sorted(whole), boundary=len(entered),
                        gathers=grew, seeds=out,
                        same=all(np.array_equal(a, b)
                                 for a, b in zip(out, ref))))
        return out

    ex._casida_seeds = probed


def lockstep_counters(gate, stats):
    """What every rank moved through `lockstep` over the whole run.

    `lockstep` is a collective whose structure every rank agrees on before any
    buffer moves, so every rank makes the same calls with the same payload;
    serial regions count nothing. Rank 0 is the source and receives nothing,
    so its megabytes are what every other rank received.
    """
    gate.section('lockstep counters of the run')
    seen = gate.everyone((stats['calls'], stats['bytes']))
    for r, (calls, nbytes) in enumerate(seen):
        gate.info(f'rank {r}: {calls} lockstep calls, {nbytes / 1e6:.3f} MB')
    if gate.size == 1:
        return
    gate.check(seen[0][0] > 0 and all(s == seen[0] for s in seen),
               'every rank made the same lockstep calls with the same payload',
               f'{seen[0][0]} calls, {seen[0][1] / 1e6:.3f} MB on rank 0')


def audited_forward(gate, mol):
    """ONE chain forward under `distributed(comm, audit=True)`.

    On a mean field each rank converged ALONE, so the locksteps have something
    to absorb across nodes: `mismatched_calls` and `max_abs_diff` report how
    often a rank's input differed from rank 0's and by how much -- zero on one
    machine, where two processes repeat each other's bits. What must be zero
    everywhere is `disagreements`: every `agreement` a kernel makes compares
    digests of inputs and outputs that are identical by construction, so one
    disagreement is a kernel handing on something it never locked.
    """
    gate.section('one audited chain forward')
    if gate.size == 1:
        gate.info('nothing to audit on one rank')
        return
    mf_own = serial(chain_scf, mol)
    lockstep_stats(reset=True)
    with distributed(gate.comm, audit=True):
        om = ExcitedStateChain(mol, chain_scf, mf=mf_own).spectrum()
    stats = lockstep_stats(reset=True)
    keys = ('audited_calls', 'mismatched_calls', 'max_abs_diff',
            'agreement_calls', 'disagreements', 'first_disagreement')
    seen = gate.everyone({k: stats.get(k, 0 if k != 'first_disagreement'
                                       else None) for k in keys})
    for r, s in enumerate(seen):
        gate.info(f"rank {r}: {s['audited_calls']} audited locksteps, "
                  f"{s['mismatched_calls']} repaired (max |d| "
                  f"{s['max_abs_diff']:.2e}); {s['agreement_calls']} agreement "
                  f"checks, {s['disagreements']} disagreements"
                  + (f" (first: {s['first_disagreement']})"
                     if s['first_disagreement'] else ''))
    gate.info(f'audited spectrum {om[0] * HARTREE_TO_EV:.6f} eV (lowest root)')
    ran = all(s['audited_calls'] > 0 and s['agreement_calls'] > 0 for s in seen)
    gate.check(ran, 'the audit ran: every rank audited its locksteps and '
               'compared agreement digests',
               f"audited {[s['audited_calls'] for s in seen]}, agreement "
               f"{[s['agreement_calls'] for s in seen]}")
    worst = max(s['disagreements'] for s in seen)
    first = next((s['first_disagreement'] for s in seen
                  if s['first_disagreement']), None)
    gate.check(worst == 0, 'audited chain forward: no kernel input or output '
               'differs between ranks',
               f'{worst} disagreements' + (f', first at {first}' if first else ''))


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    lockstep_stats(reset=True)
    with distributed(comm):
        mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
        serial(mf.kernel)
        nocc = mol.nelectron // 2
        F = region_factors(gate, mf, mol)

        scf_routes(gate, mol)
        gw_low_memory(gate, mf, mol, F)
        static_w(gate, mf, mol, F)
        bse_routes(gate, mf, mol, F)
        grid_reduction_rows(gate, mf, mol, F)
        sliced_factors_routes(gate, mf, mol, F)
        gw_grid_rows(gate, mf, mol, F)

        X_mo, D = F[0], F[1]
        eps = np.asarray(mf.mo_energy, float)
        gap = eps[nocc] - eps[nocc - 1]
        nu, wt = gauss_legendre_grid(24, w0=gap)
        grid = TimeFrequencyGrid.minimax_split(8, 0.5 * gap, eps[-1] - eps[0],
                                               nu, wt, with_sine=False,
                                               with_inverse=False)
        mu = 0.5 * (eps[nocc - 1] + eps[nocc])
        qp_set_and_rpa(gate, X_mo, D, eps, nocc, grid, nu, wt, mu)
        single_state_qp(gate, X_mo, D, eps, nocc, grid, nu, wt, mu)
        screening_and_sigma(gate, X_mo, D, eps, nocc, grid, mu)

        mf_grad, g_ex, gate_bar = chain_end_to_end(gate, mol)
        build_only_factory(gate, mol, mf_grad, g_ex, gate_bar)
        repeat = sliced_chain_routes(gate, mol, gate_bar)
        row_fit_chain_routes(gate, mol)
        grid_adjoint_routes(gate, mol, gate_bar, repeat)

        lockstep_counters(gate, lockstep_stats(reset=True))
        audited_forward(gate, mol)
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
