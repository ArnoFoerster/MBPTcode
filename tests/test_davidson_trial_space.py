"""The Casida Davidson's trial space: the bound real_eig collapses it at, the
budget that bound is sized from, and the count of those collapses.

pyscf's `real_eig` sizes its trial space from its process-wide MAX_MEMORY, 103
pairs at the chlorophyllide dimer/cc-pVDZ, and a collapse keeps only the nroots
Ritz vectors, so a large pair space collapsed every few cycles and its top
roots stalled. The Davidson hands real_eig a space of DAVIDSON_SPACE_CYCLES
increments, never less than pyscf's own, within a budget:

  * inside an allocation, DAVIDSON_SPACE_FRACTION of this rank's max_memory
    less the block action's working set, counted on the pair rows the rank
    holds (32 bytes per row per trial pair), one increment kept clear as
    real_eig's own rule keeps it; rank 0's bound on every rank;
  * outside one, DAVIDSON_SPACE_GB of the whole holders, the parent's rule.

Gated here: the fallback is the parent's rule bit for bit; the budget is the
formula for a given max_memory, rank count and working set, and the cycles
bound binds first where the allocation has room; every rank holds rank 0's
bound at 2 to 16 ranks, ranks owning no tile included; benzene/cc-pVDZ, which
collapses under a forced cap along a path that depends on the rank count, is
held whole under a budget at 1, 2, 3 and 5 ranks -- the same cycles on every
rank and at every rank count, no collapse, its roots within the Davidson's
resolution of the capped solve's; the holders take 32 x rows x max_space on
each rank and, with the working set, never more than the budget; the action's
working set is what it allocates; and the allocation is read inside SLURM,
from the mean field's max_memory where a job sets it.

The model is a Casida problem of the RPA shape -- A = D + 2K, B = 2K with
K = V^T V -- whose roots are known densely.
"""
import io
import re
import tracemalloc
import types
import warnings

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.lib import logger

import src.SingleReference.LinearResponse.davidson as davidson
from src.Base.constants import (DAVIDSON_SPACE_CYCLES, DAVIDSON_SPACE_FRACTION,
                                DAVIDSON_SPACE_GB)
from src.Base.utils.memory import allocation_max_memory_mb
from src.Base.utils.mpi_grid import contiguous_block, run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import (bse_pair_diagonal,
                                                         isdf_bse_factors,
                                                         isdf_block_action,
                                                         solve_bse_isdf)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.trial_space import PairRows, pair_tiles
from tests.test_distributed_fit_mpi import relative, roots_resolution

NOCC, NVIR, NAUX, NROOTS, CONV_TOL = 8, 150, 40, 12, 1e-6
FORCED_SPACE = 60
DIMER_PAIRS = 354 * 1426
#: The chlorophyllide dimer/cc-pVTZ: n_occ 354, n_vir 3714.
DIMER_TZ_PAIRS = 354 * 3714
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; '
           'H 1.240 2.147 0; H -1.240 2.147 0; H -2.479 0 0; '
           'H -1.240 -2.147 0; H 1.240 -2.147 0')
#: Benzene, 12 roots, converged to where two paths' roots agree to round-off:
#: at 1e-5 the capped and the whole solve differ by 3.7e-11 of |roots|, their
#: convergence error, above the 9.7e-12 resolution.
BENZENE_ROOTS, BENZENE_TOL, BENZENE_TILE = 12, 1e-9, 64
#: The forced cap under which benzene collapses, and the trial pairs the
#: budget holds serially (the whole solve needs 450).
BENZENE_CAP, BENZENE_BUDGET = 150, 600
BENZENE_SIZES = [1, 2, 3, 5]
#: The model's 1200 pairs in 12 tiles: at 16 ranks four own none.
MODEL_TILE = 100
BOUND_SIZES = [2, 3, 8, 16]
#: Trial pairs the budget holds on rank 0 in the one-bound gate.
MODEL_BUDGET = 200
#: Trial vectors of one application in the allocation census.
CENSUS_BATCH = 7
#: Bytes of Python objects a traced allocation carries beyond its arrays.
TRACE_SLACK = 64 << 10
#: MB a node is taken to hold in the simulated SLURM allocation, over two
#: tasks: small enough that the model's budget binds.
SLURM_NODE_MB, SLURM_TASKS = 40, 2


@pytest.fixture()
def model():
    """(apply_AB, diag, dense roots) of a model Casida problem."""
    rng = np.random.default_rng(11)
    eps = np.concatenate([-np.sort(rng.uniform(0.3, 2.0, NOCC))[::-1],
                          np.sort(rng.uniform(0.05, 3.0, NVIR))])
    diag = bse_pair_diagonal(eps, NOCC)
    n_ov = diag.size
    V = rng.standard_normal((NAUX, n_ov)) / np.sqrt(n_ov) * 0.5

    def apply_AB(z):
        f = z.reshape(len(z), -1)
        v = (2 * (f @ V.T) @ V).reshape(z.shape)
        return diag[None] * z + v, v

    # RPA's screened diagonal is d itself, as the production actions return
    # it; the default preconditioner is now the screened diagonal, which a
    # block action without one refuses.
    apply_AB.screened_diagonal = lambda: diag

    sqrt_d = np.sqrt(diag.ravel())
    m = sqrt_d[:, None] * (np.diag(diag.ravel()) + 4 * V.T @ V) * sqrt_d[None, :]
    return apply_AB, diag, np.sqrt(np.linalg.eigvalsh(m))


def logged_real_eig(monkeypatch):
    """Patch real_eig to log at debug level; returns the log buffer."""
    buf = io.StringIO()
    real_eig = davidson.real_eig

    def with_log(*args, **kwargs):
        kwargs['verbose'] = logger.Logger(buf, logger.DEBUG)
        return real_eig(*args, **kwargs)

    monkeypatch.setattr(davidson, 'real_eig', with_log)
    return buf


def solve(model, timings):
    apply_AB, diag, _ = model
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        return davidson._run_davidson(apply_AB, diag, NROOTS, CONV_TOL, 100,
                                      None, timings=timings)


def bound(nroots, n_pair, rows=None, max_memory=None, working=0):
    """The trial pairs real_eig holds for the MB `_trial_space_memory` hands it."""
    return davidson._real_eig_space(
        davidson._trial_space_memory(nroots, n_pair, rows, max_memory,
                                     working), nroots, n_pair)[1]


def parent_memory(nroots, n_pair):
    """The MB the parent handed real_eig: DAVIDSON_SPACE_GB of whole holders."""
    space_inc = davidson._real_eig_space(davidson.param.MAX_MEMORY, nroots,
                                         n_pair)[0]
    target = min(DAVIDSON_SPACE_CYCLES * space_inc,
                 int(DAVIDSON_SPACE_GB * 1e9 / (32 * n_pair)))
    return max(davidson.param.MAX_MEMORY,
               (target + space_inc + 0.5) * 2 * 32 * n_pair / 1e6)


def formula(nroots, n_pair, rows, max_memory, working):
    """(max_space, fits, cycles): the budget's rule spelled out -- this rank's
    rows of the holders and one increment within DAVIDSON_SPACE_FRACTION of
    max_memory less the working set, DAVIDSON_SPACE_CYCLES increments at
    most, pyscf's own bound and 4 nroots at least."""
    space_inc, own = davidson._real_eig_space(davidson.param.MAX_MEMORY,
                                              nroots, n_pair)
    fits = int((DAVIDSON_SPACE_FRACTION * max_memory * 1e6 - working)
               / (32 * rows) - space_inc)
    cycles = DAVIDSON_SPACE_CYCLES * space_inc
    return min(max(own, min(cycles, fits), 4 * nroots), n_pair), fits, cycles


def first_rows(n_pair, size, tile=None):
    """The pair rows rank 0 owns over `size` ranks, the most any rank owns."""
    tiles = pair_tiles(n_pair, tile)
    t0, t1 = contiguous_block(len(tiles), 0, size)
    return tiles[t1 - 1][1] - tiles[t0][0]


def memory_for(pairs, n_pair, rows, working, nroots):
    """The max_memory MB whose budget holds `pairs` trial pairs of `rows`,
    half a pair clear of the rounding."""
    space_inc = davidson._real_eig_space(davidson.param.MAX_MEMORY, nroots,
                                         n_pair)[0]
    return ((pairs + space_inc + 0.5) * 32 * rows + working) / (
        DAVIDSON_SPACE_FRACTION * 1e6)


def _system(geometry):
    mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(mol=mol, mf=mf, nocc=nocc, factors=factors, W_aux=W_aux,
                eps=np.asarray(mf.mo_energy, float))


def action(s, comm=None):
    """This rank's ISDF BSE@HF action on its own copies of the inputs."""
    lr = LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')
    return isdf_block_action(lr, s['nocc'], True, s['W_aux'].copy(),
                             tuple(np.array(a) for a in s['factors']),
                             comm=comm)


def guess_width(diag, nroots, n_pair):
    """The widest batch the Davidson applies: its guess or one increment."""
    return max(len(davidson._guess_indices(diag, nroots)),
               davidson._real_eig_space(davidson.param.MAX_MEMORY, nroots,
                                        n_pair)[0])


@pytest.fixture(scope='module')
def benzene():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _system(BENZENE)


@pytest.fixture(scope='module')
def benzene_runs(benzene):
    """The capped serial solve and the budgeted ones at BENZENE_SIZES, every
    rank's (omega, timings, pair rows): pyscf's floor and the fallback forced
    down to BENZENE_CAP pairs, and a max_memory whose budget holds
    BENZENE_BUDGET pairs serially.

    Every solve divides by the bare d, asked for: the default preconditioner
    is now the screened diagonal, one reduction over the ranks' grid rows
    whose last bits move with the rank count, and with it the budgeted solves
    took 21 cycles at some rank counts and 22 at others; the gates below hold
    the trial space, not the preconditioner."""
    with pytest.MonkeyPatch.context() as mp:
        act, diag = action(benzene)
        n_pair = diag.size
        mp.setattr(davidson.param, 'MAX_MEMORY', 1)
        mp.setattr(davidson, 'DAVIDSON_SPACE_GB',
                   (BENZENE_CAP + 0.5) * 32 * n_pair / 1e9)
        space_inc = davidson._real_eig_space(1, BENZENE_ROOTS, n_pair)[0]
        working = davidson._action_working_bytes(
            act, guess_width(diag, BENZENE_ROOTS, n_pair), n_pair, True)
        mm = memory_for(BENZENE_BUDGET, n_pair, n_pair, working,
                        BENZENE_ROOTS)

        def one_rank(comm, max_memory):
            act, diag = action(benzene, comm)
            t = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                om = davidson._run_davidson(
                    act, diag, BENZENE_ROOTS, BENZENE_TOL, 100, None,
                    comm=comm, timings=t, pair_tile=BENZENE_TILE,
                    max_memory=max_memory, preconditioner='bare')[0]
            rows = (PairRows(n_pair, comm, BENZENE_TILE).rows if comm
                    is not None else (0, n_pair))
            return om, t, rows[1] - rows[0]

        capped = one_rank(None, None)
        budgeted = {size: ([one_rank(None, mm)] if size == 1 else
                           run_simulated(lambda c: one_rank(c, mm), size))
                    for size in BENZENE_SIZES}
    return dict(n_pair=n_pair, max_memory=mm, space_inc=space_inc,
                capped=capped, budgeted=budgeted)


@pytest.mark.parametrize('n_pair', [40, 95, 9353, 50000, DIMER_PAIRS, 10**7])
def test_the_bound_is_never_below_pyscfs_own(n_pair):
    """Raised to the cycles-or-memory target where pyscf's own is smaller,
    and pyscf's own, bit for bit, everywhere else."""
    own = davidson._real_eig_space(davidson.param.MAX_MEMORY, NROOTS, n_pair)
    space_inc = own[0]
    target = min(DAVIDSON_SPACE_CYCLES * space_inc,
                 int(DAVIDSON_SPACE_GB * 1e9 / (32 * n_pair)))
    assert bound(NROOTS, n_pair) == min(max(own[1], target, 4 * NROOTS),
                                        n_pair)
    assert bound(NROOTS, n_pair) >= own[1]
    assert bound(NROOTS, n_pair, 1, 1.0, 1e12) >= own[1]


def test_the_dimer_pair_space_is_no_longer_collapsed_every_few_cycles():
    """At the pair space where pyscf's own bound is a handful of cycles, the
    fallback is DAVIDSON_SPACE_GB's worth of trial pairs."""
    own = davidson._real_eig_space(davidson.param.MAX_MEMORY, NROOTS,
                                   DIMER_PAIRS)
    assert own[1] < 10 * NROOTS
    assert bound(NROOTS, DIMER_PAIRS) == int(DAVIDSON_SPACE_GB * 1e9
                                             / (32 * DIMER_PAIRS))


@pytest.mark.parametrize('nroots', [3, 12, 20, 40])
@pytest.mark.parametrize('n_pair', [40, 95, 1953, 9353, 61275, 100000,
                                    DIMER_PAIRS, DIMER_TZ_PAIRS, 10**7])
def test_the_fallback_is_the_parents_rule(n_pair, nroots):
    """No allocation: the MB real_eig is handed are the parent's to the bit,
    whatever rows or working set are passed beside it -- a single machine's
    solves run as they did."""
    parent = parent_memory(nroots, n_pair)
    assert davidson._trial_space_memory(nroots, n_pair) == parent
    assert davidson._trial_space_memory(nroots, n_pair, 17, None,
                                        1e12) == parent


@pytest.mark.parametrize('n_pair,nroots,max_memory,size,working,binds', [
    (DIMER_TZ_PAIRS, 12, 72000, 8, 14.52e9, 'cycles'),
    (DIMER_TZ_PAIRS, 12, 72000, 64, 5.53e9, 'cycles'),
    (DIMER_TZ_PAIRS, 12, 35000, 8, 14.52e9, 'budget'),
    (DIMER_TZ_PAIRS, 12, 72000, 1, 60.75e9, 'own'),
    (DIMER_TZ_PAIRS, 30, 20000, 16, 6.0e9, 'budget'),
    (500000, 20, 6000, 3, 0.5e9, 'budget'),
    (500000, 20, 60000, 3, 0.5e9, 'cycles'),
    (9353, 12, 100, 2, 1e6, 'own')])
def test_the_budget_is_the_formula(n_pair, nroots, max_memory, size, working,
                                   binds):
    """Given max_memory, the rank count and the working set, the bound is
    the formula's on rank 0's rows: the DAVIDSON_SPACE_CYCLES bound where
    the budget has room for more (the dimer/cc-pVTZ at 72 GB from 8 ranks
    up), the budget where it has not, and pyscf's own where nothing fits
    (the dimer on one rank) or pyscf's is larger."""
    rows = first_rows(n_pair, size) if size > 1 else n_pair
    expected, fits, cycles = formula(nroots, n_pair, rows, max_memory,
                                     working)
    got = bound(nroots, n_pair, rows, max_memory, working)
    own = davidson._real_eig_space(davidson.param.MAX_MEMORY, nroots,
                                   n_pair)[1]
    assert got == expected
    if binds == 'cycles':
        assert got == cycles < fits
    elif binds == 'budget':
        assert max(own, 4 * nroots) < got == fits < cycles
    else:
        assert fits < got == max(own, 4 * nroots)


def test_more_ranks_hold_no_fewer_trial_pairs():
    """Rows fall with the rank count, so the same allocation and working set
    hold as many trial pairs or more, up to the cycles bound."""
    space_inc = davidson._real_eig_space(davidson.param.MAX_MEMORY, 12,
                                         DIMER_TZ_PAIRS)[0]
    got = [bound(12, DIMER_TZ_PAIRS, first_rows(DIMER_TZ_PAIRS, size),
                 30000, 8e9) for size in (2, 4, 8, 16, 32, 64)]
    assert got == sorted(got)
    assert got[0] < got[-1] == DAVIDSON_SPACE_CYCLES * space_inc


@pytest.mark.parametrize('size', BOUND_SIZES)
def test_every_rank_holds_rank_zeros_bound(model, monkeypatch, size):
    """Each rank given its own allocation, rank r (1 + r) times rank 0's:
    every rank, those owning no tile included, reports rank 0's bound and
    rank 0's cycle count, gathered from every rank, though the bound each
    would size alone differs; each holds its own rows of it."""
    monkeypatch.setattr(davidson.param, 'MAX_MEMORY', 1)
    apply_AB, diag, _ = model
    n_pair = diag.size
    working = davidson._action_working_bytes(
        apply_AB, guess_width(diag, NROOTS, n_pair), n_pair, False)
    rows0 = first_rows(n_pair, size, MODEL_TILE)
    mm = memory_for(MODEL_BUDGET, n_pair, rows0, working, NROOTS)

    def one_rank(comm):
        rank = comm.Get_rank()
        t = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            davidson._run_davidson(apply_AB, diag, NROOTS, CONV_TOL, 100,
                                   None, comm=comm, timings=t,
                                   pair_tile=MODEL_TILE,
                                   max_memory=mm * (1 + rank))
        p0, p1 = PairRows(n_pair, comm, MODEL_TILE).rows
        alone = bound(NROOTS, n_pair, p1 - p0, mm * (1 + rank), working)
        return t, p1 - p0, alone

    out = run_simulated(one_rank, size)
    assert out[0][0]['davidson_max_space'] == MODEL_BUDGET
    assert {t['davidson_max_space'] for t, _, _ in out} == {MODEL_BUDGET}
    assert len({t['davidson_iterations'] for t, _, _ in out}) == 1
    assert len({t['davidson_collapses'] for t, _, _ in out}) == 1
    assert any(alone != MODEL_BUDGET for _, _, alone in out)
    for t, rows, _ in out:
        assert t['davidson_trial_space_mb'] * 1e6 == 32 * rows * MODEL_BUDGET
    assert any(rows == 0 for _, rows, _ in out) == (size == 16)


def test_a_collapsing_solve_is_held_whole_at_every_rank_count(benzene_runs):
    """Benzene collapses under the forced cap; under the budget it is held
    whole at every rank count: no collapse, its subspace within the bound,
    and the same cycles and trial vectors on every rank of every count."""
    om_c, t_c, _ = benzene_runs['capped']
    assert t_c['davidson_max_space'] == BENZENE_CAP
    assert t_c['davidson_collapses'] > 0
    assert t_c['davidson_subspace_max'] <= BENZENE_CAP
    cycles, vectors = set(), set()
    for size, out in benzene_runs['budgeted'].items():
        assert len(out) == size
        for om, t, _ in out:
            assert t['davidson_collapses'] == 0
            assert t['davidson_subspace_max'] <= t['davidson_max_space']
            assert t['davidson_max_space'] >= BENZENE_BUDGET
            cycles.add(t['davidson_iterations'])
            vectors.add(t['davidson_vectors_applied'])
        assert len({t['davidson_max_space'] for _, t, _ in out}) == 1
    assert len(cycles) == 1 and len(vectors) == 1
    assert cycles.pop() < t_c['davidson_iterations']
    assert benzene_runs['budgeted'][1][0][1]['davidson_max_space'] == \
        BENZENE_BUDGET


def test_the_whole_solve_sits_on_the_capped_roots(benzene, benzene_runs):
    """The budgeted roots, every rank's rank 0's, within the Davidson's
    resolution of the capped solve's at every rank count."""
    om_c = benzene_runs['capped'][0]
    bar = roots_resolution(benzene['eps'], benzene['nocc'], om_c)
    for size, out in benzene_runs['budgeted'].items():
        om = out[0][0]
        assert all(np.array_equal(o, om) for o, _, _ in out)
        assert relative(om, om_c) <= bar, (size, relative(om, om_c), bar)


def test_the_holders_and_the_working_set_keep_to_the_budget(benzene_runs):
    """On every rank: the holders take 32 x its rows x max_space, the ranks'
    together one rank's whole; the reported budget is DAVIDSON_SPACE_FRACTION
    of max_memory less the working set; and the holders with the increment
    kept clear, beside the working set, never exceed it."""
    n_pair, mm = benzene_runs['n_pair'], benzene_runs['max_memory']
    space_inc = benzene_runs['space_inc']
    for size, out in benzene_runs['budgeted'].items():
        max_space = out[0][1]['davidson_max_space']
        total = 0
        for _, t, rows in out:
            held = t['davidson_trial_space_mb'] * 1e6
            working = t['davidson_action_working_mb'] * 1e6
            budget = t['davidson_space_budget_mb'] * 1e6
            assert held == 32 * rows * max_space
            assert budget == pytest.approx(
                DAVIDSON_SPACE_FRACTION * mm * 1e6 - working, rel=1e-12)
            assert held + 32 * rows * space_inc + working <= (
                DAVIDSON_SPACE_FRACTION * mm * 1e6)
            total += held
        assert total == 32 * n_pair * max_space


def test_the_working_set_is_what_the_action_allocates(benzene):
    """Traced: the arrays the serial action keeps are its `held_bytes`, and
    one application of CENSUS_BATCH vectors allocates at most its
    `vector_bytes` a vector and its `scratch_bytes`, at least its output."""
    lr = LinearResponseSolver(benzene['eps'].copy(), spin_mode='restricted')
    factors = tuple(np.array(a) for a in benzene['factors'])
    W_aux = benzene['W_aux'].copy()
    tracemalloc.start()
    try:
        c0 = tracemalloc.get_traced_memory()[0]
        act, diag = isdf_block_action(lr, benzene['nocc'], True, W_aux,
                                      factors)
        kept = tracemalloc.get_traced_memory()[0] - c0
        z = np.random.default_rng(3).standard_normal(
            (CENSUS_BATCH,) + diag.shape)
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = act(z)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert act.held_bytes <= kept <= act.held_bytes + diag.nbytes + TRACE_SLACK
    assert sum(o.nbytes for o in out) <= peak
    assert peak <= (CENSUS_BATCH * act.vector_bytes + act.scratch_bytes
                    + TRACE_SLACK)


def slurm_job(monkeypatch):
    """A simulated SLURM job of SLURM_TASKS tasks on a SLURM_NODE_MB node."""
    for key in ('SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK',
                'SLURM_TASKS_PER_NODE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', str(SLURM_NODE_MB))
    monkeypatch.setenv('SLURM_NTASKS_PER_NODE', str(SLURM_TASKS))


def test_the_allocation_sizes_the_space_inside_slurm(model, monkeypatch):
    """Inside a SLURM job with no max_memory handed over, the budget is the
    allocation's share; outside one the fallback; and the mean field's
    max_memory is the job's only inside."""
    monkeypatch.setattr(davidson.param, 'MAX_MEMORY', 1)
    apply_AB, diag, _ = model
    n_pair = diag.size
    for key in ('SLURM_JOB_ID', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU'):
        monkeypatch.delenv(key, raising=False)
    outside = {}
    davidson._run_davidson(apply_AB, diag, NROOTS, CONV_TOL, 100, None,
                           timings=outside)
    assert outside['davidson_max_space'] == bound(NROOTS, n_pair)
    assert outside['davidson_space_budget_mb'] == DAVIDSON_SPACE_GB * 1e3
    assert davidson._job_max_memory(types.SimpleNamespace(
        max_memory=12345)) is None
    slurm_job(monkeypatch)
    share = allocation_max_memory_mb()
    assert share == int(SLURM_NODE_MB / SLURM_TASKS * 0.6)
    inside = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        davidson._run_davidson(apply_AB, diag, NROOTS, CONV_TOL, 100, None,
                               timings=inside)
    working = inside['davidson_action_working_mb'] * 1e6
    expected, fits, _ = formula(NROOTS, n_pair, n_pair, share, working)
    assert inside['davidson_max_space'] == expected == fits
    assert expected != outside['davidson_max_space']
    assert inside['davidson_space_budget_mb'] * 1e6 == pytest.approx(
        DAVIDSON_SPACE_FRACTION * share * 1e6 - working, rel=1e-12)
    assert davidson._job_max_memory(types.SimpleNamespace(
        max_memory=12345)) == 12345.0


def test_the_mean_fields_max_memory_sizes_the_bse(monkeypatch):
    """The knob a job sets: water's BSE inside a SLURM job budgets against
    the mean field's max_memory, not the allocation's own share."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = _system(WATER)
        slurm_job(monkeypatch)
        s['mf'].max_memory = 777.0
        _, _, _, info = solve_bse_isdf(s['mf'], s['mol'], s['nocc'],
                                       nroots=3, qp=False, probe=False,
                                       factors=s['factors'])
    t = info['timings']
    assert t['davidson_space_budget_mb'] * 1e6 == pytest.approx(
        DAVIDSON_SPACE_FRACTION * 777.0e6
        - t['davidson_action_working_mb'] * 1e6, rel=1e-12)


def test_collapses_are_counted_as_real_eig_logs_them(model, monkeypatch):
    """A bound forced small: the count and the largest subspace are the ones
    real_eig's own log shows, and the roots are the dense ones."""
    monkeypatch.setattr(davidson.param, 'MAX_MEMORY', 1)
    monkeypatch.setattr(davidson, 'DAVIDSON_SPACE_GB',
                        (FORCED_SPACE + 0.5) * 32 * model[1].size / 1e9)
    buf = logged_real_eig(monkeypatch)
    t = {}
    omega, _, _ = solve(model, t)
    m1 = [int(m.group(1)) for m in
          re.finditer(r'real_lr_eig \d+ (\d+)', buf.getvalue())]
    logged = sum(1 for a, b in zip(m1, m1[1:]) if b < a)
    assert t['davidson_max_space'] == FORCED_SPACE
    assert t['davidson_collapses'] == logged > 0
    assert max(m1) <= t['davidson_subspace_max'] <= FORCED_SPACE
    assert np.abs(omega - model[2][:NROOTS]).max() <= CONV_TOL


def test_the_default_bound_holds_the_solve_whole(model):
    """The model's pair space fits: no collapse, and the dense roots."""
    t = {}
    omega, _, _ = solve(model, t)
    assert t['davidson_collapses'] == 0
    assert t['davidson_subspace_max'] <= t['davidson_max_space']
    assert np.abs(omega - model[2][:NROOTS]).max() <= CONV_TOL
