"""Holding BLAS at one thread is a scheduling decision, not an arithmetic one.

`Base.utils.threads.blas_single_threaded` exists because pyscf's OpenMP pool
and the BLAS pool spin against each other: on sixteen cores the exchange build
and the exchange-correlation potential cost 1.90 and 10.38 s with both pools
wide and 0.04 and 0.63 s with BLAS at one. None of that may reach a number the
code returns. What the gates here pin:

  the wrap absent    without threadpoolctl the context manager is
                     `contextlib.nullcontext` and no thread count moves, so a
                     machine without the package runs the same code path.
  off the main
  thread             the same no-op. The count is process-global and the
                     simulated-rank tests put every rank in a thread of one
                     process, where one rank limiting BLAS re-associates its
                     neighbours' GEMMs: unguarded, that scattered the
                     excited-state chain's force by 1.3e-08 Ha/Bohr past the
                     1e-08 ISDF reproducibility floor it is gated at.
  the static term    BITWISE. `static_exchange_mean_field_matrix` under the
                     wrap against the same build with `threadpool_limits(1)`
                     set from OUTSIDE -- the same BLAS thread count, therefore
                     the same summation order, therefore the same bits. A
                     comparison against an UNLIMITED build would measure BLAS
                     re-association instead and would not be bitwise.
  the ISDF fit       the wrap sits around `aux_e2` alone, which is libcint's
                     OpenMP and calls no BLAS, so M and the factors it makes
                     come back bit for bit: measured 0.0 on water/cc-pVDZ, and
                     gated at 1e-12 relative. The gate has teeth -- widening
                     the wrap over the Gram matrix and the Cholesky solve moves
                     M by 2.3e-10 between one and two BLAS threads on
                     ethylene/cc-pVTZ, two orders past it, because the balanced
                     Gram matrix amplifies a re-associated sum by its
                     conditioning. That control runs below, on a fit large
                     enough that BLAS really does spread the Gram over cores
                     and with both counts set rather than assumed. The
                     per-block F D^T is left outside the wrap for the same
                     reason, though it is too small here for BLAS to thread and
                     the gate is silent on it: there the argument carries it,
                     not the measurement.
  the gate           the threshold lives in the context manager, not at the
                     call sites, so one rule covers all of them: below
                     `BLAS_WRAP_MIN_THREADS` there are too few cores for the
                     two pools to contend over, the wrap is its entry cost
                     alone (1-2 ms of rescanning the loaded pools), and it
                     stays a no-op. That is what keeps a laptop's records
                     bitwise -- an SCF wrapped at two threads moved a recorded
                     `xc_correction_eV` by 1e-16 and failed six of the
                     thirteen checks of the frozen record it is compared with.
                     `min_threads=1` arms it anyway, which is how the mechanics
                     above are exercised at two threads.

The thread count a test observes is the one its environment set, so nothing
here asserts a particular number of threads -- only that the wrap sets one
inside when it is armed, leaves it alone when it is not, and puts back what it
found.

Run as a script, this file hands itself to pytest and exits with its verdict.
"""
import contextlib
import importlib.util
import os
import sys
import threading
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df as pyscf_df, dft, gto

import src.Base.separable_ri as separable_ri
from src.Base.constants import BLAS_WRAP_MIN_THREADS
from src.Base.separable_ri import (atomic_grid, build_separable_ri,
                                   molecular_points_covariant)
from src.Base.utils import threads
from src.Base.utils.threads import blas_single_threaded, blas_threads
import src.SingleReference.GW.qp_solve as qp_solve
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import DEFAULT_COUNTS

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
#: The control on where the wrap ends needs a fit whose Gram matrix BLAS really
#: does thread; water at cc-pVDZ is small enough that the library may serve it
#: on one thread whatever it is set to, and then the control measures nothing.
ETHYLENE = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
            'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
THREADED_BASIS, THREADED_AUX = 'cc-pvtz', 'cc-pvtz-ri'

#: The count this process was started on, read before any test has had the
#: chance to leave a limit behind -- threadpoolctl applies one as soon as the
#: object is made, so a later reading is not necessarily the environment's.
AMBIENT_BLAS_THREADS = blas_threads()

#: Relative agreement demanded of the ISDF factors across the wrap. The wrapped
#: region holds no BLAS call, so the measured difference is exactly zero; this
#: is where a re-associated GEMM that had drifted inside it would show.
FACTOR_TOL = 1e-12

_HAS_THREADPOOLCTL = threads.threadpool_limits is not None
_needs_threadpoolctl = pytest.mark.skipif(
    not _HAS_THREADPOOLCTL, reason='threadpoolctl is a soft dependency')


@pytest.fixture(scope='module')
def water():
    """Water/cc-pVDZ/PBE0, density fitted: a hybrid, so <Sigma_x - v_xc> is not
    zero by construction and both halves of the wrapped region carry a value."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUX)
    mf.kernel()
    return dict(mf=mf, mol=mol)


def isdf_case(atom, basis, aux):
    """Molecule, auxiliary molecule and the interpolation points of one fit."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    radii, origins = {}, {}
    for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
        radii[el], origins[el] = atomic_grid(el, mol.basis, aux, DEFAULT_COUNTS)
    coords = molecular_points_covariant(mol, radii, origin_by_element=origins)
    return dict(mol=mol, coords=coords,
                auxmol=pyscf_df.addons.make_auxmol(mol, auxbasis=aux))


@pytest.fixture(scope='module')
def isdf_grid():
    """The interpolation points and the auxiliary molecule the fit is built on."""
    return isdf_case(WATER, BASIS, AUX)


@pytest.fixture(scope='module')
def isdf_grid_threaded():
    """A fit whose Gram matrix is big enough for BLAS to spread over cores."""
    return isdf_case(ETHYLENE, THREADED_BASIS, THREADED_AUX)


def relative(a, b):
    """max |a - b| against the larger entry, how the fitted factors are judged."""
    scale = max(np.abs(a).max(), np.abs(b).max())
    return float(np.abs(a - b).max() / scale) if scale else 0.0


def threads_without_threadpoolctl():
    """A fresh copy of the helper module loaded with the import failing."""
    spec = importlib.util.spec_from_file_location('threads_no_threadpoolctl',
                                                  threads.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_threadpoolctl_everything_is_a_no_op(monkeypatch):
    """The soft dependency absent: a nullcontext, an unknown count, no change."""
    monkeypatch.setitem(sys.modules, 'threadpoolctl', None)
    bare = threads_without_threadpoolctl()
    assert bare.threadpool_limits is None and bare.threadpool_info is None
    assert bare.blas_threads() is None
    before = blas_threads()
    with bare.blas_single_threaded() as handle:
        assert handle is None, 'nullcontext yields nothing to a caller'
        assert blas_threads() == before, 'a no-op moved the thread count'
    assert blas_threads() == before


def openmp_threads():
    """What pyscf's own pool is set to -- the one the wrap may NOT touch."""
    return [lib['num_threads'] for lib in threads.threadpool_info()
            if lib.get('user_api') == 'openmp']


@_needs_threadpoolctl
def test_the_wrap_sets_one_thread_and_puts_back_what_it_found():
    """One thread inside, the original count outside, raise or no raise.

    `min_threads=1` because the mechanics are what is under test here and the
    ambient count on a laptop is below the threshold that arms the wrap.
    """
    before = blas_threads()
    assert before is not None
    openmp_before = openmp_threads()
    with blas_single_threaded(min_threads=1):
        assert blas_threads() == 1
        # The whole point is that pyscf keeps every core. Limiting both pools
        # is the same one-thread run the measurement rejects, and it would
        # change the summation order of the grid pass as well.
        assert openmp_threads() == openmp_before, 'the wrap took pyscf\'s pool'
    assert blas_threads() == before
    with pytest.raises(RuntimeError):
        with blas_single_threaded(min_threads=1):
            raise RuntimeError('the body fails')
    assert blas_threads() == before, 'the count did not come back after a raise'


@_needs_threadpoolctl
def test_the_default_call_leaves_the_ambient_count_where_it_is():
    """What a caller gets with no argument, at whatever this machine runs on.

    Below the threshold the two pools have too few cores to contend over, so
    the wrap does nothing and every gate that compares against a frozen record
    stays bitwise; at or above it the wrap arms and BLAS goes to one.
    """
    ambient = blas_threads()
    with blas_single_threaded():
        inside = blas_threads()
    expected = 1 if (ambient or 1) >= BLAS_WRAP_MIN_THREADS else ambient
    assert inside == expected
    assert blas_threads() == ambient


@_needs_threadpoolctl
def test_a_worker_thread_leaves_the_count_alone():
    """The setting is process-global, so it is the main thread's alone to move.

    A simulated-rank test puts every rank in its own thread of one process.
    A rank that limited BLAS there would re-associate the GEMMs of the ranks
    running beside it, and it does: wrapping the static exchange build without
    this guard scattered the excited-state chain's force by 1.3e-08 Ha/Bohr
    against a 1e-08 gate. Real ranks are processes and never see this.
    """
    before = blas_threads()
    seen = {}

    def inside_a_worker():
        with blas_single_threaded(min_threads=1) as handle:
            seen['handle'] = handle
            seen['count'] = blas_threads()

    worker = threading.Thread(target=inside_a_worker)
    worker.start()
    worker.join()
    assert seen['handle'] is None, 'a worker thread got a live limit'
    assert seen['count'] == before, 'a worker thread moved the whole process'
    assert blas_threads() == before


@_needs_threadpoolctl
def test_the_static_term_is_bitwise_the_outside_limited_build(water, monkeypatch):
    """The wrap and the same limit set from outside are the same arithmetic.

    The thread count is faked past the threshold so the wrap arms on a laptop;
    the reference is the same build under the same limit, set from OUTSIDE.
    """
    monkeypatch.setattr(threads, 'blas_threads', lambda: 16)
    inside = static_exchange_mean_field_matrix(water['mf'], water['mol'])
    monkeypatch.undo()
    with threads.threadpool_limits(limits=1, user_api='blas'):
        reference = static_exchange_mean_field_matrix(water['mf'],
                                                      water['mol'])
    assert np.array_equal(inside, reference), 'the wrap moved the static term'


@_needs_threadpoolctl
@pytest.mark.skipif((AMBIENT_BLAS_THREADS or 1) >= BLAS_WRAP_MIN_THREADS,
                    reason='this machine runs enough cores to arm the wrap')
def test_the_static_term_below_the_threshold_is_the_unwrapped_build(water,
                                                                    monkeypatch):
    """The laptop case, bitwise: nothing is limited, so nothing re-associates.

    A route audit compares `xc_correction_eV` with `==` against a frozen
    record, and a two-thread build dropped to one moves it at 1e-16. Below the
    threshold the wrapped call is the unwrapped one.

    Water/cc-pVDZ is too small for the count to reach its bits -- the two
    matrices are equal either way here -- so what carries the gate is the
    assertion that nothing was limited; the bitwise line states the intent the
    larger records enforce.
    """
    seen = []

    def probe(*args, **kwargs):
        seen.append(blas_single_threaded(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(qp_solve, 'blas_single_threaded', probe)
    wrapped = static_exchange_mean_field_matrix(water['mf'], water['mol'])
    assert seen and all(isinstance(c, contextlib.nullcontext) for c in seen), (
        'the build armed the wrap below the threshold')
    monkeypatch.setattr(qp_solve, 'blas_single_threaded',
                        contextlib.nullcontext)
    plain = static_exchange_mean_field_matrix(water['mf'], water['mol'])
    assert np.array_equal(wrapped, plain)


def armed(**kwargs):
    """Whether the wrap takes hold, entered and left so no limit survives it.

    threadpoolctl applies the limit when the object is MADE, not on entry, so
    a probe that only built one would leave the process at a single thread.
    """
    with blas_single_threaded(**kwargs) as handle:
        return handle is not None


@_needs_threadpoolctl
@pytest.mark.parametrize('count,wrapped', [(1, False), (2, False),
                                           (BLAS_WRAP_MIN_THREADS - 1, False),
                                           (BLAS_WRAP_MIN_THREADS, True),
                                           (16, True), (None, False)])
def test_the_gate_follows_the_blas_thread_count(monkeypatch, count, wrapped):
    """Below the threshold the caller is left alone; at it and above, wrapped."""
    monkeypatch.setattr(threads, 'blas_threads', lambda: count)
    assert armed() is wrapped
    # min_threads is the one handle on it, and it reaches both ways.
    assert armed(min_threads=64) is False
    assert armed(min_threads=1) is True


@_needs_threadpoolctl
@pytest.mark.parametrize('streaming', [True, False])
def test_the_isdf_factors_do_not_move_across_the_wrap(isdf_grid, streaming,
                                                      monkeypatch):
    """Both realizations of the fit: the wrapped region holds no BLAS call."""
    args = dict(auxmol=isdf_grid['auxmol'], streaming=streaming)
    monkeypatch.setattr(threads, 'blas_threads', lambda: 1)
    X, Z, M = build_separable_ri(isdf_grid['mol'], isdf_grid['coords'], **args)
    monkeypatch.setattr(threads, 'blas_threads', lambda: 16)
    X_w, Z_w, M_w = build_separable_ri(isdf_grid['mol'], isdf_grid['coords'],
                                       **args)
    for name, plain, wrapped in (('M', M, M_w), ('X', X, X_w), ('Z', Z, Z_w)):
        assert relative(plain, wrapped) < FACTOR_TOL, (
            f'{name} moved by {relative(plain, wrapped):.3e} across the wrap')


@_needs_threadpoolctl
@pytest.mark.skipif(AMBIENT_BLAS_THREADS in (None, 1),
                    reason='BLAS is already at one thread: nothing to widen')
def test_a_wrap_widened_over_the_gram_would_move_the_fit(isdf_grid_threaded):
    """The control on the gate above: the parts left outside the wrap DO care.

    Holding BLAS at one thread for the whole fit -- the Gram matrix and the
    Cholesky solve with it -- re-associates their sums, and the balanced Gram
    matrix is ill conditioned enough to turn that rounding into 2e-10 in M on
    ethylene/cc-pVTZ, two orders past the gate. So where the wrap ends is an
    answer, not a taste.

    BOTH counts are set here rather than one against whatever is ambient: the
    library serves a small enough GEMM on one thread whatever it is set to,
    and a count left to the state an earlier test finished in is not a count.
    """
    fit = (isdf_grid_threaded['mol'], isdf_grid_threaded['auxmol'],
           isdf_grid_threaded['coords'])
    with threads.threadpool_limits(limits=AMBIENT_BLAS_THREADS,
                                   user_api='blas'):
        wide = separable_ri.fit_M_streaming(*fit)
    with threads.threadpool_limits(limits=1, user_api='blas'):
        narrow = separable_ri.fit_M_streaming(*fit)
    assert relative(wide, narrow) > FACTOR_TOL, (
        f'{AMBIENT_BLAS_THREADS} BLAS threads no longer move this fit, so the '
        'gate above is blind')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
