"""Where `separable_factors` spends its wall clock, phase by phase.

The factorization is one function call, reused everywhere GW or BSE needs the
factors, and a `comm=` splits only `fit_M_streaming`'s three-centre pass.
`timings=` is a dict filled at the boundaries of every phase the call
graph actually has -- `fit_points` (`molecular_points_covariant`),
`fit_collocation` (the AO/aux collocation, the test-set selection, the
auxiliary metric LU factorization and the screening threshold, all one-time),
`fit_gram` (the Gram matrix and its balancing, read once before the
three-centre pass and once after), `fit_integrals` (the aux_e2 pass, with
`fit_blocks`/`fit_blocks_total` beside it; this rank's share of it under a
comm), `fit_integrals_reduce` (what closes the pass, the one collective under
a comm), `fit_cholesky` and `fit_solve` (`scipy`'s `posv` fuses the
two into one LAPACK call, so `fit_solve` is always 0.0 -- see
`fit_M_streaming`), `fit_assembly` (the fit's own V/Z/X tail plus
`separable_factors`'s dressed-metric and projection tail, added into one key
since the call graph does not actually keep them apart) and `fit_total`, the
whole call on whichever rank asked.

Gated here: the timers read the clock and touch no bit of the factorization,
against the same call without `timings=` serially and under a comm, and
against the pre-instrumentation code itself, extracted with `git archive` and
run in its own process; every key is present on every rank; the phases stay
inside the total; a serial fit walks every block, so `fit_blocks` is
`fit_blocks_total`; and under a comm `fit_blocks` sums over ranks to
`fit_blocks_total`, so the two cannot silently drift into counting different
things.

Water/cc-pVDZ and ethylene/cc-pVTZ at tight `block_memory_gb` budgets, so the
three-centre pass is actually many blocks rather than the single block a
generous budget gives a molecule this small.

Run as a script, this file hands itself to pytest and exits with its verdict.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.separable_ri import atomic_grid
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import DEFAULT_COUNTS, separable_factors

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')

#: (geometry, basis, block_memory_gb). The tight budgets cut the AO index one
#: shell per block, so the three-centre pass has many blocks and
#: `fit_blocks`/`fit_blocks_total` count an actual loop rather than the single
#: block a generous budget gives a molecule this small.
CASES = {'water': (WATER, 'cc-pvdz', 2e-4),
         'ethylene': (ETHYLENE, 'cc-pvtz', 1e-3)}

#: Ranks to simulate beside the serial case.
RANKS = (2, 3)

#: The phase keys that must sum to no more than `fit_total`.
PHASE_KEYS = ('fit_points', 'fit_collocation', 'fit_integrals',
              'fit_integrals_reduce', 'fit_gram', 'fit_cholesky', 'fit_solve',
              'fit_assembly')
ALL_TIMING_KEYS = PHASE_KEYS + ('fit_total', 'fit_blocks', 'fit_blocks_total')

#: s. Timers are read with time.time() at phase boundaries only, so a part may
#: exceed the whole by the time reading the clock itself costs between them.
TIMER_SLACK = 1e-3

REPO = Path(__file__).resolve().parents[1]
#: The commit before these timers were added. Pinned rather than `HEAD`: once this file's own
#: change lands, `HEAD` would include it and the comparison would stop
#: meaning anything.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
#: A shared machine: the archived probe is capped rather than left to size
#: itself against the whole node.
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: Built fresh in its own process against the extracted tree. The archived
#: tree has no `timings=` keyword at all, so this calls `separable_factors`
#: exactly as it always could. The grid goes in as explicit radii: the
#: archived code substituted the published cc-pVTZ grids for a bare call,
#: which the grid-keyword rule retired, so a bare call is a different grid in
#: the two trees at that basis. Radii alone ARE the grid in both -- no table
#: row is consulted and no nuclear point is added -- so the comparison stays
#: on the timers.
BITWISE_PROBE = '''
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.SingleReference.GW.space_time import separable_factors

warnings.simplefilter('ignore')
mol = gto.M(atom={atom!r}, basis={basis!r}, verbose=0)
auxbasis = {basis!r} + '-ri'
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
radii = {{el: {{shell: np.array(r) for shell, r in shells.items()}}
         for el, shells in {radii!r}.items()}}
X_mo, D, X_ao, coords = separable_factors(mf, mol, auxbasis=auxbasis,
                                          radii=radii,
                                          block_memory_gb={block_memory_gb!r})
np.savez({out!r}, X_mo=X_mo, D=D, X_ao=X_ao, coords=coords)
'''


@pytest.fixture(scope='module', params=sorted(CASES))
def case(request):
    """(name, mol, auxbasis, mf, block_memory_gb) for one molecule."""
    atom, basis, block_memory_gb = CASES[request.param]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    auxbasis = basis + '-ri'
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return dict(name=request.param, mol=mol, auxbasis=auxbasis, mf=mf,
                block_memory_gb=block_memory_gb)


def _factors(case, comm=None, timings=None, radii=None):
    return separable_factors(case['mf'], case['mol'], auxbasis=case['auxbasis'],
                             radii=radii,
                             block_memory_gb=case['block_memory_gb'],
                             comm=comm, timings=timings)


def _table_radii(case):
    """The shipped rows at `DEFAULT_COUNTS`, as plain lists of floats (repr
    round-trips a float64 exactly) for the archived probe to rebuild."""
    mol = case['mol']
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    return {el: {shell: [float(x) for x in np.atleast_1d(r)]
                 for shell, r in atomic_grid(el, str(mol.basis), case['auxbasis'],
                                             DEFAULT_COUNTS)[0].items()}
            for el in elements}


def _bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def _assert_factors_equal(old, new, where):
    for name, a, b in zip(('X_mo', 'D', 'X_ao', 'coords'), old, new):
        assert _bitwise(a, b), f'{name} moved {where}'


def test_serial_timings_move_no_bits(case):
    """Passing `timings=` only reads the clock; the factors it returns must
    be the ones an uninstrumented call returns."""
    without = _factors(case)
    t = {}
    with_t = _factors(case, timings=t)
    _assert_factors_equal(without, with_t, 'when timings= was passed')
    for key in ALL_TIMING_KEYS:
        assert key in t, key


@pytest.mark.parametrize('size', RANKS)
def test_distributed_timings_move_no_bits(case, size):
    """Same gate under a comm: every rank agrees with itself with and without
    `timings=`. Each rank gets its OWN dict -- a `run_simulated` rank runs in
    its own thread, and a dict shared across threads would race."""
    without = run_simulated(lambda comm: _factors(case, comm=comm), size)

    def with_timings(comm):
        t = {}
        return _factors(case, comm=comm, timings=t), t

    with_t = run_simulated(with_timings, size)
    for r in range(size):
        new, t = with_t[r]
        _assert_factors_equal(without[r], new, f'on rank {r} when timings= was passed')
        for key in ALL_TIMING_KEYS:
            assert key in t, (r, key)
        json.dumps(t)              # a caller may record it as JSON


def test_phase_boundaries_bound_the_total(case):
    """Serially one call walks every block, so `fit_blocks` is the total, and
    the phases sum to no more than the call they were cut from."""
    t = {}
    _factors(case, timings=t)
    assert sum(t[k] for k in PHASE_KEYS) <= t['fit_total'] + TIMER_SLACK
    assert t['fit_blocks'] == t['fit_blocks_total']
    assert t['fit_blocks'] > 1, 'the tight budget did not split the pass'
    assert t['fit_solve'] == 0.0


@pytest.mark.parametrize('size', RANKS)
def test_blocks_sum_to_the_total_under_a_comm(case, size):
    """Every rank's own `fit_blocks` -- the blocks IT walked -- adds up to the
    `fit_blocks_total` every rank reports identically, and the phases stay
    inside each rank's own total."""
    def one_rank(comm):
        t = {}
        _factors(case, comm=comm, timings=t)
        return t

    per_rank = run_simulated(one_rank, size)
    total = per_rank[0]['fit_blocks_total']
    assert total > max(RANKS)          # a comm with nothing to split tests nothing
    for t in per_rank:
        assert t['fit_blocks_total'] == total
        assert sum(t[k] for k in PHASE_KEYS) <= t['fit_total'] + TIMER_SLACK
    assert sum(t['fit_blocks'] for t in per_rank) == total


@pytest.fixture(scope='session')
def archive(tmp_path_factory):
    """`separable_factors` before the timing phases were added, unpacked
    into a temporary directory."""
    out = tmp_path_factory.mktemp('isdf_fit_timings_baseline')
    tar = out.parent / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
    assert (out / 'src' / 'Base' / 'separable_ri.py').is_file()
    return out


def _archived_factors(archive_dir, tmp_path, case_name, radii):
    """(X_mo, D, X_ao, coords) from the pre-instrumentation code, its own process."""
    atom, basis, block_memory_gb = CASES[case_name]
    script = tmp_path / f'probe_{case_name}.py'
    out = tmp_path / f'factors_{case_name}.npz'
    script.write_text(BITWISE_PROBE.format(archive=str(archive_dir), atom=atom,
                                           basis=basis, radii=radii,
                                           block_memory_gb=block_memory_gb,
                                           out=str(out)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(archive_dir),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    data = np.load(out)
    return data['X_mo'], data['D'], data['X_ao'], data['coords']


def test_timings_move_no_bits_against_the_archive(case, tmp_path_factory,
                                                  archive):
    """The pre-instrumentation code's own factors, run as a subprocess on the
    extracted tree, against this tree's -- called the way every production
    entry point now can, with `timings=`.

    Nothing between the two trees touches the arithmetic of the fit: the only
    change is a clock read and a dict write at phase boundaries that were
    already there as code, so this is bitwise. Both sides get the same
    explicit radii (see `BITWISE_PROBE`).
    """
    tmp = tmp_path_factory.mktemp(f'isdf_fit_timings_bitwise_{case["name"]}')
    radii = _table_radii(case)
    old = _archived_factors(archive, tmp, case['name'], radii)
    new = _factors(case, timings={},
                   radii={el: {shell: np.array(r) for shell, r in shells.items()}
                          for el, shells in radii.items()})
    _assert_factors_equal(old, new, f'relative to {BASELINE_COMMIT[:12]}')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
