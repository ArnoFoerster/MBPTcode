"""The space-time GW Dyson step split over frequencies, under simulated ranks.

`solve_qp_energy_space_time(distribute=True)` turns chi0(i.omega) into
W(i.omega) = [I - chi0(i.omega)]^-1 one frequency at a time. Each rank inverts
its own contiguous block of frequencies and the inverted rows are all-gathered
verbatim (`allgather_blocks`): nothing is summed across ranks by that step, so
every rank holds the owner's bits, and those are the bits the replicated
inversion computed.

Gated on water/cc-pVDZ and ethylene/cc-pVTZ Hartree-Fock, on the GW window
(no omega = 0 passenger) and on the whole diagonal with the static W a BSE
takes from it (the passenger row, owned by the last non-empty rank):
  * serially BITWISE against the code before the split, extracted with
    `git archive` and run in its own process on the same mean field and
    factors;
  * at 2, 3 and 8 simulated ranks BITWISE against that code's own distributed
    run at the same rank count -- its tau split with the Dyson inversion
    replicated on every rank -- the quasiparticle energies within `QP_TOL` of
    serial -- measured 0.0 Ha at every size on both molecules -- and
    W(i.omega), W(0) and the energies the same bits on every rank. That code
    predates the simulated communicator, so the extracted tree is run with
    this tree's `Base/utils/mpi_grid.py` laid over its own (the primitives it
    calls, `grid_comm`, `partition` and `reduce_sum`, keep their contracts)
    and the two constants that module reads appended to its constants;
  * `t_dyson` on every rank, and every rank's `dyson_frequencies` its own
    contiguous block, so a return to the replicated inversion, which would
    move no bit, is still seen;
  * 8 ranks on 6 frequencies, where the last ranks own an EMPTY block, invert
    nothing and leave the answer unchanged;
  * rank 1's inverted block perturbed, which must move rank 0's answer out of
    `QP_TOL`: the gate reads the rows that rank inverted.

Shown to fail: `allgather_blocks` made a no-op (still called, on a copy, so
the ranks stay in step) puts water's quasiparticle energies 1.1e-2 to 1.7e-1
Ha off serial on the window and 2.4 to 4.6 Ha on the diagonal, at 2, 3 and 8
ranks. Every rank inverting every row again (`contiguous_block` bypassed)
moves no bit and fails the `dyson_frequencies` gates alone.
"""
import copy
import os
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils.mpi_grid import (allgather_blocks, contiguous_block,
                                     run_simulated)
from src.SingleReference.GW import space_time
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

MOLECULES = {
    'water': ('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
              'cc-pvdz'),
    'ethylene': ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321',
                 'cc-pvtz'),
}
KINDS = ('window', 'diagonal')
SIZES = [2, 3, 8]
#: 8 ranks on 6 frequencies, 7 rows with the omega = 0 passenger: ranks 6 and 7
#: own nothing on the window, rank 7 nothing on the diagonal.
EMPTY_SIZE, EMPTY_NTAU = 8, 6
#: Ha. The frequency split moves no bit; what separates a distributed run from
#: the serial one is the tau reduction of chi0 upstream of it.
QP_TOL = 1e-10
#: Added to the diagonal of rank 1's inverted rows: far above the gate.
PERTURBATION = 1e-6

#: Six points is a partition fixture, not a converged grid, and says so.
pytestmark = pytest.mark.filterwarnings(
    'ignore:minimax transform fit reached only:RuntimeWarning')

REPO = Path(__file__).resolve().parents[1]
#: The last commit with the replicated inversion. Pinned rather than `HEAD`,
#: which after this change would compare the tree with itself.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
#: The constants this tree's `mpi_grid` imports, which the baseline lacks.
OVERLAY_CONSTANTS = ('AGREEMENT_DIGEST_SEED', 'AGREEMENT_DIGEST_BLOCK')
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: The same calls on the extracted tree, in its own process, on the mean field
#: and factors this process built (`{inputs}`), so only the GW route differs.
ARCHIVED_PROBE = '''
import copy
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import (solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

warnings.simplefilter('ignore')
MOLECULES = {molecules!r}
out = {{}}
for name, (atom, basis) in MOLECULES.items():
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=basis + '-ri')
    mf.kernel()
    d = np.load({inputs!r}.format(name=name))
    mf.mo_energy, mf.mo_coeff = d['mo_energy'], d['mo_coeff']
    factors = (d['X_mo'], d['D'], d['X_ao'], d['coords'])
    nocc = mol.nelectron // 2

    def route(kind, ntau, comm=None):
        m = copy.copy(mf)
        m.mo_energy = np.asarray(mf.mo_energy, float).copy()
        m.mo_coeff = np.asarray(mf.mo_coeff, float).copy()
        f = tuple(np.array(a, copy=True) for a in factors)
        kw = dict(factors=f, ntau=ntau, distribute=comm is not None, comm=comm)
        if kind == 'window':
            return solve_qp_energy_space_time(
                m, mol, nocc, np.array([nocc - 1, nocc]), **kw), np.zeros(0)
        extras = {{}}
        qp, _ = solve_qp_diagonal_space_time(m, mol, nocc, extras=extras, **kw)
        return qp, extras['w_static']

    for kind in {kinds!r}:
        for ntau, sizes in (('auto', {sizes!r}), ({empty_ntau!r}, [{empty_size!r}])):
            qp, ws = route(kind, ntau)
            out[f'{{name}}_{{kind}}_{{ntau}}_0_qp'] = qp
            out[f'{{name}}_{{kind}}_{{ntau}}_0_ws'] = ws
            for size in sizes:
                qp, ws = run_simulated(lambda c: route(kind, ntau, c), size)[0]
                out[f'{{name}}_{{kind}}_{{ntau}}_{{size}}_qp'] = qp
                out[f'{{name}}_{{kind}}_{{ntau}}_{{size}}_ws'] = ws
np.savez({out!r}, **out)
'''


@pytest.fixture(scope='module')
def systems():
    warnings.simplefilter('ignore')
    out = {}
    for name, (atom, basis) in MOLECULES.items():
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=basis + '-ri')
        mf.kernel()
        factors = separable_factors(mf, mol, auxbasis=basis + '-ri')
        out[name] = dict(mol=mol, mf=mf, factors=factors,
                         nocc=mol.nelectron // 2)
    return out


def _route(s, kind, ntau='auto', comm=None):
    """(quasiparticle energies, W(0) or an empty array, timings) on this
    rank's own copy of the mean field and factors, which a distributed solve
    overwrites in place with rank 0's."""
    mf = copy.copy(s['mf'])
    mf.mo_energy = np.asarray(s['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(s['mf'].mo_coeff, float).copy()
    factors = tuple(np.array(a, copy=True) for a in s['factors'])
    nocc, timings = s['nocc'], {}
    kw = dict(factors=factors, ntau=ntau, timings=timings,
              distribute=comm is not None, comm=comm)
    if kind == 'window':
        qp = solve_qp_energy_space_time(mf, s['mol'], nocc,
                                        np.array([nocc - 1, nocc]), **kw)
        return qp, np.zeros(0), timings
    extras = {}
    qp, _ = solve_qp_diagonal_space_time(mf, s['mol'], nocc, extras=extras,
                                         **kw)
    return qp, extras['w_static'], timings


def _bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def _gathered_rows(monkeypatch):
    """{rank: W(i.omega) as that rank holds it after the gather}."""
    seen = {}

    def gather(a, comm):
        allgather_blocks(a, comm)
        seen[0 if comm is None else comm.Get_rank()] = a.copy()
        return a

    monkeypatch.setattr(space_time, 'allgather_blocks', gather)
    return seen


@pytest.fixture(scope='module')
def archived(systems, tmp_path_factory):
    """Every call of this file on the code before the split, in one process."""
    tmp = tmp_path_factory.mktemp('dyson_baseline')
    tar = tmp / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    tree = tmp / 'tree'
    with tarfile.open(tar) as fh:
        fh.extractall(tree)
    assert (tree / 'src' / 'SingleReference' / 'GW' / 'space_time.py').is_file()
    _overlay_communicator(tree)
    for name, s in systems.items():
        X_mo, D, X_ao, coords = s['factors']
        np.savez(tmp / f'inputs_{name}.npz', mo_energy=s['mf'].mo_energy,
                 mo_coeff=s['mf'].mo_coeff, X_mo=X_mo, D=D, X_ao=X_ao,
                 coords=coords)
    out = tmp / 'archived.npz'
    script = tmp / 'probe.py'
    script.write_text(ARCHIVED_PROBE.format(
        archive=str(tree), molecules=MOLECULES, kinds=KINDS, sizes=SIZES,
        empty_ntau=EMPTY_NTAU, empty_size=EMPTY_SIZE,
        inputs=str(tmp / 'inputs_{name}.npz'), out=str(out)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tree),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(out))


def _overlay_communicator(tree):
    """This tree's `mpi_grid` over the baseline's, and the constants it reads.

    The baseline's distributed path is the one under comparison; its own
    `mpi_grid` has no simulated communicator to run it on. The overlay changes
    none of the calls that path makes -- `grid_comm`, `partition` and
    `reduce_sum` keep their contracts -- only what carries them.
    """
    grid = Path('src') / 'Base' / 'utils' / 'mpi_grid.py'
    (tree / grid).write_text((REPO / grid).read_text())
    constants = tree / 'src' / 'Base' / 'constants.py'
    ours = (REPO / 'src' / 'Base' / 'constants.py').read_text().splitlines()
    added = [line for line in ours
             if line.split(' = ')[0] in OVERLAY_CONSTANTS]
    assert len(added) == len(OVERLAY_CONSTANTS), added
    constants.write_text(constants.read_text() + '\n' + '\n'.join(added)
                         + '\n')


def _check_ranks(outs, gathered, serial, archived_key, archived, kind, ntau):
    """Every rank's answer and W: one set of bits, the archived distributed
    ones, within QP_TOL of serial; t_dyson on every rank and each rank's
    inversion count its own contiguous block."""
    size = len(outs)
    qp0, ws0, _ = outs[0]
    assert _bitwise(qp0, archived[archived_key + '_qp'])
    assert _bitwise(ws0, archived[archived_key + '_ws'])
    assert np.abs(qp0 - serial[0]).max() <= QP_TOL
    assert sorted(gathered) == list(range(size))
    nrows = outs[0][2].get('ntau_auto', ntau) + (kind == 'diagonal')
    assert gathered[0].shape[0] == nrows
    for r, (qp, ws, t) in enumerate(outs):
        assert _bitwise(qp, qp0)
        assert _bitwise(ws, ws0)
        assert _bitwise(gathered[r], gathered[0])
        assert t['t_dyson'] >= 0.0
        start, stop = contiguous_block(nrows, r, size)
        assert t['dyson_frequencies'] == stop - start
    assert sum(t['dyson_frequencies'] for _, _, t in outs) == nrows


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(MOLECULES))
def test_serial_is_the_archived_code(systems, archived, name, kind):
    """One rank inverts every frequency, which is what the code before the
    split did: the same energies and the same static W, bit for bit."""
    for ntau in ('auto', EMPTY_NTAU):
        qp, ws, t = _route(systems[name], kind, ntau)
        key = f'{name}_{kind}_{ntau}_0'
        assert _bitwise(qp, archived[key + '_qp'])
        assert _bitwise(ws, archived[key + '_ws'])
        assert t['dyson_frequencies'] == (t.get('ntau_auto', ntau)
                                          + (kind == 'diagonal'))


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(MOLECULES))
def test_dyson_over_frequencies(systems, archived, monkeypatch, name, kind,
                                size):
    """Each rank inverts its own block; the gathered W is one set of bits."""
    s = systems[name]
    serial = _route(s, kind)
    gathered = _gathered_rows(monkeypatch)
    outs = run_simulated(lambda c: _route(s, kind, comm=c), size)
    _check_ranks(outs, gathered, serial, f'{name}_{kind}_auto_{size}',
                 archived, kind, 'auto')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(MOLECULES))
def test_empty_frequency_blocks(systems, archived, monkeypatch, name, kind):
    """More ranks than frequencies: the surplus own an empty block, invert
    nothing, and the answer is the one the ranks with rows produce."""
    s = systems[name]
    serial = _route(s, kind, EMPTY_NTAU)
    gathered = _gathered_rows(monkeypatch)
    outs = run_simulated(lambda c: _route(s, kind, EMPTY_NTAU, c), EMPTY_SIZE)
    assert outs[-1][2]['dyson_frequencies'] == 0
    _check_ranks(outs, gathered, serial,
                 f'{name}_{kind}_{EMPTY_NTAU}_{EMPTY_SIZE}', archived, kind,
                 EMPTY_NTAU)


@pytest.mark.parametrize('kind', KINDS)
def test_one_ranks_block_trips_the_gate(systems, monkeypatch, kind):
    """Rank 1's inverted rows, perturbed after its inversion, reach every rank
    and move rank 0's energies out of QP_TOL."""
    s, size = systems['water'], 3
    serial = _route(s, kind)
    dyson = space_time._dyson_in_place

    def perturbed(chi0, rows, static_index=None, transform=None):
        dyson(chi0, rows, static_index, transform)
        if rows == range(*contiguous_block(chi0.shape[0], 1, size)):
            for k in rows:
                chi0[k][np.diag_indices(chi0.shape[-1])] += PERTURBATION
        return chi0

    monkeypatch.setattr(space_time, '_dyson_in_place', perturbed)
    gathered = _gathered_rows(monkeypatch)
    outs = run_simulated(lambda c: _route(s, kind, comm=c), size)
    start, stop = contiguous_block(gathered[0].shape[0], 1, size)
    assert stop > start
    for r in range(size):
        assert _bitwise(gathered[r], gathered[0])
        assert _bitwise(outs[r][0], outs[0][0])
    assert np.abs(outs[0][0] - serial[0]).max() > QP_TOL


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
