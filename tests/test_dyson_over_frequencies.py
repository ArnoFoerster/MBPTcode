"""The space-time GW Dyson step split over frequencies, under simulated ranks.

`solve_qp_energy_space_time(distribute=True)` turns chi0(i.omega) into
W(i.omega) = [I - chi0(i.omega)]^-1 one frequency at a time. Over more than one
rank each frequency is inverted by its round-robin owner (`partition`) and held
there alone (`_dyson_owned`), W(0) is broadcast from its owner, and the
screened interaction the self-energy reads is assembled from the owners' W:
nothing is summed across ranks by the Dyson step itself.

Gated on water/cc-pVDZ and ethylene/cc-pVTZ Hartree-Fock, on the GW window
(no omega = 0 passenger) and on the whole diagonal with the static W a BSE
takes from it (the passenger row, inverted by its round-robin owner):
  * serially BITWISE against the code before the split, extracted with
    `git archive` and run in its own process on the same mean field and
    factors;
  * at 2, 3 and 8 simulated ranks BITWISE against the serial run: the
    quasiparticle energies and W(0) the same bits on every rank and the serial
    ones -- at these sizes each (tau, grid tile) and (tau, block pair) item of
    the route's two reduced sums is a single tile, so the reductions add exact
    zeros -- and each frequency's W, on the one rank that owns it, the serial
    W;
  * `t_dyson` on every rank, and every rank's inverted frequencies exactly its
    round-robin share, the omega = 0 passenger apart, so a return to the
    replicated inversion, which would move no bit, is still seen;
  * 8 ranks on 6 frequencies, where the surplus ranks own nothing, invert
    nothing and leave W(0) and every W the serial bits. With more ranks than
    tau points the route also splits each point's self-energy pairs over the
    ranks, which re-associates their sums, so the energies there are gated as
    the grid-row route's own tests gate them: within `COMPOSED_GRAD_K` times
    what relabelling the grid points moves the serial ones (6.4e-9 Ha on two
    deep ethylene states), floored at `QP_BISECTION_TOL`;
  * rank 1's inverted frequencies perturbed, which must move rank 0's answer
    out of `QP_TOL`: the gate reads the W that rank inverted.

Shown to fail: every rank inverting every frequency again (`partition`
bypassed in `_dyson_owned`) moves no bit and fails the ownership gates alone,
at 2, 3 and 8 ranks and on the surplus ranks; W(0) left on its owner (the
broadcast dropped) fails the W(0) gates on the diagonal.
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

from src.Base.constants import COMPOSED_GRAD_K, QP_BISECTION_TOL
from src.Base.utils.mpi_grid import partition, run_simulated
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
#: Ha. A perturbed rank's W must move the energies past this.
QP_TOL = 1e-10
#: Added to the diagonal of rank 1's inverted rows: far above the gate.
PERTURBATION = 1e-6
#: Random orders of the grid points whose largest move of the serial energies
#: anchors the bar where a rank count re-associates them.
PERMUTATIONS = 3

#: Six points is a partition fixture, not a converged grid, and says so.
pytestmark = pytest.mark.filterwarnings(
    'ignore:minimax transform fit reached only:RuntimeWarning')

REPO = Path(__file__).resolve().parents[1]
#: The last commit with the replicated inversion. Pinned rather than `HEAD`,
#: which would compare the tree with itself.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
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

    def route(kind, ntau):
        m = copy.copy(mf)
        m.mo_energy = np.asarray(mf.mo_energy, float).copy()
        m.mo_coeff = np.asarray(mf.mo_coeff, float).copy()
        f = tuple(np.array(a, copy=True) for a in factors)
        kw = dict(factors=f, ntau=ntau)
        if kind == 'window':
            return solve_qp_energy_space_time(
                m, mol, nocc, np.array([nocc - 1, nocc]), **kw), np.zeros(0)
        extras = {{}}
        qp, _ = solve_qp_diagonal_space_time(m, mol, nocc, extras=extras, **kw)
        return qp, extras['w_static']

    for kind in {kinds!r}:
        for ntau in ('auto', {empty_ntau!r}):
            qp, ws = route(kind, ntau)
            out[f'{{name}}_{{kind}}_{{ntau}}_0_qp'] = qp
            out[f'{{name}}_{{kind}}_{{ntau}}_0_ws'] = ws
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


def _owned_rows(monkeypatch, perturb_rank=None):
    """{rank: ({frequency: W - I}, W(0), frequency count)} as each rank's
    `_dyson_owned` returns them; `perturb_rank`'s W are moved by PERTURBATION
    on the diagonal after its inversion."""
    seen, dyson = {}, space_time._dyson_owned

    def owned(chi0, static_index=None, transform=None):
        W_minus_I, w_static = dyson(chi0, static_index, transform)
        size, rank = chi0._size_rank()
        if rank == perturb_rank:
            for W in W_minus_I.values():
                W[np.diag_indices(W.shape[-1])] += PERTURBATION
        seen[rank] = ({k: W.copy() for k, W in W_minus_I.items()},
                      None if w_static is None else w_static.copy(),
                      chi0.shape[0])
        return W_minus_I, w_static

    monkeypatch.setattr(space_time, '_dyson_owned', owned)
    return seen


def _serial_rows(s, kind, ntau, monkeypatch):
    """The serial W(i.omega) - I, frequency by frequency, as `_dyson_owned`
    forms it from W: the diagonal less one after the inversion."""
    rows, dyson = {}, space_time._dyson_in_place

    def recorded(chi0, rng, static_index=None, transform=None):
        W = dyson(chi0, rng, static_index, transform)
        for k in rng:
            if k != static_index:
                Wk = W[k].copy()
                Wk[np.diag_indices(Wk.shape[-1])] -= 1.0
                rows[k] = Wk
        return W

    monkeypatch.setattr(space_time, '_dyson_in_place', recorded)
    try:
        _route(s, kind, ntau)
    finally:
        monkeypatch.setattr(space_time, '_dyson_in_place', dyson)
    return rows


@pytest.fixture(scope='module')
def archived(systems, tmp_path_factory):
    """The serial calls of this file on the code before the split, in one
    process."""
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
    for name, s in systems.items():
        X_mo, D, X_ao, coords = s['factors']
        np.savez(tmp / f'inputs_{name}.npz', mo_energy=s['mf'].mo_energy,
                 mo_coeff=s['mf'].mo_coeff, X_mo=X_mo, D=D, X_ao=X_ao,
                 coords=coords)
    out = tmp / 'archived.npz'
    script = tmp / 'probe.py'
    script.write_text(ARCHIVED_PROBE.format(
        archive=str(tree), molecules=MOLECULES, kinds=KINDS,
        empty_ntau=EMPTY_NTAU,
        inputs=str(tmp / 'inputs_{name}.npz'), out=str(out)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tree),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(out))


def _relabelled_bar(s, kind, ntau, serial_qp):
    """COMPOSED_GRAD_K times the largest move of the serial energies over
    PERMUTATIONS relabellings of the grid points, floored at
    QP_BISECTION_TOL: what re-associating the route's sums may move them."""
    moved = 0.0
    for seed in range(PERMUTATIONS):
        perm = np.random.default_rng(seed).permutation(s['factors'][0].shape[0])
        relabelled = dict(s, factors=tuple(a[perm] for a in s['factors']))
        moved = max(moved, np.abs(_route(relabelled, kind, ntau)[0]
                                  - serial_qp).max())
    return COMPOSED_GRAD_K * max(moved, QP_BISECTION_TOL)


def _check_ranks(outs, owned, serial, serial_rows, kind, ntau, qp_bar=None):
    """Every rank's answer and W(0) one set of bits, W(0) the serial one and
    the energies too, or within `qp_bar` of them where the rank count
    re-associates the self-energy; each frequency inverted once, by its
    round-robin owner, into the serial W; t_dyson on every rank and each
    rank's inversion count its own share."""
    size = len(outs)
    qp0, ws0, _ = outs[0]
    if qp_bar is None:
        assert _bitwise(qp0, serial[0])
    else:
        assert np.abs(qp0 - serial[0]).max() <= qp_bar
    assert _bitwise(ws0, serial[1])
    assert sorted(owned) == list(range(size))
    nrows = outs[0][2].get('ntau_auto', ntau) + (kind == 'diagonal')
    static = nrows - 1 if kind == 'diagonal' else None
    assert sorted(serial_rows) == [k for k in range(nrows) if k != static]
    for r, (qp, ws, t) in enumerate(outs):
        W_minus_I, w_static, n = owned[r]
        assert n == nrows
        assert _bitwise(qp, qp0)
        assert _bitwise(ws, ws0)
        if static is not None:
            assert _bitwise(w_static, ws0)
        mine = [k for k in partition(nrows, r, size) if k != static]
        assert sorted(W_minus_I) == mine
        for k, W in W_minus_I.items():
            assert _bitwise(W, serial_rows[k]), (r, k)
        assert t['t_dyson'] >= 0.0
        assert t['dyson_frequencies'] == len(mine)
    assert sum(t['dyson_frequencies'] for _, _, t in outs) == len(serial_rows)


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
def test_dyson_over_frequencies(systems, monkeypatch, name, kind, size):
    """Each rank inverts its own frequencies into the serial W; the answer is
    one set of bits."""
    s = systems[name]
    serial = _route(s, kind)
    serial_rows = _serial_rows(s, kind, 'auto', monkeypatch)
    owned = _owned_rows(monkeypatch)
    outs = run_simulated(lambda c: _route(s, kind, comm=c), size)
    _check_ranks(outs, owned, serial, serial_rows, kind, 'auto')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(MOLECULES))
def test_empty_frequency_blocks(systems, monkeypatch, name, kind):
    """More ranks than frequencies: the surplus own nothing, invert nothing,
    and the answer is the one the ranks with frequencies produce."""
    s = systems[name]
    serial = _route(s, kind, EMPTY_NTAU)
    serial_rows = _serial_rows(s, kind, EMPTY_NTAU, monkeypatch)
    owned = _owned_rows(monkeypatch)
    outs = run_simulated(lambda c: _route(s, kind, EMPTY_NTAU, c), EMPTY_SIZE)
    assert outs[-1][2]['dyson_frequencies'] == 0
    _check_ranks(outs, owned, serial, serial_rows, kind, EMPTY_NTAU,
                 qp_bar=_relabelled_bar(s, kind, EMPTY_NTAU, serial[0]))


@pytest.mark.parametrize('kind', KINDS)
def test_one_ranks_block_trips_the_gate(systems, monkeypatch, kind):
    """Rank 1's inverted W, perturbed after its inversion, reach every rank
    and move rank 0's energies out of QP_TOL."""
    s, size = systems['water'], 3
    serial = _route(s, kind)
    owned = _owned_rows(monkeypatch, perturb_rank=1)
    outs = run_simulated(lambda c: _route(s, kind, comm=c), size)
    assert owned[1][0], 'rank 1 owns frequencies'
    for r in range(size):
        assert _bitwise(outs[r][0], outs[0][0])
    assert np.abs(outs[0][0] - serial[0]).max() > QP_TOL


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
