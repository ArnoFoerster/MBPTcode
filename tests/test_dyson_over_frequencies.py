"""The space-time GW Dyson step, serially and split over frequencies under
simulated ranks.

`solve_qp_energy_space_time` turns chi0(i.omega) into W(i.omega) =
[I - chi0(i.omega)]^-1 one frequency at a time: the omega = 0 passenger (the
static W a BSE takes) is inverted dressed, and in a continuum every other
frequency is first taken into the bare gauge (`transform`). Gated on
water/cc-pVDZ and ethylene/cc-pVTZ Hartree-Fock and on water/cc-pVDZ in an
IEF-PCM water continuum, each on the GW window (no passenger in the gas
phase) and on the whole diagonal with the static W.

Serially (`_dyson_in_place`) the energies and W(0) are bitwise those of a
pinned tree whose Dyson step is not split over frequencies, unpacked into a
temporary directory and run in its own process on the mean field and factors
this process built, with the continuum's cavity handed to both (`CAVITY`) and
its W frequency axis moved onto rW, the range its transform is fitted over;
at the automatic tau count and at 6 points. One rank's `dyson_frequencies` is
every frequency.

Over more than one rank the in-core route holds chi0 by auxiliary rows
(`_qp_grid_rows`). `_dyson_owned` gathers each frequency's chi0 whole to its
round-robin owner and inverts it there, W(0) is broadcast from its owner, and
`screened_interaction_rows` folds each owner's W - I into every rank's rows of
Wt(i.tau). None of that is summed across ranks, so every gate is bitwise. At
2, 3 and 8 simulated ranks, and at 8 on 6 tau points (rank 7 owns no
frequency):
  * the frequencies the ranks invert cover every frequency once, each on its
    round-robin owner, the passenger on the rank that broadcasts W(0); each
    rank's `dyson_frequencies` counts its own and `t_dyson` is set;
  * each owner's W - I is the serial Dyson step on the same chi0, gathered
    whole, bitwise, the bare gauge included in the continuum;
  * W(0) is the serial step's dressed passenger, bitwise, on every rank and in
    what the driver hands a BSE;
  * every rank's rows of Wt(i.tau) are the rows of one rank's fold on that
    serial W - I, bitwise.
chi0 itself (proj(tau) is a sum over grid-row tiles, so it reassociates) and
the energies against serial are gated in tests/test_simulated_ranks.py.

Each defect in `PLANTS`, planted at 3 ranks, fails exactly the gates named
there; a screened frequency inverted by no rank is refused by the fold on its
owner (KeyError).
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

from src.Base.constants import PCM_LEBEDEV_ORDER
from src.Base.solvent_screening import attach_solvent_screening
from src.Base.utils.mpi_grid import (broadcast_rows, contiguous_block,
                                     partition, run_simulated)
from src.SingleReference.GW import space_time
from src.SingleReference.GW.imaginary_time import screened_interaction_rows
from src.SingleReference.GW.space_time import (_dyson_in_place, _dyson_owned,
                                               separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: (geometry, basis, continuum solvent or None for the gas phase).
SYSTEMS = {
    'water': (WATER, 'cc-pvdz', None),
    'ethylene': ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321',
                 'cc-pvtz', None),
    'water-pcm': (WATER, 'cc-pvdz', 'water'),
}
KINDS = ('window', 'diagonal')
#: Six tau points: with the passenger 7 frequencies, fewer than 8 ranks.
EMPTY_NTAU = 6
#: (ranks, tau points) of the distributed gates.
LAYOUTS = [(2, 'auto'), (3, 'auto'), (8, 'auto'), (8, EMPTY_NTAU)]
#: The planted defects' rank count: every rank owns two frequencies or more.
PLANT_SIZE = 3
#: Added to the diagonal of one owner's W - I: far above any last bit.
PERTURBATION = 1e-6

GATE_ONCE = 'every frequency inverted exactly once'
GATE_OWNER = 'each frequency inverted on its round-robin owner'
GATE_PASSENGER = 'W(0) inverted on the rank that broadcasts it'
GATE_COUNT = "each rank's dyson_frequencies its own count, t_dyson set"
GATE_W = "each owner's W - I the serial Dyson step on its chi0, bitwise"
GATE_W0 = 'W(0) the serial dressed passenger on every rank, bitwise'
GATE_WT = "every rank's Wt rows one rank's fold on the serial W - I, bitwise"
#: defect: (system, kind, `DysonSpy` plant, the gates it must fail).
PLANTS = {
    'a frequency inverted twice': (
        'water', 'diagonal', 'twice', {GATE_ONCE, GATE_OWNER, GATE_COUNT}),
    'the passenger inverted by no rank': (
        'water', 'diagonal', 'orphaned passenger',
        {GATE_ONCE, GATE_OWNER, GATE_PASSENGER, GATE_W0}),
    'W(0) not broadcast': ('water', 'diagonal', 'no broadcast', {GATE_W0}),
    'W(0) broadcast from the wrong rank': (
        'water', 'diagonal', 'wrong root', {GATE_PASSENGER, GATE_W0}),
    'a Wt row from the wrong frequency': (
        'water', 'diagonal', 'swapped', {GATE_WT}),
    "one owner's W - I perturbed": (
        'water', 'diagonal', 'perturbed', {GATE_W, GATE_WT}),
    'the bare gauge dropped': (
        'water-pcm', 'window', 'no gauge', {GATE_W, GATE_WT}),
}

#: Six points is a partition fixture, not a converged grid.
pytestmark = pytest.mark.filterwarnings(
    'ignore:minimax transform fit reached only:RuntimeWarning')

REPO = Path(__file__).resolve().parents[1]
#: A tree whose Dyson step every rank runs over every frequency.
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: The continuum's cavity, handed to both trees: the pinned tree's default
#: is pyscf's Lebedev order 29, not `PCM_LEBEDEV_ORDER`, which would
#: discretize another continuum.
CAVITY = dict(lebedev_order=PCM_LEBEDEV_ORDER)
#: The serial calls on the unpacked tree, in its own process, on the mean
#: field and factors this process built (`{inputs}`) and on `CAVITY`, so only
#: the GW route differs.
ARCHIVED_PROBE = '''
import copy
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.Base.solvent_screening import attach_solvent_screening
from src.SingleReference.GW.space_time import (solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

import src.SingleReference.GW.space_time as _archived_space_time
from src.Base.utils.grids import minimax_frequency_grid as _minimax_frequency_grid
from src.Base.utils.time_frequency import SELF_ENERGY_PAD as _PAD

# the one change since the pinned commit: W's frequencies span rW, the range
# its transform back to tau is fitted over, not the bare window
_archived_space_time.minimax_frequency_grid = (
    lambda n, e_min, e_max: _minimax_frequency_grid(n, _PAD[0] * e_min,
                                                    _PAD[1] * e_max))

warnings.simplefilter('ignore')
out = {{}}
for name, (atom, basis, solvent) in {systems!r}.items():
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=basis + '-ri')
    mf.kernel()
    if solvent is not None:
        attach_solvent_screening(mf, solvent=solvent, **{cavity!r})
    d = np.load({inputs!r}.format(name=name))
    mf.mo_energy, mf.mo_coeff = d['mo_energy'], d['mo_coeff']
    factors = (d['X_mo'], d['D'], d['X_ao'], d['coords'])
    nocc = mol.nelectron // 2
    for kind in {kinds!r}:
        for ntau in ('auto', {empty_ntau!r}):
            m = copy.copy(mf)
            kw = dict(factors=tuple(np.array(a, copy=True) for a in factors),
                      ntau=ntau, distribute=False)
            if kind == 'window':
                qp = solve_qp_energy_space_time(
                    m, mol, nocc, np.array([nocc - 1, nocc]), **kw)
                ws = np.zeros(0)
            else:
                extras = {{}}
                qp, _ = solve_qp_diagonal_space_time(m, mol, nocc,
                                                     extras=extras, **kw)
                ws = extras['w_static']
            out[f'{{name}}_{{kind}}_{{ntau}}_qp'] = qp
            out[f'{{name}}_{{kind}}_{{ntau}}_ws'] = ws
np.savez({out!r}, **out)
'''


class DysonSpy:
    """What the owned-frequency Dyson step of the grid-row route did on each
    rank (the frequencies it inverted, rank 0's chi0 gathered whole, the
    W - I and W(0) it returned, the broadcast root, the Wt rows of the fold),
    recorded in `records[rank]`, with the defect `plant` names planted:

      'twice'               rank 1 also inverts frequency 0
      'orphaned passenger'  no rank inverts the passenger
      'orphaned screened'   rank 1 skips its first frequency
      'no gauge'            the step is handed no bare-gauge transform
      'no broadcast'        the W(0) broadcast is a no-op
      'wrong root'          W(0) is broadcast from its owner's successor
      'swapped'             rank 1 hands the fold its first two W - I swapped
      'perturbed'           rank 1's first W - I moves by `PERTURBATION`
    """

    def __init__(self, monkeypatch, plant=None):
        self.records, self.plant = {}, plant
        monkeypatch.setattr(space_time, '_dyson_owned', self._dyson_owned)
        monkeypatch.setattr(space_time, 'broadcast_rows', self._broadcast_rows)
        monkeypatch.setattr(space_time, 'screened_interaction_rows',
                            self._screened)

    def _record(self, comm):
        return self.records.setdefault(_rank(comm), {'roots': []})

    def _wants(self, rank, wants, static_index):
        if self.plant == 'twice' and rank == 1:
            return wants + [0]
        if self.plant == 'orphaned passenger':
            return [k for k in wants if k != static_index]
        if self.plant == 'orphaned screened' and rank == 1:
            return wants[1:]
        return wants

    def _dyson_owned(self, chi0, static_index=None, transform=None):
        rank, rec = _rank(chi0.comm), self._record(chi0.comm)
        gather = chi0.gather_slices
        whole = np.stack([c.copy() for _, c in gather(range(chi0.shape[0]))])
        if rank == 0:
            rec['chi0'] = whole
        del whole
        inverted = []

        def gather_owned(wants):
            wants = self._wants(rank, [int(k) for k in wants], static_index)
            inverted.extend(wants)
            return gather(wants)

        chi0.gather_slices = gather_owned
        gauge = None if self.plant == 'no gauge' else transform
        W_minus_I, w_static = _dyson_owned(chi0, static_index, gauge)
        if self.plant == 'perturbed' and rank == 1:
            W = W_minus_I[min(W_minus_I)]
            W[np.diag_indices(W.shape[0])] += PERTURBATION
        rec.update(inverted=inverted, static=static_index, transform=transform,
                   W={k: v.copy() for k, v in W_minus_I.items()},
                   w0=None if w_static is None else w_static.copy())
        return W_minus_I, w_static

    def _broadcast_rows(self, a, root, comm):
        if self.plant == 'wrong root':
            root = (root + 1) % comm.Get_size()
        self._record(comm)['roots'].append(root)
        if self.plant == 'no broadcast':
            return a
        return broadcast_rows(a, root, comm)

    def _screened(self, W_minus_I, Ctw, naux, comm, *args, **kwargs):
        handed = dict(W_minus_I)
        if self.plant == 'swapped' and _rank(comm) == 1:
            a, b = sorted(handed)[:2]
            handed[a], handed[b] = handed[b], handed[a]
        Wt = screened_interaction_rows(handed, Ctw, naux, comm, *args,
                                       **kwargs)
        self._record(comm).update(wt=Wt.rows.copy(),
                                  ctw=np.array(Ctw, copy=True))
        return Wt


@pytest.fixture(scope='module')
def systems():
    """Each system's mean field, continuum attached, and separable factors."""
    warnings.simplefilter('ignore')
    out = {}
    for name, (atom, basis, solvent) in SYSTEMS.items():
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=basis + '-ri')
        mf.kernel()
        if solvent is not None:
            attach_solvent_screening(mf, solvent=solvent, **CAVITY)
        factors = separable_factors(mf, mol, auxbasis=basis + '-ri')
        out[name] = dict(mol=mol, mf=mf, factors=factors,
                         nocc=mol.nelectron // 2)
    return out


@pytest.fixture(scope='module')
def archived(systems, tmp_path_factory):
    """Every serial call of this file on the pinned tree, unpacked into a
    temporary directory and run in one process."""
    tmp = tmp_path_factory.mktemp('dyson_baseline')
    tar = tmp / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    tree = tmp / 'tree'
    with tarfile.open(tar) as fh:
        fh.extractall(tree)
    assert (tree / 'src' / 'SingleReference' / 'GW'
            / 'space_time.py').is_file()
    for name, s in systems.items():
        X_mo, D, X_ao, coords = s['factors']
        np.savez(tmp / f'inputs_{name}.npz', mo_energy=s['mf'].mo_energy,
                 mo_coeff=s['mf'].mo_coeff, X_mo=X_mo, D=D, X_ao=X_ao,
                 coords=coords)
    out = tmp / 'archived.npz'
    script = tmp / 'probe.py'
    script.write_text(ARCHIVED_PROBE.format(
        archive=str(tree), systems=SYSTEMS, cavity=CAVITY, kinds=KINDS,
        empty_ntau=EMPTY_NTAU, inputs=str(tmp / 'inputs_{name}.npz'),
        out=str(out)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tree),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(out))


def _rank(comm):
    """This rank's index, 0 without a communicator."""
    return 0 if comm is None else comm.Get_rank()


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


def _correlation_part(W):
    """W - I, the identity taken off the diagonal in place as the Dyson step
    takes it."""
    W[np.diag_indices(W.shape[0])] -= 1.0
    return W


def _dyson_gates(records, outs):
    """{gate: passed} for one distributed run: `records` the `DysonSpy`'s,
    `outs` every rank's (energies, W(0), timings). The serial reference is
    `_dyson_in_place` on rank 0's gathered chi0, and one rank's
    `screened_interaction_rows` on its W - I."""
    size = len(outs)
    recs = [records[r] for r in range(size)]
    chi0, static = recs[0]['chi0'], recs[0]['static']
    nfreq, naux = chi0.shape[0], chi0.shape[-1]
    W = _dyson_in_place(chi0.copy(), range(nfreq), static,
                        recs[0]['transform'])
    screened = [k for k in range(nfreq) if k != static]
    W_minus_I = {k: _correlation_part(W[k].copy()) for k in screened}
    wt_one = screened_interaction_rows(W_minus_I, recs[0]['ctw'], naux,
                                       None).rows
    holders = [r for r, rec in enumerate(recs) if static in rec['inverted']]
    counts = [t.get('dyson_frequencies') for _, _, t in outs]
    return {
        GATE_ONCE: sorted(k for rec in recs for k in rec['inverted'])
        == list(range(nfreq)),
        GATE_OWNER: all(sorted(rec['inverted'])
                        == [int(k) for k in partition(nfreq, r, size)]
                        for r, rec in enumerate(recs)),
        GATE_PASSENGER: static is None or (
            len(holders) == 1
            and all(rec['roots'] == holders for rec in recs)),
        GATE_COUNT: (counts == [len(rec['W']) for rec in recs]
                     and sum(counts) == len(screened)
                     and all(t.get('t_dyson', -1.0) >= 0.0
                             for _, _, t in outs)),
        GATE_W: all(sorted(rec['W']) == sorted(k for k in rec['inverted']
                                               if k != static)
                    and all(_bitwise(Wk, W_minus_I[k])
                            for k, Wk in rec['W'].items())
                    for rec in recs),
        GATE_W0: static is None or all(
            _bitwise(rec['w0'], W[static])
            and (ws.size == 0 or _bitwise(ws, W[static]))
            for rec, (_, ws, _) in zip(recs, outs)),
        GATE_WT: all(_bitwise(rec['wt'],
                              wt_one[:, slice(*contiguous_block(naux, r,
                                                                size))])
                     for r, rec in enumerate(recs)),
    }


def _failed(gates):
    """The names of the gates that did not pass."""
    return {gate for gate, passed in gates.items() if not passed}


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(SYSTEMS))
def test_serial_is_the_archived_code(systems, archived, name, kind):
    """One rank inverts every frequency: the energies and the static W are
    the pinned tree's, bit for bit."""
    passenger = kind == 'diagonal' or SYSTEMS[name][2] is not None
    for ntau in ('auto', EMPTY_NTAU):
        qp, ws, t = _route(systems[name], kind, ntau)
        key = f'{name}_{kind}_{ntau}'
        assert _bitwise(qp, archived[key + '_qp'])
        assert _bitwise(ws, archived[key + '_ws'])
        assert t['dyson_frequencies'] == t.get('ntau_auto', ntau) + passenger


@pytest.mark.parametrize('size, ntau', LAYOUTS)
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(SYSTEMS))
def test_owned_frequency_dyson(systems, monkeypatch, name, kind, size, ntau):
    """Each frequency inverted once, on its owner, as the serial step inverts
    it; W(0) on every rank; Wt's rows one rank's."""
    spy = DysonSpy(monkeypatch)
    outs = run_simulated(lambda c: _route(systems[name], kind, ntau, c), size)
    assert not _failed(_dyson_gates(spy.records, outs))
    solvated = SYSTEMS[name][2] is not None
    assert (spy.records[0]['transform'] is not None) == solvated
    assert (spy.records[0]['static'] is not None) == (solvated
                                                      or kind == 'diagonal')
    if ntau == EMPTY_NTAU:
        assert not spy.records[size - 1]['inverted']


@pytest.mark.parametrize('defect', list(PLANTS))
def test_planted_defect_fails_its_gates(systems, monkeypatch, defect):
    """The defect fails exactly the gates that cover it."""
    name, kind, plant, expected = PLANTS[defect]
    spy = DysonSpy(monkeypatch, plant)
    outs = run_simulated(lambda c: _route(systems[name], kind, comm=c),
                         PLANT_SIZE)
    assert _failed(_dyson_gates(spy.records, outs)) == expected


def test_a_screened_frequency_inverted_by_no_rank_is_refused(systems,
                                                             monkeypatch):
    """Rank 1 skipping one of its frequencies leaves the Wt fold on that
    owner without its W - I, which it refuses."""
    DysonSpy(monkeypatch, 'orphaned screened')
    with pytest.raises(KeyError):
        run_simulated(lambda c: _route(systems['water'], 'diagonal', comm=c),
                      PLANT_SIZE)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
