"""The space-time GW Dyson step, serially and split over frequencies under
simulated ranks.

`solve_qp_energy_space_time` turns chi0(i.omega) into W(i.omega) =
[I - chi0(i.omega)]^-1 one frequency at a time, in a continuum taken into the
bare gauge (`transform`) first. The static W a BSE takes and the Eq. (18)
shift reads is not on this axis: it is the one static build,
`static_screening_matrix` on `static_screening_grid`, dressed. Gated on
water/cc-pVDZ and ethylene/cc-pVTZ Hartree-Fock and on water/cc-pVDZ in an
IEF-PCM water continuum, each on the GW window and on the whole diagonal with
the static W.

Serially (`_dyson_in_place`) the energies are those of a pinned tree whose
Dyson step is not split over frequencies, unpacked into a temporary directory
and run in its own process on the mean field and factors this process built,
with the continuum's cavity handed to both (`CAVITY`) and its W frequency axis
moved onto rW, the range its transform is fitted over; at the automatic tau
count and at 6 points, bitwise but for the continuum at 6 points and a few
last bits where the root search changed. The pinned tree solved the
continuum's G0W0 equation with Eq. (18) inside it; this tree adds it after
the solve (`reaction_field.shift_after_solve`), so the continuum is compared
anchored on the mean-field spectrum (`eps_anchor`), the one placement both
trees share. One rank's `dyson_frequencies` is
every frequency, and W(0) is the one static build, to a few ulp (built
twice). The pinned tree
carried W(0) as a zero-weight omega = 0 row of the GW axis; taking that row
out leaves every energy bitwise, and taking W(0) from the static build moves
only the continuum's Eq. (18) shift on the 6-point axis, by at most
`ARCHIVE_MOVE`.

Over more than one rank the in-core route holds chi0 by auxiliary rows
(`_qp_grid_rows`). `_dyson_owned` gathers each frequency's chi0 whole to its
round-robin owner and inverts it there, and `screened_interaction_rows` folds
each owner's W - I into every rank's rows of Wt(i.tau). None of that is summed
across ranks; each gate still compares two computations (a rank's and the
serial reference's), so to a few ulp. At 2, 3 and 8 simulated ranks, and at
8 on 6 tau points (rank 7 owns no frequency):
  * the frequencies the ranks invert cover every frequency once, each on its
    round-robin owner; each rank's `dyson_frequencies` counts its own and
    `t_dyson` is set;
  * each owner's W - I is the serial Dyson step on the same chi0, gathered
    whole, to a few ulp, the bare gauge included in the continuum;
  * W(0) is the serial static build, to a few ulp, on every rank and in what the
    driver hands a BSE;
  * every rank's rows of Wt(i.tau) are the rows of one rank's fold on that
    serial W - I, to a few ulp.
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

from tests.archived_tree import require_commit
from tests.archive_radii import install_current_radii
from src.Base.numerics.constants import PCM_LEBEDEV_ORDER
from src.Base.numerics.numerics import Numerics
from src.environments.solvent_screening import attach_solvent_screening
from src.Base.utils.parallel.mpi_grid import contiguous_block, partition
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW import space_time
from src.SingleReference.GW.imaginary_time import screened_interaction_rows
from src.SingleReference.LinearResponse.space_time import (
    static_screening_grid, static_screening_matrix)
from src.SingleReference.GW.space_time import (_dyson_in_place, _dyson_owned,
                                               separable_factors,
                                               solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)
from tests.isdf_test_grid import g1
from tests.pins import RECORDING_BLAS, THREAD_CAPS, matches

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
#: Six tau points: 6 frequencies, fewer than 8 ranks.
EMPTY_NTAU = 6
#: (ranks, tau points) of the distributed gates.
LAYOUTS = [(2, 'auto'), (3, 'auto'), (8, 'auto'), (8, EMPTY_NTAU)]
#: The planted defects' rank count: every rank owns two frequencies or more.
PLANT_SIZE = 3
#: Added to the diagonal of one owner's W - I: far above any last bit.
PERTURBATION = 1e-6
#: Ha. The continuum's energies against the archived code at EMPTY_NTAU: its
#: Eq. (18) shift reads W(0) off the converged static build, the archive's off
#: a 6-point GW axis; measured 8.3e-7 (window) and 1.08e-6 (diagonal),
#: water/PCM.
ARCHIVE_MOVE = 1.2e-6
#: Ha. Every other energy against the archived code: the scan route finds each
#: root by its graphical and largest-weight scans only, where the archive fell
#: back on a plain Newton, and the vectorized scan bisects each orbital's two
#: best-ranked brackets, so a root reached by the other path lands a few last
#: bits away. Measured on the diagonal 1.1e-16
#: and 8.9e-16 (water at 6 points, orbitals 6 and 18 of 24), 8.9e-16 (ethylene
#: at the automatic count, orbital 52) and up to 8.9e-16 on six orbitals
#: (ethylene at 6 points), and on the window 1.1e-16 (ethylene LUMO at 6
#: points); the rest stays bitwise.
ROOT_SEARCH_MOVE = 4e-15

GATE_ONCE = 'every frequency inverted exactly once'
GATE_OWNER = 'each frequency inverted on its round-robin owner'
GATE_COUNT = "each rank's dyson_frequencies its own count, t_dyson set"
GATE_W = "each owner's W - I the serial Dyson step on its chi0, to a few ulp"
GATE_W0 = 'W(0) the serial static build on every rank, to a few ulp'
GATE_WT = ("every rank's Wt rows one rank's fold on the serial W - I, to a "
           'few ulp')
#: defect: (system, kind, `DysonSpy` plant, the gates it must fail).
PLANTS = {
    'a frequency inverted twice': (
        'water', 'diagonal', 'twice', {GATE_ONCE, GATE_OWNER, GATE_COUNT}),
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

try:
    from src.environments.solvent_screening import attach_solvent_screening
except ImportError:  # an older tree's module layout
    from src.Base.solvent_screening import attach_solvent_screening
from src.SingleReference.GW.space_time import (solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)

import src.SingleReference.GW.space_time as _archived_space_time
try:
    from src.Base.utils.grids.grids import minimax_frequency_grid as _minimax_frequency_grid
except ImportError:  # an older tree's module layout
    from src.Base.utils.grids import minimax_frequency_grid as _minimax_frequency_grid
try:
    from src.Base.utils.grids.time_frequency import SELF_ENERGY_PAD as _PAD
except ImportError:  # an older tree's module layout
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
    rank -- the frequencies it inverted, rank 0's chi0 gathered whole, the
    W - I it returned, the Wt rows of the fold -- recorded in
    `records[rank]`, with the defect `plant` names planted:

      'twice'               rank 1 also inverts frequency 0
      'orphaned screened'   rank 1 skips its first frequency
      'no gauge'            the step is handed no bare-gauge transform
      'swapped'             rank 1 hands the fold its first two W - I swapped
      'perturbed'           rank 1's first W - I moves by `PERTURBATION`
    """

    def __init__(self, monkeypatch, plant=None):
        self.records, self.plant = {}, plant
        monkeypatch.setattr(space_time, '_dyson_owned', self._dyson_owned)
        monkeypatch.setattr(space_time, 'screened_interaction_rows',
                            self._screened)

    def _record(self, comm):
        return self.records.setdefault(_rank(comm), {})

    def _wants(self, rank, wants):
        if self.plant == 'twice' and rank == 1:
            return wants + [0]
        if self.plant == 'orphaned screened' and rank == 1:
            return wants[1:]
        return wants

    def _dyson_owned(self, chi0, transform=None):
        rank, rec = _rank(chi0.comm), self._record(chi0.comm)
        gather = chi0.gather_slices
        whole = np.stack([c.copy() for _, c in gather(range(chi0.shape[0]))])
        if rank == 0:
            rec['chi0'] = whole
        del whole
        inverted = []

        def gather_owned(wants):
            wants = self._wants(rank, [int(k) for k in wants])
            inverted.extend(wants)
            return gather(wants)

        chi0.gather_slices = gather_owned
        gauge = None if self.plant == 'no gauge' else transform
        W_minus_I = _dyson_owned(chi0, gauge)
        if self.plant == 'perturbed' and rank == 1:
            W = W_minus_I[min(W_minus_I)]
            W[np.diag_indices(W.shape[0])] += PERTURBATION
        rec.update(inverted=inverted, transform=transform,
                   W={k: v.copy() for k, v in W_minus_I.items()})
        return W_minus_I

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
        factors = separable_factors(mf, mol, auxbasis=basis + '-ri', counts=g1(mol))
        out[name] = dict(mol=mol, mf=mf, factors=factors,
                         nocc=mol.nelectron // 2)
    return out


@pytest.fixture(scope='module')
def archived(systems, tmp_path_factory):
    """Every serial call of this file on the pinned tree, unpacked into a
    temporary directory and run in one process."""
    tmp = tmp_path_factory.mktemp('dyson_baseline')
    tar = tmp / f'{BASELINE_COMMIT}.tar'
    require_commit(REPO, BASELINE_COMMIT)
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    tree = tmp / 'tree'
    with tarfile.open(tar) as fh:
        fh.extractall(tree)
    install_current_radii(tree)
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


def _route(s, kind, ntau='auto', comm=None, inside=False):
    """(quasiparticle energies, W(0) or an empty array, timings) on this
    rank's own copy of the mean field and factors, which a distributed solve
    overwrites in place with rank 0's. `inside`: Eq. (18) inside the
    equation, anchored on the mean-field spectrum, as the pinned tree
    placed it."""
    mf = copy.copy(s['mf'])
    mf.mo_energy = np.asarray(s['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(s['mf'].mo_coeff, float).copy()
    factors = tuple(np.array(a, copy=True) for a in s['factors'])
    nocc, timings = s['nocc'], {}
    kw = dict(factors=factors, numerics=Numerics(quadrature={'ntau_st': ntau}),
              timings=timings,
              distribute=comm is not None, comm=comm)
    if inside:
        kw['eps_anchor'] = np.asarray(s['mf'].mo_energy, float).copy()
    if kind == 'window':
        qp = solve_qp_energy_space_time(mf, s['mol'], nocc,
                                        np.array([nocc - 1, nocc]), **kw)
        return qp, np.zeros(0), timings
    extras = {}
    qp, _ = solve_qp_diagonal_space_time(mf, s['mol'], nocc, extras=extras,
                                         **kw)
    return qp, extras['w_static'], timings


def _same(a, b):
    """Two computations of one array (a rank's against the serial one, two
    builds of W(0)) to a few ulp (`matches` of tests/pins.py)."""
    return matches(a, b, RECORDING_BLAS, quantity='root')


def _correlation_part(W):
    """W - I, the identity taken off the diagonal in place as the Dyson step
    takes it."""
    W[np.diag_indices(W.shape[0])] -= 1.0
    return W


def _static_w(s):
    """W(0) of the one static build on the system's factors, serially."""
    eps = np.asarray(s['mf'].mo_energy, float)
    X_mo, D = s['factors'][:2]
    return static_screening_matrix(X_mo, D, eps, s['nocc'],
                                   static_screening_grid(eps, s['nocc']))


def _dyson_gates(records, outs, w_static):
    """{gate: passed} for one distributed run: `records` the `DysonSpy`'s,
    `outs` every rank's (energies, W(0), timings), `w_static` the serial
    static build. The serial reference is `_dyson_in_place` on rank 0's
    gathered chi0, and one rank's `screened_interaction_rows` on its W - I."""
    size = len(outs)
    recs = [records[r] for r in range(size)]
    chi0 = recs[0]['chi0']
    nfreq, naux = chi0.shape[0], chi0.shape[-1]
    W = _dyson_in_place(chi0.copy(), range(nfreq), recs[0]['transform'])
    W_minus_I = {k: _correlation_part(W[k].copy()) for k in range(nfreq)}
    wt_one = screened_interaction_rows(W_minus_I, recs[0]['ctw'], naux,
                                       None).rows
    counts = [t.get('dyson_frequencies') for _, _, t in outs]
    return {
        GATE_ONCE: sorted(k for rec in recs for k in rec['inverted'])
        == list(range(nfreq)),
        GATE_OWNER: all(sorted(rec['inverted'])
                        == [int(k) for k in partition(nfreq, r, size)]
                        for r, rec in enumerate(recs)),
        GATE_COUNT: (counts == [len(rec['W']) for rec in recs]
                     and sum(counts) == nfreq
                     and all(t.get('t_dyson', -1.0) >= 0.0
                             for _, _, t in outs)),
        GATE_W: all(sorted(rec['W']) == sorted(rec['inverted'])
                    and all(_same(Wk, W_minus_I[k])
                            for k, Wk in rec['W'].items())
                    for rec in recs),
        GATE_W0: all(ws.size == 0 or _same(ws, w_static)
                     for _, ws, _ in outs),
        GATE_WT: all(_same(rec['wt'],
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
    """One rank inverts every frequency: the energies (but for the
    continuum's shift on the 6-point axis and the root search's last bits)
    are the pinned tree's, and W(0) is the static build, to a few ulp. The
    continuum keeps Eq. (18) inside the equation, as the pinned tree did."""
    solvated = SYSTEMS[name][2] is not None
    for ntau in ('auto', EMPTY_NTAU):
        qp, ws, t = _route(systems[name], kind, ntau, inside=solvated)
        key = f'{name}_{kind}_{ntau}'
        if SYSTEMS[name][2] is not None and ntau == EMPTY_NTAU:
            assert np.abs(qp - archived[key + '_qp']).max() <= ARCHIVE_MOVE
        else:
            assert np.abs(qp - archived[key + '_qp']).max() <= ROOT_SEARCH_MOVE
        assert ws.size == 0 or _same(ws, _static_w(systems[name]))
        assert t['dyson_frequencies'] == t.get('ntau_auto', ntau)


@pytest.mark.parametrize('size, ntau', LAYOUTS)
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(SYSTEMS))
def test_owned_frequency_dyson(systems, monkeypatch, name, kind, size, ntau):
    """Each frequency inverted once, on its owner, as the serial step inverts
    it; W(0) the static build on every rank; Wt's rows one rank's."""
    spy = DysonSpy(monkeypatch)
    outs = run_simulated(lambda c: _route(systems[name], kind, ntau, c), size)
    assert not _failed(_dyson_gates(spy.records, outs,
                                    _static_w(systems[name])))
    solvated = SYSTEMS[name][2] is not None
    assert (spy.records[0]['transform'] is not None) == solvated
    if ntau == EMPTY_NTAU:
        assert not spy.records[size - 1]['inverted']


@pytest.mark.parametrize('defect', list(PLANTS))
def test_planted_defect_fails_its_gates(systems, monkeypatch, defect):
    """The defect fails exactly the gates that cover it."""
    name, kind, plant, expected = PLANTS[defect]
    spy = DysonSpy(monkeypatch, plant)
    outs = run_simulated(lambda c: _route(systems[name], kind, comm=c),
                         PLANT_SIZE)
    assert _failed(_dyson_gates(spy.records, outs,
                                _static_w(systems[name]))) == expected


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
