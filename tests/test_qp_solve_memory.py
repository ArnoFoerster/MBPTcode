"""What the quasiparticle set solve holds at once, counted, on every route.

`qp_set_gradient` carries two families of large arrays: proj(tau) and its
adjoints, (ntau, naux, naux), and on the explicit residue route the
particle-hole block C_ov = B[:, occ, virt] and its adjoint, (naux, nocc*nvir).
At the chlorophyllide hexamer one of the first is 153 GB and one of the second
2.2 TB, so the count alive at once is the per-rank memory of the solve. A line
tracer walks every frame between the solve and the line being run and counts
the distinct buffers of each shape bound to a name there -- locals, containers
and the residue backends' attributes -- and the largest count over the call is
compared with what the solve is built to hold:

  * the forward pass (every weight zero): one proj(tau), no adjoint, no
    reverse sweep (`polarizability_backward` never called), and on the
    explicit route C_ov alone;
  * the reverse pass: proj(tau) and ONE projbar, whatever the number of
    states -- the Laplace residues' adjoints are recorded per push and folded
    into the integral term's projbar one tau slice at a time -- and one sweep;
    on the explicit route C_ov, at most one Cov_bar, freed after its state,
    and the product each push forms before adding it;
  * the explicit route is refused above `EXPLICIT_RESIDUE_MAX_GB` with a
    message naming the route and the block's size, and runs at the limit;
  * no (ntau, naux, naux) temporary: every tensordot of the reverse pass
    returns at most one auxiliary row's worth, (max(nfreq, ntau), naux) --
    the fold of each frequency block into projbar is one row at a time
    (`ProjRows.fold`), where it was one whole-array product per block;
  * over 2, 3 and 8 simulated ranks, proj(tau) and projbar are each rank's
    auxiliary ROWS (`ProjRows`): no rank holds a whole (ntau, naux, naux)
    array at any line, the forward holds one row block and the reverse two,
    and the bytes read off the rank's objects are its share,
    ntau (rows) naux 8 each; the census run on a ProjRows that keeps the
    whole array on every rank beside its rows (planted) finds it.

Water and ethylene/cc-pVDZ, RHF, 148 points per atom, the frontier set widened
by one deeper occupied state so that the Laplace and explicit backends take
residues. A per-state accumulator (a projbar or a Cov_bar allocated for every
state of the set) fails the reverse-pass counts.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

import src.gradients.qp_space_time as qp_space_time
import src.gradients.space_time_adjoint as space_time_adjoint
from src.Base.constants import EXPLICIT_RESIDUE_MAX_GB
from src.Base.utils.mpi_grid import contiguous_block, run_simulated
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    ProjRows, polarizability_projected_sweep)
from src.gradients.excited_state import ExcitedStateChain

BASIS = 'cc-pvdz'
MOLS = {
    'water': 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
    'ethylene': ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; '
                 'H 0 -0.9289 -1.2321'),
}
ROUTES = ('explicit', 'laplace', 'sop')
#: (ntau, naux, naux) arrays alive at once: the forward holds proj(tau)
#: alone, the reverse proj(tau) and one projbar.
PROJ_FORWARD, PROJ_REVERSE = 1, 2
#: (naux, nocc*nvir) arrays alive at once on the explicit route: C_ov in the
#: forward; in the reverse C_ov, one Cov_bar and the product a push forms
#: before adding it (`screening_chain`), whatever the number of states.
OV_FORWARD, OV_REVERSE = 1, 3
SIZES = [2, 3, 8]


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


class Census:
    """The largest number of distinct proj-shaped and ov-shaped buffers bound
    to a name in any frame between the solve and the running line: 'proj' a
    whole (ntau, naux, naux) array, 'rows' a rank's (ntau, rows, naux) share
    of one, 'ov' a particle-hole block; and the most bytes of the first two
    alive at one line."""

    def __init__(self, proj_shape, ov_shapes):
        self.proj_shape, self.ov_shapes = proj_shape, ov_shapes
        self.peak = {'proj': 0, 'rows': 0, 'ov': 0, 'bytes': 0}
        self.root = qp_space_time.qp_set_gradient.__code__

    def classify(self, a):
        base = a
        while isinstance(base.base, np.ndarray):
            base = base.base
        ntau, naux = self.proj_shape[0], self.proj_shape[-1]
        if base.shape == self.proj_shape:
            return 'proj', base
        if (base.ndim == 3 and base.shape[0] == ntau and base.shape[2] == naux
                and 0 < base.shape[1] < naux):
            return 'rows', base
        if base.shape in self.ov_shapes:
            return 'ov', base
        return None, None

    def visit(self, obj, found, depth=0):
        if isinstance(obj, np.ndarray):
            kind, base = self.classify(obj)
            if kind is not None:
                found[kind][id(base)] = base.nbytes
        elif depth > 3:
            return
        elif isinstance(obj, dict):
            for v in obj.values():
                self.visit(v, found, depth + 1)
        elif isinstance(obj, (list, tuple)) and len(obj) < 256:
            for v in obj:
                self.visit(v, found, depth + 1)
        elif ('RealScreening' in type(obj).__name__
              or isinstance(obj, ProjRows)):
            self.visit(vars(obj), found, depth + 1)

    def count(self, frame):
        found = {'proj': {}, 'rows': {}, 'ov': {}}
        f = frame
        while f is not None:
            for v in f.f_locals.values():
                self.visit(v, found)
            if f.f_code is self.root:
                break
            f = f.f_back
        for kind in found:
            self.peak[kind] = max(self.peak[kind], len(found[kind]))
        held = sum(found['proj'].values()) + sum(found['rows'].values())
        self.peak['bytes'] = max(self.peak['bytes'], held)

    def tracer(self, frame, event, arg):
        if not frame.f_code.co_filename.startswith(REPO_SRC):
            return None

        def local(frame, event, arg):
            if event in ('line', 'return'):
                self.count(frame)
            return local
        return local


class RankCensus:
    """One `Census` per rank-thread of a simulated world, counting at the lines
    of the modules that allocate proj-shaped arrays (`SOLVE_FILES`); a line
    elsewhere -- the Newton, a collective -- is seen from the frames below
    it at the next line that is counted."""

    def __init__(self, proj_shape, ov_shapes):
        self.make = lambda: Census(proj_shape, ov_shapes)
        self.of_thread, self.rank_of = {}, {}

    def tracer(self, frame, event, arg):
        if frame.f_code.co_filename not in SOLVE_FILES:
            return None
        census = self.of_thread.setdefault(threading.get_ident(), self.make())
        return census.tracer(frame, event, arg)

    def by_rank(self):
        return {self.rank_of[t]: c.peak for t, c in self.of_thread.items()
                if t in self.rank_of}


class LargestTensordot:
    """numpy, save a tensordot that records the bytes of what it returns."""

    def __init__(self):
        self.largest = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def tensordot(self, a, b, axes=2):
        out = np.tensordot(a, b, axes)
        self.largest = max(self.largest, np.asarray(out).nbytes)
        return out


REPO_SRC = str(qp_space_time.__file__).rsplit('/src/', 1)[0] + '/src/'
#: Where the solve's proj-shaped arrays are made and bound.
SOLVE_FILES = frozenset(m.__file__ for m in (qp_space_time, space_time_adjoint,
                                             ls_space_time))


@pytest.fixture(scope='module', params=sorted(MOLS))
def system(request):
    mol = gto.M(atom=MOLS[request.param], basis=BASIS, verbose=0)
    mf = scf_factory(mol)
    ch = ExcitedStateChain(mol, scf_factory, mf=mf, solver='davidson',
                           nroots=5)
    x_mo, d, eps = ch.factors_at(mol, mf)[:3]
    nocc = ch.nocc
    states = np.union1d(ch.qp_set, [nocc - 3])
    return dict(x_mo=x_mo, d=d, eps=eps, nocc=nocc, states=states,
                grid=ch.gw_grid, nu=ch.nu, wt=ch.wt,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]))


def qp_set_gradient(s, route, weights, **kw):
    ro = {}
    out = qp_space_time.qp_set_gradient(
        s['x_mo'], s['d'], s['eps'], s['nocc'], s['grid'], s['nu'], s['wt'],
        s['states'], weights, mu=s['mu'], residue_route=route, route_out=ro,
        **kw)
    return out, ro


def shapes_of(s):
    """(the whole proj(tau) shape, the ov-block shapes) of a system."""
    naux, nocc = s['d'].shape[1], s['nocc']
    nvir = s['x_mo'].shape[1] - nocc
    return ((s['grid'].ntau, naux, naux),
            {(naux, nocc * nvir), (naux, nocc, nvir)})


def counted(s, route, weights, monkeypatch):
    """(peak counts, polarizability_backward calls, route_out) of one solve."""
    naux, nocc = s['d'].shape[1], s['nocc']
    nvir = s['x_mo'].shape[1] - nocc
    census = Census((s['grid'].ntau, naux, naux),
                    {(naux, nocc * nvir), (naux, nocc, nvir)})
    calls = []
    sweep = qp_space_time.polarizability_backward

    def counting_sweep(*args, **kwargs):
        calls.append(1)
        return sweep(*args, **kwargs)

    monkeypatch.setattr(qp_space_time, 'polarizability_backward',
                        counting_sweep)
    sys.settrace(census.tracer)
    try:
        _, ro = qp_set_gradient(s, route, weights)
    finally:
        sys.settrace(None)
    return census.peak, len(calls), ro


@pytest.mark.parametrize('route', ROUTES)
def test_forward_holds_one_proj_and_runs_no_sweep(system, route, monkeypatch):
    """Every weight zero: proj(tau) alone, no adjoint, no reverse sweep."""
    peak, calls, ro = counted(system, route, np.zeros(len(system['states'])),
                              monkeypatch)
    assert calls == 0, f'{route}: {calls} reverse sweeps on a zero adjoint'
    assert peak['proj'] == PROJ_FORWARD, (route, peak)
    if 'explicit' in ro['routes'].values():
        assert peak['ov'] == OV_FORWARD, (route, peak)
    else:
        assert peak['ov'] == 0, (route, peak)


@pytest.mark.parametrize('route', ROUTES)
def test_reverse_holds_one_projbar_whatever_the_set(system, route,
                                                    monkeypatch):
    """Every weight non-zero: proj(tau) and one projbar, one sweep; on the
    explicit route C_ov, at most one Cov_bar and a push's product."""
    weights = np.linspace(0.4, 1.3, len(system['states']))
    peak, calls, ro = counted(system, route, weights, monkeypatch)
    assert calls == 1, f'{route}: {calls} reverse sweeps'
    assert peak['proj'] == PROJ_REVERSE, (route, peak)
    backends = set(ro['routes'].values())
    if route != 'sop':
        # the residue backend the route names is exercised, not bypassed
        assert route in backends, ro['routes']
    if 'explicit' in backends:
        assert peak['ov'] == OV_REVERSE, (route, peak)
    else:
        assert peak['ov'] == 0, (route, peak)


def test_explicit_route_refused_above_the_limit(system, monkeypatch):
    """Refused just above C_ov's size, with the route and the size named;
    run at exactly the limit."""
    naux, nocc = system['d'].shape[1], system['nocc']
    nvir = system['x_mo'].shape[1] - nocc
    block_gb = naux * nocc * nvir * 8 / 1e9
    assert block_gb < EXPLICIT_RESIDUE_MAX_GB
    weights = np.ones(len(system['states']))
    s = system
    single = (s['x_mo'], s['d'], s['eps'], nocc, s['grid'], s['nu'], s['wt'],
              nocc - 2)
    monkeypatch.setattr(qp_space_time, 'EXPLICIT_RESIDUE_MAX_GB',
                        block_gb * (1.0 - 1e-9))
    for run in (lambda: qp_set_gradient(s, 'explicit', weights),
                lambda: qp_space_time.qp_gradient_space_time(
                    *single, mu=s['mu'], residue_route='explicit')):
        with pytest.raises(MemoryError) as err:
            run()
        message = str(err.value)
        assert ("'explicit'" in message
                and f'{block_gb:.3g} GB' in message), message
        assert "'laplace'" in message and "'sop'" in message, message
    monkeypatch.setattr(qp_space_time, 'EXPLICIT_RESIDUE_MAX_GB', block_gb)
    out, ro = qp_set_gradient(s, 'explicit', weights)
    assert 'explicit' in ro['routes'].values()
    ro = {}
    qp_space_time.qp_gradient_space_time(*single, mu=s['mu'],
                                         residue_route='explicit',
                                         route_out=ro)
    assert ro['residue_route'] == 'explicit'


@pytest.mark.parametrize('route', ROUTES)
def test_reverse_folds_one_row_at_a_time(system, route, monkeypatch):
    """No tensordot of the solve returns more than one auxiliary row."""
    s = system
    naux = s['d'].shape[1]
    record = LargestTensordot()
    for module in (qp_space_time, space_time_adjoint, ls_space_time):
        monkeypatch.setattr(module, 'np', record)
    qp_set_gradient(s, route, np.linspace(0.4, 1.3, len(s['states'])))
    row = max(len(s['nu']), s['grid'].ntau) * naux * 8
    assert 0 < record.largest <= row, (route, record.largest, row)


def census_over_ranks(s, route, weights, size):
    """{rank: peak} of one solve on `size` simulated ranks."""
    proj_shape, ov_shapes = shapes_of(s)
    census = RankCensus(proj_shape, ov_shapes)

    def rank(comm):
        census.rank_of[threading.get_ident()] = comm.Get_rank()
        return qp_set_gradient(s, route, weights, comm=comm)

    threading.settrace(census.tracer)
    try:
        run_simulated(rank, size)
    finally:
        threading.settrace(None)
    return census.by_rank()


def assert_rows_only(s, peaks, size, blocks):
    """No whole proj on any rank; `blocks` row blocks, the rank's share."""
    ntau, naux = s['grid'].ntau, s['d'].shape[1]
    assert sorted(peaks) == list(range(size))
    for rank, peak in peaks.items():
        r0, r1 = contiguous_block(naux, rank, size)
        share = ntau * (r1 - r0) * naux * 8
        assert peak['proj'] == 0, (size, rank, peak)
        assert peak['rows'] == blocks, (size, rank, peak)
        assert peak['bytes'] == blocks * share, (size, rank, peak, share)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('route', ROUTES)
def test_ranks_hold_their_rows_alone(system, route, size):
    """Forward one row block, reverse two, per rank; never a whole proj."""
    s = system
    n = len(s['states'])
    assert_rows_only(s, census_over_ranks(s, route, np.zeros(n), size), size,
                     PROJ_FORWARD)
    assert_rows_only(s, census_over_ranks(s, route, np.linspace(0.4, 1.3, n),
                                          size), size, PROJ_REVERSE)


def test_the_census_finds_proj_held_whole(system, monkeypatch):
    """Planted: every rank keeps the whole sweep beside its rows."""
    s = system
    rows = qp_space_time.polarizability_projected_rows

    def whole_on_every_rank(X, D, eps, nocc, tau_points, mu=None,
                            tile_memory_gb=None, comm=None):
        proj = rows(X, D, eps, nocc, tau_points, mu=mu,
                    tile_memory_gb=tile_memory_gb, comm=comm)
        proj.whole = polarizability_projected_sweep(
            X, D, eps, nocc, tau_points, mu=mu,
            tile_memory_gb=tile_memory_gb)
        return proj

    monkeypatch.setattr(qp_space_time, 'polarizability_projected_rows',
                        whole_on_every_rank)
    peaks = census_over_ranks(s, 'sop', np.linspace(0.4, 1.3,
                                                    len(s['states'])), 2)
    with pytest.raises(AssertionError):
        assert_rows_only(s, peaks, 2, PROJ_REVERSE)
    assert all(peak['proj'] == 1 for peak in peaks.values()), peaks


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
