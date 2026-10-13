"""One row fit and one row-fit adjoint per force.

(i) One fit. The distributed ISDF-K SCF holds the row fit's M^T tiles;
`DistributedISDFJK.fit_tiles` hands them out read-only
(`separable_ri.FitTiles`) and `fit_rows(tiles=)` reads them wherever they are
the fit it was asked for:
  * rows read from the tiles are bitwise the rows the fit makes again
    (M^T, X_mo, D and X_ao) at 1, 2 and 3 simulated ranks, a rank owning no
    tile included; another tile edge, regularization or point set refits;
  * the state-pair chain on the row fit reads them at the reference geometry
    and runs no fit of its own there, its factors bitwise its own refit's;
  * the energy route's factor stage (`separable_factors(fit='rows',
    fit_tiles=)`) reads them, its X_mo, D and X_ao rows bitwise the separate
    fit's.
(ii) One fit adjoint. `fit_rows_adjoints` runs several targets in one pass:
  * each target is bitwise its own `fit_rows_adjoint` (D_bar and X_bar
    whole and in tiles, MT_bar, the AO collocation's and the metric's
    adjoints) at 1, 2, 3, 5 and 8 ranks (eight: a rank owning no tile),
    the same bits at every rank count;
  * a pool's seeds are summed kind by kind in a fixed order (the first
    copied, the rest added onto it) and the pool is then one target,
    bitwise `fit_rows_adjoint` of seeds the test sums by hand in that order;
  * the state-pair force with the production flags (water/cc-pVDZ on the
    ISDF-K SCF, HF, PBE0 and LRC-wPBEh, serially and on the distributed SCF
    at 2 and 3 ranks) makes one fit-adjoint call of three targets where the
    separate path makes one per skeleton and one for the chain, and every
    target of that call is bitwise the one-target call on the seeds it must
    carry (the chain's own; the excitation skeletons' seeds summed by hand
    in their order; the mean field's), so each seed is counted once and
    joined to the right ones. The pooled sum re-rounds what the fit adjoint
    amplifies by the balanced Gram matrix's conditioning, so the pooled
    force sits about 1e-12 relative from the separate calls' and is only
    printed; the finite-difference gate below checks the pooled default.
    With every deposit its own target the force is within 1e-12 relative of
    the separate path's. Every rank holds rank 0's;
  * planted, a deposit dropped or counted twice fails the bitwise target
    check and moves the force, and seeds no assembly contracted raise;
  * the total force against a 4-point central difference of its own total
    energy on C1-distorted ethylene and formaldehyde, ISDF-K HF, PBE0 and
    LRC-wPBEh, within `ISDF_GRADIENT_FLOOR`; the ISDF-K SCF's points follow
    the atomic frames of each geometry, whose curvature on formaldehyde
    needs h = 2.5e-4;
  * memory: at 2 and 3 ranks no frame of the combined call binds an array
    whole along the grid by a factor's or the fit's width, nor one whole
    along it twice; its tracemalloc peak against the separate calls'.
"""
import os
import sys
import threading
import tracemalloc
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base import separable_ri
from src.Base.constants import (ISDF_GRADIENT_FLOOR,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.distributed_isdf_jk import DistributedISDFJK
from src.Base.isdf_jk import isdf_grid, isdf_jk
from src.Base.separable_ri import (AdjointSeeds, fit_M_streaming,
                                   fit_rows_adjoint, fit_rows_adjoints,
                                   resolve_isdf_grid)
from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.mpi_grid import (current_comm, distributed, partition,
                                     run_simulated)
from src.gradients import factor_chain, isdf_derivatives
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.isdf_derivatives import (PendingFitAdjoint,
                                            isdf_fock_partial_exchange,
                                            one_fit_adjoint)
from src.properties.excitations import SurfaceSpec, surface_of
from src.SingleReference.GW.space_time import separable_factors

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
#: The state-pair gates' water (tests/test_state_pair_force_distributed_ks.py).
WATER = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
#: C1-distorted molecules of the kernel and FD gates
#: (tests/test_skeleton_tiles.py's).
GEOMS = {
    'water': 'O 0 0 0.1173; H 0 0.7872 -0.4692; H 0 -0.7572 -0.4492',
    'ethylene': ('C 0.02 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9589 1.2321; H 0.03 0.9289 -1.2321; '
                 'H 0 -0.9289 -1.2021'),
    'formaldehyde': ('C 0 0 -0.5296; O 0 0.02 0.6763; H 0 0.9357 -1.1172; '
                     'H 0.02 -0.9557 -1.1172'),
}
#: Row tile edge of the kernel gates: water's 444 points in 7 tiles.
TILE = 64
#: The handed-in rows' tile edges: 7 tiles, and 2 (a third rank owns none).
HANDED_TILES = (64, 256)
XCS = ('hf', 'pbe0', 'lrc-wpbeh')
FORCE_SIZES = (1, 2, 3)
KERNEL_SIZES = (1, 2, 3, 5, 8)
#: (step, components) of the total force's 4-point difference.
FD = {'ethylene': (1e-3, ((2, 1), (0, 2))),
      'formaldehyde': (2.5e-4, ((0, 2), (2, 1)))}
#: The Davidson residual of the FD chain, below the fit's response.
BSE_CONV_TOL = 1e-9
MAX_MEMORY = 4000
_SEPARATE = threading.local()
_MADE = threading.local()


# ------------------------------------------------------------------ helpers
def molecule(name):
    return gto.M(atom=GEOMS[name], basis=BASIS, verbose=0,
                 max_memory=MAX_MEMORY)


def isdf_factory(xc, grid_accuracy=None):
    """The ISDF-K mean field, converged here serially and left unrun over
    ranks for the distributed SCF."""
    def build(mol):
        base = dft.RKS(mol, xc=xc)
        base.max_memory = MAX_MEMORY
        kw = {}
        if grid_accuracy is not None:
            elements = sorted({mol.atom_pure_symbol(i)
                               for i in range(mol.natm)})
            kw['counts'], kw['n_start'] = resolve_isdf_grid(
                grid_accuracy, BASIS, elements, auxbasis=AUX)
        mf = isdf_jk(base, auxbasis=AUX, **kw)
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        comm = current_comm()
        if comm is None or comm.Get_size() == 1:
            mf.kernel()
        return mf
    return build


def ladder_chain(xc):
    """The state-pair surface on water: space-time chi0, SOP residues on
    the frontier states, the row fit, the grid BSE adjoint."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)
    spec = SurfaceSpec(GroundState('dft', xc), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics={'grid_accuracy': 'G1', 'sliced': True,
                                 'fit': 'rows', 'bse_adjoint': 'grid'})
    return surface_of(spec, Excitation('singlet', root=1, kernel='bse'), mol,
                      isdf_factory(xc, 'G1'))


def tiles_of(a, comm, nk, block=TILE):
    """{tile: rows} of this rank's tiles of the whole array `a`."""
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    return {int(t): a[t * block:min((t + 1) * block, nk)].copy()
            for t in partition(-(-nk // block), rank, size)}


def kernel_case(name):
    """(mol, auxmol, points, layout, random seeds) of a kernel gate."""
    mol = molecule(name)
    aux = separable_ri.df.addons.make_auxmol(mol, auxbasis=AUX)
    crd = np.asarray(isdf_grid(mol, auxbasis=AUX))
    layout = separable_ri.test_set_layout(mol, crd)
    nk, nao, naux = len(crd), mol.nao_nr(), aux.nao_nr()
    rng = np.random.default_rng(11)
    vbar = rng.standard_normal((naux, naux))
    seeds = dict(d=rng.standard_normal((nk, naux)),
                 x=rng.standard_normal((nk, nao)),
                 C=rng.standard_normal((nao, nao)),
                 mt=rng.standard_normal((nk, naux)),
                 xao=rng.standard_normal((nk, nao)), vbar=vbar + vbar.T)
    return mol, aux, crd, layout, seeds


def kernel_targets(comm, crd, seeds):
    """The chain's two forms of seeds (whole, tiles) and a skeleton's."""
    rank = 0 if comm is None else comm.Get_rank()
    nk = len(crd)
    return [
        AdjointSeeds(d_bar=seeds['d'], x_bar=seeds['x'], mo_coeff=seeds['C']),
        AdjointSeeds(d_bar=GridTileRows.from_whole(seeds['d'], TILE, comm),
                     x_bar=GridTileRows.from_whole(seeds['x'], TILE, comm),
                     mo_coeff=seeds['C']),
        AdjointSeeds(mt_bar=tiles_of(seeds['mt'], comm, nk),
                     x_bar_ao=tiles_of(seeds['xao'], comm, nk),
                     metric_bar=seeds['vbar'] if rank == 0 else None)]


def flat(adjoint):
    """Every output of a `RowFitAdjoint` in one vector."""
    parts = [adjoint.fit_centre, adjoint.fit_points]
    if adjoint.coll_centre is not None:
        parts += [adjoint.coll_centre, adjoint.coll_points]
    return np.concatenate([np.ravel(p) for p in parts])


def added_by_hand(seeds):
    """The test's own sum of mt_bar, x_bar_ao and metric_bar seeds, in the
    order given, each kind copied from the first and the others added onto
    it: the order the pooled call documents (`summed_seeds`)."""
    out = {}
    for kind in ('mt_bar', 'x_bar_ao', 'metric_bar'):
        values = [getattr(s, kind) for s in seeds
                  if getattr(s, kind) is not None]
        if not values:
            out[kind] = None
        elif isinstance(values[0], dict):
            total = {t: a.copy() for t, a in values[0].items()}
            for v in values[1:]:
                for t in total:
                    total[t] += v[t]
            out[kind] = total
        else:
            total = values[0].copy()
            for v in values[1:]:
                total += v
            out[kind] = total
    return out


def one_target(key, seeds):
    """`fit_rows_adjoint` of one `AdjointSeeds` on the fit of `key`."""
    return fit_rows_adjoint(key.mol, key.auxmol, key.coords, seeds.d_bar,
                            key.layout, x_bar=seeds.x_bar,
                            mo_coeff=seeds.mo_coeff, mt_bar=seeds.mt_bar,
                            x_bar_ao=seeds.x_bar_ao,
                            metric_bar=seeds.metric_bar, block=key.block,
                            **key.settings)


@pytest.fixture
def switches(monkeypatch):
    """`_SEPARATE.on` (per thread) runs every skeleton's own fit adjoint,
    with no pooling window; returns a log per thread of each fit-adjoint
    call's target count, each tile hand-in, and, for a call of several
    targets, whether each target is bitwise `fit_rows_adjoint` of the seeds
    it should carry: the chain's own as handed in, the excitation skeletons'
    seeds summed by hand in their order, the mean field's as produced."""
    log = {}
    lock = threading.Lock()
    pending, seeds_of = (isdf_derivatives.pending_fit_adjoint,
                         isdf_derivatives.isdf_exchange_seeds)
    contract, row_fit = (isdf_derivatives.FitKey.contract,
                         separable_ri.FitTiles.row_fit)
    def note(kind, value):
        with lock:
            log.setdefault(threading.get_ident(), []).append((kind, value))

    def maybe_pending(mf):
        return None if getattr(_SEPARATE, 'on', False) else pending(mf)

    def recorded_seeds(*args, **kwargs):
        out = seeds_of(*args, **kwargs)
        role = 'mean field' if kwargs.get('dm_other') is None else 'excitation'
        _MADE.__dict__.setdefault('seeds', []).append((role, out[0]))
        return out

    def checked(self, targets):
        note('targets', len(targets))
        outs = contract(self, targets)
        if len(targets) > 1:
            produced = _MADE.__dict__.pop('seeds', [])
            ahead = [s for role, s in produced if role == 'mean field']
            expected = []
            for target in targets:
                members = (list(target) if isinstance(target, (list, tuple))
                           else [target])
                if any(s.d_bar is not None for s in members):
                    expected.append(members[0] if len(members) == 1
                                    else None)
                elif len(members) == 1 and ahead and members[0] is ahead[-1]:
                    expected.append(members[0])
                else:
                    expected.append(AdjointSeeds(**added_by_hand(
                        [s for role, s in produced if role == 'excitation'])))
            note('bitwise', [want is not None and np.array_equal(
                flat(out), flat(one_target(self, want)))
                for out, want in zip(outs, expected)])
        return outs

    def handed(self, *args, **kwargs):
        out = row_fit(self, *args, **kwargs)
        note('handed', out is not None)
        return out

    monkeypatch.setattr(isdf_derivatives, 'pending_fit_adjoint',
                        maybe_pending)
    monkeypatch.setattr(factor_chain, 'pending_fit_adjoint', maybe_pending)
    monkeypatch.setattr(isdf_derivatives, 'isdf_exchange_seeds',
                        recorded_seeds)
    monkeypatch.setattr(isdf_derivatives.FitKey, 'contract', checked)
    monkeypatch.setattr(separable_ri.FitTiles, 'row_fit', handed)
    return log


def force(chain, separate=False):
    """The chain's total force, the window's (default) or the separate
    path's."""
    _SEPARATE.on = separate
    _MADE.seeds = []
    try:
        return np.asarray(chain.total_gradient()[0])
    finally:
        _SEPARATE.on = False


def calls(log):
    """This thread's log since the last read."""
    return log.pop(threading.get_ident(), [])


# --------------------------------------------------------- (i) one fit
@pytest.mark.parametrize('tile', HANDED_TILES)
@pytest.mark.parametrize('size', (1, 2, 3))
def test_the_handed_tiles_are_the_fit(size, tile):
    """The handle's tiles read through `fit_rows(tiles=)` are bitwise the fit
    made again; another tile edge, regularization or point set refits."""
    def rank(comm):
        warnings.simplefilter('ignore')
        mol = molecule('water')
        mf = isdf_jk(dft.RKS(mol, xc='lrc-wpbeh'), auxbasis=AUX)
        handle = DistributedISDFJK(mf.with_df, comm=comm, tile=tile)
        handle.build()
        tiles = handle.fit_tiles()
        crd, aux = np.array(handle.coords), handle.auxmol
        C = np.random.default_rng(3).standard_normal((mol.nao_nr(), 10))
        t_handed, t_fit = {}, {}
        handed = fit_M_streaming(mol, aux, crd, fit='rows', block=tile,
                                 tiles=tiles, timings=t_handed)
        fitted = fit_M_streaming(mol, aux, crd, fit='rows', block=tile,
                                 timings=t_fit)
        same = (sorted(handed.mt) == sorted(fitted.mt)
                and all(np.array_equal(handed.mt[t], fitted.mt[t])
                        for t in fitted.mt)
                and np.array_equal(handed.ao_rows(), fitted.ao_rows())
                and np.array_equal(handed.mo_rows(C), fitted.mo_rows(C))
                and np.array_equal(handed.metric_root_rows(aux),
                                   fitted.metric_root_rows(aux)))
        others = []
        for kw in (dict(block=tile // 2),
                   dict(block=tile, regularization=1e-6),
                   dict(block=tile, coords=crd + 1e-9)):
            t = {}
            fit_M_streaming(mol, aux, kw.pop('coords', crd), fit='rows',
                            tiles=tiles, timings=t, **kw)
            others.append('fit_reused_tiles' in t)
        held = handed.held['reused_MT_tiles']
        readonly = all(not a.flags.writeable for a in tiles.mt.values())
        return (same, 'fit_reused_tiles' in t_handed,
                'fit_reused_tiles' in t_fit, others, len(fitted.mt), held,
                readonly)

    out = run_simulated(rank, size)
    for same, reused, refit, others, owned, held, readonly in out:
        assert same, 'the handed rows are not the refit bits'
        assert reused and not refit
        assert others == [False, False, False], others
        assert readonly and (held > 0) == (owned > 0)
    if size == 3 and tile == 256:
        assert min(o[4] for o in out) == 0, 'no rank without a tile'


@pytest.mark.parametrize('size', (2, 3))
def test_the_chain_reads_the_scf_fit(size, switches):
    """The state-pair chain on the row fit over a distributed ISDF-K SCF
    reads the SCF's tiles at the reference and fits nothing there; its
    factors are bitwise the rows of its own refit."""
    def rank(comm):
        warnings.simplefilter('ignore')
        calls(switches)
        chain = ladder_chain('lrc-wpbeh')
        mol, mf = chain.mol0, chain.mf0
        rows = chain.factors_at(mol, mf)[0]
        log = calls(switches)
        crd, aux = chain.coords(mol), chain.auxmol(mol)
        fit = fit_M_streaming(mol, aux, crd, fit='rows',
                              block=chain.fit_block)
        again = SlicedFactors.from_rows(
            fit.mo_rows(mf.mo_coeff),
            fit.metric_root_rows(aux, chain.environment_at(mol)),
            fit.ao_rows(), crd, current_comm())
        return (log, all(np.array_equal(getattr(rows, k), getattr(again, k))
                         for k in ('X_mo', 'D', 'X_ao')),
                rows.fit_held.get('reused_MT_tiles', 0))

    out = run_simulated(rank, size)
    for log, same, held in out:
        assert ('handed', True) in log, log
        assert same, "the chain's rows are not its refit's bits"
    assert sum(o[2] for o in out) > 0


@pytest.mark.parametrize('size', (2, 3))
def test_the_energy_route_reads_the_scf_fit(size):
    """`separable_factors(fit='rows', fit_tiles=)` reads the distributed SCF's
    tiles where its grid is the SCF's, bitwise the separate fit's rows."""
    def rank(comm):
        warnings.simplefilter('ignore')
        mol = molecule('water')
        mf = isdf_factory('pbe0', 'G1')(mol)
        handle = DistributedISDFJK(mf.with_df, comm=comm)
        handle.build()
        with distributed(None):
            serial = isdf_factory('pbe0', 'G1')(mol)
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = (serial.mo_coeff,
                                                serial.mo_occ,
                                                serial.mo_energy)
        t_handed, t_fit = {}, {}
        handed = separable_factors(mf, mol, auxbasis=AUX, fit='rows',
                                   grid_accuracy='G1',
                                   fit_tiles=handle.fit_tiles(),
                                   timings=t_handed)
        fitted = separable_factors(mf, mol, auxbasis=AUX, fit='rows',
                                   grid_accuracy='G1', timings=t_fit)
        same = all(np.array_equal(getattr(handed, k), getattr(fitted, k))
                   for k in ('X_mo', 'D', 'X_ao', 'coords'))
        return same, 'fit_reused_tiles' in t_handed, \
            'fit_reused_tiles' in t_fit

    out = run_simulated(rank, size)
    assert all(o == (True, True, False) for o in out), out


# ------------------------------------------------- (ii) one fit adjoint
@pytest.mark.parametrize('name', ('water', 'ethylene'))
def test_each_target_is_its_own_call(name):
    """`fit_rows_adjoints` of three targets: each bitwise its own
    `fit_rows_adjoint`, and the same bits at every rank count."""
    mol, aux, crd, layout, seeds = kernel_case(name)

    def rank(comm, combined):
        targets = kernel_targets(comm, crd, seeds)
        if combined:
            outs = fit_rows_adjoints(mol, aux, crd, layout, targets,
                                     block=TILE)
        else:
            outs = [fit_rows_adjoint(mol, aux, crd, s.d_bar, layout,
                                     x_bar=s.x_bar, mo_coeff=s.mo_coeff,
                                     mt_bar=s.mt_bar, x_bar_ao=s.x_bar_ao,
                                     metric_bar=s.metric_bar, block=TILE)
                    for s in targets]
        return [flat(o) for o in outs]

    one = run_simulated(rank, 1, False)[0]
    for size in KERNEL_SIZES:
        for combined in (False, True):
            res = run_simulated(rank, size, combined)
            for r in res:
                assert all(np.array_equal(a, b) for a, b in zip(r, one)), \
                    (name, size, combined)


def test_a_pool_is_its_summed_seeds():
    """A pool of the chain's and a skeleton's seeds is bitwise the one-target
    call on their seeds summed by hand in the same order, at 1 and 3
    ranks."""
    mol, aux, crd, layout, seeds = kernel_case('ethylene')

    def rank(comm):
        chain, _, skel = kernel_targets(comm, crd, seeds)
        pool = fit_rows_adjoints(mol, aux, crd, layout, [[chain, skel]],
                                 block=TILE)[0]
        summed = added_by_hand([chain, skel])
        alone = fit_rows_adjoint(mol, aux, crd, chain.d_bar, layout,
                                 x_bar=chain.x_bar, mo_coeff=chain.mo_coeff,
                                 block=TILE, **summed)
        return np.array_equal(flat(pool), flat(alone))

    for size in (1, 3):
        assert all(run_simulated(rank, size)), size


def test_seeds_left_uncontracted_raise():
    """A window left with a skeleton's seeds no assembly read raises."""
    mol = molecule('water')
    mf = isdf_factory('pbe0')(mol)
    gamma = np.diag(np.linspace(0.1, 1.0, mf.mo_coeff.shape[1]))
    with pytest.raises(RuntimeError, match='no nuclear assembly contracted'):
        with one_fit_adjoint(mf):
            isdf_fock_partial_exchange(mf, gamma)
    # outside a window the skeleton contracts its own seeds
    g = isdf_fock_partial_exchange(mf, gamma)
    assert np.abs(g).max() > 0.0


@pytest.mark.parametrize('xc', XCS)
def test_the_force_makes_one_fit_adjoint_call(xc, switches, monkeypatch):
    """Serially and at 2 and 3 ranks: one call of three targets where the
    separate path makes one per skeleton and one for the chain, each target
    of it bitwise the one-target call on the seeds it should carry; with
    every deposit a target of its own the force is within 1e-12 relative of
    the separate path's; every rank holds rank 0's."""
    # each arm on a fresh chain: a second force on one chain starts its
    # Davidson from the first one's roots and lands 1e-12 away
    def rank(comm):
        warnings.simplefilter('ignore')
        calls(switches)
        pooled = force(ladder_chain(xc))
        log_pooled = calls(switches)
        separate = force(ladder_chain(xc), separate=True)
        return pooled, separate, log_pooled, calls(switches)

    def unpooled(comm):
        warnings.simplefilter('ignore')
        return (force(ladder_chain(xc)),
                force(ladder_chain(xc), separate=True))

    lines = []
    for size in FORCE_SIZES:
        out = run_simulated(rank, size)
        pooled, separate, log_p, log_s = out[0]
        for o in out[1:]:
            assert np.array_equal(o[0], pooled) and np.array_equal(o[1],
                                                                   separate)
        for o in out:
            assert [v for kind, v in o[2] if kind == 'bitwise'] == [
                [True] * 3], o[2]
        targets = [n for kind, n in log_p if kind == 'targets']
        separate_calls = [n for kind, n in log_s if kind == 'targets']
        assert targets == [3], log_p
        assert separate_calls == [1] * len(separate_calls)
        assert len(separate_calls) == (3 if xc == 'hf' else 4), log_s
        if size > 1:
            assert ('handed', True) in log_p, log_p
        d = np.abs(pooled - separate).max() / np.abs(separate).max()
        lines.append(f'{size} ranks: pooled - separate {d:.1e} rel')
    monkeypatch.setattr(isdf_derivatives.PointChain, 'same',
                        lambda self, other: self is other)
    for size in (1, 2):
        out = run_simulated(unpooled, size)
        whole, separate = out[0]
        d = np.abs(whole - separate).max() / np.abs(separate).max()
        lines.append(f'{size} ranks unpooled: {d:.1e} rel')
        assert d <= 1e-12, lines
        assert all(np.array_equal(o[0], whole) for o in out)
    print(f'\n{xc}: ' + '; '.join(lines))


@pytest.mark.parametrize('plant', ('dropped', 'twice'))
def test_a_planted_seed_breaks_the_call(plant, switches, monkeypatch):
    """A skeleton's deposit dropped from the call, or deposited twice: the
    pooled target then differs from the one-target call on the skeletons'
    own seeds summed, and the force moves."""
    deposit = PendingFitAdjoint.deposit
    seen = []

    def planted(self, key, chain, seeds):
        seen.append(1)
        if len(seen) == 1:
            if plant == 'twice':
                deposit(self, key, chain, seeds)
            else:
                return
        deposit(self, key, chain, seeds)

    with distributed(None):
        calls(switches)
        right = force(ladder_chain('lrc-wpbeh'))
        assert [v for k, v in calls(switches) if k == 'bitwise'] == [
            [True] * 3]
        monkeypatch.setattr(PendingFitAdjoint, 'deposit', planted)
        wrong = force(ladder_chain('lrc-wpbeh'))
        flags = [v for k, v in calls(switches) if k == 'bitwise']
    d = np.abs(wrong - right).max()
    print(f'\n{plant}: target checks {flags}, |d| {d:.2e}')
    assert flags and False in flags[0]
    assert d > 1e-6


@pytest.mark.parametrize('xc', XCS)
@pytest.mark.parametrize('name', ('ethylene', 'formaldehyde'))
def test_the_total_force_follows_its_energy(name, xc):
    """The excited state's total force (the mean field's and the
    excitation's skeletons on the chain's one fit-adjoint call) against a
    4-point central difference of its own total energy."""
    h, components = FD[name]
    with distributed(None):
        mol = molecule(name)
        chain = ExcitedStateChain(mol, isdf_factory(xc), spin='singlet',
                                  solver='davidson',
                                  bse_conv_tol=BSE_CONV_TOL, sliced=True,
                                  fit='rows', fit_block=TILE)
        grad = np.asarray(chain.total_gradient()[0])
        misses = []
        for atom, axis in components:
            e = {}
            for k in (2, 1, -1, -2):
                crd = mol.atom_coords().copy()
                crd[atom, axis] += k * h
                e[k] = chain.total_energy(mol.set_geom_(crd, unit='Bohr',
                                                        inplace=False))
            fd = (8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)
            misses.append(float(grad[atom, axis] - fd))
    print(f'\n{name}/{xc} (h {h:g}): analytic - FD '
          f'{np.array2string(np.array(misses), precision=2)}')
    assert np.abs(misses).max() < ISDF_GRADIENT_FLOOR
    assert np.abs(grad.sum(axis=0)).max() < 1e-9


def _whole_arrays(obj, found, npts, widths, depth=0):
    """Into `found`: arrays reachable from `obj` whole along the grid by one
    of `widths`, or whole along it twice."""
    if isinstance(obj, np.ndarray):
        shape = obj.shape
        if (obj.ndim == 2 and obj.dtype.kind == 'f'
                and ((shape[0] == npts and shape[1] in widths)
                     or (shape[1] == npts and shape[0] in widths)
                     or shape == (npts, npts))):
            found.add(shape)
    elif depth < 3 and isinstance(obj, dict):
        for v in obj.values():
            _whole_arrays(v, found, npts, widths, depth + 1)
    elif depth < 3 and isinstance(obj, (list, tuple)):
        for v in obj:
            _whole_arrays(v, found, npts, widths, depth + 1)
    elif depth < 3 and isinstance(obj, (AdjointSeeds, GridTileRows)):
        _whole_arrays(vars(obj), found, npts, widths, depth + 1)


@pytest.mark.parametrize('size', (2, 3))
def test_the_combined_call_holds_no_whole_array(size):
    """No frame of the combined call binds an array whole along the grid by
    a factor's or the fit's width, nor one whole along it twice."""
    mol, aux, crd, layout, seeds = kernel_case('water')
    npts = len(crd)
    widths = {mol.nao_nr(), aux.nao_nr(), npts}
    src = separable_ri.__file__.rsplit('/', 1)[0]

    def rank(comm):
        _, tiled, skel = kernel_targets(comm, crd, seeds)
        found = set()

        def local(frame, event, arg):
            if event in ('line', 'return'):
                for v in list(frame.f_locals.values()):
                    _whole_arrays(v, found, npts, widths)
            return local

        def tracer(frame, event, arg):
            return local if frame.f_code.co_filename.startswith(src) else None

        sys.settrace(tracer)
        try:
            fit_rows_adjoints(mol, aux, crd, layout, [tiled, skel],
                              block=TILE)
        finally:
            sys.settrace(None)
        return found

    out = run_simulated(rank, size)
    assert all(not f for f in out), out


@pytest.mark.parametrize('name', ('water', 'ethylene'))
def test_the_combined_call_memory(name):
    """tracemalloc peak of the three targets in one call against the three
    calls one after the other: one pass holds every target's row arrays
    beside the seed-free ones, and no more."""
    mol, aux, crd, layout, seeds = kernel_case(name)
    with distributed(None):
        targets = kernel_targets(None, crd, seeds)
        peaks = {}
        for label in ('separate', 'combined'):
            tracemalloc.start()
            if label == 'separate':
                held = [fit_rows_adjoints(mol, aux, crd, layout, [t],
                                          block=TILE)[0].held
                        for t in targets]
            else:
                both = fit_rows_adjoints(mol, aux, crd, layout, targets,
                                         block=TILE)[0].held
            peaks[label] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
    rows = ('MT_bar_rows', 'Q_bar_rows', 'U_rows', 'X_bar_rows', 'B_bar_rows',
            'P_bar_rows')
    per_target = max(sum(h[k] for k in rows) for h in held)
    extra = peaks['combined'] - peaks['separate']
    print(f'\n{name}: tracemalloc peak separate {peaks["separate"] / 1e6:.1f} '
          f'MB, combined {peaks["combined"] / 1e6:.1f} MB; per-target rows '
          f'{per_target / 1e6:.2f} MB, combined ledger rows '
          f'{sum(both[k] for k in rows) / 1e6:.2f} MB')
    assert extra <= 2 * len(targets) * per_target
    assert sum(both[k] for k in rows) == sum(sum(h[k] for k in rows)
                                             for h in held)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
