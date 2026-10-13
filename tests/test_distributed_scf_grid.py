"""After a distributed Kohn-Sham SCF every rank holds rank 0's quadrature grid.

Only rank 0 builds the grid inside the SCF (pyscf's own `Grids.build`, pruned
against rank 0's first density) and the other ranks receive its points inside
each quadrature. Without the grid lockstep (`_lockstep_grids`), a rank other
than 0 reaches the first Fock build after the SCF with no grid, builds its own
and prunes it against the converged density, keeping another point count
(water/PBE0/cc-pVDZ: 30128 points against rank 0's 30624); the xc
grid-response skeleton, whose tiles are gathered over the ranks, then refuses
the mismatch, and every other grid consumer integrates a different quadrature
on each rank.

Gated, water/cc-pVDZ:
  * the ISDF-K SCF (tiles of `TILE` points, 7 of them) at 1, 2, 3 and 8
    simulated ranks, where rank 7 owns no tile, PBE0 and LRC-wPBEh; the
    density-fitted SCF at 2 and 3 ranks, PBE0 and the VV10 functional
    B97M-V, whose NLC grid rank 0 builds as well;
  * every rank's grid arrays (points, weights, atoms, unpartitioned
    weights, screening mask, the same object as `screen_index` as pyscf
    leaves it) and cutoff bitwise rank 0's, for `grids` and `nlcgrids`;
  * rank 0 built and pruned each grid once, no other rank built or pruned
    one, and the next `get_fock` on every rank builds and prunes nothing and
    leaves the grid's bits where they were (on the ISDF route its
    quadrature alone, since the serial ISDFJK refuses to build under ranks);
  * one rank is the serial SCF, untouched: its grid, energy and orbitals
    bitwise a plain `mf.kernel()`'s;
  * with `_lockstep_grids` a no-op, rank 1 holds no grid after the SCF and
    builds and prunes its own on the next `get_fock`, to another point count.
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base import distributed_df
from src.Base.distributed_df import distributed_mean_field, release_distributed
from src.Base.distributed_isdf_jk import distributed_isdf_jk
from src.Base.isdf_jk import isdf_jk
from src.Base.utils.mpi_grid import distributed, run_simulated

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
#: Grid points per ISDF tile: water's 444 points in 7 tiles.
TILE = 64
CONV_TOL, CONV_TOL_GRAD = 1e-10, 1e-6
GRID_ARRAYS = ('coords', 'weights', 'atm_idx', 'quadrature_weights',
               'non0tab', 'screen_index')
ISDF_CASES = [(xc, size) for xc in ('pbe0', 'lrc-wpbeh')
              for size in (1, 2, 3, 8)]
DF_CASES = [(xc, size) for xc in ('pbe0', 'b97m_v') for size in (2, 3)]
_LOCK = threading.Lock()


def fresh(route, xc):
    """An unrun Kohn-Sham mean field on water: ISDF-K or density-fitted."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf = (isdf_jk(mf, auxbasis=AUX) if route == 'isdf'
          else mf.density_fit(auxbasis=AUX))
    mf.conv_tol, mf.conv_tol_grad = CONV_TOL, CONV_TOL_GRAD
    return mf


def watch(mf, events):
    """Count `build` and `prune_by_density_` on this mean field's grids."""
    for name in ('grids', 'nlcgrids'):
        grids = getattr(mf, name)
        for method in ('build', 'prune_by_density_'):
            original = getattr(grids, method)

            def counted(*args, _key=(name, method), _f=original, **kwargs):
                with _LOCK:
                    events[_key] = events.get(_key, 0) + 1
                return _f(*args, **kwargs)

            setattr(grids, method, counted)


def grid_state(mf):
    """Copies of every grid array, the mask aliasing and the cutoff."""
    out = {}
    for name in ('grids', 'nlcgrids'):
        grids = getattr(mf, name)
        for field in GRID_ARRAYS:
            a = getattr(grids, field)
            out[name, field] = None if a is None else np.array(a, copy=True)
        out[name, 'aliased'] = grids.screen_index is grids.non0tab
        out[name, 'cutoff'] = grids.cutoff
    return out


def bitwise(a, b):
    """Bit for bit, None only against None."""
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, np.ndarray):
        return a == b
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def run(route, xc, size):
    """Per rank: grid builds in the SCF and in the next get_fock, the grid
    after each, energy and orbitals."""
    mfs = [fresh(route, xc) for _ in range(size)]

    def one(comm):
        mf = mfs[comm.Get_rank()]
        events = {}
        watch(mf, events)
        if route == 'isdf':
            distributed_isdf_jk(mf, comm, tile=TILE)
        distributed_mean_field(mf)
        scf_events, held = dict(events), grid_state(mf)
        coords = mf.grids.coords
        if route == 'isdf':
            # the serial ISDFJK refuses to build whole under ranks, so the
            # serial grid consumer here is get_fock's quadrature alone
            dm = mf.make_rdm1()
            mf.initialize_grids(mf.mol, dm)
            mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)
        else:
            mf.get_fock()
        fock_events = {k: v - scf_events.get(k, 0) for k, v in events.items()
                       if v != scf_events.get(k, 0)}
        out = dict(scf=scf_events, fock=fock_events, held=held,
                   after=grid_state(mf), same_coords=mf.grids.coords is coords,
                   e=mf.e_tot, mo=np.asarray(mf.mo_coeff).tobytes())
        release_distributed(mf)
        return out
    return run_simulated(one, size)


def assert_rank_zeros_grid(out):
    """Every rank rank 0's grid; rank 0 alone built it; get_fock builds none."""
    size = len(out)
    built = {name for name in ('grids', 'nlcgrids')
             if out[0]['held'][name, 'coords'] is not None}
    assert 'grids' in built
    want = {(name, m): 1 for name in built
            for m in ('build', 'prune_by_density_')}
    assert out[0]['scf'] == want, out[0]['scf']
    for r in range(1, size):
        assert out[r]['scf'] == {}, (r, out[r]['scf'])
        for key, value in out[0]['held'].items():
            assert bitwise(out[r]['held'][key], value), (r, key)
    for r in range(size):
        assert out[r]['fock'] == {}, (r, out[r]['fock'])
        assert out[r]['same_coords'], r
        for key, value in out[r]['held'].items():
            assert bitwise(out[r]['after'][key], value), (r, key)
    assert all(o['e'] == out[0]['e'] and o['mo'] == out[0]['mo']
               for o in out)


@pytest.fixture(scope='module')
def serial():
    """The serial ISDF-K SCF per functional: grid, energy, orbitals."""
    out = {}
    for xc in ('pbe0', 'lrc-wpbeh'):
        with distributed(None):
            mf = fresh('isdf', xc)
            mf.kernel()
        out[xc] = (grid_state(mf), mf.e_tot,
                   np.asarray(mf.mo_coeff).tobytes())
    return out


@pytest.mark.parametrize('xc,size', ISDF_CASES)
def test_isdf_scf_leaves_rank_zeros_grid(serial, xc, size):
    """The ISDF-K SCF: rank 0's grid on every rank, rank 0 its one builder,
    and one rank the serial SCF bit for bit."""
    out = run('isdf', xc, size)
    assert_rank_zeros_grid(out)
    if size == 1:
        grid, e, mo = serial[xc]
        assert out[0]['e'] == e and out[0]['mo'] == mo
        for key, value in grid.items():
            assert bitwise(out[0]['held'][key], value), key
    else:
        n = out[0]['held']['grids', 'weights'].size
        print(f'\n{xc} at {size} ranks: {n} points on every rank')


@pytest.mark.parametrize('xc,size', DF_CASES)
def test_df_scf_leaves_rank_zeros_grid(xc, size):
    """The density-fitted SCF: the same, the VV10 grid included."""
    out = run('df', xc, size)
    assert_rank_zeros_grid(out)
    if xc == 'b97m_v':
        assert out[0]['held']['nlcgrids', 'coords'] is not None


def test_without_the_lockstep_a_rank_prunes_its_own(monkeypatch):
    """The defect this gates: with `_lockstep_grids` a no-op, rank 1 builds
    and prunes its own grid on the next get_fock, to another point count."""
    monkeypatch.setattr(distributed_df, '_lockstep_grids',
                        lambda mf, comm: None)
    out = run('isdf', 'pbe0', 2)
    assert out[1]['scf'] == {}
    assert out[1]['held']['grids', 'coords'] is None
    assert out[1]['fock'] == {('grids', 'build'): 1,
                              ('grids', 'prune_by_density_'): 1}
    n0 = out[0]['after']['grids', 'weights'].size
    n1 = out[1]['after']['grids', 'weights'].size
    assert n0 != n1, (n0, n1)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
