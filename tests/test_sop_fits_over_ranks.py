"""The pole-model fits of a quasiparticle set are striped over the ranks.

The fit is a dense least squares of about norb^3 per state, so on the
admitted window of a large molecule (hundreds of states) it must be divided
over the ranks: `fit_poles_over_ranks` hands each state's fit to one rank and
gathers the poles onto all of them. Each fit reads only that state's wc,
which is reduced and identical on every rank, so the poles, the roots and Z
are gated bitwise against serial.

Water/cc-pVDZ (HF, DF), the admitted window on the ISDF factors, a 24-point
contour grid. Over n ranks every state is fitted once in all.
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.declaration import QPStates
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import run_simulated
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.SingleReference.GW.space_time import separable_factors
import src.gradients.qp_space_time as qp_space_time
from src.gradients.qp_space_time import qp_set_gradient

SIZES = [2, 3]


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X_mo, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')[:2]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(24, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(8, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    states = np.array(resolve_qp_states(QPStates(), eps, nocc,
                                        degeneracy_tol=1e-6).explicit)
    return dict(X=X_mo, D=D, eps=eps, nocc=nocc, grid=grid, nu=nu, wt=wt,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]), states=states)


def solve(w, weights, comm=None, sop_poles=None):
    route_out = {}
    out = qp_set_gradient(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                          w['nu'], w['wt'], w['states'], weights, mu=w['mu'],
                          residue_route='sop', sop_poles=sop_poles,
                          route_out=route_out, comm=comm)
    return out, route_out


def counted_fits(monkeypatch):
    """A list that grows by one per pole fit, on any thread."""
    fits, lock = [], threading.Lock()
    real = qp_space_time.sop_from_wc

    def fit(*a, **k):
        with lock:
            fits.append(1)
        return real(*a, **k)
    monkeypatch.setattr(qp_space_time, 'sop_from_wc', fit)
    return fits


def test_the_window_is_on_the_pole_model(water):
    """More than one state, every one of them fitted: the gates below have
    something to split."""
    _, route_out = solve(water, np.zeros(len(water['states'])))
    assert len(water['states']) > 3
    assert set(route_out['routes'].values()) == {'sop'}
    assert sorted(route_out['sop_poles']) == sorted(int(p) for p in
                                                    water['states'])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('weighted', [False, True], ids=['energy', 'force'])
def test_poles_roots_and_z_bitwise_over_ranks(water, size, weighted):
    """Every rank returns the serial poles, roots and Z, bit for bit; with
    weights, the adjoints agree to the reverse pass's re-association."""
    n = len(water['states'])
    weights = (np.linspace(-1.0, 1.0, n) if weighted else np.zeros(n))
    ref, ref_out = solve(water, weights)
    got = run_simulated(lambda comm: solve(water, weights, comm=comm), size)
    for out, route_out in got:
        assert np.array_equal(out[0], ref[0])
        assert np.array_equal(route_out['z'], ref_out['z'])
        assert sorted(route_out['sop_poles']) == sorted(ref_out['sop_poles'])
        for p, poles in ref_out['sop_poles'].items():
            assert np.array_equal(route_out['sop_poles'][p], poles), p
        if weighted:
            scale = np.abs(ref[1]).max()
            assert np.abs(out[1] - ref[1]).max() <= 1e-11 * scale


@pytest.mark.parametrize('size', SIZES)
def test_each_state_is_fitted_once_over_the_ranks(water, size, monkeypatch):
    """Over `size` ranks the set costs n fits in all, not n per rank."""
    n = len(water['states'])
    fits = counted_fits(monkeypatch)
    run_simulated(lambda comm: solve(water, np.zeros(n), comm=comm), size)
    assert len(fits) == n, f'{len(fits)} fits for {n} states over {size} ranks'


def test_frozen_poles_are_not_fitted_again(water, monkeypatch):
    """A state whose poles the caller froze is read, not fitted, and its
    poles come back as they went in."""
    n = len(water['states'])
    _, ref_out = solve(water, np.zeros(n))
    frozen = {int(p): ref_out['sop_poles'][int(p)]
              for p in water['states'][::2]}
    fits = counted_fits(monkeypatch)
    out, route_out = solve(water, np.zeros(n), sop_poles=frozen)
    assert len(fits) == n - len(frozen)
    for p, poles in frozen.items():
        assert route_out['sop_poles'][p] is poles


@pytest.mark.parametrize('size', SIZES)
def test_each_slice_is_built_once_over_the_ranks(water, size, monkeypatch):
    """The three-index slices B_p, the O(M naux norb) per-state cost of the
    sweep, are striped too: n slices in all over `size` ranks, and every
    rank's are the serial ones bitwise."""
    n = len(water['states'])
    built, lock = [], threading.Lock()
    real = qp_space_time.three_index_slice

    def slice_(*a, **k):
        with lock:
            built.append(1)
        return real(*a, **k)
    ref = [real(water['X'], water['D'], int(p)) for p in water['states']]
    monkeypatch.setattr(qp_space_time, 'three_index_slice', slice_)
    got = run_simulated(lambda comm: solve(water, np.zeros(n), comm=comm),
                        size)
    assert len(built) == n, f'{len(built)} slices for {n} states'
    for _, route_out in got:
        for b, r in zip(route_out['tape'].Bps, ref):
            assert np.array_equal(b, r)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
