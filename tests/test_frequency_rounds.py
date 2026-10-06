"""The contour-deformation frequency pass over ranks goes by rounds.

`ProjRows.blocks` hands each rank its round-robin frequencies' chi0 whole:
one exchange per round of P frequencies hands every rank its frequency of the
round, and all P owners factorize at once; the reverse hands the round's
adjoints back in one exchange and folds them by serial block. Every row is
transformed and folded with the serial block's call, so no bit moves.

Water/cc-pVDZ Hartree-Fock, 8 tau points, 24 frequencies, naux 84, at 1, 2,
3, 5 and 8 simulated ranks and frequency blocks of one, five and all 24:

  * the frequency pass makes ceil(24 / P) exchanges forward and as many in
    the fold, not one per frequency;
  * `qp_set_gradient` on the pole-model and Laplace routes, adjoints in grid
    tiles: wc, the roots, Z, eps_bar and the X_bar / D_bar tiles are the
    serial bits on every rank;
  * planted, a frequency skipped, a frequency factorized twice, or a round's
    rows sent to the wrong owner each fails the gate.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import ISDF_TILE_GB
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import partition, run_simulated
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    polarizability_projected_rows)
from src.gradients.qp_space_time import qp_set_gradient

NTAU, NFREQ = 8, 24
SIZES = [1, 2, 3, 5, 8]
#: frequency blocks of one (the cc-pVTZ case), five and all, at naux = 84
TILES = {'one': 1 * 3 * 84 ** 2 * 8 / 1e9, 'five': 5 * 3 * 84 ** 2 * 8 / 1e9,
         'all': ISDF_TILE_GB}
ROWS_BLOCK = 16
LAPLACE_TOL_OF_THE_GRID = 1e-2


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')[:2]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    assert D.shape[1] == 84, 'TILES are sized for naux = 84'
    return dict(X=X, D=D, eps=eps, nocc=nocc, grid=grid, nu=nu, wt=wt, mu=mu,
                states=[nocc - 3, nocc - 2, nocc - 1, nocc, nocc + 1],
                weights=np.array([0.2, 0.3, -1.1, 0.8, 0.45]))


@pytest.fixture
def exchanges(monkeypatch):
    """{rank: exchange_rows calls ProjRows made}, counted from here on."""
    calls, real = {}, ls_space_time.exchange_rows

    def counted(send, send_ranges, recv, recv_ranges, comm):
        rank = 0 if comm is None else comm.Get_rank()
        calls[rank] = calls.get(rank, 0) + 1
        return real(send, send_ranges, recv, recv_ranges, comm)

    monkeypatch.setattr(ls_space_time, 'exchange_rows', counted)
    return calls


def frequency_pass_exchanges(w, tile, calls, comm):
    """(exchanges of the forward pass, of the forward and fold pass)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    proj = polarizability_projected_rows(w['X'], w['D'], w['eps'], w['nocc'],
                                         w['grid'].tau_points, mu=w['mu'],
                                         comm=comm)
    cosft, mine = w['grid'].cosft_wt, partition(NFREQ, rank, size)
    at = calls.get(rank, 0)
    for _ in proj.blocks(cosft, tile, mine):
        pass
    forward, at = calls.get(rank, 0) - at, calls.get(rank, 0)
    bar = proj.zeros_like()
    for fb in proj.blocks(cosft, tile, mine):
        fb.chi0[...] = 1.0
        bar.fold(cosft, fb)
    return forward, calls.get(rank, 0) - at


@pytest.mark.parametrize('size', SIZES[1:])
@pytest.mark.parametrize('tile', TILES)
def test_one_exchange_per_round(water, exchanges, size, tile):
    rounds = -(-NFREQ // size)
    got = run_simulated(lambda comm: frequency_pass_exchanges(
        water, TILES[tile], exchanges, comm), size)
    assert got == [(rounds, 2 * rounds)] * size, (size, tile, got)


def qp_outputs(w, tile, comm=None, forward=False):
    """{name: array} of the set solve on two routes, X_bar and D_bar by the
    tiles this rank holds; the forward alone (every weight zero) has none."""
    out = {}
    weights = np.zeros_like(w['weights']) if forward else w['weights']
    for route in ('sop', 'laplace'):
        ro = {}
        got = qp_set_gradient(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                              w['nu'], w['wt'], w['states'], weights,
                              mu=w['mu'], residue_route=route,
                              laplace_tol=LAPLACE_TOL_OF_THE_GRID,
                              tile_gb=tile, route_out=ro,
                              rows_block=ROWS_BLOCK, comm=comm)
        out.update({f'{route}.w': got[0], f'{route}.z': ro['z'],
                    f'{route}.eps_bar': got[1],
                    f'{route}.wc': np.stack(ro['tape'].wcs)})
        if forward:
            continue
        for name, tiles in (('X_bar', got[2]), ('D_bar', got[3])):
            out.update({f'{route}.{name}[{t}]': tiles.tile(t).copy()
                        for t in tiles.mine})
    return out


def moved(w, tile, size, forward=False):
    """(name, rank) of every output a rank holds that is not serial's bits,
    and whether the ranks hold every output serial does between them."""
    serial = qp_outputs(w, tile, forward=forward)
    ranks = run_simulated(lambda comm: qp_outputs(w, tile, comm,
                                                  forward=forward), size)
    out = [(name, rank) for rank, got in enumerate(ranks)
           for name, a in got.items() if a.tobytes() != serial[name].tobytes()]
    held = sorted(set(name for got in ranks for name in got))
    return out, held == sorted(serial)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('tile', TILES)
def test_rounds_give_the_serial_bits(water, size, tile):
    out, every_tile = moved(water, TILES[tile], size)
    assert out == [] and every_tile, (size, tile, out[:8])


def skip_one(rounds_of):
    """Rounds with the last frequency of the second round left out."""
    def planted(nfreq, size):
        rounds = rounds_of(nfreq, size)
        rounds[1] = rounds[1][:-1]
        return rounds
    return planted


def twice(rounds_of):
    """Rounds whose second round factorizes its first frequency twice."""
    def planted(nfreq, size):
        rounds = rounds_of(nfreq, size)
        rounds[1][-1] = rounds[1][0]
        return rounds
    return planted


def wrong_owner(ranges_of):
    """A round's rows sent one rank along: rank j gets frequency j + 1's."""
    def planted(n, nrows, size):
        ranges = ranges_of(n, nrows, size)
        return ranges[1:n] + ranges[:1] + ranges[n:]
    return planted


@pytest.mark.parametrize('name, plant', [
    ('frequency_rounds', skip_one), ('frequency_rounds', twice),
    ('round_ranges', wrong_owner)])
def test_a_planted_round_defect_fails(water, monkeypatch, name, plant):
    """The forward's roots and wc move; the force either moves or refuses
    (a frequency's adjoint handed back twice, or never)."""
    monkeypatch.setattr(ls_space_time, name,
                        plant(getattr(ls_space_time, name)))
    out, _ = moved(water, TILES['five'], 3, forward=True)
    assert {n.split('.')[1] for n, _ in out} >= {'w', 'wc'}, plant.__name__
    try:
        out, _ = moved(water, TILES['five'], 3)
    except (KeyError, ValueError):
        return
    assert out, plant.__name__


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
