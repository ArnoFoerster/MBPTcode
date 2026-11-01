"""No rank holds the contour-deformation slices B_p, or their screened
partners [W - I] B_p, of a whole quasiparticle set.

A slice B_p is (naux, norb), 35 MB a state at 60 atoms on an augmented
double-zeta basis, and a quasiparticle set has hundreds of states, so a list
of them on every rank is a cost no rank count divides.

  * `cd_screening_contraction_multi` holds one [W - I] B_j at a time: on a
    synthetic screening of 24 slices its tracemalloc peak stays within
    WTB_HELD_MAX slices beside its outputs and the chi0 block (holding every
    [W - I] B_j adds 24), the wc to the bit of the contraction written out;
  * `qp_set_gradient` on water/cc-pVDZ Hartree-Fock over 1, 2 and 3 simulated
    ranks: each rank holds at most its `contiguous_block` of the slices
    (the tape's and pass 1's) and of the slice adjoints B_p_bar (pass 3),
    and between them the ranks hold every one once; the roots, Z, wc,
    eps_bar and the adjoints' grid tiles are the serial bits on every rank.
"""
import os
import sys
import tracemalloc
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.utils.grids.grids import gauss_legendre_grid
from src.Base.utils.grids.time_frequency import TimeFrequencyGrid
from src.Base.utils.parallel.mpi_grid import contiguous_block, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.contour_deformation import (
    cd_screening_contraction_multi, screening_solver)
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.gwbse import qp_space_time
from src.gradients.gwbse.qp_space_time import qp_set_gradient
from tests.isdf_test_grid import g1

#: the synthetic screening: auxiliary size, orbitals, slices, frequencies
NAUX, NORB, NSLICE, NFREQ_SYNTH = 40, 300, 24, 3
#: slices the contraction may hold beside its outputs and chi0 block: one
#: [W - I] B_j, its solve's temporary and the factor
WTB_HELD_MAX = 4
NTAU, NFREQ = 8, 16
ROWS_BLOCK = 16
SIZES = [1, 2, 3]


def synthetic_screening():
    """(proj_tau, cosft_wt, slices): proj(tau) negative semidefinite, so
    1 - chi0(i.nu) is positive definite at every frequency."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((2, NAUX, NAUX)) / NAUX
    proj = -np.einsum('tPR,tQR->tPQ', a, a)
    cosft = rng.random((NFREQ_SYNTH, 2))
    slices = [rng.standard_normal((NAUX, NORB)) for _ in range(NSLICE)]
    return proj, cosft, slices


def test_one_screened_slice_at_a_time():
    proj, cosft, slices = synthetic_screening()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    got = cd_screening_contraction_multi(proj, cosft, slices)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    one = NAUX * NORB * 8
    beside = (NSLICE * NFREQ_SYNTH * NORB + 2 * NFREQ_SYNTH * NAUX ** 2) * 8
    print(f'[info] peak {(peak - beside) / one:.2f} slices beside the outputs '
          f'and chi0 for {NSLICE} slices')
    assert peak - beside < WTB_HELD_MAX * one, (peak - beside) / one
    eye = np.eye(NAUX)
    for k in range(NFREQ_SYNTH):
        solve = screening_solver(eye - np.tensordot(cosft[k], proj, axes=1))
        for s, B in enumerate(slices):
            assert np.array_equal(got[s][k],
                                  np.einsum('Pq,Pq->q', B, solve(B) - B))


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    states = list(range(nocc - 4, nocc + 3))
    weights = np.linspace(-1.0, 1.0, len(states)) + 0.05
    return dict(X=X, D=D, eps=eps, nocc=nocc, grid=grid, nu=nu, wt=wt,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]), states=states,
                weights=weights)


def held_count(slices):
    """How many slices a rank holds: a list holds all of them."""
    return len(slices.held) if hasattr(slices, 'held') else len(slices)


def census(w, seen, comm=None):
    """{name: array} of one weighted set solve on the pole-model route, the
    adjoints by the grid tiles this rank holds, and into seen[stage] the
    slices this rank holds at each stage."""
    ro = {}
    out = qp_set_gradient(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                          w['nu'], w['wt'], w['states'], w['weights'],
                          mu=w['mu'], residue_route='sop', route_out=ro,
                          rows_block=ROWS_BLOCK, comm=comm)
    seen['tape'] = held_count(ro['tape'].Bps)
    named = {'w': out[0], 'z': ro['z'], 'eps_bar': out[1],
             'wc': np.stack(ro['tape'].wcs)}
    for name, tiles in (('X_bar', out[2]), ('D_bar', out[3])):
        named.update({f'{name}[{t}]': tiles.tile(t).copy()
                      for t in tiles.mine})
    return named


@pytest.mark.parametrize('size', SIZES)
def test_no_rank_holds_the_set(water, size, monkeypatch):
    n = len(water['states'])
    seen = {r: {} for r in range(size)}
    real = qp_space_time.QPSetPass.direct_terms

    def counted(run):
        seen[run.rank]['B_p_bar'] = held_count(run.Bp_bars)
        seen[run.rank]['B_p'] = held_count(run.Bps)
        return real(run)

    monkeypatch.setattr(qp_space_time.QPSetPass, 'direct_terms', counted)
    with distributed(None):
        serial = census(water, {})
    seen[0].clear()
    ranks = run_simulated(lambda comm: census(water, seen[comm.Get_rank()],
                                              comm), size)
    for rank, out in enumerate(ranks):
        start, stop = contiguous_block(n, rank, size)
        assert seen[rank] == dict.fromkeys(('B_p_bar', 'B_p', 'tape'),
                                           stop - start), (size, rank,
                                                           seen[rank])
        moved = [name for name, a in out.items()
                 if a.tobytes() != serial[name].tobytes()]
        assert not moved, (size, rank, moved[:6])
    print(f'[info] {size} rank(s): slices held per rank '
          f'{[seen[r]["tape"] for r in range(size)]} of {n}')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
