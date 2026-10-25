"""Sums over ranks in a fixed order: the one-rank bits at every rank count.

`mpi_grid.ordered_sum` gathers every rank's indexed addends and adds them in
index order onto the first; `mpi_grid.ordered_chain_sum` carries the running
sum from one owner to the next for addends too large to gather. Serially
each is the addends' own sum in that order, so a kernel that adds its tiles
in tile order gives the one-rank bits at any rank count, where `reduce_sum`
joins rank partials in an order the rank count decides. Gated on water
/cc-pVDZ RHF at 1, 2, 3, 5 and 8 simulated ranks, every sum the serial one
to a few ulp (`assert_same`: each rank's addends are its own computation of
the serial tiles; no method tolerance). The planted
defects are shown to move the bits of the sums; under the recording BLAS
their 1e-12 moves also lie far outside the few-ulp bar:

  * the helpers on random addends, owners round-robin (`ordered_sum`) and in
    consecutive runs (`ordered_chain_sum`), and the planted defect -- rank
    1's first addend moved by MOVE -- fails both;
  * the sites that use them, every call recorded and compared with the
    serial run's: chi0(0) of the static screening and the screened
    interaction W itself (`static_screening_matrix`), the dRPA correlation
    energy's frequency sum (`rpa_correlation_energy_space_time`), the
    quasiparticle set's eps_bar (gathered) and B_p_bar (chained) frequency
    sums (`qp_set_gradient`), the fitted Fock skeleton's (natm, 3) tile sum
    (`fitted_fock_skeleton`) and the Davidson trial space's pair-tile joins
    (`PairRows.join`); the screened diagonal's tile chain is gated in
    tests/test_davidson_preconditioner.py;
  * each site fails when the helper it calls moves the largest element of
    rank 1's first addend by MOVE.

What is NOT on the ordered sums yet, and why: qp_set_gradient's other outputs and the
dRPA adjoint still join rank partials elsewhere (their residue and proj_bar
sums), and the Davidson's block action still reduce-scatters its grid
products, so the composed forces stay at RANK_SPLIT_FORCE_TOL.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import df, gto, lib, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.gradients import skeleton_tiles  # noqa: E402
from src.Base.utils.parallel import mpi_grid  # noqa: E402
from src.Base.utils.grids.grids import gauss_legendre_grid  # noqa: E402
from src.Base.utils.parallel.mpi_grid import (
    current_comm, distributed, ordered_chain_sum, ordered_sum)
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.Base.utils.grids.time_frequency import TimeFrequencyGrid  # noqa: E402
from src.SingleReference.GW.space_time import separable_factors  # noqa: E402
from src.SingleReference.LinearResponse import space_time  # noqa: E402
from src.SingleReference.LinearResponse import trial_space  # noqa: E402
from src.SingleReference.LinearResponse.space_time import (  # noqa: E402
    rpa_correlation_energy_space_time, static_screening_matrix)
from src.SingleReference.LinearResponse.trial_space import PairRows  # noqa: E402
from src.gradients.gwbse import qp_space_time  # noqa: E402
from src.gradients.gwbse.qp_space_time import qp_set_gradient  # noqa: E402
from tests.test_mpi_routes import WATER  # noqa: E402
from tests.isdf_test_grid import g1  # noqa: E402
from tests.pins import RECORDING_BLAS, assert_same, matches  # noqa: E402

SIZES = [2, 3, 5, 8]
#: Pair rows per trial-space tile: benzene-sized pair spaces in many tiles.
PAIR_TILE = 16
#: The planted defect's relative move of one element of one addend: a few
#: thousand ulp, which no rounding of the sum can absorb.
MOVE = 1e-12


@pytest.fixture(scope='module')
def water():
    """Water's RHF, its separable factors and the space-time grid."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')       # pyscf without OpenMP warns
        lib.num_threads(1)
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    with distributed(None):
        X, D = separable_factors(mf, mol, auxbasis='cc-pvdz-ri',
                                counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(24, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(8, 0.5 * gap, eps[-1] - eps[0], nu,
                                           wt, with_sine=False,
                                           with_inverse=False)
    return dict(mol=mol, mf=mf, nocc=nocc, X=X, D=D, eps=eps, nu=nu, wt=wt,
                grid=grid, mu=0.5 * (eps[nocc - 1] + eps[nocc]))


def recording(monkeypatch, module, name, plant=False):
    """Every call of `module.<name>` (an ordered sum) logged per rank as
    (call number, result copy); with `plant`, the largest element of rank
    1's first addend moved by MOVE before the sum."""
    real = getattr(mpi_grid, name)
    log = {}

    def wrapped(terms, *args, **kwargs):
        comm = args[1] if name == 'ordered_chain_sum' else args[0]
        rank = 0 if comm is None else comm.Get_rank()
        if plant and rank == 1 and terms:
            if name == 'ordered_sum':
                terms = list(terms)
                k, a = terms[0]
                a = np.array(a, dtype=float, copy=True)
                a.flat[np.abs(a).argmax()] *= 1 + MOVE
                terms[0] = (k, a)
            else:
                terms = dict(terms)
                k = min(terms)
                a = np.array(terms[k], dtype=float, copy=True)
                a.flat[np.abs(a).argmax()] *= 1 + MOVE
                terms[k] = a
        out = real(terms, *args, **kwargs)
        log.setdefault(rank, []).append(np.array(out, copy=True))
        return out

    monkeypatch.setattr(module, name, wrapped)
    return log


def same_calls(ref, got):
    """Every call's result the serial run's to a few ulp, call for call."""
    return len(ref) == len(got) and all(
        a.shape == b.shape
        and matches(b, a, RECORDING_BLAS, unit='1', quantity='root')
        for a, b in zip(ref, got))


def bitwise_calls(ref, got):
    """Every call's result the serial run's bits, call for call: what a
    planted move must break."""
    return len(ref) == len(got) and all(
        a.shape == b.shape and np.array_equal(a, b) for a, b in zip(ref, got))


# ----------------------------------------------------------------- helpers
@pytest.mark.parametrize('size', SIZES)
def test_the_helpers_are_the_one_rank_sum(size):
    """Random addends, round-robin and consecutive owners: every rank holds
    the serial sum to a few ulp."""
    rng = np.random.default_rng(size)
    n = 23
    terms = [rng.standard_normal((7, 5)) * 10.0 ** rng.integers(-8, 8)
             for _ in range(n)]
    onto = np.zeros((7, 5))
    serial = ordered_sum(list(enumerate(terms)), None, onto=onto)
    chain_serial = ordered_chain_sum(dict(enumerate(terms)), [0] * n, None,
                                     onto)
    assert_same(chain_serial, serial, unit='1', quantity='root',
                label='chained against gathered')
    owners = [r for r in range(size)
              for _ in range(*mpi_grid.contiguous_block(n, r, size))]

    def rank(comm):
        r = comm.Get_rank()
        mine = [(k, terms[k]) for k in range(r, n, size)]
        chain = {k: terms[k] for k in range(n) if owners[k] == r}
        return (ordered_sum(mine, comm, onto=onto),
                ordered_chain_sum(chain, owners, comm, onto))

    for a, b in run_simulated(rank, size):
        assert_same(a, serial, unit='1', quantity='root', label='ordered_sum')
        assert_same(b, serial, unit='1', quantity='root',
                    label='ordered_chain_sum')


def test_a_moved_addend_fails_both_helpers():
    """Rank 1's first addend moved by MOVE: neither sum is the serial one."""
    rng = np.random.default_rng(0)
    n, size = 12, 3
    terms = [rng.standard_normal(9) for _ in range(n)]
    onto = np.zeros(9)
    serial = ordered_sum(list(enumerate(terms)), None, onto=onto)
    owners = [r for r in range(size)
              for _ in range(*mpi_grid.contiguous_block(n, r, size))]

    def rank(comm):
        r = comm.Get_rank()
        moved = [t.copy() for t in terms]
        if r == 1:
            k = owners.index(1)
            moved[k][0] *= 1 + MOVE
            moved[1][0] *= 1 + MOVE
        mine = [(k, moved[k]) for k in range(r, n, size)]
        chain = {k: moved[k] for k in range(n) if owners[k] == r}
        return (ordered_sum(mine, comm, onto=onto),
                ordered_chain_sum(chain, owners, comm, onto))

    for a, b in run_simulated(rank, size):
        assert not np.array_equal(a, serial)
        assert not np.array_equal(b, serial)


# ------------------------------------------------------------------- sites
def chi0_static(w):
    return static_screening_matrix(w['X'], w['D'], w['eps'], w['nocc'])


def rpa_energy(w):
    return rpa_correlation_energy_space_time(w['X'], w['D'], w['eps'],
                                             w['nocc'], w['grid'], mu=w['mu'])


def qp_set(w):
    n = w['nocc']
    return qp_set_gradient(w['X'], w['D'], w['eps'], n, w['grid'], w['nu'],
                           w['wt'], [n - 2, n - 1, n, n + 1],
                           np.array([0.3, -1.1, 0.8, 0.45]), mu=w['mu'],
                           residue_route='explicit')


def skeleton(w):
    mol, mf = w['mol'], w['mf']
    auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    rng = np.random.default_rng(1)
    g = rng.standard_normal((mol.nao, mol.nao))
    cd = np.sqrt(2.0) * mf.mo_coeff[:, :w['nocc']]       # dm = Cd Cd^T
    return skeleton_tiles.fitted_fock_skeleton(
        mol, auxmol, g + g.T, cd @ cd.T, occ=cd, exchange=0.25, tile=8,
        comm=current_comm())


def pair_joins(w):
    """Three joins over a pair space of 300 rows in PAIR_TILE tiles: random
    per-tile addends, as the Davidson's projected blocks are."""
    comm = current_comm()
    rows = PairRows(300, comm, tile=PAIR_TILE)
    rng = np.random.default_rng(2)
    allt = [rng.standard_normal((2, 6, 4)) for _ in rows.all_tiles]
    mine = allt[rows.first_tile:rows.first_tile + len(rows.tiles)]
    return [rows.join(mine, (2, 6, 4)) for _ in range(3)]


# chi0(0) is patched where it lives: static_screening_matrix is in
# LinearResponse/space_time.py and calls space_time.ordered_chain_sum; the
# davidson's ordered_chain_sum is its block-action B chain, not chi0.
SITES = {
    'chi0(0)': (chi0_static, space_time, 'ordered_chain_sum'),
    'E_c': (rpa_energy, space_time, 'ordered_sum'),
    'qp_set eps_bar': (qp_set, qp_space_time, 'ordered_sum'),
    'qp_set B_p_bar': (qp_set, qp_space_time, 'ordered_chain_sum'),
    'fitted skeleton': (skeleton, skeleton_tiles, 'ordered_sum'),
    'trial-space joins': (pair_joins, trial_space, 'ordered_sum'),
}


@pytest.mark.parametrize('site', list(SITES))
def test_every_site_is_the_one_rank_sum(water, monkeypatch, site):
    """Every call of the site's ordered sum, at 2, 3, 5 and 8 ranks, every
    rank, the serial call to a few ulp; chi0(0)'s W and E_c themselves too."""
    run, module, name = SITES[site]
    log = recording(monkeypatch, module, name)
    with distributed(None):
        ref_out = run(water)
    ref = log.pop(0)
    assert ref, f'{site}: the serial run made no ordered sum'
    for size in SIZES:
        log.clear()
        outs = run_simulated(lambda comm: run(water), size)
        for r in range(size):
            assert same_calls(ref, log.get(r, [])), (site, size, r)
        if site in ('chi0(0)', 'E_c'):
            for out in outs:
                assert_same(out, ref_out, unit='1', quantity='root',
                            label=(site, size))
    print(f'[info] {site}: {len(ref)} ordered sum(s), every call to a few ulp of the '
          f'serial one at {SIZES} ranks')


@pytest.mark.parametrize('site', list(SITES))
def test_every_site_fails_a_moved_addend(water, monkeypatch, site):
    """Rank 1's first addend moved by MOVE at the site: some call is no
    longer the serial one."""
    run, module, name = SITES[site]
    log = recording(monkeypatch, module, name)
    with distributed(None):
        run(water)
    ref = log.pop(0)
    monkeypatch.undo()
    log = recording(monkeypatch, module, name, plant=True)
    run_simulated(lambda comm: run(water), 3)
    assert not all(bitwise_calls(ref, log.get(r, [])) for r in range(3)), site


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
