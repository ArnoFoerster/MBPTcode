"""The energy routes and the gradient chain share ONE polarizability kernel.

`LinearResponse.space_time.polarizability_projected_tau` is the N^3 sweep of
the space-time route. The energy side streams it into chi0
(`chi0_imaginary_frequency`); the gradient side keeps its output whole
(`gradients.space_time_adjoint.polarizability_tau`). Until this test existed
the gradient side carried its own copy of the loop, and nothing pinned the two
to each other -- the tau partition added on the energy side never reached the
gradient side.

What is gated:
  * the gradient forward folded with the grid's transform equals the energy
    forward at 1e-13 relative. The two accumulate in a different order, so
    bitwise is not the claim; drift is.
  * the whole sweep equals a plain per-tau loop of the kernel bitwise, and so
    does `out=` with dirty buffers against a fresh allocation.
  * a sweep split over disjoint tau subsets sums to the full sweep bitwise,
    through both entry points -- which is what makes an all-reduce over ranks
    exact.
  * the gradient package's chi0 convenience IS the energy chi0.

Random factors rather than a molecule, as in test_selfenergy_block_backward:
this gates arithmetic identity, and a physical X, D would only make the same
check slower. M is not naux, and the tile budget is set so the sweep runs five
row tiles with a short last one, because the identity has to hold across tile
boundaries and not only inside one.
"""
import numpy as np

from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.LinearResponse.space_time import (
    chi0_imaginary_frequency, polarizability_projected_sweep,
    polarizability_projected_tau, polarizability_work, split_branches,
    tile_rows)
from src.gradients.space_time_adjoint import chi0_frequency, polarizability_tau

M, NORB, NOCC, NAUX, NTAU, NFREQ = 23, 8, 3, 6, 8, 5
#: Tile budget, in GB, that puts five grid rows in a tile at this M (the kernel
#: budgets 3 * M * 8 bytes per row), so the 23 rows go through five tiles.
TILE_GB = 5 * 3 * M * 8 / 1e9
#: Relative agreement between the two accumulation orders.
REL = 1e-13
#: Ranks in the partition test.
NRANKS = 3


def system(seed=11):
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.normal(size=NORB))
    eps[NOCC:] += 2.0                       # a gap, so the branches separate
    X = 0.25 * rng.normal(size=(M, NORB))
    D = 0.25 * rng.normal(size=(M, NAUX))
    gap = eps[NOCC] - eps[NOCC - 1]
    e_max = eps[-1] - eps[0]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, gap, e_max, nu, wt,
                                           with_sine=False, with_inverse=False)
    return eps, X, D, grid


def owned(rank):
    return np.arange(rank, NTAU, NRANKS)


def test_tile_budget_gives_five_tiles():
    assert tile_rows(M, TILE_GB, 3 * M * 8) == 5
    assert M % 5 != 0                       # the last tile is short


def test_gradient_forward_equals_energy_forward():
    eps, X, D, grid = system()
    proj = polarizability_tau(X, D, eps, NOCC, grid, tile_gb=TILE_GB)
    assert proj.shape == (NTAU, NAUX, NAUX)
    folded = np.tensordot(grid.cosft_wt, proj, axes=(1, 0))
    chi0 = chi0_imaginary_frequency(X, D, eps, NOCC, grid,
                                    tile_memory_gb=TILE_GB)
    scale = np.abs(chi0).max()
    assert scale > 0
    assert np.abs(folded - chi0).max() <= REL * scale
    # and the unstreamed reference the energy side keeps for small grids
    whole = chi0_imaginary_frequency(X, D, eps, NOCC, grid, stream=False)
    assert np.abs(whole - chi0).max() <= REL * scale


def test_sweep_is_the_per_tau_kernel_bitwise():
    eps, X, D, grid = system()
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, NOCC, None)
    sweep = polarizability_projected_sweep(X, D, eps, NOCC, grid.tau_points,
                                           tile_memory_gb=TILE_GB)
    for k, tau in enumerate(grid.tau_points):
        one = polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                           tile_memory_gb=TILE_GB)
        assert np.array_equal(sweep[k], one)


def test_out_and_work_buffers_change_nothing():
    eps, X, D, grid = system()
    X_o, X_v, e_o, e_v, _, _, _ = split_branches(X, eps, NOCC, None)
    tau = grid.tau_points[NTAU // 2]
    fresh = polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                         tile_memory_gb=TILE_GB)
    # DIRTY buffers: the kernel owns their contents and must not read them
    out = np.full((NAUX, NAUX), np.nan)
    work = polarizability_work(M, TILE_GB)
    for w in work:
        w[:] = np.nan
    got = polarizability_projected_tau(X_o, X_v, e_o, e_v, D, tau,
                                       tile_memory_gb=TILE_GB, out=out,
                                       work=work)
    assert got is out
    assert np.array_equal(got, fresh)


def test_disjoint_tau_subsets_sum_to_the_whole():
    eps, X, D, grid = system()
    full = polarizability_projected_sweep(X, D, eps, NOCC, grid.tau_points,
                                          tile_memory_gb=TILE_GB)
    parts = [polarizability_projected_sweep(X, D, eps, NOCC, grid.tau_points,
                                            tau_indices=owned(r),
                                            tile_memory_gb=TILE_GB)
             for r in range(NRANKS)]
    for r, part in enumerate(parts):
        others = np.setdiff1d(np.arange(NTAU), owned(r))
        assert np.all(part[others] == 0.0)
        assert np.all(part[owned(r)] != 0.0)
    assert np.array_equal(sum(parts), full)
    # the same partition through the gradient chain's entry point
    via_gradient = sum(polarizability_tau(X, D, eps, NOCC, grid,
                                          tile_gb=TILE_GB, tau_indices=owned(r))
                       for r in range(NRANKS))
    assert np.array_equal(via_gradient, full)


def test_gradient_chi0_is_the_energy_chi0():
    eps, X, D, grid = system()
    a = chi0_frequency(X, D, eps, NOCC, grid, tile_gb=TILE_GB)
    b = chi0_imaginary_frequency(X, D, eps, NOCC, grid, tile_memory_gb=TILE_GB)
    assert np.array_equal(a, b)
