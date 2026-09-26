"""The rank and the grids, both CHOSEN rather than hardcoded.

Two things were hardcoded in the periodic THC route and are not any more, and
each was hardcoded in a way that is wrong at one end of a size series:

  * the RANK. `npoints` was an absolute integer with no default, so every
    caller wrote `ALPHA * nmo` by hand and the convention that actually
    transfers between systems -- alpha = N_mu / N_orb -- was never the thing
    being passed.
  * the TAU GRID. `ntau or 18`. R = e_max/e_min grows as the gap closes, so a
    fixed ntau is wrong at one end of any size series; and the periodic case
    is worse than the molecular one because the binding range is the
    SELF-ENERGY's, which is wider (measured 1.6x on diamond).

The accuracy the grid resolver returns is CHECKED here, not just recorded: a
resolver whose `worst_error` is discarded lets a grid that quietly failed to
reach its target read as a converged answer.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
from src.SingleReference.Periodic.pbc_isdf import resolve_npoints
from src.SingleReference.Periodic.pbc_isdf_gw import minimax_points_for_gw_k


# --- the rank ---------------------------------------------------------------

def test_resolve_npoints_accepts_either_but_not_both():
    assert resolve_npoints(None, 12, 10) == 120
    assert resolve_npoints(120, None, 10) == 120
    with pytest.raises(ValueError, match='not both and not neither'):
        resolve_npoints(None, None, 10)
    with pytest.raises(ValueError, match='not both and not neither'):
        resolve_npoints(120, 12, 10)
    with pytest.raises(ValueError, match='positive'):
        resolve_npoints(None, -1, 10)


def test_alpha_is_recorded_on_the_result(diamond_scf):
    """A result must carry the rank it was built at, not just its own number."""
    from src.SingleReference.Periodic.pbc_isdf import build_isdf_kpts
    cell, mf, kpts, mo = diamond_scf
    _, _, info = build_isdf_kpts(cell, mo, kpts, alpha=6)
    assert info['alpha'] == pytest.approx(6.0)
    assert info['npoints'] == 6 * info['nmo']


# --- the tau grid -----------------------------------------------------------

@pytest.fixture(scope='module')
def diamond_scf():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    cell.a = np.array([[0., 1.7834, 1.7834],
                       [1.7834, 0., 1.7834],
                       [1.7834, 1.7834, 0.]])
    cell.basis, cell.pseudo, cell.verbose = 'gth-szv', 'gth-pade', 0
    cell.build()
    kpts = cell.make_kpts([2, 2, 1])
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    mf.kernel()
    assert mf.converged
    return cell, mf, kpts, [np.asarray(c) for c in mf.mo_coeff]


def test_auto_ntau_resolves_and_reports_its_accuracy(diamond_scf):
    """It must return a grid AND the error it actually achieved."""
    cell, mf, kpts, _ = diamond_scf
    kplus = get_momentum_transfer_map(cell, kpts)
    n, err = minimax_points_for_gw_k(np.asarray(mf.mo_energy),
                                     np.asarray(mf.mo_occ), kplus)
    print(f"  auto ntau = {n}, worst fit error = {err:.2e}")
    assert 6 <= n <= 34
    assert err < 1e-9, err
    assert n % 2 == 0                       # minimax tables are even-sized


def test_auto_ntau_binds_on_the_self_energy_range(diamond_scf):
    """The WIDEST of the three ranges decides, and it is rS.

    Sigma = -G Wt is a product, so its decay rates are SUMS and rS exceeds
    both the transition window and the screening range. Sizing the grid on
    either of the others is a silent accuracy loss.
    """
    from src.SingleReference.Periodic.pbc_isdf_gw import self_energy_fit_ranges_k
    from src.Base.utils.time_frequency import minimax_points_for_accuracy
    from src.SingleReference.Periodic.pbc_isdf_rpa import transition_window_occ

    cell, mf, kpts, _ = diamond_scf
    e, f = np.asarray(mf.mo_energy), np.asarray(mf.mo_occ)
    kplus = get_momentum_transfer_map(cell, kpts)
    rW, rS = self_energy_fit_ranges_k(e, f)
    e_min, e_max = transition_window_occ(e, f, kplus)

    ratios = {'window': e_max / e_min, 'rW': rW[1] / rW[0], 'rS': rS[1] / rS[0]}
    assert ratios['rS'] == max(ratios.values()), ratios

    n_all, _ = minimax_points_for_gw_k(e, f, kplus)
    n_rS, _ = minimax_points_for_accuracy(1.0, ratios['rS'], target=1e-10)
    assert n_all == n_rS, (n_all, n_rS, ratios)


def test_auto_ntau_refuses_an_unreachable_target(diamond_scf):
    """An unmeetable target must raise from the driver, not pass silently."""
    from src.SingleReference.Periodic.pbc_isdf_gw import qp_energy_thc
    cell, mf, kpts, _ = diamond_scf
    with pytest.raises(ValueError, match='no tabulated minimax grid'):
        qp_energy_thc(cell, mf, alpha=4, exxdiv=None, tau_target=1e-30)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
