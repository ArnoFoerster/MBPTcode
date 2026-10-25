"""The space-time GW and the static W read their quadrature off the one
`Numerics`.

Keywords of their own (ntau, nfreq, npade, w0, tau_target and tile_gb on
`solve_qp_energy_space_time`, ntau and tau_target on `isdf_bse_factors`) with
their own defaults would let a solve that does not hand its grids on run on
those defaults: the adaptive selection's space-time GW whatever the chain's
were, and `solve_bse_isdf`'s rebuilt W whatever its GW stage's were. The
minimax target has ONE default (DEFAULT_TAU_TARGET = 1e-10), not also
`minimax_points_for_accuracy`'s own 1e-6. Both entry points take `numerics` and
read `numerics.quadrature` (and `layout.tile_gb`), `solve_bse_isdf` hands its
GW stage and its W the one object it holds, a second one inside `gw_kwargs`
is refused, and the target has one default, in constants.py.

Water/cc-pVDZ Hartree-Fock on the G1 grid, BSE@HF: no quasiparticle solve.
"""
import inspect

import pytest
from pyscf import gto, scf

from src.Base.numerics import constants
from src.Base.numerics.numerics import Numerics
from src.Base.utils.grids import time_frequency
from src.SingleReference.GW import qp_selection, space_time
from src.SingleReference.BSE import solve as bse_solve
from tests.isdf_test_grid import g1_kw

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
RETIRED = {'ntau', 'nfreq', 'npade', 'w0', 'tau_target', 'tile_gb'}


def test_the_entry_points_take_the_object():
    for fn in (space_time.solve_qp_energy_space_time,
               bse_solve.isdf_bse_factors):
        params = inspect.signature(fn).parameters
        assert 'numerics' in params, fn.__name__
        assert not RETIRED & set(params), (fn.__name__,
                                           RETIRED & set(params))
    shifts = inspect.signature(qp_selection.ac_quasiparticle_shifts)
    assert shifts.parameters['numerics'].default is inspect.Parameter.empty


def test_one_minimax_target():
    """No second default beside the one the object holds."""
    target = inspect.signature(
        time_frequency.minimax_points_for_accuracy).parameters['target']
    assert target.default is inspect.Parameter.empty
    assert Numerics().quadrature.tau_target == constants.DEFAULT_TAU_TARGET
    assert time_frequency.DEFAULT_TAU_TARGET is constants.DEFAULT_TAU_TARGET


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def test_the_rebuilt_w_reads_the_solves_quadrature(water, monkeypatch):
    """The static W `solve_bse_isdf` rebuilds sits on the grid the
    object names, not on `isdf_bse_factors`' own default."""
    mol, mf = water
    asked = []
    real = bse_solve.static_screening_grid

    def spy(eps, nocc, ntau=None, tau_target=None):
        asked.append((ntau, tau_target))
        return real(eps, nocc, ntau=ntau, tau_target=tau_target)

    monkeypatch.setattr(bse_solve, 'static_screening_grid', spy)
    numerics = Numerics(bse={'nroots': 2},
                        quadrature={'ntau_w': 10, 'tau_target': 1e-8})
    bse_solve.solve_bse_isdf(mf, mol, mol.nelectron // 2, numerics=numerics,
                            qp=False, progress=False, **g1_kw(mol, {}))
    assert asked == [(10, 1e-8)]


def test_a_second_object_in_gw_kwargs_is_refused(water):
    mol, mf = water
    with pytest.raises(TypeError, match='numerics'):
        bse_solve.solve_bse_isdf(mf, mol, mol.nelectron // 2, qp=False,
                                progress=False,
                                gw_kwargs={'numerics': Numerics()},
                                **g1_kw(mol, {}))
