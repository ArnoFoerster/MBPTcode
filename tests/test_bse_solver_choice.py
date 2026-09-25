"""One rule decides dense-vs-Davidson, in one place, fed by one constant.

`solver_choice` is that rule, and BSE_DENSE_MAX_GB is the memory it compares
against: the dense route's Casida pair (A, B), 2 * n_ov**2 * 8 bytes. The
tests below pin where the boundary sits, that Tamm-Dancoff does not move it,
that `solve_bse(solver='auto')` runs the resolved route and reports it, and
that the two explicit routes still agree on water.

The boundary test is a check that can fail, shown once on two perturbations of
`solver_choice`: weakening `<=` to `<`, which pushes the exactly-12000-pair
case -- the storage equal to the limit -- over to Davidson; and dropping the
factor 2 from the storage so only A is counted, which keeps 12001 pairs dense
and is the Tamm-Dancoff exemption this rule refuses.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import BSE_DENSE_MAX_GB, BSE_DENSE_MAX_NOV
from src.SingleReference.LinearResponse import bse as bse_module
from src.SingleReference.LinearResponse.bse import (_check_route_kwargs,
                                                    solve_bse, solver_choice)

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-jkfit'
NROOTS = 3
#: Hartree. BSE@G0W0 on water/cc-pVDZ RHF, integrals='df' in the Coulomb-fitting
#: auxiliary basis, probe off -- the three lowest roots of the front end before
#: `solver_choice` existed, and what both explicit solvers must still return.
REFERENCE_OMEGA = np.array([0.310309502912, 0.385855109783, 0.407478088639])
#: Hartree. The Davidson stops on a residual of 1e-5 (its default `conv_tol`),
#: which lands the roots this far from the dense eigh; measured here at 1.6e-13.
DAVIDSON_TOL = 1e-8
#: Hartree. The dense route is one eigh of a fixed matrix, so only the BLAS
#: scatters.
DENSE_TOL = 1e-9


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mol, mf, mol.nelectron // 2


def test_the_boundary_is_the_memory_form_of_the_pair_count():
    """One more pair than the limit is one pair too many: the storage at
    BSE_DENSE_MAX_NOV is exactly BSE_DENSE_MAX_GB and goes dense, and the next
    pair does not."""
    assert 2 * BSE_DENSE_MAX_NOV**2 * 8 / 1e9 == BSE_DENSE_MAX_GB
    assert solver_choice(BSE_DENSE_MAX_NOV) == 'dense'
    assert solver_choice(BSE_DENSE_MAX_NOV - 1) == 'dense'
    assert solver_choice(BSE_DENSE_MAX_NOV + 1) == 'davidson'
    assert solver_choice(10 * BSE_DENSE_MAX_NOV) == 'davidson'


def test_tamm_dancoff_flips_at_the_same_size():
    """TDA diagonalizes A alone but the dense route BUILDS B with it, so the
    rule counts the pair for both kernels and they cross over together."""
    for n_ov in (1, BSE_DENSE_MAX_NOV - 1, BSE_DENSE_MAX_NOV,
                 BSE_DENSE_MAX_NOV + 1, 10 * BSE_DENSE_MAX_NOV):
        assert solver_choice(n_ov, tda=True) == solver_choice(n_ov, tda=False)
    assert solver_choice(BSE_DENSE_MAX_NOV, tda=True) == 'dense'
    assert solver_choice(BSE_DENSE_MAX_NOV + 1, tda=True) == 'davidson'


def test_water_is_dense(water):
    """95 pairs is 144 kB of Casida blocks: the size at which iterating is the
    more expensive way to get three roots."""
    mol, mf, nocc = water
    n_ov = nocc * (mf.mo_coeff.shape[-1] - nocc)
    assert n_ov == 95
    assert solver_choice(n_ov) == 'dense'


def test_auto_resolves_to_dense_and_records_what_it_ran(water):
    """'auto' is resolved before any route runs, so info reports the route that
    actually ran and the numbers are the dense route's to the last bit."""
    mol, mf, nocc = water
    om_auto, x_auto, y_auto, info_auto = solve_bse(
        mf, mol, nocc, nroots=NROOTS, solver='auto', integrals='df', probe=False)
    om_dense, x_dense, y_dense, _ = solve_bse(
        mf, mol, nocc, nroots=NROOTS, solver='dense', integrals='df', probe=False)
    assert info_auto['solver'] == 'dense'
    assert np.array_equal(om_auto, om_dense)
    assert np.array_equal(x_auto, x_dense)
    assert np.array_equal(y_auto, y_dense)


def test_both_explicit_solvers_return_the_reference_roots(water):
    """The regression the one-rule change had to leave untouched: the two
    explicit routes on one mean field, agreeing with each other to the
    Davidson's convergence and with the recorded roots."""
    mol, mf, nocc = water
    om_dense = solve_bse(mf, mol, nocc, nroots=NROOTS, solver='dense',
                         integrals='df', probe=False)[0]
    om_dav = solve_bse(mf, mol, nocc, nroots=NROOTS, solver='davidson',
                       integrals='df', probe=False)[0]
    assert np.abs(np.sort(om_dense) - REFERENCE_OMEGA).max() < DENSE_TOL
    assert np.abs(np.sort(om_dav) - np.sort(om_dense)).max() < DAVIDSON_TOL


def test_the_isdf_route_takes_the_grid_and_the_mpi_keywords():
    """`solve_bse_isdf` has accepted these three all along; the front end that
    refused them was a downgrade for any distributed or grid-tailored run."""
    for key in ('grid_accuracy', 'distribute', 'comm'):
        _check_route_kwargs('davidson', 'isdf', {key: None})


def test_the_df_route_refuses_them_by_name():
    """A keyword that belongs to another route is refused, and the message
    names the keyword and the route that owns it -- never silently ignored."""
    for key in ('grid_accuracy', 'distribute', 'comm'):
        with pytest.raises(TypeError, match=key):
            _check_route_kwargs('davidson', 'df', {key: None})
    # The dense ISDF route takes the fit's keywords but not the MPI pair: there
    # is no distributed dense solve, and the fit itself takes neither.
    _check_route_kwargs('dense', 'isdf', {'grid_accuracy': 'G2'})
    for key in ('distribute', 'comm'):
        with pytest.raises(TypeError, match=key):
            _check_route_kwargs('dense', 'isdf', {key: None})


def test_grid_accuracy_reaches_the_isdf_route(water, monkeypatch):
    """Passed through unchanged, not swallowed or renamed on the way: the fit
    is asked for the level the caller named."""
    mol, mf, nocc = water
    seen = {}

    def spy(mf_, mol_, nocc_, **kwargs):
        seen.update(kwargs)
        return np.zeros(1), np.zeros((1, 1)), np.zeros((1, 1)), {}

    monkeypatch.setattr(bse_module, 'solve_bse_isdf', spy)
    solve_bse(mf, mol, nocc, nroots=NROOTS, solver='davidson', integrals='isdf',
              grid_accuracy='G2')
    assert seen['grid_accuracy'] == 'G2'
