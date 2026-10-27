"""Gates for the AO collocation the space-time self-energy reads when it is
handed the MO collocation alone (`space_time._ao_collocation`), on a mean
field that dropped linearly dependent AO combinations (nmo < nao).

Augmented bases run in the full basis (no span cut), so nmo < nao is the
normal case for a large emitter in aug-cc-pVDZ: 2CzPN dropped 13 of 990.
The chain hands the analytic continuation of the outside curve (X_mo, D)
only, and the AO self-energy then needs an X_ao with X_ao C = X_mo. The left
inverse X_mo C^T S is one, since C^T S C = I whether or not C is square; it
is the fit's X_ao projected onto the span of C, and the self-energy reads
X_ao only through orbitals in that span. Water/aug-cc-pVDZ (HF, G1 ISDF),
with every overlap eigenvalue below 3e-2 dropped (41 AOs, 37 MOs):

  * the left inverse gives back X_mo, and is not the fit's X_ao;
  * the AC roots and pole strengths on (X_mo, D) are those on the fit's
    (X_mo, D, X_ao);
  * the curve chain's forward places both outside curves.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.scf import hf as scf_hf

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.space_time import (_ao_collocation,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1, g1_numerics

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Drops the four overlap eigenvalues of water/aug-cc-pVDZ below it.
LINDEP = 3e-2


def factory(mol):
    """The mean field with the overlap eigenvalues below LINDEP dropped.

    pyscf >= 2.13 drops them inside `hf.eig` (module settings, threshold
    1e-6 by default: what dropped 2CzPN's 13) and no longer converges with
    the deprecated `remove_linear_dep_` on top; earlier pyscf has only that.
    """
    mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-8
    if hasattr(scf_hf, 'overlap_zero_eigenvalue_threshold'):
        saved = (scf_hf.remove_overlap_zero_eigenvalue,
                 scf_hf.overlap_zero_eigenvalue_threshold)
        scf_hf.remove_overlap_zero_eigenvalue = True
        scf_hf.overlap_zero_eigenvalue_threshold = LINDEP
        try:
            mf.kernel()
        finally:
            (scf_hf.remove_overlap_zero_eigenvalue,
             scf_hf.overlap_zero_eigenvalue_threshold) = saved
    else:
        mf = scf.addons.remove_linear_dep_(mf, threshold=LINDEP,
                                           lindep=LINDEP)
        mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis='aug-cc-pvdz', verbose=0)
    mf = factory(mol)
    assert mf.mo_coeff.shape == (41, 37)
    return mol, mf, separable_factors(mf, mol, counts=g1(mol))


def test_the_left_inverse_gives_back_the_mo_collocation(water):
    mol, mf, (x_mo, _, x_ao, _) = water
    x_left = _ao_collocation(x_mo, mf)
    assert np.abs(x_left @ mf.mo_coeff - x_mo).max() < 1e-12
    # the projection onto the span of C: not the fit's X_ao
    assert np.abs(x_left - x_ao).max() > 1e-2


def test_the_ac_reads_the_mo_collocation_as_the_fits(water):
    mol, mf, (x_mo, d, x_ao, _) = water
    nocc = mol.nelectron // 2
    states = np.arange(mf.mo_coeff.shape[1])
    left, z_left = solve_qp_energy_space_time(
        mf, mol, nocc, states, factors=(x_mo, d), with_z=True,
        distribute=False)
    fit, z_fit = solve_qp_energy_space_time(
        mf, mol, nocc, states, factors=(x_mo, d, x_ao), with_z=True,
        distribute=False)
    assert np.all(np.isfinite(left))
    np.testing.assert_allclose(left, fit, rtol=0, atol=1e-12)
    # Z is a central difference of the continuation: rounding amplified
    np.testing.assert_allclose(z_left, z_fit, rtol=0, atol=1e-8)


def test_the_curve_chain_places_both_curves(water):
    mol, mf, _ = water
    chain = ExcitedStateChain(
        mol, factory, mf=mf,
        numerics=g1_numerics(mol, Numerics(qp={'outside_scissor': 'curve'})))
    om = chain.forward(mol, mf)[1].omega
    assert np.all(np.isfinite(om))
    assert set(chain.resolved.outside_curve) == {'occupied', 'virtual'}
