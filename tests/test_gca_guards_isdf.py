"""Guards on the ISDF-K mean-field force and the couplings built on it."""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import BOHR_TO_ANGSTROM
from src.environments.environment import PointCharges
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.gradients.gwbse.derivative_coupling import (analytic_coupling,
                                                     analytic_ground_coupling)
from src.gradients.mean_field_response import fock_partial_skeleton
from src.gradients.gwbse.static_correction import qp_xc_correction_skeleton
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.isdf_mean_field import (isdf_mean_field_gradient,
                                           mean_field_exchange_wanted)
from tests.isdf_test_grid import g1, g1_numerics

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

AUXBASIS = 'cc-pvdz-ri'
WATER = (('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)),
         ('H', (0.0, -0.7572, -0.4692)))
#: Angstrom; the central difference's h^2 error is ~1e-7 Ha/Bohr here.
STEP = 1e-3


def water(shift=None, basis='cc-pvdz'):
    """Water, or water with one Cartesian component moved by `shift` Angstrom."""
    atom = [[s, list(c)] for s, c in WATER]
    if shift is not None:
        ia, x, d = shift
        atom[ia][1][x] += d
    return gto.M(atom=atom, basis=basis, verbose=0)


def isdf_rhf(mol):
    """A tightly converged ISDF-K Hartree-Fock mean field."""
    mf = isdf_jk_with_force(scf.RHF(mol), auxbasis=AUXBASIS, counts=g1(scf.RHF(mol)))
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    assert mf.converged
    return mf


def isdf_hse(mol):
    """ISDF-K HSE06: no long-range exact exchange (alpha = 0), 0.25 short-range."""
    mf = isdf_jk_with_force(dft.RKS(mol, xc='hse06'), auxbasis=AUXBASIS, counts=g1(dft.RKS(mol, xc='hse06')))
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    assert mf.converged
    return mf


def central_difference(energy, ia, x):
    """dE/dR of one Cartesian component, Ha/Bohr."""
    ep, em = energy((ia, x, STEP)), energy((ia, x, -STEP))
    return (ep - em) / (2.0 * STEP / BOHR_TO_ANGSTROM)


def test_isdf_mean_field_force_carries_point_charges():
    """The charges' pull on the nuclei and electrons enters the ISDF-K force."""
    charges = PointCharges([[1.5, 0.3, 1.8], [-1.2, -1.6, 0.9],
                            [0.4, 2.0, -1.9]], [0.6, -0.5, 0.4])

    def energy(shift):
        return charges.mean_field(water(shift), isdf_rhf).e_tot

    force = isdf_mean_field_gradient(charges.mean_field(water(), isdf_rhf))
    for ia, x in ((0, 0), (1, 2), (2, 1)):
        assert abs(force[ia, x] - central_difference(energy, ia, x)) < 1e-5


def test_screened_hybrid_force_carries_its_short_range_exchange():
    """alpha = 0, beta != 0: the exchange skeleton must still enter the force.

    libxc's HSE exchange hole leaves ~1e-8 Ha of noise in the energy, so the
    difference sits at 1e-5 (pyscf's own DF-HSE06 gradient shows the same);
    the dropped short-range skeleton is worth 1e-1 here.
    """
    mf = isdf_hse(water())
    force = isdf_mean_field_gradient(mf)
    for ia, x in ((0, 2), (1, 1)):
        fd = central_difference(lambda s: isdf_hse(water(s)).e_tot, ia, x)
        assert abs(force[ia, x] - fd) < 1e-4
    assert mean_field_exchange_wanted(mf)


def test_lagrangian_skeletons_refuse_an_interpolated_coulomb():
    """j_route='isdf' has no Coulomb derivative: every skeleton refuses it."""
    mf = isdf_jk_with_force(dft.RKS(water(), xc='pbe0'), auxbasis=AUXBASIS,
                 j_route='isdf', counts=g1(dft.RKS(water(), xc='pbe0')))
    mf.kernel()
    nocc = mf.mol.nelectron // 2
    gamma = np.diag(np.linspace(0.1, 1.0, mf.mo_coeff.shape[1]))
    with pytest.raises(NotImplementedError, match="j_route"):
        fock_partial_skeleton(mf, gamma, nocc)
    with pytest.raises(NotImplementedError, match="j_route"):
        qp_xc_correction_skeleton(mf, np.ones(mf.mo_coeff.shape[1]), nocc)


def fitted_rhf(mol, max_cycle=200):
    """A density-fitted Hartree-Fock mean field, stopped after `max_cycle`."""
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, max_cycle
    mf.kernel()
    return mf


def test_couplings_flag_a_loosely_converged_mean_field():
    """A coupling, like a force, warns about an SCF whose |F_ia| is left
    large, flags it in its diagnostics and the chain's `scf_flags`, and
    carries on (a numerical shortfall warns and carries a named flag)."""
    mol = water()
    chain = ExcitedStateChain(mol, fitted_rhf, state=0,
                              numerics=g1_numerics(mol, Numerics(isdf={'auxbasis': AUXBASIS},
                                                                 bse={'nroots': 3})))
    loose = fitted_rhf(mol, max_cycle=2)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        d01, diags = analytic_coupling(chain, 0, 1, mol=mol, mf=loose)
    assert np.all(np.isfinite(d01))
    assert [f['where'] for f in diags['scf_flags']] == ['analytic_coupling']
    with pytest.warns(RuntimeWarning, match='F_ia'):
        _, diags = analytic_ground_coupling(chain, 0, mol=mol, mf=loose)
    assert [f['where'] for f in diags['scf_flags']] == [
        'analytic_ground_coupling']
    assert {f['where'] for f in chain.scf_flags} == {
        'analytic_coupling', 'analytic_ground_coupling'}
