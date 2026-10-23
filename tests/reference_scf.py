"""Converged restricted Hartree-Fock references the tests share.

Each helper fixes one SCF recipe (auxiliary basis, convergence thresholds,
cycle cap); the name says which.
"""
from pyscf import scf

from src.Base.numerics.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)

#: The orbital basis the differentiable references below are built in; its
#: `-ri` and `-jkfit` partners are the auxiliary basis.
BASIS = 'cc-pvdz'


def rhf_df_jkfit(mol):
    """Density-fitted RHF on cc-pvdz-jkfit, converged to 1e-12 Ha."""
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def rhf_tight(mol):
    """A reference converged tightly enough to differentiate."""
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    return mf


def rhf_tight_default_cycles(mol):
    """RHF converged to 1e-13 Ha: the Lagrangian assumes the occupied-virtual
    Fock block vanishes."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    return mf


def rhf_conv_1e12(mol):
    """RHF converged to 1e-12 Ha."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def rhf_df_ri_1e12(mol):
    """Density-fitted RHF on cc-pvdz-ri, converged to 1e-12 Ha."""
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    mf.kernel()
    return mf


def rhf_df_differentiable(mol):
    """Density-fitted RHF on cc-pvdz-ri at the differentiable thresholds."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def rhf_df_differentiable_checked(mol):
    """`rhf_df_differentiable`, asserting that the SCF converged."""
    mf = rhf_df_differentiable(mol)
    assert mf.converged
    return mf


def rhf_df_jkfit_differentiable_checked(mol):
    """Density-fitted RHF on cc-pvdz-jkfit at the differentiable thresholds,
    asserting that the SCF converged."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-jkfit')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf
