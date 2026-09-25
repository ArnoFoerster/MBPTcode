"""Gates for src/gradients/rpa_ground_state.py -- the cubic dRPA ground-state gradient.

The analytic gradient is differenced against E_HF + E_c of the chain's own
function (SCF, factors and energy redone at each displacement), on a distorted
C1 water so that no symmetry can hide a missing term, and with both the direct
and the density-fitted skeleton, which are different code paths. The
RPA gradient on the same density-fitted energy is the physical reference:
the two differ by the interpolation grid alone, so the difference has to
shrink as the grid densifies.
"""
import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

from src.Base.separable_ri import shipped_radii_lookup
from src.SingleReference.GW.quasi_boson import build_rpa_AB
from src.SingleReference.LinearResponse.space_time import (
    rpa_correlation_energy_space_time)
from src.gradients.df_assembly import df_eri_mo
from src.gradients.grad_engine import correlation_gradient
from src.gradients.quasi_boson_adjoint import RPAAdjoint as RPA
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.targets import rpa_partials

BASIS = 'cc-pvdz'
H2O_C1 = 'O 0.03 0.02 0.117; H 0.10 0.757 -0.468; H -0.05 -0.80 -0.40'


def _converge(mf):
    """conv_tol_grad 1e-11: the Lagrangian assumes F_ia = 0."""
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def scf_df(mol):
    return _converge(scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri'))


def scf_direct(mol):
    return _converge(scf.RHF(mol))


@pytest.fixture(scope='module')
def water_df():
    mol = gto.M(atom=H2O_C1, basis=BASIS, verbose=0)
    mf = scf_df(mol)
    return mol, mf, RPAGroundStateChain(mol, scf_df, mf=mf)


def finite_difference(chain, mol, h=1e-4):
    """Five-point stencil of E_HF + E_c on every Cartesian component."""
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                d = np.zeros((mol.natm, 3))
                d[ia, x] = k * h
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(chain.energy(m)[0])
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    return fd


@pytest.mark.parametrize('scf_factory', [scf_df, scf_direct],
                         ids=['density-fitted', 'direct'])
def test_total_gradient_vs_finite_difference(scf_factory):
    """dE/dR = dE_HF/dR + dE_c/dR against a finite difference of the SUM."""
    mol = gto.M(atom=H2O_C1, basis=BASIS, verbose=0)
    chain = RPAGroundStateChain(mol, scf_factory)
    g, e, diags = chain.total_gradient()
    assert e == pytest.approx(chain.energy()[0], abs=1e-12)
    assert diags['stationarity'] < 1e-9
    assert diags['translation_residual'] < 1e-10
    fd = finite_difference(chain, mol)
    assert np.abs(fd - g).max() / np.abs(g).max() < 1e-6


def test_energy_matches_the_production_space_time_routine(water_df):
    """The chain's E_c IS the production imaginary-time energy, bitwise.

    The chain's forward pass calls `rpa_correlation_energy_space_time`, so
    there is one dRPA energy in the code and `==` is the only tolerance that
    states it: an approximate agreement here would be satisfied by a second
    implementation that had drifted.
    """
    mol, mf, chain = water_df
    crd = chain.coords(mol)
    x_ao, d = chain.factors(mol, chain.auxmol(mol), crd)
    e_prod = rpa_correlation_energy_space_time(
        x_ao @ mf.mo_coeff, d, np.asarray(mf.mo_energy, float), chain.nocc,
        chain.grid)
    assert chain.correlation_energy() == e_prod
    e, e_hf, e_c = chain.energy()
    assert e_hf == pytest.approx(mf.e_tot, abs=1e-12)
    assert e == pytest.approx(e_hf + e_c, abs=1e-12)


def _dense_df_reference(mol, mf):
    """E_c and dE_c/dR of dRPA on the density-fitted ERIs of the same auxiliary basis."""
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=BASIS + '-ri')
    eri = df_eri_mo(mol, auxmol, mf.mo_coeff)
    nocc = mol.nelectron // 2
    A, B, _ = build_rpa_AB(np.asarray(mf.mo_energy, float), eri, nocc)
    qb = RPA(A, B)
    gammaF, Gamma4 = rpa_partials(qb, nocc, mol.nao)
    g, _ = correlation_gradient(mol, mf, gammaF, Gamma4, eri_mo=eri, auxmol=auxmol)
    return qb.e_corr(), g


def test_grid_error_against_the_dense_rpa_gradient_shrinks_with_the_grid(water_df):
    """ISDF against dense on one fitted energy: the interpolation grid is the only
    difference, and densifying it must close the gap in E_c AND in the force.

    The force is the sensitive one: at 148 points/atom E_c is off by 6e-5 of
    itself and the force by 0.6% of its largest component.

    BOTH RUNGS ARE ROWS OF THE SHIPPED RADII TABLE. Its rows are optimized one
    count at a time, so neighbouring counts are not a nested ladder -- 244 and
    296 points/atom miss E_c by more than 148 does -- and an untabulated count
    is re-optimized at run time into whatever local minimum the machine or its
    radii cache holds. 148 against 592 points/atom is ordered by 3x in E_c and
    by 100x in the force.
    """
    mol, mf, chain = water_df
    e_dense, g_dense = _dense_df_reference(mol, mf)
    fine = RPAGroundStateChain(mol, scf_df, mf=mf,
                               counts={'A1': 32, 'A2': 20, 'A3': 12, 'B1': 4})
    for rung in (chain, fine):
        for el in ('O', 'H'):
            assert shipped_radii_lookup(el, BASIS, BASIS + '-ri',
                                        rung.factorization.counts) is not None
    g_coarse, e_coarse, _ = chain.correlation_gradient()
    g_fine, e_fine, _ = fine.correlation_gradient()
    assert abs(e_coarse - e_dense) < 5e-4
    assert abs(e_fine - e_dense) < abs(e_coarse - e_dense)
    d_coarse = np.abs(g_coarse - g_dense).max()
    d_fine = np.abs(g_fine - g_dense).max()
    assert d_coarse < 2e-3
    assert d_fine < d_coarse / 3
