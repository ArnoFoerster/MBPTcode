"""The ISDF-K mean field on the AO-product fit and its pivoted points
(`IsdfNumerics.test_set='ao_products'`, `pivot_points`).

Gates:
  * the mean-field force is the derivative of its energy on a selection frozen
    at the reference and carried by the frames (`reference_frames`,
    `frozen_frames`): water C1/cc-pVDZ, PBE0 (a bare exchange channel) and
    LRC-wPBEh (a bare and an attenuated one), 250 points from the G1 parent,
    five-point difference along a generic direction within 1e-8 Ha/Bohr
    (measured 8.5e-10 PBE0);
  * against the density-fitted mean field on the same auxiliary basis,
    formaldehyde (no symmetry) LRC-wPBEh/aug-cc-pVDZ, 900 points from the G3
    parent: occupied orbital energies within 1.5 meV (measured 0.75), the
    five lowest virtuals within 0.5 meV (0.32), |dP|_S below 1e-3 (5.9e-4),
    the dipole within 1e-3 D (3.5e-4), the S0 force within 3e-4 Ha/Bohr
    (7.3e-5; the tabulated G3 grid, 1776 points: 1.8e-3);
  * the mean field and the post-SCF factors on the same parent, count and
    settings share one selection and one fit (the same arrays);
  * under three simulated ranks the distributed SCF (`DistributedISDFJK`)
    gives the one-rank energy to 1e-9 Ha;
  * the excited-state chain on such a mean field (Hartree-Fock: no lab-fixed
    xc grid) shares its fit, its S1 force is the derivative of its energy
    within ISDF_GRADIENT_FLOOR (measured 5e-10 on PBE0), carries no net force
    or torque, and a rigid rotation and translation leave the energy and the
    turned force unchanged.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.factorization.distributed_df import (distributed_mean_field,
                                                   release_distributed)
from src.Base.factorization.isdf_jk import frozen_frames, reference_frames
from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.isdf_mean_field import (isdf_jk_with_force,
                                           isdf_mean_field_gradient)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.rpa.factor_chain import FrozenFactorization
from tests.test_body_fixed_frames import (ENERGY_TOL, RIGID_TOL, five_point,
                                          moved, rigid_part, rotation, water)
from tests.pins import isdf_gradient_bar
from tests.molecules import WATER_C1

H2O_C1 = WATER_C1
HCHO_C1 = ('C 0 0.02 -0.5296; O 0.01 0 0.6763; H 0 0.9339 -1.1088; '
           'H 0.03 -0.9339 -1.1188')
FD_STEP = 1e-3


def isdf_k(mol, xc, numerics, auxbasis):
    mf = isdf_jk_with_force(dft.RKS(mol, xc=xc), auxbasis=auxbasis,
                            numerics=numerics)
    mf.grids.level = 4
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    return mf


def at(mol, x):
    m = mol.copy()
    m.set_geom_(x, unit='Bohr')
    m.build(False, False)
    return m


@pytest.mark.parametrize('xc', ['pbe0', 'lrc-wpbeh'])
def test_the_mean_field_force_is_its_derivative(xc):
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    numerics = Numerics(isdf={'test_set': 'ao_products', 'pivot_points': 250},
                        scf={'grid_level': 'G1'})
    mf = isdf_k(mol, xc, numerics, 'cc-pvdz-ri')
    mf.kernel()
    frames = reference_frames(mf)
    assert frames.selection is not None
    g = isdf_mean_field_gradient(mf)
    assert np.abs(g.sum(axis=0)).max() < 1e-10

    def energy(x):
        with frozen_frames(frames):
            m = isdf_k(at(mol, x), xc, numerics, 'cc-pvdz-ri')
            m.kernel()
            return m.e_tot
    d = np.random.default_rng(0).normal(size=g.shape)
    d /= np.linalg.norm(d)
    x0 = mol.atom_coords()
    v = [energy(x0 + k * FD_STEP * d) for k in (-2, -1, 1, 2)]
    fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
    assert abs(fd - float((g * d).sum())) < 1e-8


def test_the_mean_field_against_density_fitting():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=HCHO_C1, basis='aug-cc-pvdz', verbose=0)
    ref = dft.RKS(mol, xc='lrc-wpbeh').density_fit(auxbasis='aug-cc-pvdz-ri')
    ref.grids.level = 4
    ref.conv_tol, ref.conv_tol_grad = 1e-12, 1e-9
    ref.kernel()
    g_ref = ref.Gradients()
    g_ref.grid_response = True
    g_ref = g_ref.kernel()
    mf = isdf_k(mol, 'lrc-wpbeh', Numerics(
        isdf={'test_set': 'ao_products', 'pivot_points': 900},
        scf={'grid_level': 'G3'}), 'aug-cc-pvdz-ri')
    mf.kernel()
    assert mf.converged
    reference_frames(mf)
    g = isdf_mean_field_gradient(mf)
    nocc = mol.nelectron // 2
    de = 1e3 * HARTREE_TO_EV * (mf.mo_energy - ref.mo_energy)
    assert np.abs(de[:nocc]).max() < 1.5, de[:nocc]
    assert np.abs(de[nocc:nocc + 5]).max() < 0.5, de[nocc:nocc + 5]
    S = mol.intor('int1e_ovlp')
    dP = mf.make_rdm1() - ref.make_rdm1()
    assert np.sqrt(np.einsum('ij,jk,kl,li->', dP, S, dP, S)) < 1e-3
    assert np.linalg.norm(mf.dip_moment(verbose=0)
                          - ref.dip_moment(verbose=0)) < 1e-3
    assert np.abs(g - g_ref).max() < 3e-4


def test_the_mean_field_and_the_factors_share_one_fit():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    numerics = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                              'pivot_points': 250},
                        scf={'grid_level': 'G1'})
    mf = isdf_k(mol, 'pbe0', numerics, 'cc-pvdz-ri')
    mf.kernel()
    fz = FrozenFactorization(mol, numerics)
    assert np.array_equal(fz.coords(mol), mf.with_df.coords)
    _, M, _ = fz._fit(mol, fz.auxmol(mol), fz.coords(mol))
    assert M is mf.with_df.M


def test_the_distributed_scf_is_the_one_rank_scf():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    numerics = Numerics(isdf={'test_set': 'ao_products', 'pivot_points': 250},
                        scf={'grid_level': 'G1'})
    serial = isdf_k(mol, 'pbe0', numerics, 'cc-pvdz-ri')
    serial.kernel()

    def one(comm):
        mf = isdf_k(mol, 'pbe0', numerics, 'cc-pvdz-ri')
        distributed_mean_field(mf)
        e = float(mf.e_tot)
        release_distributed(mf)
        return e
    for e in run_simulated(one, 3):
        assert abs(e - serial.e_tot) < 1e-9



def test_the_chain_on_an_isdf_k_mean_field():
    warnings.simplefilter('ignore')
    numerics = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                              'pivot_points': 250},
                        scf={'grid_level': 'G1'})

    def factory(m):
        mf = isdf_jk_with_force(scf.RHF(m), auxbasis='cc-pvdz-ri',
                                numerics=numerics)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-12, 300
        mf.kernel()
        return mf
    mol = water()
    chain = ExcitedStateChain(mol, factory, spin='singlet', numerics=numerics)
    fz = chain.factorization
    _, M, _ = fz._fit(mol, fz.auxmol(mol), fz.coords(mol))
    assert M is chain.mf0.with_df.M

    def energy(m):
        return chain.energy(m)[0]
    e0, g0 = energy(mol), chain.total_gradient(mol)[0]
    force, torque = rigid_part(mol, g0)
    assert force < RIGID_TOL and torque < RIGID_TOL, (force, torque)
    fd, d = five_point(energy, mol, np.random.default_rng(0).normal(size=g0.shape))
    assert abs(fd - float((g0 * d).sum())) < isdf_gradient_bar()
    m2 = moved(mol)
    assert abs(energy(m2) - e0) < ENERGY_TOL
    assert np.abs(chain.total_gradient(m2)[0] - g0 @ rotation().T).max() \
        < isdf_gradient_bar()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
