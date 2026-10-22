"""The density-fitted BSE@G0W0 chain at large basis sets: the memory-lean
builds give what the dense ones gave.

Gated: the Eq. (18) shift of Duchemin et al., Chem. Sci. 9, 4430 (2018), built
from two slices of the three-centre integrals over auxiliary-shell blocks, is
the dense (P|mn) build; the DF block action's (A, B) is the dense Casida pair
on the same factor and W, singlet and triplet; the imaginary-frequency route's
pole strengths ride along without moving a root.
"""
import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import dft, gto

from src.Base.environment import attach_environment
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW import reaction_field as rf
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.LinearResponse.block_actions import df_block_action
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.pins import assert_same

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS, XC = 'cc-pvdz', 'cc-pvdz-jkfit', 'pbe0'
#: Ha. Two associations of one contraction.
ROUNDOFF = 1e-11


@pytest.fixture(scope='module')
def solvated_water():
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    mf = dft.RKS(mol, xc=XC).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    attach_environment(mf, SolventScreening(mol, solvent='toluene'))
    return mol, mf, mol.nelectron // 2


def dense_shift(mf, mol, nocc):
    """Eq. (18) through the whole fitted (P|pq), the reference for the lean
    build."""
    env = mf.with_screening
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    V = auxmol.intor('int2c2e', aosym='s1')
    t = rf.bare_gauge_transform(auxmol, env, V=V)
    mo, naux = mf.mo_coeff, V.shape[0]
    three = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    fit = np.linalg.solve(V, three.reshape(-1, naux).T).reshape(naux, *mo.shape[:1] * 2)
    coeff = np.einsum('PQ,Qmn,mp,nq->Ppq', aux_metric_sqrt(auxmol, env, V=V),
                      fit, mo, mo, optimize=True)
    w = np.asarray(LinearResponseSolver(mf.mo_energy, coeff_df=coeff,
                                        spin_mode='restricted')
                   .static_screening_aux(nocc))
    return rf.congruent_quasiparticle_shift(np.einsum('Qpp->Qp', coeff), w, t,
                                            nocc)


def test_lean_reaction_field_shift_is_the_dense_one(solvated_water):
    mol, mf, nocc = solvated_water
    lean = rf.environment_quasiparticle_shift(mf, mol, nocc)
    assert np.abs(lean).max() > 1e-3
    np.testing.assert_allclose(lean, dense_shift(mf, mol, nocc), atol=ROUNDOFF)


def test_three_centre_blocks_do_not_move_the_slices(solvated_water):
    """One auxiliary shell per block against all of them in one."""
    mol, mf, nocc = solvated_water
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    occ, virt = np.arange(nocc), np.arange(nocc, mf.mo_coeff.shape[1])
    sets = [(mf.mo_coeff, occ, virt)]
    whole = rf._three_center_ov_and_diagonal(mol, auxmol, sets)[0]
    shells = rf._three_center_ov_and_diagonal(mol, auxmol, sets,
                                              tile_gb=1e-12)[0]
    for a, b in zip(whole, shells):
        np.testing.assert_allclose(a, b, atol=ROUNDOFF)


@pytest.mark.parametrize('tile_gb', [None, 1e-12])
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_df_block_action_is_the_dense_casida_pair(solvated_water, spin, tile_gb):
    """The swapped exchange contracted through (naux, nocc, nocc) is B, and
    W C_vv built a virtual at a time is the whole product."""
    mol, mf, nocc = solvated_water
    coeff = get_density_fitting_coefficients(mol, mf)
    lr = LinearResponseSolver(get_orbital_energies(mf, representation='spatial'),
                              coeff_df=coeff, spin_mode='restricted')
    w = lr.static_screening_aux(nocc)
    a, b = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w,
                                    triplet=spin == 'triplet')
    tile = {} if tile_gb is None else {'tile_memory_gb': tile_gb}
    apply_ab, diag = df_block_action(lr, nocc, True, w, spin=spin, **tile)
    z = np.random.default_rng(7).standard_normal((3,) + diag.shape)
    az, bz = apply_ab(z)
    flat = z.reshape(3, -1)
    np.testing.assert_allclose(az.reshape(3, -1), flat @ a.T, atol=ROUNDOFF)
    np.testing.assert_allclose(bz.reshape(3, -1), flat @ b.T, atol=ROUNDOFF)


def test_imaginary_axis_pole_strengths_leave_the_roots(solvated_water):
    mol, mf, nocc = solvated_water
    states = np.arange(nocc - 2, nocc + 2)
    roots = solve_qp_energy_imaginary_axis(mf, mol, nocc, states)
    with_z, z = solve_qp_energy_imaginary_axis(mf, mol, nocc, states,
                                               with_z=True)
    # two solves through the continuation: the same roots to a few ulp
    assert_same(with_z, roots, quantity='continued', label='with_z')
    assert np.all((z > 0.5) & (z < 1.0))
