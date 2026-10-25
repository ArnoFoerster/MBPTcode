"""The auxiliary representation of the PCM reaction field, measured and gated.

Every route reads vtilde between pair densities as c^T vtilde_PQ c, c the
Coulomb fit; vtilde_PQ is the exact cavity contraction, so the error is the
fitted density's potential on the cavity. `fit_residual` measures it against
the exact grid potential of the pair densities, and `require_fit` refuses a
solvated run whose auxiliary set misses the basis's diffuse tail.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import df, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import SOLVENT_RI_RESIDUAL_MAX
from src.environments.solvent_screening import SolventScreening

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def orbitals(mol):
    """Gas-phase RHF orbitals and the doubly occupied count: the residual is
    a property of the pair densities and the cavity, not of the SCF."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.mo_coeff, mol.nelectron // 2


def test_the_residual_is_the_error_of_the_ri_reaction_field():
    """fit_residual's two potentials ARE the two vtilde's: G Q G^T is
    kernel_mo's ov block, G_fit Q G_fit^T is c^T aux_kernel c."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mo, nocc = orbitals(mol)
    env = SolventScreening(mol, solvent='toluene')
    auxmol = df.addons.make_auxmol(mol, 'cc-pvdz-ri')
    o, v = mo[:, :nocc], mo[:, nocc:]
    n = nocc * v.shape[1]
    exact = env.kernel_mo(mol, mo)[:nocc, nocc:, :nocc, nocc:].reshape(n, n)
    int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1')
    p_ia = np.einsum('mnP,mi,na->iaP', int3c, o, v).reshape(n, -1)
    c = np.linalg.solve(auxmol.intor('int2c2e'), p_ia.T)
    ri = c.T @ env.aux_kernel(auxmol) @ c
    residual = env.fit_residual(mol, auxmol, o, v)
    element = np.linalg.norm(ri - exact) / np.linalg.norm(exact)
    # vtilde is quadratic in the potential: its error is about twice the residual
    assert 0.05 < residual < 0.07
    assert 1.0 < element / residual < 2.0


def test_the_residual_converges_with_the_auxiliary_set():
    """The diffuse tail is what is missing: aug-cc-pvdz-ri cuts the residual
    five-fold and an even-tempered set further, both under the bar."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mo, nocc = orbitals(mol)
    env = SolventScreening(mol, solvent='toluene')
    got = [env.fit_residual(mol, df.addons.make_auxmol(mol, aux),
                            mo[:, :nocc], mo[:, nocc:])
           for aux in ('cc-pvdz-ri', 'aug-cc-pvdz-ri',
                       df.aug_etb(mol, beta=1.6))]
    assert got[0] > 4 * got[1]
    assert got[1] < 0.015
    assert got[0] < SOLVENT_RI_RESIDUAL_MAX


def test_the_basis_own_ri_set_passes():
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mo, nocc = orbitals(mol)
    env = SolventScreening(mol, solvent='toluene')
    residual = env.require_fit(mol, df.addons.make_auxmol(mol, 'cc-pvdz-ri'),
                               mo, nocc)
    assert residual < SOLVENT_RI_RESIDUAL_MAX


def test_a_set_without_the_basis_diffuse_functions_is_refused():
    """cc-pvdz-ri under aug-cc-pVDZ: by name, and -- where the names cannot
    be compared -- by the measured residual (0.17 on water)."""
    mol = gto.M(atom=H2O, basis='aug-cc-pvdz', verbose=0)
    mo, nocc = orbitals(mol)
    env = SolventScreening(mol, solvent='toluene')
    with pytest.raises(ValueError, match='auxiliary partner'):
        env.require_fit(mol, df.addons.make_auxmol(mol, 'cc-pvdz-ri'),
                        mo, nocc)
    per_element = gto.M(atom=H2O, basis={'O': 'aug-cc-pvdz',
                                         'H': 'aug-cc-pvdz'}, verbose=0)
    with pytest.raises(ValueError, match='SOLVENT_RI_RESIDUAL_MAX'):
        env.require_fit(per_element,
                        df.addons.make_auxmol(per_element, 'cc-pvdz-ri'),
                        mo, nocc)
