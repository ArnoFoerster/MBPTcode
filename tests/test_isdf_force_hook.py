"""The ISDF mean-field force reaches Base and the environments as an object,
never as an import.

`src.Base.factorization.isdf_jk.isdf_jk` builds the mean field and refuses
pySCF's own gradient by name: that gradient differentiates the fitted
interaction, not the interpolated one the energy comes from. The gradient
layer's `attach_isdf_gradient` binds the ISDF force and leaves itself on the
mean field as `force_hook`, which `SolventScreening` calls on the mean field it
wraps in a continuum; `isdf_jk_with_force` is both steps.
"""
import ast
import pathlib

import pytest
from pyscf import gto, scf

from src.Base.factorization.isdf_jk import ISDFForceNotAttached, isdf_jk
from src.gradients.isdf_mean_field import attach_isdf_gradient, isdf_jk_with_force
from tests.isdf_test_grid import g1

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                 basis='cc-pvdz', verbose=0)


def test_pyscfs_gradient_of_a_bare_isdf_mean_field_is_refused_by_name(mol):
    mf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri', counts=g1(scf.RHF(mol)))
    assert mf.force_hook is None
    for ask in (mf.Gradients, mf.nuc_grad_method):
        with pytest.raises(ISDFForceNotAttached, match='attach_isdf_gradient'):
            ask()


def test_the_attached_force_carries_its_hook(mol):
    mf = isdf_jk_with_force(scf.RHF(mol), auxbasis='cc-pvdz-ri',
                            counts=g1(scf.RHF(mol)))
    assert mf.force_hook is attach_isdf_gradient
    assert type(mf.Gradients()).__name__ == '_ISDFGradients'


def test_base_and_environments_never_import_the_gradient_layer():
    """The hook replaces the two call-time imports that pointed up."""
    for path in ('src/Base/factorization/isdf_jk.py',
                 'src/environments/solvent_screening.py'):
        tree = ast.parse((REPO / path).read_text())
        modules = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                    for a in n.names]
        assert not [m for m in modules if m.startswith('src.gradients')], path
