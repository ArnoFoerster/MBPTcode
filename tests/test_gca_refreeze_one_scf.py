"""A composed surface's refreeze converges ONE mean field at the new geometry.

`RPABSESurface` and `RPAQPSurface` are E_0^dRPA plus an excitation on one
mean field; two SCFs at one geometry would be two solutions, and every energy
of the surface is a difference of them. The refreeze builds the new ground
half on one fresh SCF, hands that mean field to the excited half, and the
excited half carries its conventions over on the mean field the walk
evaluated on at that geometry, which the ground half already holds. Each half
converging its own, and the
excited half a third for the carry-over, would be three solutions.

H2/cc-pVDZ Hartree-Fock: the smallest system the composed surface accepts,
with a unique SCF solution, so the composed refreeze and the two halves
refrozen alone (each on its own SCF) are one surface: same energies.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.gradients.gwbse.rpa_bse_surface import RPABSESurface, RPAQPSurface
from tests.isdf_test_grid import g1_numerics

BASIS = 'cc-pvdz'
H2 = 'H 0 0 0; H 0 0 0.74'
#: Bohr the second hydrogen moves along the bond: the walk's last geometry.
STEP = 0.05


def counted_factory(calls):
    """A converged density-fitted RHF per call, each call appended to `calls`."""
    def build(mol):
        calls.append(np.array(mol.atom_coords()))
        mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged
        return mf
    return build


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom=H2, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def moved(h2):
    m = h2.copy()
    c = h2.atom_coords().copy()
    c[1, 2] += STEP
    m.set_geom_(c, unit='Bohr')
    m.build(False, False)
    return m


@pytest.mark.parametrize('surface_cls', [RPABSESurface, RPAQPSurface],
                         ids=['bse', 'qp'])
def test_a_composed_refreeze_converges_one_scf(h2, moved, surface_cls):
    calls = []
    s = surface_cls(h2, counted_factory(calls), state=0,
                    numerics=g1_numerics(h2))
    # the walk evaluated the last geometry: its mean field is held
    s.energy(moved)
    del calls[:]
    r = s.refreeze(moved)
    assert len(calls) == 1, f'{len(calls)} SCFs in one composed refreeze'
    assert np.array_equal(calls[0], moved.atom_coords())
    assert r.excited.mf0 is r.ground.mf0, 'the halves stand on two SCFs'


@pytest.mark.parametrize('surface_cls', [RPABSESurface, RPAQPSurface],
                         ids=['bse', 'qp'])
def test_the_refrozen_surface_is_the_halves_refrozen_alone(h2, moved,
                                                           surface_cls):
    """On a unique SCF solution one shared mean field changes no number: the
    composed refreeze equals each half refrozen on its own SCF."""
    calls = []
    s = surface_cls(h2, counted_factory(calls), state=0,
                    numerics=g1_numerics(h2))
    before = s.energy(moved)
    r = s.refreeze(moved)
    fac = s.ground.factorization.rebuilt_at(moved)
    alone = surface_cls(moved, s._scf, state=0, factorization=fac,
                        ground=s.ground.refreeze(moved, factorization=fac),
                        excited=s.excited.refreeze(moved, factorization=fac))
    composed, apart = r.energy(moved), alone.energy(moved)
    assert np.allclose(composed, apart, rtol=0.0, atol=1e-10), (composed,
                                                                apart)
    # the refreeze drift at the walk's last geometry is unchanged with it
    assert abs((composed[0] - before[0]) - (apart[0] - before[0])) < 1e-10


def test_a_mean_field_of_another_geometry_is_refused(h2, moved):
    """`mean_field(mol, mf)` hands `mf` through to kernels that read `mol`
    and to skeletons that read `mf.mol`: of two geometries, the force would
    be a mixture of both with the right shape."""
    s = RPABSESurface(h2, counted_factory([]), state=0, numerics=g1_numerics(h2))
    elsewhere = counted_factory([])(moved)
    with pytest.raises(ValueError, match='another geometry'):
        s.ground.mean_field(h2, elsewhere)
    with pytest.raises(ValueError, match='another geometry'):
        s.excited.mean_field(None, elsewhere)
    assert s.ground.mean_field(moved, elsewhere)[1] is elsewhere
