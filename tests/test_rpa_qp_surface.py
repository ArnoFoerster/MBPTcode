"""The SIGN of the quasiparticle energy on the composed G0W0 surface.

E^{N-1} = E_0 - eps_h and E^{N+1} = E_0 + eps_l: the electron count's sign,
reversed. A surface that adds eps for every state reports a NEGATIVE ionization
potential -- eps_HOMO is about -0.5 Ha -- and, worse, relaxes a cation on the
wrong sign of the quasiparticle force, so it converges to something that looks
like a minimum. The dense `QuasiparticleSurface` carries this as `charge_change`
and is the convention the cubic surface has to agree with.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.gradients.dense_surfaces import QuasiparticleSurface
from src.gradients.rpa_bse_surface import RPAQPSurface

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


def hf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def surface(water):
    """One mean field and one factorization for every check below."""
    return RPAQPSurface(water, hf_factory, state=0, mf=hf_factory(water))


def test_removing_an_electron_costs_a_positive_ionization_potential(surface):
    """E^{N-1} - E_0 = -eps_HOMO > 0, which is the whole content of the sign."""
    surface.state = 0
    e, e_0, qp = surface.energy()
    assert e - e_0 == pytest.approx(-qp, abs=1e-12)
    assert e - e_0 > 0.0, f'ionization potential {e - e_0:.4f} Ha is not positive'
    assert qp < 0.0, 'eps_HOMO must be negative for the sign to matter'


def test_adding_an_electron_enters_with_the_opposite_sign(surface):
    """E^{N+1} - E_0 = +eps_LUMO: the offset's occupancy decides, not the class."""
    surface.state = +1
    e, e_0, qp = surface.energy()
    assert e - e_0 == pytest.approx(+qp, abs=1e-12)
    surface.state = 0


def test_the_correlated_gradient_flips_sign_with_the_occupancy(surface):
    """The force must answer the energy it belongs to.

    The composed gradient is the dRPA ground state's plus the quasiparticle's,
    and the second enters with the same sign as in the energy -- otherwise the
    surface is stationary for neither functional and an optimizer converges
    silently on the wrong geometry.
    """
    surface.state = 0
    g_total = surface.total_gradient()[0]
    g_ground = surface.ground.total_gradient()[0]
    g_qp = surface.excited.quasiparticle_gradient(0)[0]
    assert np.allclose(g_total - g_ground, -g_qp, atol=1e-12, rtol=0.0), \
        f'worst {np.abs(g_total - g_ground + g_qp).max():.2e} Ha/Bohr'
    # and it is not the sum, which is what the surface returned before
    assert np.abs(g_qp).max() > 1e-4, 'the two signs would be indistinguishable'


def test_the_charge_change_follows_the_dense_convention(water, surface):
    """`QuasiparticleSurface` is the reference convention; an adiabatic IP is a
    difference of two minima and only means something if both routes agree on
    which direction the electron went."""
    nocc = water.nelectron // 2
    for state, charge_change in ((0, -1), (+1, +1)):
        surface.state = state
        dense = QuasiparticleSurface(water, hf_factory,
                                     charge_change=charge_change,
                                     orbital=nocc - 1 + state)
        assert surface.charge_change == dense.charge_change
        assert surface.sign == dense.sign
    surface.state = 0
    assert 'E^(N-1)' in surface.label()
