"""Every seam that combines two products refuses a numerics mismatch.

One `Numerics` is passed once; two products of different
identities are two answers to two questions. The composed surface's halves,
the refreeze, the spin-orbit and derivative-coupling state pairs, the
adiabatic gap and a force handed to another surface each refuse
(`NumericsMismatch`) where they combine, and the products carry the identity
they were computed on. `compare_surfaces` is not among them: it compares an
oracle against production and keeps reporting.

H2/cc-pVDZ Hartree-Fock on the G1 grid.
"""
import types

import numpy as np
import pytest
from pyscf import gto

from src.Base import numerics as nm
from src.Base.numerics import Numerics
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.derivative_coupling import analytic_coupling
from src.gradients.excited_state import ExcitedStateChain, FirstPoint
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties import hessian
from src.properties.spin_orbit import spin_orbit_couplings
from src.properties.vibronic import adiabatic_gap
from tests.reference_scf import rhf_df_differentiable_checked

BASIS = 'cc-pvdz'


@pytest.fixture(scope='module')
def h2():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def g1():
    return Numerics(isdf={'counts': resolve_isdf_grid('G1', BASIS, ['H'])[0]})


@pytest.fixture(scope='module')
def other(g1):
    return g1.derive('a finer self-energy grid', quadrature={'ntau_gw': 30})


def test_the_halves_of_a_composed_surface_refuse_other_numerics(h2, g1,
                                                                other):
    ground = RPAGroundStateChain(h2, rhf_df_differentiable_checked, numerics=other)
    with pytest.raises(nm.NumericsMismatch, match='quadrature.ntau_gw'):
        RPABSESurface(h2, rhf_df_differentiable_checked, numerics=g1, ground=ground)
    assert RPABSESurface(h2, rhf_df_differentiable_checked, numerics=g1)


def test_a_refreeze_onto_other_numerics_is_refused(h2, g1, other,
                                                   monkeypatch):
    chain = ExcitedStateChain(h2, rhf_df_differentiable_checked, numerics=g1)
    original = ExcitedStateChain._refrozen

    def drifting(self, mol, factorization, mf, walked, qp_floor):
        out = original(self, mol, factorization, mf, walked, qp_floor)
        out.__dict__['settings'] = other
        return out
    monkeypatch.setattr(ExcitedStateChain, '_refrozen', drifting)
    with pytest.raises(nm.NumericsMismatch, match='refreeze'):
        chain.refreeze(h2)


def test_a_spin_orbit_pair_of_other_numerics_is_refused():
    one = nm.Stamped((np.zeros(1), None, None), 'a' * 64)
    two = nm.Stamped((np.zeros(1), None, None), 'b' * 64)
    with pytest.raises(nm.NumericsMismatch, match='spin-orbit'):
        spin_orbit_couplings(None, None, 1, one, two)


def test_a_coupling_off_an_evaluation_of_other_numerics_is_refused(h2, g1):
    chain = ExcitedStateChain(h2, rhf_df_differentiable_checked, numerics=g1)
    evaluation = types.SimpleNamespace(numerics_identity='c' * 64)
    with pytest.raises(nm.NumericsMismatch, match='derivative coupling'):
        analytic_coupling(chain, 0, 1, evaluation=evaluation)


def test_an_adiabatic_gap_of_other_numerics_is_refused():
    one = {'e_total': -1.0, 'numerics_identity': 'a' * 64}
    two = {'e_total': -0.9, 'numerics_identity': 'b' * 64}
    with pytest.raises(nm.NumericsMismatch, match='adiabatic gap'):
        adiabatic_gap(one, two)
    assert adiabatic_gap(one, dict(two, numerics_identity='a' * 64)) \
        == pytest.approx(-0.1)
    assert adiabatic_gap({'e_total': -1.0}, two) == pytest.approx(-0.1)


def test_a_force_of_other_numerics_is_not_handed_over(h2, g1):
    surface = RPABSESurface(h2, rhf_df_differentiable_checked, numerics=g1)
    force = (np.zeros((2, 3)), -1.0, {'numerics_identity': 'd' * 64})
    point = FirstPoint(np.array(h2.atom_coords()), np.array(h2.atom_charges()),
                       'singlet', 0, force)
    with pytest.raises(nm.NumericsMismatch, match='first point'):
        surface.first_point = point
    stamped = (np.zeros((2, 3)), -1.0,
               {'numerics_identity': nm.numerics_identity(surface)})
    surface.first_point = FirstPoint(point.coords, point.charges, 'singlet',
                                     0, stamped)


def test_a_force_names_the_numerics_it_was_taken_on(h2, g1):
    chain = ExcitedStateChain(h2, rhf_df_differentiable_checked, numerics=g1)
    _, _, diags = chain.total_gradient(h2)
    assert diags['numerics_identity'] == g1.identity()


def test_the_hessian_stamp_names_the_numerics(g1):
    assert hessian.fd_hessian_stamp() == {'pair_layout': 'reference',
                                  'frames': 'reference'}
    assert hessian.fd_hessian_stamp(g1)['numerics'] == g1.identity()
