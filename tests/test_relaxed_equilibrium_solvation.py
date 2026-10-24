"""Equilibrium solvation of a relaxed excited state: the excitation's response
at eps_static (`SolventScreening(equilibrium=True)`, `equilibrium_partner`).

WHAT EACH GATE IS FOR:

- eps_static = eps_inf makes the equilibrium object the non-equilibrium one,
  bit for bit, in the energy AND the force: the switch moves one constant and
  nothing else;
- the equilibrium excitation gradient is the derivative of its own energy
  (water in water, eps_static = 78.4 in the response): the force needs no new
  code because every adjoint is the continuum's own.

The partner's own gates, the switch through a `SurfaceSpec` and the
charge-transfer shift on ethylene/F2 are tests/test_equilibrium_solvation.py's.
"""
import numpy as np
import pytest
from pyscf import gto

from src.environments.solvent_screening import SolventScreening
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.Base.numerics.numerics import numerics_of
from tests.blas_bar import assert_same
from tests.isdf_test_grid import g1
from tests.reference_scf import rhf_tight

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


@pytest.mark.parametrize('spin', ('singlet', 'triplet'))
def test_equal_constants_are_the_non_equilibrium_object(water, spin):
    eps = 1.7764
    kw = dict(eps=eps, eps_static=eps)
    out = []
    for equilibrium in (False, True):
        env = SolventScreening(water, equilibrium=equilibrium, **kw)
        chain = ExcitedStateChain(water, rhf_tight, environment=env, spin=spin, numerics=numerics_of({'solver': 'dense', 'counts': g1(water)}))
        out.append(chain.excitation_gradient(water)[0])
        out.append(np.atleast_1d(chain.excitation(water)))
    # The same object: bitwise on the recording BLAS (two chains, two SCF
    # solves); on another BLAS the gradient and the excitation energy each hold
    # to their class bar.
    for a, b, quantity in zip(out[:2], out[2:], ('gradient', 'energy')):
        assert_same(a, b, quantity=quantity, label=quantity)


@pytest.mark.parametrize('spin', ('singlet', 'triplet'))
def test_equilibrium_gradient_is_the_derivative_of_its_energy(water, spin):
    """Water in water with the response at eps_static, 5-point stencil."""
    env = SolventScreening(water, solvent='water').equilibrium_partner()
    chain = ExcitedStateChain(water, rhf_tight, environment=env, spin=spin, numerics=numerics_of({'solver': 'dense', 'counts': g1(water)}))
    analytic, _ = chain.excitation_gradient(water)
    h = 1e-4
    fd = np.zeros((water.natm, 3))
    for ia in range(water.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                m = water.copy()
                d = np.zeros((water.natm, 3))
                d[ia, x] = k * h
                m.set_geom_(water.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                v.append(chain.excitation(m))
            fd[ia, x] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)
    assert np.abs(analytic - fd).max() / np.abs(analytic).max() < 1e-6
