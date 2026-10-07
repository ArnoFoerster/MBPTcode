"""The pole model's Eq. (27) verdict is frozen at the reference geometry.

A state takes the pole model ('sop') when every residue its contour sweeps
lies within one particle-hole gap of the Newton start (`compressible`). The
reference solve starts each Newton at eps_p, every later solve at the frozen
reference root, and the quasiparticle correction can carry a state across
that limit: on formaldehyde/cc-pVDZ, PBE0, orbitals 4, 5 and 10 have reach
0.96, 0.76 and 0.76 at eps_p and 1.35, 1.16 and 1.15 at the root. The verdict
is therefore frozen with the root and the poles (`frozen_on_pole_model`), as
the scissor tiers are, so the energy and its force use one route. Checked:

- every state's route at a second solve, at the reference geometry and at a
  displaced one, is the reference solve's;
- the second solve at the reference geometry, starting at the frozen roots,
  repeats them to the Newton tolerance;
- the rule itself: frozen poles keep the pole model past the limit, a frozen
  root without poles keeps the quadrature, nothing frozen reads Eq. (27) at
  the start.

Run: python tests/test_pole_model_route_frozen.py   (or pytest)
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import (QP_CD_NEWTON_TOL,  # noqa: E402
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import QPStates  # noqa: E402
from src.SingleReference.GW.qp_states import resolve_qp_states  # noqa: E402
from src.SingleReference.GW.sum_over_poles import compressible  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402
from src.gradients.qp_space_time import (  # noqa: E402
    frozen_on_pole_model, pole_model_route)

H2CO = 'C 0 0 -0.5296; O 0 0 0.6763; H 0 0.9339 -1.1088; H 0 -0.9339 -1.1088'
#: The states whose Eq. (27) verdict differs between eps_p and the root.
CROSSING = (4, 5, 10)
#: The displacement of the second geometry, Bohr, on the carbon along z.
STEP = 1e-3


def factory(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    return mf


def set_solve(chain, mol, mf):
    """(roots, routes) of one solve of the chain's set at `mol`."""
    x_mo, d, eps, mu = chain._factors_for(mol, mf)[:4]
    out, route_out = chain._qp_set_solve(
        x_mo, d, eps, mu, chain._xc_correction(mf, chain.qp_set),
        np.zeros(len(chain.qp_set)))
    return (np.asarray(out[0], float),
            {int(p): r for p, r in route_out['routes'].items()})


@pytest.fixture(scope='module')
def solved():
    """(chain, the reference solve's roots and routes, the eps of R0)."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
    mf = factory(mol)
    eps = np.asarray(mf.mo_energy, float)
    window = list(resolve_qp_states(QPStates(), eps, mol.nelectron // 2,
                                    degeneracy_tol=1e-6).explicit)
    chain = ExcitedStateChain(mol, factory, mf=mf, qp_window=window,
                              residue_route='sop')
    roots, routes = set_solve(chain, mol, mf)
    return chain, roots, routes, eps


def test_the_case_crosses_the_wall(solved):
    chain, _, routes, eps = solved
    nocc = chain.nocc
    for p in CROSSING:
        assert routes[p] == 'sop', (p, routes[p])
        assert compressible(eps[p], eps, nocc)[0], p
        assert not compressible(chain.qp_seeds[p], eps, nocc)[0], p


def test_the_reference_geometry_repeats_its_routes_and_roots(solved):
    chain, roots, routes, _ = solved
    again, routes_again = set_solve(chain, chain.mol0, chain.mf0)
    assert routes_again == routes
    np.testing.assert_allclose(again, roots, rtol=0, atol=QP_CD_NEWTON_TOL)


def test_a_displaced_geometry_keeps_the_reference_routes(solved):
    chain, _, routes, _ = solved
    mol = chain.mol0.copy()
    coords = mol.atom_coords()
    coords[0, 2] += STEP
    mol.set_geom_(coords, unit='Bohr')
    mol.build(False, False)
    _, routes_there = set_solve(chain, mol, factory(mol))
    assert routes_there == routes


def test_the_frozen_verdict_rule():
    eps = np.array([-1.0, -0.5, 0.2, 0.9])
    nocc = 2
    past = -1.3                  # 0.8 from eps_1 on a 0.7 gap: reach 1.14
    assert not compressible(past, eps, nocc)[0]
    assert compressible(-0.9, eps, nocc)[0]
    poles = {0: np.array([0.7, 1.4])}
    assert frozen_on_pole_model(0, {0: past}, poles) is True
    assert frozen_on_pole_model(0, {0: past}, {}) is False
    assert frozen_on_pole_model(0, None, None) is None
    # an orbital the mappings do not name has nothing frozen
    assert frozen_on_pole_model(1, {0: past}, poles) is None
    route = lambda w0, sop_poles: pole_model_route(
        'sop', None, 0, eps, nocc, w0, None, sop_poles)[0]
    assert route({0: past}, poles) == 'sop'
    assert route({0: -0.9}, {}) == 'auto'
    assert route(None, None) == 'sop'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
