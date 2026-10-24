"""`relax(engine='auto')` hands each optimizer the keywords it takes.

WHAT BREAKS IF THIS FAILS. With geomeTRIC importable, 'auto' forwarding the
Cartesian optimizer's keywords (`trust`, `max_cycle`, `conv`) to
`optimize_geometric`, which has no **kw, kills the relaxation with a
TypeError -- the path `calc_adiabatic_excitation(**optimizer)` takes. Both
branches filter, and the first test passes.
"""
import numpy as np
import pytest
from pyscf import gto

import src.properties.optimize as optimize_module
from src.Base.numerics.numerics import Numerics
from src.properties.optimize import relax

pytest.importorskip('geometric')


class Bond:
    """E = k (r - r_e)^2 / 2 in the bond length."""

    def __init__(self, mol):
        self.mol0 = mol

    def total_gradient(self, mol=None, mf=None):
        here = mol.atom_coords()
        d = here[1] - here[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros_like(here)
        grad[1] = 0.4 * (r - 2.2) * d / r
        grad[0] = -grad[1]
        return grad, 0.2 * (r - 2.2) ** 2, {}

    def refreeze(self, mol):
        return self


@pytest.fixture
def n2():
    return gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                 verbose=0)


def test_auto_drops_the_cartesian_keywords_for_geometric(n2):
    _, info = relax(Bond(n2), n2, engine='auto', verbose=False, nudge=0,
                    numerics=Numerics(walk={'trust': 0.2, 'max_cycle': 30}),
                    conv={'opt_grad_max': 1e-4})
    assert info['optimizer'] == 'geometric' and info['converged']


def test_auto_drops_the_geometric_keywords_for_cartesian(n2, monkeypatch):
    monkeypatch.setattr(optimize_module, 'HAVE_GEOMETRIC', False)
    _, info = relax(Bond(n2), n2, engine='auto', verbose=False, nudge=0,
                    maxiter=30, coordsys='cart', numerics=Numerics(walk={'trust': 0.2}))
    assert info['optimizer'] == 'internal' and info['converged']


def test_auto_refuses_a_keyword_no_optimizer_takes(n2):
    with pytest.raises(TypeError, match='trust_radius'):
        relax(Bond(n2), n2, engine='auto', verbose=False, trust_radius=0.2)
