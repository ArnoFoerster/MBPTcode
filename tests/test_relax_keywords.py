"""`relax(engine='auto')` hands the walk's settings to whichever optimizer runs.

WHAT BREAKS IF THIS FAILS. With geomeTRIC importable, 'auto' dropped
`numerics` and `conv`, so a walk capped by `Numerics.walk.max_cycle` ran on
geomeTRIC's own 100 steps and its own thresholds; without it, `maxiter` and
`coordsys` were dropped. Now both engines take the walk's settings, the
Cartesian fallback drops only geomeTRIC's coordinate system and file place,
and any other keyword the engine does not take is refused by name.
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


def test_auto_hands_geometric_the_walk_numerics_and_thresholds(n2):
    _, info = relax(Bond(n2), n2, engine='auto', verbose=False, nudge=0,
                    numerics=Numerics(walk={'trust': 0.2, 'max_cycle': 30}),
                    conv={'opt_grad_max': 1e-7})
    assert info['optimizer'] == 'geometric' and info['converged']
    # geomeTRIC's own GAU set stops at 4.5e-4 Ha/Bohr
    assert info['opt_grad_max'] < 1e-7
    _, info = relax(Bond(n2), n2, engine='auto', verbose=False, nudge=0,
                    numerics=Numerics(walk={'max_cycle': 1}))
    assert info['optimizer'] == 'geometric' and not info['converged']
    assert info['status'] == 'maxiter (1) reached without convergence'


def test_auto_on_cartesian_drops_only_geometric_placement(n2, monkeypatch):
    monkeypatch.setattr(optimize_module, 'HAVE_GEOMETRIC', False)
    _, info = relax(Bond(n2), n2, engine='auto', verbose=False, nudge=0,
                    coordsys='cart', workdir='unused',
                    numerics=Numerics(walk={'trust': 0.2, 'max_cycle': 30}))
    assert info['optimizer'] == 'internal' and info['converged']
    with pytest.raises(TypeError, match=r"optimize takes no \['converge'\]"):
        relax(Bond(n2), n2, engine='auto', verbose=False, converge='GAU')


@pytest.mark.parametrize('geometric', [True, False])
def test_auto_refuses_a_keyword_the_engine_does_not_take(n2, monkeypatch,
                                                         geometric):
    monkeypatch.setattr(optimize_module, 'HAVE_GEOMETRIC', geometric)
    for unknown in ('trust_radius', 'maxiter'):
        with pytest.raises(TypeError, match=unknown):
            relax(Bond(n2), n2, engine='auto', verbose=False,
                  **{unknown: 0.2})
