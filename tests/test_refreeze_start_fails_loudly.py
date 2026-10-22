"""A refrozen surface that cannot be evaluated at the previous minimum raises.

WHAT BREAKS IF THIS FAILS. A walk whose very first evaluation raises returns a
record without an energy. Inside the refreeze that record was then read
for its energy, and the run died with `KeyError: 'energy'`: the exception the
refrozen surface raised, and the traceback saying where, were thrown away. A
refrozen surface is rebuilt AT the converged geometry of the surface before
it, so failing to evaluate there is a broken surface, not a walk that went
nowhere: the original exception has to come out of `relax`. The first walk's
start that fails to evaluate is still a reported status, as before.

A relaxation that swallowed the exception would make
`test_a_refrozen_surface_that_does_not_evaluate_raises` fail for both engines
with `KeyError: 'energy'`.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.step_failed import StepFailed
from src.properties.optimize import HAVE_GEOMETRIC, relax

#: Ha/Bohr^2 and Bohr: the toy bond's stiffness and minimum.
STIFFNESS = 0.4
R_E = 2.2
#: The message the broken surface raises with, to find it again.
MESSAGE = 'the quasiparticle root of orbital 46 was not found'

ENGINES = pytest.mark.parametrize('engine', ['cartesian', 'geometric'])


class BrokenOnEvaluation(StepFailed):
    """Stands in for a lost root at the refrozen surface's first evaluation:
    the one failure a walk does not raise at its own start."""


class HarmonicBond:
    """E = k (r - r_e)^2 / 2 in the bond length; `refreeze` returns a surface
    that raises on its first evaluation when `breaks`."""

    def __init__(self, mol, breaks=True, broken=False):
        self.mol0, self.breaks, self.broken = mol, breaks, broken

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), mf

    def total_gradient(self, mol=None, mf=None):
        if self.broken:
            raise BrokenOnEvaluation(MESSAGE)
        mol = self.mol0 if mol is None else mol
        d = mol.atom_coords()[1] - mol.atom_coords()[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros((mol.natm, 3))
        grad[1] = STIFFNESS * (r - R_E) * d / r
        grad[0] = -grad[1]
        return grad, 0.5 * STIFFNESS * (r - R_E) ** 2, {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol):
        return type(self)(mol, breaks=self.breaks, broken=self.breaks)

    def label(self):
        return 'harmonic bond toy'


@pytest.fixture
def n2():
    return gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                 verbose=0)


def _relax(engine, surface, mol, **kw):
    if engine == 'geometric' and not HAVE_GEOMETRIC:
        pytest.skip('geomeTRIC is not installed')
    return relax(surface, mol, engine=engine, verbose=False, nudge=0, **kw)


@ENGINES
def test_a_refrozen_surface_that_does_not_evaluate_raises(engine, n2):
    """The refrozen surface's own exception, not a KeyError, ends the run."""
    with pytest.raises(BrokenOnEvaluation, match=MESSAGE) as caught:
        _relax(engine, HarmonicBond(n2), n2, refreeze=1)
    # the traceback reaches the frame that raised
    assert caught.traceback[-1].name == 'total_gradient'


@ENGINES
def test_a_first_walk_that_does_not_evaluate_is_reported(engine, n2):
    """A start that fails on the first surface stays a reported status."""
    surface = HarmonicBond(n2, broken=True)
    _, info = _relax(engine, surface, n2, refreeze=1)
    assert info['converged'] is False
    assert info['status'] == f'BrokenOnEvaluation: {MESSAGE}'


@ENGINES
def test_a_refrozen_surface_that_evaluates_still_relaxes(engine, n2):
    """The control: the same toy without the break measures its refreeze."""
    _, info = _relax(engine, HarmonicBond(n2, breaks=False), n2, refreeze=1)
    assert info['converged'] and info['refreeze'] == 'measured'
