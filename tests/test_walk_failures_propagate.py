"""Only a `StepFailed` shortens a step; every other exception leaves the walk.

WHAT BREAKS IF THIS FAILS. Both walkers caught `Exception` around every
evaluation and read it as "the step was too long": the Cartesian walk cut its
trust radius by four until it collapsed, geomeTRIC's walk ended with the
message as its status. A NotImplementedError from a dropped quasiparticle
member, a MemoryError and a plain bug then all ended as "trust radius
collapsed", a non-convergence, and were patched as one. The failures that
ARE about the step -- the Newton losing its root, the Casida problem losing
positive-definiteness or its Davidson stopping short, the followed state
leaving the solved roots -- raise `StepFailed`, and only those are caught.

The negative control on a real chain: formaldehyde S1, HF/cc-pVDZ, whose
quasiparticle solve is made to raise NotImplementedError at the first trial
step. On the code before the fix the Cartesian walk returns 'trust radius
collapsed' instead of raising.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.step_failed import StepFailed
from src.SingleReference.LinearResponse.casida import CasidaInstability
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.qp_set import QuasiparticleRootLost
from src.gradients.gwbse.state_following import RootFollowingLost
from src.properties.optimize import (HAVE_GEOMETRIC, optimize,
                                     optimize_geometric)
from src.SingleReference.LinearResponse.davidson import (DavidsonRefusal,
                                                         CasidaBreakdown)
from src.Solvers.qp_equation import QPRootNotConverged
from tests.isdf_test_grid import g1_numerics

#: Ha/Bohr^2 and Bohr: the toy bond's stiffness and minimum; it starts 0.2
#: Bohr inside it.
STIFFNESS = 0.4
R_E = 2.2
#: Bohr: a trial step longer than this fails, a shorter one evaluates.
TOO_FAR = 0.05

ENGINES = pytest.mark.parametrize('engine', ['internal', 'geometric'])


class FailingBond:
    """E = k (r - r_e)^2 / 2; an evaluation farther than TOO_FAR from the
    last one that evaluated raises `failure`."""

    def __init__(self, mol, failure):
        self.mol0, self.failure = mol, failure
        self.last = mol.atom_coords().copy()

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), mf

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        here = mol.atom_coords()
        if np.abs(here - self.last).max() > TOO_FAR:
            raise self.failure('evaluated too far from the last geometry')
        self.last = here.copy()
        d = here[1] - here[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros((mol.natm, 3))
        grad[1] = STIFFNESS * (r - R_E) * d / r
        grad[0] = -grad[1]
        return grad, 0.5 * STIFFNESS * (r - R_E) ** 2, {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol):
        return type(self)(mol, self.failure)

    def label(self):
        return 'failing bond toy'


@pytest.fixture
def n2():
    return gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                 verbose=0)


def _walk(engine, surface, mol):
    if engine == 'geometric':
        if not HAVE_GEOMETRIC:
            pytest.skip('geomeTRIC is not installed')
        return optimize_geometric(surface, mol, verbose=False, nudge=0)
    return optimize(surface, mol, verbose=False, nudge=0, trust=0.3)


def test_the_step_failures_are_step_failed():
    """The four places that mean "shorter step" raise a `StepFailed`."""
    for kind in (RootFollowingLost, QuasiparticleRootLost, DavidsonRefusal,
                 CasidaBreakdown, QPRootNotConverged, CasidaInstability):
        assert issubclass(kind, StepFailed), kind
    # the dense instability is still the ValueError its callers caught
    assert issubclass(CasidaInstability, ValueError)


@ENGINES
def test_a_defect_in_an_evaluation_leaves_the_walk(engine, n2):
    """A NotImplementedError at a trial step propagates out of the walk."""
    with pytest.raises(NotImplementedError, match='too far'):
        _walk(engine, FailingBond(n2, NotImplementedError), n2)


def test_a_failed_step_is_retried_shorter(n2):
    """A StepFailed at a trial step shortens it, and the walk converges."""
    _, info = _walk('internal', FailingBond(n2, StepFailed), n2)
    assert info['converged'] and info['rejected'] > 0


def test_geometric_shortens_a_failed_step(n2):
    """geomeTRIC restarts shorter from the last accepted geometry on a
    StepFailed, and the walk converges."""
    _, info = _walk('geometric', FailingBond(n2, StepFailed), n2)
    assert info['converged'] and info['rejected'] > 0


def test_a_start_that_fails_with_a_defect_raises(n2):
    """A defect at the very first evaluation is raised too, not reported."""
    surface = FailingBond(n2, NotImplementedError)
    surface.last = surface.last + 1.0
    with pytest.raises(NotImplementedError):
        optimize(surface, n2, verbose=False, nudge=0)


def scf_factory(mol):
    """HF converged for gradient work."""
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def test_a_defect_in_the_quasiparticle_solve_leaves_a_real_walk():
    """The negative control on a real chain: formaldehyde S1."""
    mol = gto.M(atom='C 0 0 -0.529; O 0 0 0.6746; H 0 0.9376 -1.1188; '
                     'H 0 -0.9376 -1.1188', basis='cc-pvdz', verbose=0)
    chain = ExcitedStateChain(mol, scf_factory, spin='singlet',
                              mf=scf_factory(mol), numerics=g1_numerics(mol))
    solve = chain.quasiparticles.qp_set_solve
    first = []

    def at_the_start_only(x_mo, d_sigma, eps, *args, **kw):
        # the orbital energies tell the start from a displaced geometry
        first.append(np.array(eps))
        if not np.array_equal(eps, first[0]):
            raise NotImplementedError('a member with no pole model here')
        return solve(x_mo, d_sigma, eps, *args, **kw)

    chain.quasiparticles.qp_set_solve = at_the_start_only
    with pytest.raises(NotImplementedError, match='no pole model'):
        optimize(chain, max_cycle=5, verbose=False)
