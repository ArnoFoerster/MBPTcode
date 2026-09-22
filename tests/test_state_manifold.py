"""Several roots of ONE solve, and the interstate slot that needs them.

The manifold's whole claim is that its roots come off a single forward pass and
are therefore eigenvectors of a common matrix. Two things have to be true for
that to be worth anything: the shared pass must give the SAME answers as
independent surfaces (otherwise sharing broke something), and it must actually
be shared (otherwise the class is an expensive alias). Both are asserted below,
the second by counting forward passes rather than by timing, which is the only
form of that measurement that does not flake.

`ExcitedStateChain` exercises the pinned path (`bse` fixture below).

A COMPOSED surface is the third case and the one that failed silently: it keeps
a root index of its own and reads the energy off the chain in `excited`, so a
manifold driving the outer copy returned root 0 for every state -- 2.08 eV low
on water's S1, with nothing raised anywhere.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV, ISDF_GRADIENT_FLOOR
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.state_manifold import (StateManifold, driven_chain,
                                          own_root_attribute, pinned_forward,
                                          root_attribute)


def rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                 basis='cc-pvdz', verbose=0)


# ------------------------------------------------------- which attribute
def test_a_chain_with_no_root_attribute_is_refused():
    """Better than defaulting to root 0, which would silently make every state
    of the manifold the same one."""
    class Nothing:
        pass
    with pytest.raises(AttributeError, match='ROOT_ATTR'):
        root_attribute(Nothing())


def test_an_ambiguous_chain_is_refused_rather_than_ordered():
    """A chain carrying both names has two meanings and the manifold would drive
    whichever came first in the candidate list."""
    class Both:
        state, root = 0, 0
    with pytest.raises(AttributeError, match='ROOT_ATTR'):
        root_attribute(Both())


def test_a_declaration_that_does_not_exist_is_refused():
    class Lying:
        ROOT_ATTR = 'nowhere'
    with pytest.raises(AttributeError, match='nowhere'):
        root_attribute(Lying())


# ------------------------------------------------------- the shared pass
def test_the_forward_pass_is_pinned_and_then_released():
    """Counted, not timed. A chain whose `_forward` is called once per root has
    no reason to exist as a manifold."""
    class Fake:
        ROOT_ATTR = 'state'
        mol0 = 'MOL'

        def __init__(self):
            self.state, self.calls = 0, 0

        def mean_field(self, mol=None, mf=None):
            return (mol or self.mol0), mf

        def _forward(self, mol, mf):
            self.calls += 1
            return np.array([1.0, 2.0, 3.0]), {'pass': self.calls}

        def total_energy(self, mol=None, mf=None):
            return float(self._forward(mol, mf)[0][self.state])

        def total_gradient(self, mol=None, mf=None):
            om, pieces = self._forward(mol, mf)
            return np.zeros((1, 3)), float(om[self.state]), pieces

    fake = Fake()
    man = StateManifold(fake, states=(0, 1, 2))
    assert man.energies() == {0: 1.0, 1: 2.0, 2: 3.0}
    assert man.chain.calls == 1, 'one forward pass must serve every root'

    man.gradients()
    assert man.chain.calls == 2, 'one more pass for the gradients, not three'

    # released afterwards: an unpinned call recomputes
    man.chain.total_energy()
    assert man.chain.calls == 3
    assert '_forward' not in man.chain.__dict__


def test_a_pinned_pass_refuses_a_different_geometry():
    """Otherwise a displaced geometry would silently get the reference
    spectrum, which is a finite difference of exactly zero."""
    class Fake:
        ROOT_ATTR = 'state'

        def __init__(self):
            self.state = 0

        def _forward(self, mol, mf):
            return np.zeros(2), {}

    fake = Fake()
    with pinned_forward(fake, 'MOL_A', None) as pinned:
        assert pinned
        with pytest.raises(RuntimeError, match='different geometry'):
            fake._forward('MOL_B', None)


# ------------------------------------------- the real chain the plan aims at
@pytest.fixture(scope='module')
def bse(mol):
    """One mean field shared, so the manifold's saving is the forward pass and
    not an SCF it avoided."""
    from src.gradients.excited_state import ExcitedStateChain
    return ExcitedStateChain(mol, rhf, spin='singlet', state=0, mf=rhf(mol))


def test_the_bse_chains_root_attribute_is_inferred(bse):
    """`ExcitedStateChain` declares no ROOT_ATTR and carries only `state`, so
    inference resolves it. If it ever grows a second name this fails loudly
    rather than driving the wrong one."""
    assert root_attribute(bse) == 'state'


def test_bse_roots_off_one_pass_equal_independent_surfaces(bse, mol):
    """The claim, on production code: sharing the solve changes no number.

    Bitwise, not to a tolerance -- the roots come from the SAME Casida
    solution, so any difference at all would mean the pin leaked.
    """
    from src.gradients.excited_state import ExcitedStateChain
    calls, orig = {'n': 0}, type(bse)._forward

    def counted(self, m, f):
        calls['n'] += 1
        return orig(self, m, f)

    type(bse)._forward = counted
    try:
        man = StateManifold(bse, states=(0, 1, 2))
        got = man.energies()
        assert calls['n'] == 1, 'three roots must cost ONE forward pass'
    finally:
        type(bse)._forward = orig

    for n in (0, 1, 2):
        ref = ExcitedStateChain(mol, rhf, spin='singlet', state=n,
                                mf=bse.mf0).total_energy()
        assert got[n] == ref


def test_bse_gradients_off_one_pass_equal_independent_surfaces(bse, mol):
    from src.gradients.excited_state import ExcitedStateChain
    man = StateManifold(bse, states=(0, 1))
    got = man.gradients()
    for n in (0, 1):
        ref = ExcitedStateChain(mol, rhf, spin='singlet', state=n,
                                mf=bse.mf0).total_gradient()
        # E_0 + Omega: two chains' gradients, so two reproducibility floors
        assert np.abs(got[n][0] - ref[0]).max() < 2 * ISDF_GRADIENT_FLOOR
        # the seeded Newton root sits one ulp from the unseeded one
        assert abs(got[n][1] - ref[1]) < 1e-12
        assert np.abs(got[n][0].sum(axis=0)).max() < 1e-9
    assert np.abs(got[1][0] - got[0][0]).max() > 1e-3


# ------------------------------------------ a COMPOSED surface: TWO objects
@pytest.fixture(scope='module')
def composed(mol):
    """E_HF + E_c^dRPA + Omega. The root index on the outer object is a copy;
    the excitation is read off the chain in `excited`."""
    return RPABSESurface(mol, rhf, state=0, mf=rhf(mol))


@pytest.fixture(scope='module')
def root1(mol, composed):
    """An INDEPENDENT surface at root 1, sharing only the mean field, so what
    the manifold saves is the forward pass and not an SCF."""
    return RPABSESurface(mol, rhf, state=1, mf=composed.ground.mf0)


@pytest.fixture(scope='module')
def root1_reference(root1):
    """(E_1, dE_1/dR) of that independent surface: one gradient serves both."""
    g, e, _ = root1.total_gradient()
    return e, g


def test_the_driven_half_holds_the_spectrum(composed):
    """The outer index RESOLVES, which is why driving it looked right."""
    assert own_root_attribute(composed) == 'state'
    assert driven_chain(composed) is composed.excited
    assert root_attribute(driven_chain(composed)) == 'state'
    # a plain chain is its own driven half, so nothing above changes for one
    assert driven_chain(composed.excited) is composed.excited


def test_a_composed_root_is_that_root_and_not_root_zero(composed, root1_reference):
    """The defect, on the single-root entry points: `energy(n)` returned root
    0's energy for every state and raised nothing."""
    e_ref, _ = root1_reference
    man = StateManifold(composed, states=(0, 1))
    e1, e0 = man.energy(1), man.energy(0)
    assert e1 == pytest.approx(e_ref, abs=1e-12)
    assert e0 == pytest.approx(composed.total_energy(), abs=1e-12)
    assert (e1 - e0) * HARTREE_TO_EV > 0.5, 'the two roots came back as one'


def test_composed_energies_come_off_one_forward_pass(composed, root1_reference):
    """Counted on the EXCITED half, which is where the pass now has to be
    pinned; pinning the outer object would leave one pass per root."""
    e_ref, _ = root1_reference
    calls, orig = {'n': 0}, type(composed.excited)._forward

    def counted(self, m, f):
        calls['n'] += 1
        return orig(self, m, f)

    type(composed.excited)._forward = counted
    try:
        got = StateManifold(composed, states=(0, 1)).energies()
        assert calls['n'] == 1, 'two roots must cost ONE forward pass'
    finally:
        type(composed.excited)._forward = orig
    assert got[1] == pytest.approx(e_ref, abs=1e-12)
    assert got[0] < got[1]


def test_a_composed_roots_gradient_is_that_roots_own(composed, root1_reference):
    _, g_ref = root1_reference
    man = StateManifold(composed, states=(0, 1))
    g1 = man.gradient(1)[0]
    assert np.abs(g1 - g_ref).max() < 2 * ISDF_GRADIENT_FLOOR
    assert np.abs(g1 - man.gradient(0)[0]).max() > 1e-3
    assert np.abs(g1.sum(axis=0)).max() < 1e-9


def test_two_composed_surfaces_are_independent(composed):
    """A shared `excited` half would make one optimizer retune the other's
    state, which is the failure `surface` exists to prevent."""
    man = StateManifold(composed, states=(0, 1))
    s0, s1 = man.surface(0), man.surface(1)
    assert s0.excited is not s1.excited
    assert (s0.excited.state, s1.excited.state) == (0, 1)
    assert (s0.state, s1.state) == (0, 1), 'the outer copy has to agree'
    man.energy(0)
    assert (s0.excited.state, s1.excited.state) == (0, 1)
    # and both roots still share the one mean field and the one factorization
    assert s0.ground.mf0 is s1.ground.mf0
    assert s0.excited.factorization is s1.excited.factorization


def test_the_manifold_does_not_move_the_composed_surface_it_was_given(composed):
    """The caller still holds it; a driven `excited` half would move its state
    under the surface the caller is comparing against."""
    man = StateManifold(composed, states=(0, 1))
    man.energy(1)
    man.surface(1)
    assert (composed.state, composed.excited.state) == (0, 0)
