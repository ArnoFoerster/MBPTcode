"""Several states of ONE electronic structure: the root set the interstate
quantities need.

A `PotentialEnergySurface` is one state. A non-adiabatic coupling, a spin-orbit
matrix element or a singlet-triplet gap is a property of TWO, and building them
from two independently constructed surfaces is wrong twice over.

It is wasteful, because the expensive half does not depend on which root is
selected: an excited-state chain's forward pass computes the quasiparticle
set, the screened interaction and the whole Casida spectrum, and only the
index into that spectrum changes with the state. Two surfaces pay for two
quasiparticle sets to keep one.

It is also unsound, because nothing makes the two agree. Each surface resolves
its own quasiparticle set, its own frozen frames and its own interpolation
layout; two vectors taken from two such solutions are not eigenvectors of a
common Hamiltonian, so their overlap carries a numerical difference that reads
as physics. An interstate matrix element is only defined between roots of ONE
solve.

`StateManifold` is that one solve. It pins a single forward pass and evaluates
every requested root against it, so the roots are eigenvectors of the same
matrix and the cost of n of them is one forward pass plus n reverse passes
rather than n of each.

    man = StateManifold(chain, states=(0, 1, 2))
    man.energies()                  # {n: E_n}, one forward pass
    man.gradient(1)                 # (dE_1/dR, E_1, diagnostics)
    man.surface(1)                  # root 1 as a PotentialEnergySurface
    man.coupling(0, 1)              # <0| d/dR |1>, (natm, 3), by eigenvector overlap

WHICH ATTRIBUTE NAMES THE ROOT. The chains spell it differently -- `state` on
the BSE chains, `root` on the downfolded ones -- so a manifold declares it as
`ROOT_ATTR` on the chain class. A chain that declares nothing is accepted only
when exactly one of the two names is present, and refused by name otherwise
rather than guessed at.

AND WHICH OBJECT CARRIES IT. A composed surface -- E_HF + E_c^dRPA + Omega --
keeps a root index of its own and reads the energy off the chain in `excited`.
Setting the outer copy moves nothing there, so the manifold drives the half
that holds the spectrum, pins its forward pass, and takes the mean field from
the composed surface, which owns the environment both halves are evaluated in.

The per-root view is the chain itself with that one attribute set, so it is a
surface by construction and needs no adapter: whatever the surface protocol
guarantees for the chain holds for every root of its manifold.
"""
import contextlib
import copy

import numpy as np

from src.Base.constants import NUCLEAR_FD_STEP
from src.properties import nonadiabatic
from src.properties.surface import (driven_chain, own_root_attribute,
                                    root_attribute)

#: Sentinel for "this instance had no `_forward` of its own before pinning".
_UNSET = object()


def select_root(surface, n):
    """Set root `n` on `surface` and on the half that holds its spectrum.

    BOTH, because the two composed surfaces read the index from different
    places -- the BSE one from `excited.state`, the quasiparticle one from its
    own, where it also fixes the sign of eps^QP -- and an object whose two
    copies disagree reports one state's energy under another's label.
    """
    half = driven_chain(surface)
    setattr(half, root_attribute(half), n)
    if half is not surface:
        outer = own_root_attribute(surface)
        if outer is not None:
            setattr(surface, outer, n)


def copy_for_root(surface):
    """A shallow copy of `surface` whose spectrum half is copied too.

    Shallow, so every root shares the mean field, the grids and every frozen
    convention -- that sharing is what makes the roots eigenvectors of one
    matrix. But the root lives on the `excited` half of a composed surface, so
    copying the outer object alone leaves two nominally independent surfaces
    driving ONE chain, and relaxing either retunes the other.
    """
    out = copy.copy(surface)
    half = driven_chain(surface)
    if half is not surface:
        out.excited = copy.copy(half)
    return out


@contextlib.contextmanager
def pinned_forward(chain, mol, mf):
    """Evaluate one `_forward(mol, mf)` and hold it under every root inside.

    Legitimate because the forward pass does not depend on the selected root:
    a chain's `_casida` returns the whole spectrum and reads the root only to
    check that a Davidson run reaches it. That check is made here instead,
    once, since the pinned pass would otherwise carry the guard of whichever
    root ran first.

    Yields False, having done nothing, for a chain with no `_forward` -- the
    downfolded chains rebuild their model per root -- so the caller falls back
    to a pass each and the manifold stays correct either way.
    """
    if not hasattr(chain, '_forward'):
        yield False
        return
    om, pieces = chain._forward(mol, mf)
    held = chain.__dict__.get('_forward', _UNSET)

    def replay(m, f=None):
        # A pinned pass belongs to ONE geometry; anything else is a caller bug
        # that would otherwise return the reference geometry's spectrum.
        if m is not mol:
            raise RuntimeError('the pinned forward pass belongs to a different '
                               'geometry than the one asked for')
        return om, pieces

    chain._forward = replay
    try:
        yield True
    finally:
        if held is _UNSET:
            del chain._forward
        else:
            chain._forward = held


class StateManifold:
    """A root set of one chain, and the interstate slot that needs it."""

    def __init__(self, chain, states=(0,)):
        """`states` are root indices into the chain's own ordering, ascending.

        The chain is COPIED, shallowly, and so is the half that holds its
        spectrum: the manifold drives a root attribute and must not move the
        objects the caller still holds. The copies share the mean field, the
        grids and every frozen convention, which is the point -- all roots must
        sit on the same ones.
        """
        self.chain = copy_for_root(chain)
        self.driven = driven_chain(self.chain)
        self.root_attr = root_attribute(self.driven)
        states = (range(int(states)) if isinstance(states, (int, np.integer))
                  else states)
        self.states = tuple(sorted({int(n) for n in states}))
        if not self.states:
            raise ValueError('a manifold needs at least one state')
        if self.states[0] < 0:
            raise ValueError(f'root indices must be non-negative, got '
                             f'{self.states[0]}')
        self._require_reachable()

    def _require_reachable(self):
        """A Davidson chain that solves fewer roots than were asked for would
        raise from inside the first gradient, after the forward pass was paid."""
        nroots = getattr(self.driven, 'nroots', None)
        solver = getattr(self.driven, 'solver', None)
        if nroots is not None and solver != 'dense' and nroots <= self.states[-1]:
            raise ValueError(
                f'states up to {self.states[-1]} were asked for but the chain '
                f'solves nroots={nroots}; raise nroots to at least '
                f'{self.states[-1] + 1}')

    @property
    def mol0(self):
        return self.chain.mol0

    def scf_factory(self, mol):
        return self.chain.scf_factory(mol)

    def _mean_field(self, mol=None, mf=None):
        """(mol, mf) through the surface's own environment.

        The composed surface answers this itself and its ground-state half is
        the fallback; `scf_factory` is not an answer -- it is the raw factory
        and carries no environment, so the shared pass would be pinned on a
        mean field none of the roots are evaluated on.
        """
        owner = self.chain
        if not hasattr(owner, 'mean_field'):
            owner = owner.ground
        return owner.mean_field(mol, mf)

    def _at(self, n):
        """The chain with root `n` selected. Mutates the manifold's own copy."""
        if n not in self.states:
            raise ValueError(f'root {n} is not in this manifold: {self.states}')
        select_root(self.chain, n)
        return self.chain

    def surface(self, n):
        """Root `n` as an INDEPENDENT `PotentialEnergySurface`.

        A separate copy, spectrum half included, so that holding two of them
        and driving one does not move the other -- an optimizer relaxing S1
        must not retune the T1 surface it is being compared against.
        """
        if n not in self.states:
            raise ValueError(f'root {n} is not in this manifold: {self.states}')
        out = copy_for_root(self.chain)
        select_root(out, n)
        return out

    def energy(self, n, mol=None, mf=None):
        """Total energy of root `n` in Hartree, at `mol` or at the reference."""
        return self._at(n).total_energy(mol, mf)

    def gradient(self, n, mol=None, mf=None):
        """(dE_n/dR, E_n, diagnostics) for root `n`."""
        return self._at(n).total_gradient(mol, mf)

    def energies(self, mol=None, mf=None):
        """{n: E_n} for every root, off ONE forward pass."""
        mol, mf = self._mean_field(mol, mf)
        with pinned_forward(self.driven, mol, mf):
            return {n: self._at(n).total_energy(mol, mf) for n in self.states}

    def gradients(self, mol=None, mf=None):
        """{n: (dE_n/dR, E_n, diagnostics)} for every root, off ONE forward pass.

        The reverse pass is per root and genuinely different for each; only the
        forward half is shared, which is the expensive half.
        """
        mol, mf = self._mean_field(mol, mf)
        with pinned_forward(self.driven, mol, mf):
            return {n: self._at(n).total_gradient(mol, mf) for n in self.states}

    def gaps(self, mol=None, mf=None):
        """{(m, n): E_n - E_m} over the roots, in Hartree, off ONE forward pass.

        VERTICAL differences at one geometry. An ADIABATIC gap is a difference
        of two RELAXED minima and belongs to `src.properties.vibronic`, which
        takes the two surfaces this manifold hands out.
        """
        e = self.energies(mol, mf)
        return {(m, n): e[n] - e[m]
                for m in self.states for n in self.states if m < n}

    def coupling(self, m, n, mol=None, step=NUCLEAR_FD_STEP):
        """<Psi_m | d/dR Psi_n>, the derivative coupling between roots `m` and
        `n`, as an (natm, 3) array in 1/Bohr.

        The slot the manifold exists to make well posed: both roots come off
        one solve, so their vectors are eigenvectors of a common matrix and an
        off-diagonal element between them means something. It is evaluated by
        `src.properties.nonadiabatic.derivative_couplings`, the overlap of the
        reference eigenvectors with those at displaced geometries, with root
        tracking across the displacement. The spin-orbit element is not this:
        it goes through `src.properties.spin_orbit`, which takes two manifolds.
        """
        for k in (m, n):
            if k not in self.states:
                raise ValueError(f'root {k} is not in this manifold: {self.states}')
        nac = nonadiabatic.derivative_couplings(self, mol=mol, states=(m, n),
                                               step=step)
        i, j = (1, 2) if m < n else (2, 1)
        return nac[i, j] if m != n else nac[1, 1]

    def refreeze(self, mol):
        """The whole manifold with the chain's frozen conventions rebuilt at `mol`.

        One refreeze for all roots, so they keep sharing the quadratures and the
        window -- refreezing each surface separately is how a manifold stops
        being one solve.
        """
        return type(self)(self.chain.refreeze(mol), states=self.states)

    def label(self):
        """The method and the root set, for a log line."""
        base = self.chain.label()
        return f'{base} | manifold over roots {list(self.states)}'
