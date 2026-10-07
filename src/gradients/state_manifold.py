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

SPINS AND ROOTS OFF ONE EVALUATION. On a chain whose forward pass splits at
the Casida step (`ExcitedStateChain._shared_forward`, alone or as the excited
half of `RPABSESurface`) a state is a (spin, root) target, and `evaluate`
serves several of them at one geometry:

    man = StateManifold(chain, states=(('singlet', 0), ('triplet', 0)))
    ev = man.evaluate(gradients=(('singlet', 0), ('triplet', 0)))
    ev.energy[t], ev.gradient[t], ev.info[t]     # per target, total
    ev.g0                                        # the ground-state force, once
    ev.spectrum['triplet']                       # (omega, X, Y)
    man.surface(('triplet', 0), first_point=ev)  # a PES whose first force is ev's

The mean field once, the shared forward once (the factors, the static W, the
reaction field, the quasiparticle set with its tape), one Casida solve per
spin, one reverse pass per force target, all reading the tape the forward
built, and the ground-state force once. Every number is the one the chain of
that spin and root returns evaluated alone, bit for bit. An integer state
keeps meaning a root of the chain's own spin.
"""
import contextlib
import copy

import numpy as np

from src.Base.constants import KAPPA, NUCLEAR_FD_STEP
from src.gradients.excited_state import FirstPoint
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


class StateEvaluation:
    """What `StateManifold.evaluate` computed at one geometry, by the
    manifold's state keys: `energy` (E_0 + Omega, Hartree), `omega`, `root`
    (the root the state is at here), `gradient` and `info` for the force
    targets, `interstate` {(m, n): (d<m|H|n>/dR, diagnostics)}, `spectrum`
    {spin: (omega, X, Y)}, `g0` the ground-state force (None without a force
    target) and `target` {key: (spin, root)}."""

    def __init__(self, mol, mf):
        self.mol, self.mf = mol, mf
        self.energy, self.omega, self.root, self.target = {}, {}, {}, {}
        self.gradient, self.info, self.interstate = {}, {}, {}
        self.spectrum = {}
        self.g0 = None

    def first_point(self, key):
        """The `FirstPoint` of force target `key`: this geometry, its spin
        and root, and its (dE/dR, E, diagnostics)."""
        if key not in self.gradient:
            raise ValueError(f'no force was evaluated for {key!r}; the force '
                             f'targets were {tuple(self.gradient)}')
        spin, root = self.target[key]
        return FirstPoint(np.array(self.mol.atom_coords()),
                          np.array(self.mol.atom_charges()), spin, root,
                          (self.gradient[key], self.energy[key],
                           self.info[key]))


class StateManifold:
    """A root set of one chain, and the interstate slot that needs it."""

    def __init__(self, chain, states=(0,)):
        """`states` are root indices into the chain's own ordering, ascending,
        or (spin, root) targets in the order given (`evaluate`).

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
        states = list(states)
        if all(isinstance(n, (int, np.integer)) for n in states):
            self.states = tuple(sorted({int(n) for n in states}))
        else:
            self.states = tuple(dict.fromkeys(self._spin_root(n)
                                              for n in states))
        if not self.states:
            raise ValueError('a manifold needs at least one state')
        lowest = min(self._root_of(n) for n in self.states)
        if lowest < 0:
            raise ValueError(f'root indices must be non-negative, got '
                             f'{lowest}')
        # the per-spin Casida solvers and the per-target views, taken off the
        # driven chain once its first forward has frozen the conventions
        self._solvers, self._views = {}, {}
        self._require_reachable()

    @property
    def shares_forward(self):
        """Whether the driven chain's forward splits at the Casida step and
        the surface composes a total `evaluate` knows how to assemble."""
        return (hasattr(self.driven, '_shared_forward')
                and hasattr(self.driven, 'spin_view')
                and (self.driven is self.chain
                     or hasattr(self.chain, 'composed')))

    def _spin_root(self, n):
        """(spin, root) of a target given as one, refused by name otherwise."""
        if not self.shares_forward:
            raise ValueError(
                f'a (spin, root) state needs a chain whose forward pass is '
                f'shared below the Casida step; {type(self.driven).__name__} '
                f'takes root indices')
        spin, root = n
        if spin not in KAPPA:
            raise ValueError(f'spin {spin!r} not in {tuple(KAPPA)}')
        return (str(spin), int(root))

    @staticmethod
    def _root_of(n):
        return n[1] if isinstance(n, tuple) else int(n)

    def _target(self, n):
        """(spin, root) of a state key: an integer is a root of the chain's
        own spin."""
        return n if isinstance(n, tuple) else (self.driven.spin, int(n))

    def _require_reachable(self):
        """A Davidson chain that solves fewer roots than were asked for would
        raise from inside the first gradient, after the forward pass was paid."""
        nroots = getattr(self.driven, 'nroots', None)
        solver = getattr(self.driven, 'solver', None)
        top = max(self._root_of(n) for n in self.states)
        if nroots is not None and solver != 'dense' and nroots <= top:
            raise ValueError(
                f'states up to {top} were asked for but the chain '
                f'solves nroots={nroots}; raise nroots to at least '
                f'{top + 1}')

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

    def surface(self, n, first_point=None):
        """State `n` as an INDEPENDENT `PotentialEnergySurface`.

        A separate copy, spectrum half included, so that holding two of them
        and driving one does not move the other -- an optimizer relaxing S1
        must not retune the T1 surface it is being compared against.

        A (spin, root) state, or any state with `first_point`, is a copy of
        its target's view, which needs the conventions frozen by an
        evaluation first. first_point: a `StateEvaluation` holding this
        state's force; the surface's first `total_gradient` at that geometry
        is that force (`FirstPoint`), so a walk started there does not
        evaluate it again.
        """
        if n not in self.states:
            raise ValueError(f'root {n} is not in this manifold: {self.states}')
        if not isinstance(n, tuple) and first_point is None:
            out = copy_for_root(self.chain)
            select_root(out, n)
            return out
        out = self._spin_surface(n)
        if first_point is not None:
            out.first_point = first_point.first_point(n)
        return out

    def _spin_surface(self, n):
        """An independent copy of the target view of `n`, inside a copy of
        the composed surface where there is one."""
        spin, root = self._target(n)
        view = self._views.get((spin, root))
        half = (self.driven.spin_view(spin, state=root) if view is None
                else copy.copy(view))
        half.follow_log = list(half.follow_log)
        half.davidson_solves = list(half.davidson_solves)
        if self.driven is self.chain:
            return half
        out = copy.copy(self.chain)
        out.excited = half
        outer = own_root_attribute(out)
        if outer is not None:
            setattr(out, outer, root)
        if hasattr(out, 'spin'):
            out.spin = spin
        return out

    def energy(self, n, mol=None, mf=None):
        """Total energy of state `n` in Hartree, at `mol` or at the reference."""
        if isinstance(n, tuple):
            if n not in self.states:
                raise ValueError(f'root {n} is not in this manifold: '
                                 f'{self.states}')
            return self.evaluate(mol, mf, states=(n,)).energy[n]
        return self._at(n).total_energy(mol, mf)

    def gradient(self, n, mol=None, mf=None):
        """(dE_n/dR, E_n, diagnostics) for state `n`."""
        if isinstance(n, tuple):
            if n not in self.states:
                raise ValueError(f'root {n} is not in this manifold: '
                                 f'{self.states}')
            ev = self.evaluate(mol, mf, states=(n,), gradients=(n,))
            return ev.gradient[n], ev.energy[n], ev.info[n]
        return self._at(n).total_gradient(mol, mf)

    def energies(self, mol=None, mf=None):
        """{n: E_n} for every state, off ONE forward pass."""
        if self.shares_forward:
            return dict(self.evaluate(mol, mf).energy)
        mol, mf = self._mean_field(mol, mf)
        with pinned_forward(self.driven, mol, mf):
            return {n: self._at(n).total_energy(mol, mf) for n in self.states}

    def gradients(self, mol=None, mf=None):
        """{n: (dE_n/dR, E_n, diagnostics)} for every state, off ONE forward
        pass.

        The reverse pass is per state and genuinely different for each; only
        the forward half is shared, which is the expensive half.
        """
        if self.shares_forward:
            ev = self.evaluate(mol, mf, gradients=self.states)
            return {n: (ev.gradient[n], ev.energy[n], ev.info[n])
                    for n in self.states}
        mol, mf = self._mean_field(mol, mf)
        with pinned_forward(self.driven, mol, mf):
            return {n: self._at(n).total_gradient(mol, mf) for n in self.states}

    def gaps(self, mol=None, mf=None):
        """{(m, n): E_n - E_m} over the states, in Hartree, off ONE forward
        pass, m before n in the manifold's order.

        VERTICAL differences at one geometry. An ADIABATIC gap is a difference
        of two RELAXED minima and belongs to `src.properties.vibronic`, which
        takes the two surfaces this manifold hands out.
        """
        e = self.energies(mol, mf)
        return {(m, n): e[n] - e[m]
                for i, m in enumerate(self.states)
                for n in self.states[i + 1:]}

    # ------------------------------------------------ one shared evaluation
    def evaluate(self, mol=None, mf=None, states=None, gradients=(),
                 couplings=()):
        """A `StateEvaluation` of several states at one geometry.

        states: the keys whose energies are wanted (every state of the
        manifold by default); gradients: the force targets; couplings: (m, n)
        pairs of one spin whose interstate numerator d<m|H|n>/dR is wanted
        (`ExcitedStateChain.interstate_gradient`). Every rank runs it whole,
        in the same order: the mean field, the composed surface's ground-state
        force, the shared forward, one Casida solve per spin in the order the
        spins first appear, the force targets in the order given, then the
        couplings. The quasiparticle tape is held from the forward to the
        last reverse pass and released when this returns.

        Each spin's Casida solve reads the highest root asked of it: that
        decides which roots may refuse an unconverged residual and not the
        iterations, so every root is the one a chain of that spin and root
        returns alone. The mean field's force rides the first force target's
        fit-adjoint call (`one_fit_adjoint(mean_field=True)`), as it rides the
        single force in `ExcitedStateChain.total_gradient`.
        """
        if not self.shares_forward:
            raise TypeError(
                f'{type(self.driven).__name__} has no forward pass shared '
                'below the Casida step; take its roots one at a time')
        keys = self.states if states is None else tuple(states)
        gradients = tuple(gradients)
        couplings = tuple(tuple(c) for c in couplings)
        wanted = tuple(dict.fromkeys(
            keys + gradients + tuple(k for c in couplings for k in c)))
        for k in wanted:
            if k not in self.states:
                raise ValueError(f'root {k} is not in this manifold: '
                                 f'{self.states}')
        for m, n in couplings:
            if m == n or self._target(m)[0] != self._target(n)[0]:
                raise ValueError(
                    f'an interstate element is taken between two roots of ONE '
                    f'Casida solve; got {m!r} and {n!r}')
        mol, mf = self._mean_field(mol, mf)
        if gradients or couplings:
            self.driven.require_differentiable_environment()
        composed = self.driven is not self.chain
        ev = StateEvaluation(mol, mf)
        ground = None
        if composed and gradients:
            ground = self.chain.ground.total_gradient(mol, mf)
        shared = self.driven._shared_forward(mol, mf)
        try:
            solved = self._solve_spins(shared, wanted, ev)
            # An adaptive explicit set is checked on these eigenvectors at the
            # reference geometry; one that grows is solved and checked again,
            # and the spin views taken on the old set are dropped.
            verify = getattr(self.driven, 'verify_selection', None)
            while verify is not None and verify(shared, solved):
                shared.release()
                self._solvers.clear()
                self._views.clear()
                shared = self.driven._shared_forward(mol, mf)
                solved = self._solve_spins(shared, wanted, ev)
            for k in wanted:
                ev.target[k] = self._target(k)
                om, pieces = solved[ev.target[k][0]]
                root = self._view(k).tracked_state(mol, mf, om, pieces)
                ev.root[k], ev.omega[k] = int(root), float(om[root])
            if not composed:
                e_0 = mf.e_tot
            elif ground is not None:
                e_0 = ground[1]
            else:
                e_0 = self.chain.ground.energy(mol, mf)[0]
            for k in wanted:
                ev.energy[k] = e_0 + ev.omega[k]
            for k in gradients:
                om, pieces = solved[ev.target[k][0]]
                total = self._force(k, mf, om, pieces, ev, ground)
                ev.gradient[k], ev.energy[k], ev.info[k] = total
            if ground is not None:
                ev.g0 = ground[0]
            for m, n in couplings:
                spin = ev.target[m][0]
                om, pieces = solved[spin]
                ev.interstate[(m, n)] = self._solvers[spin]._interstate_gradient(
                    pieces, om, ev.root[m], ev.root[n], release_tape=False)
        finally:
            shared.release()
        return ev

    def _solve_spins(self, shared, wanted, ev):
        """{spin: (omega, pieces)}: one Casida solve per spin on `shared`."""
        solved = {}
        for spin in dict.fromkeys(self._target(k)[0] for k in wanted):
            top = max(self._target(k)[1] for k in wanted
                      if self._target(k)[0] == spin)
            solver = self._solvers.get(spin)
            if solver is None:
                solver = self._solvers[spin] = self.driven.spin_view(spin)
            solver.state, solver.timer = top, self.driven.timer
            om, pieces = solver._casida_forward(shared)
            solved[spin] = (om, pieces)
            ev.spectrum[spin] = (om, pieces[10], pieces[11])
        return solved

    def _view(self, k):
        """The target view of state `k`: its own root-following history, the
        Davidson record of its spin's solver."""
        spin, root = self._target(k)
        view = self._views.get((spin, root))
        if view is None:
            view = self._views[(spin, root)] = self.driven.spin_view(
                spin, state=root)
            view.davidson_solves = self._solvers[spin].davidson_solves
        view.timer = self.driven.timer
        return view

    def _force(self, k, mf, om, pieces, ev, ground):
        """(dE_k/dR, E_k, diagnostics) of one force target off the shared
        pieces; the mean field's force is computed with the first and kept in
        `ev.g0`."""
        view = self._view(k)
        root = ev.root[k]
        if ground is not None:
            return self.chain.composed(
                ground, view._root_gradient(pieces, om, root,
                                            release_tape=False))
        if ev.g0 is None:
            with view.one_fit_adjoint(mf, mean_field=True):
                g_om, diags = view._root_gradient(pieces, om, root,
                                                  release_tape=False)
                ev.g0 = view.mean_field_gradient(mf)
        else:
            g_om, diags = view._root_gradient(pieces, om, root,
                                              release_tape=False)
        return view.composed_total(mf, ev.g0, g_om, diags)

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
