"""What a property routine is allowed to know about the electronic structure
underneath it.

A geometry optimization, a Huang-Rhys spectrum, an adiabatic singlet-triplet
gap or a Marcus rate needs the same four things from a method: the TOTAL energy
of one state as a function of the nuclei, its derivative, a way to rebuild
whatever the method froze at a reference geometry, and a name to print. Nothing
in that list mentions BSE@GW, a factorization, or even the existence of an
analytic gradient, which is why the protocol lives here rather than next to any
one chain.

TOTAL energy, never an excitation energy. Two states relax to different minima,
so E_0 does not cancel in an adiabatic gap and cannot be left out of the
surface; a method that reports Omega alone is not a surface.

REFREEZE BELONGS TO THE METHOD. Every cubic chain fixes discrete choices at a
reference geometry -- the quasiparticle set, the frame orientation, the
interpolation pair layout, the Newton branch, the residue backend -- because
each is a discontinuity in the surface otherwise. An optimizer that walks far
enough leaves the geometry those choices were made for, and the only honest
answer is to rebuild them at the new geometry and measure how far the minimum
moves. Only the method knows what it froze, so `refreeze` is its method and the
optimizer merely calls it.

ONE EVALUATION, HOWEVER MANY RANKS. Inside `with distributed(comm):` every rank
runs the property routine and the surface whole, and `evaluate` is where a
property routine hands a geometry to the surface: it locksteps the geometry on
the way in and the energy, force and diagnostics on the way out, so every rank
evaluates rank 0's geometry and every rank's next decision -- a step, a trust
radius, a convergence test -- is taken from rank 0's numbers.
"""
from abc import ABC, abstractmethod

import numpy as np

from src.Base.numerics.constants import NUCLEAR_FD_STEP
from src.Base.geometry.geometry import displaced
from src.Base.utils.parallel.mpi_grid import lockstep


# ------------------------------------------------ the protocol vocabulary
# Protocol vocabulary a surface is driven through: which attribute selects its
# root, which half of a composed surface owns the spectrum, which root it
# followed to a geometry, and the walk's acceptance of a geometry.
#
# Below the property layer, because both a manifold in `src.gradients` and the
# property routines in `src.properties` drive a surface through it; a copy on
# either side would be two answers to which half owns the spectrum.

def root_attribute(surface):
    """The name of the attribute that selects `surface`'s root: its class's
    declared `ROOT_ATTR`.

    Declared, never guessed: the chains spell it differently -- `state` on the
    BSE chains, `root` on the downfolded ones -- and a fragment-diabatic
    surface's `state` names a diabat, not a root. A class that declares none
    (None, or no declaration) has no root to drive, and a declaration the
    instance does not carry is refused.
    """
    declared = getattr(type(surface), 'ROOT_ATTR', None)
    if declared is None:
        raise AttributeError(
            f'{type(surface).__name__} declares no root attribute, so the '
            f'manifold cannot tell which selects the state; declare '
            f'ROOT_ATTR = \'<name>\' on the class')
    if not hasattr(surface, declared):
        raise AttributeError(
            f'{type(surface).__name__}.ROOT_ATTR names {declared!r}, which '
            f'the instance does not have')
    return declared


def own_root_attribute(surface):
    """`root_attribute` where the class declares one, None where it does not.

    A composed surface may carry a root index of its own or none at all, and a
    caller has to set the one that exists without refusing the one that does not.
    """
    try:
        return root_attribute(surface)
    except AttributeError:
        return None


def driven_chain(surface):
    """The half of `surface` whose root attribute selects the state: the
    attribute its class declares as `DRIVEN_HALF`, else the surface itself.

    A COMPOSED surface holds its ground state in `ground` and its state in
    `excited` (`DRIVEN_HALF = 'excited'`). Driving the outer index alone moves
    nothing on the BSE surface, whose excitation is read from
    `excited.state`: every root then comes back as root 0, which on water is
    2.08 eV low and raises nothing anywhere. The half that owns the spectrum
    is the one to drive.
    """
    half = getattr(type(surface), 'DRIVEN_HALF', None)
    return surface if half is None else getattr(surface, half)


def followed_root(surface, mol):
    """The root `surface` followed at `mol`: the newest root-following entry
    evaluated nearest to `mol`, the constructor's `state` with none.

    An optimizer evaluates the steps it rejects too, so the LAST entry after a
    trust radius collapsed, or a geomeTRIC step it took back, is the root of a
    geometry the walk did not stay at. The walk returns a geometry it
    evaluated (geomeTRIC's up to the Angstrom round trip), so the nearest
    entry is that geometry's own. A surface that keeps no coordinates beside
    its log (`follow_coords`) is read at its last entry.
    """
    log = getattr(surface, 'follow_log', None) or []
    if not log:
        return int(surface.state)
    at = getattr(surface, 'follow_coords', None) or []
    if len(at) != len(log):
        return int(log[-1]['index'])
    here = np.asarray(mol.atom_coords(), float)
    far = [float(np.abs(np.asarray(c) - here).max()) for c in at]
    # the newest of the nearest: an entry evaluated twice at one geometry
    # keeps the later one
    k = len(far) - 1 - int(np.argmin(far[::-1]))
    return int(log[k]['index'])


def commit(surface, mol):
    """Tell the state-following half of `surface` that a walk accepted `mol`
    (`ExcitedStateChain.commit`); a surface that follows nothing ignores it."""
    half = driven_chain(getattr(surface, 'surface', surface))
    accept = getattr(half, 'commit', None)
    if accept is not None:
        accept(mol)


#: What a property may ask of a surface beyond `PotentialEnergySurface`'s
#: members (total_energy, total_gradient, refreeze, label, mol0), by name. A
#: method that provides an ingredient is interchangeable with any other that
#: does (GW-BSE, ADC); one that lacks it is refused by name at the call
#: that needs it (`require`), never by a type check deeper down.
INGREDIENTS = {
    'mean_field': 'mean_field(mol, mf) -> (mol, mf), the mean field its energy '
                  'is evaluated on',
    'energy': 'energy(mol, mf) -> (E_0 + Omega, E_0, Omega) of an excited '
              'state, off one evaluation',
    'excitation': 'excitation(mol, mf) -> Omega, an excited state\'s '
                  'excitation energy',
    'excited_states': 'excited_states(mol, mf) -> (omega, X, Y, nocc), every '
                      'excited state solved at a geometry',
    'state_specific_correction': 'a classical state-specific term '
                                 '(scheme B) at a geometry',
    'to_record': 'to_record() -> the record a checkpoint restores it from',
    'physics': 'physics, its declared `SurfacePhysics`',
    'conventions': 'conventions, the frozen conventions of a cubic chain',
    'qp_set': 'qp_set, the explicitly solved quasiparticle set of a dense '
              'surface',
    'phase': 'phase(name), the stage timer of a chain',
    'shared_forward': 'shared_forward(mol, mf), a forward shared below the '
                      'Casida step',
    'spin_view': 'spin_view(spin, state), the same chain for another spin '
                 'or root',
    'composed': 'composed(ground, excited), the total of a composed surface',
    'solver_used': 'solver_used(n_ov), the eigensolver a chain resolves',
    'spin': 'spin, the spin of the states a chain solves',
}


class MissingIngredient(TypeError):
    """A property asked a surface for an ingredient (`INGREDIENTS`) its method
    does not provide."""


def provides(surface, ingredient):
    """Whether `surface` provides `ingredient`, one of `INGREDIENTS`; a name
    outside the vocabulary is a programming error and refused."""
    if ingredient not in INGREDIENTS:
        raise ValueError(f'{ingredient!r} is not a surface ingredient '
                         f'({", ".join(sorted(INGREDIENTS))})')
    declared = getattr(type(surface), 'INGREDIENTS', None)
    if declared is not None:
        # a surface class declares what it provides; an instance still has to
        # hold it (an attribute set by one constructor branch only)
        return ingredient in declared and hasattr(surface, ingredient)
    return hasattr(surface, ingredient)


def require(surface, ingredient, where):
    """`surface`'s `ingredient`, or `MissingIngredient` naming the surface's
    class, the ingredient and what `where` needed it for."""
    if not provides(surface, ingredient):
        raise MissingIngredient(
            f'{type(surface).__name__} provides no {ingredient} '
            f'({INGREDIENTS[ingredient]}), which {where} needs')
    return getattr(surface, ingredient)


class PotentialEnergySurface(ABC):
    """One state's total energy over the nuclei, and the conventions behind it.

    `mol0` is the reference geometry the frozen conventions were chosen at.

    `mean_field(mol) -> (mol, mf)` IS THE ACCESSOR for the mean field a surface
    evaluates its energy on, and the only one a property routine may use --
    through `surface_mean_field` below, which is what `vibronic` and
    `FiniteDifferenceGradient` call. When the surface stands in an environment
    that mean field is the environment's ground state: PCM at the static
    dielectric constant for a continuum, the classical charges in h_core for a
    point-charge field. Every surface that a property routine asks for its
    mean field owes this accessor, a gas-phase one returning its factory's
    mean field; there is no fallback to `scf_factory`. It is not a member of
    the structural check below, which a walk-only toy satisfies.

    Two class attributes declare how a manifold or a coupling drives the
    surface (`src.gradients.surface`): `ROOT_ATTR`, the attribute
    that selects its root (absent or None: it has none), and `DRIVEN_HALF`,
    the attribute holding the half that owns the spectrum on a composed
    surface (absent or None: the surface itself).

    `scf_factory(mol) -> mf` IS NOT PART OF THIS PROTOCOL. It is the raw user
    function each surface keeps so that `refreeze` and a displaced geometry can
    rebuild themselves, and a solvated surface's own mean field is NOT that one
    -- the dRPA ground state of water/cc-pVDZ B3LYP in PCM(1.78/78.39) sits
    0.2385 eV away from it. For a gradient the factory must converge the ORBITAL
    gradient to ~1e-11, since the Lagrangian assumes the occupied-virtual Fock
    block vanishes.

    THE BASE CLASS EVERY SURFACE INHERITS: `total_energy`,
    `total_gradient`, `refreeze`, `label` and `mean_field` are abstract, so a
    surface that lacks one fails when it is created, not at the call deep
    inside a walk that needs it. `evaluate` is the one evaluation a property
    routine hands a geometry to (`evaluate` below), inherited.

    A class that mixes parts in lists `PotentialEnergySurface` LAST among its bases, so the
    methods its parts define come before the abstract ones in its MRO.
    """

    #: The ingredients (`surface.INGREDIENTS`) a class provides,
    #: declared on the class: `provides` reads this. `mean_field` is one of the
    #: five members, so every surface has it.
    INGREDIENTS = frozenset({'mean_field'})

    @abstractmethod
    def total_energy(self, mol=None, mf=None):
        """Total energy of THIS state in Hartree, at `mol` or at the reference."""

    @abstractmethod
    def total_gradient(self, mol=None, mf=None):
        """(dE/dR of shape (natm, 3), E, diagnostics), everything in Hartree."""

    @abstractmethod
    def refreeze(self, mol):
        """The same surface with every frozen convention rebuilt at `mol`."""

    @abstractmethod
    def label(self):
        """A short name for the state and the method, for logs and records."""

    @abstractmethod
    def mean_field(self, mol=None, mf=None):
        """(mol, mf): the mean field the energy is evaluated on."""

    def evaluate(self, mol, mf=None):
        """(dE/dR, E, diagnostics) at `mol`, rank 0's on every rank."""
        return evaluate(self, mol, mf)


class FiniteDifferenceGradient(PotentialEnergySurface):
    """A gradient for an ENERGY-ONLY surface, by central differences.

    The solvated chain is the case this exists for: its cavity moves with the
    atoms, its reverse pass does not carry that motion, and returning the
    gas-phase force would be wrong by the entire reaction-field response while
    looking perfectly reasonable. Differencing the energy is exact instead --
    every displaced geometry rebuilds its own cavity -- and it is the reference
    an analytic term has to reproduce.

    IT COSTS 6 natm ENERGIES PER GRADIENT (one displaced pair per Cartesian
    component) plus one at the reference point, each with its own SCF. That is
    a few hundred BSE@GW evaluations for a geometry optimization of anything
    past a handful of atoms, so it is a validation route and a last resort, not
    a way to run production.

    `h` is a Bohr displacement. The two-point difference has an O(h^2)
    truncation error and an O(eps/h) noise floor from the energy's own
    convergence, so a mean field converged to 1e-14 supports 1e-3 comfortably
    and a loose one does not.

    `map_fn` takes `map`'s signature and runs the 6 natm displaced energies:
    the default evaluates them in order in this process; a process pool or
    `functools.partial(mpi_grid.mpi_map, comm=comm)` spreads them over ranks,
    since no displaced energy depends on another. The wrapped surface must
    not itself distribute over the same communicator (see `mpi_map`).
    """

    #: The ingredients this surface provides (`surface.INGREDIENTS`).
    INGREDIENTS = frozenset({'mean_field'})

    def __init__(self, surface, h=NUCLEAR_FD_STEP, map_fn=None):
        self.surface, self.h, self.map_fn = surface, h, map_fn

    @property
    def mol0(self):
        return self.surface.mol0

    def scf_factory(self, mol):
        return self.surface.scf_factory(mol)

    def mean_field(self, mol=None, mf=None):
        """(mol, mf) from the WRAPPED surface, never from this wrapper's factory.

        The solvated chain is the case: its mean field is the PCM one, and a
        record built from `scf_factory` would carry the bare one under a
        solvated energy.
        """
        mol = self.surface.mol0 if mol is None else mol
        return mol, (surface_mean_field(self.surface, mol) if mf is None else mf)

    def total_energy(self, mol=None, mf=None):
        """The wrapped surface's energy, unchanged."""
        return self.surface.total_energy(mol, mf)

    def total_gradient(self, mol=None, mf=None):
        """(dE/dR, E, diagnostics) with dE/dR differenced, not differentiated.

        EVERY DISPLACED ENERGY IS LEFT TO BUILD ITS OWN MEAN FIELD. Handing it
        `scf_factory`'s would difference a gas-phase surface under a solvated
        reference energy -- 0.24 eV apart on water in PCM -- which is not a
        derivative of anything.
        """
        mol = self.surface.mol0 if mol is None else mol
        e0 = self.surface.total_energy(mol, mf)
        crd = np.asarray(mol.atom_coords())

        def displaced_energy(job):
            ia, x, sign = job
            d = np.zeros((mol.natm, 3))
            d[ia, x] = sign * self.h
            return self.surface.total_energy(displaced(mol, d, crd))

        # The 6 natm displaced energies are independent; the pair of one
        # component is differenced afterwards, in the same arithmetic as before.
        jobs = [(ia, x, sign) for ia in range(mol.natm) for x in range(3)
                for sign in (-1.0, 1.0)]
        runner = map if self.map_fn is None else self.map_fn
        energy = dict(zip(jobs, runner(displaced_energy, jobs)))
        grad = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            for x in range(3):
                grad[ia, x] = ((energy[(ia, x, 1.0)] - energy[(ia, x, -1.0)])
                               / (2.0 * self.h))
        return grad, e0, {'fd_step': self.h}

    def refreeze(self, mol):
        """Refreeze the wrapped surface and keep differencing it."""
        return FiniteDifferenceGradient(self.surface.refreeze(mol), self.h,
                                        map_fn=self.map_fn)

    def label(self):
        return f'{self.surface.label()} [finite difference, h = {self.h} Bohr]'


def surface_mean_field(surface, mol):
    """The mean field `surface` itself evaluates at `mol`, through `mean_field`.

    THE PROTOCOL'S ACCESSOR IS `mean_field(mol) -> (mol, mf)`; `scf_factory` is
    the raw user factory and is not part of the protocol. A surface standing in
    an environment builds its mean field THROUGH it -- PCM at the static
    dielectric for a continuum, the classical charges in h_core for a point-
    charge field -- so the factory's is a different surface. Measured on
    water/cc-pVDZ B3LYP in PCM(1.78/78.39), the dRPA ground state is
    -76.31502422 Ha on its own mean field and -76.30626038 Ha on the factory's,
    0.2385 eV apart; a record, an off-diagonal energy or a reorganization
    energy built from the second is not on the surface the optimizer relaxed.
    """
    return require(surface, 'mean_field', 'surface_mean_field')(mol)[1]


def lockstep_geometry(mol):
    """`mol` at rank 0's nuclear coordinates, written into it where this rank's
    differ; `mol` is returned.

    The geometry is the one input of a surface evaluation that no kernel
    locksteps: the SCF, the collocation and every integral derivative read the
    nuclei straight off the Mole, so a rank whose optimizer arithmetic moved a
    coordinate by one ulp would evaluate a different molecule and add its
    partials to everyone's reductions -- measured on water/cc-pVDZ with the
    SCF converged over two ranks, the BSE@GW force then moves 8.7e-9 Ha/Bohr
    on every rank, rank 0 included. Written in Bohr, so the repaired Mole's
    `unit` is 'Bohr' and its coordinates are rank 0's to the bit. Serially,
    and on every rank that already agrees, the Mole is not touched.
    """
    coords = mol.atom_coords()
    locked = lockstep(coords.copy())
    if not np.array_equal(locked, coords):
        mol.set_geom_(locked, unit='Bohr')
    return mol


def evaluate(surface, mol, mf=None):
    """(dE/dR, E, diagnostics) of `surface` at `mol`: rank 0's on every rank.

    THE BOUNDARY BETWEEN A PROPERTY ROUTINE AND THE KERNELS. The geometry is
    locked to rank 0's on the way in (`lockstep_geometry`) and the result on
    the way out: the kernels hand every rank the same bits, but the pieces a
    surface adds on its own -- the mean-field force, a dense eigensolve, a sum
    of two chains -- carry each node's last bits, and an optimizer stepping on
    those would take a different step on every node. One `lockstep` of the
    result makes the walk that follows one walk. Serially both are no-ops and
    this is `surface.total_gradient(mol, mf)`.

    `mf` is handed through untouched: a chain locks a given mean field to rank
    0's orbitals itself (`FactorChain.mean_field`), and whatever a surface
    forms from one it did not lock is covered by the lockstep of the result.
    """
    lockstep_geometry(mol)
    return lockstep(surface.total_gradient(mol, mf))
