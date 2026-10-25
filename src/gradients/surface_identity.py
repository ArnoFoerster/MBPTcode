"""What a force is a force of: ONE answer for every place that asks.

A force handed from one surface to another (`FirstPoint`), a force kept on
disk for a finite-difference Hessian (`ForceCache`) and a walk's checkpoint
all have to say which surface the force belongs to, and two forces of one
geometry belong to the same surface exactly when everything the surface froze
or was asked for agrees:

    conventions  the frozen discrete choices keyed by orbital identity
                 (`FrozenConventions`), where the chain holds them
    grid         the factorization: settings, radii, placed points, frames,
                 pair layout (`FrozenFactorization.to_record`, its wall times
                 dropped, which are no convention)
    settings     the physics and the chain's records (`settings_record`), its
                 `Numerics` as the identity hash (layout and provenance are
                 not a choice the force depends on)
    environment  its label (`environment_label`)
    mean field   the reference's class, functional and J/K route (density
                 fitting's auxiliary basis, or ISDF-K's grid by content)
    scf layout   a digest of the ISDF-K pair layout every SCF of the walk is
                 fitted on (`FactorChain.scf_layout`), None without one
    scf frames   the atomic frames every SCF's ISDF-K points are placed on
                 (`FactorChain.scf_frames`, `FrozenFrames.to_record`), None
                 without an ISDF-K fit
    span         a digest of the canonical span every SCF is solved in
                 (`FactorChain.span_basis`), None where nothing is cut

per half of a composed surface. Every surface answers it itself
(`PotentialEnergySurface.identity`): a factor chain from `chain_identity`, a
composed surface from its halves (`composed_identity`), a surface that freezes
nothing None. The walk's history -- which root was followed where -- is not in
it: it says how the surface was walked, not which surface it is.
"""
from dataclasses import dataclass

import numpy as np

from src.environments.environment import environment_label
from src.Base.numerics.numerics import Numerics
from src.Base.utils.record_io import record_hash

#: What a factorization record holds that is not a convention.
NOT_A_CONVENTION = ('seconds',)


def force_identity(surface):
    """What a force of `surface` is a force of, as a record
    (`record_io.record_hash` makes it one string): the surface's own answer
    (`PotentialEnergySurface.identity`), None for a surface that freezes
    nothing."""
    return surface.identity()


def chain_identity(chain, conventions=None):
    """The identity of one factor chain (module docstring): its class, grid,
    conventions (the chain's `FrozenConventions`, None for a chain that holds
    none), settings, environment, mean field, SCF layout, SCF frames and
    span."""
    factorization = chain.factorization
    return {
        'class': class_name(chain),
        'grid': None if factorization is None else grid_record(factorization),
        'conventions': (None if conventions is None
                        else conventions.to_record()),
        'settings': numerics_view(chain.settings_record()),
        'environment': environment_label(chain.environment),
        'mean_field': mean_field_identity(chain.mf0),
        'scf_layout': digest(chain.scf_layout()),
        'scf_frames': frames_record(chain.scf_frames),
        'span': digest(chain.span_basis)}


def composed_identity(surface, **halves):
    """The identity of a surface composed of halves: its class and each
    half's own (`identity`), by the half's name."""
    return dict({'class': class_name(surface)},
                **{name: half.identity() for name, half in halves.items()})


def numerics_view(record):
    """`record` with every `Numerics` in it replaced by its identity (the
    choices a result depends on, the layout and the provenance left out), so a
    force names the numerics it was computed on by one hash."""
    if isinstance(record, Numerics):
        return {'numerics': record.identity()}
    if isinstance(record, dict):
        return {k: numerics_view(v) for k, v in record.items()}
    return record


def frames_record(frames):
    """The frozen SCF frames' record, None without them."""
    return None if frames is None else frames.to_record()


def mean_field_identity(mf):
    """The reference mean field's class, functional and J/K route: the
    density fit's auxiliary basis, or an ISDF-K grid by its counts and the
    digest of its radii; None without a mean field."""
    if mf is None:
        return None
    with_df = getattr(mf, 'with_df', None)
    radii = (None if with_df is None else
             getattr(with_df, 'grid_radii', None)
             or getattr(with_df, 'radii', None))
    return {'class': type(mf).__name__, 'xc': getattr(mf, 'xc', None),
            'jk': None if with_df is None else type(with_df).__name__,
            'auxbasis': (None if with_df is None
                         else str(getattr(with_df, 'auxbasis', None))),
            'counts': (None if with_df is None
                       else getattr(with_df, 'counts', None)),
            'radii': digest(radii)}


def digest(value):
    """sha256 of a record value (`record_hash`), None for None."""
    return None if value is None else record_hash(value)


def grid_record(factorization):
    """The factorization's record without what is no convention; refused for
    a factorization that writes none, whose forces could not say which grid
    they stand on (a force cache of them: cache_dir=None)."""
    if not hasattr(factorization, 'to_record'):
        raise ValueError(
            f'{type(factorization).__name__} writes no record, so a force '
            'on it cannot say which conventions it was computed on; keep no '
            'cache of its forces (cache_dir=None)')
    return {k: v for k, v in factorization.to_record().items()
            if k not in NOT_A_CONVENTION}


def class_name(obj):
    return f'{type(obj).__module__}:{type(obj).__qualname__}'


@dataclass(eq=False)
class FirstPoint:
    """(dE/dR, E, diagnostics) of one state already evaluated at one
    geometry, handed to the surface that walks from there: its first
    `total_gradient` at that geometry, for that spin and root, on the same
    frozen conventions, is this one (`replayed_first_point`), so a T1
    relaxation started where the S1 force was taken does not evaluate the T1
    force there again. `conventions`: the hash of the conventions the force
    stands on (`conventions_hash`), stamped by the surface it is handed to
    (`HandedOverPoint`) unless its maker stamped it."""

    coords: np.ndarray
    charges: np.ndarray
    spin: str
    state: int
    result: tuple
    conventions: str = None

    def mismatch(self, mol, spin, state, conventions):
        """What differs from the point asked for: 'geometry', 'spin',
        'root', 'conventions'; empty when this is it."""
        out = []
        if not (np.array_equal(mol.atom_charges(), self.charges)
                and np.array_equal(mol.atom_coords(), self.coords)):
            out.append('geometry')
        if spin != self.spin:
            out.append('spin')
        if int(state) != self.state:
            out.append('root')
        if conventions != self.conventions:
            out.append('conventions')
        return out
