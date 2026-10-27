"""One evaluation per geometry: what a surface computed at the last geometries
it was asked about, keyed by the nuclei themselves.

An optimizer, the record of its minimum and the vibronic analysis after it all
ask one surface about the same geometry. An SCF is free to land on another
solution, so an SCF and a forward pass per asking would give the walk's energy
and the record's as two numbers from separate SCFs at one geometry. A surface
that keeps what it computed at a geometry, keyed by the bits of its
coordinates and charges, hands every caller there the same mean field, the
same roots and the same force.
"""
import collections
import hashlib
from dataclasses import dataclass, field

import numpy as np

from src.Base.numerics.constants import EVALUATION_CACHE_SIZE


@dataclass(eq=False)
class Evaluation:
    """What a chain computed at one geometry on one mean field: the forward
    `shared` (a `SharedForward`, None once dropped) and its Casida `solution`
    (a `CasidaSolution`, the spectrum kept when the forward is dropped), how
    many of the lowest
    roots the solve guaranteed converged (`read`, None for all), the forces
    asked for there, by root, and the state each chain sharing the cache
    matched there, by its token (what `ExcitedStateChain.commit` adopts)."""

    mf: object
    shared: object = None
    solution: object = None
    read: object = None
    forces: dict = field(default_factory=dict)
    followed: dict = field(default_factory=dict)


class GeometryCache:
    """The values of the last `size` geometries, by `geometry_key`; the
    oldest is dropped when another comes in."""

    def __init__(self, size=EVALUATION_CACHE_SIZE):
        self.size = int(size)
        self._entries = collections.OrderedDict()

    def get(self, mol):
        """The value held for the nuclei of `mol`, or None."""
        key = geometry_key(mol)
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, mol, value):
        """Hold `value` for the nuclei of `mol`; returns it."""
        key = geometry_key(mol)
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.size:
            self._entries.popitem(last=False)
        return value

    def values(self):
        """The values held, oldest first."""
        return list(self._entries.values())

    def __len__(self):
        return len(self._entries)


def geometry_key(mol):
    """The nuclei of `mol` and its electron count as one hashable value:
    the coordinates' bits in Bohr, the charges, the total charge and spin."""
    return (np.asarray(mol.atom_coords(), float).tobytes(),
            np.asarray(mol.atom_charges()).tobytes(), int(mol.charge),
            int(mol.spin))


def geometry_hash(mol):
    """sha256 of `geometry_key` as hex: the nuclei and electron count of
    `mol` as a string a record holds."""
    coords, charges, charge, spin = geometry_key(mol)
    return hashlib.sha256(coords + charges
                          + f'{charge} {spin}'.encode()).hexdigest()
