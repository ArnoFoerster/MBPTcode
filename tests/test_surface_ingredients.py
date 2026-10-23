"""One declared surface protocol.

A property that needs only energies, gradients, states and overlaps takes any
surface that provides them; what it asks beyond `PotentialEnergySurface`'s
members is a named ingredient (`surface.INGREDIENTS`), and a surface
lacking one is refused by name at the call (`MissingIngredient`), never by a
type check deeper down. No SCF here: stand-in surfaces.
"""
import numpy as np
import pytest

from src.gradients.surface import (INGREDIENTS, MissingIngredient,
                                   driven_chain, provides, require,
                                   root_attribute)
from src.properties import characters
from src.properties.excitations import spectrum_of


class States:
    """Any method's excited states: the one ingredient `spectrum_of` and
    `characters.roots` read."""

    ROOT_ATTR = 'state'

    def __init__(self):
        self.state = 0

    def excited_states(self, mol=None, mf=None):
        x = np.eye(4)[:, :2]
        return np.array([0.2, 0.3]), x, 0 * x, 2


class Bare:
    """A surface with energies only."""

    def total_energy(self, mol=None, mf=None):
        return 0.0


class Composed:
    DRIVEN_HALF = 'excited'

    def __init__(self):
        self.excited = States()


def test_any_surface_with_states_serves_the_spectrum():
    for surface in (States(), Composed()):
        omega, x, y, nocc = spectrum_of(surface, None, None)
        assert omega.tolist() == [0.2, 0.3] and nocc == 2
    assert characters.roots(States())[3] == 2
    assert root_attribute(driven_chain(Composed())) == 'state'


def test_a_missing_ingredient_is_refused_by_name():
    with pytest.raises(MissingIngredient,
                       match='Bare provides no excited_states.*spectrum_of'):
        spectrum_of(Bare(), None, None)
    assert not provides(Bare(), 'mean_field')
    with pytest.raises(MissingIngredient, match='Bare provides no mean_field'):
        require(Bare(), 'mean_field', 'a walk')


def test_the_vocabulary_is_closed():
    with pytest.raises(ValueError, match='not a surface ingredient'):
        provides(States(), 'forward')
    assert 'excited_states' in INGREDIENTS and 'mean_field' in INGREDIENTS
