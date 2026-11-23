"""The two particle-hole masks of the downfolding, over an ActiveSpace.

Every later downfolding step reads these index conventions, so the test pins
them: the pair order of `active_pair_mask` and the local-index meaning of
`ph_element_mask`. The degeneracy refusal of an active space is tested with
the class, in test_active_space.py.
"""
import numpy as np

from src.EffectiveHamiltonian.active_space import (
    ActiveSpace, active_pair_mask, ph_element_mask)


def test_masks():
    space = ActiveSpace.from_indices(6, 3, [1, 2], [3, 5])
    assert active_pair_mask(space).reshape(3, 3).tolist() == [
        [0, 0, 0], [1, 0, 1], [1, 0, 1]]
    m = ph_element_mask(space)
    assert m.shape == (4,) * 4
    assert m[0, 2, 3, 1] and m[2, 0, 0, 3] and m[2, 0, 3, 1] and m[0, 2, 1, 3]
    assert not m[0, 1, 2, 3] and not m[0, 0, 2, 2]
    # exactly (2 * n_occ * n_virt)^2 elements
    assert m.sum() == (2 * 2 * 2) ** 2
