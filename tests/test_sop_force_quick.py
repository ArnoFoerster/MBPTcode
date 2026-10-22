"""The fast twin of tests/test_sop_force_follows_its_energy.py.

The SOP excitation force on the production flag set is the derivative of the
surface it walks, and a pole set re-fitted at every geometry fails that gate.
Distorted formaldehyde/cc-pVDZ on Hartree-Fock, with one difference along a
seeded direction over every coordinate (tests/directional_difference.py)
instead of three components. Water, ethylene's pole-basis noise, the Kohn-Sham
and range-separated references stay nightly.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests import directional_difference as dd
from tests.test_sop_force_follows_its_energy import (HF_GATES,
                                                     HF_TRANSLATION_BAR,
                                                     molecule, sop_chain)

NAME = 'formaldehyde'
_, _, STEP, BAR = next(g for g in HF_GATES if g[0] == NAME)


@pytest.fixture(scope='module')
def formaldehyde():
    mol = molecule(NAME)
    chain = sop_chain(mol, 'hf')
    grad, _ = chain.excitation_gradient()
    return mol, chain, grad, dd.direction(mol)


def test_the_hartree_fock_sop_force_follows_its_energy_along_a_direction(
        formaldehyde):
    mol, chain, grad, d = formaldehyde
    assert chain.resolved.sop_poles, 'no state took the pole model'
    translation = np.abs(grad.sum(axis=0)).max()
    assert translation < HF_TRANSLATION_BAR, translation
    miss = dd.along(grad, d) - dd.difference(chain.excitation, mol, d, STEP)
    print(f'\n{NAME}/HF: analytic - FD {miss:.1e} Ha/Bohr, translation '
          f'{translation:.1e}')
    assert abs(miss) < BAR, miss


def test_poles_refit_per_geometry_fail_the_gate(formaldehyde):
    """The gate can fail: re-fitting the poles at each displaced geometry."""
    mol, chain, grad, d = formaldehyde
    frozen = {p: v.copy() for p, v in chain.conventions.sop_poles.items()}

    def refit(m):
        chain.conventions.sop_poles.clear()
        return chain.excitation(m)

    try:
        miss = dd.along(grad, d) - dd.difference(refit, mol, d, STEP)
    finally:
        chain.conventions.sop_poles.clear()
        chain.conventions.sop_poles.update(frozen)
    assert abs(miss) > 10 * BAR, miss


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
