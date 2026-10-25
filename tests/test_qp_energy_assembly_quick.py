"""The fast twin of gate (d) in tests/test_qp_energy_assembly.py.

On a Kohn-Sham reference in toluene, with inside- and outside-scissor orbitals
present, the S1 force of the window chain is a central difference of its own
energy to FD_TOL. PBE0 in PCM(toluene) only, the column that carries every
term the gas phase does plus the reaction field; the gas phase and LRC-wPBEh
stay nightly. One difference along a seeded direction over every coordinate
(tests/directional_difference.py) instead of four components.
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gradients.gwbse.surface_factory import reference_mean_field
from tests import directional_difference as dd
from tests.test_qp_energy_assembly import (FD_STEP, FD_TOL, WINDOWS, build,
                                           environment, factory_for, molecule)


def test_the_force_is_the_derivative_of_its_energy_along_a_direction():
    warnings.simplefilter('ignore')
    factory = factory_for('pbe0')
    mol = molecule('formaldehyde')
    env = environment(mol, 'toluene')
    chain = build('window', mol, factory, reference_mean_field(
        mol, factory, env), env, WINDOWS['formaldehyde'])
    analytic, _ = chain.excitation_gradient()
    assert chain.resolved.scissor_map
    assert chain.resolved.outside_shift
    d = dd.direction(mol)
    miss = dd.along(analytic, d) - dd.difference(chain.excitation, mol, d,
                                                  FD_STEP)
    print(f'\nformaldehyde/PBE0 in toluene: analytic - FD {miss:.1e} Ha/Bohr')
    assert abs(miss) < FD_TOL, miss


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
