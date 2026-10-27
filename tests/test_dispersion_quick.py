"""The fast twin of the dRPA force gate in tests/test_dispersion.py.

dE_0/dR against a difference of the E_0 the chain REPORTS, on an ISDF
Kohn-Sham reference, at the nightly file's grid levels (PBE0 at 3, LRC-wPBEh
at 6). One Richardson difference along a seeded direction over every
coordinate (tests/directional_difference.py) instead of all nine components:
four energies instead of thirty-six. The gate is the finer step's own h^2
truncation along that direction, as in the nightly file.
"""
import os
import sys

import pytest
from pyscf import gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.Base.numerics.numerics import numerics_of
from tests import directional_difference as dd
from tests.isdf_test_grid import g1
from tests.test_dispersion import FD_STEPS, H2O_C1, isdf_kohn_sham


@pytest.mark.parametrize('xc, level', [('pbe0', 3), ('lrc-wpbeh', 6)])
def test_the_drpa_force_follows_its_reported_energy_along_a_direction(xc, level):
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    factory = isdf_kohn_sham(xc, level)
    chain = RPAGroundStateChain(mol, factory, mf=factory(mol), numerics=numerics_of({'counts': g1(mol)}))
    g, e, _ = chain.total_gradient()
    assert e == pytest.approx(chain.energy()[0], abs=1e-12)
    d = dd.direction(mol)
    fd, truncation = dd.richardson(lambda m: chain.energy(m)[0], mol, d,
                                   FD_STEPS)
    miss = dd.along(g, d) - fd
    print(f'\n{xc} level {level}: analytic - FD {miss:.1e}, truncation '
          f'{truncation:.1e} Ha/Bohr')
    assert abs(miss) < truncation, (miss, truncation)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
