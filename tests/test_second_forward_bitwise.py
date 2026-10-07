"""A second forward pass at the reference geometry is the first, bit for bit.

The reference geometry's quasiparticle solve freezes its roots as the Newton
seeds every later solve starts from. The first solve starts at eps_p pushed
by the pole guard, a later one at the root itself, and a Newton started
elsewhere lands an ulp away (on water 1.1e-16 in eps_QP, 6e-14 in Omega,
5.3e-13 in a derivative coupling, and on the Hartree-Fock Davidson route a
flipped eigenvector). The freezing solve is repeated from its own seed, so
the reference's roots are the ones that seed reaches.

Water/cc-pVDZ: the Hartree-Fock Davidson route on the grid adjoint, and the
production surface (ISDF-K LRC-wPBEh, sum-over-poles residues, frontier set
with the outside scissor).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_shared_evaluation_consumers import (fresh, molecule,
                                                    prototype)


@pytest.mark.parametrize('case', ['hf-davidson-grid', 'production'])
def test_a_second_forward_at_the_reference_is_the_first(case):
    chain = fresh(prototype(case, molecule()), ('singlet', 0))
    mol, mf = chain.mean_field()
    om1, first = chain._forward(mol, mf)
    om2, second = chain._forward(mol, mf)
    assert np.array_equal(first[7], second[7]), 'eps_QP'
    assert np.array_equal(om1, om2), 'Omega'
    assert np.array_equal(first[10], second[10]), 'X'
    assert np.array_equal(first[11], second[11]), 'Y'


def test_the_first_quasiparticle_call_is_the_second():
    chain = fresh(prototype('production', molecule()), ('singlet', 0))
    assert chain.quasiparticle(0) == chain.quasiparticle(0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
