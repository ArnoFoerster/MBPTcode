"""A second forward pass at the reference geometry is the first, to a few ulp.

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
from tests.pins import assert_same, assert_same_tree


@pytest.mark.parametrize('case', ['hf-davidson-grid', 'production'])
def test_a_second_forward_at_the_reference_is_the_first(case):
    chain = fresh(prototype(case, molecule()), ('singlet', 0))
    mol, mf = chain.mean_field()
    first, sol1 = chain.forward(mol, mf)
    second, sol2 = chain.forward(mol, mf)
    # two passes: a few ulp. The vectors are held
    # without sign alignment ('root'), so a flipped eigenvector still fails.
    assert_same(second.eps_qp, first.eps_qp, label='eps_QP')
    assert_same(sol2.omega, sol1.omega, quantity='root', label='Omega')
    assert_same(sol2.x, sol1.x, unit='1', quantity='root', label='X')
    assert_same(sol2.y, sol1.y, unit='1', quantity='root', label='Y')


def test_the_first_quasiparticle_call_is_the_second():
    chain = fresh(prototype('production', molecule()), ('singlet', 0))
    assert_same_tree(chain.quasiparticle(0), chain.quasiparticle(0),
                     label='two quasiparticle calls')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
