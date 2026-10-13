"""A Casida eigenvector's sign is a convention of the vector, not of the path
the eigensolver took to it.

A Davidson root comes back with whatever sign its last Rayleigh-Ritz step
produced, and the path follows the last bits of the input: on water/cc-pVDZ
Hartree-Fock, 34 of 48 one-ulp moves of a single eps_QP flip at least one
of the four roots without a convention. A state's energy and force do not
see the sign; the
interstate numerator d<m|H|n>/dR, the derivative coupling and a spin-orbit
element do, and a sign that depends on the call history or on an ulp of the
input is a wrong sign for them half the time. `casida_phase` makes the
largest |X| element of each root positive (`CASIDA_PHASE_TIE_TOL` ties go to
the lowest index), on the Davidson and the dense route alike.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gradients.excited_state import casida_phase
from tests.test_shared_evaluation_consumers import (fresh, molecule,
                                                    prototype)

#: (orbital, direction) one-ulp moves of eps_QP that flip a root without the
#: convention: orbital 1 down flips all four.
FLIPPED = ((0, -1), (1, -1), (3, 1), (4, 1))


def _setup():
    chain = fresh(prototype('hf-davidson-grid', molecule()), ('singlet', 0))
    mol, mf = chain.mean_field()
    return chain, chain._shared_forward(mol, mf)


def _leads_positive(xn):
    for k in range(xn.shape[1]):
        mag = np.abs(xn[:, k])
        lead = np.flatnonzero(mag >= (1 - 1e-8) * mag.max())[0]
        assert xn[lead, k] > 0, k


def test_an_ulp_of_eps_qp_moves_no_sign():
    chain, shared = _setup()
    args = (shared.x_mo, shared.d)
    _, x0, _, _ = chain._casida(*args, shared.eps_qp, shared.w_aux)
    for p, direction in FLIPPED:
        eps = np.array(shared.eps_qp, float)
        eps[p] = np.nextafter(eps[p], direction * np.inf)
        _, x, _, _ = chain._casida(*args, eps, shared.w_aux)
        dots = np.einsum('ik,ik->k', x0, x)
        assert (dots > 0).all(), (p, direction, dots)


def test_both_solvers_hand_back_the_convention():
    chain, shared = _setup()
    args = (shared.x_mo, shared.d, shared.eps_qp, shared.w_aux)
    _, x_dav, _, _ = chain._casida(*args)
    _leads_positive(x_dav)
    chain.solver = 'dense'
    _, x_dense, _, _ = chain._casida(*args)
    _leads_positive(x_dense)
    nroots = x_dav.shape[1]
    assert (np.einsum('ik,ik->k', x_dav, x_dense[:, :nroots]) > 0).all()


def test_the_rule_flips_x_and_y_together():
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((6, 3)), 0.1 * rng.standard_normal((6, 3))
    x[2, 1] = -10.0
    om, xf, yf = casida_phase(np.arange(3.0), x, y)
    assert np.array_equal(xf[:, 1], -x[:, 1])
    assert np.array_equal(yf[:, 1], -y[:, 1])
    # a tie within the tolerance goes to the lowest index
    x = np.zeros((4, 1))
    x[1, 0], x[3, 0] = -1.0, 1.0 + 1e-12
    _, xf, _ = casida_phase(np.zeros(1), x, np.zeros((4, 1)))
    assert xf[1, 0] == 1.0


def test_the_first_interstate_gradient_on_a_fresh_chain_is_the_second():
    chain = fresh(prototype('hf-davidson-grid', molecule()), ('singlet', 0))
    first, _ = chain.interstate_gradient(0, 1)
    second, _ = chain.interstate_gradient(0, 1)
    assert np.array_equal(first, second)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
