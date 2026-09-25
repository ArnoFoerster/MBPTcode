"""Which combination of (X, Y) the spin-orbit element contracts, and what it costs.

Himmelsbach and Holzer (J. Chem. Phys. 161, 244105 (2024)) write the
state-to-state density in a form that differs from this repo's outside the
Tamm-Dancoff approximation, and that difference is what this file measures.

TWO STATEMENTS, AND THEY HAVE DIFFERENT SIZES.

The GROUND-state element takes L = X - Y, not R = X + Y (their Eq. 37): the
SOMF operator is skew-symmetric and R is symmetric, so the R contraction is the
wrong combination -- PySOC's choice, and worth a few per cent here. The gate
below is the one that catches someone "fixing" this back.

The INTERSTATE element's density is XX + YY here and XX - YY in their Eqs. (19)
and (21). Both reduce to the same expression at Y = 0, where the determinant
construction in `test_soc_spin_factors.py` has measured this repo correct, so
the anchor is exact and the disagreement is the YY term alone. h_SO is
antisymmetric, so only the antisymmetric part of the density survives the trace
and their explicit antisymmetrization is redundant -- the whole content of the
difference is that sign.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_CM
from src.Base.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR,
                                       GROUND_SPIN_FACTOR, _blocks, coupling,
                                       ground_state_element,
                                       interstate_element, soc_operator_mo)
from test_spin_orbit import CH2O


def element(h_mo, nocc, xs, xt, ys, yt, y_sign):
    """Tr[h gamma^IJ]; y_sign +1 is this repo's XX + YY, -1 is Holzer's XX - YY."""
    h_oo, h_vv, _ = _blocks(np.asarray(h_mo), nocc)
    xs, xt = np.reshape(xs, (nocc, -1)), np.reshape(xt, (nocc, -1))
    ys, yt = np.reshape(ys, (nocc, -1)), np.reshape(yt, (nocc, -1))
    return (-np.einsum('xji,ia,ja->x', h_oo, xs, xt, optimize=True)
            - y_sign * np.einsum('xji,ia,ja->x', h_oo, ys, yt, optimize=True)
            + np.einsum('xab,ia,ib->x', h_vv, xs, xt, optimize=True)
            + y_sign * np.einsum('xab,ia,ib->x', h_vv, ys, yt, optimize=True))


@pytest.fixture(scope='module')
def formaldehyde_bse():
    """Singlet and triplet BSE manifolds with Y kept, and the SOC operator.

    Formaldehyde because its S1 x T2 is El-Sayed allowed and large enough to
    read a per-cent effect off, and its S1 x T1 is forbidden.
    """
    mol = gto.M(atom=CH2O, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='def2-svp-ri')
    mf.conv_tol = 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    lr = LinearResponseSolver(
        np.asarray(mf.mo_energy), spin_mode='restricted',
        coeff_df=get_density_fitting_coefficients(mol, mf,
                                                  representation='spatial'))
    kw = dict(nroots=3, polarizability='BSE', conv_tol=1e-6, max_cycle=400,
              W_aux=lr.static_screening_aux(nocc))
    singlet = solve_casida_davidson(lr, nocc, spin='singlet', **kw)
    triplet = solve_casida_davidson(lr, nocc, spin='triplet', **kw)
    return soc_operator_mo(mf, mol), nocc, singlet, triplet


def test_the_local_form_is_the_modules_own(formaldehyde_bse):
    """`element` re-spells `interstate_element` so the YY sign can be flipped.
    Without this the copy could drift from the module and every comparison
    below would still pass, measuring the copy against itself."""
    h, nocc, (_, xs, ys), (_, xt, yt) = formaldehyde_bse
    for i in range(2):
        for j in range(2):
            assert np.allclose(
                element(h, nocc, xs[:, i], xt[:, j], ys[:, i], yt[:, j], +1),
                interstate_element(h, nocc, xs[:, i], xt[:, j],
                                   ys[:, i], yt[:, j]), rtol=0, atol=0)


def test_the_two_density_forms_are_one_expression_at_tda(formaldehyde_bse):
    """The anchor. At Y = 0 the YY term is absent from both, so any difference
    here would be a transcription error rather than the approximation."""
    h, nocc, (_, xs, _), (_, xt, _) = formaldehyde_bse
    z = np.zeros_like(xs)
    for i in range(2):
        for j in range(2):
            a = (h, nocc, xs[:, i], xt[:, j], z[:, i], z[:, j])
            assert np.array_equal(element(*a, +1), element(*a, -1))


def test_the_yy_sign_is_worth_under_a_per_cent(formaldehyde_bse):
    """The size of the approximation, pinned: under 1% in |V| is under 2% in a
    rate. Small, but not zero, and a change of form would move every rate."""
    h, nocc, (_, xs, ys), (_, xt, yt) = formaldehyde_bse
    seen = []
    for i in range(2):
        for j in range(2):
            a = (h, nocc, xs[:, i], xt[:, j], ys[:, i], yt[:, j])
            ours, theirs = (coupling(element(*a, s), EXCITED_SPIN_FACTOR)
                            for s in (+1, -1))
            if theirs * HARTREE_TO_CM > 1e-3:      # skip El-Sayed forbidden
                seen.append(abs(ours / theirs - 1.0))
    assert seen, 'no allowed pair survived to measure'
    assert max(seen) < 0.01, f'the density form moved |V| by {max(seen):.1%}'


def test_the_module_takes_the_minus_combination_from_the_ground_state(
        formaldehyde_bse):
    """THE ONE THAT CATCHES A REGRESSION. X + Y is the symmetric combination
    and the SOMF operator is skew-symmetric, so it is the wrong one -- but it
    is what PySOC contracts, it differs by several per cent, and nothing else
    in the suite would notice the swap."""
    h, nocc, _, (_, xt, yt) = formaldehyde_bse
    h_ov = _blocks(np.asarray(h), nocc)[2]
    x, y = np.reshape(xt[:, 0], (nocc, -1)), np.reshape(yt[:, 0], (nocc, -1))
    take = lambda v: coupling(np.einsum('xia,ia->x', h_ov, v),
                              GROUND_SPIN_FACTOR)
    minus, plus = take(x - y), take(x + y)
    assert coupling(ground_state_element(h, nocc, xt[:, 0], yt[:, 0]),
                    GROUND_SPIN_FACTOR) == pytest.approx(minus, rel=1e-12)
    assert abs(plus / minus - 1.0) > 0.02, (
        'the two combinations are too close here to gate the choice')
