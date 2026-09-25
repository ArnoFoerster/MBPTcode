"""`rsh_split` and `xc_hybrid_coeff`, the two coefficient readers.

WHAT THESE ARE NOT ANY MORE: a pin on a refusal. The skeleton derivatives
CARRY a range-separated hybrid now, channel by channel -- see
`test_range_separated_exchange.py`, which gates that, and
`isdf_derivatives.exchange_channels`, which is what they are built from. What
survives here is the reading of the coefficients themselves, and the fact that
`xc_hybrid_coeff` returns ALPHA and so is not a weight: that is the trap the
refusal was really guarding, and it outlived the refusal.
"""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.gradients.isdf_derivatives import (require_no_range_separation,
                                            rsh_split, xc_hybrid_coeff)


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='sto-3g',
                 verbose=0)


def _rks(mol, xc):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    return mf


def test_rsh_split_sees_what_hybrid_coeff_cannot(mol):
    """One scalar cannot carry a two-range operator; this is the evidence."""
    for xc, omega_expected in (('pbe0', 0.0), ('b3lyp', 0.0),
                               ('cam-b3lyp', 0.33), ('wb97x', 0.3)):
        mf = _rks(mol, xc)
        omega, alpha, beta = rsh_split(mf)
        assert abs(omega - omega_expected) < 1e-6, xc
        # what the gradient skeleton would have used, and what it misses
        _, a_x = xc_hybrid_coeff(mf)
        assert abs(a_x - alpha) < 1e-6, xc
        if omega_expected:
            assert abs(beta) > 0.4, xc      # the dropped term is not small


def test_range_separated_gradients_refuse(mol):
    for xc in ('cam-b3lyp', 'wb97x'):
        mf = _rks(mol, xc)
        with pytest.raises(NotImplementedError, match='range-separated'):
            require_no_range_separation(mf, 'a skeleton derivative')


def test_global_hybrids_and_hartree_fock_still_pass(mol):
    """The refusal must be narrow: only omega != 0 with a nonzero beta."""
    for xc in ('pbe0', 'b3lyp', 'bhandhlyp', 'pbe'):
        require_no_range_separation(_rks(mol, xc), 'a skeleton derivative')
    mf = scf.RHF(mol)
    mf.kernel()
    assert rsh_split(mf) == (0.0, 1.0, 0.0)
    require_no_range_separation(mf, 'a skeleton derivative')
