"""A refrozen surface takes over the followed STATE, not an index.

WHAT BREAKS IF THIS FAILS. If `ExcitedStateChain.refreeze` handed the new
surface only a root index, read off the root-following log, the new surface
would adopt whatever root sits at that index at its first evaluation, with
no overlap check. At a near-degenerate pair (the LE/CT pair of DMABN) the
refrozen walk could start on the other state, and nothing in the record
could tell. The committed eigenvector travels with the refrozen chain and
its first evaluation matches it by overlap, at the same geometry.

The case: water S1 (HF/cc-pVDZ), followed by overlap, refrozen where it was
evaluated. The log entry the index is read from is made to name root 1, as
an entry evaluated at a rejected step can. A refrozen surface that
trusted the index would report root 1's excitation energy; this one finds its
state at root 0 with an overlap of one.
"""
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import SCF_DIFFERENTIABLE_GRAD_TOL, SCF_TIGHT_CONV_TOL
from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics

#: Hartree: two evaluations of one state at one geometry on two SCFs.
SAME_STATE = 1e-8


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def test_the_refrozen_surface_matches_the_followed_state():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    chain = ExcitedStateChain(
        mol, scf_factory, spin='singlet', state=0, mf=scf_factory(mol),
        numerics=g1_numerics(mol, Numerics(bse={'track': 'overlap'})))
    omega = chain.excitation(mol)
    # the index the refreeze is handed names another root
    chain.follow_log[-1]['index'] = 1
    refrozen = chain.refreeze(mol)
    assert refrozen.excitation(mol) == pytest.approx(omega, abs=SAME_STATE)
    first = refrozen.follow_log[0]
    assert first['index'] == 0
    assert first['weight'] == pytest.approx(1.0, abs=1e-6)
