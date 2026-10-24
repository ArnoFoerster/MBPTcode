"""The legacy chain's defaults and the renamed keys.

WHAT BREAKS IF THIS FAILS. Two names would mean two things at once, and each
would cost a comparison that looks sound:

  `solver`    `ExcitedStateChain` defaulting to 'dense' while every CLI above
              it asks for 'auto' would let the class and the caller that
              dispatched it run different eigensolvers under one record. The
              class defaults to production's ONE memory rule
              (`bse.solver_choice`), which at the pair counts a
              finite-difference gate runs resolves to the dense route, so the
              default moves no digit of the energy or the gradient, which is
              gated here to a few ulp against a chain that spells `solver='dense'`.
  `grad_max`  the driving force at an input geometry in one half of the
              repository and the residual at a converged one in the other, and
              the geometry optimizer's THRESHOLD in `constants.py`. The
              constant's key is `opt_grad_max`, the residual it is compared
              against; `grad_max` in a walk's `conv` is refused as a name the
              walk does not hold.

The qp_window default stays the frontier set, QPStates('frontier', 2), which
this class solves; it is gated here.

What breaks each equivalence gate:

  * test_the_chain_defaults_are_the_production_ones:
    src/gradients/gwbse/excited_state.py with the constructor at solver='dense'
    gives 'dense' where 'auto' is asserted.
  * test_auto_is_the_dense_route_at_this_pair_count and
    test_the_solver_default_moved_no_number: src/gradients/gwbse/excited_state.py
    with `solver_used` returning 'davidson' for 'auto' whatever the rule
    says. The first sees 'davidson' where 'dense' is asserted; the second
    sees Omega, 0.3111593161151571 by the matrix-free route against
    0.3111593161151481 by the dense one. THE TOTAL ENERGY STILL MATCHES: 9e-15
    Ha is below the last bit of E_0 + Omega at 75.7 Ha, which is why Omega is
    compared on its own.
  * test_the_old_spelling_is_refused: src/properties/optimize.py with
    `resolved_conv` merging an unknown key silently leaves the default
    threshold in force under a name the caller thinks is read.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import (GEOM_OPT_CONV, SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.SingleReference.LinearResponse.casida import solver_choice
from src.Base.orbital_spaces import get_occ_virt_indices
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.optimize import resolved_conv
from tests.isdf_test_grid import g1_numerics
from tests.pins import assert_same

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'

#: A quartic bond, in Bohr and Hartree: cheap enough to relax twice per
#: spelling of the threshold, and -- unlike a harmonic one, which an RFO step
#: solves exactly in one cycle whatever it is asked for -- it approaches its
#: minimum slowly enough that the FORCE THRESHOLD is what stops the run. A run
#: that stops on the step criterion instead would agree under both spellings
#: without either of them being read.
QUARTIC = 1.0
R_E = 2.2


def scf_factory(mol):
    """A mean field converged for gradient work, as the chain requires."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


def pair_count(mol, mf):
    """n_ov, the dimension the memory rule is applied to."""
    eps = np.asarray(mf.mo_energy, float)
    occ, virt = get_occ_virt_indices(eps, mol.nelectron // 2)
    return len(occ) * len(virt)


# --------------------------------------------------------- the class defaults
def test_the_chain_defaults_are_the_production_ones(water):
    """What a chain built with nothing but defaults declares it will do.

    `solver='auto'` is the memory rule; `residue_route='explicit'` takes the
    real-axis residues as they stand rather than picking a backend per state;
    `qp_window=2` is the frontier set this class solves, unchanged, and the
    orbitals beyond it carry the frozen scissor, the only treatment there is.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    assert chain.solver == 'auto'
    assert chain.residue_route == 'explicit'
    assert chain.qp_window == 2
    assert chain.conventions.outside_record() == {'outside': 'scissor'}


def test_auto_is_the_dense_route_at_this_pair_count(water):
    """At water/cc-pVDZ the rule resolves to dense, so the flip is a no-op.

    The pair count is read off the molecule rather than spelled: a number
    copied into a test stops tracking the basis it came from.
    """
    mol, mf = water
    n_ov = pair_count(mol, mf)
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    assert chain.solver_used(n_ov) == 'dense'
    assert chain.solver_used(n_ov) == solver_choice(n_ov)


def test_the_solver_default_moved_no_number(water):
    """The default chain and an explicitly dense one, to a few ulp.

    Not a method tolerance: the two must run the SAME eigensolver on the same
    blocks, so the energy, the excitation energy and every gradient component
    are the same float up to the last bits two runs may move.
    A looser tolerance here would accept exactly the divergence
    the flip has to be proved not to have introduced.
    """
    mol, mf = water
    default = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    dense = ExcitedStateChain(
        mol, scf_factory, mf=mf, numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    g_auto, e_auto, d_auto = default.total_gradient()
    g_dense, e_dense, d_dense = dense.total_gradient()
    assert_same(e_auto, e_dense, label='energy')
    assert_same(d_auto['omega'], d_dense['omega'], label='omega')
    assert_same(g_auto, g_dense, quantity='gradient', label='gradient')


# ------------------------------------------------- the optimizer's threshold
def test_the_constant_names_the_residual():
    """One number under one key, the residual the convergence test reads."""
    assert 'grad_max' not in GEOM_OPT_CONV
    assert resolved_conv(None)['opt_grad_max'] == GEOM_OPT_CONV['opt_grad_max']


def test_the_old_spelling_is_refused():
    """A threshold under a name the walk does not read would leave the
    default in force."""
    with pytest.raises(ValueError, match='no convergence'):
        resolved_conv({'grad_max': 1e-3})


