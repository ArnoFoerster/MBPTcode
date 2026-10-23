"""The legacy chain's defaults, the renamed keys, and the aliases beside them.

WHAT BREAKS IF THIS FAILS. Three names would mean two things at once, and each
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
              against; `grad_max` is accepted as an alias and sets the same
              number.

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
  * test_an_optimization_runs_the_same_under_either_spelling:
    src/properties/optimize.py with `resolved_conv` dropping the alias key
    instead of mapping it gives 9 cycles under 'opt_grad_max' against 10
    under 'grad_max': the alias silently leaves the default threshold in
    force and the run relaxes past the one it was given.

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
from src.properties.optimize import optimize, resolved_conv
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


class QuarticSurface:
    """E = a (r - r_e)^4 / 4 in the bond length of a diatomic."""

    def __init__(self, mol, a=QUARTIC, r_e=R_E):
        self.mol0, self.a, self.r_e = mol, a, r_e

    def scf_factory(self, mol):
        """No mean field: the surface is a function of the nuclei alone."""
        return None

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), mf

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        d = mol.atom_coords()[1] - mol.atom_coords()[0]
        r = float(np.linalg.norm(d))
        grad = np.zeros((mol.natm, 3))
        grad[1] = self.a * (r - self.r_e) ** 3 * d / r
        grad[0] = -grad[1]
        return grad, 0.25 * self.a * (r - self.r_e) ** 4, {}

    def label(self):
        return f'quartic toy, r_e = {self.r_e} Bohr'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


@pytest.fixture
def n2_toy():
    """A homonuclear pair 0.8 Bohr inside the minimum of the toy surface."""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.4', unit='Bohr', basis='sto-3g',
                verbose=0)
    return QuarticSurface(mol), mol


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
def test_the_constant_names_the_residual_and_keeps_the_old_spelling():
    """One number under two keys, and the unambiguous one is canonical."""
    assert GEOM_OPT_CONV['grad_max'] == GEOM_OPT_CONV['opt_grad_max']
    resolved = resolved_conv(None)
    assert resolved['opt_grad_max'] == GEOM_OPT_CONV['opt_grad_max']
    assert resolved['grad_max'] == resolved['opt_grad_max']


def test_the_old_spelling_still_sets_the_threshold_and_says_it_is_old():
    """A caller's stored `conv` keeps working, and hears that it is superseded."""
    with pytest.warns(DeprecationWarning, match='opt_grad_max'):
        out = resolved_conv({'grad_max': 1e-3})
    assert out['opt_grad_max'] == 1e-3
    assert out['grad_max'] == 1e-3
    assert out['step_max'] == GEOM_OPT_CONV['step_max']


def test_the_two_spellings_disagreeing_are_refused():
    """Two thresholds under one meaning: one of them is not in force, and
    which is not something a caller can be left to guess."""
    with pytest.raises(ValueError, match='one threshold under two names'):
        resolved_conv({'opt_grad_max': 1e-3, 'grad_max': 1e-5})
    # The allowed case: the same number twice is the mirrored dict, not a
    # contradiction, and passes silently.
    agreeing = resolved_conv({'opt_grad_max': 1e-3, 'grad_max': 1e-3})
    assert agreeing['opt_grad_max'] == 1e-3


def test_an_optimization_runs_the_same_under_either_spelling(n2_toy):
    """The alias is not decoration: it has to reach the convergence test.

    The threshold is looser than GEOM_OPT_CONV's own and the step criteria are
    wide open, so the run stops on the FORCE and on the number this dict names.
    The residual it stops at is checked to sit above the default: a spelling
    that never reached the convergence test would relax further and land on the
    default threshold instead, which is what makes this a measurement of the
    alias rather than of the optimizer.
    """
    surface, mol = n2_toy
    loose = {'opt_grad_max': 1e-3, 'grad_rms': 1e-3,
             'step_max': 1.0, 'step_rms': 1.0}
    new_mol, new_info = optimize(surface, mol, conv=dict(loose), verbose=False)
    old = dict(loose)
    old['grad_max'] = old.pop('opt_grad_max')
    with pytest.warns(DeprecationWarning):
        old_mol, old_info = optimize(surface, mol, conv=old, verbose=False)
    assert new_info['converged'] and old_info['converged']
    assert new_info['cycles'] == old_info['cycles']
    assert_same(new_info['opt_grad_max'], old_info['opt_grad_max'],
                quantity='gradient', label='opt_grad_max')
    assert_same(new_mol.atom_coords(), old_mol.atom_coords(), unit='1',
                quantity='root', label='geometry')
    assert (GEOM_OPT_CONV['opt_grad_max'] < old_info['opt_grad_max']
            < loose['opt_grad_max'])
