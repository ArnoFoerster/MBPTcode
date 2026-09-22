"""The legacy chain's defaults, the renamed keys, and the aliases beside them.

WHAT BREAKS IF THIS FAILS. Three names in this repository meant two things at
once, and each cost a comparison that looked sound:

  `solver`    `ExcitedStateChain` defaulted to 'dense' while every CLI above it
              asked for 'auto', so the class and the caller that dispatched it
              could run different eigensolvers under one record. The class now
              defaults to production's ONE memory rule (`bse.solver_choice`),
              which at the pair counts a finite-difference gate runs resolves
              to the dense route it replaces -- so the flip must move no digit
              of the energy or the gradient, which is gated here bitwise
              against a chain that spells `solver='dense'`.
  `grad_max`  the driving force at an input geometry in one half of the
              repository and the residual at a converged one in the other, and
              the geometry optimizer's THRESHOLD in `constants.py`. The
              constant's key is `opt_grad_max`, the residual it is compared
              against; the old spelling is accepted for one release and sets
              the same number.

`tools/gradient_tests/systems.py` (the `tier`/`size_class` alias on the
benchmark table) is out of scope for this port, so that gate is not here.

The qp_window default is NOT flipped: this class solves the frontier set the
campaigns ran, QPStates('frontier', 2), and it is gated here so a later
readthrough does not read the flip as incomplete.

THE EQUIVALENCE GATES WERE SHOWN TO FAIL. Each was run once against a
deliberately broken copy of the file named, which was then restored from a
backup and `cmp` confirmed byte-identical:

  test_the_chain_defaults_are_the_production_ones
      src/gradients/excited_state.py: the constructor back at solver='dense'
      -> FAILED: assert 'dense' == 'auto'.
  test_auto_is_the_dense_route_at_this_pair_count
  test_the_solver_default_moved_no_number
      src/gradients/excited_state.py: `solver_used` returning 'davidson' for
      'auto' whatever the rule says -> FAILED: assert 'davidson' == 'dense'
      for the first, and the second on Omega, 0.3111593161151571 by the
      matrix-free route against 0.3111593161151481 by the dense one. THE
      TOTAL ENERGY STILL MATCHED there: 9e-15 Ha is below the last bit of
      E_0 + Omega at 75.7 Ha, which is why Omega is compared on its own.
  test_an_optimization_runs_the_same_under_either_spelling
      src/properties/optimize.py: `resolved_conv` dropping the old key instead
      of mapping it -> FAILED: 9 cycles under 'opt_grad_max' against 10 under
      'grad_max', the old spelling silently leaving the default threshold in
      force and the run relaxing past the one it was given.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import GEOM_OPT_CONV
from src.SingleReference.LinearResponse.bse import solver_choice
from src.SingleReference.base import get_occ_virt_indices
from src.gradients.excited_state import ExcitedStateChain
from src.properties.optimize import optimize, resolved_conv

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
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
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
    `qp_window=2` is the frontier set this class solves, unchanged, and
    `outside='mean-field'` is what the orbitals beyond it carry.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, mf=mf)
    assert chain.solver == 'auto'
    assert chain.residue_route == 'explicit'
    assert chain.qp_window == 2
    assert chain.outside == 'mean-field'


def test_auto_is_the_dense_route_at_this_pair_count(water):
    """At water/cc-pVDZ the rule resolves to dense, so the flip is a no-op.

    The pair count is read off the molecule rather than spelled: a number
    copied into a test stops tracking the basis it came from.
    """
    mol, mf = water
    n_ov = pair_count(mol, mf)
    chain = ExcitedStateChain(mol, scf_factory, mf=mf)
    assert chain.solver_used(n_ov) == 'dense'
    assert chain.solver_used(n_ov) == solver_choice(n_ov)


def test_the_solver_default_moved_no_number(water):
    """The default chain and an explicitly dense one, to the last bit.

    Not a tolerance: the two must run the SAME eigensolver on the same blocks,
    so the energy, the excitation energy and every gradient component are the
    same float. A tolerance here would accept exactly the divergence the flip
    has to be proved not to have introduced.
    """
    mol, mf = water
    default = ExcitedStateChain(mol, scf_factory, mf=mf)
    dense = ExcitedStateChain(mol, scf_factory, mf=mf, solver='dense')
    g_auto, e_auto, d_auto = default.total_gradient()
    g_dense, e_dense, d_dense = dense.total_gradient()
    assert e_auto == e_dense
    assert d_auto['omega'] == d_dense['omega']
    assert np.array_equal(g_auto, g_dense)


# ------------------------------------------------- the optimizer's threshold
def test_the_constant_names_the_residual_and_keeps_the_old_spelling():
    """One number under two keys, and the unambiguous one is canonical."""
    assert GEOM_OPT_CONV['opt_grad_max'] == 4.5e-4
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
    assert new_info['opt_grad_max'] == old_info['opt_grad_max']
    assert np.array_equal(new_mol.atom_coords(), old_mol.atom_coords())
    assert (GEOM_OPT_CONV['opt_grad_max'] < old_info['opt_grad_max']
            < loose['opt_grad_max'])
