"""The derivative couplings refuse what they cannot define.

- d_0n couples the closed-shell ground state to a singlet: a triplet chain is
  refused on the chain branch as on the evaluation branch, and the chain
  branch refuses a non-differentiable environment as `analytic_coupling` does.
- The amplitude term divides by Omega_n - Omega_m: two roots closer than
  COUPLING_GAP_MIN are refused, off an evaluation and off a chain.
- The configuration term has no canonical rotation for a degenerate orbital
  pair: gamma on such a pair is refused, as `multipliers` refuses its
  right-hand side.
- The configuration term's Z-vector is solved on the evaluation's own
  `Numerics`: one whose multiplier_max_iter stops it short warns
  CphfNotConverged and flags 'orbital_response_not_converged', and a
  coupling with no `Numerics` at all is refused.

Stub chains and evaluations for the first two (each refusal fires before any
electronic structure), water/cc-pVDZ DF-RHF for the last two.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base import numerical_flags
from src.Base.numerics.constants import COUPLING_GAP_MIN
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.cphf import CphfNotConverged
from src.gradients.gwbse.state_following import CasidaSolution
from src.gradients.gwbse.derivative_coupling import (analytic_coupling,
                                                     analytic_ground_coupling,
                                                     configuration_coupling,
                                                     evaluated_coupling)
from tests.reference_scf import rhf_df_jkfit
from tests.molecules import WATER

S0, S1 = ('singlet', 0), ('singlet', 1)


class Reached(Exception):
    """The stub's electronic structure was asked for: no refusal fired."""


class NotDifferentiable(Exception):
    pass


class StubChain:
    """A chain whose every step past the guards raises `Reached`."""

    def __init__(self, spin='singlet', differentiable=True, gap=None):
        self.spin, self.differentiable, self.gap = spin, differentiable, gap

    def require_differentiable_environment(self):
        if not self.differentiable:
            raise NotDifferentiable

    def mean_field(self, mol, mf):
        if self.gap is None:
            raise Reached
        return None, None

    def require_scf_quality(self, mf, where):
        return []

    def forward(self, mol, mf):
        return None, CasidaSolution(np.zeros(2), np.zeros((4, 2)),
                                    np.zeros((4, 2)), {})

    @property
    def reverse(self):
        """The stub is its own reverse pass (`ExcitedStateChain.reverse`)."""
        return self

    def interstate_gradient_of(self, shared, solution, m, n):
        return np.zeros((3, 3)), {'gap': self.gap}


def test_a_triplet_chain_has_no_ground_state_coupling():
    with pytest.raises(ValueError, match='singlet'):
        analytic_ground_coupling(StubChain('triplet'), 0)


def test_the_ground_coupling_refuses_a_non_differentiable_environment():
    with pytest.raises(NotDifferentiable):
        analytic_ground_coupling(StubChain(differentiable=False), 0)


def test_degenerate_roots_are_refused_off_a_chain():
    with pytest.raises(RuntimeError, match='COUPLING_GAP_MIN'):
        analytic_coupling(StubChain(gap=0.1 * COUPLING_GAP_MIN), 0, 1)


class StubEvaluation:
    """Two singlet roots `gap` apart, and nothing else."""

    def __init__(self, gap):
        x = np.zeros((4, 2))
        self.interstate = {(S0, S1): (np.zeros((3, 3)), {})}
        self.root, self.target = {S0: 0, S1: 1}, {S0: S0, S1: S1}
        self.spectrum = {'singlet': (np.array([0.3, 0.3 + gap]), x, x)}
        self.mol = self.mf = None

    def require_scf_quality(self, where):
        return []


def test_degenerate_roots_are_refused_off_an_evaluation():
    with pytest.raises(RuntimeError, match='COUPLING_GAP_MIN'):
        evaluated_coupling(StubEvaluation(0.0), S0, S1)
    with pytest.raises(RuntimeError, match='COUPLING_GAP_MIN'):
        evaluated_coupling(StubEvaluation(-0.5 * COUPLING_GAP_MIN), S0, S1)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER,
                basis='cc-pvdz', verbose=0)
    return mol, rhf_df_jkfit(mol)


def test_density_on_a_degenerate_orbital_pair_is_refused(water):
    mol, mf = water
    nocc = mol.nelectron // 2
    nvir = len(mf.mo_energy) - nocc
    held = mf.mo_energy.copy()
    try:
        # the first two virtuals made degenerate
        mf.mo_energy = held.copy()
        mf.mo_energy[nocc + 1] = mf.mo_energy[nocc]
        goo = np.zeros((nocc, nocc))
        gvv = np.zeros((nvir, nvir))
        gvv[0, 1] = gvv[1, 0] = 0.3
        with pytest.raises(RuntimeError, match='degenerate orbital pair'):
            configuration_coupling(mol, mf, nocc, goo, gvv, Numerics())
    finally:
        mf.mo_energy = held


class WaterEvaluation(StubEvaluation):
    """Two singlet roots of random amplitudes on water's mean field, held
    with `numerics`."""

    def __init__(self, mol, mf, numerics):
        nocc = mol.nelectron // 2
        n_ov = nocc * (len(mf.mo_energy) - nocc)
        rng = np.random.default_rng(7)
        x, y = rng.standard_normal((2, n_ov, 2))
        super().__init__(0.2)
        self.spectrum = {'singlet': (np.array([0.3, 0.5]), x, 0.1 * y)}
        self.interstate = {(S0, S1): (np.zeros((mol.natm, 3)), {})}
        self.mol, self.mf, self.numerics = mol, mf, numerics


def test_the_coupling_solves_its_z_vector_on_the_evaluations_numerics(water):
    mol, mf = water
    short = Numerics(scf={'multiplier_max_iter': 1})
    flags = []
    with pytest.warns(CphfNotConverged), numerical_flags.collecting(flags):
        d, _ = evaluated_coupling(WaterEvaluation(mol, mf, short), S0, S1)
    assert [f['flag'] for f in flags] == ['orbital_response_not_converged']
    assert np.isfinite(d).all()
    flags = []
    with numerical_flags.collecting(flags):
        evaluated_coupling(WaterEvaluation(mol, mf, Numerics()), S0, S1)
    assert flags == []


def test_a_coupling_without_numerics_is_refused(water):
    mol, mf = water
    with pytest.raises(TypeError, match="needs the run's Numerics"):
        evaluated_coupling(WaterEvaluation(mol, mf, None), S0, S1)
