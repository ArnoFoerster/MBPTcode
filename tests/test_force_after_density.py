"""A GW-BSE state's force after its relaxed density finishes the density's
reverse pass (`ExcitedStateChain.excitation_gradient`, `HeldReverse`).

A solvation loop's last cycle takes the state's relaxed density; the force
then follows at the same charges, on the same mean field and Casida solution.

Water (bent, non-symmetric), cc-pVDZ, Hartree-Fock reference (density fitted),
G0W0-BSE S1 in fixed toluene charges (S0's equilibrium), as
`GWBSEState` sees them:

  1. the force after the density runs no quasiparticle backward and no
     multiplier solve of its own, and is the full reverse pass's force bit for
     bit; the relaxed density is unchanged.
  2. over two and three simulated ranks on the row fit: the same.
  3. a mean field whose continuum responds in the force's kernel but not in
     the frozen-field density's (a classical state-specific chain): the
     multiplier does not serve, the force solves its own and is the full
     pass's bit for bit.
  4. a new forward pass drops the held pass, and every held forward before
     it runs: a loop's cycle never holds the last cycle's factors and tape
     beside its own.

Run: python -m pytest tests/test_force_after_density.py
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.environments import solvation_loop as sl
from src.environments.environment import solvent_scheme
from src.environments.reaction_field import ContinuumField
from src.environments.solvent_screening import SolventScreening
from src.gradients import mean_field_response
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.reverse_fold import ReversePass
from src.gradients.gwbse.state_model import GWBSEState
from tests.isdf_test_grid import g1

ATOMS = 'O 0.0 0.05 0.1173; H 0.02 0.7572 -0.4692; H 0.0 -0.7672 -0.4592'


def molecule():
    return gto.M(atom=ATOMS, basis='cc-pvdz', verbose=0)


def make_scf(mol):
    mf = scf.RHF(mol).density_fit('cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-15, 1e-12, 200, 0
    return mf


def numerics(mol, rows=False):
    """The chain's Numerics (dense Casida, G1), on the row fit where asked."""
    isdf, layout = {'counts': g1(mol)}, {}
    if rows:
        isdf['fit'], layout['sliced'] = 'rows', True
    return Numerics(bse={'solver': 'dense'}, isdf=isdf, layout=layout,
                    solvent={'loop_tol_q': 1e-10, 'loop_tol_energy': 1e-12})


class Counted:
    """Counts the calls of the quasiparticle backward and the multiplier
    solve while it is entered."""

    def __init__(self, monkeypatch):
        self.n = {'qp': 0, 'lambda': 0}
        fold, solve = ReversePass._fold_quasiparticle, mean_field_response.solve_lambda

        def counted_fold(*a, **kw):
            self.n['qp'] += 1
            return fold(*a, **kw)

        def counted_solve(*a, **kw):
            self.n['lambda'] += 1
            return solve(*a, **kw)
        monkeypatch.setattr(ReversePass, '_fold_quasiparticle', counted_fold)
        monkeypatch.setattr(mean_field_response, 'solve_lambda', counted_solve)

    def reset(self):
        self.n = {'qp': 0, 'lambda': 0}


def model_in_charges(mol, num):
    """(S1 model, h, e) in S0's equilibrium charges in toluene."""
    field = ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))
    ground = GWBSEState(mol, make_scf, numerics=num)
    q = sl.equilibrate(ground, field, numerics=num).q
    state = GWBSEState(mol, make_scf, 'singlet', 0,
                       lr_environment=field.transition_environment(),
                       numerics=num)
    h, e = field.operator(q)
    return state, h, e


def full_and_held(state, h, e, counted=None):
    """(the full reverse pass's dOmega/dR, the force's after the density,
    the density): the full pass taken on the model's own forward and
    solution before the force finishes the held one."""
    r = state(h, e)
    chain, mf = state.chain, state.mf
    root, shared, solution = chain.follower.tracked(state.mol, mf)
    full = chain.reverse.root_gradient(shared, solution, root,
                                       release_tape=False)[0]
    if counted is not None:
        counted.reset()
    held = chain.excitation_gradient(state.mol, mf)[0]
    return full, held, r.info['delta_density']


def test_the_force_finishes_the_density_pass(monkeypatch):
    mol = molecule()
    state, h, e = model_in_charges(mol, numerics(mol))
    counted = Counted(monkeypatch)
    full, held, drho = full_and_held(state, h, e, counted)
    assert counted.n == {'qp': 0, 'lambda': 0}, counted.n
    assert np.array_equal(full, held), np.abs(full - held).max()
    # the density itself: a fresh model's, the same SCF and pass
    fresh, _, _ = model_in_charges(mol, numerics(mol))
    assert np.array_equal(fresh(h, e).info['delta_density'], drho)


@pytest.mark.parametrize('ranks', [2, 3])
def test_over_ranks_on_the_row_fit(monkeypatch, ranks):
    counted = Counted(monkeypatch)

    def one(comm):
        mol = molecule()
        state, h, e = model_in_charges(mol, numerics(mol, rows=True))
        full, held, _ = full_and_held(state, h, e)
        return full, held
    out = run_simulated(one, ranks)
    for full, held in out:
        assert np.array_equal(full, held), np.abs(full - held).max()
        assert np.array_equal(held, out[0][1])
    assert counted.n['lambda'] > 0


def test_a_responding_kernel_solves_its_own_multiplier(monkeypatch):
    mol = molecule()
    optical = SolventScreening(mol, solvent='toluene')
    env = solvent_scheme(optical, 'classical-ss', ss_equilibrium=True)
    mf = env.mean_field(mol, lambda m: make_scf(m).run())
    chain = ExcitedStateChain(mol, None, mf=mf, spin='singlet',
                              environment=env, numerics=numerics(mol))
    counted = Counted(monkeypatch)
    chain.excitation_density(mol, mf)
    root, shared, solution = chain.follower.tracked(mol, mf)
    full = chain.reverse.root_gradient(shared, solution, root,
                                       release_tape=False)[0]
    counted.reset()
    held = chain.excitation_gradient(mol, mf)[0]
    assert counted.n == {'qp': 0, 'lambda': 1}, counted.n
    assert np.array_equal(full, held), np.abs(full - held).max()


def test_a_new_forward_drops_the_held_pass():
    mol = molecule()
    state, h, e = model_in_charges(mol, numerics(mol))
    state(h, e)
    held = state.chain.follower.held(mol, state.mf)
    assert list(held.reverse) == [state.chain.follower.tracked(mol, state.mf)[0]]
    state(h * 1.001, e)
    # the old mean field's pass is gone; the new one holds its own
    assert not held.reverse
    assert len(state.chain.follower.held(mol, state.mf).reverse) == 1


def test_a_new_forward_runs_with_no_forward_held(monkeypatch):
    mol = molecule()
    state, h, e = model_in_charges(mol, numerics(mol))
    state(h, e)
    follower = type(state.chain.follower)
    run = follower._forward_pass
    seen = []

    def checked(self, m, mf):
        seen.append([ev.shared for ev in self.chain.evaluations.values()])
        return run(self, m, mf)
    monkeypatch.setattr(follower, '_forward_pass', checked)
    state(h * 1.001, e)
    assert seen and all(s is None for s in seen[0]), seen
