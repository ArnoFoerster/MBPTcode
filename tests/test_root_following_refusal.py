"""A followed BSE state that leaves the solved roots is searched for again,
then refused -- never replaced by whichever root overlaps it best.

`ExcitedStateChain.tracked_root` matches the previous step's eigenvector
against the roots of this geometry. A best overlap below
ROOT_FOLLOW_WEIGHT_MIN means the state is not among them: the walk used to
warn and carry on with that best root, which put another state's force into
the step (DMABN S1, overlaps of 0.00 to 0.49 against force spikes of
1.45 Ha/Bohr). Now the same forward is solved again for
ROOT_FOLLOW_RESOLVE_FACTOR times the roots and matched again, and a state
still missing raises `RootFollowingLost`.

The Casida step is planted: the roots are columns of one orthogonal matrix
at one geometry, so every overlap is exactly 0 or 1 and the test reads the
decision, not a solver.
"""
import types

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import (ROOT_FOLLOW_RESOLVE_FACTOR,
                                         ROOT_FOLLOW_WEIGHT_MIN)
from src.gradients.gwbse.conventions import FrozenConventions
from src.gradients.evaluation import GeometryCache
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_following import CasidaSolution, RootFollowingLost

#: Roots a matrix-free solve returns; the planted pair space holds 10.
NROOTS = 3


@pytest.fixture(scope='module')
def mean_field():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


@pytest.fixture(scope='module')
def roots(mean_field):
    """An orthonormal basis of the pair space: the planted eigenvectors."""
    mol, _ = mean_field
    nocc = mol.nelectron // 2
    n_ov = nocc * (mol.nao - nocc)
    q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((n_ov, n_ov)))
    return q


def follower(mol):
    """An overlap-following chain with nothing built but what following reads."""
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.settings = Numerics()
    chain.conventions = FrozenConventions()
    chain.track, chain.state, chain.nroots = 'overlap', 0, NROOTS
    chain.nocc = mol.nelectron // 2
    chain._followed = chain._anchor = None
    chain._evaluations, chain._owner = GeometryCache(), object()
    chain.follow_log, chain.follow_coords = [], []
    chain.timer = None
    return chain


def forward(mol, mf, x):
    """(shared, `CasidaSolution`): the geometry, and the roots `x` (Y = 0)
    with energies 0, 1, ..."""
    shared = types.SimpleNamespace(mol=mol, mf=mf, x_mo=None, d=None,
                                   eps_qp=None, w_aux=None, qp_h=None)
    return shared, CasidaSolution(np.arange(x.shape[1], dtype=float), x,
                                  np.zeros_like(x), {})


def plant(chain, columns, calls):
    """The chain's Casida step returns `columns` of the planted roots."""
    def casida(x_mo, d, eps_qp, w_aux, at=None):
        calls.append(chain.nroots)
        x = columns
        return np.arange(x.shape[1], dtype=float), x, np.zeros_like(x), {}
    chain.follower._casida = casida


def anchored(mean_field, roots):
    mol, mf = mean_field
    chain = follower(mol)
    chain.tracked_root(mol, mf, *forward(mol, mf, roots[:, :NROOTS]))
    return chain


def test_a_lost_state_is_found_by_the_wider_solve(mean_field, roots):
    mol, mf = mean_field
    chain = anchored(mean_field, roots)
    # the state (column 0) has left the three roots solved here ...
    narrow = roots[:, 1:NROOTS + 1]
    # ... and is the fourth root of the wider solve
    wide = roots[:, [1, 2, 3, 0, 4, 5]]
    calls = []
    plant(chain, wide, calls)
    root, found = chain.tracked_root(mol, mf, *forward(mol, mf, narrow))
    assert calls == [ROOT_FOLLOW_RESOLVE_FACTOR * NROOTS]
    assert chain.nroots == NROOTS, 'the wider solve must not stick'
    assert root == 3
    assert found.x is wide and len(found.omega) == wide.shape[1]
    step = chain.follow_log[-1]
    assert step['weight'] > 1.0 - 1e-10
    assert step['widened_to'] == wide.shape[1]


def test_a_state_missing_from_the_wider_solve_is_refused(mean_field, roots):
    mol, mf = mean_field
    chain = anchored(mean_field, roots)
    calls = []
    plant(chain, roots[:, 1:2 * NROOTS + 1], calls)
    with pytest.raises(RootFollowingLost, match='refused'):
        chain.tracked_root(mol, mf, *forward(mol, mf, roots[:, 1:NROOTS + 1]))
    assert calls == [ROOT_FOLLOW_RESOLVE_FACTOR * NROOTS]
    assert len(chain.follow_log) == 1, 'a refused step is not logged'


def test_every_root_solved_is_refused_without_a_resolve(mean_field, roots):
    """The dense route returns the whole pair space: nothing wider exists."""
    mol, mf = mean_field
    chain = anchored(mean_field, roots)
    calls = []
    plant(chain, roots, calls)
    # a full basis over which the state is spread evenly: no root holds it
    n = roots.shape[1]
    spread = np.eye(n)
    spread[:, 0] = 1.0 / np.sqrt(n)
    q, _ = np.linalg.qr(spread)
    full = roots @ q.T
    assert np.abs(roots[:, 0] @ full).max() < ROOT_FOLLOW_WEIGHT_MIN
    with pytest.raises(RootFollowingLost):
        chain.tracked_root(mol, mf, *forward(mol, mf, full))
    assert calls == []


def test_a_followed_state_is_returned_as_before(mean_field, roots):
    """A state that stays in the window costs no second solve."""
    mol, mf = mean_field
    chain = anchored(mean_field, roots)
    calls = []
    plant(chain, roots, calls)
    order = [2, 0, 1]
    x = roots[:, order]
    root, found = chain.tracked_root(mol, mf, *forward(mol, mf, x))
    assert root == 1 and calls == [] and found.x is x
    assert 'widened_to' not in chain.follow_log[-1]
