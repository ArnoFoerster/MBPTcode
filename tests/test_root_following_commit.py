"""Root following moves only on a geometry the walk accepted.

WHAT BREAKS IF THIS FAILS. `ExcitedStateChain.tracked_root` replaced the
followed eigenvector on EVERY forward pass: a trial step the walk then
rejected, an energy read elsewhere, each of the 6N displacements of a
Hessian. The next step was then matched against a geometry the walk never
stood at, and near a crossing that ratchets onto the neighbouring state. Now
every evaluation is matched against the last COMMITTED geometry and the
walkers commit their start and each accepted step (`surface.commit`). The
match is held with the geometry's evaluation, so a commit adopts what was
matched at the accepted geometry whatever was evaluated after it, and a
commit with no match there is refused.

The Casida step is planted as in `test_root_following_refusal.py`: the roots
are columns of one orthogonal matrix at one geometry, so every overlap is a
number set by hand. A rejected trial mixes the followed state with its
neighbour by ANGLE; the next evaluation holds the state pure. Matched
against the committed anchor it overlaps 1; matched against the rejected
trial, as the code before the fix did, cos(ANGLE).
"""
import copy

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics import Numerics
from src.gradients.conventions import FrozenConventions
from src.gradients.evaluation import GeometryCache
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_following import ForwardPieces
from src.properties.optimize import (HAVE_GEOMETRIC, optimize,
                                     optimize_geometric)

#: Roots planted per geometry.
NROOTS = 3
#: Radians: how far the rejected trial rotates the state into its neighbour,
#: more than 45 degrees, so that the neighbour's root is the one matched there.
ANGLE = np.deg2rad(50.0)
#: Bohr: two geometries apart for the cache, the same for the overlaps.
TINY = 1e-5


@pytest.fixture(scope='module')
def mean_field():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


@pytest.fixture(scope='module')
def roots(mean_field):
    mol, _ = mean_field
    nocc = mol.nelectron // 2
    n_ov = nocc * (mol.nao - nocc)
    q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((n_ov, n_ov)))
    return q


def follower():
    """An overlap-following chain with nothing built but what following reads."""
    chain = ExcitedStateChain.__new__(ExcitedStateChain)
    chain.settings = Numerics()
    chain.conventions = FrozenConventions()
    chain.track, chain.state, chain.nroots = 'overlap', 0, NROOTS
    chain.nocc = 5
    chain._followed = chain._anchor = None
    chain._evaluations, chain._owner = GeometryCache(), object()
    chain.follow_log, chain.follow_coords = [], []
    chain.timer = None
    return chain


def pieces(mol, mf, x):
    body = [None] * 17
    body[0], body[1] = mol, mf
    body[10], body[11] = x, np.zeros_like(x)
    return ForwardPieces(tuple(body))


def match(chain, mol, mf, x):
    """(root, weight) of the followed state among the planted roots `x`."""
    root, _, _ = chain.tracked_root(mol, mf, np.arange(x.shape[1], dtype=float),
                                    pieces(mol, mf, x))
    return root, chain.follow_log[-1]['weight']


def shifted(mol, d):
    """`mol` with every coordinate moved by `d` Bohr."""
    out = mol.copy()
    out.set_geom_(mol.atom_coords() + d, unit='Bohr')
    return out


def mixed(roots):
    """The trial's roots: the state rotated by ANGLE into its neighbour."""
    c, s = np.cos(ANGLE), np.sin(ANGLE)
    x = roots[:, :NROOTS].copy()
    x[:, 0] = c * roots[:, 0] + s * roots[:, 1]
    x[:, 1] = -s * roots[:, 0] + c * roots[:, 1]
    return x


def test_an_uncommitted_evaluation_moves_nothing(mean_field, roots):
    mol, mf = mean_field
    chain = follower()
    match(chain, mol, mf, roots[:, :NROOTS])
    match(chain, mol, mf, mixed(roots))
    root, weight = match(chain, mol, mf, roots[:, :NROOTS])
    assert root == 0
    assert weight == pytest.approx(1.0, abs=1e-10)


def test_a_committed_evaluation_is_what_the_next_one_matches(mean_field,
                                                             roots):
    mol, mf = mean_field
    chain = follower()
    match(chain, mol, mf, roots[:, :NROOTS])
    trial, _ = match(chain, mol, mf, mixed(roots))
    assert trial == 1
    assert chain.commit(mol)
    root, weight = match(chain, mol, mf, roots[:, :NROOTS])
    assert root == 0
    assert weight == pytest.approx(np.sin(ANGLE), abs=1e-10)


def test_a_commit_elsewhere_is_refused_and_moves_nothing(mean_field, roots):
    mol, mf = mean_field
    chain = follower()
    match(chain, mol, mf, roots[:, :NROOTS])
    match(chain, mol, mf, mixed(roots))
    with pytest.raises(RuntimeError, match='no state matched'):
        chain.commit(shifted(mol, 0.1))
    assert match(chain, mol, mf, roots[:, :NROOTS])[1] == pytest.approx(
        1.0, abs=1e-10)


def test_a_trial_step_is_adopted_after_an_evaluation_elsewhere(mean_field,
                                                              roots):
    """The match is held with the geometry, not in one slot: an energy read
    at another geometry between a trial step and its acceptance used to
    leave `commit` matching against the wrong geometry, and say False."""
    mol, mf = mean_field
    chain = follower()
    match(chain, mol, mf, roots[:, :NROOTS])
    trial = shifted(mol, TINY)
    assert match(chain, trial, mf, mixed(roots))[0] == 1
    match(chain, shifted(mol, -TINY), mf, roots[:, :NROOTS])
    assert chain.commit(trial)
    root, weight = match(chain, mol, mf, roots[:, :NROOTS])
    assert root == 0
    assert weight == pytest.approx(np.sin(ANGLE), abs=1e-4)


def test_copies_sharing_the_cache_adopt_their_own_match(mean_field, roots):
    mol, mf = mean_field
    chain = follower()
    match(chain, mol, mf, roots[:, :NROOTS])
    other = copy.copy(chain)
    assert other._evaluations is chain._evaluations
    trial = shifted(mol, TINY)
    assert match(chain, trial, mf, mixed(roots))[0] == 1
    assert match(other, trial, mf, roots[:, :NROOTS])[0] == 0
    assert chain.commit(trial) and other.commit(trial)
    assert match(chain, mol, mf, roots[:, :NROOTS])[1] == pytest.approx(
        np.sin(ANGLE), abs=1e-4)
    assert match(other, mol, mf, roots[:, :NROOTS])[1] == pytest.approx(
        1.0, abs=1e-4)


class Bowl:
    """E = k (r - r_e)^2 / 2 in the bond, plus 1 Ha at the first trial step
    alone, which the walk therefore rejects; records every geometry it was
    told to commit."""

    def __init__(self, mol):
        self.mol0 = mol
        self.evaluated, self.committed = [], []

    def total_gradient(self, mol=None, mf=None):
        here = mol.atom_coords()
        d = here[1] - here[0]
        r = float(np.linalg.norm(d))
        bump = 1.0 if len(self.evaluated) == 1 else 0.0
        grad = np.zeros_like(here)
        grad[1] = 0.4 * (r - 2.2) * d / r
        grad[0] = -grad[1]
        self.evaluated.append(here.copy())
        return grad, 0.2 * (r - 2.2) ** 2 + bump, {}

    def commit(self, mol):
        self.committed.append(mol.atom_coords().copy())

    def refreeze(self, mol):
        return self


@pytest.mark.parametrize('engine', ['internal', 'geometric'])
def test_the_walkers_commit_the_start_and_accepted_steps_only(engine):
    mol = gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                verbose=0)
    surface = Bowl(mol)
    if engine == 'geometric':
        if not HAVE_GEOMETRIC:
            pytest.skip('geomeTRIC is not installed')
        mol_opt, info = optimize_geometric(surface, mol, verbose=False,
                                           nudge=0)
    else:
        mol_opt, info = optimize(surface, mol, verbose=False, nudge=0)
    assert info['converged']
    assert surface.committed, 'nothing was committed'
    assert np.array_equal(surface.committed[0], mol.atom_coords())
    rejected = surface.evaluated[1]
    assert not any(np.array_equal(c, rejected) for c in surface.committed)
    assert all(any(np.array_equal(c, e) for e in surface.evaluated)
               for c in surface.committed)
    # the minimum the walk returns is committed
    assert np.abs(surface.committed[-1] - mol_opt.atom_coords()).max() < 1e-6
