"""Numerical events in the excited-state chain warn, carry a named flag in the
record, and let the run carry on
(refuse inputs, warn numerics); the adjoint reads the forward's
floor.

- a member leaving its frozen quasiparticle-Hamiltonian group at a
  displaced geometry: warned about, the drop in `qp_subspace_drops` (which
  `qp_bookkeeping` reports as 'dropped_here').
- an element whose ISDF atomic-grid fit failed (fit error at
  ISDF_FIT_ERROR_FAILED): warned about, `fit_failed` on the factorization and
  in the grid it holds (`held_grid`, which the vertical-excitation record carries).
- two roots within ROOT_FOLLOW_MARGIN_MIN of each other share the tracked
  state: warned about, the step flagged 'mixed' in `follow_log`.
- the square-root adjoint truncates the auxiliary metric's spectrum at
  the forward's own floor, AUX_METRIC_ROOT_FLOOR, not a copy of its value.
- a mean-field exchange skeleton contracted ahead of a force that never
  asked for it warns on leaving the fit-adjoint block. (A handed-over first
  point that does not answer: tests/test_gca_first_point.py.)
- a member leaving its group with a diagonal root of Z outside (0, 1]
  keeps it, flagged 'z_outside' on its drop; a group's eigenvectors of Z
  outside (0, 1], and a group's Newton that stops short of self-consistency
  on a finite H, keep their H and are logged in `qp_hamiltonian_log`, which
  the record carries (`qp_bookkeeping`'s and the emission record's
  'qp_hamiltonian_warnings'); a non-finite H still refuses the step. A mean
  field above `scf_force_tol` under a force lands in the record's
  'scf_flags'. A matrix-free Casida solve that leaves a root the force reads
  above read_tol, or unconfirmed as the lowest, keeps its roots, warns and
  is logged in `casida_log`, which the vertical and emission records carry
  as 'casida_warnings'; a root the force does not read is not logged.

Stubs, except G5: water/cc-pVDZ on its shipped G1 radii, whose fit error is
reported as failed.
"""
import numpy as np
import pytest
from pyscf import gto

import src.Base.constants as constants
import src.gradients.factor_chain as factor_chain
from src.Base.numerics import Numerics
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.isdf_derivatives import one_fit_adjoint, sqrtm_adjoint
import src.gradients.qp_set as qp_set
from src.gradients.qp_set import QPSetMixin, QuasiparticleRootLost
from src.gradients.state_following import StateFollowingMixin
from src.Base.step_failed import DavidsonNotConverged
from src.properties.excitations import (casida_warnings, held_grid,
                                        qp_hamiltonian_warnings, scf_flags)
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_jkfit

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def test_a_member_leaving_its_frozen_group_warns_and_is_recorded():
    class Stub:
        qp_subspace_drops = {'dropped': []}

    stub = Stub()
    with pytest.warns(RuntimeWarning, match='leave their frozen'):
        QPSetMixin._dropped_here(stub, frozen=[(3, 4)], kept=[], off={4},
                                 routes={3: 'sop', 4: 'cd'},
                                 route_out={'z': [0.9, 0.9]},
                                 pos={3: 0, 4: 1})
    assert [d['orbital'] for d in stub.qp_subspace_drops['dropped']] == [3, 4]


def test_a_failed_isdf_fit_warns_and_is_flagged(monkeypatch):
    real = factor_chain.runtime_atomic_radii

    def failed(element, *args, **kwargs):
        radii, _, origin = real(element, *args, **kwargs)
        return radii, constants.ISDF_FIT_ERROR_FAILED, origin

    monkeypatch.setattr(factor_chain, 'runtime_atomic_radii', failed)
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    with pytest.warns(RuntimeWarning, match='FAILED its fit'):
        chain = ExcitedStateChain(mol, rhf_df_jkfit, mf=rhf_df_jkfit(mol),
                                  numerics=g1_numerics(mol, Numerics(
                                      bse={'solver': 'dense'})))
    assert chain.factorization.fit_failed == ['H', 'O']
    assert held_grid(chain)['fit_failed'] == ['H', 'O']


class Follower(StateFollowingMixin):
    """A chain following one state, its match planted: the best root above
    the weight floor and the runner-up within the margin."""
    track, state, nocc = 'overlap', 0, 1

    def __init__(self):
        self.follow_log, self.follow_coords = [], []
        x = np.eye(4)[:, :2]
        self._followed = self._anchor = (None, np.eye(2), x, 0 * x)

    def _match_followed(self, here, pieces):
        return 0, 0.6, 0.5 * constants.ROOT_FOLLOW_MARGIN_MIN

    def _matched_at(self, mol, mf, followed):
        pass


def test_two_mixed_roots_warn_and_are_flagged(monkeypatch):
    import src.gradients.state_following as sf
    monkeypatch.setattr(sf, 'body_overlap', lambda *a: np.eye(2))
    monkeypatch.setattr(sf, 'state_overlap', lambda *a: np.eye(2))

    class MeanField:
        mo_coeff = np.eye(2)

    class Mol:
        @staticmethod
        def atom_coords():
            return np.zeros((1, 3))

    pieces = {10: np.eye(4)[:, :2], 11: np.zeros((4, 2))}
    chain = Follower()
    with pytest.warns(RuntimeWarning, match='mixed'):
        root, _, _ = chain.tracked_root(Mol(), MeanField(), np.zeros(2),
                                        pieces, widen=False)
    assert root == 0
    assert chain.follow_log[-1]['mixed'] is True


def test_the_square_root_adjoint_reads_the_forward_floor():
    assert sqrtm_adjoint.__defaults__[0] is constants.AUX_METRIC_ROOT_FLOOR


def test_an_unasked_mean_field_skeleton_warns():
    mf = object()
    with pytest.warns(RuntimeWarning, match='never asked'):
        with one_fit_adjoint(mf, mean_field=True) as pending:
            pending.stash = ('request', 'skeleton')


def test_a_dropped_member_of_z_outside_keeps_its_root_and_is_flagged():
    class Stub:
        qp_subspace_drops = {'dropped': []}

    stub = Stub()
    with pytest.warns(RuntimeWarning, match=r'outside \(0, 1\]: \[\(4, 1.3\)\]'):
        QPSetMixin._dropped_here(stub, frozen=[(3, 4)], kept=[], off={4},
                                 routes={3: 'sop', 4: 'cd'},
                                 route_out={'z': [0.9, 1.3]},
                                 pos={3: 0, 4: 1})
    drops = {d['orbital']: d for d in stub.qp_subspace_drops['dropped']}
    assert drops[4]['z_outside'] is True and drops[4]['z'] == 1.3
    assert drops[3]['z_outside'] is False


class View:
    """A resolved view holding one frozen group (3, 4) and its seed."""
    geometry = 'geometry-a'
    qp_set = [3, 4]

    def group_seed(self, group):
        return np.diag([-0.5, 0.2]) if tuple(group) == (3, 4) else None


class Conventions:
    gw_grid, nu = None, None

    def record_groups(self, view, groups):
        self.groups = groups


class GroupChain(QPSetMixin):
    """A chain whose quasiparticle Hamiltonian has one group (3, 4)."""
    is_ks, tile_gb, nocc = False, None, 3

    def __init__(self):
        self.resolved, self.conventions = View(), Conventions()

    def _routes_taken(self, route_out):
        return {3: 'sop', 4: 'sop'}

    def _frozen_qp_subspaces(self, eps, routes):
        return [(3, 4)]

    def _group_poles(self, group, wc, eps, of=None):
        return np.array([1.0])

    def _group_newton(self, group, e0, amp, d, roots, of=None):
        return np.diag([-0.5, 0.2]), 3


def test_a_group_of_z_outside_keeps_its_h_and_is_logged(monkeypatch):
    monkeypatch.setattr(qp_set, 'pair_contractions', lambda *a: [np.zeros(2)])
    monkeypatch.setattr(qp_set, 'static_term', lambda xc, i: 0.0)
    monkeypatch.setattr(qp_set, 'subspace_residues', lambda *a: np.zeros(1))
    monkeypatch.setattr(qp_set, 'pole_denominators',
                        lambda *a: np.zeros((1, 1)))
    monkeypatch.setattr(qp_set, 'subspace_pole_strengths',
                        lambda *a: np.array([0.8, 1.2]))

    class Tape:
        wcs = [np.zeros(2), np.zeros(2)]

    class MeanField:
        class mol:
            @staticmethod
            def atom_coords():
                return np.zeros((1, 3))

    chain = GroupChain()
    with pytest.warns(RuntimeWarning, match=r'outside \(0, 1\], kept'):
        qp_h = chain._qp_hamiltonian(
            MeanField(), np.linspace(-1, 1, 6), [-0.5, 0.2],
            {'tape': Tape(), 'xc_correction': None})
    assert qp_h.groups[0].z.tolist() == [0.8, 1.2]
    entry, = chain.qp_hamiltonian_log
    assert entry['kind'] == 'z_outside' and entry['orbitals'] == [3, 4]
    assert entry['z'] == [0.8, 1.2] and entry['geometry'] == 'geometry-a'


def test_a_group_short_of_self_consistency_carries_on_and_is_logged(
        monkeypatch):
    h_end = np.array([[-0.5, 1e-6], [1e-6, 0.2]])
    monkeypatch.setattr(qp_set, 'subspace_hamiltonian',
                        lambda e0, amp, d, seed, max_iter=200:
                        (h_end, max_iter, False))
    chain = GroupChain()
    with pytest.warns(RuntimeWarning, match='no self-consistency'):
        h, it = QPSetMixin._group_newton(chain, (3, 4), np.eye(2), None,
                                         None, None)
    assert h is h_end and it == 200
    entry, = chain.qp_hamiltonian_log
    assert entry['kind'] == 'not_self_consistent'
    assert entry['newton_steps'] == 200 and entry['last_step'] == 0.0
    # what is left of a group starts from its block of the frozen seed
    with pytest.warns(RuntimeWarning, match=r'left of \[3, 4\]'):
        QPSetMixin._group_newton(chain, (4,), np.eye(1), None, None, None,
                                 of=(3, 4))
    assert [e['orbitals'] for e in chain.qp_hamiltonian_log] == [[3, 4], [4]]
    surface = type('Surface', (), {'qp_hamiltonian_log':
                                   chain.qp_hamiltonian_log})()
    assert qp_hamiltonian_warnings(surface, surface) == chain.qp_hamiltonian_log


def test_a_non_finite_group_h_still_refuses_the_step(monkeypatch):
    monkeypatch.setattr(qp_set, 'subspace_hamiltonian',
                        lambda e0, amp, d, seed, max_iter=200:
                        (np.full((2, 2), np.nan), max_iter, False))
    with pytest.raises(QuasiparticleRootLost, match='non-finite'):
        QPSetMixin._group_newton(GroupChain(), (3, 4), np.eye(2), None, None,
                                 None)


def test_scf_flags_reach_the_record_once_each():
    flag = {'where': 'total_gradient', 'residual': 1e-5, 'tol': 1e-8,
            'coords': np.zeros((1, 3))}
    excited = type('Chain', (), {'scf_flags': [flag]})()
    ground = type('Chain', (), {'scf_flags': [dict(flag)]})()
    got = scf_flags(excited, ground)
    assert got == [dict(flag, coords=[[0.0, 0.0, 0.0]])]


class Solver(StateFollowingMixin):
    """A chain on the matrix-free Casida route at state 1, unfollowed."""
    nocc, nroots, state, track, timer = 1, 3, 1, None, None
    spin, bse_tda, probe, settings, bse_conv_tol = 'singlet', False, False, None, 1e-8

    def __init__(self):
        self.davidson_solves = []
        self.resolved = View()

    def solver_used(self, n_ov):
        return 'davidson'


def stuck_davidson(converged, residuals):
    """A Davidson that leaves the roots `converged` marks False at their
    `residuals`, refusing a read one above read_tol as the real one does."""
    def solve(lr, nocc, refuse_unconverged=True, read_roots=None,
              read_tol=None, timings=None, **kw):
        read = len(converged) if read_roots is None else read_roots
        if refuse_unconverged and any(
                not c and r > read_tol
                for c, r in zip(converged[:read], residuals[:read])):
            raise DavidsonNotConverged('Refused', [1], [residuals[1]],
                                       read_tol)
        if timings is not None:
            timings.update(davidson_converged=list(converged),
                           davidson_residuals=list(residuals))
        return np.array([0.3, 0.4, 0.5]), np.eye(4)[:, :3], np.zeros((4, 3))
    return solve


def test_a_read_root_left_unconverged_is_kept_warned_and_recorded(
        monkeypatch):
    import src.gradients.state_following as sf
    monkeypatch.setattr(sf, 'solve_casida_davidson', stuck_davidson(
        [True, False, False], [1e-9, 1e-3, 1e-2]))
    chain = Solver()
    with pytest.warns(RuntimeWarning, match=r'Casida roots \[1\] the force '
                      r'reads were left above read_tol'):
        om, _, _, _ = chain._casida((None, None), None,
                                    np.array([-0.5, 0.1, 0.2, 0.3, 0.4]), None)
    assert om.tolist() == [0.3, 0.4, 0.5]
    entry, = chain.casida_log
    assert entry['kind'] == 'not_converged' and entry['roots'] == [1]
    assert entry['residuals'] == [1e-3] and entry['geometry'] == 'geometry-a'
    assert entry['read_tol'] == max(1e-8, constants.BSE_FORCE_RESIDUAL_TOL)
    # the record carries it, once for the vertical and the relaxed surface
    assert casida_warnings(chain, chain) == [entry]


def test_an_unconfirmed_read_root_is_logged(monkeypatch):
    import src.gradients.state_following as sf
    monkeypatch.setattr(sf, 'solve_casida_davidson', stuck_davidson(
        [False, True, True], [1e-9, 1e-9, 1e-9]))
    chain = Solver()
    with pytest.warns(RuntimeWarning, match='not confirmed as the lowest'):
        chain._casida((None, None), None,
                      np.array([-0.5, 0.1, 0.2, 0.3, 0.4]), None)
    assert [e['kind'] for e in chain.casida_log] == ['unconfirmed']


def test_a_read_root_both_unconverged_and_unconfirmed_is_logged(monkeypatch):
    """Davidson records each root's unconfirmed flag apart
    (`davidson_unconfirmed`): a read root left unconverged within read_tol
    that an unsettled Ritz pair may lie below is logged as unconfirmed,
    which the merged flag alone could not tell."""
    import src.gradients.state_following as sf
    stuck = stuck_davidson([True, False, True], [1e-9, 1e-7, 1e-9])

    def solve(*args, timings=None, **kw):
        out = stuck(*args, timings=timings, **kw)
        timings.update(davidson_unconfirmed=[False, True, False])
        return out

    monkeypatch.setattr(sf, 'solve_casida_davidson', solve)
    chain = Solver()
    with pytest.warns(RuntimeWarning, match='not confirmed as the lowest'):
        chain._casida((None, None), None,
                      np.array([-0.5, 0.1, 0.2, 0.3, 0.4]), None)
    entry, = chain.casida_log
    assert entry['kind'] == 'unconfirmed' and entry['roots'] == [1]
