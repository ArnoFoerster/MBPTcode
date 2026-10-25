"""The Davidson returns the LOWEST roots, not merely converged ones.

A root whose guess pair is dressed far below its own diagonal by couplings to
high pairs starts as a Ritz value above the next root up. A Hermitian Davidson
that corrects only the lowest `nroots` Ritz pairs never refines it and
converges, every residual below conv_tol, on the next root instead:
formaldehyde/cc-pVDZ triplet in toluene on the excited-state chain's own
spectrum, 4 Tamm-Dancoff roots, gave 0.366009 Ha for a fourth root at
0.363695 Ha. The Tamm-Dancoff Davidson tracks extra Ritz pairs
(`tracked_roots`) until their residual intervals clear the highest root.

The full (Casida) Davidson (`trial_space.casida_rows`) tracks the same extra
Ritz pairs. A solver that corrects max(nroots, 20) Ritz pairs a cycle but
stops once the lowest nroots converged (pyscf's real_eig) skips the trap where
the requested roots converge first (exact in the first cycle)
or where nroots >= 20 leaves no corrected pair beyond them. The loop settles
an extra pair by its residual interval in the metric S = [[A, B], [B, A]],
in which J S is self-adjoint, rather than by |r| <= conv_tol, which stalled
benzene/cc-pVDZ BSE triplet at 12 roots on the degenerate pairs above the
12th.

Gated here, against eigensolutions of the same action built whole:

  * model pair spaces with that trap: Tamm-Dancoff at 3, 4 and 5 roots,
    full at 4 and 24 roots, and the full one at 2 and 3 simulated ranks, the
    same bits on every rank;
  * every residual interval the full loop measured holds an eigenvalue;
  * a full solve stopped at the cycle cap with an extra pair unsettled
    refuses (DavidsonRefusal) under refuse_unconverged and warns otherwise,
    and a Tamm-Dancoff one is refused or warned about as unconverged;
  * an unstable (A - B) is refused as the dense CasidaSolver refuses it
    (CasidaInstability);
"""
import numpy as np
import pytest
from pyscf import scf

from src.Base.numerics.constants import SCF_DIFFERENTIABLE_GRAD_TOL, SCF_TIGHT_CONV_TOL
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.casida import (CasidaInstability,
                                                       CasidaSolver)
from src.SingleReference.LinearResponse.davidson import (DavidsonNotConverged,
                                                         DavidsonRefusal,
                                                         _dense_casida,
                                                         run_davidson)

BASIS = 'cc-pvdz'
MOLECULES = {
    'water': 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24',
    'formaldehyde': 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.59; H 0 -0.94 -0.59'}
#: Ha: the Davidson roots against the dense ones, at conv_tol = CONV_TOL
ROOT_TOL = 1e-7
CONV_TOL = 1e-9
NROOTS = (3, 4, 6, 8)
#: Pair rows per tile where the full trap is split over simulated ranks.
TRAP_TILE = 100


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


def trapped_pair_space(no=4, nv=10, seed=0):
    """Symmetric A whose pair 3 (diagonal 0.50 Ha) is coupled only to the 8
    highest pairs and dressed to the second root, 0.203 Ha."""
    n = no * nv
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0.6, 2.0, n))
    d[:6] = [0.20, 0.30, 0.40, 0.50, 0.47, 0.55]
    a = np.diag(d)
    high = np.arange(n - 8, n)
    a[3, high] = a[high, 3] = 0.25
    rest = np.setdiff1d(np.arange(n), np.r_[3, high])
    g = 0.002 * rng.standard_normal((len(rest), len(rest)))
    a[np.ix_(rest, rest)] += g + g.T
    return a, no, nv


@pytest.mark.parametrize('nroots', [3, 4, 5])
def test_tda_davidson_finds_a_root_dressed_below_its_diagonal(nroots):
    """The model trap: the dressed pair is the second root, and its unit
    vector's Ritz value starts at 0.50 Ha, above the next three roots."""
    a, no, nv = trapped_pair_space()
    ref = np.linalg.eigvalsh(a)[:nroots]

    def apply_a(z):
        return (z.reshape(len(z), -1) @ a).reshape(len(z), no, nv), None

    om = run_davidson(apply_a, np.diag(a).reshape(no, nv).copy(), nroots,
                       CONV_TOL, 100, None, preconditioner='bare',
                       tda=True)[0]
    assert np.abs(np.sort(om) - ref).max() < ROOT_TOL


def trapped_casida(nroots, b_diag=0.0, noise=0.0, no=4, nv=300, seed=0):
    """Casida (A, B) whose pair `nroots`, just above the requested diagonals,
    is coupled only to the 8 highest pairs and dressed to the second root;
    B is b_diag on the diagonal, the other pairs couple by `noise`."""
    n = no * nv
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0.6, 2.0, n))
    d[:nroots + 1] = np.linspace(0.2, 0.55, nroots + 1)
    a = np.diag(d)
    high = np.arange(n - 8, n)
    a[nroots, high] = a[high, nroots] = 0.25
    rest = np.setdiff1d(np.arange(n), np.r_[nroots, high])
    g = noise * rng.standard_normal((len(rest), len(rest)))
    a[np.ix_(rest, rest)] += g + g.T
    return a, b_diag * np.eye(n), no, nv


def casida_action(a, b, no, nv):
    def apply_ab(z):
        z = z.reshape(len(z), -1)
        return ((z @ a).reshape(len(z), no, nv),
                (z @ b).reshape(len(z), no, nv))
    return apply_ab, np.diag(a).reshape(no, nv).copy()


@pytest.mark.parametrize('nroots, b_diag, noise', [(4, 0.05, 0.0),
                                                   (24, 0.0, 1e-5)],
                         ids=['converged-first', 'beyond-space-inc'])
def test_full_davidson_finds_a_root_dressed_below_its_diagonal(nroots, b_diag,
                                                               noise):
    """The trap on the full Casida problem: at 4 roots the requested pairs are
    exact in the first cycle, at 24 the trapped pair lies beyond the 24 Ritz
    pairs real_eig corrected; real_eig returned the third root for the
    second in both."""
    a, b, no, nv = trapped_casida(nroots, b_diag, noise)
    apply_ab, diag = casida_action(a, b, no, nv)
    ref = _dense_casida(apply_ab, diag, no * nv, 1.0, 256)[0][:nroots]
    om = run_davidson(apply_ab, diag, nroots, CONV_TOL, 100, None,
                       preconditioner='bare')[0]
    assert np.abs(np.sort(om) - ref).max() < ROOT_TOL


@pytest.mark.parametrize('size', [2, 3])
def test_full_trap_is_found_at_every_rank_count(size):
    """The converged-first trap with its pair space cut into tiles over
    simulated ranks: every rank returns rank 0's bits, the dense roots."""
    a, b, no, nv = trapped_casida(4, 0.05)
    apply_ab, diag = casida_action(a, b, no, nv)
    ref = _dense_casida(apply_ab, diag, no * nv, 1.0, 256)[0][:4]

    def one_rank(comm):
        return run_davidson(apply_ab, diag, 4, CONV_TOL, 100, None,
                             comm=comm, pair_tile=TRAP_TILE,
                             preconditioner='bare')

    out = run_simulated(one_rank, size)
    for om, X, Y in out:
        for got, want in zip((om, X, Y), out[0]):
            assert got.tobytes() == want.tobytes()
    assert np.abs(np.sort(out[0][0]) - ref).max() < ROOT_TOL


def test_every_residual_interval_holds_an_eigenvalue():
    """Each r_S the full loop measured on the trap is an interval
    [theta - r_S, theta + r_S] holding an eigenvalue of the Casida problem."""
    a, b, no, nv = trapped_casida(4, 0.05)
    apply_ab, diag = casida_action(a, b, no, nv)
    dense = _dense_casida(apply_ab, diag, no * nv, 1.0, 256)[0]
    counters = davidson._davidson_counters()
    davidson._davidson_root(apply_ab, diag, 4, CONV_TOL, 100, None,
                            counters=counters, preconditioner='bare')
    measured = [s for s in counters['trace'] if s['r_s'] is not None]
    assert measured
    for step in measured:
        theta = step['omega'][4:4 + len(step['r_s'])]
        for th, r in zip(theta, step['r_s']):
            if r > 0:
                assert np.abs(dense - th).min() <= r


def test_unsettled_extra_pair_at_the_cap_is_loud():
    """Stopped after one cycle, the requested roots of the trap are exact
    eigenpairs (|r| = 0) while the trapped extra pair is unsettled: the solve
    refuses under refuse_unconverged, whatever the read roots' residuals,
    and with refuse_unconverged=False warns and records the doubted roots
    unconverged."""
    a, b, no, nv = trapped_casida(4, 0.05)
    apply_ab, diag = casida_action(a, b, no, nv)
    kw = dict(preconditioner='bare')
    with pytest.raises(DavidsonRefusal, match='could not confirm'):
        run_davidson(apply_ab, diag, 4, CONV_TOL, 1, None,
                      refuse_unconverged=True, read_roots=2, read_tol=1e-3,
                      **kw)
    timings = {}
    with pytest.warns(RuntimeWarning, match='could not confirm'):
        run_davidson(apply_ab, diag, 4, CONV_TOL, 1, None, timings=timings,
                      refuse_unconverged=False, **kw)
    assert timings['davidson_verdict'] == 'unconfirmed'
    assert not all(timings['davidson_converged'])


def test_the_tda_trap_at_the_cap_is_loud_and_converges_past_it():
    """The Tamm-Dancoff solve is the one symmetric Davidson, which corrects
    every tracked pair each cycle and calls no root converged in its first:
    stopped there it refuses under refuse_unconverged and warns otherwise,
    and four cycles converge every requested root, the trapped one
    included."""
    a, _, no, nv = trapped_casida(4, 0.0)

    def apply_a(z):
        return (z.reshape(len(z), -1) @ a).reshape(len(z), no, nv), None

    diag = np.diag(a).reshape(no, nv).copy()
    kw = dict(preconditioner='bare', tda=True)
    with pytest.raises(DavidsonNotConverged):
        run_davidson(apply_a, diag, 4, CONV_TOL, 1, None,
                      refuse_unconverged=True, **kw)
    timings = {}
    with pytest.warns(RuntimeWarning, match='unconverged'):
        run_davidson(apply_a, diag, 4, CONV_TOL, 1, None, timings=timings,
                      refuse_unconverged=False, **kw)
    assert timings['davidson_verdict'] == 'unconverged'
    om = run_davidson(apply_a, diag, 4, CONV_TOL, 4, None, **kw)[0]
    assert np.abs(np.sort(om) - np.linalg.eigvalsh(a)[:4]).max() < ROOT_TOL


def test_an_unstable_problem_is_refused_as_the_dense_solver_refuses():
    """(A - B) with a negative eigenvalue: the dense CasidaSolver and the
    Davidson both raise CasidaInstability, before any root comes back."""
    a, b, no, nv = trapped_casida(4)
    b[0, 0] = a[0, 0] + 0.1
    with pytest.raises(CasidaInstability):
        CasidaSolver(a, b).solve()
    apply_ab, diag = casida_action(a, b, no, nv)
    with pytest.raises(CasidaInstability, match=r'min eig\(A-B\)'):
        run_davidson(apply_ab, diag, 4, CONV_TOL, 100, None,
                      preconditioner='bare')


