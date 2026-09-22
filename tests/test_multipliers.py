"""Gates for src/gradients/multipliers.py -- the shared orbital-response solve.

The two gradient engines used to carry one copy each of the Z-vector solve
antisym(Y_E + Y[fold(Lambda)]) = 0. Both copies are embedded below verbatim,
and the shared solver is required to reproduce them BITWISE on each route's
own inputs: the dense four-index matvec on H4/sto-3g, and the Coulomb/exchange
matvec `fock_partial_Y` on water/cc-pVDZ. The only edit to the copied bodies
is that the tolerances are spelled as the constants rather than as digits;
they are the same numbers, which is what `test_the_constants_are_the_numbers`
asserts.

Bitwise, not approximate: lgmres is deterministic for identical inputs and
call order, so anything that moved -- a different preconditioner, a different
right-hand side, an extra float conversion -- shows up in the last bit rather
than hiding under a tolerance. Every gate here was shown to fail under a
perturbation of the shared solver, each restored and `cmp`-verified:

  drop the preconditioner (M=Mop -> M=None)      both equivalence gates fail,
      by 6.9e-16 (dense) and 2.0e-12 (cubic) -- the same equation converged
      along a different Krylov path
  flip the right-hand side (rhs = -antisym -> +antisym)   both equivalence
      gates and both stationarity checks fail
  drop the degenerate-pair refusal               its gate fails
  drop the residual check                        its gate fails

The OVERALL sign of the antisymmetrization is a gauge of this equation and
nothing can see it: `M - M^T -> M^T - M` negates the right-hand side and the
operator together, so Lambda is unchanged to the last bit and all seven tests
still pass. Only a sign applied on ONE side of the equation is a defect.

The last two gates are the solver's refusals. A degenerate orbital pair is
projected out because the equation cannot determine a rotation inside a
degenerate set; the right-hand side there vanishes by the same symmetry, so
one that does not is a broken assumption and must refuse. And an operator
blind to one pair converges, reports info=0, and leaves a finite residual --
which is exactly what the residual check is for, since the multiplier enters
the force linearly and a stagnated solve gives a plausible wrong gradient.
"""
import numpy as np
import pytest
import scipy.sparse.linalg
from pyscf import ao2mo, gto, scf

from src.Base.constants import (ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                                ORBITAL_MULTIPLIER_MAX_ITER,
                                ORBITAL_MULTIPLIER_RESIDUAL_TOL,
                                ORBITAL_MULTIPLIER_TOL)
from src.gradients.grad_engine import (_y_fold_lambda, fold_fock, orbital_Y,
                                       solve_multipliers)
from src.gradients.isdf_derivatives import fock_partial_Y, solve_lambda
from src.gradients.multipliers import solve_orbital_multipliers
from src.gradients.qb_core import RPA, build_rpa_AB
from src.gradients.targets import rpa_partials

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
H4 = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'


# ---------------------------------------------------------------------------
# the two pre-consolidation implementations, copied verbatim
# ---------------------------------------------------------------------------

def _antisym_to_vec(Mmat, iu):
    A = Mmat - Mmat.T
    return A[iu]


def _vec_to_sym(v, norb, iu):
    Lam = np.zeros((norb, norb))
    Lam[iu] = v
    return Lam + Lam.T


def _reference_solve_multipliers(Y_E, F_mo, eri_mo, nocc, eps,
                                 tol=ORBITAL_MULTIPLIER_TOL, verbose=False):
    """`grad_engine.solve_multipliers` as it stood before consolidation."""
    norb = F_mo.shape[0]
    iu = np.triu_indices(norb, k=1)
    rhs = -_antisym_to_vec(Y_E, iu)

    de = eps[iu[1]] - eps[iu[0]]           # eps_q - eps_p for p<q (row p, col q)
    deg = np.abs(de) < ORBITAL_MULTIPLIER_DEGENERACY_TOL
    if deg.any():
        bad = np.abs(rhs[deg]).max() if rhs[deg].size else 0.0
        if bad > ORBITAL_MULTIPLIER_RESIDUAL_TOL:
            raise RuntimeError(f"degenerate orbital pair carries gradient {bad:.2e}; "
                               "symmetry-adapted treatment needed")
        rhs = np.where(deg, 0.0, rhs)

    def matvec(v):
        v = np.where(deg, 0.0, v)
        Lam = _vec_to_sym(v, norb, iu)
        out = _antisym_to_vec(_y_fold_lambda(Lam, F_mo, eri_mo, nocc), iu)
        return np.where(deg, v, out)

    def precond(v):
        return np.where(deg, v, v / (2.0 * np.where(deg, 1.0, -de)))

    n = rhs.size
    op = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)
    Mop = scipy.sparse.linalg.LinearOperator((n, n), matvec=precond)
    x, info = scipy.sparse.linalg.lgmres(
        op, rhs, M=Mop, rtol=0.0, atol=tol * max(1.0, np.linalg.norm(rhs)),
        maxiter=ORBITAL_MULTIPLIER_MAX_ITER)
    res = np.linalg.norm(matvec(x) - rhs)
    if verbose:
        print(f"    [multipliers] n={n} info={info} |res|={res:.3e}")
    if info != 0 or res > ORBITAL_MULTIPLIER_RESIDUAL_TOL * max(1.0, np.linalg.norm(rhs)):
        raise RuntimeError(f"multiplier solve failed: info={info}, res={res:.3e}")
    return _vec_to_sym(np.where(deg, 0.0, x), norb, iu), res


def _reference_solve_lambda(mf, Y_E, nocc, tol=ORBITAL_MULTIPLIER_TOL,
                            max_iter=ORBITAL_MULTIPLIER_MAX_ITER, verbose=False):
    """`isdf_derivatives.solve_lambda` as it stood before consolidation."""
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    iu = np.triu_indices(norb, k=1)
    de = eps[iu[1]] - eps[iu[0]]
    deg = np.abs(de) < ORBITAL_MULTIPLIER_DEGENERACY_TOL

    def to_vec(Mx):
        return (Mx - Mx.T)[iu]

    rhs = -to_vec(Y_E)
    if deg.any():
        bad = np.abs(rhs[deg]).max() if rhs[deg].size else 0.0
        if bad > ORBITAL_MULTIPLIER_RESIDUAL_TOL:
            raise RuntimeError(f"degenerate orbital pair carries gradient "
                               f"{bad:.2e}; symmetry-adapted treatment needed")
        rhs = np.where(deg, 0.0, rhs)

    def matvec(v):
        v = np.where(deg, 0.0, v)
        Lam = np.zeros((norb, norb))
        Lam[iu] = v
        Lam = Lam + Lam.T
        return np.where(deg, v, to_vec(fock_partial_Y(mf, Lam, nocc)))

    n = rhs.size
    op = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)
    prec = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=lambda v: np.where(deg, v, v / (2.0 * np.where(deg, 1.0, -de))))
    x, info = scipy.sparse.linalg.lgmres(
        op, rhs, M=prec, rtol=0.0, atol=tol * max(1.0, np.linalg.norm(rhs)),
        maxiter=max_iter)
    res = np.linalg.norm(matvec(x) - rhs)
    if info != 0 or res > ORBITAL_MULTIPLIER_RESIDUAL_TOL * max(1.0, np.linalg.norm(rhs)):
        raise RuntimeError(f"Lambda solve failed: info={info}, res={res:.3e}")
    Lam = np.zeros((norb, norb))
    Lam[iu] = np.where(deg, 0.0, x)
    if verbose:
        print(f"    [Lambda] res={res:.3e}")
    return Lam + Lam.T, res


# ---------------------------------------------------------------------------
# inputs: one case per route, built the way its own driver builds it
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def dense_case():
    """H4/sto-3g dRPA: the four-index route's Y_E, exactly as
    `correlation_gradients` assembles it before the multiplier solve."""
    mol = gto.M(atom=H4, unit='Bohr', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    norb, nocc = mol.nao, mol.nelectron // 2
    C = mf.mo_coeff
    eri = ao2mo.general(mol, (C,) * 4, compact=False).reshape((norb,) * 4)
    h_mo = C.T @ mf.get_hcore() @ C
    F_mo = np.diag(mf.mo_energy)
    gammaF, Gamma4 = rpa_partials(RPA(*build_rpa_AB(mf.mo_energy, eri, nocc)[:2]),
                                  nocc, norb)
    gh_E, _ = fold_fock(gammaF, nocc, Gamma4)
    Y_E = orbital_Y(gh_E, Gamma4, h_mo, eri)
    return Y_E, F_mo, eri, nocc, mf.mo_energy


@pytest.fixture(scope='module')
def cubic_case():
    """Water/cc-pVDZ density-fitted: the cubic route's Y_E, the orbital-energy
    adjoint of the HOMO folded through `fock_partial_Y` as the epsilon chain
    does it."""
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    nocc = mol.nelectron // 2
    eps_bar = np.zeros(len(mf.mo_energy))
    eps_bar[nocc - 1] = 1.0
    return mf, fock_partial_Y(mf, np.diag(eps_bar), nocc), nocc


def _hartree_fock_matvec(eps):
    """Y[Lambda] = 2 F Lambda with F = diag(eps): the response with the
    two-electron part switched off, whose antisymmetric part is exactly
    2 (eps_p - eps_q) Lambda_pq -- the equation the preconditioner inverts."""
    F = np.diag(eps)
    return lambda Lam: 2.0 * (F @ Lam)


def _pair_blind_matvec(eps, pair):
    """The same response with one pair's antisymmetric part erased: an
    operator that has silently dropped a term. It is singular there, so lgmres
    stagnates at a finite residual instead of failing outright."""
    exact = _hartree_fock_matvec(eps)
    p, q = pair

    def matvec(Lam):
        Y = exact(Lam)
        Y[p, q] = Y[q, p] = 0.5 * (Y[p, q] + Y[q, p])
        return Y
    return matvec


# ---------------------------------------------------------------------------
# equivalence with the pre-consolidation implementations
# ---------------------------------------------------------------------------

def test_the_dense_route_reproduces_its_old_solver(dense_case):
    """The four-index matvec through the shared solver, bitwise."""
    Y_E, F_mo, eri, nocc, eps = dense_case
    lam_ref, res_ref = _reference_solve_multipliers(Y_E, F_mo, eri, nocc, eps)
    lam_new, res_new = solve_multipliers(Y_E, F_mo, eri, nocc, eps)
    assert np.array_equal(lam_new, lam_ref)
    assert res_new == res_ref
    assert np.abs(lam_new).max() > 1e-6          # there is a multiplier to compare


def test_the_cubic_route_reproduces_its_old_solver(cubic_case):
    """`fock_partial_Y` through the shared solver, bitwise."""
    mf, Y_E, nocc = cubic_case
    lam_ref, res_ref = _reference_solve_lambda(mf, Y_E, nocc)
    lam_new, res_new = solve_lambda(mf, Y_E, nocc)
    assert np.array_equal(lam_new, lam_ref)
    assert res_new == res_ref
    assert np.abs(lam_new).max() > 1e-6


def test_the_constants_are_the_numbers_the_two_copies_carried():
    """The consolidation moved four hard-coded numbers; it did not change one."""
    assert ORBITAL_MULTIPLIER_TOL == 1e-11
    assert ORBITAL_MULTIPLIER_MAX_ITER == 3000
    assert ORBITAL_MULTIPLIER_DEGENERACY_TOL == 1e-8
    assert ORBITAL_MULTIPLIER_RESIDUAL_TOL == 1e-7


# ---------------------------------------------------------------------------
# the two refusals
# ---------------------------------------------------------------------------

def test_a_degenerate_pair_is_projected_out_when_its_gradient_vanishes():
    """A rotation inside a degenerate set does not change the energy, so the
    equation cannot determine it -- and does not have to, because its
    right-hand side vanishes by the same symmetry."""
    eps = np.array([-0.7, -0.7, 0.3, 1.1])
    Y_E = np.array([[0.0, 0.25, 0.4, -0.2],
                    [0.25, 0.0, 0.1, 0.7],
                    [-0.3, 0.6, 0.0, 0.15],
                    [0.5, -0.1, 0.9, 0.0]])
    assert Y_E[0, 1] == Y_E[1, 0]                 # the degenerate pair carries none
    Lam, res = solve_orbital_multipliers(_hartree_fock_matvec(eps), Y_E, eps)
    assert Lam[0, 1] == 0.0                       # projected out, not guessed
    assert res < ORBITAL_MULTIPLIER_RESIDUAL_TOL
    # every pair the equation DOES determine is stationary
    Y_tot = Y_E + _hartree_fock_matvec(eps)(Lam)
    A = Y_tot - Y_tot.T
    A[0, 1] = A[1, 0] = 0.0
    assert np.abs(A).max() < 1e-10


def test_a_degenerate_pair_carrying_gradient_refuses():
    """The paired failure: the same degenerate pair with a right-hand side
    that does not vanish is not a case to project out, it is a broken
    assumption, and a Lambda that ignored it would be silently wrong."""
    eps = np.array([-0.7, -0.7, 0.3, 1.1])
    Y_E = np.array([[0.0, 0.25, 0.4, -0.2],
                    [0.55, 0.0, 0.1, 0.7],
                    [-0.3, 0.6, 0.0, 0.15],
                    [0.5, -0.1, 0.9, 0.0]])
    assert Y_E[0, 1] != Y_E[1, 0]
    with pytest.raises(RuntimeError, match='degenerate orbital pair'):
        solve_orbital_multipliers(_hartree_fock_matvec(eps), Y_E, eps)


def test_an_operator_blind_to_one_pair_is_caught_by_the_residual():
    """lgmres reports info=0 on stagnation as well as on convergence, so the
    residual is recomputed and checked. Without that the force comes back
    smooth, plausible and wrong."""
    eps = np.array([-0.7, -0.2, 0.3, 1.1])
    Y_E = np.array([[0.0, 0.25, 0.4, -0.2],
                    [0.55, 0.0, 0.1, 0.7],
                    [-0.3, 0.6, 0.0, 0.15],
                    [0.5, -0.1, 0.9, 0.0]])
    with pytest.raises(RuntimeError, match=r'multiplier solve failed.*res='):
        solve_orbital_multipliers(_pair_blind_matvec(eps, (0, 1)), Y_E, eps,
                                  max_iter=5)


def test_the_same_right_hand_side_solves_with_the_whole_operator():
    """The paired healthy case: nothing about that right-hand side is bad."""
    eps = np.array([-0.7, -0.2, 0.3, 1.1])
    Y_E = np.array([[0.0, 0.25, 0.4, -0.2],
                    [0.55, 0.0, 0.1, 0.7],
                    [-0.3, 0.6, 0.0, 0.15],
                    [0.5, -0.1, 0.9, 0.0]])
    Lam, res = solve_orbital_multipliers(_hartree_fock_matvec(eps), Y_E, eps,
                                         max_iter=5)
    assert res < ORBITAL_MULTIPLIER_RESIDUAL_TOL
    assert np.array_equal(Lam, Lam.T)             # symmetric by construction
    assert np.abs(np.diag(Lam)).max() == 0.0      # zero diagonal by construction
    Y_tot = Y_E + _hartree_fock_matvec(eps)(Lam)
    assert np.abs(Y_tot - Y_tot.T).max() < 1e-10
