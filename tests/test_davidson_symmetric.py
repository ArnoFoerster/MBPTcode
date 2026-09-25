"""The one symmetric Davidson (src/Solvers/davidson.py) that both ADC routes
iterate on, and the two routes' use of it.

There was no test here at all while the ee route carried its own hand-written
solver, and the route that used it (spin-free, matrix-free) reached the
iteration only on systems the other routes send to a dense eigh -- so the
iterative path was exercised only at a dimension where the subspace nearly
spans the problem. `dense_limit=0` forces it on a system small enough to also
solve densely, which is the only way to compare the two.

Checks:
  1. lowest-k against a dense eigh, with a degenerate pair among the roots
  2. the convergence flag is REAL -- a starved solve reports False and warns
  3. tol_residual is reachable: 1e-8 means 1e-8, not pyscf's 1e-7 lindep floor
  4. overlap_pick follows an interior root instead of the lowest
  5. ee-ADC matrix-free == dense, including the singlet/triplet channels,
     each solved in its flip-pair basis
  6. charged ADC matrix-free == dense, and last_result['converged'] is the
     solver's own flag rather than a hardcoded all-True
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Solvers.davidson import overlap_pick, solve_symmetric
from src.SingleReference.ADC import ADCSolver
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

H2O = 'O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471'


def _test_matrix(n=400, offdiag=0.02, seed=0):
    """Diagonally dominant with a wide diagonal spread -- the shape of an ADC
    supermatrix -- carrying an EXACTLY degenerate pair among its lowest roots.
    Rows 1 and 2 are decoupled and given the same diagonal, so the degeneracy
    is a property of the spectrum rather than of the couplings; one seed
    vector per root is what collapses such a pair, which is why the solver
    seeds wider than nroots."""
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=offdiag, size=(n, n))
    A = 0.5 * (A + A.T)
    A[np.diag_indices(n)] = np.sort(rng.uniform(-1.0, 3.0, n))
    for row in (1, 2):
        A[row, :] = 0.0
        A[:, row] = 0.0
    # clear of the whole coupled spectrum, not just of the lowest diagonal
    # entry: the off-diagonal block pushes the lowest Ritz values a few tenths
    # below min(diag), so a small offset would leave the pair buried mid-cluster
    A[1, 1] = A[2, 2] = A.diagonal().min() - 0.5
    return A


def check_dense_reference():
    A = _test_matrix()
    d = np.diag(A).copy()
    w = np.linalg.eigvalsh(A)
    assert abs(w[0] - w[1]) < 1e-12, 'the reference itself must be degenerate'
    e, X, conv = solve_symmetric(lambda v: A @ v, d, nroots=6, tol_residual=1e-8)
    dE = np.max(np.abs(e - w[:6]))
    res = np.max(np.linalg.norm(A @ X - X * e[None, :], axis=0))
    # both members of the degenerate pair, not one of them twice
    pair = np.abs(e - w[0]) < 1e-9
    both = pair.sum() == 2 and abs(X[:, pair][:, 0] @ X[:, pair][:, 1]) < 1e-6
    ok = conv.all() and dE < 1e-10 and res < 1e-8 and both
    print(f"lowest-6 vs dense eigh: dE={dE:.2e} |r|={res:.2e} conv={conv.all()} "
          f"degenerate pair resolved={both}: {'OK' if ok else 'FAIL'}")
    return ok


def check_convergence_flag_is_real():
    """Starve the solve and it must SAY so. The flag this replaced was
    np.ones_like(e) -- a check that cannot fail."""
    A = _test_matrix()
    d = np.diag(A).copy()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, _, conv = solve_symmetric(lambda v: A @ v, d, nroots=6,
                                     tol_residual=1e-10, max_cycle=2)
    warned = any(issubclass(r.category, RuntimeWarning) for r in rec)
    ok = (not conv.all()) and warned
    print(f"starved solve reports failure: conv.all()={conv.all()} warned={warned}: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_tolerance_is_reachable():
    """pyscf reuses its `lindep` as a 'this correction is too small' drop, so
    at its default a residual below ~1e-7 is unreachable and the solve stalls
    above the tolerance it was given. solve_symmetric scales lindep with the
    tolerance; this pins that it worked."""
    A = _test_matrix()
    d = np.diag(A).copy()
    e, X, conv = solve_symmetric(lambda v: A @ v, d, nroots=4, tol_residual=1e-9)
    res = np.max(np.linalg.norm(A @ X - X * e[None, :], axis=0))
    ok = conv.all() and res < 1e-9
    print(f"tol_residual=1e-9 actually reached: |r|={res:.2e} conv={conv.all()}: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_root_following():
    """overlap_pick must land on the root the reference vector points at, not
    on the lowest one."""
    A = _test_matrix(offdiag=0.002)
    d = np.diag(A).copy()
    w, v = np.linalg.eigh(A)
    row = 300
    target = int(np.argmax(np.abs(v[row, :])))
    ref = np.zeros(A.shape[0])
    ref[row] = 1.0
    e, X, conv = solve_symmetric(lambda v_: A @ v_, d, nroots=1, x0=ref,
                                 pick=overlap_pick(ref), tol_residual=1e-8)
    dE = abs(e[0] - w[target])
    ov = abs(X[:, 0] @ v[:, target])
    interior = target > 10
    ok = conv.all() and dE < 1e-9 and ov > 0.99 and interior
    print(f"root-following hits interior root {target} (not 0): dE={dE:.2e} "
          f"overlap={ov:.6f}: {'OK' if ok else 'FAIL'}")
    return ok


def check_ee_matrix_free_matches_dense():
    cases = [
        ('sto-3g', dict(level='adc3'), 1e-10),
        ('6-31g', dict(level='adc2'), 1e-10),
        ('6-31g', dict(level='adc2', spin='singlet'), 1e-9),
        ('6-31g', dict(level='adc2', spin='triplet'), 1e-9),
    ]
    all_ok = True
    for basis, kw, tol in cases:
        mol = gto.M(atom=H2O, basis=basis, verbose=0)
        mf = scf.RHF(mol).run()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            e_dense, _ = solve_ee_adc(mf, mol, nroots=4, matrix_free=False, **kw)
            e_iter, _ = solve_ee_adc(mf, mol, nroots=4, matrix_free=True,
                                     dense_limit=0, conv_tol=1e-8, **kw)
        stalled = [r for r in rec if issubclass(r.category, RuntimeWarning)]
        d = np.max(np.abs(np.asarray(e_dense) - np.asarray(e_iter)))
        ok = d < tol and not stalled
        all_ok &= ok
        label = f"{basis}/{kw['level']}/{kw.get('spin', 'spin-free')}"
        print(f"ee-ADC {label:<26} dense vs Davidson: {d:.2e}"
              f"{' STALLED' if stalled else ''}: {'OK' if ok else 'FAIL'}")
    return all_ok


def check_charged_adc_matrix_free_matches_dense():
    from src.SingleReference.ADC.solve import davidson_follow

    mol = gto.M(atom=H2O, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    dense = ADCSolver(mf, level='adc3', matrix_free=False)
    e_dense, _ = dense.solve(nroots=1)
    # A dense solve returns the whole pole spectrum, and its lowest eigenvalue
    # is not the root the matrix-free route follows -- that one is picked by
    # overlap with the HOMO, so select the dense pole the same way.
    vec = dense.last_result['vec']
    homo = dense.nocc - 1
    target = float(e_dense[int(np.argmax(np.abs(vec[homo, :])))])

    solver = ADCSolver(mf, level='adc3', matrix_free=True)
    e_iter, _ = solver.solve(nroots=1, conv_tol=1e-8)
    conv = solver.last_result['converged']
    d = abs(target - float(e_iter[0]))
    ok = d < 1e-8 and bool(np.all(conv))
    print(f"charged ADC(3) dense HOMO pole vs root-following Davidson: {d:.2e} "
          f"converged={np.all(conv)}: {'OK' if ok else 'FAIL'}")

    # and that flag is the solver's own, not the np.ones_like it replaced
    A = _test_matrix(n=200)
    dg = np.diag(A).copy()
    ref = np.zeros(200)
    ref[0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, _, starved = davidson_follow(lambda v: A @ v, dg, 200, 10, 0,
                                           ref, 1, conv_tol=1e-12, max_cycle=1)
        _, _, _, healthy = davidson_follow(lambda v: A @ v, dg, 200, 10, 0,
                                           ref, 1, conv_tol=1e-8)
    ok2 = not bool(np.all(starved)) and bool(np.all(healthy))
    print(f"   davidson_follow's flag distinguishes the two: starved="
          f"{bool(np.all(starved))} healthy={bool(np.all(healthy))}: "
          f"{'OK' if ok2 else 'FAIL'}")
    return ok and ok2


def main():
    all_ok = True
    for check in (check_dense_reference,
                  check_convergence_flag_is_real,
                  check_tolerance_is_reachable,
                  check_root_following,
                  check_ee_matrix_free_matches_dense,
                  check_charged_adc_matrix_free_matches_dense):
        all_ok &= bool(check())
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
