import os
import sys
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.SingleReference.LinearResponse.casida import CasidaSolver


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def synthetic(n, kind, rng):
    """(A, B): 'diag' has A-B exactly diagonal (the RPA branch), 'chol' takes the
    Cholesky branch, 'fallback' has an indefinite A+B (the shifted branch). The
    1/sqrt(n) scale keeps V's and W's spectral norms n-independent, so A+B stays
    positive definite for 'diag'/'chol' and indefinite for 'fallback' at any n."""
    d = np.sort(rng.uniform(0.2, 2.0, n))
    V = rng.standard_normal((n, n)) * (0.01 / np.sqrt(n))
    V = V + V.T
    if kind == 'diag':
        return np.diag(d) + 2.0 * V, 2.0 * V
    W = rng.standard_normal((n, n)) * (0.005 / np.sqrt(n))
    W = W + W.T
    if kind == 'chol':
        return np.diag(d) + 2.0 * V - W, 2.0 * V - 0.5 * W
    return np.diag(d - 1.5) + 2.0 * V - W, 2.0 * V - 0.5 * W


def peak_over_inputs(fn, unit):
    """Traced peak during fn() minus the traced size before it, in units of `unit` bytes."""
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return (peak - base) / unit, out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    all_ok = True

    # --- contract on a small problem: definiteness, residuals, normalization ---
    for kind in ('diag', 'chol', 'fallback'):
        A, B = synthetic(120, kind, rng)
        min_eig = np.linalg.eigvalsh(A + B).min()
        want_definite = kind != 'fallback'
        all_ok &= check(min_eig > 0 if want_definite else min_eig < 0,
                        f'{kind}: A+B definiteness as intended', f'{min_eig:.1e}')
        omega, X, Y = CasidaSolver(A, B).solve()
        if kind != 'fallback':
            r1 = np.max(np.abs(A @ X + B @ Y - X * omega[None, :]))
            r2 = np.max(np.abs(B @ X + A @ Y + Y * omega[None, :]))
            all_ok &= check(r1 < 1e-9 and r2 < 1e-9, f'{kind}: Casida residuals', f'{r1:.1e}, {r2:.1e}')
        nrm = np.max(np.abs(X.T @ X - Y.T @ Y - np.eye(len(omega))))
        all_ok &= check(nrm < 1e-9, f'{kind}: X^T X - Y^T Y = 1', f'{nrm:.1e}')

    # --- keep_intermediates gates the instance attributes, not the result ---
    A, B = synthetic(120, 'chol', rng)
    s_default = CasidaSolver(A, B)
    res_default = s_default.solve()
    s_keep = CasidaSolver(A, B, keep_intermediates=True)
    res_keep = s_keep.solve()
    all_ok &= check(s_default.Z is None, 'default: Z not stored')
    all_ok &= check(s_keep.Z is not None and s_keep.Z.shape == A.shape, 'keep_intermediates: Z stored')
    all_ok &= check(all(np.array_equal(a, b) for a, b in zip(res_default, res_keep)),
                    'keep_intermediates does not change omega, X, Y')

    # --- tracemalloc ratchets at n = 1500, in units of one n x n float64 array ---
    n = 1500
    unit = 8 * n * n
    RATCHET = {'diag': 4.2, 'chol': 5.2, 'tda': 3.2}    # pinned in Task 1 step 4
    for kind, tda in (('diag', False), ('chol', False), ('tda', True)):
        A, B = synthetic(n, 'diag' if kind == 'diag' else 'chol', rng)
        over, _ = peak_over_inputs(lambda: CasidaSolver(A, B).solve(tda=tda), unit)
        all_ok &= check(over <= RATCHET[kind], f'{kind}: peak over caller-held A, B <= {RATCHET[kind]}',
                        f'{over:.2f} arrays')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
