import os
import sys
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.constants import get_method_info
from src.SingleReference.GW.qp_energy import _casida_spectrum
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" +
          (f'   ({detail})' if detail else ''))
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
    """Traced peak during fn() minus the traced size before it, in units of
    `unit` bytes."""
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
            all_ok &= check(r1 < 1e-9 and r2 < 1e-9, f'{kind}: Casida residuals',
                            f'{r1:.1e}, {r2:.1e}')
        nrm = np.max(np.abs(X.T @ X - Y.T @ Y - np.eye(len(omega))))
        # 'fallback' takes the shifted-Cholesky branch, whose normalization
        # residual runs a few x higher than diag/chol's.
        nrm_tol = 1e-6 if kind == 'fallback' else 1e-9
        all_ok &= check(nrm < nrm_tol, f'{kind}: X^T X - Y^T Y = 1', f'{nrm:.1e}')

    # --- keep_intermediates gates the instance attributes, not the result ---
    A, B = synthetic(120, 'chol', rng)
    s_default = CasidaSolver(A, B)
    res_default = s_default.solve()
    s_keep = CasidaSolver(A, B, keep_intermediates=True)
    res_keep = s_keep.solve()
    all_ok &= check(s_default.Z is None, 'default: Z not stored')
    all_ok &= check(s_keep.Z is not None and s_keep.Z.shape == A.shape,
                    'keep_intermediates: Z stored')
    all_ok &= check(all(np.array_equal(a, b) for a, b in zip(res_default, res_keep)),
                    'keep_intermediates does not change omega, X, Y')
    all_ok &= check(s_default.A is None and s_default.B is None,
                    'default: A, B released by solve')
    try:
        s_default.solve()
        raised = False
    except RuntimeError:
        raised = True
    all_ok &= check(raised, 'default: a second solve() raises RuntimeError')
    all_ok &= check(s_keep.A is not None and s_keep.B is not None,
                    'keep_intermediates: A, B kept')

    # --- tracemalloc ratchets at n = 1500, in units of one n x n float64 array ---
    n = 1500
    unit = 8 * n * n
    RATCHET = {'diag': 4.2, 'chol': 5.2, 'tda': 3.2}
    for kind, tda in (('diag', False), ('chol', False), ('tda', True)):
        A, B = synthetic(n, 'diag' if kind == 'diag' else 'chol', rng)
        over, _ = peak_over_inputs(lambda: CasidaSolver(A, B).solve(tda=tda), unit)
        all_ok &= check(over <= RATCHET[kind],
                        f'{kind}: peak over caller-held A, B <= {RATCHET[kind]}',
                        f'{over:.2f} arrays')

    # --- ownership handoff: nothing but the solver holds A, B during the solve ---
    def handoff_peak(kind, tda, unit):
        """Traced peak of solve() alone, A and B included, in units of `unit`
        bytes: synthetic()'s own construction memory is excluded by starting
        the peak window after the inputs exist, on a box the solver empties."""
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            box = [synthetic(n, kind, rng)]
            tracemalloc.reset_peak()
            CasidaSolver(*box.pop()).solve(tda=tda)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return (peak - base) / unit
    RATCHET_HANDOFF = {'diag': 4.1, 'chol': 5.1, 'tda': 4.1}
    for kind, tda in (('diag', False), ('chol', False), ('tda', True)):
        over = handoff_peak('diag' if kind == 'diag' else 'chol', tda, unit)
        all_ok &= check(
            over <= RATCHET_HANDOFF[kind],
            f'{kind}: peak with A, B handed to the solver <= {RATCHET_HANDOFF[kind]}',
            f'{over:.2f} arrays, A and B included')

    # --- tracemalloc ratchet on _casida_spectrum's singlet solve, same
    # synthetic DF system as test_casida_build_inplace.py's build ratchet ---
    naux_s, norb_s, nocc_s = 400, 100, 30
    n_pair_s = nocc_s * (norb_s - nocc_s)
    spec_rng = np.random.default_rng(0)
    coeff_s = spec_rng.standard_normal((naux_s, norb_s, norb_s))
    coeff_s = coeff_s + coeff_s.transpose(0, 2, 1)
    eps_s = np.sort(spec_rng.uniform(-1.0, 1.0, norb_s))
    lr_s = LinearResponseSolver(eps_s, coeff_df=coeff_s, spin_mode='restricted')
    methods_s = ['GW']
    method_infos_s = {'GW': get_method_info('GW')}
    spec_unit = 8 * n_pair_s * n_pair_s
    RATCHET_SPECTRUM = 4.2

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        spectrum = _casida_spectrum(lr_s, nocc_s, 'RPA', None, False,
                                    method_infos_s, methods_s, False, True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    del spectrum
    over = (peak - base) / spec_unit
    all_ok &= check(over <= RATCHET_SPECTRUM,
                    f'_casida_spectrum RPA singlet: peak <= {RATCHET_SPECTRUM}',
                    f'{over:.2f} arrays')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
