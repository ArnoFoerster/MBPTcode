"""diagonalize_matrix on the GPU (MBPT_EIGH_DEVICE) reproduces the host solve.

    python tests/test_eigh_device.py

1. MBPT_EIGH_DEVICE: an unknown value raises ValueError; 'cpu', and 'auto' below
   the threshold, never reach the device.
2. Without CuPy or without a visible GPU: 'gpu' raises RuntimeError and 'auto'
   solves on the host without a warning; the device rows print a skip line.
3. With a GPU: 'gpu' on a real symmetric and a complex Hermitian (N, N) matrix,
   eigenvalues to 1e-12 relative to LAPACK's, |M Z - Z diag(w)| / |M| and
   |Z^H Z - 1| below 1e-12, host arrays out; 'auto' above the threshold takes the
   device; the Casida A-B diagonal, A-B full and TDA branches under 'gpu' against
   'cpu', omega to 1e-12 relative, X and Y to 1e-10 after aligning column signs;
   a device out-of-memory or a cuSOLVER refusal under 'auto' falls back to the
   host with one RuntimeWarning, and under 'gpu' raises.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg as la

from src.Base.utils.linearAlgebra import diagonalization as diag
from src.Base.utils.linearAlgebra.diagonalization import diagonalize_matrix
from src.SingleReference.LinearResponse.casida import CasidaSolver

N = 300
rng = np.random.default_rng(3)


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def sym(scale, dtype=float):
    """Symmetric or Hermitian (N, N) matrix, entries of order `scale`."""
    G = rng.standard_normal((N, N)) * scale
    if dtype is complex:
        G = G + 1j * rng.standard_normal((N, N)) * scale
    return (G + G.conj().T) / 2


def solve(device, M, threshold=5000):
    """(w, Z) of diagonalize_matrix under MBPT_EIGH_DEVICE=device."""
    os.environ['MBPT_EIGH_DEVICE'] = device
    w, Z, _, _, _ = diagonalize_matrix(M, threshold=threshold)
    return w, Z


def column_signs(X_ref, X):
    """Per-column sign (phase-free for real vectors) that maps X onto X_ref."""
    s = np.sign(np.sum(X_ref * X, axis=0))
    s[s == 0] = 1.0
    return s[None, :]


keep_env = os.environ.get('MBPT_EIGH_DEVICE')
all_ok = True
M_real = np.diag(np.linspace(1.0, 3.0, N)) + sym(0.02)
w_ref, Z_ref = la.eigh(M_real)

# 1. the switch
try:
    solve('tpu', M_real)
    all_ok &= check(False, 'unknown MBPT_EIGH_DEVICE raises ValueError')
except ValueError:
    all_ok &= check(True, 'unknown MBPT_EIGH_DEVICE raises ValueError')

keep_gpu, calls = diag._eigh_gpu, []
diag._eigh_gpu = lambda *a, **k: calls.append(1)
try:
    w_cpu, _ = solve('cpu', M_real, threshold=10)
    w_low, _ = solve('auto', M_real, threshold=N + 1)
finally:
    diag._eigh_gpu = keep_gpu
all_ok &= check(not calls and np.allclose(w_cpu, w_ref, rtol=0, atol=1e-12)
                and np.allclose(w_low, w_ref, rtol=0, atol=1e-12),
                "'cpu', and 'auto' below the threshold, never reach the device",
                f'{len(calls)} device calls')

cp = diag._cupy()
if cp is None:
    # 2. no CuPy, or no visible GPU
    try:
        solve('gpu', M_real)
        all_ok &= check(False, "'gpu' without a GPU raises RuntimeError")
    except RuntimeError:
        all_ok &= check(True, "'gpu' without a GPU raises RuntimeError")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w_auto, _ = solve('auto', M_real, threshold=10)
    all_ok &= check(np.allclose(w_auto, w_ref, rtol=0, atol=1e-12) and not caught,
                    "'auto' without a GPU solves on the host, silently",
                    f'{len(caught)} warnings')
    print("  [skip] the device rows: CuPy is not importable or sees no GPU")
else:
    # 3. on the device
    name = cp.cuda.runtime.getDeviceProperties(0)['name']
    print(f"  device: {name.decode() if isinstance(name, bytes) else name}, "
          f'CuPy {cp.__version__}')
    for label, M in (('real symmetric', M_real),
                     ('complex Hermitian', np.diag(np.linspace(1.0, 3.0, N))
                      + sym(0.02, complex))):
        w0 = la.eigvalsh(M)
        w, Z = solve('gpu', M)
        scale = np.abs(w0).max()
        dw = np.abs(w - w0).max() / scale
        res = np.abs(M @ Z - Z * w[None, :]).max() / np.abs(M).max()
        orth = np.abs(Z.conj().T @ Z - np.eye(N)).max()
        host = type(w) is np.ndarray and type(Z) is np.ndarray
        all_ok &= check(dw < 1e-12 and res < 1e-12 and orth < 1e-12 and host,
                        f"'gpu' {label}, eigenpairs against LAPACK, host arrays",
                        f'eigenvalues {dw:.1e} rel, residual {res:.1e}, '
                        f'orthonormality {orth:.1e}')

    keep_eigh, used = cp.linalg.eigh, []
    cp.linalg.eigh = lambda a: (used.append(1), keep_eigh(a))[1]
    try:
        w_auto, _ = solve('auto', M_real, threshold=10)
    finally:
        cp.linalg.eigh = keep_eigh
    all_ok &= check(len(used) == 1 and np.abs(w_auto - w_ref).max() < 1e-12,
                    "'auto' above the threshold takes the device",
                    f'{len(used)} device eigh calls')

    D = np.diag(np.linspace(1.0, 3.0, N))
    K = sym(0.02)
    for label, (A, B, tda) in {'A-B diagonal': (D + K, K, False),
                               'A-B full': (D + K, sym(0.02), False),
                               'TDA': (D + K, K, True)}.items():
        os.environ['MBPT_EIGH_DEVICE'] = 'cpu'
        ref = CasidaSolver(A.copy(), B.copy()).solve(threshold=10, tda=tda)
        os.environ['MBPT_EIGH_DEVICE'] = 'gpu'
        got = CasidaSolver(A.copy(), B.copy()).solve(threshold=10, tda=tda)
        s = column_signs(ref[1], got[1])
        domega = np.abs(got[0] - ref[0]).max() / np.abs(ref[0]).max()
        dvec = max(np.abs(got[1] * s - ref[1]).max(), np.abs(got[2] * s - ref[2]).max())
        all_ok &= check(domega < 1e-12 and dvec < 1e-10,
                        f"Casida {label}: 'gpu' against 'cpu'",
                        f'omega {domega:.1e} rel, X and Y {dvec:.1e}')

    def refusing(exc):
        def eigh(a):
            raise exc
        return eigh

    # INVALID_VALUE is what Xsyevd raises on a size it rejects
    for label, exc in (('a device out-of-memory',
                        cp.cuda.memory.OutOfMemoryError(M_real.nbytes, 0)),
                       ('a cuSOLVER refusal', cp.cuda.cusolver.CUSOLVERError(3))):
        cp.linalg.eigh = refusing(exc)
        try:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    w_fall, _ = solve('auto', M_real, threshold=10)
                runtime = [c for c in caught
                           if issubclass(c.category, RuntimeWarning)]
                ok = np.abs(w_fall - w_ref).max() < 1e-12 and len(runtime) == 1
                detail = (f'{len(runtime)} RuntimeWarning'
                          + (f': {runtime[0].message}' if runtime else ''))
            except type(exc) as e:
                ok, detail = False, f'{type(e).__name__} escaped: {e}'
            all_ok &= check(ok, f"'auto' falls back to the host on {label}", detail)
            try:
                solve('gpu', M_real)
                all_ok &= check(False, f"'gpu' raises on {label}")
            except type(exc):
                all_ok &= check(True, f"'gpu' raises on {label}")
        finally:
            cp.linalg.eigh = keep_eigh

if keep_env is None:
    os.environ.pop('MBPT_EIGH_DEVICE', None)
else:
    os.environ['MBPT_EIGH_DEVICE'] = keep_env
print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
sys.exit(0 if all_ok else 1)
