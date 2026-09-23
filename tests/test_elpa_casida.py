"""The distributed Casida and ADC dense solves reproduce the serial ones.

    python tests/test_elpa_casida.py               # serial path only
    mpirun -n 4 python tests/test_elpa_casida.py   # the real check

Needs pyelpa and mpi4py for the distributed cells. Launched on more than one
rank, a cell that fell back to the serial path fails; on one rank either path
passes, so the serial run checks the comparisons only. Two phases. Unserved,
every rank solves, as a driver that never parks its workers does: rank 0 must
get the serial result, every other rank the eigenvalues and None for the rest.
Served, ranks > 0 park in serve_distributed_solves and rank 0 solves alone.
Cases: Casida A-B diagonal (the in-place branch), A-B full (the Cholesky
branch) and TDA, and ADC's diag_dense, the dense solve both ADC drivers call.
Served only: a second solve on a solver that keeps its intermediates, and a
complex Hermitian matrix, which must stay on the local solve. Eigenvectors are
compared after aligning the sign of every column.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.Base.utils.linearAlgebra.diagonalization import (diagonalize_matrix,
                                                          serve_distributed_solves,
                                                          release_workers)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.ADC import solve as adc_solve

N = 300
NORB = 20
TOL = 1e-8
rng = np.random.default_rng(1)


def sym(scale):
    """Symmetric (N, N) matrix from the module rng, entries of order `scale`."""
    G = rng.standard_normal((N, N)) * scale
    return (G + G.T) / 2


def column_signs(X_ref, X):
    """Per-column sign that maps X onto X_ref; applied to X and Y alike."""
    s = np.sign(np.sum(X_ref * X, axis=0))
    s[s == 0] = 1.0
    return s[None, :]


# every rank draws the same matrices; rank 0's are the ones a solve reads
D = np.diag(np.linspace(1.0, 3.0, N))
K = sym(0.02)
cases = {'A-B diagonal': (D + K, K, False),
         'A-B full': (D + K, sym(0.02), False),
         'TDA': (D + K, K, True)}

# diag_dense does not report its route, so record it where it asks for one.
routes = []
_diagonalize = adc_solve.diagonalize_matrix


def recording_diagonalize(M, threshold=5000):
    """diagonalize_matrix, noting in `routes` whether it ran distributed."""
    out = _diagonalize(M, threshold=threshold)
    routes.append(out[2])
    return out


adc_solve.diagonalize_matrix = recording_diagonalize


def solve_all(threshold):
    """{case: (values, vectors, second block, distributed)} at `threshold`.

    Casida gives (omega, X, Y); ADC gives (poles, eigenvectors, pole strengths).
    """
    out = {}
    for label, (A, B, tda) in cases.items():
        r = CasidaSolver(A, B).solve(threshold=threshold, tda=tda)
        out[label] = (r[0], r[1], r[2], r.is_distributed)
    e, Z, V = adc_solve.diag_dense(D + K, NORB, threshold=threshold)
    out['ADC diag_dense'] = (e, V, Z, routes[-1])
    return out


def deviations(ref, res):
    """Largest |res - ref| of the values, the vectors and the second block."""
    s = column_signs(ref[1], res[1])
    s2 = s if res[2].ndim == 2 else 1.0             # pole strengths carry no sign
    return (np.max(np.abs(res[0] - ref[0])), np.max(np.abs(res[1] * s - ref[1])),
            np.max(np.abs(res[2] * s2 - ref[2])))


def check(ok, label, detail=''):
    """Print one verdict line; returns ok as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{tail}")
    return bool(ok)


ref = solve_all(10**9)                                # serial, on every rank
unserved = solve_all(10)                              # every rank solves
try:
    from mpi4py import MPI      # after a solve above its threshold imported pyelpa
    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()
except ImportError:
    comm, rank, n_ranks = None, 0, 1

# ranks > 0: the eigenvalues, None for everything else, and the distributed flag
mine = all(np.max(np.abs(r[0] - ref[k][0])) < TOL and r[1] is None
           and r[2] is None and r[3] for k, r in unserved.items()) if rank else True
workers_ok = comm.allreduce(mine, op=MPI.LAND) if n_ranks > 1 else True

if serve_distributed_solves():
    sys.exit(0)                     # a worker rank, released by rank 0 at the end

all_ok = True
on = f"on {n_ranks} rank(s)"
for phase, results in (('unserved', unserved), ('served', solve_all(10))):
    for label, res in results.items():
        d = deviations(ref[label], res)
        all_ok &= check(max(d) < TOL and (res[3] or n_ranks == 1),
                        f"{phase} {label}: distributed={res[3]} {on}",
                        'max |d| values, vectors, second block = '
                        + ', '.join(f'{x:.1e}' for x in d))
all_ok &= check(workers_ok, f"unserved, ranks > 0: eigenvalues, None otherwise {on}")

A, B, _ = cases['A-B full']
kept, kept_ref = (CasidaSolver(A, B, keep_intermediates=True) for _ in range(2))
first_ref = kept_ref.solve(threshold=10**9)
first, second = kept.solve(threshold=10), kept.solve(threshold=10)
s = column_signs(kept_ref.Z, kept.Z)
d_Z = np.max(np.abs(kept.Z * s - kept_ref.Z))
d_2 = max(np.max(np.abs(second[0] - first_ref[0])),
          np.max(np.abs(second[1] * column_signs(first_ref[1], second[1])
                        - first_ref[1])))
all_ok &= check(d_Z < TOL and d_2 < TOL and (second.is_distributed or n_ranks == 1),
                f"keep_intermediates, two solves: distributed={second.is_distributed}",
                f'|dZ| = {d_Z:.1e}, second solve |d| = {d_2:.1e}')

G = rng.standard_normal((N, N)) * 0.02
H = D + K + 1j * (G - G.T) / 2                        # Hermitian, not real
w_ref = diagonalize_matrix(H, threshold=10**9)[0]
w, _, distributed, _, _ = diagonalize_matrix(H, threshold=10)
d_w = np.max(np.abs(w - w_ref))
all_ok &= check(not distributed and d_w < TOL,
                f"complex Hermitian stays local: distributed={distributed}",
                f'|d w| = {d_w:.1e}')

release_workers()
print('\nAll checks passed.' if all_ok else '\nFAILURES DETECTED')
sys.exit(0 if all_ok else 1)
