"""The served distributed Casida and ADC dense solves reproduce the serial ones.

    python tests/test_elpa_casida.py               # serial path only
    mpirun -n 4 python tests/test_elpa_casida.py   # the real check

Needs pyelpa and mpi4py for the distributed cells. Launched on more than one
rank, a cell that fell back to the serial path fails; on one rank either path
passes, so the serial run checks the comparison only. Casida: A-B diagonal (the
in-place branch), A-B full (the Cholesky branch), and TDA; ADC: diag_dense, the
dense solve both ADC drivers call. Eigenvectors are compared after aligning the
sign of every column.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.Base.utils.linearAlgebra.diagonalization import (serve_distributed_solves,
                                                          release_workers)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.ADC import solve as adc_solve

if serve_distributed_solves():
    sys.exit(0)                     # a worker rank, released by rank 0 at the end

try:
    from mpi4py import MPI          # after the setup above, which imports pyelpa first
    n_ranks = MPI.COMM_WORLD.Get_size()
except ImportError:
    n_ranks = 1

N = 300
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


D = np.diag(np.linspace(1.0, 3.0, N))
K = sym(0.02)
cases = {'A-B diagonal': (D + K, K, False),
         'A-B full': (D + K, sym(0.02), False),
         'TDA': (D + K, K, True)}

ok = True
for label, (A, B, tda) in cases.items():
    ref = CasidaSolver(A, B).solve(threshold=10**9, tda=tda)      # serial
    res = CasidaSolver(A, B).solve(threshold=10, tda=tda)         # served
    s = column_signs(ref[1], res[1])
    d_omega = np.max(np.abs(res[0] - ref[0]))
    d_X = np.max(np.abs(res[1] * s - ref[1]))
    d_Y = np.max(np.abs(res[2] * s - ref[2]))
    good = (d_omega < TOL and d_X < TOL and d_Y < TOL
            and (res.is_distributed or n_ranks == 1))
    ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {label}: distributed={res.is_distributed}"
          f" on {n_ranks} rank(s)  |d omega| = {d_omega:.1e}  |dX| = {d_X:.1e}"
          f"  |dY| = {d_Y:.1e}")

# diag_dense does not report its route, so record it where it asks for one.
routes = []
_diagonalize = adc_solve.diagonalize_matrix


def recording_diagonalize(M, threshold=5000):
    """diagonalize_matrix, noting in `routes` whether it ran distributed."""
    out = _diagonalize(M, threshold=threshold)
    routes.append(out[2])
    return out


adc_solve.diagonalize_matrix = recording_diagonalize
NORB = 20
e_ref, Z_ref, V_ref = adc_solve.diag_dense(D + K, NORB, threshold=10**9)
e, Z, V = adc_solve.diag_dense(D + K, NORB, threshold=10)
s = column_signs(V_ref, V)
d_e = np.max(np.abs(e - e_ref))
d_Z = np.max(np.abs(Z - Z_ref))
d_V = np.max(np.abs(V * s - V_ref))
good = d_e < TOL and d_Z < TOL and d_V < TOL and (routes[-1] or n_ranks == 1)
ok &= good
print(f"  [{'ok' if good else 'FAIL'}] ADC diag_dense: distributed={routes[-1]}"
      f" on {n_ranks} rank(s)  |d e| = {d_e:.1e}  |dZ| = {d_Z:.1e}"
      f"  |dV| = {d_V:.1e}")

release_workers()
print('\nAll checks passed.' if ok else '\nFAILURES DETECTED')
sys.exit(0 if ok else 1)
