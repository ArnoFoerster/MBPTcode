"""
ADCSolverUnrestricted.downfolded_seeds: on-shell non-diagonal downfold
seeds (solve.downfolded_seed_vectors) for root-following toward orbital-mixed
states. Checks: seeds orthonormal; every seeded matrix-free solve lands on a
dense eigenpair; the union of seeded solves contains the dense lowest
Z>=0.05 IP state; the EN-dressed route is exercised.
"""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC import ADCSolver

Z_MIN = 0.05


@lru_cache(maxsize=None)
def oh_solvers():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    dense = ADCSolver(mf, mol=mol, level='adc3', matrix_free=False)
    e_d, Z_d = dense.solve()
    mfree = ADCSolver(mf, mol=mol, level='adc3', matrix_free=True)
    return mf, mol, np.asarray(e_d), np.asarray(Z_d), mfree


def test_seeds_orthonormal_and_embedded():
    _, _, _, _, solver = oh_solvers()
    e_eff, seeds = solver.downfolded_seeds()
    assert seeds.shape[1] == solver.nocc == e_eff.size
    G = seeds.T @ seeds
    assert np.max(np.abs(G - np.eye(solver.nocc))) < 1e-12
    # zeros outside the orbital window (occupied rows only by default)
    assert np.max(np.abs(seeds[solver.nocc:])) == 0.0
    assert np.all(np.diff(e_eff) >= 0)


def test_every_seeded_solve_hits_a_dense_eigenpair():
    _, _, e_d, Z_d, solver = oh_solvers()
    _, seeds = solver.downfolded_seeds()
    found = []
    for k in range(seeds.shape[1]):
        e, Z = solver.solve(nroots=1, ref_vec=seeds[:, k], tol=1e-8)
        j = int(np.argmin(np.abs(e_d - float(e[0]))))
        assert abs(e_d[j] - float(e[0])) < 5e-6
        assert abs(Z_d[j] - float(Z[0])) < 1e-4
        found.append(j)
    # the dense lowest Z>=Z_MIN IP state is among the seeded results
    ip = np.where((e_d < 0) & (Z_d >= Z_MIN))[0]
    lowest = ip[np.argmax(e_d[ip])]
    assert lowest in found


def test_en_dressed_route():
    mf, mol, _, _, _ = oh_solvers()
    en = ADCSolver(mf, mol=mol, level='adc3', matrix_free=True,
                   en_dress={'hh': True, 'singles': False})
    e_eff, seeds = en.downfolded_seeds()
    e, Z = en.solve(nroots=1, ref_vec=seeds[:, -1], tol=1e-7)
    assert e.shape == (1,) and 0.0 < float(Z[0]) <= 1.0
    assert float(e[0]) < 0.0


if __name__ == '__main__':
    test_seeds_orthonormal_and_embedded()
    test_every_seeded_solve_hits_a_dense_eigenpair()
    test_en_dressed_route()
    print('OK')
