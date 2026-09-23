"""The static exchange of the QP step: every route must agree, and scale.

On the isdf route the QP step used to take Sigma_x from the mean field's own
ISDF-K, which puts the grid's K error into every QP energy at FIRST order: the
v_xc part of that error cancels against the eigenvalue shift, the Sigma_x part
does not (benzene HOMO 8.5 meV; a 178-atom conjugated macrocycle ~30 meV on
every BSE root). The alternatives that do not carry the grid are 'df' --
density fitting with the STORED three-index tensor, 1.6 TB at 4736 basis
functions, so a cross-check only -- and 'df-direct', which streams the
integrals once against the QP states and holds naux x nstates x nao at most.
They are the same RI expression, so they must agree to numerical precision, and
the streamed one must not depend on how the pass is blocked.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto
from threadpoolctl import threadpool_limits

from src.Base.constants import HARTREE_TO_EV
from src.Base.isdf_jk import isdf_jk
from src.SingleReference.GW.qp_solve import (static_exchange_correction,
                                             static_exchange_diagonal,
                                             static_exchange_matrix)

BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; H 1.240 2.147 0; '
           'H -1.240 2.147 0; H -2.479 0 0; H -1.240 -2.147 0; H 1.240 -2.147 0')

if __name__ == '__main__':
    mol = gto.M(atom=BENZENE, basis='cc-pvdz', verbose=0)
    nocc = mol.nelectron // 2
    states = list(range(nocc - 3, nocc + 3))
    mf = isdf_jk(dft.RKS(mol, xc='LRC-WPBEh'), auxbasis='cc-pvdz-ri', n_start=6)
    # C and H at n_start=6 aren't in the shipped table, so this falls through
    # to optimize_atomic_radii's live search (lazily run on first use, inside
    # kernel()) or its git-ignored cache. That search is a numerically
    # differentiated descent under threaded BLAS: a different reduction order
    # moves L-BFGS-B onto a different local minimum (see
    # optimize_atomic_radii's docstring in src/Base/separable_ri.py), and
    # check 4 below moves by tens of meV between minima. Pin BLAS to one
    # thread so the radii search -- and this test -- are reproducible.
    with threadpool_limits(limits=1, user_api='blas'):
        mf.kernel()
    all_ok = True

    # 1. Streamed and stored density fitting are one expression.
    stored = np.diag(static_exchange_matrix(mf, mol, exchange='df'))[states]
    streamed = static_exchange_diagonal(mf, mol, states, exchange='df-direct',
                                        block_memory_gb=4.0)
    d1 = float(np.abs(streamed - stored).max())
    ok1 = d1 < 1e-10
    all_ok &= ok1
    print(f'1. df-direct vs stored-tensor df over {len(states)} states: '
          f'{d1:.2e} Ha   {"OK" if ok1 else "FAIL"}')

    # 2. The pass is a sum over AO blocks; the cut must not change the answer.
    tiny = static_exchange_diagonal(mf, mol, states, exchange='df-direct',
                                    block_memory_gb=0.05)
    d2 = float(np.abs(tiny - streamed).max())
    ok2 = d2 < 1e-12
    all_ok &= ok2
    print(f'2. block size 50 MB vs 4 GB: {d2:.2e} Ha   {"OK" if ok2 else "FAIL"}')

    # 3. The routing: 'mf' through the diagonal entry point is the matrix's
    #    diagonal, and the single-state helper is one element of it.
    mat = np.diag(static_exchange_matrix(mf, mol))[states]
    via = static_exchange_diagonal(mf, mol, states)
    one = static_exchange_correction(mf, mol, states[3], exchange='df-direct')
    d3 = float(np.abs(via - mat).max())
    d3b = abs(one - streamed[3])
    ok3 = d3 < 1e-13 and d3b < 1e-13
    all_ok &= ok3
    print(f"3. 'mf' diagonal == matrix diagonal: {d3:.2e}; single-state helper == "
          f'diagonal[p]: {d3b:.2e}   {"OK" if ok3 else "FAIL"}')

    # 4. And the route effect is real and has the measured size: the ISDF-K
    #    Sigma_x differs from the density-fitted one on the HOMO by several meV.
    h = states.index(nocc - 1)
    route = (via[h] - streamed[h]) * HARTREE_TO_EV * 1e3
    ok4 = 2.0 < abs(route) < 30.0
    all_ok &= ok4
    print(f'4. HOMO <Sigma_x - v_xc>, ISDF-K minus DF: {route:+.1f} meV   '
          f'{"OK" if ok4 else "FAIL"}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
