"""BSE@G0W0 needs two energy sets, and the 4-center builder has to be told.

The BSE diagonal carries the quasiparticle energies while W is screened at the
mean-field ones -- that split is what G0W0-BSE means. A three-index solver gets
it by construction, because its whole kernel comes from the W_aux it is handed.
The 4-center builder does not: it rebuilds the direct term's polarizability
itself and, before `eps_screen`, could only do so at the energies on its own
diagonal. Screening at the quasiparticle energies instead moves these roots by
24 meV -- larger than the auxiliary basis error it would be mistaken for, and
flat under grid refinement, since it is not a fitting error at all.

So the pins are: the default is unchanged, the argument reaches the direct term,
the two spellings of "screen at my own energies" agree to a few ulp (two
builds), and once the
split is right the residual gap to density fitting is the fitting set alone --
the same size whether it is measured here or at a single energy set.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_MEV
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.numerics import numerics_of
from tests.isdf_test_grid import g1
from tests.pins import RECORDING_BLAS, matches

NROOTS = 4

if __name__ == '__main__':
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='cc-pvdz', verbose=0)
    nocc = mol.nelectron // 2
    # cc-pVDZ-RI on purpose: it is an MP2 correlation-fitting set, so its error
    # on the BSE direct term is large enough (21 meV) to be measured twice over
    # in check 4. A Coulomb-fitting set would put it near the solver threshold.
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    info = solve_bse_isdf(mf, mol, nocc, probe='sign', counts=g1(mf),
                          numerics=numerics_of({'nroots': NROOTS}))[3]
    eps_qp, eps_mf = info['eps'], info['eps_mf']

    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    full = lambda eps: LinearResponseSolver(eps, eri_chemist=eri,
                                            spin_mode='restricted')
    df = lambda eps: LinearResponseSolver(eps, coeff_df=coeff,
                                          spin_mode='restricted')
    W_full = full(eps_mf).static_screening_aux(nocc)
    W_df = df(eps_mf).static_screening_aux(nocc)
    resp = full(eps_qp)
    roots = lambda A, B: np.sort(CasidaSolver(A, B).solve()[0])[:NROOTS]
    all_ok = True

    # 1. Omitting the argument must leave the pre-existing route untouched, and
    #    naming the diagonal's own energies must be the same computation.
    base = resp.build_casida_matrices(nocc, lBSE=True, W_aux=W_full)
    same = resp.build_casida_matrices(nocc, lBSE=True, W_aux=W_full,
                                      eps_screen=eps_qp)
    ok1 = all(matches(a, b, RECORDING_BLAS, quantity='root')
              for a, b in zip(base, same))
    all_ok &= ok1
    print(f'1. eps_screen=None and =self.eps agree to a few ulp   '
          f'{"OK" if ok1 else "FAIL"}')

    # 2. And the argument has to reach the direct term, i.e. the A matrix, not
    #    only the swap term that W_aux already fed.
    split = resp.build_casida_matrices(nocc, lBSE=True, W_aux=W_full,
                                       eps_screen=eps_mf)
    d = (roots(*split) - roots(*base)) * HARTREE_TO_MEV
    shift = d[np.argmax(np.abs(d))]
    ok2 = 10.0 < abs(shift) < 60.0
    all_ok &= ok2
    print(f'2. screening at eps_mf instead of eps_qp moves the roots by '
          f'{shift:+.1f} meV   {"OK" if ok2 else "FAIL"}')

    # 3. It is meaningless for a three-index solver and unimplemented for the
    #    unrestricted block; both must say so rather than silently ignore it.
    ok3 = True
    try:
        df(eps_qp).build_casida_matrices(nocc, lBSE=True, W_aux=W_df,
                                         eps_screen=eps_mf)
        ok3 = False
    except ValueError:
        pass
    try:
        LinearResponseSolver((eps_qp, eps_qp), eri_chemist=(eri, eri, eri),
                             spin_mode='unrestricted').build_casida_matrices(
            (nocc, nocc), lBSE=True, W_aux=W_full, eps_screen=(eps_mf, eps_mf))
        ok3 = False
    except NotImplementedError:
        pass
    all_ok &= ok3
    print(f'3. rejected for a three-index and for an unrestricted solver   '
          f'{"OK" if ok3 else "FAIL"}')

    # 4. With the split right, what separates the two integral routes is the
    #    auxiliary basis and nothing else -- so it must not depend on which
    #    energies the measurement is taken at.
    gap_qp = abs(roots(*split) - roots(*df(eps_qp).build_casida_matrices(
        nocc, lBSE=True, W_aux=W_df))).max() * HARTREE_TO_MEV
    gap_mf = abs(roots(*full(eps_mf).build_casida_matrices(
        nocc, lBSE=True, W_aux=W_full))
        - roots(*df(eps_mf).build_casida_matrices(
            nocc, lBSE=True, W_aux=W_df))).max() * HARTREE_TO_MEV
    ok4 = abs(gap_qp - gap_mf) < 2.0
    all_ok &= ok4
    print(f'4. DF-to-full gap is the fitting set: {gap_qp:.1f} meV at G0W0 vs '
          f'{gap_mf:.1f} meV at one eps   {"OK" if ok4 else "FAIL"}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
