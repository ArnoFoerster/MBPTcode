"""All the ways one can run a BSE calculation:

    SOLVER      Davidson  never forms A or B, returns the few lowest roots
                dense     builds A and B, returns the whole spectrum, O(N^6)

    INTEGRALS   ISDF      the separable factorization's own three-index factor
                DF        pyscf's cderi
                full      the 4-center ERI tensor

Five of those six combinations run; Davidson with full ERIs is the exception.

Both three-index routes are fitted in a COULOMB-fitting auxiliary basis rather
than the MP2 correlation-fitting <basis>-ri, because the BSE direct term
contracts (ij|ab), a contraction that does not appear in MP2. With the -ri
set, the lowest excitation energy moves by 21 meV between DF and the exact
tensor at cc-pVDZ and by 6 meV at cc-pVTZ; with the -jkfit set used here, by
2 meV.

    python examples/11_bse_davidson_vs_dense.py
"""
import os
import sys

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV, ISDF_GRID_ACCURACY
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.BSE.solve import isdf_df_coefficients, solve_bse_isdf
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

NROOTS = 4
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvtz', verbose=0)
nocc = mol.nelectron // 2

# One auxiliary basis for the mean field and the separable fit
AUXBASIS = str(mol.basis) + '-jkfit'

mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
mf.kernel()
ev = lambda w: ', '.join(f'{x * HARTREE_TO_EV:.3f}' for x in np.sort(w)[:NROOTS])

# # Default: GW space-time using ISDF, followed by Davidson
# The grid is named: the G1 shell counts of cc-pVTZ. The radii table holds no
# row for the jkfit auxiliary set, so the radii are optimized at run time,
# from six starts; that has to be declared (runtime_radii), or it is refused.
COUNTS = dict(zip(('A1', 'A2', 'A3', 'B1'), ISDF_GRID_ACCURACY['cc-pvtz']['G1']))
numerics = Numerics(isdf={'runtime_radii': True}, bse={'nroots': NROOTS})
omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, numerics=numerics, probe='sign',
                                   auxbasis=AUXBASIS, n_start=6, counts=COUNTS)
eps_qp, W_isdf = info['eps'], info['W_aux']
X_mo, D = info['factors'][:2]

######### The more indirect routes ############### 

# get fit coefficients from mean field
B_df = get_density_fitting_coefficients(mol, mf, representation='spatial')
eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

# construct W
W_df = LinearResponseSolver(info['eps_mf'], coeff_df=B_df,spin_mode='restricted').static_screening_aux(nocc)
W_full = LinearResponseSolver(info['eps_mf'], eri_chemist=eri,spin_mode='restricted').static_screening_aux(nocc)

# Initialize linear response solvers

# ISDF
resp_isdf = LinearResponseSolver(eps_qp, spin_mode='restricted',
                                 coeff_df=isdf_df_coefficients(X_mo, D))

# DF
resp_df = LinearResponseSolver(eps_qp, spin_mode='restricted', coeff_df=B_df)

# Full integrals
resp_full = LinearResponseSolver(eps_qp, spin_mode='restricted', eri_chemist=eri)

# Davidson, using ISDF
dav_isdf = solve_casida_davidson(resp_isdf, nocc, numerics=numerics, polarizability='BSE',
                                 W_aux=W_isdf, isdf_factors=(X_mo, D))[0]

# Davidson, using standard DF
dav_df = solve_casida_davidson(resp_df, nocc, numerics=numerics, polarizability='BSE',
                               W_aux=W_df)[0]

# Dense: build A and B, then diagonalize
A_isdf, B_isdf = resp_isdf.build_casida_matrices(nocc, lBSE=True, W_aux=W_isdf)
dense_isdf = CasidaSolver(A_isdf, B_isdf).solve()[0]
A_df, Bmat_df = resp_df.build_casida_matrices(nocc, lBSE=True, W_aux=W_df)
dense_df = CasidaSolver(A_df, Bmat_df).solve()[0]
A_full, B_full = resp_full.build_casida_matrices(nocc, lBSE=True, W_aux=W_full,eps_screen=info['eps_mf'])
dense_full = CasidaSolver(A_full, B_full).solve()[0]

for label, w in (('Davidson, ISDF', dav_isdf),
                 ('Davidson, DF  ', dav_df),
                 ('dense,    ISDF', dense_isdf),
                 ('dense,    DF  ', dense_df),
                 ('dense,    full', dense_full)):
    print(f'BSE@G0W0   {label}   {ev(w)} eV')
