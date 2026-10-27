"""Cubic-scaling GW and BSE on one ISDF factorization.

The route for a large molecule: the SCF, the GW and the BSE all run on the SAME
factors. Reusing them is not only cheaper -- a second fit comes back in a
different gauge, and mixing gauges gives a wrong answer that still converges.

Two arguments of `solve_bse_isdf` are easy to confuse:

    factors=  the ISDF fit ONLY. Says nothing about whether a GW has run.
    qp=       the BSE diagonal. 'G0W0' (default) runs the GW itself; an array
              of orbital energies uses yours; False gives BSE@mean-field.

W is never passed: it is always built at the mean-field energies, which is what
G0W0-BSE means.

    python examples/09_isdf_gw_space_time.py
"""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV, SURFACE_GRID_ACCURACY
from src.Base.factorization.isdf_jk import separable_factors_from_jk
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.space_time import (solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)
from src.SingleReference.BSE.solve import solve_bse_isdf

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
nocc = mol.nelectron // 2

# 1. SCF with an ISDF with_df: J and K from the factors, no cderi stored.
#    The interpolation grid is named, here the level the surfaces use.
counts = resolve_isdf_grid(SURFACE_GRID_ACCURACY, str(mol.basis))[0]
mf = isdf_jk_with_force(scf.RHF(mol), auxbasis='cc-pvdz-ri', counts=counts)
mf.kernel()
print(f'ISDF-SCF   E = {mf.e_tot:.8f} Ha')

# 2. Hand the GW and BSE the factorization the SCF already built.
factors = separable_factors_from_jk(mf)

# 3. G0W0 for the states you want. One self-energy serves a whole window.
qp = solve_qp_energy_space_time(mf, mol, nocc, [nocc - 2, nocc - 1],
                                factors=factors)
print('G0W0       HOMO-1, HOMO = '
      + ', '.join(f'{e * HARTREE_TO_EV:.3f}' for e in qp) + ' eV')

# 4a. BSE in two steps: the QP diagonal first, then the BSE on it.
eps_qp, _ = solve_qp_diagonal_space_time(mf, mol, nocc, factors=factors)  # every orbital
numerics = Numerics(bse={'nroots': 3})     # every numerical choice in one object
omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, numerics=numerics, qp=eps_qp,
                                   factors=factors)
print('BSE@G0W0   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega) + ' eV')

# 4b. Or in one call, which runs that same GW itself.
omega1, _, _, _ = solve_bse_isdf(mf, mol, nocc, numerics=numerics, factors=factors)
print('one call   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega1) + ' eV')
