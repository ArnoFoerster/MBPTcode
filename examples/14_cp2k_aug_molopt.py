"""BSE@G0W0@PBE of formaldehyde in CP2K's aug-SZV-MOLOPT-ae basis with its RI tier.

The orbital and auxiliary sets are read from CP2K at run time (see
src/Base/basis/cp2k_basis.py for the source, the license notice and the cache);
nothing is shipped with MBPTcode. By default `register` takes the tightest RI tier
of every element; this example asks for the smallest tier within the paper's
Delta-I threshold of 1e-4, and it serves the mean field, W and the BSE kernel
alike. Delta-I is an MP2 criterion: against the exact 4-center tensor this tier
leaves 27 meV on the lowest singlets and 6 meV on HOMO and LUMO, the tightest tiers
1.6 meV and under 1 meV (see the module docstring and `min_lmax` for the angular
rule).

    python examples/14_cp2k_aug_molopt.py
"""
import os
import sys

import numpy as np
from pyscf import dft, gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV
from src.Base.basis.cp2k_basis import pick_ri_tier, register
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

ATOMS = 'H 0 0.934473 -0.588078; H 0 -0.934473 -0.588078; C 0 0 0; O 0 0 1.221104'
BASIS, ELEMENTS, MAX_ERROR = 'aug-SZV-MOLOPT-ae', ['H', 'C', 'O'], 1e-4

for el in ELEMENTS:
    name, n, err, pat = pick_ri_tier(BASIS, el, MAX_ERROR)
    print(f'{el}: {n} auxiliary functions, Delta-I {err:.1e}, {pat}')
basis, aux = register(BASIS, max_error=MAX_ERROR)
mol = gto.M(atom=ATOMS, basis=basis, verbose=0)
mf = dft.RKS(mol, xc='PBE').density_fit(auxbasis=aux)
mf.kernel()
nocc = mol.nelectron // 2
print(f'{mol.nao} functions, {mf.with_df.auxmol.nao} auxiliary, E = {mf.e_tot:.6f} Ha')

qp = calc_qp_energy(mf, state=list(range(mol.nao)))
eps_qp = np.array([qp[p]['GW'] for p in range(mol.nao)]) / HARTREE_TO_EV
print(f'G0W0 HOMO = {qp[nocc - 1]["GW"]:.3f} eV   LUMO = {qp[nocc]["GW"]:.3f} eV')

coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
eps_ks = get_orbital_energies(mf, representation='spatial')
w_aux = LinearResponseSolver(eps_ks, coeff_df=coeff,
                             spin_mode='restricted').static_screening_aux(nocc)
A, B = LinearResponseSolver(eps_qp, coeff_df=coeff, spin_mode='restricted') \
    .build_casida_matrices(nocc, lBSE=True, W_aux=w_aux)
omega = np.sort(CasidaSolver(A, B).solve()[0])[:3] * HARTREE_TO_EV
print('BSE@G0W0 lowest singlets = ' + ', '.join(f'{w:.3f}' for w in omega) + ' eV')
