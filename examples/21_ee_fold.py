"""The folded EE solver: ADC(2), the GF2 supermatrix and the one-doubles-set
BSE@GW of Monino and Loos, J. Chem. Phys. 159, 034105 (2023), eqs 53 and 66, each
root solved on the singles space at its own frequency, the doubles never stored;
the folded ADC(2) roots beside the full solve, which carries the doubles."""
import os
import sys

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_fold, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.SingleReference.ADC.eeADC.ee_gw_pieces import build_pieces_gw

HARTREE_TO_EV = 27.211386245988
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit().run()
eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
B = DFIntegrals.from_scf(mol, mf).B_aa
nocc = mol.nelectron // 2

# the pieces of the fold: the singles block M, the coupling V and its transpose,
# the doubles diagonal D
pieces = {
    'adc2': ee_r_sigma_df.build_operator(eps, B, nocc, level='adc2',
                                         pieces=True)[3],
    'gf2': ee_r_sigma_df.build_operator(eps, B, nocc, level='gf2',
                                        pieces=True)[3],
    'gw, eq 66, TDA screening': build_pieces_gw(eps, B, nocc, screening='tda'),
}
for name, P in pieces.items():
    for spin in ('singlet', 'triplet'):
        res = ee_fold.solve_folded(P, 3, spin=spin)
        print(f'{name}, {spin}: ' + '   '.join(
            f'{w * HARTREE_TO_EV:.3f} eV (T1 {t1:.3f})'
            for w, t1 in zip(res.omega, res.t1)))

# the folded ADC(2) roots equal the full solve
e_full, _ = solve_ee_adc(mf, level='adc2', df=True, nroots=3, spin='singlet')
print('adc2, singlet, full solve: ' + '   '.join(
    f'{w * HARTREE_TO_EV:.3f} eV' for w in e_full))

# fold=True: the same fold with every doubles-shaped quantity built a tile at a
# time from the DF factors (ee_ri_fold), the route for molecules of a thousand
# basis functions; here with the 1s core frozen and the correlation fit
e_tiled, _ = solve_ee_adc(mf, level='adc2', fold=True, nroots=3, spin='singlet',
                          frozen=1, auxbasis='cc-pvdz-ri')
print('adc2, singlet, tiled fold, frozen 1s: ' + '   '.join(
    f'{w * HARTREE_TO_EV:.3f} eV' for w in e_tiled))

# in toluene: the SCF relaxes at the static dielectric constant, the singlet
# singles block takes the optical reaction field of the transition density,
# and the second-order terms keep the bare interaction
from src.Base.solvent_screening import SolventScreening      # noqa: E402

env = SolventScreening(mol, solvent='toluene')
mf_tol = env.mean_field(mol, lambda m: scf.RHF(m).density_fit().run())
e_tol, _ = solve_ee_adc(mf_tol, level='adc2', fold=True, nroots=3,
                        spin='singlet', frozen=1, auxbasis='cc-pvdz-ri',
                        environment=env)
print('adc2, singlet, toluene: ' + '   '.join(
    f'{w * HARTREE_TO_EV:.3f} eV' for w in e_tol))
