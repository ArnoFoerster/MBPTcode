"""Electronic-excitation ADC(2)/ADC(3): the polarization-propagator route,
spin-free, matrix-free, density-fitted -- the production path."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

HARTREE_TO_EV = 27.211386245988
basis = 'aug-cc-pvdz'
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis=basis, verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-ri').run()

# singlet and triplet: one operator, each channel solved in the flip-pair
# basis of the alpha<->beta involution
print(basis)
for level in ('adc2', 'adc3'):
    e_s, _ = solve_ee_adc(mf, level=level, df=True, nroots=1, spin='singlet')
    e_t, _ = solve_ee_adc(mf, level=level, df=True, nroots=1, spin='triplet')
    print(f'{level}: S1 = {e_s[0] * HARTREE_TO_EV:.3f} eV   '
          f'T1 = {e_t[0] * HARTREE_TO_EV:.3f} eV   '
          f'E_ST = {(e_s[0] - e_t[0]) * HARTREE_TO_EV:.3f} eV')
