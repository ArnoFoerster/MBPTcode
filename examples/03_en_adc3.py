"""EN-ADC(3): Epstein-Nesbet denominators, spin-adapted. The production variant.

ONE dict configures the whole run: the solver dresses the U^(2) amplitude
denominators, the static builder dresses the MP2 density ('singles' is
consumed there). hh/pp: which ladder dresses; shift: 'sum' (2J-K) is
calibrated best, 'mean' (J-K/2) and 'opposite' (J) are the alternatives."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.ADC import ADCSolver, build_static_correction


DRESS = {'hh': True, 'pp': True, 'singles': False,
         'spin_adapted': True, 'shift': 'sum'}

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

solver = ADCSolver(mf, level='adc3', en_dress=DRESS)
static = build_static_correction(mf, kind='mp2_relaxed', en_dress=DRESS)

eGF, Z = solver.solve(static_correction=static)
print(f'EN-ADC(3) IP = {-eGF[0] * HARTREE_TO_EV:.3f} eV   Z = {Z[0]:.3f}')
