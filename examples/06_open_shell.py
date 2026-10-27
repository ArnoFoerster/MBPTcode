"""Open-shell (UHF) ADC(3) IP -- dispatches to the spin-orbital branch."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.ADC import ADCSolver, build_static_correction


mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1, verbose=0)
mf = scf.UHF(mol).run()

solver = ADCSolver(mf, level='adc3')       # UHF -> spin-orbital branch
static = build_static_correction(mf, kind='mp2_relaxed')

eGF, Z = solver.solve(static_correction=static)
print(f'UHF ADC(3) IP = {-eGF[0] * HARTREE_TO_EV:.3f} eV   Z = {Z[0]:.3f}')
