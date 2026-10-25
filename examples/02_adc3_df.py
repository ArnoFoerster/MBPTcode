"""ADC(3) with density fitting -- the route for large bases (aug-cc-pVQZ)."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.ADC import ADCSolver, build_static_correction


mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
# df=True requires a density-fitted mean field (one aux basis end to end)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()

solver = ADCSolver(mf, level='adc3', df=True)
static = build_static_correction(mf, kind='mp2_relaxed', B_aa=solver.B_aa)

eGF, Z = solver.solve(static_correction=static)
print(f'IP = {-eGF[0] * HARTREE_TO_EV:.3f} eV   Z = {Z[0]:.3f}   (DF)')
