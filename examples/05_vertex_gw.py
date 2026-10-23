"""Vertex-corrected quasiparticle energies on a BSE-screened polarizability, with DF."""
import os
import sys

from pyscf import cc, gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference import calc_qp_energy

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvqz', verbose=0)
# df=True needs a density-fitted mean field (one aux basis end to end)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri').run()

# polarizability: Use BSE polarizbility in all self-energy variants.
# PSD2 also needs triplets
qp = calc_qp_energy(mf, selfenergy=['GW@BSE', 'GWGammaInf', 'PSD1', 'PSD2'],
                    polarizability='BSE', state='homo', df=True)

for method, e in qp[mol.nelectron // 2 - 1].items():
    print(f'{method:<12} IP = {-e:.3f} eV')

# pySCF's CCSD, to judge the methods against
mycc = cc.CCSD(mf).run()
print(f'{"EOM-IP-CCSD":<12} IP = {mycc.ipccsd(nroots=1)[0] * HARTREE_TO_EV:.3f} eV')

#expected outcome
#GW@BSE       IP = 12.493 eV
#GWGammaInf   IP = 12.770 eV 
#PSD1         IP = 12.922 eV
#PSD2         IP = 12.636 eV
#EOM-IP-CCSD  IP = 12.637 eV --> underestimates IP a bit, true IP closer to 12.75, GWGammaInf matches perfectly