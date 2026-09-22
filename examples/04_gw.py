"""G0W0 quasiparticle energies -- the default route."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.SingleReference import calc_qp_energy, evgw_eigenvalues

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

# returns eV; the HOMO quasiparticle energy is minus the ionization potential
qp = calc_qp_energy(mf, state='homo')
print(f'G0W0 IP = {-qp:.3f} eV')

# a list of states returns {orbital: {method: energy}}
for p, e in calc_qp_energy(mf, state=[2, 3, 4]).items():
    print(f'  orbital {p}: QP = {e["GW"]:.3f} eV')
