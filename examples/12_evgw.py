"""evGW: the quasiparticle energies reinjected into G and P0
    python examples/12_evgw.py
"""
import os
import sys

from pyscf import dft, gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV
from src.SingleReference import calc_qp_energy, evgw_eigenvalues

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
nocc = mol.nelectron // 2

mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()
print(f'PBE0   E = {mf.e_tot:.8f} Ha')
print(f'       HOMO {mf.mo_energy[nocc - 1] * HARTREE_TO_EV:8.3f} eV   '
      f'gap {(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV:6.3f} eV\n')


# the default routes
g0w0 = calc_qp_energy(mf, mode='casida', state='homo')
evgw = calc_qp_energy(mf, mode='casida', state='homo', self_consistency='evGW')
print(f'HOMO   G0W0 {g0w0:8.3f} eV      evGW {evgw:8.3f} eV\n')

# The loop itself with more details on convergence
print('cycle-by-cycle (the residual is max |delta eps| over HOMO and LUMO):')
eps, info = evgw_eigenvalues(mf, mol, mode='casida', verbose=True)
print(f"\nconverged in {info['cycles']} cycles: {info['converged']}")

ks = mf.mo_energy
single = calc_qp_energy(mf, mode='casida', state=[nocc - 1, nocc])
print(f'\n{"":14s}{"KS":>10s}{"G0W0":>10s}{"evGW":>10s}')
for label, p in (('HOMO', nocc - 1), ('LUMO', nocc)):
    print(f'{label:14s}{ks[p] * HARTREE_TO_EV:10.3f}'
          f'{single[p]["GW"]:10.3f}{eps[p] * HARTREE_TO_EV:10.3f}')
print(f'{"gap":14s}{(ks[nocc] - ks[nocc - 1]) * HARTREE_TO_EV:10.3f}'
      f'{single[nocc]["GW"] - single[nocc - 1]["GW"]:10.3f}'
      f'{(eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV:10.3f}')
