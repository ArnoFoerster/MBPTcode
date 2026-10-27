"""Several ionization states: seed Davidson from each occupied orbital in turn."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.ADC import ADCSolver, build_static_correction

QP_Z_MIN = 0.05

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2

solver = ADCSolver(mf, level='adc3')
static = build_static_correction(mf, kind='mp2_relaxed')

# never match states by nearest eigenvalue -- it silently jumps roots
states = []
for orb in range(nocc - 1, nocc - 4, -1):
    eGF, Z = solver.solve(static_correction=static, homo_index=orb)
    if Z[0] >= QP_Z_MIN:
        states.append((-eGF[0] * HARTREE_TO_EV, Z[0]))

for i, (ip, z) in enumerate(sorted(states), start=1):
    print(f'state {i}:  IP = {ip:7.3f} eV   Z = {z:.3f}')
