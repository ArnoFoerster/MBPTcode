import os
import sys
from pyscf import gto, scf, df

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.SingleReference.GW.qp_energy import calc_qp_energy

# Run a quick RHF on Ne
mol = gto.Mole()
mol.atom = "Ne 0 0 0"
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = df.make_auxbasis(mol)
mf.run()

print("1. Testing calc_qp_energy with defaults (GW, RPA, df=True, state=homo)...")
val1 = calc_qp_energy(mf)
print(f"Result: {val1:.6f} eV")

print("\n2. Testing calc_qp_energy with vertex correction PSD1, polarizability=BSE...")
val2 = calc_qp_energy(mf, selfenergy="PSD1", polarizability="BSE")
print(f"Result: {val2:.6f} eV")

print("\n3. Testing calc_qp_energy with state list and method list...")
val3 = calc_qp_energy(mf, selfenergy=["GW", "PSD1", "PSD2"], polarizability="BSE", state=[3, 4])
print(f"Result dictionary:\n{val3}")

print("\n4. Testing calc_qp_energy with printSpectralFunction=True...")
calc_qp_energy(mf, selfenergy="GW", state=4, printSpectralFunction=True)
print("\nAll tests completed!")
