import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto, scf, df

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.Base.numerics.constants import HARTREE_TO_EV

benzene_geom = """
C  0.000  1.396  0.000
C  1.209  0.698  0.000
C  1.209 -0.698  0.000
C  0.000 -1.396  0.000
C -1.209 -0.698  0.000
C -1.209  0.698  0.000
H  0.000  2.479  0.000
H  2.147  1.240  0.000
H  2.147 -1.240  0.000
H  0.000 -2.479  0.000
H -2.147 -1.240  0.000
H -2.147  1.240  0.000
"""

if __name__ == '__main__':
    # w0 defaults to gap_scaled_w0 (None) throughout -- a fixed w0 needs a
    # much larger nfreq to converge on small-gap systems like benzene;
    # gap-scaled w0 gets sub-meV/few-meV agreement even at nfreq=20.
    cases = [
        ('HF/6-31g HOMO', 'H 0 0 0; F 0 0 0.9', '6-31g', 'homo', 30),
        ('HF/6-31g LUMO', 'H 0 0 0; F 0 0 0.9', '6-31g', 'lumo', 30),
        ('Ne/cc-pvdz HOMO', 'Ne 0 0 0', 'cc-pvdz', 'homo', 30),
        ('benzene/cc-pvtz HOMO', benzene_geom, 'cc-pvtz', 'homo', 20),
    ]

    all_ok = True
    for label, atom, basis, which, nfreq in cases:
        mol = gto.M(atom=atom, basis=basis)
        mf = scf.RHF(mol).density_fit()
        mf.with_df.auxbasis = df.make_auxbasis(mol)
        mf.run()

        nocc = mol.nelectron // 2
        p_state = nocc - 1 if which == 'homo' else nocc

        qp_analytic = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True, state=p_state)
        qp_ac = solve_qp_energy_imaginary_axis(mf, mol, nocc, p_state, numerics=Numerics(imagfreq={'nfreq': nfreq})) * HARTREE_TO_EV

        diff_mev = (qp_ac - qp_analytic) * 1000
        ok = abs(diff_mev) < 2.0
        all_ok &= ok
        print(f"{label:22s} nfreq={nfreq:3d}: AC={qp_ac:.6f} eV  analytic={qp_analytic:.6f} eV  diff={diff_mev:+.3f} meV  {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
