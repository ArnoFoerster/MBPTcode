import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, cc, df

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix

# The 6 self-energy variants requested: GW under each of the 3 screening
# choices (RPA/TDHF/BSE, all vertex_mode='GW'), plus the two vertex-corrected
# families GWGammaInf and PSD1/PSD2 (kept at their usual BSE-screened vertex).
METHODS = {
    'GW@RPA':     dict(selfenergy='GW@RPA',     polarizability='RPA'),
    'GW@TDHF':    dict(selfenergy='GW@TDHF',    polarizability='TDHF'),
    'GW@BSE':     dict(selfenergy='GW@BSE',     polarizability='BSE'),
    'GWGammaInf': dict(selfenergy='GWGammaInf', polarizability='BSE'),
    'PSD1':       dict(selfenergy='PSD1',       polarizability='BSE'),
    'PSD2':       dict(selfenergy='PSD2',       polarizability='BSE'),
}


def build_density_corrections(mf, mol):
    """None, CCSD, GW-linearized, GW-relaxed -- CCSD needs an RHF reference
    even when mf itself is a KS object, so it is built from a separate RHF
    object on the same mol and applied as a source-agnostic AO density
    correction on top of whichever mf is being tested.
    """
    mf_hf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf_hf).run()
    dm_ccsd = mf_hf.mo_coeff @ mycc.make_rdm1() @ mf_hf.mo_coeff.T

    dm_gw_lin = compute_gw_density_matrix(mf, mol, polarizability='RPA', df=True, relax=False)
    dm_gw_rel = compute_gw_density_matrix(mf, mol, polarizability='RPA', df=True, relax=True,
                                          numerics=Numerics())

    return {
        'none': None,
        'CCSD': dm_ccsd,
        'GW-linearized': dm_gw_lin,
        'GW-relaxed': dm_gw_rel,
    }


if __name__ == '__main__':
    all_ok = True
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='cc-pvdz')

    starting_points = {
        'HF': scf.RHF(mol).density_fit(),
    }
    mf_pbe0 = scf.RKS(mol).density_fit()
    mf_pbe0.xc = 'pbe0'
    starting_points['PBE0'] = mf_pbe0

    for sp_name, mf in starting_points.items():
        mf.with_df.auxbasis = df.make_auxbasis(mol)
        mf.run()
        print(f"\n{'='*90}\nStarting point: {sp_name} (E={mf.e_tot:.6f})\n{'='*90}")

        corrections = build_density_corrections(mf, mol)

        header = f"{'method':<12s}" + "".join(f"{c:>16s}" for c in corrections)
        print(header)
        rows = {}
        for method_label, kwargs in METHODS.items():
            row = []
            for corr_label, dm in corrections.items():
                val = calc_qp_energy(mf, state='homo', df=True, dm_correction=dm, **kwargs)
                ok = isinstance(val, float) and val == val and -100 < val < 0
                all_ok &= ok
                row.append(val)
            rows[method_label] = row
            print(f"{method_label:<12s}" + "".join(f"{v:16.6f}" for v in row))

        # Sanity relationships: a density correction should actually move the
        # QP energy (dm_correction wired up and doing something), and the two
        # GW density variants shouldn't be identical to each other (CPHF ran).
        for method_label, row in rows.items():
            uncorrected, ccsd, gw_lin, gw_rel = row
            ok = abs(uncorrected - ccsd) > 1e-4
            all_ok &= ok
            if not ok:
                print(f"  [{sp_name}/{method_label}] CCSD correction had no effect: FAIL")
            ok = abs(gw_lin - gw_rel) > 1e-6
            all_ok &= ok
            if not ok:
                print(f"  [{sp_name}/{method_label}] GW-linearized == GW-relaxed (CPHF had no effect): FAIL")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
