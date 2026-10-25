"""Tests for the EOM-CC polarizability and the G0W@CC self-energy
(src/SingleReference/GW/cc_polarizability.py), following Lewis & Berkelbach,
JCTC 15, 2925 (2019), doi:10.1021/acs.jctc.9b00099.

For the 2-electron systems H2 and He the EOM-CC polarizability is EXACT, so:
- alpha(0) must match a finite-field FCI polarizability, and
- the G0WCC@HF ionization potential, with every EE state in the Lehmann sum,
  must match G0W@HF screened by the FCI response (tests/fci_screened_gw.py)
  and the paper's Table 1 (def2-SVP, GW100 geometry): H2 15.97 eV, He
  23.82 eV. The G0W0@HF column (H2 16.24, He 24.32) cross-checks this repo's
  own GW@RPA on the same systems.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, fci

from src.SingleReference.CC.eom import EOMCC
from src.Base.pyscf.pyscf_interface import get_dipole_integrals
from src.SingleReference.GW.qp_energy import calc_qp_energy
from tests.fci_screened_gw import fci_screened_g0w_ip


def check_alpha_vs_finite_field_fci():
    """H2/sto-3g, not 6-31g: with sto-3g's minimal virtual space (nv=2), the
    default nroots=4 EXACTLY spans the whole EE excitation manifold, so the
    Lehmann sum is complete. A larger basis (e.g. 6-31g) has far more
    possible excitations than 4, and truncating the sum to just the lowest
    4 roots is a genuine (large) truncation error, not a code bug --
    confirmed the hard way (nroots=4 on 6-31g gave alpha_zz off by ~2x from
    the finite-field reference, purely from missing higher-lying states)."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    eom = EOMCC(mf, level='ccsdt', t_stopping_eps=1e-11)
    res = eom.kernel('ee', nroots=4)
    alpha = res.polarizability(get_dipole_integrals(mf.mol, mf, representation='spin'))[0]

    def e_field(F):
        molf = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        h = (molf.intor('int1e_kin') + molf.intor('int1e_nuc')
             + F * molf.intor('int1e_r')[2])
        mff = scf.RHF(molf)
        mff.get_hcore = lambda *args: h
        mff.run()
        return fci.FCI(mff).kernel()[0]

    F = 5e-3
    alpha_ff = -(e_field(F) - 2 * e_field(0.0) + e_field(-F)) / F**2
    d = abs(alpha[2, 2] - alpha_ff)
    ok = d < 1e-4
    print(f"H2/sto-3g alpha_zz(0): Lehmann {alpha[2,2]:.6f} vs finite-field FCI "
          f"{alpha_ff:.6f} (diff {d:.2e}): {'OK' if ok else 'FAIL'}")
    return ok


def check_gwcc_vs_fci_and_paper():
    """H2 and He at the GW100 geometry in def2-SVP have exactly degenerate
    excitations (spin multiplets, p-type shells), so the left and right EOM
    solves return unrelated bases within a cluster; transition_densities
    pairs them per cluster. nroots = the whole EE manifold makes the Lehmann
    sum complete (a dense diagonalization), as in the paper; the IP must
    then equal the FCI-screened one and the paper's."""
    all_ok = True
    # GW100 geometry R(H2) = 0.74144 A, def2-SVP, HF reference
    cases = [
        ('H2', 'H 0 0 0; H 0 0 0.74144', 16.24, 15.97),
        ('He', 'He 0 0 0', 24.32, 23.82),
    ]
    for name, atom, ip_g0w0_paper, ip_gwcc_paper in cases:
        mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)

        ip_rpa = -calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                                 df=False, state='homo')
        ok = abs(ip_rpa - ip_g0w0_paper) < 0.05
        all_ok &= ok
        print(f"{name} G0W0@HF (repo GW@RPA): {ip_rpa:.3f} eV vs paper "
              f"{ip_g0w0_paper:.2f}: {'OK' if ok else 'FAIL'}")

        nstates = EOMCC(mf, level='ccsdt').manifold_dim('ee')
        ip_cc = -calc_qp_energy(mf, selfenergy='GW', polarizability='CCSDT',
                                state='homo', nroots=nstates)
        ip_fci = fci_screened_g0w_ip(mf)
        ok = abs(ip_cc - ip_fci) < 1e-5 and abs(ip_cc - ip_gwcc_paper) < 0.05
        all_ok &= ok
        print(f"{name} G0WCC@HF (exact CC screening, {nstates} states): {ip_cc:.6f} eV "
              f"vs FCI-screened {ip_fci:.6f} (diff {ip_cc - ip_fci:.1e}) and paper "
              f"{ip_gwcc_paper:.2f}: {'OK' if ok else 'FAIL'}")
    return all_ok


def check_ccsd_screening_is_refused():
    """EOM-CCSD is not in MBPTcode (pyscf has it); CC
    screening at the CCSD level is refused at the EOM build, by name."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    try:
        calc_qp_energy(mf, selfenergy='GW', polarizability='CCSD', state='homo', nroots=4)
        ok = False
    except ValueError as exc:
        ok = 'pyscf' in str(exc)
    print(f"H2/sto-3g: G0WCC with EOM-CCSD screening is refused: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    all_ok = True
    all_ok &= check_alpha_vs_finite_field_fci()
    all_ok &= check_gwcc_vs_fci_and_paper()
    all_ok &= check_ccsd_screening_is_refused()
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
