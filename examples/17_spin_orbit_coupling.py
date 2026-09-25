"""
Spin-orbit coupling, and El-Sayed's rule: <S1|H_SO|T1> and <S1|H_SO|T2>,

Formaldehyde's S1 and T1 are the SAME configuration, n -> pi*, just opposite
spin, so they can not couple according to El-Sayed's rule
Both states can still be spin-orbit coupled, mediated through higher-lying 
excited states. This coupling is mediated through vibrational modes.

    python examples/17_spin_orbit_coupling.py
"""
import logging
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_CM
from src.Base.declaration import Excitation, GroundState
from src.properties.excitations import SurfaceSpec, calc_adiabatic_gap
from src.properties.optimize import relax_ground_state
from src.properties.spin_orbit import (promoting_mode_couplings,
                                       spin_orbit_couplings)
from src.properties.vibronic import normal_modes
from src.SingleReference.LinearResponse.bse import solve_bse

mol = gto.M(atom='C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588',
            basis='aug-cc-pvdz', verbose=0)


def factory(m):
    mf = scf.RHF(m)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def socme_at(m, nroots_t=1):
    """The (2, nroots_t) |<S_I|H_SO|T_J>| table at geometry `m`, S1 vs T1..Tn."""
    mf = factory(m)
    nocc = m.nelectron // 2
    singlet = solve_bse(mf, mol=m, nroots=1, solver='dense', integrals='full',spin='singlet')[:3]
    triplet = solve_bse(mf, mol=m, nroots=nroots_t, solver='dense',integrals='full', spin='triplet')[:3]
    return spin_orbit_couplings(mf, m, nocc, singlet, triplet)['socme']

# specify the PES
spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',factorization='four-index')

# calculate diabatic gap
logging.disable(logging.WARNING)
gap = calc_adiabatic_gap(spec, Excitation('singlet'), spec,Excitation('triplet'), mol, factory, refreeze=0)
logging.disable(logging.NOTSET)

socme_s1 = socme_at(gap['state_a']['mol_excited_minimum'], nroots_t=2) * HARTREE_TO_CM
socme_t1 = socme_at(gap['state_b']['mol_excited_minimum'], nroots_t=2) * HARTREE_TO_CM

print(f"Delta-E_ST (adiabatic, S1 - T1, each at its own minimum): "
      f"{gap['gap_eV']:.3f} eV")
print(f"\n{'':22s}{'<S1|H_SO|T1>':>16s}{'<S1|H_SO|T2>':>16s}   (cm^-1)")
print(f"{'at R*_S1 (fwd ISC)':22s}{socme_s1[1, 0]:16.4f}{socme_s1[1, 1]:16.4f}")
print(f"{'at R*_T1 (rISC)':22s}{socme_t1[1, 0]:16.4f}{socme_t1[1, 1]:16.4f}")
print(f"\nsame configuration (n->pi*/n->pi*) vanishes by El-Sayed's rule; "
      f"different configuration (n->pi*/pi->pi*) does not.")

logging.disable(logging.WARNING)     
mol0, mf0 = relax_ground_state(mol, factory)
logging.disable(logging.NOTSET)

# pyscf's own analytic RHF Hessian
omega, modes, masses, _ = normal_modes(mf0, mol=mol0)
v0, promoting = promoting_mode_couplings(mol0, omega, modes, masses, socme_at, 1, 0)

print(f"\nGround-state (Franck-Condon) <S1|H_SO|T1> = {v0 * HARTREE_TO_CM:.4f} "
      f"cm^-1, {len(promoting)} modes:")
print(f"{'mode':>5s}{'omega/cm-1':>13s}{'|dV/dq|/cm-1':>15s}")
for k, wk, dv in sorted(promoting, key=lambda r: -r[2]):
    print(f"{k:5d}{wk * HARTREE_TO_CM:13.1f}{dv * HARTREE_TO_CM:15.4f}")
print(f"\nthe largest |dV/dq| marks the promoting mode: the vibration that "
      f"lends this El-Sayed-forbidden pair the symmetry it lacks.")
