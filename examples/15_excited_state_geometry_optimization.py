"""
Excited-state geometry optimization: singlet and triplet relaxed
independently for the adiabatic gap between them

One high-level entry point, built on the BSE@GW gradient:

  `calc_adiabatic_gap(spec, Excitation('singlet'), spec, Excitation('triplet'), ...)`
      relaxes BOTH a singlet and a triplet, each to its OWN minimum, and
      differences their relaxed total energies directly:

          Delta-E_ST = E_S1(R*_S1) - E_T1(R*_T1)

      NOT the vertical gap at one geometry -- S1 and T1 relax by different
      amounts, so neither surface cancels and both are carried in full. This
      is the quantity a reverse-intersystem-crossing (rISC/TADF) rate depends
      on exponentially. Each state's own `vertical` (Omega(R0), Franck-Condon)
      and `emission` (Omega(R*), fluorescence/phosphorescence) block falls out
      of the same relaxation, at no extra cost -- S1 and T1 are each relaxed
      exactly once, and no ground-state geometry is ever optimized.

Formaldehyde's n -> pi* transitions (both S1 and T1) empty a C=O oxygen lone
pair and lengthen the bond -- a mild, textbook relaxation on every surface
here, so `emission` is always below `vertical` and nothing dissociates.

Two routes compute the same physics at two costs: the DENSE quasi-boson route
(O(N^6), exact within the model) and the cubic-scaling ISDF/space-time route
with the sum-over-poles continuation (`residues='sop'`), the one meant for
systems where the dense route does not fit. Delta-E_ST agrees between them to
~1 meV here -- tighter than either vertical gap alone, because the ISDF error
common to both states' Omega mostly cancels in the difference.

geomeTRIC's own progress log is suppressed below so only
`src/properties/optimize.py`'s own one-line-per-cycle print survives:
iteration, total energy, excitation energy and the residual force.

    python examples/15_excited_state_geometry_optimization.py
"""
import logging
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.declaration import Excitation, GroundState
from src.properties.excitations import SurfaceSpec, calc_adiabatic_gap

mol = gto.M(atom='C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588',
            basis='cc-pvdz', verbose=0)


def factory(m):
    mf = scf.RHF(m)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


ROUTES = {
    'dense, O(N^6)':
        SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',factorization='four-index'),
    'ISDF/space-time + SOP, cubic':
        SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                    residues='sop', factorization='isdf',numerics={'auxbasis': 'cc-pvdz-ri'}),
}

logging.disable(logging.WARNING)          # geomeTRIC's own log, not ours
gaps = {}
for name, spec in ROUTES.items():
    print(f'\n{name}: relaxing S1, then T1')
    gaps[name] = calc_adiabatic_gap(spec, Excitation('singlet'), spec,
                                    Excitation('triplet'), mol, factory,
                                    refreeze=0)
logging.disable(logging.NOTSET)

print(f"\n{'route':30s}{'state':>6s}{'vertical':>10s}{'emission':>10s}")
for name in ROUTES:
    gap = gaps[name]
    s1, t1 = gap['state_a'], gap['state_b']
    print(f"{name:30s}{'S1':>6s}{s1['omega_eV']:10.3f}{s1['emission_eV']:10.3f}")
    print(f"{'':30s}{'T1':>6s}{t1['omega_eV']:10.3f}{t1['emission_eV']:10.3f}")
    print(f"{'':30s}{'ΔE_ST':>6s}{gap['gap_eV']:10.3f}  (adiabatic, S1 - T1,"
          f" each at its own minimum)")
