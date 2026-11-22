"""Downfolding from an unrestricted reference: the OH radical in cc-pVDZ on a
UHF mean field, where the alpha and beta orbitals, their energies and
occupations differ, and so do the alpha and beta blocks of the effective
Hamiltonian (see docs/methods/downfolding.md, "Unrestricted references").

The active space is one ActiveSpace per spin, here the contiguous one with
five electrons in four orbitals, (N_alpha, N_beta) = (3, 2): per spin the
highest occupied and the lowest virtual orbitals. H_eff is downfolded from HF
(the CASCI Hamiltonian of the UHF determinant in its own orbitals) and from
the RPA with bare, cRPA, cRPA_ph and mRPA screening, and with cRPA plus the
static environment self-energy Sigma^E of each spin (sigma='srg').

The columns E_core, E_h, E_x and E_c are the terms of E_0 beside the nuclear
repulsion, each the total over both spins; -TrSig is -Tr[Sigma_bar gamma^A,0],
present with Sigma^E only. The solver is fci.direct_uhf: in spin-polarized
orbitals H_eff conserves S_z only, so its states are not spin eigenfunctions,
and <S^2> is printed for each. It is that of the whole molecule's state, the
active CI vector with the inactive core determinant, from the alpha-beta
overlaps of the orbitals (space['spin_overlap']); 0.75 is a pure doublet, and
the UHF determinant itself is printed for comparison.

E(1) is the lowest state of H_eff at this S_z and 'split' the gap to the
next one. The ground state of OH is 2-Pi, doubly degenerate: the hole can sit
in either pi orbital, and both are in the active space. The UHF determinant
puts it in one of them, so its orbitals, and an H_eff built on it, treat the
two components differently: 'split' is zero in the exact theory, and what
is printed is how much of the reference's broken symmetry each H_eff keeps.

The rows marked 'if' are the density-fitted imaginary-frequency route and
those marked 'st' the space-time route (ISDF factors on the G1 grid), both on
the same orbitals with a density fit attached; there 'srg' is s = infinity,
continued by Pade from the imaginary axis."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import unrestricted_from_counts
from src.Solvers.active_space import solve

METHODS = (('HF (UCASCI)', 'hf', {}),
           ('RPA bare', 'rpa', {'screening': 'bare'}),
           ('RPA cRPA', 'rpa', {'screening': 'crpa'}),
           ('RPA cRPA_ph', 'rpa', {'screening': 'crpa_ph'}),
           ('RPA mRPA', 'rpa', {'screening': 'mrpa'}),
           ('cRPA+SRG', 'rpa', {'screening': 'crpa', 'sigma': 'srg'}),
           ('cRPA if', 'rpa_if', {'screening': 'crpa'}),
           ('cRPA+SRG if', 'rpa_if', {'screening': 'crpa', 'sigma': 'srg'}),
           ('cRPA st', 'rpa_st', {'screening': 'crpa'}),
           ('cRPA+SRG st', 'rpa_st', {'screening': 'crpa', 'sigma': 'srg'}))

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, cart=False,
            verbose=0)
mf = scf.UHF(mol).run()
nocc = tuple(int((occ > 0).sum()) for occ in mf.mo_occ)

# one ActiveSpace per spin; any other choice is two ActiveSpace.from_indices
active = unrestricted_from_counts(mf.mo_energy, nocc, 4, (3, 2))
print(f'OH radical, UHF/cc-pVDZ: E = {mf.e_tot:.6f}, <S^2> = '
      f'{mf.spin_square()[0]:.4f}, occupied orbitals (alpha, beta) = {nocc}')
for spin, space in zip(('alpha', 'beta'), active):
    print(f'  {spin} active orbitals: occupied {list(space.occ_active)}, '
          f'virtual {list(space.virt_active)}')

# one low-level run per method and route serves every screening
fitted = mf.density_fit()
lows = {'hf': run_low_level(mf, method='hf'),
        'rpa': run_low_level(mf, method='rpa', mode='casida', df=False),
        'rpa_if': run_low_level(fitted, method='rpa', mode='imagfrequency'),
        'rpa_st': run_low_level(fitted, method='rpa', mode='space-time',
                                numerics=Numerics(isdf={'level': 'G1'}))}

print(f'E_nuc = {mf.energy_nuc():.6f}')
print(f'{"method":<13}{"E_core":>12}{"E_h":>12}{"E_x":>11}{"E_c":>10}'
      f'{"-TrSig":>10}{"E_0":>12}{"E(1)":>12}{"<S^2>":>8}{"split eV":>10}'
      f'{"<S^2>":>8}')
for label, method, opts in METHODS:
    g_eff, space, en = build_effective_hamiltonian(lows[method], active,
                                                   **opts)
    out = {}
    roots = solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
                  spin_free=space['spin_free'], nroots=2, out=out,
                  spin_overlap=space['spin_overlap'])
    s2 = out['spin_square']
    print(f'{label:<13}{en.e_core:12.6f}{en.eh:12.6f}{en.ex:11.6f}'
          f'{en.ec:10.6f}{0.0 - en.e_sigma_trace:10.6f}{en.e0:12.6f}'
          f'{roots[0]:12.6f}{s2[0]:8.4f}'
          f'{(roots[1] - roots[0]) * HARTREE_TO_EV:10.3f}{s2[1]:8.4f}')
