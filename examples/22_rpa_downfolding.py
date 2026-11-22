"""Downfolding onto a pi space: the effective Hamiltonian of ethylene (2,2) and
benzene (6,6) in cc-pVDZ, downfolded from HF (CASCI) and from the RPA with bare,
cRPA, cRPA_ph and mRPA (moment-conserving, Casida only) screening, and with
cRPA, cRPA_ph and mRPA plus the static environment self-energy Sigma^E
(sigma='srg'; see docs/methods/downfolding.md).
The columns E_core, E_h, E_x and E_c are the terms of E_0 beside the nuclear
repulsion: the one-electron energy of the environment, and the Hartree,
exchange and correlation energies with their double-counting corrections
(`energies.eh`, `.ex`, `.ec`; each part is a field of its own, e.g.
`e_hartree` and `e_hartree_dc`). -TrSig is -Tr[Sigma_bar gamma^A,0], the last
term of E_0, present with Sigma^E only. The rows marked 'if' are the
density-fitted imaginary-frequency route (mode='imagfrequency'), where 'srg'
is s = infinity, continued by Pade from the imaginary axis; they match the
Casida rows above them to the density fit. The rows marked 'st' are the
space-time route (mode='space-time': separable ISDF factors on the G1 grid,
imaginary time), the same continuation of Sigma^E; they differ from the 'if'
rows by the ISDF fit, mostly in E_0, much less in the excitation energies.

One run_low_level per method, route and molecule serves every screening. The pi
orbitals are picked by Loewdin population on the carbons' p shell, and the
lowest singlet and triplet excitation energies come from the singlet and
triplet sectors of the active-space FCI."""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.solvers.backends import PyscfFCISolver

# both molecules lie in the xy plane, so the pi orbitals are the carbon p_z
ETHYLENE = [('C', (0, 0.6695, 0)), ('C', (0, -0.6695, 0)),
            ('H', (0.92892559, 1.23207733, 0)), ('H', (-0.92892559, 1.23207733, 0)),
            ('H', (0.92892559, -1.23207733, 0)), ('H', (-0.92892559, -1.23207733, 0))]
BENZENE = [('C', (0, 1.39250319, 0)), ('C', (-1.20594314, 0.6962516, 0)),
           ('C', (-1.20594314, -0.6962516, 0)), ('C', (0, -1.39250319, 0)),
           ('C', (1.20594314, -0.6962516, 0)), ('C', (1.20594314, 0.6962516, 0)),
           ('H', (-2.14171677, 1.23652075, 0)), ('H', (-2.14171677, -1.23652075, 0)),
           ('H', (0, -2.47304151, 0)), ('H', (2.14171677, -1.23652075, 0)),
           ('H', (2.14171677, 1.23652075, 0)), ('H', (0, 2.47304151, 0))]
MOLECULES = (('ethylene (2,2)', ETHYLENE, 1), ('benzene (6,6)', BENZENE, 3))
METHODS = (('HF (CASCI)', 'hf', {}),
           ('RPA bare', 'rpa', {'screening': 'bare'}),
           ('RPA cRPA', 'rpa', {'screening': 'crpa'}),
           ('RPA cRPA_ph', 'rpa', {'screening': 'crpa_ph'}),
           ('RPA mRPA', 'rpa', {'screening': 'mrpa'}),
           ('cRPA+SRG', 'rpa', {'screening': 'crpa', 'sigma': 'srg'}),
           ('cRPA_ph+SRG', 'rpa', {'screening': 'crpa_ph', 'sigma': 'srg'}),
           ('mRPA+SRG', 'rpa', {'screening': 'mrpa', 'sigma': 'srg'}),
           ('cRPA if', 'rpa_if', {'screening': 'crpa'}),
           ('cRPA+SRG if', 'rpa_if', {'screening': 'crpa', 'sigma': 'srg'}),
           ('cRPA st', 'rpa_st', {'screening': 'crpa'}),
           ('cRPA+SRG st', 'rpa_st', {'screening': 'crpa', 'sigma': 'srg'}))


def excitations(g_eff, space):
    """(E_S0, S1 - S0, T1 - S0) in Eh of the effective Hamiltonian: the
    singlet sector gives S0 and S1, the Ms = 1 sector the lowest triplet. The
    FCI is run on h_eff and v_eff alone, so E_0 is added to S0 here (solve()
    adds it itself)."""
    n, nelec = space['n_orbs'], space['n_elec_active']
    h1, eri = g_eff[1], g_eff[2].transpose(0, 2, 1, 3)   # chemist order
    singlets = PyscfFCISolver().roots(h1, eri, n, nelec, nroots=2, spin=0)[0]
    triplet = PyscfFCISolver().roots(h1, eri, n, nelec, nroots=1, spin=2)[0]
    return (singlets[0] + g_eff[0], singlets[1] - singlets[0],
            triplet[0] - singlets[0])


for name, atoms, n_pi in MOLECULES:
    mol = gto.M(atom=atoms, basis='cc-pvdz', unit='A', cart=False, verbose=0)
    mf = scf.RHF(mol).run()
    # the n_pi highest occupied and lowest virtual orbitals on the carbon p shell
    carbons = [i for i, (s, _) in enumerate(atoms) if s == 'C']
    active = ActiveSpace.from_population(mf, atoms=carbons, ao_type='p',
                                         n_occ=n_pi, n_virt=n_pi)
    print(f'\n{name}: pi orbitals (canonical) occupied {list(active.occ_active)}, '
          f'virtual {list(active.virt_active)}; E(RHF) = {mf.e_tot:.6f}')

    # one low-level run per method and route; casida on exact 4-centre
    # integrals (df=False); imagfrequency is density fitted by construction,
    # so it gets the same orbitals with a density fit attached (without one
    # it would decompose the exact MO integrals: minutes for benzene);
    # space-time builds ISDF factors on the grid numerics.isdf names
    lows = {'hf': run_low_level(mf, method='hf'),
            'rpa': run_low_level(mf, method='rpa', mode='casida', df=False),
            'rpa_if': run_low_level(mf.density_fit(), method='rpa',
                                    mode='imagfrequency'),
            'rpa_st': run_low_level(mf.density_fit(), method='rpa',
                                    mode='space-time',
                                    numerics=Numerics(isdf={'level': 'G1'}))}
    print(f'E_nuc = {mf.energy_nuc():.6f}')
    print(f'{"method":<13}{"E_core":>12}{"E_h":>12}{"E_x":>11}{"E_c":>10}'
          f'{"-TrSig":>10}{"E_0":>12}{"E(S0)":>12}{"S1 (eV)":>9}{"T1 (eV)":>9}')
    for label, method, opts in METHODS:
        g_eff, space, en = build_effective_hamiltonian(lows[method], active, **opts)
        e_s0, s1, t1 = excitations(g_eff, space)
        print(f'{label:<13}{en.e_core:12.6f}{en.eh:12.6f}{en.ex:11.6f}'
              f'{en.ec:10.6f}{0.0 - en.e_sigma_trace:10.6f}{en.e0:12.6f}'
              f'{e_s0:12.6f}{s1 * HARTREE_TO_EV:9.3f}{t1 * HARTREE_TO_EV:9.3f}')
