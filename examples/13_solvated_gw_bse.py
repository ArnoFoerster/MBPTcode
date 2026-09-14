"""GW and BSE in a polarizable continuum

Non-equilibrium solvation is 2 dielectric constants:

  * the GROUND STATE relaxes inside PCM(eps_static), because the solvent nuclei
    have had time to reorient around a state that is already there. That is
    `env.mean_field(mol, factory)`.
  * the RESPONSE to an added charge or an excitation is fast, so only the
    solvent's electrons follow it: eps_infinity = n^2, 1.78 for water against
    78.4. That is the vtilde `attach_environment` puts into the interaction.

Taking one without the other is a different physical model, so this example
runs the two halves separately and prints both. For a charged excitation the
response half dominates -- water moves its own HOMO by 2 meV through the ground
state and by 1.5 eV through the reaction field. For a neutral excitation it is
the other way round.

Inside GW the reaction field enters as Duchemin, Guido, Jacquemin and Blase,
Chem. Sci. 9, 4430 (2018) Eq. (18): the self-polarization of the orbital
carrying the added charge in the SCREENED reaction field, with the self-energy
itself screened by the BARE interaction. Screening Sigma dynamically as well
would count the same polarization twice.

    python examples/13_solvated_gw_bse.py
"""
import os
import sys

from pyscf import dft, gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import attach_environment
from src.Base.solvent_screening import SolventScreening
from src.SingleReference import calc_qp_energy
from src.SingleReference.LinearResponse.davidson import solve_bse_df

BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
NROOTS = 3
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis=BASIS, verbose=0)
nocc = mol.nelectron // 2


def factory(m):
    mf = dft.RKS(m, xc='pbe0').density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


# A named solvent carries both constants, so they cannot be taken apart.
env = SolventScreening(mol, solvent='water')
print(f'water: eps_inf = {env.eps:.4f} (the response), '
      f'eps_static = {env.eps_static:.3f} (the ground state)\n')

gas = factory(mol)
frozen = env.mean_field(mol, factory)        # ground state only
solvated = env.mean_field(mol, factory)      # and the response on top
attach_environment(solvated, env)

print(f'{"":28s}{"gas":>12s}{"+ground":>12s}{"+response":>12s}')

# ---- the ionization potential: a charged excitation ----------------------
ip = [-calc_qp_energy(m, mode='casida', state='homo')
      for m in (gas, frozen, solvated)]
print(f'{"IP (GW@PBE0) / eV":28s}' + ''.join(f'{v:12.3f}' for v in ip))
print(f'{"  shift from gas / eV":28s}'
      + ''.join(f'{v - ip[0]:12.3f}' for v in ip))

# ---- the lowest excitations: neutral -------------------------------------
rows = []
for m in (gas, frozen, solvated):
    omega = solve_bse_df(m, mol, nocc, nroots=NROOTS, probe=False)[0]
    rows.append(sorted(omega * HARTREE_TO_EV))
for k in range(NROOTS):
    print(f'{f"BSE@G0W0 root {k + 1} / eV":28s}'
          + ''.join(f'{r[k]:12.3f}' for r in rows))
    print(f'{"  shift from gas / eV":28s}'
          + ''.join(f'{r[k] - rows[0][k]:12.3f}' for r in rows))
