"""Planar-interface screening for a periodic slab: v -> v + vtilde with PBC.

The periodic counterpart of 13_solvated_gw_bse.py. A 3D crystal has no
outside, so there is no cavity to embed; the meaningful periodic case is a
SLAB, where the environment fills the vacuum region -- a monolayer in an
electrolyte, or a layer between an electrolyte above and a metal electrode
below, which is the electrode/electrolyte geometry.

The environment enters as an image-charge boundary condition at each planar
interface (not a closed cavity), and because the whole periodic pipeline reads
only PBCDFIntegrals.L, replacing L by its screened counterpart gives screened
RPA, screened W^Q, screened BSE blocks and screened Sigma_c at once.

Two things a slab needs that a molecule does not:
  * a non-negative Coulomb kernel -- pyscf's get_coulG puts a NEGATIVE value
    at G = 0 for cell.dimension < 3, which is not a usable RI-V metric. The
    AUTO Fermi-Dirac damped kernel supplies one, and check_low_dim_support
    verifies its real-space support still fits the vacuum at this k-mesh.
  * a cavity that actually encloses the density. leaked_density_fraction
    reports what is left outside; an over-tight cavity is refused rather than
    silently over-screened.
"""
import os
import sys

import numpy as np
from pyscf.pbc import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_rpa import ri_rpa_ecorr_from_dfints
from src.SingleReference.Periodic.pbc_rpa_damping import (nyquist_params,
                                                          make_coulG_damped,
                                                          check_low_dim_support)
from src.SingleReference.Periodic.pbc_solvent_screening import (
    SlabDielectricEnvironment, build_dfintegrals_screened, solvent_cohsex_kpts)
from src.Base.constants import HARTREE_TO_EV

KMESH = [2, 2, 1]

# A layer of H2 units with 24 bohr of vacuum. cell.dimension = 2 keeps the
# default low_dim_ft_type: 'inf_vacuum' would give a non-negative kernel but
# breaks pyscf's own periodic SCF, so the damped kernel below is the route.
cell = gto.Cell()
cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
cell.a = np.diag([4.0, 4.0, 24.0])
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.dimension = 2
cell.mesh = [13, 13, 72]
cell.verbose = 0
cell.build()

mf = scf.KRHF(cell, cell.make_kpts(KMESH), exxdiv=None).density_fit()
mf.kernel()

r0, beta, Rc = nyquist_params(cell, KMESH)
check_low_dim_support(cell, KMESH, r0, beta)
coulG = make_coulG_damped(r0, beta)
bare = build_dfintegrals_coulG(mf, coulG_fn=coulG)
ec_gas = ri_rpa_ecorr_from_dfints(bare, nw=24)
nocc = bare.nocc[0]

print(f'slab: {cell.natm} atoms, {KMESH} k-mesh, damping Rc = {Rc:.2f} bohr')
print(f'E(KRHF) = {mf.e_tot:.8f}   Ec(RPA) = {ec_gas:.8f}\n')
print(f"{'environment':>28}   {'Ec(RPA)':>12}  {'HOMO shift':>11} {'LUMO shift':>11}")
print('-' * 68)

for label, kwargs in (('vacuum', dict(eps=1.0)),
                      ('water both sides', dict(solvent='water')),
                      ('water | metal electrode', dict(solvent='water', eps_bot=np.inf))):
    env = SlabDielectricEnvironment(cell, z_center=0.0, z_half_width=4.0, **kwargs)
    screened = build_dfintegrals_screened(mf, env, coulG_fn=coulG)
    sigma = solvent_cohsex_kpts(bare, screened)
    diag = np.array([np.diag(sigma[k]).real for k in range(bare.nkpts)])
    print(f'{label:>28}   {ri_rpa_ecorr_from_dfints(screened, nw=24):12.8f}  '
          f'{diag[:, nocc - 1].mean() * HARTREE_TO_EV:+10.3f}  '
          f'{diag[:, nocc].mean() * HARTREE_TO_EV:+10.3f}   eV')

print('-' * 68)
print('Occupied levels rise and virtual levels fall: the gap closes, and the')
print('metal electrode screens harder than the electrolyte. Add the diagonal of')
print('solvent_cohsex_kpts to exchange_minus_vxc in qp_energy_g0w0 for QP energies.')
env = SlabDielectricEnvironment(cell, solvent='water', z_center=0.0, z_half_width=4.0)
print(f'\ndensity left outside the cavity: {env.leaked_density_fraction(mf):.2%}')
