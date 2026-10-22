"""Four ways to the same G0W0 quasiparticle energy

    mode='casida', df=False    4-center ERIs, explicit Casida problem   O(N^6)
    mode='casida', df=True     three-index factor, same problem         O(N^6)
    mode='imagfrequency'       Sigma on an imaginary-frequency grid     O(N^4)
    mode='space-time'          Sigma as a product in imaginary time     O(N^3)

    python examples/10_gw_routes.py
"""
import os
import sys
from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV, SURFACE_GRID_ACCURACY
from src.Base.isdf_jk import isdf_jk, separable_factors_from_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.SingleReference import calc_qp_energy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.GW.space_time import solve_qp_energy_space_time

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
nocc = mol.nelectron // 2
# The space-time route interpolates on a grid it is told: the surfaces' level.
counts = resolve_isdf_grid(SURFACE_GRID_ACCURACY, str(mol.basis))[0]

# One exact mean field, so the four routes differ only in the GW step.
mf = scf.RHF(mol).run()
print(f'HF          E = {mf.e_tot:.8f} Ha   HOMO(KS) = 'f'{mf.mo_energy[nocc - 1] * HARTREE_TO_EV:.3f} eV\n')

# 1. Casida with the full 4-center ERIs
homo_full = calc_qp_energy(mf, state='homo', df=False)

# 2. Casida with the three-index factor (THE DEFAULT)
homo_df = calc_qp_energy(mf, state='homo')

# 3. Sigma integrated on an imaginary-frequency grid
homo_iw = calc_qp_energy(mf, state='homo', mode='imagfrequency')

# 4. Sigma through the space-time route
homo_st = calc_qp_energy(mf, state='homo', mode='space-time', counts=counts)

for label, homo in (('casida, full ERIs', homo_full),
                    ('casida, three-index', homo_df),
                    ('imaginary frequency', homo_iw),
                    ('space-time (ISDF)', homo_st)):
    print(f'{label:22s} HOMO = {homo:.3f} eV')

# The low-scaling routes are callable directly
qp_iw = solve_qp_energy_imaginary_axis(mf, mol, nocc, nocc - 1)
qp_st = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1, counts=counts)

# A window of states costs one self-energy on the space-time route
window = solve_qp_energy_space_time(mf, mol, nocc, [nocc - 2, nocc - 1, nocc],
                                    counts=counts)

print(f'\ncalled directly        {qp_iw * HARTREE_TO_EV:8.3f} eV (imaginary frequency)')
print(f'                       {qp_st * HARTREE_TO_EV:8.3f} eV (space-time)')
print('window HOMO-1..LUMO    '+ ', '.join(f'{e * HARTREE_TO_EV:.3f}' for e in window) + ' eV')

# The SCF's integrals are an independent choice. 

# With an ISDF mean field the static exchange <Sigma_x - v_xc> would inherit the SCF's ISDF-K
# which puts the interpolation error into the quasiparticle energy at first order.
# 'df-direct' rebuilds that one term from the three-index integrals instead.
mf_isdf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri', counts=counts)
mf_isdf.kernel()
factors = separable_factors_from_jk(mf_isdf)
qp_mf = solve_qp_energy_space_time(mf_isdf, mol, nocc, nocc - 1, factors=factors)
qp_dd = solve_qp_energy_space_time(mf_isdf, mol, nocc, nocc - 1, factors=factors,sigma_x='df-direct')
print(f'\nISDF SCF    E = {mf_isdf.e_tot:.8f} Ha')
print(f'space-time on it       {qp_mf * HARTREE_TO_EV:8.3f} eV (sigma_x=mf)')
print(f'                       {qp_dd * HARTREE_TO_EV:8.3f} eV (sigma_x=df-direct)')
