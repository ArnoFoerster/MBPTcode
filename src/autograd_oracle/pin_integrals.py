"""Pin the torch integral engine against pyscf at one geometry (development gate)."""
import os
import sys
import time
os.environ['OMP_NUM_THREADS'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
MBPTCODE_ROOT = os.environ.get('MBPTCODE_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
if MBPTCODE_ROOT not in sys.path:
    sys.path.insert(0, MBPTCODE_ROOT)
import numpy as np
import torch
from pyscf import gto, df
from pyscf.dft import gen_grid
torch.set_num_threads(1)
from src.autograd_oracle import torch_integrals as ti
from src.autograd_oracle.gate_nuclear import shell_table_from_mole


def cmp(name, mine, ref):
    mine, ref = np.asarray(mine), np.asarray(ref)
    err = np.abs(mine - ref).max(); sc = np.abs(ref).max()
    print(f'  {name:22s} max|d| {err:9.2e}  on max|ref| {sc:9.2e}  rel {err / sc:8.1e}', flush=True)


def main():
    mol = gto.M(atom='O 0.02 0.01 0.117; H 0.03 0.757 -0.468; H -0.05 -0.757 -0.45',
                basis=sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz', verbose=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    coords = torch.as_tensor(mol.atom_coords())
    tab, taux = (ti.ShellTable(*shell_table_from_mole(mol)),
                 ti.ShellTable(*shell_table_from_mole(auxmol)))
    bas, aux = ti.Basis(tab, coords), ti.Basis(taux, coords)
    print(f'nao {tab.nao} (pyscf {mol.nao}), naux {taux.nao} (pyscf {auxmol.nao_nr()}), '
          f'shells {tab.nsh}/{taux.nsh}, l max {max(tab.ls)}/{max(taux.ls)}')
    pts = torch.as_tensor(np.random.default_rng(0).normal(size=(50, 3)) * 1.5)
    t0 = time.perf_counter()
    cmp('AO values', bas.values(pts), mol.eval_gto('GTOval_sph', pts.numpy()))
    cmp('aux values', aux.values(pts), auxmol.eval_gto('GTOval_sph', pts.numpy()))
    S, T = ti.overlap_kinetic(bas)
    cmp('overlap', S, mol.intor('int1e_ovlp'))
    cmp('kinetic', T, mol.intor('int1e_kin'))
    V = ti.nuclear_attraction(bas, torch.as_tensor(mol.atom_charges(), dtype=torch.float64), coords)
    cmp('nuclear attraction', V, mol.intor('int1e_nuc'))
    V2 = ti.two_centre(aux)
    cmp('(P|Q)', V2, auxmol.intor('int2c2e', aosym='s1'))
    e3 = ti.three_centre(bas, aux)
    ref3 = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1').reshape(mol.nao, mol.nao, -1)
    cmp('(mu nu|P)', e3, ref3)
    # Gaussian surface charges as pyscf's PCM builds them
    ng = gen_grid.LEBEDEV_ORDER[29]
    xi = torch.as_tensor(np.full(60, 4.9) / np.sqrt(np.linspace(0.02, 0.05, 60)))
    gpts = pts[:60] if len(pts) >= 60 else torch.cat([pts, pts[:10]])
    fake = gto.fakemol_for_charges(gpts.numpy(), expnt=(xi.numpy()) ** 2)
    cmp('(P|g_k)', ti.single_potential_on_charges(aux, gpts, xi),
        gto.mole.intor_cross('int2c2e', auxmol, fake))
    cmp('(mu nu|g_k)', ti.pair_potential_on_charges(bas, gpts, xi),
        df.incore.aux_e2(mol, fake, intor='int3c2e', aosym='s1'))
    fake_n = gto.fakemol_for_charges(mol.atom_coords())
    vn = np.dot(mol.atom_charges(), gto.mole.intor_cross('int2c2e', fake_n, fake))
    cmp('nuclei on g_k', ti.nuclear_potential_on_charges(coords, torch.as_tensor(mol.atom_charges(), dtype=torch.float64), gpts, xi), vn)
    print(f'  ({time.perf_counter() - t0:.1f} s)')
    sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
