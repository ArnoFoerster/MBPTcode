"""The torch side of the NUCLEAR oracle: pin every stage, then differentiate
Omega and E_0 with respect to the nuclear coordinates by autograd.

    python torch_nuclear_oracle.py frozen.npz out.npz

Single-threaded and ended with os._exit for the reasons in torch_oracle.py.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
MBPTCODE_ROOT = os.environ.get('MBPTCODE_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
if MBPTCODE_ROOT not in sys.path:
    sys.path.insert(0, MBPTCODE_ROOT)

import numpy as np
import torch

from src.autograd_oracle import torch_chain as tc
from src.autograd_oracle import torch_nuclear as tn

torch.set_num_threads(1)


def pin(name, mine, ref, tol=1e-9):
    mine = np.asarray(mine.detach().numpy() if torch.is_tensor(mine) else mine, float)
    ref = np.asarray(ref, float)
    if mine.shape != ref.shape:
        print(f'  {name:26s} SHAPE {mine.shape} vs {ref.shape}')
        return False
    err = float(np.abs(mine - ref).max())
    sc = float(np.abs(ref).max()) or 1.0
    ok = err / sc < tol
    print(f'  {name:26s} max|d| {err:9.2e}  on {sc:9.2e}  rel {err / sc:8.1e}  '
          f'{"ok" if ok else "FAIL"}', flush=True)
    return ok


def main():
    f_in, f_out = sys.argv[1], sys.argv[2]
    x_mo, d, d_bare, eps_ref, fz, extra = tc.load(f_in)
    z = np.load(f_in, allow_pickle=False)
    c = tn.Constants(z)
    target = str(z['target'])
    qp_orb = int(z['qp_orb'])
    coords = torch.as_tensor(np.asarray(z['coords'], float)).clone().requires_grad_(True)

    t0 = time.perf_counter()
    pins = {}
    value, E0, info = tn.surface(coords, c, fz, target=target, qp_orb=qp_orb, pins=pins)
    t_fwd = time.perf_counter() - t0
    print(f'torch forward from the nuclear coordinates: {t_fwd:.1f} s')

    print('\n== pinning every stage against production at the reference geometry')
    allok = True
    allok &= pin('Fock(D*)', pins['F'], z['fock'], 1e-10)
    allok &= pin('E_0 (e_tot)', pins['e_tot'].reshape(1), np.array([float(z['e_tot'])]), 1e-11)
    allok &= pin('eps', pins['eps'], z['mo_energy'], 1e-9)
    allok &= pin('|C|', torch.abs(pins['C']), np.abs(z['mo_coeff']), 1e-8)
    resid = float(pins['resid'].abs().max())
    print(f'  {"fixed point |g(D*) - D*|":26s} {resid:9.2e}   (the SCF converged to 1e-11)')
    if c.solvated:
        allok &= pin('cavity points', pins['grid'], z['grid_ref'], 1e-12)
        allok &= pin('switching F', pins['Fsw'], z['F_ref'], 1e-12)
        allok &= pin('areas A', pins['A'], z['A_ref'], 1e-12)
        allok &= pin('K (optical)', pins['K_o'], z['K_o'], 1e-12)
        allok &= pin('R (optical)', pins['R_o'], z['R_o'], 1e-12)
        allok &= pin('K (static)', pins['K_s'], z['K_s'], 1e-12)
        allok &= pin('R (static)', pins['R_s'], z['R_s'], 1e-12)
        allok &= pin('Q (optical)', pins['Q_o'], z['Q_o'], 1e-10)
        allok &= pin('vtilde_aux', pins['vtilde'], z['vtilde_ref'], 1e-9)
        allok &= pin('e_pcm', pins['e_pcm'].reshape(1), np.array([float(z['e_pcm'])]), 1e-9)
    # The fit's Gram matrix has cond ~ 2e8, so integrals that agree to 1e-14
    # give an M that agrees to ~1e-8: the conditioning floor, not a convention.
    allok &= pin('fit M (cond 2e8)', pins['Mfit'], z['Mfit_ref'], 1e-7)
    allok &= pin('D dressed', pins['Dd'], z['D_ref_dressed'], 1e-7)
    if c.solvated:
        allok &= pin('D bare', pins['Db'], z['D_ref_bare'], 1e-7)
    # X_mo up to column signs
    xm, xr = pins['X_mo'].detach().numpy(), np.asarray(z['X_mo_ref'])
    sign = np.sign(np.einsum('kp,kp->p', xm, xr))
    allok &= pin('X_mo (signs aligned)', xm * sign[None, :], xr, 1e-8)
    vp = float(z['value_prod'])
    print(f'  {"value":26s} torch {float(value):.12f}  prod {vp:.12f}  d {float(value) - vp:+.2e}')

    print('\n== autograd with respect to the nuclear coordinates')
    t0 = time.perf_counter()
    g_value, = torch.autograd.grad(value, coords, retain_graph=True)
    g_e0, = torch.autograd.grad(E0, coords)
    print(f'  backward {time.perf_counter() - t0:.1f} s')
    g_value, g_e0 = g_value.numpy(), g_e0.numpy()
    g_prod, g_scf = np.asarray(z['g_omega']), np.asarray(z['g_scf'])
    name = 'dOmega/dR' if target == 'bse' else 'd eps^QP/dR'
    print(f'  atom ax   {name:>13s} torch     production     d           dE_0/dR torch    pyscf          d')
    for ia in range(coords.shape[0]):
        for x in range(3):
            print(f'  {ia:4d} {"xyz"[x]}   {g_value[ia, x]:+.9f}   {g_prod[ia, x]:+.9f}   '
                  f'{g_value[ia, x] - g_prod[ia, x]:+.2e}   {g_e0[ia, x]:+.9f}   '
                  f'{g_scf[ia, x]:+.9f}   {g_e0[ia, x] - g_scf[ia, x]:+.2e}')
    dv, de = np.abs(g_value - g_prod).max(), np.abs(g_e0 - g_scf).max()
    print(f'  {name}: max|torch - prod| {dv:.2e}  rel {dv / np.abs(g_prod).max():.1e}   '
          f'transl. residual torch {np.abs(g_value.sum(0)).max():.1e}')
    print(f'  dE_0/dR:  max|torch - pyscf| {de:.2e}  rel {de / np.abs(g_scf).max():.1e}   '
          f'transl. residual torch {np.abs(g_e0.sum(0)).max():.1e}')
    np.savez(f_out, g_value=g_value, g_e0=g_e0, value=float(value), pins_ok=bool(allok))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
