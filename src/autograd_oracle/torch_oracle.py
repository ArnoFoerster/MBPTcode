"""The torch side of the oracle: a process that never imports pyscf.

    python torch_oracle.py frozen.npz torch_adjoints.npz [--root-mode ..] [--newton-steps ..]

Reads the factors, the frozen conventions and (when present) the production
adjoints written by `gate_factors.py`, evaluates the chain in torch,
differentiates it by autograd, prints the forward and adjoint comparisons, and
writes its own adjoints for the swap-in step. Pinned to ONE torch thread (the
x86 build under Rosetta segfaults in backward with two) and ends with
os._exit (its Intel OpenMP runtime deadlocks the normal exit on this stack).
"""
import argparse
import os
import sys
import time

# ONE THREAD, NOT A CHOICE. The x86_64 torch in this environment runs under
# Rosetta, and its backward pass segfaults reproducibly (3/3) with two intra-op
# threads and never (3/3) with one. The oracle is small; it does not need them.
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

torch.set_num_threads(1)


def compare(name, prod, other, floor=1e-300):
    """One adjoint: scale, worst absolute and relative deviation, alignment."""
    prod, other = np.asarray(prod, float), np.asarray(other, float)
    scale = float(np.abs(prod).max())
    diff = np.abs(prod - other)
    rel = float(diff.max() / max(scale, floor))
    big = np.abs(prod) > 1e-2 * scale
    worst = float((diff[big] / np.abs(prod[big])).max()) if big.any() else float('nan')
    cos = float((prod.ravel() @ other.ravel())
                / max(np.linalg.norm(prod) * np.linalg.norm(other), floor))
    print(f'  {name:12s} max|prod| {scale:10.3e}   max|d| {diff.max():9.2e}   '
          f'rel {rel:8.1e}   worst entry(>1% of max) {worst:8.1e}   '
          f'1-cos {1.0 - cos:8.1e}', flush=True)
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--root-mode', default='newton', choices=['newton', 'ift'])
    ap.add_argument('--newton-steps', type=int, default=3)
    args = ap.parse_args()

    x_mo, d, d_bare, eps, fz, extra = tc.load(args.inp)
    target = str(extra['target'])
    qp_orb = int(extra['qp_orb'])
    t0 = time.perf_counter()
    X_t, Dd_t, eps_t = tc.as_t(x_mo), tc.as_t(d), tc.as_t(eps)
    Db_t = None if d_bare is None else tc.as_t(d_bare)
    value_t, info, adj = tc.adjoints(X_t, Dd_t, Db_t, eps_t, fz, target=target,
                                     qp_orb=qp_orb, root_mode=args.root_mode,
                                     newton_steps=args.newton_steps)
    value_prod = float(extra['value_prod'])
    print(f'torch      {target}: value {value_t:.10f} Ha   forward+autograd '
          f'{time.perf_counter() - t0:.1f} s   root mode {args.root_mode}, '
          f'{args.newton_steps} steps   threads {torch.get_num_threads()}', flush=True)

    print('\n== forward', flush=True)
    print(f'  value        prod {value_prod:.12f}  torch {value_t:.12f}  '
          f'd {value_t - value_prod:+.2e} Ha')
    if extra.get('shift_prod') is not None:
        sh_t = info['shift'].detach().numpy()
        sh_p = extra['shift_prod']
        print(f'  Eq.(18)      max|d| {np.abs(sh_t - sh_p).max():.2e} Ha on '
              f'max|shift| {np.abs(sh_p).max():.3e}')
    for p in info['roots']:
        w_t, z_t = float(info['roots'][p]), float(info['z'][p])
        zp = fz.z_prod.get(p, float('nan'))
        print(f'  eps^QP[{p:2d}]   prod {fz.roots[p]:.12f}  torch {w_t:.12f}  '
              f'd {w_t - fz.roots[p]:+.2e}   Z prod {zp:.9f} torch {z_t:.9f} '
              f'd {z_t - zp:+.1e}   Newton |f| {info["newton_resid"][p]:.1e}')
    if 'eps_qp' in info and extra.get('eps_qp_prod') is not None:
        print(f'  eps^QP diag  max|d| '
              f'{np.abs(info["eps_qp"].detach().numpy() - extra["eps_qp_prod"]).max():.2e} Ha')
    if 'omega' in info and extra.get('omega_prod') is not None:
        om_t = info['omega'].detach().numpy()
        n = min(6, len(om_t), len(extra['omega_prod']))
        print(f'  Omega[:{n}]   max|d| {np.abs(om_t[:n] - extra["omega_prod"][:n]).max():.2e} Ha')

    rels, worst = [], np.nan
    if extra.get('eps_bar') is not None:
        print('\n== adjoints: production manual reverse pass vs torch.autograd', flush=True)
        rels = [compare('eps_bar', extra['eps_bar'], adj['eps']),
                compare('X_bar', extra['x_bar'], adj['x']),
                compare('D_bar', extra['d_bar'], adj['d'])]
        if extra.get('d_bar_bare') is not None:
            rels.append(compare('D_bar_bare', extra['d_bar_bare'], adj['d_bare']))
        worst = float(max(rels)) if np.all(np.isfinite(rels)) else float('nan')  # NaN must not hide behind max
        print(f'  WORST ADJOINT REL {worst:.1e}  ->  '
              + ('AGREE' if worst < 1e-8 else 'DISAGREE (inspect the table)'), flush=True)

    np.savez(args.out, value=value_t, eps_bar=adj['eps'], x_bar=adj['x'],
             d_bar=adj['d'],
             d_bar_bare=(np.zeros(0) if adj['d_bare'] is None else adj['d_bare']),
             worst_rel=(worst if rels else np.nan))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
