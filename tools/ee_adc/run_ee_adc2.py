"""Vertical ADC(2) singlets and triplets of one molecule, gas phase or in a
continuum, written as one JSON record.

    python tools/ee_adc/run_ee_adc2.py --xyz mol.xyz --environment toluene \
        --nroots 6 --out-dir results/mol/toluene

The route is solve_ee_adc(fold=True): the doubles folded into the singles
problem and every doubles-shaped quantity built a tile at a time from the DF
factors, so a thousand basis functions fit on one machine. The reference is
DF-RHF on the JK fit (--jk-auxbasis); the ADC integrals use the correlation
fit (--auxbasis). In a continuum (a solvent name from pyscf's SMD table) the
SCF relaxes at the solvent's static dielectric constant and the excitation's
reaction field answers at the optical one, in the singlet singles block; the
second-order terms keep the bare interaction (see ee_ri_fold).

--state-specific vertical|equilibrium adds each root's state-specific term,
the reaction field of its unrelaxed ISR-ADC(2) difference density
(ee_state_specific): at the optical constant with the ground state's slow
polarization frozen (vertical) or at the static one (equilibrium). The
channel record keeps the eigenvalue (omega_eV, which holds the LR term of
singlets), the shifts of both modes and the one added (omega_ss_eV); each
root's active-space difference density goes to density_<spin>.npz.

Frozen core by default: the 1s shell of every atom past He (and the 2s2p shell
past Ne). Restartable: the SCF density is kept in scf.npz and each finished
spin channel in channel_<spin>.json; a rerun reuses both. --estimate-only
prints the problem's size, the memory the operator holds and the floating
point work of one product, and exits.

--distribute runs over MPI ranks (srun -n N): every rank converges the same
SCF and takes rank 0's orbitals, holds the DF factors whole and works on its
share of the occupied tiles of each product; rank 0 alone iterates and writes.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time


def _repo_root(start):
    """The directory holding src/, walking up from `start`."""
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, 'src')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit('run_ee_adc2.py: no src/ above ' + start)
        d = parent


ROOT = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                  # noqa: E402
from pyscf import df, gto, lib, scf                                 # noqa: E402

from src.Base.constants import EE_RI_TILE_GB, HARTREE_TO_EV        # noqa: E402
from src.Base.utils.mpi_grid import (distributed, grid_comm,        # noqa: E402
                                     lockstep_mean_field)
from src.Base.utils.threads import blas_threads, openmp_threads    # noqa: E402
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc   # noqa: E402


def frozen_core(mol):
    """Doubly occupied core orbitals: 1s past He, 1s2s2p past Ne, 1s..3d past Ar
    (the last only for Ga onwards; K-Zn keep their 3s3p frozen too)."""
    n = 0
    for a in range(mol.natm):
        z = mol.atom_charge(a)
        n += 1 if z > 2 else 0
        n += 4 if z > 10 else 0
        n += 4 if z > 18 else 0
        n += 5 if z > 30 else 0
    return n


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_sha():
    snap = os.path.join(ROOT, 'SNAPSHOT.txt')
    if os.path.isfile(snap):
        with open(snap) as fh:
            return fh.read().strip()
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:                                   # noqa: BLE001
        return None


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if platform.system() == 'Darwin' else 1024 ** 2)


def estimate(mol, nfrozen, auxbasis, jk_auxbasis, tile_gb):
    """Sizes, memory one process holds (GB) and the work of one product (flop).

    The SCF holds the JK fit's AO tensor; the factor build the correlation
    fit's AO tensor beside the MO factors; a product the factors, one tile
    and two (nocc nvir, naux) arrays of the vector it is on."""
    no = mol.nelectron // 2 - nfrozen
    nv = mol.nao - mol.nelectron // 2
    naux = df.make_auxmol(mol, auxbasis).nao
    naux_jk = df.make_auxmol(mol, jk_auxbasis).nao
    gb = 8.0 / 1024 ** 3
    npair = mol.nao * (mol.nao + 1) / 2
    factors = (no * nv * naux + 2 * no * no * naux + nv * nv * naux) * gb
    tile = min(tile_gb, 5 * no * nv * nv * no * gb)
    per_vector = 2 * no * nv * naux * gb
    flop_m = 2.0 * (no * nv) ** 2 * naux
    peak = max(npair * naux_jk * gb,
               factors + max(npair * naux * gb, tile + per_vector))
    return dict(nocc_active=no, nvir=nv, naux=naux, naux_jk=naux_jk,
                nfrozen=nfrozen, factors_gb=factors, tile_gb=tile,
                per_vector_gb=per_vector, cderi_jk_gb=npair * naux_jk * gb,
                cderi_ri_gb=npair * naux * gb, peak_gb=peak,
                flop_singles_product=flop_m, flop_folded_product=4 * flop_m)


def build_reference(mol, args, env, out_dir, root):
    """DF-RHF on the JK fit, relaxed in the continuum's eps_static if any;
    over ranks every rank converges it and takes rank 0's orbitals."""
    path = os.path.join(out_dir, 'scf.npz')
    dm0 = np.load(path)['dm'] if os.path.isfile(path) else None

    def factory(m):
        mf = scf.RHF(m).density_fit(auxbasis=args.jk_auxbasis)
        mf.conv_tol = args.scf_conv_tol
        mf.conv_tol_grad = args.scf_conv_tol_grad
        mf.max_cycle = 200
        mf.max_memory = args.max_memory
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError('the gas-phase SCF did not converge')
        return mf

    mf = factory(mol) if env is None else env.mean_field(mol, factory)
    if not mf.converged:
        raise RuntimeError('the SCF did not converge')
    lockstep_mean_field(mf)
    if root:
        np.savez(path, dm=mf.make_rdm1(), mo_energy=mf.mo_energy, e_tot=mf.e_tot)
    # the JK fit is not needed past the SCF; the ADC builds its own factors
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None and hasattr(with_df, 'reset'):
        with_df.reset()
    return mf


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--xyz', required=True)
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--basis', default='aug-cc-pvdz')
    p.add_argument('--auxbasis', default='aug-cc-pvdz-ri',
                   help='correlation fit of the ADC integrals')
    p.add_argument('--jk-auxbasis', default='aug-cc-pvdz-jkfit',
                   help='JK fit of the reference')
    p.add_argument('--environment', default='gas',
                   help="'gas' or a solvent name of pyscf's SMD table")
    p.add_argument('--spins', nargs='+', default=['singlet', 'triplet'])
    p.add_argument('--nroots', type=int, default=6)
    p.add_argument('--state-specific', default='off',
                   choices=('off', 'vertical', 'equilibrium'),
                   help='add the state-specific reaction field of each '
                        "root's difference density (needs a continuum)")
    p.add_argument('--frozen', default='auto',
                   help="'auto' (core shells), or a number of orbitals")
    p.add_argument('--conv-tol', type=float, default=1e-6,
                   help='residual and frequency tolerance of the fold (Hartree)')
    p.add_argument('--scf-conv-tol', type=float, default=1e-10)
    p.add_argument('--scf-conv-tol-grad', type=float, default=1e-6)
    p.add_argument('--max-memory', type=int, default=4000, help='MB')
    p.add_argument('--tile-gb', type=float, default=EE_RI_TILE_GB)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--label', default=None)
    p.add_argument('--estimate-only', action='store_true')
    p.add_argument('--distribute', action='store_true',
                   help='run over the MPI ranks of the launch')
    args = p.parse_args()
    comm = grid_comm()[0] if args.distribute else None
    with distributed(comm):
        return run(args, comm)


def run(args, comm):
    rank = 0 if comm is None else comm.Get_rank()
    root = rank == 0
    say = print if root else (lambda *a, **k: None)

    mol = gto.M(atom=args.xyz, basis=args.basis, charge=args.charge,
                verbose=4 if root and not args.estimate_only else 0,
                max_memory=args.max_memory)
    nfrozen = frozen_core(mol) if args.frozen == 'auto' else int(args.frozen)
    est = estimate(mol, nfrozen, args.auxbasis, args.jk_auxbasis, args.tile_gb)
    say(json.dumps(dict(estimate=est, nao=mol.nao, natm=mol.natm,
                        ranks=1 if comm is None else comm.Get_size(),
                        blas_threads=blas_threads(),
                        omp_threads=openmp_threads())), flush=True)
    if args.state_specific != 'off' and args.environment == 'gas':
        raise SystemExit('--state-specific needs --environment (a continuum)')
    if args.estimate_only:
        need = est['peak_gb']
        if need * 1024 > args.max_memory:
            say(f'refused: about {need:.0f} GB needed, --max-memory '
                f'{args.max_memory} MB', flush=True)
            return 2
        return 0

    if root:
        os.makedirs(args.out_dir, exist_ok=True)
    if comm is not None:
        comm.Barrier()
    record_path = os.path.join(args.out_dir, 'record.json')
    env = None
    env_info = {'name': 'gas'}
    if args.environment != 'gas':
        from src.Base.solvent_screening import SolventScreening
        env = SolventScreening(mol, solvent=args.environment)
        env_info = dict(name=args.environment, eps_optical=env.eps,
                        eps_static=env.eps_static, method=env.method,
                        cavity=dict(env._cavity, cavity_atoms=None),
                        ngrids=env.ngrids,
                        convention='SCF at eps_static; LR transition reaction '
                                   'field at eps_optical in the singlet singles '
                                   'block; second order with the bare '
                                   'interaction; state-specific term '
                                   '(1/2) dV Q dV of the unrelaxed ISR-ADC(2) '
                                   'difference density: ' + {
                                       'off': 'not added',
                                       'vertical': 'Q at eps_optical, slow '
                                                   'polarization frozen',
                                       'equilibrium': 'Q at eps_static'}[
                                       args.state_specific])

    record = dict(label=args.label, xyz=os.path.abspath(args.xyz),
                  xyz_sha256=sha256(args.xyz), basis=args.basis,
                  auxbasis=args.auxbasis, jk_auxbasis=args.jk_auxbasis,
                  method='ADC(2) (folded doubles, density fitted)',
                  state_specific=args.state_specific,
                  environment=env_info, nroots=args.nroots,
                  conv_tol=args.conv_tol, code=git_sha(), estimate=est,
                  host=platform.node(), threads=dict(blas=blas_threads(),
                                                     omp=openmp_threads()),
                  ranks=1 if comm is None else comm.Get_size())
    t0 = time.time()
    mf = build_reference(mol, args, env, args.out_dir, root)
    record['scf'] = dict(e_tot=float(mf.e_tot), wall_s=time.time() - t0,
                         homo_eV=float(mf.mo_energy[mol.nelectron // 2 - 1]
                                       * HARTREE_TO_EV),
                         lumo_eV=float(mf.mo_energy[mol.nelectron // 2]
                                       * HARTREE_TO_EV))
    say(json.dumps({'scf': record['scf']}), flush=True)

    channels = {}
    for spin in args.spins:
        path = os.path.join(args.out_dir, f'channel_{spin}.json')
        if os.path.isfile(path):
            with open(path) as fh:
                stored = json.load(fh)
            if stored.get('state_specific', 'off') == args.state_specific:
                channels[spin] = stored
                say(f'{spin}: reused {path}', flush=True)
                continue
        t = time.time()
        e, _, res = solve_ee_adc(mf, mol, level='adc2', spin=spin,
                                 nroots=args.nroots, fold=True, frozen=nfrozen,
                                 auxbasis=args.auxbasis, environment=env,
                                 conv_tol=args.conv_tol, return_fold=True,
                                 tile_gb=args.tile_gb,
                                 state_specific=args.state_specific)
        op = res.operator
        ch = dict(omega_eV=(np.asarray(res.omega) * HARTREE_TO_EV).tolist(),
                  singles_weight_T1=np.asarray(res.t1).tolist(),
                  converged=np.asarray(res.converged).tolist(),
                  newton_steps=np.asarray(res.steps).tolist(),
                  loop=list(res.loop), level=np.asarray(res.level).tolist(),
                  e_mp2=op.e_mp2, product_columns=op.matvecs, tile_k=op.kt,
                  wall_s=time.time() - t, peak_rss_gb=peak_rss_gb(),
                  state_specific=args.state_specific)
        if args.state_specific != 'off':
            ch['ss_shifts_eV'] = {m: (np.asarray(v) * HARTREE_TO_EV).tolist()
                                  for m, v in res.ss_shifts.items()}
            ch['ss_shift_eV'] = (np.asarray(res.ss_shift) * HARTREE_TO_EV).tolist()
            ch['omega_ss_eV'] = (np.asarray(e) * HARTREE_TO_EV).tolist()
            if root:
                np.savez(os.path.join(args.out_dir, f'density_{spin}.npz'),
                         delta_density=np.asarray(res.delta_density),
                         omega=res.omega, t1=res.t1, nfrozen=nfrozen)
        if root:
            with open(path, 'w') as fh:
                json.dump(ch, fh, indent=1)
        channels[spin] = ch
        say(json.dumps({spin: ch}), flush=True)

    record['channels'] = channels
    if 'singlet' in channels and 'triplet' in channels:
        key = 'omega_ss_eV' if args.state_specific != 'off' else 'omega_eV'
        s, tr = channels['singlet'][key], channels['triplet'][key]
        record['S'] = s[:3]
        record['T'] = tr[:3]
        record['dEST_eV'] = s[0] - tr[0]
    record['wall_s'] = time.time() - t0
    record['peak_rss_gb'] = peak_rss_gb()
    if root:
        with open(record_path, 'w') as fh:
            json.dump(record, fh, indent=1)
    say(json.dumps({k: record.get(k) for k in ('S', 'T', 'dEST_eV', 'wall_s',
                                                  'peak_rss_gb')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
