"""Vertical ADC(2) singlets and triplets of one molecule, gas phase or in a
continuum, written as one JSON record.

    python -m src.SingleReference.ADC.eeADC --xyz mol.xyz \
        --environment toluene --nroots 6 --out-dir results/mol/toluene

from the repository root (or with it on PYTHONPATH); `main(argv)` is the
same run from Python.

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
--ss-density relaxed builds the term from the orbital-relaxed difference
density instead (the ground-state surface charges frozen; relaxed+pcm lets the
continuum respond inside the orbital response, a bound on that choice): the
channel record then also keeps the unrelaxed shifts (ss_shifts_unrelaxed_eV)
and both dipole changes (delta_dipole_au), and density_<spin>.npz the relaxed
densities in the full MO basis (delta_density_relaxed).

--spin-scaling scs|sos runs SCS- or SOS-ADC(2) (ee_ri_fold; the factors of
ADC2_SPIN_SCALING), gas phase or with the LR term, not with --state-specific.

Frozen core by default: the 1s shell of every atom past He (and the 2s2p shell
past Ne). Restartable: the SCF density is kept in scf.npz and each finished
spin channel in channel_<spin>.json; a rerun reuses both. --estimate-only
prints the problem's size, the memory the operator holds and the floating
point work of one product, and exits.

--distribute runs over MPI ranks (srun -n N): every rank converges the same
SCF and takes rank 0's orbitals, holds L and B_oo once per node, B_vv in its
share of the auxiliary chunks and the MP1 amplitudes of its occupied tiles, and
works on those tiles in each product; rank 0 alone iterates and writes.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import time
import warnings

import numpy as np
from pyscf import df, gto, scf

from src.SingleReference.common.active_space import frozen_core
from src.Base.numerics.constants import (ADC2_SPIN_SCALING, BYTES_PER_GB,
                                         EE_RI_AMPLITUDE_CACHE_GB, EE_RI_FOLD_TILE_ARRAYS,
                                         EE_RI_TILE_GB, HARTREE_TO_EV, MB_PER_GB)
from src.Base.utils.record_io import code_id
from src.Base.utils.parallel.mpi_grid import (distributed, grid_comm,
                                              lockstep_mean_field)
from src.Base.utils.parallel.threads import blas_threads, openmp_threads
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc

# the record's words for the state-specific term's density and its mode
SS_DENSITY_CONVENTIONS = {
    'unrelaxed': 'unrelaxed ISR-ADC(2)',
    'relaxed': 'orbital-relaxed ADC(2) (ground-state charges frozen)',
    'relaxed+pcm': 'orbital-relaxed ADC(2) (continuum responding)'}
SS_MODE_CONVENTIONS = {
    'off': 'not added',
    'vertical': 'Q at eps_optical, slow polarization frozen',
    'equilibrium': 'Q at eps_static'}


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS, in kB on Linux
    return r / (BYTES_PER_GB if platform.system() == 'Darwin' else BYTES_PER_GB / 1024)


def estimate(mol, nfrozen, auxbasis, jk_auxbasis, tile_gb, ranks=1):
    """Sizes, memory one process of `ranks` holds (GB) and the work of one
    product (flop).

    The SCF holds the JK fit's AO tensor; the factor build the correlation
    fit's AO tensor beside the MO factors; a product the factors (L and B_oo
    whole, B_vv and the kept MP1 amplitudes this rank's share), one tile and
    two (nocc nvir, naux) arrays of the vector it is on; the fold diagonal's
    setup a batch of whole B_vv rows of at most one tile."""
    no = mol.nelectron // 2 - nfrozen
    nv = mol.nao - mol.nelectron // 2
    naux = df.make_auxmol(mol, auxbasis).nao
    naux_jk = df.make_auxmol(mol, jk_auxbasis).nao
    gb = 8.0 / BYTES_PER_GB
    npair = mol.nao * (mol.nao + 1) / 2
    amplitudes = min(EE_RI_AMPLITUDE_CACHE_GB, (no * nv) ** 2 * gb / ranks)
    factors = (no * nv * naux + no * no * naux + nv * nv * naux / ranks) * gb
    tile = min(tile_gb, EE_RI_FOLD_TILE_ARRAYS * no * nv * nv * no * gb)
    per_vector = 2 * no * nv * naux * gb
    fold_diag = min(tile_gb, nv * nv * naux * gb / ranks) + naux * naux * gb
    flop_m = 2.0 * (no * nv) ** 2 * naux
    peak = max(npair * naux_jk * gb,
               factors + amplitudes + max(npair * naux * gb, tile + per_vector,
                                          fold_diag))
    return dict(nocc_active=no, nvir=nv, naux=naux, naux_jk=naux_jk,
                nfrozen=nfrozen, ranks=ranks, factors_gb=factors,
                amplitudes_gb=amplitudes, tile_gb=tile,
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
        return mf

    mf = factory(mol) if env is None else env.mean_field(mol, factory)
    if not mf.converged:
        # a numerical event: warned about, flagged in the record's 'scf'
        # (converged false) and the run carries on (inputs are refused,
        # numerical events are warned about)
        warnings.warn('the SCF did not converge; every number of this run is '
                      'flagged (record scf.converged)', RuntimeWarning)
    lockstep_mean_field(mf)
    if root:
        np.savez(path, dm=mf.make_rdm1(), mo_energy=mf.mo_energy, e_tot=mf.e_tot)
    # the JK fit is not needed past the SCF; the ADC builds its own factors
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None and hasattr(with_df, 'reset'):
        with_df.reset()
    return mf


def main(argv=None):
    """Parse `argv` (omitted: the command line) and run; the exit status."""
    p = argparse.ArgumentParser(prog='python -m src.SingleReference.ADC.eeADC',
                                description=__doc__.split('\n')[0])
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
    p.add_argument('--ss-density', default='unrelaxed',
                   choices=('unrelaxed', 'relaxed', 'relaxed+pcm'),
                   help='difference density of the state-specific term: '
                        'unrelaxed ISR-ADC(2), orbital-relaxed with the '
                        'ground-state charges frozen, or relaxed with the '
                        'continuum responding')
    p.add_argument('--spin-scaling', default='none', choices=('none', 'scs', 'sos'),
                   help='spin-component scaled ADC(2), factors (c_os, c_ss) of '
                        'ADC2_SPIN_SCALING')
    p.add_argument('--frozen', default='auto',
                   help="'auto' (core shells), or a number of orbitals")
    p.add_argument('--conv-tol', type=float, default=1e-6,
                   help='residual and frequency tolerance of the fold (Hartree)')
    p.add_argument('--scf-conv-tol', type=float, default=1e-10)
    p.add_argument('--scf-conv-tol-grad', type=float, default=1e-6)
    p.add_argument('--max-memory', type=int, default=4000, help='MB')
    p.add_argument('--tile-gb', type=float, default=EE_RI_TILE_GB)
    p.add_argument('--device', default='cpu', choices=('cpu', 'gpu'),
                   help='gpu: the doubles kernel on each rank\'s GPU (CuPy)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--label', default=None)
    p.add_argument('--estimate-only', action='store_true')
    p.add_argument('--ranks', type=int, default=None,
                   help='ranks the estimate divides the split factors over '
                        '(default: the launch\'s)')
    p.add_argument('--distribute', action='store_true',
                   help='run over the MPI ranks of the launch')
    args = p.parse_args(argv)
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
    ranks = args.ranks or (1 if comm is None else comm.Get_size())
    est = estimate(mol, nfrozen, args.auxbasis, args.jk_auxbasis, args.tile_gb, ranks)
    say(json.dumps(dict(estimate=est, nao=mol.nao, natm=mol.natm,
                        ranks=1 if comm is None else comm.Get_size(),
                        blas_threads=blas_threads(),
                        omp_threads=openmp_threads())), flush=True)
    if args.state_specific != 'off' and args.environment == 'gas':
        raise SystemExit('--state-specific needs --environment (a continuum)')
    if args.ss_density != 'unrelaxed' and args.state_specific == 'off':
        raise SystemExit('--ss-density needs --state-specific')
    if args.estimate_only:
        need = est['peak_gb']
        if need * MB_PER_GB > args.max_memory:
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
        from src.environments.solvent_screening import SolventScreening
        env = SolventScreening(mol, solvent=args.environment)
        env_info = dict(name=args.environment, eps_optical=env.eps,
                        eps_static=env.eps_static, method=env.method,
                        cavity=dict(env.cavity_settings, cavity_atoms=None),
                        ngrids=env.ngrids,
                        convention='SCF at eps_static; LR transition reaction '
                                   'field at eps_optical in the singlet singles '
                                   'block; second order with the bare '
                                   'interaction; state-specific term '
                                   '(1/2) dV Q dV of the '
                                   + SS_DENSITY_CONVENTIONS[args.ss_density]
                                   + ' difference density: '
                                   + SS_MODE_CONVENTIONS[args.state_specific])

    record = dict(label=args.label, xyz=os.path.abspath(args.xyz),
                  xyz_sha256=sha256(args.xyz), basis=args.basis,
                  auxbasis=args.auxbasis, jk_auxbasis=args.jk_auxbasis,
                  method=('ADC(2)' if args.spin_scaling == 'none' else
                          args.spin_scaling.upper() + '-ADC(2)')
                  + ' (folded doubles, density fitted)',
                  spin_scaling=dict(name=args.spin_scaling, factors=(
                      None if args.spin_scaling == 'none'
                      else list(ADC2_SPIN_SCALING[args.spin_scaling]))),
                  state_specific=args.state_specific,
                  ss_density=args.ss_density,
                  environment=env_info, nroots=args.nroots,
                  conv_tol=args.conv_tol, code=code_id(), estimate=est,
                  host=platform.node(), threads=dict(blas=blas_threads(),
                                                     omp=openmp_threads()),
                  ranks=1 if comm is None else comm.Get_size())
    t0 = time.time()
    mf = build_reference(mol, args, env, args.out_dir, root)
    record['scf'] = dict(e_tot=float(mf.e_tot), converged=bool(mf.converged),
                         wall_s=time.time() - t0,
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
            if (stored.get('state_specific', 'off') == args.state_specific
                    and stored.get('ss_density', 'unrelaxed') == args.ss_density
                    and stored.get('spin_scaling', 'none') == args.spin_scaling):
                channels[spin] = stored
                say(f'{spin}: reused {path}', flush=True)
                continue
        t = time.time()
        e, _, res = solve_ee_adc(mf, mol, level='adc2', spin=spin,
                                 nroots=args.nroots, fold=True, frozen=nfrozen,
                                 auxbasis=args.auxbasis, environment=env,
                                 conv_tol=args.conv_tol, return_fold=True,
                                 tile_gb=args.tile_gb, device=args.device,
                                 state_specific=args.state_specific,
                                 ss_density=args.ss_density,
                                 spin_scaling=(None if args.spin_scaling == 'none'
                                               else args.spin_scaling))
        op = res.operator
        ch = dict(omega_eV=(np.asarray(res.omega) * HARTREE_TO_EV).tolist(),
                  singles_weight_T1=np.asarray(res.t1).tolist(),
                  converged=np.asarray(res.converged).tolist(),
                  # the fold's numerical events by name (ee_fold.FoldResult.flags),
                  # and the SCF's: a rerun reuses this channel, its SCF or not
                  flags=list(getattr(res, 'flags', [])) + ([] if mf.converged else [
                      {'flag': 'scf_not_converged',
                       'message': 'the channel was solved on an SCF that did '
                                  'not converge'}]),
                  newton_steps=np.asarray(res.steps).tolist(),
                  loop=list(res.loop), level=np.asarray(res.level).tolist(),
                  e_mp2=op.e_mp2, product_columns=op.matvecs,
                  image_columns=op.image_columns, node_images=op.node_images,
                  tile_k=op.kt,
                  wall_s=time.time() - t, peak_rss_gb=peak_rss_gb(),
                  state_specific=args.state_specific,
                  ss_density=args.ss_density,
                  spin_scaling=args.spin_scaling)
        if args.state_specific != 'off':
            ch['ss_shifts_eV'] = {m: (np.asarray(v) * HARTREE_TO_EV).tolist()
                                  for m, v in res.ss_shifts.items()}
            ch['ss_shift_eV'] = (np.asarray(res.ss_shift) * HARTREE_TO_EV).tolist()
            ch['omega_ss_eV'] = (np.asarray(e) * HARTREE_TO_EV).tolist()
            densities = dict(delta_density=np.asarray(res.delta_density))
            if args.ss_density != 'unrelaxed':
                ch['ss_shifts_unrelaxed_eV'] = {
                    m: (np.asarray(v) * HARTREE_TO_EV).tolist()
                    for m, v in res.ss_shifts_unrelaxed.items()}
                ch['delta_dipole_au'] = {k: np.asarray(v).tolist()
                                         for k, v in res.delta_dipole.items()}
                densities['delta_density_relaxed'] = np.asarray(
                    res.delta_density_relaxed)
            if root:
                np.savez(os.path.join(args.out_dir, f'density_{spin}.npz'),
                         omega=res.omega, t1=res.t1, nfrozen=nfrozen,
                         **densities)
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
        # the caveats dEST is built on: the SCF and both channels' fold flags
        record['dEST_flags'] = sorted(
            {f['flag'] for c in ('singlet', 'triplet')
             for f in channels[c].get('flags', [])}
            | (set() if record['scf']['converged'] else {'scf_not_converged'}))
    record['wall_s'] = time.time() - t0
    record['peak_rss_gb'] = peak_rss_gb()
    if root:
        with open(record_path, 'w') as fh:
            json.dump(record, fh, indent=1)
    say(json.dumps({k: record.get(k) for k in ('S', 'T', 'dEST_eV', 'wall_s',
                                                  'peak_rss_gb')}), flush=True)
    return 0
