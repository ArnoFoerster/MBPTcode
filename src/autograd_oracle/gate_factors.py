"""Gate the production manual adjoints against a torch.autograd oracle.

The production solvated BSE@GW chain (`ExcitedStateChain`) is run once; its
reverse pass is intercepted at `nuclear_gradient`, so the adjoints compared
are exactly what production hands to the nuclear assembly -- (eps_bar, X_bar,
D_bar, D_bar_bare). The factors, the frozen conventions and those adjoints are
written to an .npz, and `torch_oracle.py` -- a SEPARATE PROCESS that never
imports pyscf -- evaluates the same function in torch (`torch_chain.py`),
differentiates it by autograd and prints the comparison. Two processes because
torch's x86 Intel OpenMP runtime and pyscf/BLAS deadlock at interpreter exit
when they share one.

Checks, in order:
  forward   Omega (or eps^QP), eps^QP on the window, Eq. (18) shift, Z
  adjoints  each of the four arrays, entry by entry
  swap-in   torch adjoints through production's OWN nuclear assembly against
            production's force -- what the adjoint differences are worth in
            Ha/Bohr
  --fd      optional four-point finite difference of ONE nuclear component of
            the whole chain (SCF, cavity, factors, QP, Casida redone)

Run as `python -m src.autograd_oracle.gate_factors` from the repo root, or by
path from anywhere; `main(argv)` returns the numbers for a test.
"""
import argparse
import os
import subprocess
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '2')

HERE = os.path.dirname(os.path.abspath(__file__))
MBPTCODE_ROOT = os.environ.get('MBPTCODE_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
if MBPTCODE_ROOT not in sys.path:
    sys.path.insert(0, MBPTCODE_ROOT)

import numpy as np
from pyscf import gto, scf

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.geometry import molecule_at
from src.Base.numerics import Numerics
from src.Base.constants import HARTREE_TO_EV, KAPPA, SURFACE_GRID_ACCURACY
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.real_screening import residue_set
from src.SingleReference.GW.qp_states import frozen_scissor
from src.gradients.excited_state import ExcitedStateChain
from src.autograd_oracle.frozen import FrozenConventions, dump


def say(*args, **kwargs):
    """print, flushed: the progress lines must interleave with the torch process."""
    print(*args, flush=True, **kwargs)


#: water bent out of C2v: a C1 system, so no symmetry hides a wrong component
H2O_C1 = 'O 0.02 0.01 0.117; H 0.03 0.757 -0.468; H -0.05 -0.757 -0.45'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='bse', choices=['bse', 'qp'])
    ap.add_argument('--gas', action='store_true', help='gas-phase control (no continuum)')
    ap.add_argument('--solvent', default='water')
    ap.add_argument('--spin', default='singlet', choices=['singlet', 'triplet'])
    ap.add_argument('--tda', action='store_true')
    ap.add_argument('--state', type=int, default=0)
    ap.add_argument('--qp-offset', type=int, default=0,
                    help='for --target qp: orbital relative to the HOMO')
    ap.add_argument('--nfreq-cd', type=int, default=None)
    ap.add_argument('--root-mode', default='newton', choices=['newton', 'ift'])
    ap.add_argument('--newton-steps', type=int, default=3)
    ap.add_argument('--fd', default=None, metavar='IA,X|all',
                    help='finite-difference one nuclear component (e.g. 0,2) or '
                         'all of them; with all, the total excited-state force '
                         'E_0 + Omega is differenced too')
    ap.add_argument('--h', type=float, default=1e-3, help='FD step in Bohr')
    ap.add_argument('--atom', default=H2O_C1)
    ap.add_argument('--basis', default='cc-pvdz')
    ap.add_argument('--grid-accuracy', default=SURFACE_GRID_ACCURACY,
                    help='validated ISDF grid level (no level exists for sto-3g)')
    ap.add_argument('--auxbasis', default=None,
                    help='default: basis + -ri; sto-3g has none, use cc-pvdz-ri')
    ap.add_argument('--workdir', default=os.environ.get('TMPDIR', HERE),
                    help='where the two .npz files go')
    args = ap.parse_args(argv)
    BASIS = args.basis
    AUX = args.auxbasis or paired_auxbasis(BASIS)

    def scf_factory(mol):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
        mf.kernel()
        assert mf.converged
        return mf

    # --------------------------------------------------------------- production
    t0 = time.perf_counter()
    mol = gto.M(atom=args.atom, basis=BASIS, verbose=0)
    env = None if args.gas else SolventScreening(mol, solvent=args.solvent)
    say('building the production chain (radii, layout, frames, PCM ground state)')
    chain = ExcitedStateChain(
        mol, scf_factory, spin=args.spin, state=args.state, bse_tda=args.tda,
        environment=env,
        numerics=Numerics(quadrature={'nfreq_cd': args.nfreq_cd or None},
                          bse={'solver': 'dense'},
                          isdf={'basis': BASIS, 'auxbasis': AUX,
                                'counts': resolve_isdf_grid(args.grid_accuracy, BASIS)[0]}))
    mol, mf = chain.mean_field()
    nocc, norb = chain.nocc, len(mf.mo_energy)
    say(f'{"gas phase" if args.gas else "PCM(" + args.solvent + ")"}: '
          f'{chain.label()}   nao {mol.nao}  M {chain.M}  naux {chain.naux}  '
          f'nocc {nocc}   {time.perf_counter() - t0:.1f} s')

    captured = {}
    _orig_nuclear_gradient = chain.nuclear_gradient

    def spy(mol_, mf_, auxmol, crd, x_mo, eps_bar, x_bar, d_bar, **assembly):
        # every keyword the chain hands its assembly (y_extra, g_extra,
        # d_bar_bare, root_bar, kernel_bar, gauges) is kept and re-fed below,
        # so a re-run differs from production in the swapped adjoints alone
        d_bar_bare = assembly.get('d_bar_bare')
        captured.update(auxmol=auxmol, crd=crd, x_mo=x_mo,
                        eps_bar=np.array(eps_bar), x_bar=np.array(x_bar),
                        d_bar=np.array(d_bar), assembly=dict(assembly),
                        d_bar_bare=None if d_bar_bare is None else np.array(d_bar_bare))
        return _orig_nuclear_gradient(mol_, mf_, auxmol, crd, x_mo, eps_bar, x_bar,
                                      d_bar, **assembly)

    chain.nuclear_gradient = spy

    say('production gradient')
    t0 = time.perf_counter()
    qp_orb = nocc - 1 + args.qp_offset
    if args.target == 'qp':
        g_prod, diags = chain.quasiparticle_gradient(args.qp_offset)
        value_prod = diags['qp_energy']
    else:
        g_prod, diags = chain.excitation_gradient()
        value_prod = diags['omega']
    say(f'production {args.target}: value {value_prod:.10f} Ha '
          f'({value_prod * HARTREE_TO_EV:.6f} eV)   gradient {time.perf_counter() - t0:.1f} s'
          f'   transl. residual {diags["translation_residual"]:.1e}   nfreq_cd {chain.conventions.nfreq_cd}')

    say('reading the frozen conventions')
    # every frozen convention, read off the chain AFTER its first solve
    om_all, pieces = chain._forward(mol, mf)
    (_, _, auxmol, crd, x_mo, d, eps, eps_qp_prod, w_aux, cache, xn, yn, mu,
     d_bare, shift_prod, screening, _) = pieces
    d_sigma = d if d_bare is None else d_bare
    states = [int(p) for p in chain.resolved.qp_set]
    if args.target == 'qp' and qp_orb not in states:
        states = sorted(set(states) | {qp_orb})
    xc_set = chain._xc_correction(mf, states, shift_prod)
    (ws, _, _, _), ro = chain._qp_set_solve(x_mo, d_sigma, eps, mu, xc_set,
                                            np.zeros(len(states)), states=states)
    roots = {int(p): float(w) for p, w in zip(states, ws)}
    z_prod = {int(p): float(z) for p, z in zip(states, ro['z'])}
    routes = ro['routes']
    shift_vec = np.zeros(norb) if shift_prod is None else np.asarray(shift_prod, float)
    xc_all = np.asarray(chain._xc_correction(mf, np.arange(norb), shift_vec), float)
    residues = {p: residue_set(eps, nocc, roots[p]) for p in roots}
    for p in roots:
        if routes.get(p) not in ('explicit', 'none') and residues[p]:
            sys.exit(f'orbital {p} took the {routes.get(p)!r} residue route; the '
                     f'oracle ports the explicit backend only')
    # the distance of each root from the nearest other orbital energy (Ha): the
    # one QP root finder has no guard band to compare it with
    gaps = {p: np.delete(np.abs(eps - roots[p]), p).min() for p in roots}
    say(f'window {states}   residues ' + ', '.join(
        f'{p}:{len(residues[p])}' for p in roots) + '   nearest level ' + ', '.join(
        f'{p}:{gaps[p]:.1e}' for p in roots))

    fz = FrozenConventions(
        nocc=nocc, mu=mu, states=states, xc_const=xc_all, roots=roots,
        residues=residues, gw_tau=chain.conventions.gw_grid.tau_points,
        gw_cosft=chain.conventions.gw_grid.cosft_wt, nu=chain.conventions.nu, wt=chain.conventions.wt,
        w_tau=chain.w_grid.tau_points, w_cosft=chain.w_grid.cosft_wt,
        spin_kappa=KAPPA[args.spin], bse_tda=args.tda, state=args.state,
        outside_window=chain._outside_window(norb),
        scissor=[frozen_scissor(chain.resolved.outside_shift, p) or 0.0
                 for p in range(norb)], z_prod=z_prod)

    # ------------------------------------------------- the torch process
    os.makedirs(args.workdir, exist_ok=True)
    tag = f'{args.target}_{"gas" if args.gas else args.solvent}_{BASIS}'
    f_in = os.path.join(args.workdir, f'oracle_{tag}_frozen.npz')
    f_out = os.path.join(args.workdir, f'oracle_{tag}_torch.npz')
    dump(f_in, x_mo, d, d_bare, eps, fz, target=args.target, qp_orb=qp_orb,
         value_prod=value_prod, shift_prod=shift_prod, eps_qp_prod=eps_qp_prod,
         omega_prod=om_all, eps_bar=captured['eps_bar'], x_bar=captured['x_bar'],
         d_bar=captured['d_bar'], d_bar_bare=captured['d_bar_bare'])
    say(f'torch process on {f_in}')
    cmd = [sys.executable, os.path.join(HERE, 'torch_oracle.py'), f_in, f_out,
           '--root-mode', args.root_mode, '--newton-steps', str(args.newton_steps)]
    # the torch process pins itself to one thread (its backward segfaults with two
    # on this Rosetta stack); the trace flag is passed through
    rc = subprocess.run(cmd, env=dict(os.environ, OMP_NUM_THREADS='1',
                                      TORCH_CHAIN_TRACE=os.environ.get(
                                          'TORCH_CHAIN_TRACE', ''))).returncode
    if rc != 0 or not os.path.exists(f_out):
        sys.exit(f'the torch process failed (rc {rc})')
    tor = np.load(f_out)
    adj = {'eps': tor['eps_bar'], 'x': tor['x_bar'], 'd': tor['d_bar'],
           'd_bare': None if tor['d_bar_bare'].size == 0 else tor['d_bar_bare']}

    say('\n== swap-in: torch adjoints through production\'s nuclear assembly')
    t0 = time.perf_counter()
    g_swap, dg = _orig_nuclear_gradient(
        mol, mf, captured['auxmol'], captured['crd'], captured['x_mo'], adj['eps'],
        adj['x'], adj['d'], **dict(captured['assembly'], d_bar_bare=adj['d_bare']))
    say(f'  production force  max|g| {np.abs(g_prod).max():.6e} Ha/Bohr')
    say(f'  |g_swap - g_prod| max {np.abs(g_swap - g_prod).max():.2e}   '
          f'rel {np.abs(g_swap - g_prod).max() / np.abs(g_prod).max():.1e}   '
          f'transl. residual {dg["translation_residual"]:.1e}   '
          f'({time.perf_counter() - t0:.1f} s)')
    # The assembly's own floor: the Lagrangian is an iterative solve, so adjoints
    # that differ at round-off can still move the force by more than round-off.
    # Re-feeding production's adjoints must be bitwise; 1e-13 noise on them shows
    # what a round-off-level adjoint difference is worth in Ha/Bohr.
    rng = np.random.default_rng(0)
    noisy = {k: (None if v is None else v * (1.0 + 1e-13 * rng.standard_normal(v.shape)))
             for k, v in (('eps', captured['eps_bar']), ('x', captured['x_bar']),
                          ('d', captured['d_bar']), ('d_bare', captured['d_bar_bare']))}
    g_same = _orig_nuclear_gradient(mol, mf, captured['auxmol'], captured['crd'],
                                    captured['x_mo'], captured['eps_bar'],
                                    captured['x_bar'], captured['d_bar'],
                                    **captured['assembly'])[0]
    g_noisy = _orig_nuclear_gradient(mol, mf, captured['auxmol'], captured['crd'],
                                     captured['x_mo'], noisy['eps'], noisy['x'], noisy['d'],
                                     **dict(captured['assembly'],
                                            d_bar_bare=noisy['d_bare']))[0]
    say(f'  assembly floor: same adjoints again {np.abs(g_same - g_prod).max():.1e}   '
          f'adjoints x (1 + 1e-13 noise) {np.abs(g_noisy - g_prod).max():.1e} Ha/Bohr')

    if args.fd:
        h = args.h
        comps = ([(ia, x) for ia in range(mol.natm) for x in range(3)]
                 if args.fd == 'all' else [tuple(int(v) for v in args.fd.split(','))])
        total = args.fd == 'all' and args.target != 'qp'
        if total:
            g_tot, e_ex0, _ = chain.total_gradient()
        fd = np.full((mol.natm, 3), np.nan)
        fd_tot = np.full((mol.natm, 3), np.nan)
        t0 = time.perf_counter()
        for ia, x in comps:
            vals, tots = [], []
            for k in (-2, -1, 1, 2):
                dR = np.zeros((mol.natm, 3))
                dR[ia, x] = k * h
                m = molecule_at(mol, mol.atom_coords() + dR)
                if args.target == 'qp':
                    vals.append(chain.quasiparticle(args.qp_offset, m))
                elif total:
                    e_ex, _, om = chain.energy(m)
                    vals.append(om)
                    tots.append(e_ex)
                else:
                    vals.append(chain.excitation(m))
            fd[ia, x] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
            if total:
                fd_tot[ia, x] = (tots[0] - 8 * tots[1] + 8 * tots[2] - tots[3]) / (12 * h)
        say(f'\n== finite difference of the whole chain (SCF in the continuum, cavity, '
              f'factors, QP, Casida redone), h = {h:g} Bohr, {len(comps)} component(s), '
              f'{time.perf_counter() - t0:.0f} s')
        say('  atom ax        FD            production     swap-in        d(prod-FD)   d(swap-FD)')
        for ia, x in comps:
            say(f'  {ia:4d} {"xyz"[x]}   {fd[ia, x]:+.9f}   {g_prod[ia, x]:+.9f}   '
                  f'{g_swap[ia, x]:+.9f}   {g_prod[ia, x] - fd[ia, x]:+.2e}    '
                  f'{g_swap[ia, x] - fd[ia, x]:+.2e}')
        ok = np.isfinite(fd)
        scale = np.abs(g_prod[ok]).max()
        say(f'  dOmega/dR: max|prod - FD| {np.abs(g_prod[ok] - fd[ok]).max():.2e}  '
              f'rel {np.abs(g_prod[ok] - fd[ok]).max() / scale:.1e}   '
              f'max|swap - FD| {np.abs(g_swap[ok] - fd[ok]).max():.2e}   '
              f'FD transl. residual {np.abs(np.nansum(fd, axis=0)).max():.1e}')
        if total:
            say(f'  total force dE_ex/dR (E_0 in PCM + Omega): max|g| {np.abs(g_tot).max():.4e}   '
                  f'max|prod - FD| {np.abs(g_tot - fd_tot).max():.2e}   '
                  f'rel {np.abs(g_tot - fd_tot).max() / np.abs(g_tot).max():.1e}   '
                  f'transl. residual {np.abs(g_tot.sum(axis=0)).max():.1e}')

    worst = float(tor['worst_rel'])
    say(f'\nWORST ADJOINT REL {worst:.1e}  ->  '
          + ('AGREE' if worst < 1e-8 else 'DISAGREE (inspect the table)'))
    return {'worst_rel': worst, 'g_prod': g_prod, 'g_swap': g_swap,
            'value_prod': value_prod, 'value_torch': float(tor['value'])}


if __name__ == '__main__':
    main()
