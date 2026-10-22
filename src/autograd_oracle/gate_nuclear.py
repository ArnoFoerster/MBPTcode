"""Production side of the NUCLEAR autograd oracle: run the chain, dump everything.

Builds the solvated `ExcitedStateChain` exactly as `gate_factors.py` does, evaluates production's dOmega/dR and the total force, and writes to an
.npz: the shell tables, the frozen cavity, the frozen ISDF layout, the
converged density, the frozen many-body conventions, reference intermediates
for pinning, and the production gradients. Then it runs `torch_nuclear_oracle.py`
in a separate process (torch never shares a process with pyscf here). Run as
`python -m src.autograd_oracle.gate_nuclear`; `main(argv)` returns the numbers.
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
from pyscf.solvent import pcm as pyscf_pcm
from pyscf.data import radii as pyscf_radii

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.geometry import molecule_at
from src.Base.numerics import Numerics
from src.Base.constants import (HARTREE_TO_EV, ISDF_FIT_REGULARIZATION, KAPPA,
                                SURFACE_GRID_ACCURACY)
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.real_screening import residue_set
from src.SingleReference.GW.qp_states import frozen_scissor
from src.gradients.excited_state import ExcitedStateChain
from src.autograd_oracle.frozen import (FrozenConventions, dump,
                                        shell_table_arrays)


def say(*args, **kwargs):
    """print, flushed: the progress lines must interleave with the torch process."""
    print(*args, flush=True, **kwargs)


H2O_C1 = 'O 0.02 0.01 0.117; H 0.03 0.757 -0.468; H -0.05 -0.757 -0.45'


def shell_table_from_mole(mol):
    """(atom_of, ls, exps, coeffs, c2s) of a Mole in the engine's conventions:
    libcint's stored coefficients carry the radial normalization that
    `bas_ctr_coeff` strips off."""
    atom_of, ls, exps, coeffs, c2s = [], [], [], [], []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        atom_of.append(mol.bas_atom(ib))
        ls.append(l)
        exps.append(mol.bas_exp(ib))
        coeffs.append(mol._libcint_ctr_coeff(ib))
        c2s.append(gto.cart2sph(l))
    return atom_of, ls, exps, coeffs, c2s



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='bse', choices=['bse', 'qp'])
    ap.add_argument('--gas', action='store_true')
    ap.add_argument('--solvent', default='water')
    ap.add_argument('--spin', default='singlet', choices=['singlet', 'triplet'])
    ap.add_argument('--tda', action='store_true')
    ap.add_argument('--state', type=int, default=0)
    ap.add_argument('--qp-offset', type=int, default=0)
    ap.add_argument('--atom', default=H2O_C1)
    ap.add_argument('--basis', default='cc-pvdz')
    ap.add_argument('--auxbasis', default=None)
    ap.add_argument('--workdir', default=os.environ.get('TMPDIR', HERE))
    ap.add_argument('--fd', default=None, metavar='IA,X|all',
                    help='also finite-difference the whole production chain')
    ap.add_argument('--h', type=float, default=1e-3)
    args = ap.parse_args(argv)
    BASIS = args.basis
    AUX = args.auxbasis or paired_auxbasis(BASIS)

    def scf_factory(mol):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
        mf.kernel()
        assert mf.converged
        return mf

    t0 = time.perf_counter()
    mol = gto.M(atom=args.atom, basis=BASIS, verbose=0)
    env = None if args.gas else SolventScreening(mol, solvent=args.solvent)
    chain = ExcitedStateChain(
        mol, scf_factory, spin=args.spin, state=args.state, bse_tda=args.tda,
        environment=env,
        numerics=Numerics(
            bse={'solver': 'dense'},
            isdf={'basis': BASIS, 'auxbasis': AUX,
                  'counts': resolve_isdf_grid(SURFACE_GRID_ACCURACY, BASIS)[0]}))
    mol, mf = chain.mean_field()
    nocc, norb = chain.nocc, len(mf.mo_energy)
    say(f'{"gas phase" if args.gas else "PCM(" + args.solvent + ")"}: {chain.label()}   '
          f'nao {mol.nao}  M {chain.M}  naux {chain.naux}   {time.perf_counter() - t0:.1f} s')

    # ------------------------------------------------ production gradients
    t0 = time.perf_counter()
    qp_orb = nocc - 1 + args.qp_offset
    if args.target == 'qp':
        g_omega, diags = chain.quasiparticle_gradient(args.qp_offset)
        value_prod = diags['qp_energy']
    else:
        g_omega, diags = chain.excitation_gradient()
        value_prod = diags['omega']
    g_scf = np.asarray(mf.Gradients().kernel())
    say(f'production {args.target}: value {value_prod:.10f} Ha ({value_prod * HARTREE_TO_EV:.6f} eV)'
          f'   gradients {time.perf_counter() - t0:.1f} s   nfreq_cd {chain.conventions.nfreq_cd}')

    # ------------------------------------------------ frozen many-body conventions
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
    for p, r in ro['routes'].items():
        if r not in ('explicit', 'none') and residue_set(eps, nocc, roots[p]):
            sys.exit(f'orbital {p} took the {r!r} residue route; only explicit is ported')
    shift_vec = np.zeros(norb) if shift_prod is None else np.asarray(shift_prod, float)
    xc_all = np.asarray(chain._xc_correction(mf, np.arange(norb), shift_vec), float)
    residues = {p: residue_set(eps, nocc, roots[p]) for p in roots}
    fz = FrozenConventions(
        nocc=nocc, mu=mu, states=states, xc_const=xc_all, roots=roots, residues=residues,
        gw_tau=chain.conventions.gw_grid.tau_points, gw_cosft=chain.conventions.gw_grid.cosft_wt, nu=chain.conventions.nu,
        wt=chain.conventions.wt, w_tau=chain.w_grid.tau_points, w_cosft=chain.w_grid.cosft_wt,
        spin_kappa=KAPPA[args.spin], bse_tda=args.tda, state=args.state,
        outside_window=chain._outside_window(norb),
        scissor=[frozen_scissor(chain.resolved.outside_shift, p) or 0.0
                 for p in range(norb)], z_prod=z_prod)
    say(f'window {states}   residues ' + ', '.join(f'{p}:{len(residues[p])}' for p in roots))

    # ------------------------------------------------ constants for the torch side
    extra = {}
    extra.update(shell_table_arrays(*shell_table_from_mole(mol), 'mol'))
    extra.update(shell_table_arrays(*shell_table_from_mole(auxmol), 'aux'))
    extra.update(coords=mol.atom_coords(), charges=mol.atom_charges().astype(float),
                 nelec=mol.nelectron, solvated=not args.gas, target=args.target, qp_orb=qp_orb,
                 value_prod=value_prod, g_omega=g_omega, g_scf=g_scf, e_tot=float(mf.e_tot),
                 mo_energy=np.asarray(mf.mo_energy), mo_coeff=np.asarray(mf.mo_coeff),
                 D_ref=mf.make_rdm1(), fock=mf.get_fock(),
                 regularization=ISDF_FIT_REGULARIZATION)
    # ISDF layout, points, frames
    mu_i, nu_i, w_l = chain.layout
    extra.update(lay_mu=mu_i, lay_nu=nu_i, lay_w=w_l,
                 pts_local=np.vstack(chain.pts_local), pt_owner=chain.owner, frames=chain.frames)
    if chain.body is not None and not chain.body.fixed:
        extra.update(ref_coords=chain.body.reference)
    _, Mfit_ref, V_ref = chain.factorization.shareable_factors(mol, auxmol, crd)
    extra.update(Mfit_ref=Mfit_ref, D_ref_dressed=d, X_mo_ref=x_mo,
                 D_ref_bare=(np.zeros(0) if d_bare is None else d_bare))
    # cavity
    if args.gas:
        for k in ('cav_owner', 'cav_normals', 'cav_weights', 'cav_r_vdw', 'cav_xi', 'cav_R_J',
                  'cav_R_in_J', 'cav_R_sw_J'):
            extra[k] = np.zeros(0)
        extra.update(eps_opt=1.0, eps_static=1.0, pcm_method='IEF-PCM')
    else:
        pcm = env._pcm
        surf = pcm.surface
        owner = np.concatenate([np.full(p1 - p0, ia) for ia, (p0, p1) in enumerate(surf['gslice_by_atom'])])
        rad = env._cavity['vdw_scale'] * pyscf_pcm.modified_Bondi + env._cavity['r_probe'] / pyscf_radii.BOHR
        extra.update(cav_owner=owner, cav_normals=surf['norm_vec'], cav_weights=surf['weights'],
                     cav_r_vdw=surf['R_vdw'], cav_xi=surf['charge_exp'],
                     cav_R_J=rad[mol.atom_charges()], cav_R_in_J=surf['R_in_J'],
                     cav_R_sw_J=surf['R_sw_J'], eps_opt=env.eps, eps_static=env.eps_static,
                     pcm_method=env.method,
                     K_o=pcm._intermediates['K'], R_o=pcm._intermediates['R'],
                     Q_o=env.response_matrix(), vtilde_ref=env.aux_kernel(auxmol),
                     grid_ref=surf['grid_coords'], F_ref=surf['switch_fun'],
                     A_ref=pcm._intermediates['A'],
                     K_s=mf.with_solvent._intermediates['K'], R_s=mf.with_solvent._intermediates['R'],
                     e_pcm=float(mf.with_solvent.e), scf_grid=mf.with_solvent.surface['grid_coords'])
        if not np.allclose(surf['grid_coords'], mf.with_solvent.surface['grid_coords']):
            sys.exit('the SCF cavity and the screening cavity differ')

    os.makedirs(args.workdir, exist_ok=True)
    tag = f'{args.target}_{"gas" if args.gas else args.solvent}_{BASIS}'
    f_in = os.path.join(args.workdir, f'nuclear_{tag}_frozen.npz')
    f_out = os.path.join(args.workdir, f'nuclear_{tag}_torch.npz')
    dump(f_in, x_mo, d, d_bare, eps, fz, **extra)
    say(f'torch process on {f_in}')
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'torch_nuclear_oracle.py'), f_in, f_out],
                        env=dict(os.environ, OMP_NUM_THREADS='1')).returncode
    if rc != 0 or not os.path.exists(f_out):
        sys.exit(f'the torch process failed (rc {rc})')

    tor = np.load(f_out)
    if args.fd:
        h = args.h
        comps = ([(ia, x) for ia in range(mol.natm) for x in range(3)]
                 if args.fd == 'all' else [tuple(int(v) for v in args.fd.split(','))])
        fd = np.full((mol.natm, 3), np.nan)
        fd_e0 = np.full((mol.natm, 3), np.nan)
        t0 = time.perf_counter()
        for ia, x in comps:
            vals, e0s = [], []
            for k in (-2, -1, 1, 2):
                dR = np.zeros((mol.natm, 3))
                dR[ia, x] = k * h
                m = molecule_at(mol, mol.atom_coords() + dR)
                if args.target == 'qp':
                    vals.append(chain.quasiparticle(args.qp_offset, m))
                    e0s.append(chain.mean_field(m)[1].e_tot)
                else:
                    e_ex, e_0, om = chain.energy(m)
                    vals.append(om)
                    e0s.append(e_0)
            fd[ia, x] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
            fd_e0[ia, x] = (e0s[0] - 8 * e0s[1] + 8 * e0s[2] - e0s[3]) / (12 * h)
        ok = np.isfinite(fd)
        g_t = tor['g_value']
        say(f'\n== finite difference of the whole production chain, h = {h:g}, '
              f'{len(comps)} component(s), {time.perf_counter() - t0:.0f} s')
        say(f'  d{args.target}/dR:  max|prod - FD| {np.abs(g_omega[ok] - fd[ok]).max():.2e}   '
              f'max|torch - FD| {np.abs(g_t[ok] - fd[ok]).max():.2e}   '
              f'FD transl. residual {np.abs(np.nansum(fd, axis=0)).max():.1e}')
        say(f'  dE_0/dR:   max|pyscf - FD| {np.abs(g_scf[ok] - fd_e0[ok]).max():.2e}   '
              f'max|torch - FD| {np.abs(tor["g_e0"][ok] - fd_e0[ok]).max():.2e}')
    return {'g_omega_prod': g_omega, 'g_omega_torch': tor['g_value'],
            'g_e0_pyscf': g_scf, 'g_e0_torch': tor['g_e0'],
            'pins_ok': bool(tor['pins_ok']), 'value_prod': value_prod}


if __name__ == '__main__':
    main()
