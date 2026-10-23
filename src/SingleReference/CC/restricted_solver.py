"""Restricted (spin-blocked) CCSDT T-solver + full Lambda-CCSDT solver + 1-RDM.

Mirrors amplitudes.py::kernel + solver.py::solve_lambda_ccsdt/ccsdt_one_rdm,
but over 6 T-blocks (t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab,
t3_abbabb) and 6 L-blocks (l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab,
l3_abbabb) instead of 3+3, mirroring the T side now that Lambda3 is
generated. Wherever a
generated residual function needs a '_bb'/'_bbbb'/'_bbbbbb'-suffixed
argument, the matching '_aa'/'_aaaa'/'_aaaaaa' array is passed in directly --
a genuine numerical identity for a closed-shell reference (alpha and beta
share the same spatial orbitals), not an approximation.

Final 1-RDM: only the '_aa'-block D1 is generated (off-diagonal spin blocks
of a closed-shell 1-RDM vanish exactly, and the beta-diagonal block is
numerically identical to the alpha one), so the full spatial 1-RDM block is
2x the generated block. D1 now includes full Lambda1/Lambda2/Lambda3
feedback.
"""
import numpy as np

from .integrals import build_restricted_integrals_from_mf, energy_denominators
from .generated_restricted import amplitudes_restricted as TR
from .generated_restricted import lambda_residual_restricted as LR
from .generated_restricted import d1_blocks_restricted as D1R
from .fixed_point import (TO_L_LAYOUT, iterate, jacobi_step, lambda_denominators,
                          solve_lambda)


def _t_residual_args(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                     f_aa, g_aaaa, g_abab, o, v):
    """Bundles the positional args every generated T-residual function
    expects, substituting the '_aa'-type array for its '_bb'-type twin."""
    return (t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_aaaaaa,
           f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)


def kernel_t_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                        f_aa, g_aaaa, g_abab, o, v, hf_energy, max_iter=100,
                        stopping_eps=1e-9, diis_size=8, diis_start_cycle=2, verbose=True,
                        out=None):
    """Iterate the restricted CCSDT T-amplitude equations to convergence.

    out: optional dict; receives 'converged' and 'n_iter'. Non-convergence
    warns (fixed_point.CCNotConverged) and returns the last iterate."""
    nv, no = t1_aa.shape
    e_ai, e_abij, e_abcijk = energy_denominators(f_aa, no, nv)
    denominators = [(e, np.reciprocal(e)) for e in
                    (e_ai, e_abij, e_abij, e_abcijk, e_abcijk, e_abcijk)]

    def energy(ts):
        return hf_energy + TR.cc_energy_restricted(ts[0], ts[0], ts[1], ts[2], ts[1],
                                                   f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)

    def step(ts):
        args = _t_residual_args(*ts, f_aa, g_aaaa, g_abab, o, v)
        rs = [TR.t1_aa_residual(*args), TR.t2_aaaa_residual(*args), TR.t2_abab_residual(*args),
              TR.t3_aaaaaa_residual(*args), TR.t3_aabaab_residual(*args),
              TR.t3_abbabb_residual(*args)]
        res_norm = (np.linalg.norm(rs[0]) + np.linalg.norm(rs[1]) + np.linalg.norm(rs[2])
                    + np.linalg.norm(rs[3]) + np.linalg.norm(rs[4]) + np.linalg.norm(rs[5]))
        return [jacobi_step(r, t, d, fd) for r, t, (d, fd) in zip(rs, ts, denominators)], res_norm

    old_energy = energy([t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb])

    def is_converged(it, ts, res_norm):
        nonlocal old_energy
        current_energy = energy(ts)
        delta_e = np.abs(old_energy - current_energy)
        if verbose:
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(
                it, current_energy - hf_energy, delta_e, res_norm), flush=True)
        if delta_e < stopping_eps and res_norm < stopping_eps:
            return True
        old_energy = current_energy
        return False

    if verbose:
        print("    ==> Restricted CCSDT T amplitude equations (DIIS) <==", flush=True)
        print("     Iter               Energy                 |dE|                 |dT|", flush=True)

    ts, converged, n_iter = iterate(
        step, [t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb], is_converged,
        max_iter, diis_size=diis_size, diis_start_cycle=diis_start_cycle,
        what='Restricted CCSDT T amplitude iterations')
    if out is not None:
        out['converged'], out['n_iter'] = converged, n_iter
    return tuple(ts)


def solve_lambda_ccsdt_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                                  f_aa, g_aaaa, g_abab, o, v, max_iter=300,
                                  stopping_eps=1e-8, diis_size=15, diis_start_cycle=2,
                                  verbose=True, out=None):
    """Iterate the restricted full Lambda-CCSDT (Lambda1/Lambda2/Lambda3)
    equations to convergence: the same algorithm as solver.solve_lambda_ccsdt
    (fixed_point.solve_lambda) over the six spin blocks.

    out: optional dict; receives 'converged' and 'n_iter'. Non-convergence
    warns (fixed_point.CCNotConverged) and returns the last iterate."""
    nv, no = t1_aa.shape
    e_ai, e_abij, e_abcijk = energy_denominators(f_aa, no, nv)
    to_l = TO_L_LAYOUT
    ranks = (1, 2, 2, 3, 3, 3)
    denominators = lambda_denominators(ranks, e_ai, e_abij, e_abcijk)

    # standard CC-Lambda starting point
    t_blocks = (t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb)
    l_blocks = [t.transpose(to_l[r]).copy() for t, r in zip(t_blocks, ranks)]

    t_args = _t_residual_args(*t_blocks, f_aa, g_aaaa, g_abab, o, v)
    residual_fns = (LR.l1_aa_residual, LR.l2_aaaa_residual, LR.l2_abab_residual,
                    LR.l3_aaaaaa_residual, LR.l3_aabaab_residual, LR.l3_abbabb_residual)

    def residuals(ls):
        l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb = ls
        l_args = t_args + (l1_aa, l1_aa, l2_aaaa, l2_abab, l2_aaaa,
                          l3_aaaaaa, l3_aabaab, l3_abbabb, l3_aaaaaa)
        return [fn(*l_args).transpose(to_l[r]) for fn, r in zip(residual_fns, ranks)]

    ls, converged, n_iter = solve_lambda(
        residuals, l_blocks, denominators, ['L1', 'L2aa', 'L2ab', 'L3aaa', 'L3aab', 'L3abb'],
        max_iter, stopping_eps, diis_size, diis_start_cycle, verbose, 'Restricted Lambda-CCSDT')
    if out is not None:
        out['converged'], out['n_iter'] = converged, n_iter
    return tuple(ls)


def ccsdt_one_rdm_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                             l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb, o, v):
    """Restricted spatial 1-RDM (nmo x nmo, nmo = no+nv spatial orbitals).
    Off-diagonal spin blocks vanish exactly for a closed-shell reference;
    each generated D1 block is the alpha (== beta) diagonal block, so the
    spatial density is 2x it. Includes full Lambda1/Lambda2/Lambda3
    feedback."""
    nv, no = t1_aa.shape
    nmo = no + nv
    d_aa = np.eye(nmo)
    args = (t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_aaaaaa,
           None, None, None, None, None, o, v,
           l1_aa, l1_aa, l2_aaaa, l2_abab, l2_aaaa,
           l3_aaaaaa, l3_aabaab, l3_abbabb, l3_aaaaaa, d_aa)
    # D1 blocks don't reference f/g at all (CC 1-body density only involves
    # t/l amplitudes + the identity, same as the generalized d1_blocks.py) --
    # pass None for f_aa/f_bb/g_aaaa/g_abab/g_bbbb positions, never touched.
    opdm = np.zeros((nmo, nmo))
    opdm[:no, :no] = 2.0 * D1R.d1_oo_aa(*args)
    opdm[no:, no:] = 2.0 * D1R.d1_vv_aa(*args)
    opdm[:no, no:] = 2.0 * D1R.d1_ov_aa(*args)
    opdm[no:, :no] = 2.0 * D1R.d1_vo_aa(*args)
    return opdm


def solve_ccsdt_1rdm_restricted_from_ints(ints, t_stopping_eps=1e-9, l_stopping_eps=1e-8,
                                          max_iter=300, verbose=True, out=None):
    """Core restricted solve: given the integral dict from
    build_restricted_integrals_from_mf, converge T-CCSDT, full Lambda-CCSDT,
    and assemble the spatial 1-RDM.

    out: optional dict; receives 't_converged' and 'lambda_converged'."""
    f_aa, g_aaaa, g_abab = ints['f_aa'], ints['g_aaaa'], ints['g_abab']
    nocc, nvir = ints['nocc'], ints['nvir']
    o, v = ints['o'], ints['v']

    t1_aa = np.zeros((nvir, nocc))
    t2_aaaa = np.zeros((nvir, nvir, nocc, nocc))
    t2_abab = np.zeros((nvir, nvir, nocc, nocc))
    t3_aaaaaa = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t3_aabaab = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t3_abbabb = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))

    t_out, l_out = {}, {}
    t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb = kernel_t_restricted(
        t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
        f_aa, g_aaaa, g_abab, o, v, ints['hf_energy'],
        max_iter=max_iter, stopping_eps=t_stopping_eps, verbose=verbose, out=t_out)

    l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb = solve_lambda_ccsdt_restricted(
        t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
        f_aa, g_aaaa, g_abab, o, v,
        max_iter=max_iter, stopping_eps=l_stopping_eps, verbose=verbose, out=l_out)
    if out is not None:
        out['t_converged'] = t_out['converged']
        out['lambda_converged'] = l_out['converged']

    opdm = ccsdt_one_rdm_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                                    l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb, o, v)
    return opdm, (t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb)


def compute_ccsdt_density_matrix_restricted(mf, symmetrize=True, **kwargs):
    """Restricted CCSDT AO-basis 1-RDM from an already-converged, closed-shell
    spin-restricted pyscf mean-field object. Mirrors pipeline.py's
    compute_ccsdt_density_matrix, but via the restricted (spin-blocked)
    T-CCSDT + full Lambda-CCSDT pipeline instead of full generalized
    spin-orbital CCSDT. kwargs go to solve_ccsdt_1rdm_restricted_from_ints
    (including out= for the convergence flags)."""
    ints = build_restricted_integrals_from_mf(mf)
    opdm, _ = solve_ccsdt_1rdm_restricted_from_ints(ints, **kwargs)
    dm_ao = ints['mo_coeff'] @ opdm @ ints['mo_coeff'].T
    if symmetrize:
        dm_ao = 0.5 * (dm_ao + dm_ao.T)
    return dm_ao
