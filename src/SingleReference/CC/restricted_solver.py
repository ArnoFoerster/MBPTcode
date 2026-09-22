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
from .diis import DIIS


def _t_residual_args(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                     f_aa, g_aaaa, g_abab, o, v):
    """Bundles the positional args every generated T-residual function
    expects, substituting the '_aa'-type array for its '_bb'-type twin."""
    return (t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_aaaaaa,
           f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)


def kernel_t_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                        f_aa, g_aaaa, g_abab, o, v, hf_energy, max_iter=100,
                        stopping_eps=1e-9, diis_size=8, diis_start_cycle=2, verbose=True):
    """Iterate the restricted CCSDT T-amplitude equations to convergence."""
    nv, no = t1_aa.shape
    e_ai, e_abij, e_abcijk = energy_denominators(f_aa, no, nv)
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)

    shapes = [t1_aa.shape, t2_aaaa.shape, t2_abab.shape,
             t3_aaaaaa.shape, t3_aabaab.shape, t3_abbabb.shape]
    sizes = [int(np.prod(s)) for s in shapes]

    diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
    old_vec = np.hstack([t1_aa.ravel(), t2_aaaa.ravel(), t2_abab.ravel(),
                        t3_aaaaaa.ravel(), t3_aabaab.ravel(), t3_abbabb.ravel()])

    old_energy = hf_energy + TR.cc_energy_restricted(t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
                                                     f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)

    if verbose:
        print("    ==> Restricted CCSDT T amplitude equations (DIIS) <==")
        print("     Iter               Energy                 |dE|                 |dT|")

    for it in range(max_iter):
        args = _t_residual_args(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                                f_aa, g_aaaa, g_abab, o, v)
        r1 = TR.t1_aa_residual(*args)
        r2aa = TR.t2_aaaa_residual(*args)
        r2ab = TR.t2_abab_residual(*args)
        r3aaa = TR.t3_aaaaaa_residual(*args)
        r3aab = TR.t3_aabaab_residual(*args)
        r3abb = TR.t3_abbabb_residual(*args)

        res_norm = (np.linalg.norm(r1) + np.linalg.norm(r2aa) + np.linalg.norm(r2ab)
                   + np.linalg.norm(r3aaa) + np.linalg.norm(r3aab) + np.linalg.norm(r3abb))

        new_t1 = (r1 + fock_e_ai * t1_aa) * e_ai
        new_t2aa = (r2aa + fock_e_abij * t2_aaaa) * e_abij
        new_t2ab = (r2ab + fock_e_abij * t2_abab) * e_abij
        new_t3aaa = (r3aaa + fock_e_abcijk * t3_aaaaaa) * e_abcijk
        new_t3aab = (r3aab + fock_e_abcijk * t3_aabaab) * e_abcijk
        new_t3abb = (r3abb + fock_e_abcijk * t3_abbabb) * e_abcijk

        vec = np.hstack([new_t1.ravel(), new_t2aa.ravel(), new_t2ab.ravel(),
                        new_t3aaa.ravel(), new_t3aab.ravel(), new_t3abb.ravel()])
        error_vec = old_vec - vec
        vec = diis_update.compute_new_vec(vec, error_vec)
        old_vec = vec

        idx = 0
        t1_aa = vec[idx:idx + sizes[0]].reshape(shapes[0]); idx += sizes[0]
        t2_aaaa = vec[idx:idx + sizes[1]].reshape(shapes[1]); idx += sizes[1]
        t2_abab = vec[idx:idx + sizes[2]].reshape(shapes[2]); idx += sizes[2]
        t3_aaaaaa = vec[idx:idx + sizes[3]].reshape(shapes[3]); idx += sizes[3]
        t3_aabaab = vec[idx:idx + sizes[4]].reshape(shapes[4]); idx += sizes[4]
        t3_abbabb = vec[idx:idx + sizes[5]].reshape(shapes[5]); idx += sizes[5]

        current_energy = hf_energy + TR.cc_energy_restricted(
            t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa, f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if verbose:
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(
                it, current_energy - hf_energy, delta_e, res_norm))

        if delta_e < stopping_eps and res_norm < stopping_eps:
            break
        old_energy = current_energy
    else:
        raise ValueError("Restricted CCSDT T amplitude iterations did not converge")

    return t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb


def solve_lambda_ccsdt_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                                  f_aa, g_aaaa, g_abab, o, v, max_iter=300,
                                  stopping_eps=1e-8, diis_size=15, diis_start_cycle=2,
                                  verbose=True):
    """Iterate the restricted full Lambda-CCSDT (Lambda1/Lambda2/Lambda3)
    equations to convergence."""
    nv, no = t1_aa.shape
    e_ai, e_abij, e_abcijk = energy_denominators(f_aa, no, nv)
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)
    le_ai = e_ai.transpose(1, 0)
    lfock_e_ai = fock_e_ai.transpose(1, 0)
    le_abij = e_abij.transpose(2, 3, 0, 1)
    lfock_e_abij = fock_e_abij.transpose(2, 3, 0, 1)
    le_abcijk = e_abcijk.transpose(3, 4, 5, 0, 1, 2)
    lfock_e_abcijk = fock_e_abcijk.transpose(3, 4, 5, 0, 1, 2)

    # standard CC-Lambda starting point
    l1_aa = t1_aa.transpose(1, 0).copy()
    l2_aaaa = t2_aaaa.transpose(2, 3, 0, 1).copy()
    l2_abab = t2_abab.transpose(2, 3, 0, 1).copy()
    l3_aaaaaa = t3_aaaaaa.transpose(3, 4, 5, 0, 1, 2).copy()
    l3_aabaab = t3_aabaab.transpose(3, 4, 5, 0, 1, 2).copy()
    l3_abbabb = t3_abbabb.transpose(3, 4, 5, 0, 1, 2).copy()

    shapes = [l1_aa.shape, l2_aaaa.shape, l2_abab.shape,
             l3_aaaaaa.shape, l3_aabaab.shape, l3_abbabb.shape]
    sizes = [int(np.prod(s)) for s in shapes]
    diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
    old_vec = np.hstack([l1_aa.ravel(), l2_aaaa.ravel(), l2_abab.ravel(),
                        l3_aaaaaa.ravel(), l3_aabaab.ravel(), l3_abbabb.ravel()])

    if verbose:
        print("    ==> Restricted Lambda-CCSDT amplitude equations (DIIS) <==")
        print("     Iter          |dL1|          |dL2aa|        |dL2ab|        "
              "|dL3aaa|       |dL3aab|       |dL3abb|")

    t_args = (t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
             t3_aaaaaa, t3_aabaab, t3_abbabb, t3_aaaaaa,
             f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)

    for it in range(max_iter):
        l_args = t_args + (l1_aa, l1_aa, l2_aaaa, l2_abab, l2_aaaa,
                          l3_aaaaaa, l3_aabaab, l3_abbabb, l3_aaaaaa)
        r1 = LR.l1_aa_residual(*l_args).transpose(1, 0)
        r2aa = LR.l2_aaaa_residual(*l_args).transpose(2, 3, 0, 1)
        r2ab = LR.l2_abab_residual(*l_args).transpose(2, 3, 0, 1)
        r3aaa = LR.l3_aaaaaa_residual(*l_args).transpose(3, 4, 5, 0, 1, 2)
        r3aab = LR.l3_aabaab_residual(*l_args).transpose(3, 4, 5, 0, 1, 2)
        r3abb = LR.l3_abbabb_residual(*l_args).transpose(3, 4, 5, 0, 1, 2)

        l1_new = (r1 + lfock_e_ai * l1_aa) * le_ai
        l2aa_new = (r2aa + lfock_e_abij * l2_aaaa) * le_abij
        l2ab_new = (r2ab + lfock_e_abij * l2_abab) * le_abij
        l3aaa_new = (r3aaa + lfock_e_abcijk * l3_aaaaaa) * le_abcijk
        l3aab_new = (r3aab + lfock_e_abcijk * l3_aabaab) * le_abcijk
        l3abb_new = (r3abb + lfock_e_abcijk * l3_abbabb) * le_abcijk

        d1 = np.linalg.norm(l1_new - l1_aa)
        d2aa = np.linalg.norm(l2aa_new - l2_aaaa)
        d2ab = np.linalg.norm(l2ab_new - l2_abab)
        d3aaa = np.linalg.norm(l3aaa_new - l3_aaaaaa)
        d3aab = np.linalg.norm(l3aab_new - l3_aabaab)
        d3abb = np.linalg.norm(l3abb_new - l3_abbabb)

        vec = np.hstack([l1_new.ravel(), l2aa_new.ravel(), l2ab_new.ravel(),
                        l3aaa_new.ravel(), l3aab_new.ravel(), l3abb_new.ravel()])
        error_vec = old_vec - vec
        try:
            vec = diis_update.compute_new_vec(vec, error_vec)
        except np.linalg.LinAlgError:
            # tiny systems can converge so fast the DIIS error vectors become
            # exactly degenerate (singular B-matrix) -- fall back to the
            # undamped update for this step rather than crashing (same
            # fallback as solver.py's generalized solve_lambda_ccsdt).
            pass
        old_vec = vec

        idx = 0
        l1_aa = vec[idx:idx + sizes[0]].reshape(shapes[0]); idx += sizes[0]
        l2_aaaa = vec[idx:idx + sizes[1]].reshape(shapes[1]); idx += sizes[1]
        l2_abab = vec[idx:idx + sizes[2]].reshape(shapes[2]); idx += sizes[2]
        l3_aaaaaa = vec[idx:idx + sizes[3]].reshape(shapes[3]); idx += sizes[3]
        l3_aabaab = vec[idx:idx + sizes[4]].reshape(shapes[4]); idx += sizes[4]
        l3_abbabb = vec[idx:idx + sizes[5]].reshape(shapes[5]); idx += sizes[5]

        if verbose:
            print(f"     {it:4d}   {d1:.3e}      {d2aa:.3e}      {d2ab:.3e}      "
                 f"{d3aaa:.3e}      {d3aab:.3e}      {d3abb:.3e}")

        if (d1 < stopping_eps and d2aa < stopping_eps and d2ab < stopping_eps
               and d3aaa < stopping_eps and d3aab < stopping_eps and d3abb < stopping_eps):
            break
    else:
        raise ValueError("Restricted Lambda-CCSDT iterations did not converge")

    return l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb


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
                                          max_iter=300, verbose=True):
    """Core restricted solve: given the integral dict from
    build_restricted_integrals_from_mf, converge T-CCSDT, full Lambda-CCSDT,
    and assemble the spatial 1-RDM."""
    f_aa, g_aaaa, g_abab = ints['f_aa'], ints['g_aaaa'], ints['g_abab']
    nocc, nvir = ints['nocc'], ints['nvir']
    o, v = ints['o'], ints['v']

    t1_aa = np.zeros((nvir, nocc))
    t2_aaaa = np.zeros((nvir, nvir, nocc, nocc))
    t2_abab = np.zeros((nvir, nvir, nocc, nocc))
    t3_aaaaaa = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t3_aabaab = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t3_abbabb = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))

    t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb = kernel_t_restricted(
        t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
        f_aa, g_aaaa, g_abab, o, v, ints['hf_energy'],
        max_iter=max_iter, stopping_eps=t_stopping_eps, verbose=verbose)

    l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb = solve_lambda_ccsdt_restricted(
        t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
        f_aa, g_aaaa, g_abab, o, v,
        max_iter=max_iter, stopping_eps=l_stopping_eps, verbose=verbose)

    opdm = ccsdt_one_rdm_restricted(t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
                                    l1_aa, l2_aaaa, l2_abab, l3_aaaaaa, l3_aabaab, l3_abbabb, o, v)
    return opdm, (t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb)


def compute_ccsdt_density_matrix_restricted(mf, symmetrize=True, **kwargs):
    """Restricted CCSDT AO-basis 1-RDM from an already-converged, closed-shell
    spin-restricted pyscf mean-field object. Mirrors pipeline.py's
    compute_ccsdt_density_matrix, but via the restricted (spin-blocked)
    T-CCSDT + Lambda-CCSD(T3-informed) pipeline instead of full generalized
    spin-orbital CCSDT."""
    ints = build_restricted_integrals_from_mf(mf)
    opdm, _ = solve_ccsdt_1rdm_restricted_from_ints(ints, **kwargs)
    dm_ao = ints['mo_coeff'] @ opdm @ ints['mo_coeff'].T
    if symmetrize:
        dm_ao = 0.5 * (dm_ao + dm_ao.T)
    return dm_ao
