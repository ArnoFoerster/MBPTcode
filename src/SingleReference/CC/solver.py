"""Lambda-CCSDT iterative solve + CCSDT 1-RDM assembly, given converged T1/T2/T3.

lambdaK_residual() outputs shape (nv,..,no,..) (matching T's convention), but
l1/l2/l3 themselves are consumed elsewhere (e.g. d1_blocks.py) in the generator's
canonical (no,..,nv,..) order -- transpose each residual before use.

Plain fixed-point iteration on the Lambda equations is not contractive at
this system size; DIIS is not optional here.
"""
import numpy as np

from . import lambda1_residual as L1M
from . import lambda2_residual as L2M
from . import lambda3_residual as L3M
from . import d1_blocks as D1M
from .fixed_point import TO_L_LAYOUT, lambda_denominators, solve_lambda


def solve_lambda_ccsdt(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk,
                       max_iter=300, stopping_eps=1e-8, verbose=True,
                       diis_size=15, diis_start_cycle=2, out=None):
    """Lambda-CCSDT multipliers (l1, l2, l3) for converged T.

    out: optional dict; receives 'converged' and 'n_iter'. Non-convergence
        warns (fixed_point.CCNotConverged) and returns the last iterate.
    """
    to_l = TO_L_LAYOUT
    denominators = lambda_denominators((1, 2, 3), e_ai, e_abij, e_abcijk)

    # standard CC-Lambda starting point
    l1 = t1.transpose(1, 0).copy()
    l2 = t2.transpose(2, 3, 0, 1).copy()
    l3 = t3.transpose(3, 4, 5, 0, 1, 2).copy()
    kd = np.eye(t1.shape[0] + t1.shape[1])

    def residuals(ls):
        l1, l2, l3 = ls
        return [L1M.lambda1_residual(t1, t2, t3, l1, l2, l3, fock, g, o, v).transpose(to_l[1]),
                L2M.lambda2_residual(t1, t2, t3, l1, l2, l3, fock, g, o, v).transpose(to_l[2]),
                L3M.lambda3_residual(t1, t2, t3, l1, l2, l3, fock, g, kd, o, v).transpose(to_l[3])]

    (l1, l2, l3), converged, n_iter = solve_lambda(
        residuals, [l1, l2, l3], denominators, ['L1', 'L2', 'L3'], max_iter, stopping_eps,
        diis_size, diis_start_cycle, verbose, 'Lambda-CCSDT')
    if out is not None:
        out['converged'], out['n_iter'] = converged, n_iter
    return l1, l2, l3


def ccsdt_one_rdm(t1, t2, t3, l1, l2, l3, o, v):
    nv, no = t1.shape
    nmo = no + nv
    kd = np.eye(nmo)

    opdm = np.zeros((nmo, nmo))
    opdm[:no, :no] = D1M.d1_oo(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[no:, no:] = D1M.d1_vv(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[:no, no:] = D1M.d1_ov(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[no:, :no] = D1M.d1_vo(t1, t2, t3, l1, l2, l3, kd, o, v)
    return opdm
