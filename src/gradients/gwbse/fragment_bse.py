"""The GW-BSE singles operator of the fragment partition, Tamm-Dancoff or
full: `BSEOperator`, one implementation of the `SinglesOperator` protocol of
`src.fragments.partition` (the partition, its diabats, A_eff(Omega_0), the pole
guard and the dense Feshbach oracle are there, the same for every method).

The BSE is the pencil K v = Omega S v of that module: K = A, S = 1 in the
Tamm-Dancoff approximation; v = (X, Y), K = [[A, B], [B, A]] and
S = diag(1, -1) for the full BSE. K does not depend on Omega (`omega` is
None), so dA_eff/dOmega is the resolvent's -y^T S y alone.

DENSE AND MATRIX-FREE ARE ONE CODE PATH. `BSEOperator` applies K to a block
of vectors, densely from `bse_blocks` for small systems and through the ISDF
block action of the Davidson solver otherwise.

WHICH KERNEL is the chain's: `ExcitedStateChain(bse_tda=...)`, with the
quasiparticle Hamiltonian's off-diagonal one-body blocks where the chain
solves one (`qp.hamiltonian`): A_ia,jb gains delta_ij H_ab - delta_ab H_ij
inside each group (`with_one_body`), so the diabats and A_eff do not depend
on how a group's orbitals rotate into each other, as the chain's own roots
do not.

WHAT IS NOT HERE. No nuclear derivatives: the method-neutral Lagrangian is
`src.gradients.fragment_diabatic` and the GW-BSE reverse pass it seeds is
`src.gradients.gwbse.fragment_diabatic`; their finite-difference reference is
`tests/fragment_diabatic_fd.py`.
"""
import numpy as np

from src.Base.numerics.constants import FRAGMENT_DENSE_MAX
from src.Base.factorization.sliced_factors import SlicedFactors
from src.Base.utils.parallel.mpi_grid import lockstep
from src.gradients.gwbse.bse_isdf import bse_blocks
from src.gradients.gwbse.state_following import dense_one_body
from src.SingleReference.LinearResponse.block_actions import isdf_block_action
from src.SingleReference.LinearResponse.davidson import with_one_body
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)


class BSEOperator:
    """y = K x for the canonical BSE of one spin channel, x of shape (dim, k).

    Tamm-Dancoff (`tda=True`): K = A and dim = n_ov. Full BSE: x stacks the
    (X, Y) halves, K = [[A, B], [B, A]] and dim = 2 n_ov. Built from a chain's
    `shared_forward`, so it is the same kernel that chain's roots and gradients
    use: the same quasiparticle energies and quasiparticle Hamiltonian, the
    same static screening, the same factors.
    """

    #: K does not depend on Omega (the `SinglesOperator` protocol)
    omega = None

    def __init__(self, nocc, eps_qp, apply, dense=None, tda=True,
                 spin='singlet'):
        self.nocc = int(nocc)
        self.spin = spin
        self.eps_qp = np.asarray(eps_qp, float)
        self.nvir = len(self.eps_qp) - self.nocc
        self._apply = apply
        self.dense = dense
        self.tda = bool(tda)

    @property
    def eps(self):
        """The quasiparticle energies: K's zeroth-order diagonal is
        e_a - e_i of these."""
        return self.eps_qp

    @property
    def n_ov(self):
        return self.nocc * self.nvir

    @property
    def dim(self):
        return self.n_ov if self.tda else 2 * self.n_ov

    @property
    def metric(self):
        """(dim,) the diagonal of S: ones, and -1 on a full BSE's Y half."""
        one = np.ones(self.n_ov)
        return one if self.tda else np.concatenate([one, -one])

    def apply(self, x):
        x = np.asarray(x, float)
        vec = x.ndim == 1
        # the ISDF action is collective: every rank hands it rank 0's vectors
        x = lockstep(np.ascontiguousarray(x.reshape(self.dim, -1)), check=True)
        y = self._apply(x)
        return y[:, 0] if vec else y

    def apply_domega(self, x):
        """dK/dOmega: K does not depend on Omega, and the partition never asks
        (`omega` is None)."""
        raise ValueError('the BSE kernel does not depend on Omega')

    @classmethod
    def from_chain(cls, chain, mol=None, mf=None, route='auto'):
        """(operator, `SharedForward`) at one geometry; route 'auto' |
        'dense' | 'isdf'.
        The kernel, Tamm-Dancoff or full, is the chain's (`bse_tda`)."""
        mol, mf = chain.mean_field(mol, mf)
        shared = chain.shared_forward(mol, mf)
        x_mo, d, eps_qp, w_aux = shared.x_mo, shared.d, shared.eps_qp, shared.w_aux
        nocc, tda = chain.nocc, bool(chain.bse_tda)
        nv = len(eps_qp) - nocc
        n_ov = nocc * nv
        one_body = None if shared.qp_h is None else shared.qp_h.one_body(nocc)
        if route == 'auto':
            route = ('dense' if (n_ov if tda else 2 * n_ov) <= FRAGMENT_DENSE_MAX
                     else 'isdf')
        if route == 'dense':
            a, b = bse_blocks(x_mo, d, eps_qp, w_aux, nocc, spin=chain.spin,
                              bse_tda=tda)[:2]
            if one_body is not None:
                a = dense_one_body(a, one_body, nocc, nv)
            k = 0.5 * (a + a.T)
            if not tda:
                b = 0.5 * (b + b.T)
                k = np.block([[k, b], [b, k]])
            return cls(nocc, eps_qp, lambda x: k @ x, dense=k, tda=tda,
                       spin=chain.spin), shared
        if route != 'isdf':
            raise ValueError(f"route must be 'auto', 'dense' or 'isdf', got "
                             f"{route!r}")
        lr = LinearResponseSolver(np.asarray(eps_qp, float),
                                  spin_mode='restricted')
        factors = x_mo if isinstance(x_mo, SlicedFactors) else (x_mo, d)
        apply_ab = with_one_body(isdf_block_action(lr, nocc, True, w_aux,
                                                   factors,
                                                   spin=chain.spin)[0],
                                 one_body)

        def apply(x):
            k = x.shape[1]
            cols = x if tda else np.hstack([x[:n_ov], x[n_ov:]])
            z = np.ascontiguousarray(cols.T.reshape(-1, nocc, nv))
            az, bz = apply_ab(z)[:2]
            az = np.asarray(az).reshape(-1, n_ov).T
            if tda:
                return az
            bz = np.asarray(bz).reshape(-1, n_ov).T
            return np.vstack([az[:, :k] + bz[:, k:], bz[:, :k] + az[:, k:]])
        return cls(nocc, eps_qp, apply, tda=tda, spin=chain.spin), shared
