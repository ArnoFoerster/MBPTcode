"""AB-G0W0: the dRPA bosons compressed into an RI basis, no frequency integral.

Toelle and Chan, J. Chem. Phys. 160, 164108 (2024). In the quasi-boson picture
G0W0 is frequency independent,

  Sigma_pp(w) = sum_{j,nu} (W^nu_pj)^2/(w - eps_j + Om_nu)
              + sum_{b,nu} (W^nu_pb)^2/(w - eps_b - Om_nu),

exact once the Om_nu are the Casida eigenvalues, and O(n_ov^3) for that reason.
The AB expansion replaces the boson operators by a smaller RI basis,
b^+_nu ~= sum_Q C^Q_nu b^+_Q (their Eqs. 17-24), and re-solves the dRPA exactly
inside it (their Eq. 28), so the boson count falls from n_ov to at most naux --
quadratic to linear in system size.

With the symmetric orthogonalization carried out, the basis is the dominant
eigenvectors of S = C_ov C_ov^T carried into the particle-hole space,

  S = P E P^T,   C_ab = C_ov^T P E^{-1/2},

orthonormal by construction, and capped by the RANK of the three-index factors.
The compression is therefore a different, smaller space in which the whole dRPA
is re-solved, not a subset of the exact bosons: selecting the M lowest exact
bosons fails because the coupling weight is spread across the spectrum.

The route carries no validity wall and reaches core states, which is what a
fit to imaginary-axis data cannot do. Accuracy is set by the auxiliary basis: a
standard RI basis leaves tens of meV, and the paper's even-tempered AB bases
(their Eqs. 34-37) are what reach the few-meV level.

Poles and amplitudes come out in the sum-over-poles layout -- (Om_nu, |W^nu_pq|^2)
-- so a sum-over-poles self-energy serves this route unchanged.
"""
import numpy as np

from src.Base.constants import AB_RCOND
from src.SingleReference.LinearResponse.casida import CasidaSolver


def _drpa_bosons(a, b):
    """(Om, X+Y) for factor-2 singlet blocks (A, B), from the production Casida solve.

    The quasi-boson amplitude of Eq. (28) is Casida's X+Y in the standard RPA
    normalization (X+Y)^T (A+B) (X+Y) = Om, so no separate eigenproblem is
    needed. The sign of each column is the eigensolver's gauge; every consumer
    here squares the coupling built from it.
    """
    om, x, y = CasidaSolver(a, b).solve()
    return om, x + y


def ab_basis(c_ov, rcond=AB_RCOND, n_bosons=None):
    """C_ab, (n_ov, N_AB): the orthonormal auxiliary boson basis, Eqs. (21)-(24).

    S^{-1/2} and the eigenvectors of S collapse to one scaling, so no inverse
    square root is formed.

    n_bosons: keep only this many, largest eigenvalue first.
    """
    c_ov = np.asarray(c_ov, float)
    s = c_ov @ c_ov.T
    e, p = np.linalg.eigh(0.5 * (s + s.T))
    order = np.argsort(e)[::-1]
    e, p = e[order], p[:, order]
    keep = e > rcond * e[0]
    if n_bosons is not None:
        keep[int(n_bosons):] = False
    return c_ov.T @ (p[:, keep] / np.sqrt(e[keep]))


def ab_bosons(c_ov, d, c_ab):
    """(Om_Q, XY): the dRPA solved exactly inside the AB basis, Eq. (28).

    Factor-2 singlet blocks, A = diag(d) + 2 V and B = 2 V, as everywhere else
    here, so the boson energies compare directly with a full Casida solve.
    """
    cc = np.asarray(c_ov, float) @ c_ab                     # (naux, N_AB)
    v = cc.T @ cc
    a = (c_ab * np.asarray(d, float)[:, None]).T @ c_ab + 2.0 * v
    return _drpa_bosons(a, 2.0 * v)


def ab_couplings(bp, c_ov, c_ab, xy):
    """W^Q_pq, (norb, N_AB): electron-boson coupling in the AB basis, Eqs. (26), (29).

    The bare V_pq,nu = (pq|ia) never appears in the particle-hole basis; it goes
    from the three-index factors straight into the compressed one.
    """
    return np.sqrt(2.0) * ((np.asarray(bp, float).T @ (c_ov @ c_ab)) @ xy)


def ab_from_factors(bp, c_ov, d, n_bosons=None, rcond=AB_RCOND, c_ab=None):
    """(poles, amplitudes) for one state, in the sum-over-poles layout.

    amplitudes[Q, q] = (W^Q_pq)^2. C_ab does not depend on p, so a caller
    solving a quasiparticle set builds it once and passes it in.
    """
    if c_ab is None:
        c_ab = ab_basis(c_ov, rcond=rcond, n_bosons=n_bosons)
    om, xy = ab_bosons(c_ov, d, c_ab)
    w = ab_couplings(bp, c_ov, c_ab, xy)
    return om, (w ** 2).T


def exact_bosons(c_ov, d):
    """(Om_nu, XY) from the full Casida problem: the O(n_ov^3) reference.

    What the AB expansion approximates. n_ov is quadratic in system size.
    """
    v = np.asarray(c_ov, float).T @ np.asarray(c_ov, float)
    return _drpa_bosons(np.diag(np.asarray(d, float)) + 2.0 * v, 2.0 * v)


def exact_from_factors(bp, c_ov, d):
    """(poles, amplitudes) for the full boson set, same layout as AB."""
    om, xy = exact_bosons(c_ov, d)
    w = np.sqrt(2.0) * ((np.asarray(bp, float).T @ c_ov) @ xy)
    return om, (w ** 2).T
