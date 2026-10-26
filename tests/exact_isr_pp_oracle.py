"""Exact intermediate-state representation (ISR) of the polarization
propagator in the determinant space, order by order.

What it computes. For a small closed-shell molecule in the Moller-Plesset
partitioning H(lambda) = F + lambda (H - F) (spin orbitals, occupied first),
the excited intermediate states of Schirmer (Phys. Rev. A 43, 4647 (1991);
Schirmer and Trofimov, J. Chem. Phys. 120, 11449 (2004)) built exactly as
power series in lambda:

    Psi_0         the normalized Rayleigh-Schroedinger ground state
    class 1       C_ia Psi_0 - Psi_0 <Psi_0|C_ia|Psi_0>, Loewdin
                  orthonormalized within the class
    class 2       C_ijab Psi_0 orthogonalized to Psi_0 and to class 1, then
                  Loewdin orthonormalized within the class

with C_ia = a^dag_a a_i and C_ijab = a^dag_a a^dag_b a_j a_i (i < j, a < b),
and, in that basis, the series of

    M(lambda)       <Psi~_I|H - E_0|Psi~_J>
    D^pq(lambda)    <Psi~_I|a^dag_p a_q|Psi~_J> - delta_IJ <Psi_0|a^dag_p a_q|Psi_0>
    F^pq(lambda)    <Psi_0|a^dag_p a_q|Psi~_J>

under the ADC(2) truncation at lambda = 1, by block (singles/singles
through second order, singles/doubles through first, doubles/doubles at
zeroth order; the ground-to-excited row through second order on the singles
and first on the doubles): M whole (`truncated_secular`), D and F
contracted with two vectors (`transition_density`, `ground_transition_density`).

What it is for. The reference of the ISR-ADC(2) transition densities
(eeADC/ee_transition_density): the truncated M is the ADC(2) secular matrix of
eeADC/ee_u_dense_full (checked element by element), and the truncated D and
F, contracted with its eigenvectors, are the state-to-state and
ground-to-excited one-particle transition densities, the antisymmetric parts
included (a state's own density fixes only the symmetric part of a
transition density, so no density-only reference can check them).
sto-3g systems only: the determinant space is dense. The reference is an RHF
converged to an orbital gradient of 1e-11 (`build_system`).
"""
import numpy as np
import scipy.sparse as sp
from pyscf import ao2mo, gto, scf

from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_integrals,
                                            get_orbital_energies)
from src.SingleReference.ADC.eeADC import ee_utils
from src.SingleReference.ADC.isr_engine import lowdin_sqrt_inv_series
from src.SingleReference.CC.determinant_space import apply_ops
from tests.exact_isr_oracle import rspt_series, series_reciprocal
from tests.lambda_oracle import LambdaSelfEnergyOracle

#: Orders 0..ORDER of every series (ADC(2) reads up to second order).
ORDER = 2


def build_system(atom, basis):
    """(eps, h1, g, nocc) in spin orbitals of an RHF converged to an orbital
    gradient of 1e-11, so that the Brillouin condition holds below the
    checks' bars."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-14, 1e-11
    mf.kernel()
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n, n, n, n)
    h1, g = get_antisymmetrized_spin_integrals(
        mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, eri)
    return get_orbital_energies(mf, representation='spin'), h1, g, mol.nelectron


def _op(basis, cre, ann):
    """Sparse a^dag_{cre...} a_{ann...} on the N-electron sector."""
    rows, cols, vals = [], [], []
    for col, det in enumerate(basis.dets):
        r = apply_ops(det, list(cre), list(ann))
        if r is None:
            continue
        ph, nd = r
        i = basis.index.get(nd)
        if i is None:
            continue
        rows.append(i)
        cols.append(col)
        vals.append(float(ph))
    return sp.csr_matrix((vals, (rows, cols)), shape=(basis.dim, basis.dim))


def _mul(a, b):
    """Series product, truncated at ORDER."""
    return [sum(a[i] @ b[k - i] for i in range(k + 1)) for k in range(ORDER + 1)]


def _t(a):
    return [x.T for x in a]


def _lowdin(P):
    """P S^-1/2 with S = P^T P, as series (S^(0) is the identity)."""
    S = _mul(_t(P), P)
    if np.max(np.abs(S[0] - np.eye(S[0].shape[0]))) > 1e-12:
        raise ValueError('precursors not orthonormal at zeroth order')
    Y = lowdin_sqrt_inv_series({k: S[k] for k in range(1, ORDER + 1)}, ORDER)
    return _mul(P, [Y[k] for k in range(ORDER + 1)])


class ExactPPISR:
    """The exact ISR of one molecule (module docstring)."""

    def __init__(self, atom, basis='sto-3g'):
        eps, h1, g, nocc = build_system(atom, basis)
        self.eps, self.g, self.nocc = eps, g, nocc
        self.norb = norb = len(eps)
        orc = LambdaSelfEnergyOracle(eps, h1, g, nocc)
        self.basis = b = orc.bN
        psi, E, nser, _ = rspt_series(orc, nord=ORDER + 1)
        inv_n = series_reciprocal(nser)
        # n^(-1/2) from 1/n: y_m = (z_m - sum_{p=1}^{m-1} y_p y_{m-p}) / 2
        c = [1.0]
        for m in range(1, ORDER + 1):
            c.append(0.5 * (inv_n[m] - sum(c[p] * c[m - p] for p in range(1, m))))
        self.psi0 = [sum(c[k - i] * psi[i] for i in range(k + 1))[:, None]
                     for k in range(ORDER + 1)]
        self.e0 = [E[k] for k in range(ORDER + 1)]
        self.Hf = sp.csr_matrix(orc.Hf['N'])
        self.W = sp.csr_matrix(orc.Hfull['N'] - orc.Hf['N'])
        i_s, a_s = ee_utils.configs_singles(nocc, norb)
        i_d, j_d, a_d, b_d = ee_utils.configs_doubles(nocc, norb)
        singles = [_op(b, [a], [i]) for i, a in zip(i_s, a_s)]
        doubles = [_op(b, [a, bb], [j, i]) for i, j, a, bb in zip(i_d, j_d, a_d, b_d)]
        self.n_s, self.n_d = len(singles), len(doubles)
        ground = self.psi0

        def raw(ops):
            return [np.column_stack([o @ ground[k][:, 0] for o in ops])
                    for k in range(ORDER + 1)]

        def project_out(X, B):
            return [x - y for x, y in zip(X, _mul(B, _mul(_t(B), X)))]

        P1 = project_out(raw(singles), ground)
        psi1 = _lowdin(P1)
        P2 = project_out(project_out(raw(doubles), ground), psi1)
        psi2 = _lowdin(P2)
        self.isr = [np.hstack([a, d]) for a, d in zip(psi1, psi2)]

    # ------------------------------------------------------------ series
    def secular(self):
        """<Psi~_I|H(lambda) - E_0(lambda)|Psi~_J> as a series."""
        hf = _mul(_t(self.isr), [self.Hf @ x for x in self.isr])
        w = _mul(_t(self.isr), [self.W @ x for x in self.isr])
        n = hf[0].shape[0]
        return [hf[k] + (w[k - 1] if k else 0.0) - self.e0[k] * np.eye(n)
                for k in range(ORDER + 1)]

    def truncated_secular(self):
        """The ADC(2) truncation of `secular` at lambda = 1: singles/singles
        through second order, singles/doubles through first, doubles/doubles
        at zeroth."""
        m, ns = self.secular(), self.n_s
        out = np.zeros_like(m[0])
        out[:ns, :ns] = sum(m[k][:ns, :ns] for k in range(3))
        out[:ns, ns:] = sum(m[k][:ns, ns:] for k in range(2))
        out[ns:, :ns] = sum(m[k][ns:, :ns] for k in range(2))
        out[ns:, ns:] = m[0][ns:, ns:]
        return out

    # ------------------------------------------------------ densities
    def _rdm(self, x, y):
        """<x|a^dag_p a_q|y> for every p, q of two determinant vectors."""
        if not hasattr(self, '_ops'):
            self._ops = [[_op(self.basis, [p], [q]) for q in range(self.norb)]
                         for p in range(self.norb)]
        return np.array([[x @ (o @ y) for o in row] for row in self._ops])

    def _parts(self, v):
        """The ISR vector v on the determinants, by class and order."""
        ns = self.n_s
        return ([x[:, :ns] @ v[:ns] for x in self.isr],
                [x[:, ns:] @ v[ns:] for x in self.isr])

    def _rho0(self):
        if not hasattr(self, '_rho0_series'):
            g = [x[:, 0] for x in self.psi0]
            self._rho0_series = [sum(self._rdm(g[a], g[k - a]) for a in range(k + 1))
                                 for k in range(ORDER + 1)]
        return self._rho0_series

    def transition_density(self, vl, vr):
        """gamma_pq = <l|a^dag_p a_q|r> of two ISR-basis vectors under the
        ADC(2) truncation of D~ - rho_0 1 (module docstring), (norb, norb)."""
        ns = self.n_s
        ls, ld = self._parts(vl)
        rs, rd = self._parts(vr)
        rho = self._rho0()
        out = sum(self._rdm(ls[a], rs[b]) for a in range(3) for b in range(3 - a))
        out = out - sum(rho[:3]) * float(vl[:ns] @ vr[:ns])
        out = out + sum(self._rdm(ls[a], rd[b]) + self._rdm(ld[a], rs[b])
                        for a in range(2) for b in range(2 - a))
        return out + self._rdm(ld[0], rd[0]) - rho[0] * float(vl[ns:] @ vr[ns:])

    def ground_transition_density(self, vr):
        """gamma^0r_pq = <Psi_0|a^dag_p a_q|r> under the ADC(2) truncation of
        the ground row (singles through second order, doubles through
        first)."""
        g = [x[:, 0] for x in self.psi0]
        rs, rd = self._parts(vr)
        return (sum(self._rdm(g[a], rs[b]) for a in range(3) for b in range(3 - a))
                + sum(self._rdm(g[a], rd[b]) for a in range(2) for b in range(2 - a)))


def build(atom, basis='sto-3g'):
    return ExactPPISR(atom, basis)
