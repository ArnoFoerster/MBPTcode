"""G0W@CC: GW self-energy built from a coupled-cluster polarizability (Lewis & Berkelbach, JCTC 15, 2925 (2019), Eq. 12).

Same Lehmann structure as GW@RPA/GW@BSE, but with Casida excitation
energies/amplitudes replaced by EOM-CC ones (CCSDT or CCSD, from eom.py):

    Sigma_c,pq(w) = sum_{n,i} V^n_pi Vt^n_qi / (w - eps_i + Omega_n - i eta)
                  + sum_{n,a} V^n_pa Vt^n_qa / (w - eps_a - Omega_n + i eta)
    V^n_pq  = sum_rs (pq|rs) rho_n(rs),   rho_n(rs)  = <Psi~_0|r^dag s|Psi_n>
    Vt^n_pq = sum_rs (pq|rs) rho*_n(rs),  rho*_n(rs) = <Psi~_n|r^dag s|Psi_0>

rho != rho* since Hbar is non-Hermitian. HF reference: no v_xc correction,
so QP eq. is w = eps_p + Re Sigma_c,pp(w).

Spin-orbital, full-ERI. EOM-CCSDT (or CCSD) via eom.py's generated
sigma-vector Davidson solver -- scales far better than exact determinant
enumeration, though still full-ERI/no-symmetry so realistically small-to-
medium systems only. Validated against the paper's Table 1 (H2/He, def2-SVP)
by tests/test_cc_polarizability.py.

Reached from the usual front end as
calc_qp_energy(mf, selfenergy='GW', polarizability='ccsdt', state=...); this
module is the physics object behind that branch, the CC analog of
self_energy.py's SelfEnergySolver.
"""
import numpy as np

from src.Base.numerics.constants import DEFAULT_BROADENING_ETA
from src.Base.pyscf.pyscf_interface import get_two_electron_integrals_chemist
from src.Solvers.qp_equation import solve_qp_equation_pole_strength
from src.SingleReference.CC.eom import EOMCC


class GWCCSelfEnergy:
    """Correlation self-energy with CC screening (Eq. 12 of the paper).

    Parameters: an EOMCC instance (or mf + level to build one). All EE
    eigenstates and transition densities are computed once at construction;
    sigma_c(w) evaluations are then cheap.
    """

    def __init__(self, mf, level='ccsdt', eom=None, nroots=8, verbose=False):
        """nroots: EE states included in the Lehmann sum. Pad generously
        beyond the number of "real" low-lying states you care about --
        non-Hermitian Davidson needs elbow room past the exact size of a
        degenerate cluster (e.g. a p-shell excitation manifold) to reliably
        resolve every member of it. With too few, a member of a cluster may
        find no left eigenvector to pair with: transition_densities warns
        (`UnpairedEOMStates`), gives that state zero transition densities and
        lists it in `unpaired_states`, so its pole drops out of Sigma_c.
        He/def2-SVP's lowest excitation, exactly 3-fold degenerate, is such
        a cluster."""
        self.eom = eom if eom is not None else EOMCC(mf, level=level, verbose=verbose)
        self.res = self.eom.kernel('ee', nroots=nroots)
        rho, rho_star = self.res.transition_densities()

        # Through get_two_electron_integrals_chemist rather than ao2mo directly so
        # that an attached solvent screening (v -> v + vtilde) reaches this
        # self-energy too -- see src/environments/solvent_screening.py.
        eri_so = get_two_electron_integrals_chemist(mf.mol, mf, representation='spin')
        # Kept complex: degenerate EOM roots give complex-conjugate eigenvector
        # pairs whose Im*Im cross term is dropped if truncated to real early.
        self.V = np.einsum('pqrs,nrs->npq', eri_so, rho, optimize=True)
        self.Vt = np.einsum('pqrs,nrs->npq', eri_so, rho_star, optimize=True)

        self.omega_n = self.res.omega            # excitation energies > 0
        self.eps = np.diagonal(self.eom.ints['fock']).copy()
        self.nocc = self.eom.nocc

    def sigma_c(self, w, p, q=None, eta=DEFAULT_BROADENING_ETA):
        """Sigma_c,pq(w) (complex). Diagonal element if q is None."""
        if q is None:
            q = p
        o = slice(None, self.nocc)
        v = slice(self.nocc, None)
        num_o = self.V[:, p, o] * self.Vt[:, q, o]      # (n, i)
        num_v = self.V[:, p, v] * self.Vt[:, q, v]      # (n, a)
        den_o = w - self.eps[o][None, :] + self.omega_n[:, None] - 1j * eta
        den_v = w - self.eps[v][None, :] - self.omega_n[:, None] + 1j * eta
        return np.sum(num_o / den_o) + np.sum(num_v / den_v)

    def solve_qp(self, p, eta=DEFAULT_BROADENING_ETA, static_shift=0.0):
        """Solve w = eps_p + static_shift + Re Sigma_c,pp(w) (HF reference).
        Returns Hartree. static_shift carries any state-independent-of-w term
        the caller adds, e.g. a solvent reaction field."""
        func = lambda w: w - self.eps[p] - static_shift - self.sigma_c(w, p, eta=eta).real
        return solve_qp_equation_pole_strength(func, self.eps[p])
