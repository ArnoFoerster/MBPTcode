"""G0W@HF screened by the exact (FCI) density response of a 2-electron closed
shell: the oracle of the G0W@CC route, whose EOM-CC polarizability is exact for
two electrons.

The Lehmann sum runs over every FCI excited state of pyscf's direct_spin1
solver, rho_n(rs) = <0|r^dag s|n> and rho*_n(rs) = <n|r^dag s|0> spin-summed,
V^n_pq = sum_rs (pq|rs) rho_n(rs), with the quasiparticle equation of
src/SingleReference/GW/cc_polarizability.py (HF orbital energies, broadening
eta), solved by bracketing on a grid around the HF orbital energy.
"""
import numpy as np
from pyscf import ao2mo, fci
from scipy.optimize import brentq

from src.Base.numerics.constants import DEFAULT_BROADENING_ETA, HARTREE_TO_EV


def fci_screened_g0w_ip(mf, eta=DEFAULT_BROADENING_ETA):
    """HOMO IP (eV) of G0W@HF screened by the FCI response of the 2-electron
    closed shell mf: the root nearest the HF orbital energy."""
    mol = mf.mol
    norb, nocc = mf.mo_coeff.shape[1], mol.nelectron // 2
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-13
    ndet = norb * norb
    e, c = solver.kernel(h1, eri, norb, (1, 1), nroots=ndet, max_space=ndet + 10)
    V, Vt, omega = [], [], []
    for en, cn in zip(e[1:], c[1:]):
        rho = solver.trans_rdm1(c[0], cn, norb, (1, 1))
        rho_star = solver.trans_rdm1(cn, c[0], norb, (1, 1))
        V.append(np.einsum('pqrs,rs->pq', eri, rho))
        Vt.append(np.einsum('pqrs,rs->pq', eri, rho_star))
        omega.append(en - e[0])
    V, Vt, omega = np.array(V), np.array(Vt), np.array(omega)
    eps, p = mf.mo_energy, nocc - 1

    def qp(w):
        num_o = V[:, p, :nocc] * Vt[:, p, :nocc]
        num_v = V[:, p, nocc:] * Vt[:, p, nocc:]
        sigma = (np.sum(num_o / (w - eps[None, :nocc] + omega[:, None] - 1j * eta))
                 + np.sum(num_v / (w - eps[None, nocc:] - omega[:, None] + 1j * eta)))
        return w - eps[p] - sigma.real
    grid = np.linspace(eps[p] - 0.1, eps[p] + 0.1, 2001)
    values = [qp(w) for w in grid]
    roots = [brentq(qp, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if values[i] * values[i + 1] < 0]
    return -min(roots, key=lambda w: abs(w - eps[p])) * HARTREE_TO_EV
