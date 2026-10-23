"""EOM-CCSDT (EE/IP/EA) regression tests: src/SingleReference/CC/eom.py.

Architecture (see eom.py's module docstring): generated sigma-vector
equations (src/SingleReference/CC/generated_eom/, produced by
the generator) diagonalized with pyscf's non-Hermitian Davidson
(pyscf.lib.davidson_nosym1) -- the same matvec+diag+davidson_nosym1
architecture pyscf's own eom_rccsd/eom_gccsd use.

Oracle: full CCSDT vs determinant_space. EOM-CCSD is retired here (it is
pyscf's); level='ccsd' is refused, which check_ccsd_level_is_refused gates.
  Full CCSDT (level='ccsdt', genuinely nonzero T3) vs
     src/SingleReference/CC/determinant_space.py, an exact (exponential-scaling,
     small-system-only) determinant-space diagonalization of the SAME
     projected Hbar -- the only available check on the triples-containing
     sigma equations, since pyscf has no EOM-CCSDT.

Also exercises transition densities / static polarizability (needed for
src/SingleReference/GW/cc_polarizability.py): validated against determinant_space
directly (machine precision) and, independently, against finite-field FCI for
H2 (where EOM-CCSDT is exact) -- a fully independent numerical check with no
shared code path.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, fci

from src.SingleReference.CC.eom import EOMCC
from src.SingleReference.CC import determinant_space as ds
from src.Base.pyscf.pyscf_interface import get_dipole_integrals

LIH_STO3G = dict(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g', verbose=0)
H2_STO3G = dict(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)


def check_ccsd_level_is_refused():
    """EOM-CCSD is retired here (it is pyscf's): level='ccsd' is refused."""
    mf = scf.RHF(gto.M(**H2_STO3G)).run()
    try:
        EOMCC(mf, level='ccsd')
    except ValueError as exc:
        ok = 'pyscf' in str(exc)
    else:
        ok = False
    print(f"level='ccsd' refused with a pointer to pyscf: {'OK' if ok else 'FAIL'}")
    return ok


def check_left_right_consistency():
    mol = gto.M(**LIH_STO3G)
    mf = scf.RHF(mol).run()
    eom = EOMCC(mf, level='ccsdt', t_stopping_eps=1e-10, max_iter=200)
    all_ok = True
    for sector in ('ee', 'ip', 'ea'):
        # at CCSDT the EA roots need the tighter settings the determinant-space
        # check uses (default tol/max_space leaves 4e-7 between the sides)
        opts = dict(tol=1e-12, max_cycle=200, max_space=40)
        res_r = eom.kernel(sector, nroots=4, left=False, **opts)
        res_l = eom.kernel(sector, nroots=4, left=True, **opts)
        d = np.max(np.abs(np.sort(res_r.omega) - np.sort(res_l.omega)))
        ok = d < 1e-7
        all_ok &= ok
        print(f"{sector.upper()} left/right eigenvalue consistency (max diff {d:.2e}): "
              f"{'OK' if ok else 'FAIL'}")
    return all_ok


def check_ccsdt_vs_determinant_space():
    mol = gto.M(**LIH_STO3G)
    mf = scf.RHF(mol).run()
    eom = EOMCC(mf, level='ccsdt', t_stopping_eps=1e-10, max_iter=200)
    no, nv = eom.nocc, eom.nvir
    norb = no + nv
    t1, t2, t3 = eom.t1, eom.t2, eom.t3
    ok_t3 = np.linalg.norm(t3) > 1e-3
    print(f"T3 genuinely nonzero for LiH (|t3|={np.linalg.norm(t3):.2e}): {'OK' if ok_t3 else 'FAIL'}")

    # LiH's EE spectrum has a 6-fold degenerate cluster just past the 4th
    # root: request extra EE roots so a dropped member of it doesn't land
    # inside the compared window, and a wider Krylov space (max_space=60) so
    # reaching that cluster still converges the lowest 6 roots to the
    # default tol.
    nstates_ee = 6
    margin = 4
    res_ee = eom.kernel('ee', nroots=nstates_ee + margin, max_space=60)
    res_ip = eom.kernel('ip', nroots=4)
    res_ea = eom.kernel('ea', nroots=4, tol=1e-12, max_cycle=200, max_space=40)

    basis_ee = ds.SectorBasis(norb, no, no)
    Hbar_ee, _, _ = ds.build_hbar(basis_ee, eom.ints['soei'], eom.ints['g'], t1, t2, t3, no)
    idx = basis_ee.manifold_indices(3)
    idx = idx[idx != basis_ee.index[(1 << no) - 1]]
    w_ee = np.sort(np.linalg.eigvals(np.asarray(Hbar_ee[np.ix_(idx, idx)].todense())).real) - eom.e_cc

    basis_ip = ds.SectorBasis(norb, no - 1, no)
    Hbar_ip, _, _ = ds.build_hbar(basis_ip, eom.ints['soei'], eom.ints['g'], t1, t2, t3, no)
    idx = basis_ip.manifold_indices(2)
    w_ip = np.sort(np.linalg.eigvals(np.asarray(Hbar_ip[np.ix_(idx, idx)].todense())).real) - eom.e_cc

    basis_ea = ds.SectorBasis(norb, no + 1, no)
    Hbar_ea, _, _ = ds.build_hbar(basis_ea, eom.ints['soei'], eom.ints['g'], t1, t2, t3, no)
    idx = basis_ea.manifold_indices(3)
    w_ea = np.sort(np.linalg.eigvals(np.asarray(Hbar_ea[np.ix_(idx, idx)].todense())).real) - eom.e_cc

    d_ee = np.max(np.abs(np.sort(res_ee.omega)[:nstates_ee] - w_ee[:nstates_ee]))
    d_ip = np.max(np.abs(np.sort(res_ip.omega) - w_ip[:4]))
    d_ea = np.max(np.abs(np.sort(res_ea.omega) - w_ea[:4]))

    ok = d_ee < 1e-6 and d_ip < 1e-6 and d_ea < 1e-6
    print(f"CCSDT EE vs determinant_space (max diff {d_ee:.2e}): {'OK' if d_ee < 1e-6 else 'FAIL'}")
    print(f"CCSDT IP vs determinant_space (max diff {d_ip:.2e}): {'OK' if d_ip < 1e-6 else 'FAIL'}")
    print(f"CCSDT EA vs determinant_space (max diff {d_ea:.2e}): {'OK' if d_ea < 1e-6 else 'FAIL'}")
    return ok and ok_t3


def check_transition_density_vs_determinant_space():
    """rho_n(pq)/rho_star_n(pq) (from the generated_eom/ee_density.py
    equations, see eom.py's transition_densities()) vs the exact
    determinant_space sandwich <(1+Lambda)|e^-T p^dag q e^T|R_n>, full
    CCSDT level, LiH/sto-3g."""
    from src.SingleReference.CC import solver as ccsolver
    from src.SingleReference.CC.integrals import energy_denominators
    from src.SingleReference.CC.generated_eom import ee_density as ED

    mol = gto.M(**LIH_STO3G)
    mf = scf.RHF(mol).run()
    eom = EOMCC(mf, level='ccsdt', t_stopping_eps=1e-10, max_iter=200)
    no, nv = eom.nocc, eom.nvir
    norb = no + nv
    o, v = slice(None, no), slice(no, None)
    t1, t2, t3 = eom.t1, eom.t2, eom.t3
    e_ai, e_abij, e_abcijk = energy_denominators(eom.ints['fock'], no, nv)
    lam1, lam2, lam3 = ccsolver.solve_lambda_ccsdt(t1, t2, t3, eom.ints['fock'], eom.ints['g'],
                                                    o, v, e_ai, e_abij, e_abcijk,
                                                    stopping_eps=1e-9, verbose=False)

    res_r = eom.kernel('ee', nroots=4)
    n = 3
    amps = res_r.amplitudes(n)
    r1n, r2n, r3n = amps['r1'], amps['r2'], amps['r3']

    kd = np.eye(norb)
    mine = np.zeros((norb, norb))
    mine[o, o] = ED.rho_oo(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
    mine[v, v] = ED.rho_vv(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
    mine[o, v] = ED.rho_ov(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
    mine[v, o] = ED.rho_vo(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)

    basis = ds.SectorBasis(norb, no, no)
    Hbar, E, Einv = ds.build_hbar(basis, eom.ints['soei'], eom.ints['g'], t1, t2, t3, no)
    ref = (1 << no) - 1
    iref = basis.index[ref]
    bra = np.zeros(basis.dim)
    bra[iref] = 1.0
    lam_t = ds.build_t_operator(basis, lam1.T, lam2.transpose(2, 3, 0, 1),
                                lam3.transpose(3, 4, 5, 0, 1, 2), no)
    bra_full = bra + (lam_t @ bra)
    r_t = ds.build_t_operator(basis, r1n, r2n, r3n, no)
    ket_r = r_t @ bra
    bra_lifted = Einv.T @ bra_full
    ket_lifted = E @ ket_r

    exact = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            Npq = ds.build_one_body_operator(basis, p, q)
            exact[p, q] = bra_lifted @ (Npq @ ket_lifted)

    d = np.max(np.abs(mine - exact))
    ok = d < 1e-8
    print(f"CCSDT transition density rho_n vs determinant_space (max diff {d:.2e}): "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_polarizability_vs_finite_field_fci():
    """H2/sto-3g static polarizability alpha_zz: EOM-CCSDT (exact for 2
    electrons) vs a fully independent finite-field FCI calculation -- no
    shared code path with rho_n/transition_densities at all."""
    mol = gto.M(**H2_STO3G)
    mf = scf.RHF(mol).run()
    eom = EOMCC(mf, level='ccsdt', t_stopping_eps=1e-11, max_iter=200)
    res_r = eom.kernel('ee', nroots=4)
    dip_so = get_dipole_integrals(mf.mol, mf, representation='spin')
    alpha = res_r.polarizability(dip_so, omega_grid=[0.0])[0]

    def fci_energy(field):
        h1 = mf.get_hcore() + field * mol.intor('int1e_r')[2]
        mf2 = scf.RHF(mol)
        mf2.get_hcore = lambda *args: h1
        mf2.verbose = 0
        mf2.kernel()
        e, _ = fci.FCI(mf2).kernel()
        return e

    fld = 0.005
    ep, e0, em = fci_energy(fld), fci_energy(0.0), fci_energy(-fld)
    alpha_zz_fd = -(ep - 2 * e0 + em) / fld ** 2

    d = abs(alpha[2, 2] - alpha_zz_fd)
    ok = d < 1e-4
    print(f"H2 static alpha_zz: EOM-CCSDT={alpha[2,2]:.6f}, finite-field FCI={alpha_zz_fd:.6f} "
          f"(diff {d:.2e}, dominated by finite-difference discretization): {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    all_ok = True
    all_ok &= check_ccsd_level_is_refused()
    all_ok &= check_left_right_consistency()
    all_ok &= check_ccsdt_vs_determinant_space()
    all_ok &= check_transition_density_vs_determinant_space()
    all_ok &= check_polarizability_vs_finite_field_fci()
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
