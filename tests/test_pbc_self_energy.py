"""Validation of the periodic GW correlation self-energy Sigma_c.

Machine-precision oracle: at the Gamma-only (nk=1, q=0) limit the periodic
Sigma_c must reduce exactly to the molecular GW.self_energy.SelfEnergySolver's
diagonal GW self-energy, fed the same DF factor and RPA eigenpairs. This
validates the spectral self-energy assembly (chi_a contraction, energy
denominators, occ/virt sign, prefactor).

Finite-q note: a machine-precision finite-q self-energy check (supercell
folding / exact krgw_ac agreement) additionally requires the q->0 Coulomb head
correction (Patterson Sec. II.F) and the exxdiv exchange term, which are NOT
implemented here -- without them the finite-q QP energies
carry a finite-size-dependent shift (occupied ~0.4 Ha from exxdiv, plus a
~10-30 meV head-correction/AC level residual on virtuals vs krgw_ac). The
finite-q momentum bookkeeping of chi_a itself is exercised by the smoke test in
test_pbc_amplitudes and reduces correctly at nk=1 here.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf

from src.SingleReference.Periodic.pbc_integrals import PBCDFIntegrals
from src.SingleReference.Periodic import pbc_self_energy as pse
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.base import get_occ_virt_indices


class _MolSE(SelfEnergySolver):
    def __init__(self, eps, df):
        self.spin_mode = 'restricted'
        self.eps = eps
        self.df_coeff = df
        self.naux = df.shape[0]
        self.block_size = 10000
        self.eta = 1e-3

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)


if __name__ == '__main__':
    all_ok = True

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.2; He 1.5 0 0.6'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.diag([4.0, 4.0, 3.6])
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    dfints = PBCDFIntegrals.from_scf(cell, mf)
    e = np.asarray(mf.mo_energy)
    nocc = dfints.nocc[0]
    L = dfints.L[0][0].real
    eps = e[0].real

    eig = pse.solve_rpa_all_q(dfints, mo_energy=e)
    Om, X, Y = eig[0]

    mol = _MolSE(eps, L)
    chi_a_mol = mol.get_chi_a(nocc, X.real, Y.real)

    # Sigma_c over several orbitals and frequencies
    max_err = 0.0
    for n in range(dfints.nmo):
        for w in (eps[n] - 0.2, eps[n], eps[n] + 0.3):
            s_pbc = pse.sigma_c_diag(dfints, eig, 0, n, w, mo_energy=e)
            s_mol = mol.calculate_self_energy(n, w, nocc, Om.real, chi_a_mol, None, vertex_mode='GW')
            max_err = max(max_err, abs(s_pbc - s_mol))
    ok = max_err < 1e-10
    print(f"1. Sigma_c q=0 vs molecular SelfEnergySolver (all orb, 3 freqs): "
          f"maxerr={max_err:.2e} {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # QP linearization runs and is finite
    qp = pse.qp_energy_g0w0(dfints, eig, 0, nocc - 1, mo_energy=e)
    ok = np.isfinite(qp)
    print(f"2. G0W0 QP linearization runs (HOMO QP={qp:.4f} Ha): {'OK' if ok else 'FAIL'}")
    all_ok &= ok

    # ---- 3. FINITE-q G0W0@HF QP vs pyscf krgw_ac (fc=False), 1x1x2 mesh ----
    # Both omit the q->0 head correction, so they agree at the analytic-
    # continuation level (~few x10 meV). Occupied require the pyscf-computed
    # Sigma_x - v_xc term (get_exchange_minus_vxc); without it they are off by
    # the Madelung constant. This exercises chi_a's finite-q momentum against
    # independent pyscf code.
    from pyscf.pbc.gw import krgw_ac
    cell2 = gto.Cell()
    cell2.atom = 'H 0 0 0; H 0 0 1.2'
    cell2.basis = 'gth-dzvp'
    cell2.pseudo = 'gth-pade'
    cell2.a = np.diag([3.0, 3.0, 2.4])
    cell2.verbose = 0
    cell2.build()
    kpts2 = cell2.make_kpts([1, 1, 2])
    nk2 = len(kpts2)
    mf2 = scf.KRHF(cell2, kpts2, exxdiv='ewald').density_fit()
    mf2.kernel()
    gw = krgw_ac.KRGWAC(mf2)
    gw.fc = False
    gw.verbose = 0
    gw.kernel()
    qp_ref = np.array(gw.mo_energy)

    df2 = PBCDFIntegrals.from_scf(cell2, mf2)
    e2 = np.asarray(mf2.mo_energy)
    eig2 = pse.solve_rpa_all_q(df2, mo_energy=e2)
    vxc2 = pse.get_exchange_minus_vxc(mf2, exxdiv=None)
    nocc2 = df2.nocc[0]
    max_gw = 0.0
    for kn in range(nk2):
        for n in (nocc2 - 1, nocc2):
            mine = pse.qp_energy_g0w0(df2, eig2, kn, n, mo_energy=e2,
                                      exchange_minus_vxc=vxc2[kn, n])
            max_gw = max(max_gw, abs(mine - qp_ref[kn, n]))
    # 5 meV, NOT the old 0.05 Ha. The old tolerance was loose enough to pass at
    # 36.7 meV while Sigma_c was too large by a factor of nkpts; the residual
    # was misread as
    # analytic-continuation error. With the nkpts normalization fixed this lands
    # at ~0.65 meV, which IS the AC level, so the tolerance can be tight enough
    # to catch a recurrence.
    ok = max_gw < 0.005
    print(f"3. finite-q G0W0@HF QP (HOMO/LUMO, 1x1x2) vs krgw_ac fc=False: "
          f"max diff={max_gw*1e3:.1f} meV {'OK' if ok else 'FAIL'} (AC-limited)")
    all_ok &= ok

    # ---- 4. VERTEX self-energy (GWGammaInf, PSD1) q=0 vs molecular ----
    from src.SingleReference.Periodic import pbc_casida as pc
    W_all = pc.build_static_W_all_Q(dfints, mo_energy=e)
    eig_bse = pse.solve_bse_all_q(dfints, W_all, mo_energy=e)
    Omb, Xb, Yb = eig_bse[0]
    chi_a_mol = mol.get_chi_a(nocc, Xb.real, Yb.real)
    chi_b_mol = mol.get_chi_b_vertex(nocc, Xb.real, Yb.real, eri_w=W_all[0].real)
    for mode in ('GWGammaInf', 'PSD1'):
        mx = 0.0
        for n in range(dfints.nmo):
            for w in (eps[n] - 0.2, eps[n], eps[n] + 0.3):
                sp = pse.sigma_vertex_diag(dfints, eig_bse, W_all, 0, n, w,
                                           vertex_mode=mode, mo_energy=e)
                sm = mol.calculate_self_energy(n, w, nocc, Omb.real, chi_a_mol,
                                               chi_b_mol, vertex_mode=mode)
                mx = max(mx, abs(sp - sm))
        ok = mx < 1e-10
        print(f"4. {mode} vertex Sigma q=0 vs molecular (all orb, 3 freqs): "
              f"maxerr={mx:.2e} {'OK' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- 5. finite-q vertex self-energy runs + finite on 1x1x3 ----
    prim = gto.Cell()
    prim.atom = 'H 0 0 0; H 0.9 0.9 0.9'
    prim.basis = 'gth-szv'
    prim.pseudo = 'gth-pade'
    prim.a = np.eye(3) * 2.5
    prim.verbose = 0
    prim.build()
    kp3 = prim.make_kpts([1, 1, 3])
    mf3 = scf.KRHF(prim, kp3).density_fit()
    mf3.kernel()
    df3 = PBCDFIntegrals.from_scf(prim, mf3)
    e3 = np.asarray(mf3.mo_energy)
    W3 = pc.build_static_W_all_Q(df3, mo_energy=e3)
    eig3 = pse.solve_bse_all_q(df3, W3, mo_energy=e3)
    smoke = True
    for kn in range(3):
        for mode in ('GWGammaInf', 'PSD1'):
            v = pse.sigma_vertex_diag(df3, eig3, W3, kn, df3.nocc[kn] - 1,
                                      e3[kn][df3.nocc[kn] - 1].real, vertex_mode=mode,
                                      mo_energy=e3)
            if not np.isfinite(v):
                smoke = False
    print(f"5. finite-q vertex Sigma runs+finite (1x1x3, GWGammaInf/PSD1): "
          f"{'OK' if smoke else 'FAIL'}")
    all_ok &= smoke

    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
