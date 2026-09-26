"""Validation of the self-contained periodic RI-RPA (k-point direct RPA) and the
AUTO Fermi-Dirac Coulomb damping.

Oracles:
  A  Gamma-only RI-RPA  ==  molecular pyscf gw.rpa.RPA        (absolute scale)
  B  2x1x1 RI-RPA       ==  GDF k-RPA via pyscf krgw_ac        (q-summation)
  C  2x1x1 RI-RPA /cell ==  Gamma 2x1x1 supercell             (kernel-agnostic, exact)
  G  gauss_legendre     ==  minimax (repo grids) at converged nw
  D  fixed-r0 damped 2x1x1/cell == Gamma supercell            (damped kernel assembly)
  E  indefinite Coulomb metric raises instead of being silently truncated

Run from the repo root:  python tests/test_pbc_rpa.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import super_cell

from src.SingleReference.Periodic.pbc_rpa import (ri_rpa_ecorr, make_auxcell,
                                                  coulomb_metric_inv_sqrt,
                                                  ri_rpa_ecorr_from_dfints)
from src.SingleReference.Periodic.pbc_rpa_damping import (make_coulG_damped,
                                                          nyquist_params)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG

NW = 24
MESH = [15, 15, 15]


def _hchain_cell():
    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.diag([4.0, 4.0, 2.8])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()
    return cell


# ---- GDF k-RPA oracle (rides pyscf krgw_ac; validation cross-check only) ----
def _gdf_krpa_ecorr(kmf, nw=NW):
    from pyscf import lib
    from pyscf.pbc.gw import krgw_ac
    from pyscf.pbc.gw.krgw_ac import get_rho_response, _conc_mos
    from pyscf.ao2mo import _ao2mo
    from src.Base.utils.grids import gauss_legendre_grid
    einsum = lib.einsum

    gw = krgw_ac.KRGWAC(kmf); gw.fc = False
    mf = gw._scf
    mo_energy = np.array(mf.mo_energy); mo_coeff = np.array(mf.mo_coeff)
    nkpts = gw.nkpts; kpts = gw.kpts; nmo = gw.nmo; mydf = gw.with_df
    kscaled = gw.mol.get_scaled_kpts(kpts); kscaled -= kscaled[0]
    freqs, wts = gauss_legendre_grid(nw, w0=0.5)
    e_corr = 0.0
    for kL in range(nkpts):
        Lij = []; kidx = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                kc = -kscaled[i] + kscaled[j] + kscaled[kL]
                if np.linalg.norm(np.round(kc) - kc) < 1e-12:
                    kidx[i] = j
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop(
                            [kpti, kptj], max_memory=0.1 * mf.max_memory, compact=False):
                        Lpq.append(LpqR + LpqI * 1.0j)
                    Lpq = np.vstack(Lpq).reshape(-1, nmo ** 2)
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, [], None, out=None)
                    Lij.append(Lij_out.reshape(-1, nmo, nmo))
        Lij = np.asarray(Lij); naux = Lij.shape[1]
        for w in range(nw):
            Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
            sign, logdet = np.linalg.slogdet(np.eye(naux) - Pi)
            e_corr += wts[w] / (2.0 * np.pi) * (logdet + np.trace(Pi).real)
    return e_corr / nkpts


def test_gamma_vs_molecular():
    """A: Gamma-only periodic RI-RPA reproduces the molecular RPA correlation energy."""
    cell = _hchain_cell()
    kmf = scf.KRHF(cell, cell.make_kpts([1, 1, 1])); kmf.exxdiv = None; kmf.kernel()
    ec = ri_rpa_ecorr(kmf, nw=NW, mesh=MESH)
    from pyscf.gw.rpa import RPA
    mf = scf.RHF(cell, exxdiv=None).density_fit(); mf.kernel()
    rpa = RPA(mf); rpa.kernel(nw=NW)
    diff = abs(ec - rpa.e_corr)
    print(f"  [A] RI-RPA Gamma={ec: .8f}  molecular={rpa.e_corr: .8f}  diff={diff:.2e}")
    assert diff < 5e-4, diff        # RI + FFT-mesh vs GDF


def test_kmesh_vs_gdf_and_supercell():
    """B + C: q-summation vs GDF krpa, and exact k-mesh <-> Gamma-supercell identity."""
    cell = _hchain_cell()
    kmf = scf.KRHF(cell, cell.make_kpts([2, 1, 1])); kmf.exxdiv = None; kmf.kernel()
    ec_k = ri_rpa_ecorr(kmf, nw=NW, mesh=MESH)

    kmf2 = scf.KRHF(cell, cell.make_kpts([2, 1, 1])).density_fit()
    kmf2.exxdiv = None; kmf2.kernel()
    ec_gdf = _gdf_krpa_ecorr(kmf2, nw=NW)
    print(f"  [B] RI-RPA 2x1x1={ec_k: .8f}  GDF krpa={ec_gdf: .8f}  diff={abs(ec_k-ec_gdf):.2e}")
    assert abs(ec_k - ec_gdf) < 5e-4, abs(ec_k - ec_gdf)

    scell = super_cell(cell, [2, 1, 1]); scell.verbose = 0
    smf = scf.KRHF(scell, scell.make_kpts([1, 1, 1])); smf.exxdiv = None; smf.kernel()
    ec_s = ri_rpa_ecorr(smf, nw=NW, mesh=[MESH[0]*2, MESH[1], MESH[2]]) / 2.0
    print(f"  [C] RI-RPA 2x1x1/cell={ec_k: .8f}  Gamma supercell={ec_s: .8f}  diff={abs(ec_k-ec_s):.2e}")
    assert abs(ec_k - ec_s) < 1e-7, abs(ec_k - ec_s)


def test_grids_gauss_legendre_vs_minimax():
    """G: the repo gauss_legendre and minimax grids agree at converged nw."""
    cell = _hchain_cell()
    kmf = scf.KRHF(cell, cell.make_kpts([2, 1, 1])); kmf.exxdiv = None; kmf.kernel()
    ec_gl = ri_rpa_ecorr(kmf, nw=16, mesh=MESH, grid='gauss_legendre')
    ec_mm = ri_rpa_ecorr(kmf, nw=16, mesh=MESH, grid='minimax')
    print(f"  [G] gauss_legendre={ec_gl: .8f}  minimax={ec_mm: .8f}  diff={abs(ec_gl-ec_mm):.2e}")
    assert abs(ec_gl - ec_mm) < 1e-5, abs(ec_gl - ec_mm)


def test_damped_kernel_supercell_identity():
    """D: at fixed r0 the damped kernel preserves the k-mesh <-> supercell identity."""
    cg = make_coulG_damped(r0=8.0, beta=2.0)
    cell = _hchain_cell()
    kmf = scf.KRHF(cell, cell.make_kpts([2, 1, 1])); kmf.exxdiv = None; kmf.kernel()
    ec_k = ri_rpa_ecorr(kmf, nw=NW, mesh=MESH, coulG_fn=cg)
    scell = super_cell(cell, [2, 1, 1]); scell.verbose = 0
    smf = scf.KRHF(scell, scell.make_kpts([1, 1, 1])); smf.exxdiv = None; smf.kernel()
    ec_s = ri_rpa_ecorr(smf, nw=NW, mesh=[MESH[0]*2, MESH[1], MESH[2]], coulG_fn=cg) / 2.0
    print(f"  [D] damped 2x1x1/cell={ec_k: .8f}  Gamma supercell={ec_s: .8f}  diff={abs(ec_k-ec_s):.2e}")
    assert abs(ec_k - ec_s) < 1e-6, abs(ec_k - ec_s)


def test_weighted_q_sum_reduces_to_uniform():
    """[F] The weighted-q sum reduces EXACTLY to the hardcoded uniform average.

    `ri_rpa_ecorr_from_dfints` sums E_c(Q) over whatever transfers its integrals
    object carries, with weights; `ri_rpa_ecorr` hardcodes a uniform 1/nkpts
    average over the regular mesh. Handed regular-grid integrals the two are the
    same sum, so they must agree to machine precision.

    This is the ONLY thing pinning the weighted path -- w_Q = 1/nkpts is the only weighting anything constructs, so
    a regression in the weighted sum would otherwise be invisible.
    """
    cell = _hchain_cell()
    kmf = scf.KRHF(cell, cell.make_kpts([2, 1, 1]))
    kmf.exxdiv = None
    kmf.conv_tol = 1e-10
    kmf.kernel()
    ref = ri_rpa_ecorr(kmf, nw=NW, mesh=MESH)
    ec = ri_rpa_ecorr_from_dfints(build_dfintegrals_coulG(kmf, mesh=MESH), nw=NW)
    print(f"  [F] weighted-q={ec: .10f}  uniform={ref: .10f}  diff={abs(ec - ref):.2e}")
    assert abs(ec - ref) < 1e-12, abs(ec - ref)


def test_low_dim_coulomb_metric_rejected():
    """[E] A 2D cell makes the RI-V metric indefinite; that must raise, not truncate.

    pyscf's get_coulG returns v(G=0) = -pi L_z^2 / 2 for cell.dimension == 2, so
    J picks up a negative eigenvalue -- and it is the LARGEST in magnitude, i.e.
    exactly what an `e > tol * e.max()` null-space filter would throw away without
    a word. Also pins the positive-semidefinite path as bit-identical to that
    filter, so 3D results are untouched.
    """
    from pyscf.pbc import tools
    from pyscf.pbc.df import ft_ao
    from pyscf import lib

    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.diag([4.0, 4.0, 12.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = [9, 9, 25]
    cell.verbose = 0
    cell.build()

    aux = make_auxcell(cell)
    Gv, _, kws = cell.get_Gv_weights(cell.mesh)
    q = np.zeros(3)
    vG = tools.get_coulG(cell, k=q, mesh=cell.mesh, Gv=Gv) * np.asarray(kws)
    assert vG.min() < 0, "2D get_coulG is expected to be negative at G=0"
    auxG = ft_ao.ft_ao(aux, Gv, kpt=q)
    J = lib.einsum('gP,gQ->PQ', auxG.conj() * vG[:, None], auxG)

    e = np.linalg.eigvalsh(J)
    assert (e < 0).any() and abs(e.min()) > e.max(), (e.min(), e.max())
    try:
        coulomb_metric_inv_sqrt(J)
    except ValueError as err:
        print(f"  [E] 2D metric rejected (min eig {e.min():.4g}, max {e.max():.4g})")
        assert 'not positive definite' in str(err)
    else:
        raise AssertionError("indefinite Coulomb metric was silently accepted")

    # positive-semidefinite path: unchanged relative to the plain eigenvalue filter
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12)) + 1j * rng.normal(size=(30, 12))
    P = X.conj().T @ X
    P = np.vstack([np.hstack([P, P[:, :2]]), np.hstack([P[:2], P[:2, :2]])])
    ev, U = np.linalg.eigh(P)
    keep = ev > 1e-10 * ev.max()
    ref = (U[:, keep] * (ev[keep] ** -0.5)) @ U[:, keep].conj().T
    assert np.array_equal(ref, coulomb_metric_inv_sqrt(P))
    print("  [E] rank-deficient PSD metric: bit-identical to the old filter")


if __name__ == '__main__':
    print("test_pbc_rpa:")
    test_gamma_vs_molecular()
    test_kmesh_vs_gdf_and_supercell()
    test_grids_gauss_legendre_vs_minimax()
    test_damped_kernel_supercell_identity()
    test_weighted_q_sum_reduces_to_uniform()
    test_low_dim_coulomb_metric_rejected()
    print("ALL PASSED")
