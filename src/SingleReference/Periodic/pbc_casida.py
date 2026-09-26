"""Periodic RPA polarizability at momentum transfer q -- the k-point
generalization of LinearResponse.linear_response's RPA blocks and
solve_rpa_screening_df, feeding the complex-Hermitian CasidaSolver
(Sander/Maggio/Kresse, PRB 92, 045209).

Two products:

- build_chi0_aux: the non-interacting RPA polarizability Pi0^q in the DF
  auxiliary basis, Pi0 = (4/nkpts) sum L_ov f L_ov^H. This is the ingredient
  the screened W^q is built from, and it matches
  pyscf.pbc.gw.krgw_ac.get_rho_response to machine precision -- the trustworthy
  finite-q oracle for the momentum bookkeeping (pyscf's tdscf get_ab is BUGGY
  at finite kshift and disagrees with its own gen_vind, so it is NOT used).

- build_rpa_matrices: the Kresse time-reversal-adapted Casida blocks A(q),B(q)
  for the RPA/TDDFT bubble. Per Kresse Eqs. 32-33/56-58, in that basis A and B
  are both Hermitian and, for the Hartree (bubble) kernel, the coupling is the
  SAME Hermitian matrix V in A and B (only A carries the extra diagonal
  A - B = energy differences). So:

      A = diag(e_a - e_i) + (2/nkpts) V_dir,   B = (2/nkpts) V_dir,
      V_dir[(ki,ia),(kj,jb)] = (a_ka i_ki | j_kj b_kb) = (M^H M)[.,.],

  with M[L,(ki,i,a)] = L_ov(ki -> ka=kconserv[ki]) the occ-virt DF factor
  (bra conjugated). A + B coupling is then (4/nkpts) V_dir, consistent with
  build_chi0_aux's (4/nkpts) prefactor, and reduces to the molecular
  A = D + 2 V_iajb at nkpts=1.

The transition space at momentum transfer q is the electron-hole pairs
(i @ ki, a @ ka=kconserv[ki]) over all ki, flattened (ki, i, a). `kconserv`
is passed in directly (dfints.kconserv_pair[q] in production), so this code is
agnostic to the q-labeling convention -- Lblock resolves the k-pairs.
"""
import numpy as np
from pyscf.pbc.lib.kpts_helper import conj_mapping
from src.SingleReference.LinearResponse.casida import CasidaSolver


def _transition_layout(dfints, kconserv):
    """(offsets, N, nocc, nvirt) for the (ki,i,a) flattened transition space."""
    nk = dfints.nkpts
    nocc = dfints.nocc
    nvirt = dfints.nmo - nocc
    sizes = [int(nocc[ki] * nvirt[ki]) for ki in range(nk)]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return offsets, int(offsets[-1]), nocc, nvirt


def _stacked_Lov(dfints, kconserv):
    """M[L, (ki,i,a)]: occ-virt DF factor stacked over ki, and the flat layout.

    M is the bra-conjugated occ-virt block so that V_dir = M^H M reproduces the
    direct Coulomb (a_ka i_ki | j_kj b_kb) and Pi0 = (4/nk) M f M^H reproduces
    get_rho_response.
    """
    offsets, N, nocc, nvirt = _transition_layout(dfints, kconserv)
    # naux from the blocks actually used at THIS transfer, not from L[0]: naux
    # can differ between momentum transfers in GDF, so L[0] need not carry the
    # right auxiliary momentum.
    naux = dfints.Lblock(0, kconserv[0]).shape[0]
    M = np.zeros((naux, N), dtype=np.complex128)
    for ki in range(dfints.nkpts):
        ka = kconserv[ki]
        Lov = dfints.Lblock(ki, ka)[:, :nocc[ki], nocc[ka]:]     # (naux, no, nv)
        M[:, offsets[ki]:offsets[ki + 1]] = Lov.reshape(naux, -1)
    return M, offsets, N, nocc, nvirt


def _diag_energies(dfints, kconserv, mo_energy):
    offsets, N, nocc, nvirt = _transition_layout(dfints, kconserv)
    d = np.zeros(N)
    for ki in range(dfints.nkpts):
        ka = kconserv[ki]
        dia = (mo_energy[ka, nocc[ka]:][None, :] - mo_energy[ki, :nocc[ki]][:, None])
        d[offsets[ki]:offsets[ki + 1]] = dia.ravel()
    return d


def build_chi0_aux(dfints, kconserv, omega, mo_energy=None, imaginary=True, eta=0.0):
    """Non-interacting RPA polarizability Pi0^q(omega) in the DF auxiliary basis.

    Matches pyscf.pbc.gw.krgw_ac.get_rho_response on the imaginary axis:
        Pi0_PQ = (4/nkpts) sum_{ki, i in occ, a in virt}
                 L_ov[P,ia] * f(omega, e_i - e_a) * conj(L_ov[Q,ia]),
    with f = d/(omega^2 + d^2) (imaginary axis, d = e_i - e_a < 0), or the
    real-axis f = 1/(omega - d + i eta) - 1/(omega + d + i eta) style response
    when imaginary=False. Hermitian (imaginary axis) / complex (real axis).
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    nk = dfints.nkpts
    naux = dfints.Lblock(0, kconserv[0]).shape[0]   # this transfer's aux dim, not L[0]'s

    Pi = np.zeros((naux, naux), dtype=np.complex128)
    for ki in range(nk):
        ka = kconserv[ki]
        nocc_i = dfints.nocc[ki]
        nocc_a = dfints.nocc[ka]
        Lov = dfints.Lblock(ki, ka)[:, :nocc_i, nocc_a:]         # (naux, no, nv)
        d = mo_energy[ki, :nocc_i][:, None] - mo_energy[ka, nocc_a:][None, :]   # e_i - e_a
        if imaginary:
            f = d / (omega**2 + d * d)
        else:
            f = (1.0 / (omega - d + 1j * eta) - 1.0 / (omega + d + 1j * eta))
        Pf = Lov * f[None]
        Pi += 4.0 / nk * np.einsum('Pia,Qia->PQ', Pf, Lov.conj())
    return Pi


def build_rpa_matrices(dfints, kconserv, mo_energy=None):
    """Kresse RPA/TDDFT Casida blocks A(q), B(q), both complex Hermitian.

    A = diag(e_a - e_i) + (2/nkpts) V_dir,  B = (2/nkpts) V_dir, with
    V_dir = M^H M the direct Coulomb bubble coupling. A - B is diagonal
    (positive energy differences), so the CasidaSolver squaring trick applies
    at essentially no cost (Kresse Sec. II.D). Feed straight to
    CasidaSolver(A, B).solve().
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    w = 1.0 / dfints.nkpts

    M, offsets, N, nocc, nvirt = _stacked_Lov(dfints, kconserv)
    d_all = _diag_energies(dfints, kconserv, mo_energy)

    V_dir = M.conj().T @ M                       # Hermitian PSD, (ai|jb)
    A = np.diag(d_all).astype(np.complex128) + 2.0 * w * V_dir
    B = 2.0 * w * V_dir
    return A, B


def solve_rpa_spectral(dfints, kconserv, mo_energy=None):
    """RPA eigenpairs at momentum transfer q, projected to the aux basis.

    Solves the Kresse RPA Casida problem (build_rpa_matrices) with the complex
    CasidaSolver and returns (Omega, rho) where Omega are the RPA excitation
    energies and rho[P, S] = sum_{ki,ia} M[P,(ki,ia)] (X+Y)_S[(ki,ia)] is the
    aux-basis projection of the transition eigenvectors -- the periodic
    analogue of LinearResponse.solve_rpa_spectral_df's XplusY_proj.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    M, offsets, N, nocc, nvirt = _stacked_Lov(dfints, kconserv)
    A, B = build_rpa_matrices(dfints, kconserv, mo_energy=mo_energy)
    Omega, X, Y = CasidaSolver(A, B).solve()
    rho = M @ (X + Y)
    return Omega, rho


def build_W_aux_spectral(dfints, Omega, rho, omega=0.0, imaginary=True, eta=0.0):
    """Screened interaction W^q(omega) in the aux basis from RPA eigenpairs.

    Spectral form (cross-checked against the (1-Pi0)^{-1} inversion to 1e-15 at
    all frequencies; the -4/nkpts constant is fixed by that match):

        W(i nu) = I - (4/nkpts) sum_S rho_S rho_S^H * Omega_S / (nu^2 + Omega_S^2)

    On the real axis (imaginary=False), the retarded continuation
    (i nu -> omega) of Omega/(nu^2+Omega^2) = 1/2 [1/(Omega-omega) +
    1/(Omega+omega)]:

        W(omega) = I - (4/nkpts) sum_S rho_S rho_S^H * 1/2
                       [1/(Omega_S - omega - i eta) + 1/(Omega_S + omega + i eta)]

    Both forms coincide at omega = nu = 0, the static limit. rho, Omega from
    solve_rpa_spectral.
    """
    naux = rho.shape[0]
    c = 4.0 / dfints.nkpts
    if imaginary:
        denom = Omega / (omega**2 + Omega**2)
    else:
        denom = 0.5 * (1.0 / (Omega - omega - 1j * eta)
                       + 1.0 / (Omega + omega + 1j * eta))
    W = np.eye(naux, dtype=np.complex128) - c * (rho * denom[None]) @ rho.conj().T
    return W


def build_W_aux_inversion(dfints, kconserv, omega, mo_energy=None, imaginary=True, eta=0.0):
    """Screened interaction W^q(omega) in the aux basis by direct inversion,
    W = (I - Pi0(omega))^{-1}, with Pi0 from build_chi0_aux. Independent of the
    Casida eigensolve -- the cross-check oracle for build_W_aux_spectral."""
    Pi = build_chi0_aux(dfints, kconserv, omega, mo_energy=mo_energy,
                        imaginary=imaginary, eta=eta)
    naux = Pi.shape[0]
    return np.linalg.inv(np.eye(naux) - Pi)


# ---------------------------------------------------------------------------
# Full BSE per momentum transfer q
# ---------------------------------------------------------------------------

def build_static_W_all_Q(dfints, mo_energy=None):
    """Static screened interaction W^Q in the aux basis for every momentum
    transfer Q, as a list indexed by Q (= dfints.kconserv_pair row index).

    W^Q[P,P'] is what screens the BSE direct term at internal transfer Q =
    k - k'. Built from the Kresse RPA eigenpairs at omega=0 (imaginary axis
    static limit), so W^Q is Hermitian in the aux basis at momentum Q, shared
    with every L-block of a k-pair at that transfer.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    W_all = []
    for Q in range(dfints.nkpts):
        Omega, rho = solve_rpa_spectral(dfints, dfints.kconserv_pair[Q], mo_energy=mo_energy)
        W_all.append(build_W_aux_spectral(dfints, Omega, rho, omega=0.0, imaginary=True))
    return W_all


def _screened_block(dfints, W_all, k1, k2, k3, k4, s1, s2, s3, s4):
    """Screened 4-index MO block sum_{PP'} L^{k1,k2}_{P,pq} W^Q_{PP'}
    L^{k3,k4}_{P',rs} for orbital ranges s1..s4, with Q the momentum transfer
    of the (k1,k2) pair. Reduces to eri_block when W^Q = I.

    Requires k1-k2+k3-k4 = 0 so the two L-blocks share aux momentum Q = k2-k1.
    """
    Q = dfints._pair_to_q(k1, k2)
    L12 = dfints.Lblock(k1, k2)[:, s1, s2]
    L34 = dfints.Lblock(k3, k4)[:, s3, s4]
    return np.einsum('Ppq,PR,Rrs->pqrs', L12, W_all[Q], L34)


def build_bse_matrices(dfints, q, W_all, mo_energy=None):
    """Full (non-TDA) BSE Casida blocks A(q), B(q) at momentum transfer q, in
    the Kresse time-reversal-adapted basis (both complex Hermitian).

    Singlet BSE kernel (2 v - W), the k-point generalization of
    LinearResponse._build_block_df's lBSE branch:

        A[K,J] = delta (e_a,ka - e_i,ki)
                 + (2/nk) (a_ka i_ki | j_kj b_kb)          [e-h exchange, v^q]
                 - (1/nk) W_{ij,ab}                         [screened direct, Fig 2(c)]
        B[K,J] = (2/nk) (a_ka i_ki | j_kj b_kb)
                 - (1/nk) W_{i b~, j~ a}                    [screened, Fig 2(d)]

    with K=(i,a,ki), J=(j,b,kj), ka=kconserv[ki], kb=kconserv[kj]. The exchange
    (Hartree) carries the BSE transfer q and is the SAME Hermitian block in A
    and B (Kresse Eq. 57). The screened direct term of A carries internal
    transfer kj-ki. For B, the antiresonant partner J sits at time-reversed
    momenta (Kresse Eq. 58: the Y/antiresonant component lives at -k-q), so its
    virtual b and occupied j are pulled from the conjugate k-points
    b~ = conj(kb), j~ = conj(kj); this is what makes B momentum-conserving AND
    Hermitian at finite q (the naive kb,kj port carries 2q and is neither).
    Both screened terms carry 1/nk (bare part 1/nk, correlation 1/nk^2 via W's
    own 1/nk), reducing to the molecular factors at nk=1. Validated at q=0
    against the molecular LinearResponseSolver BSE and at finite q by supercell
    folding (tests/test_pbc_bse.py).

    W_all: static W^Q for all Q from build_static_W_all_Q.
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)
    nk = dfints.nkpts
    kconserv = dfints.kconserv_pair[q]
    cmap = conj_mapping(dfints.cell, dfints.kpts)     # kbar = index of -k
    w = 1.0 / nk

    # e-h exchange v^q = (a_ka i_ki | j_kj b_kb) = M^H M, in (ki,i,a) layout
    M, offsets, N, nocc, nvirt = _stacked_Lov(dfints, kconserv)
    V_dir = M.conj().T @ M
    d_all = _diag_energies(dfints, kconserv, mo_energy)

    A = np.diag(d_all).astype(np.complex128) + 2.0 * w * V_dir
    B = 2.0 * w * V_dir.copy()

    def occ(k):
        return slice(0, nocc[k])

    def vir(k):
        return slice(nocc[k], dfints.nmo)

    for ki in range(nk):
        ka = kconserv[ki]
        sl_i = slice(offsets[ki], offsets[ki + 1])
        for kj in range(nk):
            kb = kconserv[kj]
            sl_j = slice(offsets[kj], offsets[kj + 1])

            # A screened direct: W_{ij,ab}, aux momentum kj-ki, -> [i,a,j,b]
            Wd = _screened_block(dfints, W_all, ki, kj, ka, kb,
                                 occ(ki), occ(kj), vir(ka), vir(kb))
            A_scr = Wd.transpose(0, 2, 1, 3)
            A[sl_i, sl_j] -= w * A_scr.reshape(A_scr.shape[0] * A_scr.shape[1], -1)

            # B screened (Fig 2d): antiresonant J at time-reversed momenta,
            # W_{i b~ | j~ a} with b~=conj(kb), j~=conj(kj), -> [i,a,j,b]
            kbb, kjb = cmap[kb], cmap[kj]
            Ws = _screened_block(dfints, W_all, ki, kbb, kjb, ka,
                                 occ(ki), vir(kbb), occ(kjb), vir(ka))
            B_scr = Ws.transpose(0, 3, 2, 1)
            B[sl_i, sl_j] -= w * B_scr.reshape(B_scr.shape[0] * B_scr.shape[1], -1)

    return A, B
