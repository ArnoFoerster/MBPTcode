"""Periodic transition amplitudes chi_a / chi_b -- the k-point generalization of
GW.transition_amplitudes.AmplitudeGenerator, built on the BSE (or RPA)
eigenpairs from pbc_casida.

chi_a: the GW transition amplitude. Molecular form
    chi_a^S_{p,r} = sum_ia (X+Y)^S_{ia} (ia|pr) = sum_P rho^S_P L_{p,r},
    rho^S_P = sum_ia L_ov[P,ia] (X+Y)^S_{ia}.
Periodic: the exciton S carries momentum transfer q, so the amplitude connects
p @ kp to m @ km = kp - q,
    chi_a^{S,q}_{p kp, m km} = sum_P rho^{S,q}_P L^{kp,km}_{p,m},
with rho^{S,q} the aux-basis projection of the transfer-q eigenvectors
(pbc_casida.solve_rpa_spectral's rho, or project_XpY here for BSE vectors).
The (transfer-q) rho contracts directly with the (transfer -q) L^{kp,km} block
via the same aux-index contraction as PBCDFIntegrals.eri_block.
"""
import numpy as np
from src.SingleReference.Periodic.pbc_casida import _stacked_Lov, _transition_layout


def kpoint_minus_q(dfints, q, kp):
    """Index km with kpts[km] = kpts[kp] - kpts[q] (mod G), i.e. kconserv_pair[q, km] = kp.

    The transfers ARE k-points, so km is recovered by inverting kconserv_pair.
    """
    matches = np.where(dfints.kconserv_pair[q] == kp)[0]
    if len(matches) != 1:
        raise RuntimeError(f"no unique k - q for kp={kp}, q={q}")
    return int(matches[0])


def project_XpY(dfints, kconserv, X, Y):
    """Aux-basis projection rho[P, S] = sum_{ki,ia} L_ov(ki->ka)[P,ia] (X+Y)^S_{ki,ia}.

    Same object as pbc_casida.solve_rpa_spectral's rho, but for an arbitrary set
    of eigenvectors (e.g. the BSE X, Y at transfer q). kconserv = kconserv_pair[q].
    """
    M, offsets, N, nocc, nvirt = _stacked_Lov(dfints, kconserv)
    return M @ (X + Y)


def get_chi_a(dfints, q, rho, kp):
    """GW transition amplitude chi_a^{S,q}[p @ kp, m @ km=kp-q] for target k-point kp.

    rho[P, S] : aux-basis projection of the transfer-q eigenvectors (project_XpY
    or solve_rpa_spectral). Returns (nstates, nmo, nmo), the amplitude over all
    orbitals p @ kp (rows) and m @ km (cols).
    """
    km = kpoint_minus_q(dfints, q, kp)
    L = dfints.Lblock(kp, km)                       # (naux, nmo, nmo)
    return np.einsum('PS,Ppm->Spm', rho, L)


def get_chi_b_vertex(dfints, q, W_all, X, Y, kp):
    """Vertex-correction transition amplitude chi_b^{S,q}[r @ kr, p @ kp] for
    target k-point kp, with kr = kp - q the internal k-point.

    Periodic port of GW.transition_amplitudes.get_chi_b_vertex_df (restricted,
    W-dressed). The vertex W-dressing entangles the exciton (i,a) with the
    internal r and target p, so momentum conservation selects the exciton
    ki = kr slice (a @ ka = ki + q = kp), and each of the four terms carries its
    OWN internal W-transfer (occ/virt X vs Y). The four
    molecular terms, with exciton i @ kr, a @ kp:

        chi_occ[k,p] = sum_ia X_ia (a_kp k_kr | i_kr p_kp)_W    [W at kr-kp = -q]
                     + sum_ia Y_ia (a_kp p_kp | i_kr k_kr)_W    [W at 0]
        chi_virt[c,p]= sum_ia X_ia (a_kp p_kp | i_kr c_kr)_W    [W at 0]
                     + sum_ia Y_ia (a_kp c_kr | i_kr p_kp)_W    [W at kr-kp = -q]

    Returns (nstates, nmo, nmo): internal r @ kr (rows, occ then virt), target
    p @ kp (cols). Reduces exactly to the molecular get_chi_b_vertex at nk=1.
    Finite-q momentum is confirmed end-to-end by the self-energy comparison
    against pyscf's krgw_ac (tests/test_pbc_self_energy.py).
    """
    nk = dfints.nkpts
    nmo = dfints.nmo
    kconserv = dfints.kconserv_pair[q]
    kr = kpoint_minus_q(dfints, q, kp)            # internal k
    ka = kconserv[kr]                             # = kp (exciton virtual k)
    assert ka == kp
    no_r, nv_r = dfints.nocc[kr], nmo - dfints.nocc[kr]
    no_i = dfints.nocc[kr]
    nv_a = nmo - dfints.nocc[kp]

    # exciton ki=kr slice, reshaped (i in occ@kr, a in virt@kp, S)
    offsets, N, nocc, nvirt = _transition_layout(dfints, kconserv)
    sl = slice(offsets[kr], offsets[kr + 1])
    nstates = X.shape[1]
    X_s = X[sl].reshape(no_i, nv_a, nstates)
    Y_s = Y[sl].reshape(no_i, nv_a, nstates)

    o_r = slice(0, dfints.nocc[kr])
    v_r = slice(dfints.nocc[kr], nmo)
    o_i = slice(0, dfints.nocc[kr])
    v_a = slice(dfints.nocc[kp], nmo)

    Q1 = dfints._pair_to_q(kp, kr)                # W transfer for the X/Y "-q" terms
    Q0 = dfints._pair_to_q(kp, kp)                # W transfer 0
    W1 = W_all[Q1]
    W0 = W_all[Q0]

    def scr(k1, k2, s1, s2, k3, k4, s3, s4, W):
        L12 = dfints.Lblock(k1, k2)[:, s1, s2]
        L34 = dfints.Lblock(k3, k4)[:, s3, s4]
        return np.einsum('Pxy,PR,Rzw->xyzw', L12, W, L34)

    # occ block: internal k @ kr (occ)
    int_occ_X = scr(kp, kr, v_a, o_r, kr, kp, o_i, slice(0, nmo), W1)   # (a,k,i,p)
    chi_occ = np.einsum('akip,iaS->kpS', int_occ_X, X_s)
    int_occ_Y = scr(kp, kp, v_a, slice(0, nmo), kr, kr, o_i, o_r, W0)   # (a,p,i,k)
    chi_occ += np.einsum('apik,iaS->kpS', int_occ_Y, Y_s)

    # virt block: internal c @ kr (virt)
    int_virt_X = scr(kp, kp, v_a, slice(0, nmo), kr, kr, o_i, v_r, W0)  # (a,p,i,c)
    chi_virt = np.einsum('apic,iaS->cpS', int_virt_X, X_s)
    int_virt_Y = scr(kp, kr, v_a, v_r, kr, kp, o_i, slice(0, nmo), W1)  # (a,c,i,p)
    chi_virt += np.einsum('acip,iaS->cpS', int_virt_Y, Y_s)

    chi = np.zeros((nstates, nmo, nmo), dtype=np.complex128)
    chi[:, o_r, :] = chi_occ.transpose(2, 0, 1)
    chi[:, v_r, :] = chi_virt.transpose(2, 0, 1)
    return chi
