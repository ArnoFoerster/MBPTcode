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
from pyscf.pbc.lib.kpts_helper import conj_mapping
from src.SingleReference.Periodic.pbc_casida import (_screened_block, _stacked_Lov,
                                                      _transition_layout)


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

    Periodic form of GW.transition_amplitudes.get_chi_b_vertex_df (restricted,
    W-dressed). The exciton S of transfer q is the eigenvector v_S of the pair
    space of that transfer, X on the excitations (a @ ki+q, i @ ki) and Y on
    the de-excitations (j @ k+q, b @ k), which the time-reversal-adapted layout
    stores in the slice kj = -(k+q). The amplitude closes the exciton with the
    screened exchange, sum_pairs v (v p|W|r w)^*, over EVERY slice: the W
    transfer kr - k changes from slice to slice. A virtual internal r takes
    the pole at +Omega_S, eigenvector (X, Y); an occupied one the pole at
    -Omega_S, eigenvector (Y, X):

        chi_virt[c,p] = sum_k [ X_k,ia (a p|c i)^* + Y_-(k+q),jb (j p|c b)^* ]
        chi_occ[l,p]  = sum_k [ Y_k,ia (a p|l i)^* + X_-(k+q),jb (j p|l b)^* ]

    with a @ k+q, i @ k, j @ k+q, b @ k. At nk = 1 these are the molecular
    X (a p|i c) + Y (a c|i p) and X (a l|i p) + Y (a p|i l). Returns
    (nstates, nmo, nmo): internal r @ kr (rows, occ then virt), target p @ kp
    (cols).
    """
    nmo = dfints.nmo
    kconserv = dfints.kconserv_pair[q]
    kr = kpoint_minus_q(dfints, q, kp)            # internal k
    cmap = conj_mapping(dfints.cell, dfints.kpts)
    offsets, N, nocc, nvirt = _transition_layout(dfints, kconserv)
    nstates = X.shape[1]
    allp = slice(0, nmo)
    o_r, v_r = slice(0, nocc[kr]), slice(nocc[kr], nmo)

    chi = np.zeros((nstates, nmo, nmo), dtype=np.complex128)
    for k in range(dfints.nkpts):
        kq = kconserv[k]                          # k + q
        kt = cmap[kq]                             # slice of the pair (j @ k+q, b @ k)
        sd = slice(offsets[k], offsets[k + 1])
        st = slice(offsets[kt], offsets[kt + 1])
        Xd = X[sd].reshape(nocc[k], nvirt[kq], nstates)
        Yd = Y[sd].reshape(nocc[k], nvirt[kq], nstates)
        Xt = X[st].reshape(nocc[kq], nvirt[k], nstates)
        Yt = Y[st].reshape(nocc[kq], nvirt[k], nstates)
        # (a p|r i)^* and (j p|r b)^*, W at transfer kr - k
        Ia = _screened_block(dfints, W_all, kq, kp, kr, k, slice(nocc[kq], nmo),
                             allp, allp, slice(0, nocc[k])).conj()
        Ij = _screened_block(dfints, W_all, kq, kp, kr, k, slice(0, nocc[kq]),
                             allp, allp, slice(nocc[k], nmo)).conj()
        chi[:, v_r] += (np.einsum('apri,iaS->Srp', Ia[:, :, v_r], Xd)
                        + np.einsum('jprb,jbS->Srp', Ij[:, :, v_r], Yt))
        chi[:, o_r] += (np.einsum('apri,iaS->Srp', Ia[:, :, o_r], Yd)
                        + np.einsum('jprb,jbS->Srp', Ij[:, :, o_r], Xt))
    return chi
