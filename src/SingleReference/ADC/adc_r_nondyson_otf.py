"""Restricted non-Dyson IP-ADC(3) with the 1h-1h block applied on the fly, density-fitted.

adc_r_nondyson_df precomputes the static 1h-1h block once, a step of order O^2 V^3 N_aux
(the MP3-like step of non-Dyson ADC(3)), and then iterates at a lower cost. Here the
block is never formed: its third-order terms are applied to each trial vector, at
O V^3 N_aux per product, so the whole calculation scales as the number of Davidson
iterations times N^5, as Dyson ADC(3) does. This is the first of the two implementations
of Dempwolff, Schneider, Hodecker and Dreuw, J. Chem. Phys. 150, 064108 (2019).

The block is their Eq. (A8), summed over spin for a closed shell, with first-order
amplitudes T[i,k,a,b] = (ia|kb) / (e_a + e_b - e_i - e_k):

    I = -f + I2 + I_rho + sum over the pp, ph, hh and x terms of (A + A^T)

I2 (second order) and I_rho (the second-order density terms, equal to the Sigma(inf)
block of the unrelaxed MP2 density) are precomputed at O^3 V^2. The four third-order
terms would cost O^2 V^4 or O^3 V^3 to precompute and are applied instead: the trial
vector x is first contracted into the amplitudes, tau[k,c,d] = sum_j x_j T[j,k,c,d], and
every later step is at most O V^3 N_aux, with intermediates of size O V^2 or O^3. The
amplitudes are regenerated from the DF factor in chunks of occupied indices, three
passes per product, never held whole.

Eq. (A8) and the symmetrized fold of adc_r_nondyson_df are two third-order forms of one
block: equal on the diagonal and different off it at third order (1e-4 Hartree for
water/6-31G, with IPs equal to 0.05 meV).

Parallelism: inside distributed(comm) the occupied chunks of every amplitude pass and the
auxiliary-index ranges of the DF contractions are split over the ranks and summed, so
every rank returns the whole product.
"""
import numpy as np

from src.Base.utils.parallel.mpi_grid import contiguous_block, current_comm, reduce_sum
from src.SingleReference.CC.cached_einsum import einsum as _einsum
from src.SingleReference.common.eri_blocks import b_block
from src.SingleReference.common.mp_amplitudes import chunk_ranges, occ_chunk

# memory of one auxiliary-index slab of the DF contractions below
_SLAB_BYTES = 1 << 30


def _psum(*arrays):
    """Sum the ranks' partials in place (no-op without a distribution)."""
    comm = current_comm()
    if comm is not None and comm.Get_size() > 1:
        for a in arrays:
            reduce_sum(a, comm)


def _q_slabs(naux, per_q_bytes):
    """This rank's auxiliary-index slabs: a contiguous block per rank, cut so that one
    slab's intermediate stays below _SLAB_BYTES."""
    comm = current_comm()
    size = 1 if comm is None else comm.Get_size()
    q0, q1 = (0, naux) if size == 1 else contiguous_block(naux, comm.Get_rank(), size)
    step = max(1, int(_SLAB_BYTES // max(1, per_q_bytes)))
    return [slice(q, min(q + step, q1)) for q in range(q0, q1, step)]


class _Blocks:
    """Views of the DF factor and the amplitude stream of one closed-shell system."""

    def __init__(self, B, eps, O):
        self.B, self.O = B, O
        self.V = B.shape[1] - O
        self.naux = B.shape[0]
        o, v = slice(0, O), slice(O, None)
        self.oo, self.ov = b_block(B, o, o), b_block(B, o, v)
        self.vo, self.vv = b_block(B, v, o), b_block(B, v, v)
        self.eo, self.ev = eps[:O], eps[O:]

    def amplitudes(self):
        """(lo, hi, T[lo:hi]) with T[i,k,a,b] = (ia|kb) / (e_a + e_b - e_i - e_k)."""
        evv = self.ev[:, None] + self.ev[None, :]
        for lo, hi in chunk_ranges(self.O, occ_chunk(self.O)):
            T = _einsum('Qia,Qkb->ikab', self.ov[:, lo:hi], self.ov, optimize=True)
            T /= evv[None, None] - self.eo[lo:hi, None, None, None] - self.eo[None, :, None, None]
            yield lo, hi, T


def static_part(B, eps, O, sigma_inf_oo):
    """-f + I2 + I_rho of Eq. (A8), excitation sign (eigenvalues are IPs), (O, O).
    sigma_inf_oo: the occupied block of Sigma(inf) of the unrelaxed MP2 density, whose
    negative is I_rho."""
    blk = _Blocks(B, eps, O)
    A2 = np.zeros((O, O))
    for lo, hi, T in blk.amplitudes():
        Ua = _einsum('ikab,Qkb->iQa', T, blk.ov, optimize=True)
        Ub = _einsum('ikab,Qka->iQb', T, blk.ov, optimize=True)
        A2[lo:hi] = (_einsum('iQa,Qja->ij', Ua, blk.ov, optimize=True)
                     - 0.5 * _einsum('iQb,Qjb->ij', Ub, blk.ov, optimize=True))
    _psum(A2)
    return -np.diag(blk.eo) + A2 + A2.T - sigma_inf_oo


def third_order_apply(B, eps, O, x):
    """(pp + ph + hh + x terms of Eq. (A8)) @ x for one vector x (O,), excitation sign."""
    blk = _Blocks(B, eps, O)
    V, naux = blk.V, blk.naux

    # pass 1: the vector into the amplitudes, tau[k,c,d] = sum_j x_j T[j,k,c,d]
    tau = np.zeros((O, V, V))
    for lo, hi, T in blk.amplitudes():
        tau += _einsum('j,jkcd->kcd', x[lo:hi], T, optimize=True)
    _psum(tau)
    tau_s = np.ascontiguousarray(tau.transpose(0, 2, 1))

    # DF contractions without amplitudes, auxiliary index split over the ranks
    X = np.zeros((O, V, V))            # every term of the form sum_kab T[ikab] X[k,a,b]
    E1, E2 = np.zeros((2, O, V, V)), np.zeros((2, O, V, V))     # [tau, tau_s]
    gam1, gam2 = np.zeros((O, V, V)), np.zeros((O, V, V))
    sig = (tau, tau_s)
    theta = 2.0 * tau - tau_s
    Lth = np.zeros((O, V, V))
    for q in _q_slabs(naux, 8 * O * V * V):
        Bvv, Boo, Bov, Bvo = blk.vv[q], blk.oo[q], blk.ov[q], blk.vo[q]
        # pp: Gamma = L(2 tau - tau_s), L phi[k,a,b] = sum_cd (ac|bd) phi[k,c,d]
        Z = _einsum('Qbd,kcd->Qkcb', Bvv, theta, optimize=True)
        Lth += _einsum('Qac,Qkcb->kab', Bvv, Z, optimize=True)
        for s, phi in enumerate(sig):
            # ph: E1[k,a,b] = sum_lc phi[l,a,c] (kl|cb), E2[k,a,b] = sum_lc phi[l,a,c] (kb|cl)
            Y = _einsum('lac,Qcb->Qlab', phi, Bvv, optimize=True)
            E1[s] += _einsum('Qkl,Qlab->kab', Boo, Y, optimize=True)
            Za = _einsum('lac,Qcl->Qa', phi, Bvo, optimize=True)
            E2[s] += _einsum('Qa,Qkb->kab', Za, Bov, optimize=True)
        # x, A^T x: gam1[k,c,a] = sum_j x_j (kj|ca), gam2[k,a,c] = sum_j x_j (ka|cj)
        gam1 += _einsum('Qkj,j,Qca->kca', Boo, x, Bvv, optimize=True)
        gam2 += _einsum('Qka,Qcj,j->kac', Bov, Bvo, x, optimize=True)
    _psum(Lth, E1, E2, gam1, gam2)
    X -= Lth                                               # pp: 2 A x = -sum T Gamma
    H = 2 * E1[0] - 4 * E2[0] - E1[1] + 2 * E2[1]
    Ht = -E1[0] + 2 * E2[0] + 2 * E1[1] - E2[1]
    X += H + Ht.transpose(0, 2, 1)                         # ph: 2 A x

    # pass 2: amplitude-dependent intermediates of the hh and x terms
    R1, R2 = np.zeros((O, O, O)), np.zeros((O, O, O))       # [k, l, m]
    W = np.zeros((4, O, V, V))      # W[tau], W[tau_s], Wbar[tau], Wbar[tau_s]   [k, a, c]
    P = np.zeros((4, O, V, V))      # P1, Pbar1, P2, Pbar2                        [l, b, c]
    S1, S2 = np.zeros((O, V, V)), np.zeros((O, V, V))       # [m, a, b]
    om1 = _einsum('Qkm,Qlj,j->klm', blk.oo, blk.oo, x, optimize=True)   # (km|lj) x_j
    om2 = _einsum('Qkj,j,Qlm->klm', blk.oo, x, blk.oo, optimize=True)   # (kj|lm) x_j
    for lo, hi, T in blk.amplitudes():
        Tsw = T.transpose(0, 1, 3, 2)
        R1[lo:hi] = _einsum('klab,mab->klm', T, tau_s, optimize=True)
        R2[lo:hi] = _einsum('klab,mab->klm', T, tau, optimize=True)
        W[0, lo:hi] = _einsum('klab,lbc->kac', T, tau, optimize=True)
        W[1, lo:hi] = _einsum('klab,lbc->kac', T, tau_s, optimize=True)
        W[2, lo:hi] = _einsum('klab,lbc->kac', Tsw, tau, optimize=True)
        W[3, lo:hi] = _einsum('klab,lbc->kac', Tsw, tau_s, optimize=True)
        P[0] += _einsum('kca,klab->lbc', gam1[lo:hi], T, optimize=True)
        P[1] += _einsum('kca,klab->lbc', gam1[lo:hi], Tsw, optimize=True)
        P[2] += _einsum('kac,klab->lbc', gam2[lo:hi], T, optimize=True)
        P[3] += _einsum('kac,klab->lbc', gam2[lo:hi], Tsw, optimize=True)
        S1 += _einsum('klm,klab->mab', om1[lo:hi], T, optimize=True)
        S2 += _einsum('klm,klab->mab', om2[lo:hi], T, optimize=True)
    _psum(R1, R2, W, P, S1, S2)

    y = np.zeros(O)
    # hh, A x: -1/2 sum_klm [(km|li)(2R1 - R2) + (ki|lm)(2R2 - R1)]
    Ra, Rb = 2 * R1 - R2, 2 * R2 - R1
    # x, A x: -sum_kac [(ki|ca) Y1 + (ka|ci) Y2]
    Y1 = W[0] - 2 * W[1] - 2 * W[2] + W[3]
    Y2 = -2 * W[0] + 4 * W[1] + W[2] - 2 * W[3]
    for q in _q_slabs(naux, 8 * O * O):
        Boo, Bvv, Bov, Bvo = blk.oo[q], blk.vv[q], blk.ov[q], blk.vo[q]
        y -= 0.5 * _einsum('Qli,Qkm,klm->i', Boo, Boo, Ra, optimize=True)
        y -= 0.5 * _einsum('Qki,Qlm,klm->i', Boo, Boo, Rb, optimize=True)
        y -= _einsum('Qki,Qca,kac->i', Boo, Bvv, Y1, optimize=True)
        y -= _einsum('Qci,Qka,kac->i', Bvo, Bov, Y2, optimize=True)
    # hh, A^T x: -1/2 sum_mab T[imab] Xi[m,a,b]
    Xi = (2 * S1 - S2).transpose(0, 2, 1) + (2 * S2 - S1)
    X -= 0.5 * Xi
    # x, A^T x: -sum_lbc T[ilbc] (Z1 + Z2^T)[l,b,c]
    Z1 = P[0] - 2 * P[1] - 2 * P[2] + P[3]
    Z2 = -2 * P[0] + P[1] + 4 * P[2] - 2 * P[3]
    X -= Z1 + Z2.transpose(0, 2, 1)

    # pass 3: every term with the amplitudes on the output side
    for lo, hi, T in blk.amplitudes():
        y[lo:hi] += _einsum('ikab,kab->i', T, X, optimize=True)
    _psum(y)
    return y
